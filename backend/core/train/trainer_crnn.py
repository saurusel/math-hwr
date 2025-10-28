import os, io, time, json, math
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from ..data.dataset import HWRDataset
from ..data.collate import collate_ctc
from ..data.augmentations import augment_training_batch  # NEW: Augmentations
from ..models.crnn_ctc import CRNN_CTC
from ..tokenizer import TOKEN_LIST
from ..metrics import cer as cer_fn, wer as wer_fn
from ..parser import is_valid_token_stream
from ..decoders.ctc_decode import ctc_greedy_decode
from ..event_logger import EventLogger

def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _png_from_tensor(chw):
    # chw in [0,1], single-channel
    arr = (1.0 - chw.squeeze(0).clamp(0,1)).mul(255).byte().cpu().numpy()  # back to black ink
    return Image.fromarray(arr, mode="L")

def _png_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL for frontend display."""
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64_str}"

def train_crnn_ctc(config: Dict[str, Any], run_dir: str, emit: Callable[[Dict[str,Any]], None]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]
    img_h = int(data_cfg.get("img_h", 64)); img_w_max = int(data_cfg.get("img_w_max", 512))

    # NEW: Support max_samples to limit dataset size
    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    # Initialize EventLogger for persistence
    events_file = os.path.join(run_dir, "events.jsonl")
    logger = EventLogger(events_file)
    logger.log_status("RUNNING", "Training started")

    # Check for stop signal file
    stop_file = os.path.join(run_dir, "STOP_REQUESTED")
    def should_stop():
        return os.path.exists(stop_file)

    train_ds = HWRDataset(train_dir, img_h=img_h, img_w_max=img_w_max, max_samples=max_train)
    val_ds   = HWRDataset(val_dir,   img_h=img_h, img_w_max=img_w_max, max_samples=max_val)

    bs = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_ctc)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate_ctc)

    blank_id = len(TOKEN_LIST)
    num_classes = blank_id + 1

    model = CRNN_CTC(num_classes=num_classes).to(device)
    amp = bool(config["train"].get("amp", True))
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    opt = optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=config.get("optimizer",{}).get("weight_decay", 1e-2))

    # Scheduler
    sched_cfg = config.get("scheduler", {"type":"onecycle"})
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(config["train"]["epochs"])
    scheduler = None
    if sched_cfg.get("type","onecycle") == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            opt, max_lr=config["train"]["lr"],
            steps_per_epoch=steps_per_epoch, epochs=int(config["train"]["epochs"]),
            pct_start=float(sched_cfg.get("warmup_pct", 0.1))
        )
    elif sched_cfg.get("type") == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    grad_clip = float(config["train"].get("grad_clip_norm", 1.0))

    epochs = int(config["train"]["epochs"])
    emit_every = int(config["train"].get("emit_every", 25))
    ckpt_every_steps = int(config["train"].get("checkpoint_every_steps", 0))  # 0 = only by epoch

    # === Resume support ===
    start_epoch = 0
    global_step = 0
    resume_from: Optional[str] = config.get("resume_from")
    ckpt_to_load = None
    if resume_from:
        cand = os.path.join("runs", resume_from, "checkpoint_latest.pt")
        if os.path.exists(cand):
            ckpt_to_load = cand
    else:
        cand = os.path.join(run_dir, "checkpoint_latest.pt")
        if os.path.exists(cand):
            ckpt_to_load = cand

    if ckpt_to_load:
        state = torch.load(ckpt_to_load, map_location="cpu")
        if "model" in state: model.load_state_dict(state["model"])
        if "opt" in state:   opt.load_state_dict(state["opt"])
        if "scaler" in state and hasattr(scaler, "load_state_dict"):
            try: scaler.load_state_dict(state["scaler"])
            except Exception: pass
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        emit({"event":"resumed", "from": ckpt_to_load, "epoch": start_epoch, "global_step": global_step})

    # === Dataset stats & receptive field sanity ===
    def _length_stats(ds: HWRDataset):
        lens = []
        for r in ds.rows:
            tgt = r.get("target_canonical") or r["target"]
            lens.append(len(tgt.split()))
        return {"count": len(lens), "min": int(min(lens) if lens else 0), "max": int(max(lens) if lens else 0), "mean": float(sum(lens)/max(1,len(lens)))}
    train_len = _length_stats(train_ds); val_len = _length_stats(val_ds)

    # measure typical T on a small batch
    model.eval()
    with torch.no_grad():
        val_x, _, _, _ = next(iter(val_loader))
        val_x = val_x.to(device)
        logits, T_ex = model(val_x)
    model.train()

    stats_event = {
        "event": "dataset_stats",
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "train_tokens": train_len,
        "val_tokens": val_len,
        "typical_T": int(T_ex)
    }
    emit(stats_event)
    logger.log_dataset_stats(len(train_ds), len(val_ds), train_len)

    # quick validation (restore training flag after)
    def quick_val(save_png_step: Optional[int]=None, return_samples: bool=False):
        was_training = model.training
        model.eval()
        val_x, _, _, val_tokens = next(iter(val_loader))
        val_x = val_x.to(device)
        with torch.no_grad():
            val_logits, Tval = model(val_x)
            hyps_idx = ctc_greedy_decode(val_logits, blank_id)

        # Process multiple samples if requested
        samples_list = []
        if return_samples:
            num_samples = min(10, val_x.size(0))  # Up to 10 samples
            for b in range(num_samples):
                tgt_tokens_b = val_tokens[b]
                tgt_str_b = " ".join(tgt_tokens_b)
                hyp_tokens_b = [TOKEN_LIST[i] for i in hyps_idx[b]]
                hyp_str_b = " ".join(hyp_tokens_b)

                # Convert image to base64 for frontend
                img_b = _png_from_tensor(val_x[b].detach().cpu())
                img_b64 = _png_to_base64(img_b)

                samples_list.append({
                    "id": f"batch_{b}",
                    "target": tgt_str_b,
                    "pred": hyp_str_b,
                    "image_b64": img_b64,
                    "ok": tgt_str_b == hyp_str_b
                })

        # First sample for metrics
        tgt_tokens = val_tokens[0]
        tgt_str = " ".join(tgt_tokens)
        hyp_tokens = [TOKEN_LIST[i] for i in hyps_idx[0]]
        hyp_str = " ".join(hyp_tokens)
        m = {
            "cer": cer_fn(tgt_str, hyp_str),
            "wer": wer_fn(tgt_tokens, hyp_tokens),
            "exact": 1.0 if tgt_str == hyp_str else 0.0,
            "valid": 1.0 if (len(hyp_tokens) > 0 and is_valid_token_stream(hyp_tokens)) else 0.0,
            "tree_f1": 0.0
        }

        png_path = ""
        if save_png_step is not None:
            img = _png_from_tensor(val_x[0].detach().cpu())
            png_path = os.path.join(run_dir, "samples", f"step_{save_png_step}.png")
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            img.save(png_path)

        if was_training:
            model.train()

        if return_samples:
            return m, tgt_str, hyp_str, png_path, samples_list
        return m, tgt_str, hyp_str, png_path

    for epoch in range(start_epoch+1, epochs+1):
        # Check for stop signal
        if should_stop():
            logger.log_status("STOPPED", f"Training stopped by user at epoch {epoch}")
            logger.log_text(f"Early stopping requested. Saving checkpoint at epoch {epoch-1}")
            emit({"event": "stopped", "epoch": epoch - 1, "message": "Training stopped by user"})
            break

        epoch_start_time = time.time()
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch in train_loader:
            # Check stop signal during epoch
            if should_stop():
                logger.log_text(f"Stop signal detected during epoch {epoch}, finishing current epoch...")
                break
            if not model.training:
                model.train()
            x, y, y_lens, tokens = batch
            x = x.to(device); y = y.to(device)

            # AUGMENTATIONS (NEW!) - Apply before training
            augment_config = config.get("augment", {"invert": True, "noise": True, "blur": True})
            x = augment_training_batch(x, augment_config)

            # CTC requires input length T >= target length
            with torch.no_grad():
                dummy_logits, Tcur = model(x[:1])
                max_tgt_len = int(y_lens.max().item())
            if Tcur < max_tgt_len:
                emit({"event":"warning","kind":"ctc_short_T","T":int(Tcur),"max_tgt_len":int(max_tgt_len)})
                # continue anyway; but this is a red flag for architecture

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits, T = model(x)            # (T,B,C)
                log_probs = logits.log_softmax(dim=-1)
                inp_lens = torch.full((x.size(0),), T, dtype=torch.long, device=logits.device)
                loss = criterion(log_probs, y, inp_lens, y_lens)

            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            epoch_loss_sum += float(loss.detach().cpu().item())
            epoch_batches += 1

            if emit_every > 0 and (global_step % emit_every == 0):
                result = quick_val(save_png_step=global_step, return_samples=True)
                m, tgt_str, hyp_str, png_path, samples_list = result if len(result) == 5 else (*result, [])

                # Emit metrics
                emit({
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "metrics": {"loss": float(loss.detach().cpu().item()), **m}
                })

                # Emit sample predictions with images
                if samples_list:
                    emit({
                        "event": "sample_pred",
                        "epoch": epoch,
                        "items": samples_list
                    })

                    # IMPORTANT: Also save to events.jsonl for history
                    logger.log_sample_pred(epoch, samples_list)

            if ckpt_every_steps > 0 and (global_step % ckpt_every_steps == 0):
                _atomic_save({
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scaler": getattr(scaler, "state_dict", lambda: {})(),
                    "epoch": epoch,
                    "global_step": global_step
                }, os.path.join(run_dir, "checkpoint_latest.pt"))
                _atomic_save({"model": model.state_dict()}, os.path.join(run_dir, f"checkpoint_step_{global_step}.pt"))

        # epoch end validation
        model.eval()
        tot = 0; cer_sum=0; wer_sum=0; exact_sum=0; valid_sum=0
        with torch.no_grad():
            for val_x, _, _, val_tokens in val_loader:
                val_x = val_x.to(device)
                val_logits, Tval = model(val_x)
                hyps_idx = ctc_greedy_decode(val_logits, blank_id)
                B = val_x.size(0)
                for b in range(B):
                    tgt = " ".join(val_tokens[b])
                    hyp_tokens = [TOKEN_LIST[i] for i in hyps_idx[b]]
                    hyp = " ".join(hyp_tokens)
                    cer_sum += cer_fn(tgt, hyp)
                    wer_sum += wer_fn(val_tokens[b], hyp_tokens)
                    exact_sum += 1.0 if tgt == hyp else 0.0
                    valid_sum += 1.0 if (len(hyp_tokens) > 0 and is_valid_token_stream(hyp_tokens)) else 0.0
                    tot += 1

        _atomic_save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": getattr(scaler, "state_dict", lambda: {})(),
            "epoch": epoch,
            "global_step": global_step
        }, os.path.join(run_dir, "checkpoint_latest.pt"))
        _atomic_save({"model": model.state_dict()}, os.path.join(run_dir, f"checkpoint_epoch_{epoch}.pt"))

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = epoch_loss_sum / max(1, epoch_batches)

        epoch_metrics = {
            "train_loss": avg_train_loss,
            "val_loss": float(loss.detach().cpu().item()) if tot > 0 else 0,
            "cer": cer_sum/max(1,tot),
            "wer": wer_sum/max(1,tot),
            "exact": exact_sum/max(1,tot),
            "valid": valid_sum/max(1,tot),
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_time_sec": epoch_duration,
            "samples_processed": tot,
            "batches_processed": epoch_batches
        }

        emit({
            "event": "epoch_end",
            "epoch": epoch,
            "metrics": epoch_metrics
        })

        # Persist to events.jsonl
        logger.log_metric(epoch, epoch_metrics)
        logger.log_text(f"Epoch {epoch}/{epochs} completed - CER: {(epoch_metrics['cer']*100):.2f}%")

    final_ckpt = os.path.join(run_dir, "crnn_final.pt")
    _atomic_save({"model": model.state_dict(), "vocab": TOKEN_LIST}, final_ckpt)

    # Log final checkpoint
    logger.log_checkpoint("final", final_ckpt, epochs, epoch_metrics if 'epoch_metrics' in locals() else {})
    logger.log_status("FINISHED", f"Training completed after {epochs} epochs")
    logger.close()

    emit({ "event": "finished", "epoch": epochs, "checkpoint": final_ckpt })
