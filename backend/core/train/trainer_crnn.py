import os, io, time, json, math
from typing import Dict, Any, Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from ..data.dataset import HWRDataset
from ..data.collate import collate_ctc
from ..models.crnn_ctc import CRNN_CTC
from ..tokenizer import TOKEN_LIST
from ..metrics import cer as cer_fn, wer as wer_fn
from ..parser import is_valid_token_stream
from ..decoders.ctc_decode import ctc_greedy_decode

def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _png_from_tensor(chw):
    # chw in [0,1], single-channel
    arr = (1.0 - chw.squeeze(0).clamp(0,1)).mul(255).byte().cpu().numpy()  # back to black ink
    return Image.fromarray(arr, mode="L")

def train_crnn_ctc(config: Dict[str, Any], run_dir: str, emit: Callable[[Dict[str,Any]], None]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]
    img_h = int(data_cfg.get("img_h", 64)); img_w_max = int(data_cfg.get("img_w_max", 512))

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    train_ds = HWRDataset(train_dir, img_h=img_h, img_w_max=img_w_max)
    val_ds   = HWRDataset(val_dir,   img_h=img_h, img_w_max=img_w_max)

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

    emit({"event":"dataset_stats", "train_tokens": train_len, "val_tokens": val_len, "typical_T": int(T_ex)})

    # quick validation (restore training flag after)
    def quick_val(save_png_step: Optional[int]=None):
        was_training = model.training
        model.eval()
        val_x, _, _, val_tokens = next(iter(val_loader))
        val_x = val_x.to(device)
        with torch.no_grad():
            val_logits, Tval = model(val_x)
            hyps_idx = ctc_greedy_decode(val_logits, blank_id)
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
            img.save(png_path)
        if was_training:
            model.train()
        return m, tgt_str, hyp_str, png_path

    for epoch in range(start_epoch+1, epochs+1):
        model.train()
        for batch in train_loader:
            if not model.training:
                model.train()
            x, y, y_lens, tokens = batch
            x = x.to(device); y = y.to(device)

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

            if emit_every > 0 and (global_step % emit_every == 0):
                m, tgt_str, hyp_str, png_path = quick_val(save_png_step=global_step)
                emit({
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "metrics": {"loss": float(loss.detach().cpu().item()), **m},
                    "samples": [{
                        "png_path": png_path.replace("\\","/"),
                        "target": tgt_str,
                        "pred": hyp_str,
                        "ok": m["exact"] > 0.5
                    }]
                })

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

        emit({
            "event": "epoch_end",
            "epoch": epoch,
            "metrics": {
                "loss": float(loss.detach().cpu().item()),
                "cer": cer_sum/max(1,tot),
                "wer": wer_sum/max(1,tot),
                "exact": exact_sum/max(1,tot),
                "valid": valid_sum/max(1,tot),
                "tree_f1": 0.0
            }
        })

    _atomic_save({"model": model.state_dict(), "vocab": TOKEN_LIST}, os.path.join(run_dir, "crnn_final.pt"))
    emit({ "event": "finished", "epoch": epochs, "checkpoint": os.path.join(run_dir, "crnn_final.pt") })
