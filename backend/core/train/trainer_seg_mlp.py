"""
M2 Trainer: Segmentation + MLP Classifier
Classical OCR approach with character-level segmentation and classification.
"""
import os, io, time, json, math
from typing import Dict, Any, Callable, Optional, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from ..data.seg_dataset import SegmentedCharDataset, collate_seg_mlp
from ..data.dataset import HWRDataset  # For validation on full expressions
from ..data.augmentations import augment_training_batch  # NEW: Augmentations
from ..models.seg_mlp import SegmentationOCR
from ..tokenizer import TOKEN_LIST
from ..metrics import cer as cer_fn, wer as wer_fn
from ..parser import is_valid_token_stream
from ..segmentation_improved import segment_and_prepare_improved
from ..event_logger import EventLogger


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _png_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 data URL."""
    import base64
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{b64_str}"


def train_seg_mlp(config: Dict[str, Any], run_dir: str, emit: Callable[[Dict[str,Any]], None]):
    """
    Train M2: Segmentation + MLP Classifier model.

    Training strategy:
    1. Segment all training expressions into individual characters
    2. Train MLP to classify each character
    3. Validate on full expressions (segment + classify + combine)

    Args:
        config: Training configuration
        run_dir: Output directory for this run
        emit: Callback for SSE events
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = config.get("data", {})
    train_dir = data_cfg["train_dir"]
    val_dir = data_cfg["val_dir"]

    max_train = data_cfg.get("max_train_samples")
    max_val = data_cfg.get("max_val_samples")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "samples"), exist_ok=True)

    # Initialize EventLogger
    events_file = os.path.join(run_dir, "events.jsonl")
    logger = EventLogger(events_file)
    logger.log_status("RUNNING", "Training M2 (Segmentation + MLP) started")

    # Emit RUNNING status immediately so UI updates
    emit({"event": "status", "status": "RUNNING", "message": "Training started"})

    # Stop signal
    stop_file = os.path.join(run_dir, "STOP_REQUESTED")
    def should_stop():
        return os.path.exists(stop_file)

    # Load datasets
    # Training: segmented characters (individual symbols)
    train_ds = SegmentedCharDataset(train_dir, target_size=(32, 32), max_samples=max_train)

    # Validation: full expressions (for end-to-end validation)
    val_full_ds = HWRDataset(val_dir, img_h=64, img_w_max=512, max_samples=max_val)

    bs = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_seg_mlp)

    # Create token->id mapping
    token_to_id = {tok: i for i, tok in enumerate(TOKEN_LIST)}
    num_classes = len(TOKEN_LIST)

    # Model
    model = SegmentationOCR(num_classes=num_classes, input_size=32*32, dropout=0.3).to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-2)

    # Scheduler
    epochs = int(config["train"]["epochs"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * epochs

    sched_type = config.get("train", {}).get("scheduler", "cosine")
    if sched_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=config["train"]["lr"],
                                                   steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Training settings
    grad_clip = float(config["train"].get("grad_clip_norm", 1.0))
    emit_every = int(config["train"].get("emit_every", 25))

    # Emit dataset stats
    stats_event = {
        "event": "dataset_stats",
        "train_samples": len(train_ds),
        "val_samples": len(val_full_ds),
        "train_characters": len(train_ds),
        "model_type": "M2_Segmentation_MLP"
    }
    emit(stats_event)
    logger.log_dataset_stats(
        train_samples=len(train_ds),
        val_samples=len(val_full_ds),
        stats={"train_characters": len(train_ds), "model_type": "M2_Segmentation_MLP"}
    )

    # Training loop
    global_step = 0
    best_char_acc = 0.0
    best_seq_acc = 0.0

    for epoch in range(1, epochs + 1):
        if should_stop():
            emit({"event": "stopped", "epoch": epoch})
            logger.log_status("STOPPED", f"Stopped at epoch {epoch}")
            break

        # === TRAINING ===
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            if should_stop():
                break

            images = images.to(device)  # (batch_size, 1, H, W)

            # AUGMENTATIONS (NEW!) - Apply before training
            augment_config = config.get("augment", {"invert": True, "noise": True, "blur": True})
            images = augment_training_batch(images, augment_config)

            # Convert labels to IDs
            label_ids = []
            for label in labels:
                if label in token_to_id:
                    label_ids.append(token_to_id[label])
                else:
                    label_ids.append(0)  # Unknown token -> class 0

            targets = torch.tensor(label_ids, dtype=torch.long, device=device)

            # Forward
            opt.zero_grad()
            logits = model(images)  # (batch_size, num_classes)
            loss = criterion(logits, targets)

            # Backward
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            scheduler.step()

            # Metrics
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            epoch_loss += loss.item()

            global_step += 1

            # Emit progress
            if global_step % emit_every == 0:
                emit({
                    "event": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss.item(),
                    "char_acc": 100.0 * correct / max(1, total),
                    "lr": scheduler.get_last_lr()[0] if scheduler else config["train"]["lr"]
                })

        train_loss = epoch_loss / max(1, len(train_loader))
        train_char_acc = 100.0 * correct / max(1, total)

        # === VALIDATION ===
        model.eval()
        val_correct_chars = 0
        val_total_chars = 0
        val_correct_seqs = 0
        val_total_seqs = 0
        val_correct_seg = 0
        samples_list = []  # For sample predictions

        with torch.no_grad():
            for val_idx, val_row in enumerate(val_full_ds.rows[:min(50, len(val_full_ds.rows))]):
                # Load full expression image
                # val_row["image"] already contains "images/filename.png"
                img_path = os.path.join(val_dir, val_row["image"])
                if not os.path.exists(img_path):
                    continue

                img_pil = Image.open(img_path).convert("L")
                img_arr = np.array(img_pil)

                # Ground truth - flat tokens (no grouping)
                gt_tokens = (val_row.get("target_canonical") or val_row["target"]).split()

                # Segment and classify using IMPROVED segmentation
                segments = segment_and_prepare_improved(
                    img_arr,
                    target_size=(32, 32),
                    adaptive=True,
                    split_wide=False,
                    morph_strength="light"
                )

                if not segments:
                    val_total_seqs += 1
                    continue

                # Prepare batch
                seg_tensors = [torch.from_numpy(seg[0]).unsqueeze(0).to(device) for seg in segments]
                if not seg_tensors:
                    val_total_seqs += 1
                    continue

                batch_segs = torch.stack(seg_tensors)  # (num_segments, 1, H, W)

                # Classify
                logits = model(batch_segs)
                pred_ids = torch.argmax(logits, dim=1).cpu().tolist()
                pred_tokens = [TOKEN_LIST[pid] if pid < len(TOKEN_LIST) else "?" for pid in pred_ids]

                # Save sample predictions (first 10)
                if len(samples_list) < 10:
                    import io, base64
                    buf = io.BytesIO()
                    img_pil.save(buf, format="PNG")
                    img_b64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

                    samples_list.append({
                        "id": f"val_{val_idx}",
                        "target": " ".join(gt_tokens),
                        "pred": " ".join(pred_tokens),
                        "ok": pred_tokens == gt_tokens,
                        "image_b64": img_b64
                    })

                # Check if segmentation count matches
                if len(pred_tokens) == len(gt_tokens):
                    val_correct_seg += 1

                    # Character-level accuracy
                    for pt, gt in zip(pred_tokens, gt_tokens):
                        val_total_chars += 1
                        if pt == gt:
                            val_correct_chars += 1

                # Sequence-level accuracy (exact match)
                val_total_seqs += 1
                if pred_tokens == gt_tokens:
                    val_correct_seqs += 1

        val_char_acc = 100.0 * val_correct_chars / max(1, val_total_chars)
        val_seq_acc = 100.0 * val_correct_seqs / max(1, val_total_seqs)
        val_seg_acc = 100.0 * val_correct_seg / max(1, val_total_seqs)

        # Update best metrics
        improved = False
        if val_char_acc > best_char_acc:
            best_char_acc = val_char_acc
            improved = True
        if val_seq_acc > best_seq_acc:
            best_seq_acc = val_seq_acc

        # Emit epoch metrics
        epoch_metrics = {
            "event": "epoch_end",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_char_acc": train_char_acc,
            "val_char_acc": val_char_acc,
            "val_seq_acc": val_seq_acc,
            "val_seg_acc": val_seg_acc,
            "lr": scheduler.get_last_lr()[0] if scheduler else config["train"]["lr"],
            "best_char_acc": best_char_acc,
            "best_seq_acc": best_seq_acc
        }
        emit(epoch_metrics)
        logger.log_metric(epoch, {
            "train_loss": train_loss,
            "train_char_acc": train_char_acc,
            "val_char_acc": val_char_acc,
            "val_seq_acc": val_seq_acc,
            "val_seg_acc": val_seg_acc,
            "lr": scheduler.get_last_lr()[0] if scheduler else config["train"]["lr"],
            "best_char_acc": best_char_acc,
            "best_seq_acc": best_seq_acc
        })

        # Emit sample predictions with images
        if samples_list:
            emit({
                "event": "sample_pred",
                "epoch": epoch,
                "items": samples_list
            })
            logger.log_sample_pred(epoch, samples_list)

        # Save checkpoint
        ckpt_data = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "best_char_acc": best_char_acc,
            "best_seq_acc": best_seq_acc
        }

        # Latest checkpoint
        _atomic_save(ckpt_data, os.path.join(run_dir, "checkpoint_latest.pt"))

        # Best checkpoint
        if improved:
            _atomic_save(ckpt_data, os.path.join(run_dir, "seg_mlp_best.pt"))
            emit({"event": "new_best", "epoch": epoch, "char_acc": val_char_acc})
            logger.log_checkpoint(
                kind="best",
                path="seg_mlp_best.pt",
                epoch=epoch,
                metrics={"char_acc": val_char_acc, "seq_acc": val_seq_acc}
            )

    # === FINAL ===
    # Save final model
    final_data = {
        "model": model.state_dict(),
        "config": config,
        "best_char_acc": best_char_acc,
        "best_seq_acc": best_seq_acc,
        "num_classes": num_classes
    }
    _atomic_save(final_data, os.path.join(run_dir, "seg_mlp_final.pt"))

    emit({"event": "finished", "best_char_acc": best_char_acc, "best_seq_acc": best_seq_acc})
    logger.log_status("FINISHED", f"Training completed. Best char acc: {best_char_acc:.2f}%, seq acc: {best_seq_acc:.2f}%")
