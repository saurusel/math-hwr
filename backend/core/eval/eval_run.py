
import os, time, json
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.dataset import HWRDataset
from ..data.collate import collate_ctc
from ..models.crnn_ctc import CRNN_CTC
from ..tokenizer import TOKEN_LIST
from ..metrics import cer as cer_fn, wer as wer_fn
from ..parser import is_valid_token_stream
from ..decoders.ctc_decode import ctc_greedy_decode

@torch.no_grad()
def evaluate_crnn_ctc(
    ckpt_path: str,
    val_dir: str,
    img_h: int = 64,
    img_w_max: int = 512,
    batch_size: int = 64,
    max_batches: Optional[int] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Evaluate CRNN+CTC checkpoint on provided val split."""
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")

    ds = HWRDataset(val_dir, img_h=img_h, img_w_max=img_w_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_ctc)

    blank_id = len(TOKEN_LIST)
    num_classes = blank_id + 1
    model = CRNN_CTC(num_classes=num_classes).to(device_t).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        model.load_state_dict(state["model"])
    else:
        # allow loading a bare state dict
        model.load_state_dict(state)

    tot = 0
    cer_sum=0.0; wer_sum=0.0; exact_sum=0.0; valid_sum=0.0
    n_images=0
    t_infer=0.0

    samples: List[Dict[str,Any]] = []

    for b_idx, (x, _, _, tgt_tokens) in enumerate(dl, start=1):
        x = x.to(device_t)

        if device_t.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits, T = model(x)
        if device_t.type == "cuda":
            torch.cuda.synchronize()
        t_infer += (time.time() - t0)

        hyps = ctc_greedy_decode(logits, blank_id)

        B = x.size(0)
        for i in range(B):
            tgt = " ".join(tgt_tokens[i])
            hyp_tokens = [TOKEN_LIST[j] for j in hyps[i]]
            hyp = " ".join(hyp_tokens)
            cer_sum += cer_fn(tgt, hyp)
            wer_sum += wer_fn(tgt_tokens[i], hyp_tokens)
            exact_sum += 1.0 if tgt == hyp else 0.0
            valid_sum += 1.0 if (len(hyp_tokens) > 0 and is_valid_token_stream(hyp_tokens)) else 0.0
            tot += 1

            if len(samples) < 16:
                samples.append({
                    "target": tgt,
                    "pred": hyp,
                    "ok": tgt == hyp
                })

        n_images += B
        if (max_batches is not None) and (b_idx >= max_batches):
            break

    metrics = {
        "cer": cer_sum / max(1, tot),
        "wer": wer_sum / max(1, tot),
        "exact": exact_sum / max(1, tot),
        "valid": valid_sum / max(1, tot),
        "latency_ms_per_image": (t_infer / max(1, n_images)) * 1000.0,
    }
    return {"metrics": metrics, "n_images": n_images, "samples": samples}
