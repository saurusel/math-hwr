# backend/core/models/attn_seq2seq.py
# -*- coding: utf-8 -*-
"""
Минимальный encoder-decoder с вниманием (greedy-инференс).
Цель: совместимый вывод со схемой токенов как у CRNN-CTC (space separated).
Только инференс (обучение добавим отдельным шагом).

Зависимости: torch, torchvision (для нормализации можно и без неё)
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Вокаб ----
# Вокаб должен включать спец-токены:
# PAD=0, BOS=1, EOS=2. Остальные — обычные токены из твоего набора (цифры, x,y,z, +, -, ^, (, ), =, ÷, *, и т.п.).
# В реальном чекпоинте мы будем хранить токены, а здесь дадим заглушку + возможность загрузить из state_dict.
SPECIAL_TOKENS = {
    "PAD": 0,
    "BOS": 1,
    "EOS": 2,
}

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def build_from_tokens(cls, tokens: List[str]) -> "Vocab":
        # ожидаем, что tokens не включает спец-символы, добавим их в начало
        itos = ["<PAD>", "<BOS>", "<EOS>"] + tokens
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    @property
    def pad_id(self) -> int: return SPECIAL_TOKENS["PAD"]
    @property
    def bos_id(self) -> int: return SPECIAL_TOKENS["BOS"]
    @property
    def eos_id(self) -> int: return SPECIAL_TOKENS["EOS"]

    def encode(self, toks: List[str]) -> List[int]:
        return [self.stoi.get(t, self.stoi.get("<UNK>", self.pad_id)) for t in toks]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<PAD>" for i in ids]


# ---- Простейший CNN-энкодер → фичи (B, C, H, W) ----
class CNNEncoder(nn.Module):
    def __init__(self, in_ch=1, feat_ch=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(256, feat_ch, 3, 1, 1), nn.ReLU(inplace=True),
            # высоту уменьшаем ещё раз, ширину оставляем для "времени"
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        f = self.conv(x)  # (B, C, H', W')
        return f


# ---- Внимание (Luong dot) ----
class DotAttention(nn.Module):
    def __init__(self, dec_dim: int, enc_dim: int):
        super().__init__()
        # приведём размеры к одному пространству
        self.proj_enc = nn.Linear(enc_dim, dec_dim, bias=False)

    def forward(self, dec_h: torch.Tensor, enc_seq: torch.Tensor, enc_mask: Optional[torch.Tensor] = None):
        """
        dec_h: (B, dec_dim) — текущее скрытое состояние декодера
        enc_seq: (B, T, enc_dim)
        enc_mask: (B, T) 1=valid, 0=pad (опционально)
        """
        B, T, E = enc_seq.size()
        proj = self.proj_enc(enc_seq)             # (B, T, dec_dim)
        scores = torch.bmm(proj, dec_h.unsqueeze(2)).squeeze(2)  # (B, T)
        if enc_mask is not None:
            scores = scores.masked_fill(enc_mask == 0, -1e9)
        attn = F.softmax(scores, dim=1)           # (B, T)
        ctx = torch.bmm(attn.unsqueeze(1), enc_seq).squeeze(1)   # (B, enc_dim)
        return ctx, attn


# ---- Декодер (LSTM + attention) ----
class AttnDecoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, dec_dim: int, enc_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTMCell(emb_dim + enc_dim, dec_dim)
        self.attn = DotAttention(dec_dim, enc_dim)
        self.out = nn.Linear(dec_dim + enc_dim, vocab_size)

    def forward_step(
        self,
        y_prev: torch.Tensor,     # (B,) id предыдущего токена
        h: torch.Tensor, c: torch.Tensor,
        enc_seq: torch.Tensor,    # (B, T, enc_dim)
        enc_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(y_prev)              # (B, emb_dim)
        # внимание на текущем h
        ctx, attn = self.attn(h, enc_seq, enc_mask)   # ctx: (B, enc_dim)
        x = torch.cat([emb, ctx], dim=-1)         # (B, emb_dim+enc_dim)
        h, c = self.lstm(x, (h, c))               # (B, dec_dim), (B, dec_dim)
        logits = self.out(torch.cat([h, ctx], dim=-1))  # (B, vocab_size)
        return logits, h, c, attn


# ---- Полная модель ----
class AttnSeq2Seq(nn.Module):
    def __init__(self, vocab: Vocab, img_h: int = 64, enc_ch: int = 256, emb_dim: int = 128, dec_dim: int = 256):
        super().__init__()
        self.vocab = vocab
        self.img_h = img_h
        self.encoder = CNNEncoder(in_ch=1, feat_ch=enc_ch)
        self.enc_dim = enc_ch
        self.emb_dim = emb_dim
        self.dec_dim = dec_dim
        self.decoder = AttnDecoder(vocab_size=len(vocab.itos), emb_dim=emb_dim, dec_dim=dec_dim, enc_dim=self.enc_dim)

    def forward_encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,1,H,W)
        return:
          enc_seq: (B, T, C) — T = ширина признаков, C=enc_dim
          enc_mask: (B, T) — здесь все валидно (если нет паддинга справа, можно 1s)
        """
        feats = self.encoder(x)        # (B, C, H', W')
        # "сворачиваем" по высоте: среднее по H'
        feats = feats.mean(dim=2)      # (B, C, W')
        enc_seq = feats.permute(0, 2, 1).contiguous()  # (B, T, C)
        enc_mask = torch.ones(enc_seq.size(0), enc_seq.size(1), device=enc_seq.device, dtype=torch.bool)
        return enc_seq, enc_mask

    @torch.no_grad()
    def greedy_decode(self, x: torch.Tensor, max_len: int = 128) -> List[List[str]]:
        """
        x: (B,1,H,W)
        return: список списков токенов (без BOS/EOS)
        """
        self.eval()
        enc_seq, enc_mask = self.forward_encoder(x)  # (B,T,C),(B,T)
        B = x.size(0)
        device = x.device

        h = torch.zeros(B, self.dec_dim, device=device)
        c = torch.zeros(B, self.dec_dim, device=device)

        y_prev = torch.full((B,), fill_value=self.vocab.bos_id, dtype=torch.long, device=device)
        results: List[List[int]] = [[] for _ in range(B)]

        for _ in range(max_len):
            logits, h, c, _ = self.decoder.forward_step(y_prev, h, c, enc_seq, enc_mask)
            next_ids = torch.argmax(logits, dim=-1)  # (B,)
            for b in range(B):
                tok = next_ids[b].item()
                if tok == self.vocab.eos_id:
                    # закончили гипотезу
                    pass
                else:
                    results[b].append(tok)
            y_prev = next_ids
            # если все закончили на EOS — можно прервать (упрощенно игнорим, пусть добегает макс-длину)

        # декодим в строку: пропускаем PAD/BOS/EOS
        out_tokens: List[List[str]] = []
        for ids in results:
            toks = []
            for i in ids:
                if i in (self.vocab.pad_id, self.vocab.bos_id, self.vocab.eos_id):
                    continue
                toks.append(self.vocab.itos[i])
            out_tokens.append(toks)
        return out_tokens

    # ---- Загрузка чекпоинта ----
    @classmethod
    def from_checkpoint(cls, ckpt_path: str, device: torch.device, default_tokens: Optional[List[str]] = None, img_h: int = 64) -> "AttnSeq2Seq":
        """
        Ожидаемый формат чекпоинта (рекомендуется):
        {
            "model_state": ...,
            "vocab_tokens": [...],   # без спец-символов (они добавятся)
            "img_h": 64,
            "meta": { ... }
        }
        Если полей нет — используем default_tokens.
        """
        state = torch.load(ckpt_path, map_location="cpu")
        tokens = state.get("vocab_tokens", None)
        if tokens is None:
            if default_tokens is None:
                # минимальный набор на всякий случай; лучше всегда класть vocab_tokens в чекпоинт
                tokens = list("0123456789") + ["x","y","z","+","-","*","÷","^","(",")","="]
            else:
                tokens = default_tokens

        vocab = Vocab.build_from_tokens(tokens)
        model = cls(vocab=vocab, img_h=state.get("img_h", img_h))
        model.load_state_dict(state.get("model_state", state), strict=False)
        model.to(device)
        model.eval()
        return model
