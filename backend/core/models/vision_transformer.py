"""M3 - Vision Transformer for Math Expression Recognition"""
import torch
import torch.nn as nn
from typing import List


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them."""

    def __init__(self, img_h=64, img_w=512, patch_size=16, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches_h = img_h // patch_size
        self.num_patches_w = img_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 1, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformerSeq2Seq(nn.Module):
    """M3 - Vision Transformer with sequence decoder."""

    def __init__(
        self,
        vocab_size: int,
        img_h: int = 64,
        img_w: int = 512,
        patch_size: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_h, img_w, patch_size, d_model)

        # Positional encoding for patches
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)

        # Token embedding for decoder
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(128, d_model) * 0.02)  # Max seq len

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 0

    def forward(self, x, tgt):
        """
        x: (B, 1, H, W) - images
        tgt: (B, T) - target tokens with BOS/EOS
        """
        # Encode image
        patches = self.patch_embed(x)  # (B, num_patches, d_model)
        patches = patches + self.pos_embed  # Add positional encoding
        src = patches.transpose(0, 1)  # (num_patches, B, d_model)

        # Decode
        tgt_emb = self.token_embed(tgt)  # (B, T, d_model)
        seq_len = tgt.size(1)
        tgt_emb = tgt_emb + self.pos_encoding[:seq_len].unsqueeze(0)
        tgt_seq = tgt_emb.transpose(0, 1)  # (T, B, d_model)

        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Transformer
        output = self.transformer(src, tgt_seq, tgt_mask=tgt_mask)  # (T, B, d_model)

        # Project to vocab
        logits = self.fc_out(output)  # (T, B, vocab_size)

        return logits

    @torch.no_grad()
    def greedy_decode(self, x, max_len=128):
        """
        Greedy decoding for inference.
        x: (B, 1, H, W)
        Returns: List of token sequences
        """
        self.eval()
        B = x.size(0)
        device = x.device

        # Encode
        patches = self.patch_embed(x) + self.pos_embed
        src = patches.transpose(0, 1)

        # Start with BOS
        tgt_tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(max_len):
            # Decode
            tgt_emb = self.token_embed(tgt_tokens) + self.pos_encoding[:tgt_tokens.size(1)].unsqueeze(0)
            tgt_seq = tgt_emb.transpose(0, 1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tokens.size(1)).to(device)

            output = self.transformer(src, tgt_seq, tgt_mask=tgt_mask)
            logits = self.fc_out(output[-1])  # Last position

            next_token = logits.argmax(dim=-1)  # (B,)

            # Check for EOS
            if (next_token == self.eos_id).all():
                break

            tgt_tokens = torch.cat([tgt_tokens, next_token.unsqueeze(1)], dim=1)

        # Convert to token lists
        results = []
        for b in range(B):
            tokens = tgt_tokens[b].tolist()
            # Remove BOS, EOS, PAD
            tokens = [t for t in tokens if t not in (self.bos_id, self.eos_id, self.pad_id)]
            results.append(tokens)

        return results
