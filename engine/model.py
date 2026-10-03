"""NOVA-0: tiny text-conditioned video diffusion model.

All trainable weights are created by PyTorch with standard random
initialization. No pretrained checkpoint is loaded here.
"""

import math
import torch
import torch.nn as nn

from .config import NovaConfig

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        x = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return emb

class NovaVideoModel(nn.Module):
    """Small pixel-space denoiser for short research clips."""

    def __init__(self, cfg: NovaConfig):
        super().__init__()
        self.cfg = cfg

        self.text_embedding = nn.Embedding(cfg.vocab_size, cfg.text_dim, padding_idx=0)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.text_dim,
                nhead=4,
                dim_feedforward=cfg.text_dim * 4,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )

        self.time_embedding = SinusoidalTimeEmbedding(cfg.model_dim)
        self.condition = nn.Linear(cfg.text_dim + cfg.model_dim, cfg.model_dim)

        self.in_proj = nn.Conv3d(cfg.channels, cfg.model_dim, kernel_size=3, padding=1)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.model_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.video_encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

        self.out_norm = nn.GroupNorm(8, cfg.model_dim)
        self.out_proj = nn.Conv3d(cfg.model_dim, cfg.channels, kernel_size=3, padding=1)

    def forward(self, video, text_ids, timesteps):
        # video: [B, C, T, H, W]
        x = self.in_proj(video)
        b, d, t, h, w = x.shape

        text = self.text_embedding(text_ids)
        text = self.text_encoder(text)
        mask = text_ids.ne(0).unsqueeze(-1)
        pooled = (text * mask).sum(1) / mask.sum(1).clamp_min(1)

        time = self.time_embedding(timesteps)
        cond = self.condition(torch.cat([pooled, time], dim=1))

        # Flatten spatiotemporal positions for a compact global attention block.
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + cond.unsqueeze(1)
        tokens = self.video_encoder(tokens)

        x = tokens.transpose(1, 2).reshape(b, d, t, h, w)
        x = self.out_norm(x)
        return self.out_proj(x)

def build_model(cfg=None):
    cfg = cfg or NovaConfig()
    return NovaVideoModel(cfg)
