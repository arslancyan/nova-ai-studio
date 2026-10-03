"""NOVA native latent video diffusion model.

The model is initialized from random weights. It operates on the compact
spatiotemporal latent produced by NovaVideoAutoencoder.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device)
            / max(half - 1, 1)
        )
        x = timesteps.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class NovaLatentVideoModel(nn.Module):
    """Small text-conditioned denoiser for NOVA latent video tensors."""

    def __init__(
        self,
        latent_channels: int = 8,
        text_vocab_size: int = 97,
        text_dim: int = 128,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        max_text_tokens: int = 96,
        latent_tokens: int = 1024,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_tokens = latent_tokens

        self.text_embedding = nn.Embedding(
            text_vocab_size, text_dim, padding_idx=0
        )
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_dim,
                nhead=4,
                dim_feedforward=text_dim * 4,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
        )
        self.time_embedding = SinusoidalTimeEmbedding(model_dim)
        self.condition = nn.Linear(text_dim + model_dim, model_dim)

        self.in_proj = nn.Conv3d(latent_channels, model_dim, 1)
        self.position = nn.Parameter(torch.zeros(1, latent_tokens, model_dim))
        nn.init.normal_(self.position, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.video_transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers
        )
        self.out_norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, latent_channels)

    def forward(
        self,
        latent: torch.Tensor,
        text_ids: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = self.in_proj(latent)
        batch, dim, frames, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.latent_tokens:
            raise ValueError(
                f"Expected {self.latent_tokens} latent tokens, got {tokens.shape[1]}"
            )

        text = self.text_embedding(text_ids)
        text = self.text_encoder(text)
        mask = text_ids.ne(0).unsqueeze(-1)
        pooled = (text * mask).sum(1) / mask.sum(1).clamp_min(1)

        time = self.time_embedding(timesteps)
        cond = self.condition(torch.cat([pooled, time], dim=1))

        tokens = tokens + self.position + cond.unsqueeze(1)
        tokens = self.video_transformer(tokens)
        tokens = self.out_proj(self.out_norm(tokens))

        return tokens.transpose(1, 2).reshape(
            batch, self.latent_channels, frames, height, width
        )


def build_latent_model(**kwargs) -> NovaLatentVideoModel:
    return NovaLatentVideoModel(**kwargs)
