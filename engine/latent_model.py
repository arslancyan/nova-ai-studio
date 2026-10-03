"""NOVA native latent video diffusion model.

The model is initialized from random weights. It uses factorized spatial and
temporal attention so the first native training stage is practical on modest
GPU memory without relying on a pretrained video backbone.
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
    """Text-conditioned denoiser with factorized spatial/temporal attention."""

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
        latent_frames: int = 2,
        latent_height: int = 8,
        latent_width: int = 8,
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_tokens = latent_tokens
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width

        expected_tokens = latent_frames * latent_height * latent_width
        if expected_tokens != latent_tokens:
            raise ValueError(
                f"latent_tokens={latent_tokens} does not match "
                f"{latent_frames}x{latent_height}x{latent_width}"
            )

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
        self.spatial_position = nn.Parameter(
            torch.zeros(1, latent_height * latent_width, model_dim)
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, latent_frames, model_dim)
        )
        nn.init.normal_(self.spatial_position, std=0.02)
        nn.init.normal_(self.temporal_position, std=0.02)

        def make_layer():
            return nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                batch_first=True,
                norm_first=True,
            )

        self.spatial_transformer = nn.TransformerEncoder(
            make_layer(), num_layers=num_layers
        )
        self.temporal_transformer = nn.TransformerEncoder(
            make_layer(), num_layers=max(1, num_layers // 2)
        )
        self.text_cross_attention = nn.MultiheadAttention(
            model_dim, num_heads, batch_first=True
        )
        self.text_projection = nn.Linear(text_dim, model_dim)

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
        if (
            frames != self.latent_frames
            or height != self.latent_height
            or width != self.latent_width
        ):
            raise ValueError(
                "Latent shape does not match model configuration: "
                f"got {(frames, height, width)}, expected "
                f"{(self.latent_frames, self.latent_height, self.latent_width)}"
            )

        spatial_tokens = height * width
        tokens = x.permute(0, 2, 3, 4, 1).reshape(
            batch * frames, spatial_tokens, dim
        )
        tokens = tokens + self.spatial_position
        tokens = self.spatial_transformer(tokens)
        tokens = tokens.reshape(batch, frames, spatial_tokens, dim)

        tokens = tokens + self.temporal_position[:, :, None, :]
        temporal = tokens.permute(0, 2, 1, 3).reshape(
            batch * spatial_tokens, frames, dim
        )
        temporal = self.temporal_transformer(temporal)
        tokens = temporal.reshape(batch, spatial_tokens, frames, dim).permute(
            0, 2, 1, 3
        )

        text = self.text_embedding(text_ids)
        text = self.text_encoder(text)
        text = self.text_projection(text)
        text_mask = text_ids.eq(0)
        tokens_flat = tokens.reshape(batch, frames * spatial_tokens, dim)
        attended, _ = self.text_cross_attention(
            tokens_flat, text, text, key_padding_mask=text_mask
        )

        time = self.time_embedding(timesteps)
        pooled = self.text_embedding(text_ids)
        mask = text_ids.ne(0).unsqueeze(-1)
        pooled = (pooled * mask).sum(1) / mask.sum(1).clamp_min(1)
        cond = self.condition(torch.cat([pooled, time], dim=1))
        attended = attended + cond.unsqueeze(1)

        out = self.out_proj(self.out_norm(attended))
        return out.transpose(1, 2).reshape(
            batch, self.latent_channels, frames, height, width
        )


def build_latent_model(**kwargs) -> NovaLatentVideoModel:
    return NovaLatentVideoModel(**kwargs)
