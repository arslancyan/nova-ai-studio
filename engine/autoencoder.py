"""NOVA native video autoencoder.

This model is initialized from random weights and is trained only on NOVA data.
It learns a compact spatiotemporal latent representation before diffusion training.
"""
from __future__ import annotations
import torch
import torch.nn as nn

class NovaVideoAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.SiLU(),
            nn.Conv3d(32, 48, 4, stride=(2,2,2), padding=1), nn.SiLU(),
            nn.Conv3d(48, 64, 3, padding=1), nn.SiLU(),
            nn.Conv3d(64, latent_channels, 4, stride=(2,2,2), padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(latent_channels, 64, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(64, 48, 4, stride=(2,2,2), padding=1), nn.SiLU(),
            nn.Conv3d(48, 32, 3, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(32, in_channels, 4, stride=(2,2,2), padding=1), nn.Tanh(),
        )

    def encode(self, video):
        return self.encoder(video)

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, video):
        latent = self.encode(video)
        reconstruction = self.decode(latent)
        return reconstruction, latent

def build_autoencoder(in_channels=3, latent_channels=8):
    return NovaVideoAutoencoder(in_channels, latent_channels)
