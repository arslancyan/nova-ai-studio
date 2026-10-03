"""Sample NOVA latent diffusion and decode it with the native autoencoder.

Requires a trained NOVA autoencoder and latent denoiser checkpoint.
"""
from __future__ import annotations

import torch

from .autoencoder import build_autoencoder
from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode


@torch.no_grad()
def generate(
    autoencoder_checkpoint,
    latent_checkpoint,
    prompt,
    output="nova_latent_sample.pt",
    frames=8,
    height=32,
    width=32,
    timesteps=1000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ae_pack = torch.load(
        autoencoder_checkpoint, map_location=device, weights_only=True
    )
    autoencoder = build_autoencoder(
        in_channels=3,
        latent_channels=int(ae_pack.get("latent_channels", 8)),
    ).to(device)
    autoencoder.load_state_dict(ae_pack["state_dict"])
    autoencoder.eval()

    latent_pack = torch.load(
        latent_checkpoint, map_location=device, weights_only=True
    )
    latent_channels = int(latent_pack["latent_channels"])
    latent_tokens = int(latent_pack["latent_tokens"])

    model = build_latent_model(
        latent_channels=latent_channels,
        latent_tokens=latent_tokens,
    ).to(device)
    model.load_state_dict(latent_pack["state_dict"])
    model.eval()

    text_ids = torch.tensor([encode(prompt)], device=device)
    diffusion = GaussianDiffusion(timesteps)

    latent_frames = max(1, frames // 4)
    latent_height = max(1, height // 4)
    latent_width = max(1, width // 4)

    x = torch.randn(
        1,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
        device=device,
    )

    for step in reversed(range(timesteps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        x = diffusion.step(model, x, text_ids, t)

    video = autoencoder.decode(x).clamp(-1, 1)
    torch.save(video.cpu(), output)
    return video


if __name__ == "__main__":
    generate(
        "checkpoints/nova_ae.pt",
        "checkpoints/nova_latent.pt",
        "a red circle moving in a dark scene",
    )
