"""Native NOVA latent diffusion sampler.

Requires trained NOVA autoencoder and latent denoiser checkpoints.
The sampler is intentionally separate from the web renderer so model
experiments remain reproducible and do not silently fall back to demo output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

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
    seed=0,
):
    if frames < 4 or height < 4 or width < 4:
        raise ValueError("frames/height/width must be at least 4")
    if frames % 4 or height % 4 or width % 4:
        raise ValueError("frames/height/width must be divisible by 4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)

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

    latent_frames = frames // 4
    latent_height = height // 4
    latent_width = width // 4

    expected_tokens = latent_frames * latent_height * latent_width
    if expected_tokens != latent_tokens:
        raise ValueError(
            "Checkpoint latent shape does not match requested output: "
            f"checkpoint tokens={latent_tokens}, requested={expected_tokens}"
        )

    model = build_latent_model(
        latent_channels=latent_channels,
        latent_tokens=latent_tokens,
        latent_frames=latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
    ).to(device)
    model.load_state_dict(latent_pack["state_dict"])
    model.eval()

    text_ids = torch.tensor([encode(prompt)], device=device)
    diffusion = GaussianDiffusion(timesteps)

    x = torch.randn(
        1,
        latent_channels,
        latent_frames,
        latent_height,
        latent_width,
        device=device,
        generator=generator,
    )

    for step in reversed(range(timesteps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        x = diffusion.step(model, x, text_ids, t)

    video = autoencoder.decode(x).clamp(-1, 1)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(video.cpu(), output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autoencoder", default="checkpoints/nova_ae.pt")
    parser.add_argument("--latent", default="checkpoints/nova_latent.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="outputs/nova_native.pt")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    path = generate(
        args.autoencoder,
        args.latent,
        args.prompt,
        args.output,
        args.frames,
        args.height,
        args.width,
        args.timesteps,
        args.seed,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
