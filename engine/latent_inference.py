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
    sampler="ddpm",
    sampling_steps=None,
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
        model_dim=int(latent_pack.get("model_dim", 128)),
        text_dim=int(latent_pack.get("text_dim", latent_pack.get("model_dim", 128))),
        num_heads=int(latent_pack.get("num_heads", 4)),
        num_layers=int(latent_pack.get("num_layers", 4)),
    ).to(device)
    model.load_state_dict(latent_pack["state_dict"])
    model.eval()

    text_ids = torch.tensor([encode(prompt)], device=device)
    trained_timesteps = int(latent_pack.get("diffusion_steps", timesteps))
    if timesteps != trained_timesteps:
        timesteps = trained_timesteps
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

    if sampler not in {"ddpm", "ddim"}:
        raise ValueError("sampler must be ddpm or ddim")
    if sampling_steps is None:
        sampling_steps = timesteps if sampler == "ddpm" else min(50, timesteps)
    sampling_steps = max(2, min(int(sampling_steps), timesteps))
    if sampler == "ddpm":
        schedule = list(range(timesteps - 1, -1, -1))
    else:
        schedule = torch.linspace(timesteps - 1, 0, sampling_steps, device=device).round().long().tolist()
    for index, step in enumerate(schedule):
        t = torch.full((1,), int(step), device=device, dtype=torch.long)
        if sampler == "ddim":
            prev_step = int(schedule[index + 1]) if index + 1 < len(schedule) else 0
            prev_t = torch.full((1,), prev_step, device=device, dtype=torch.long)
            x = diffusion.ddim_step(model, x, text_ids, t, prev_t, eta=0.0)
        else:
            x = diffusion.step(model, x, text_ids, t)

    video = autoencoder.decode(x).clamp(-1, 1)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(video.cpu(), output_path)
    return output_path


def build_native_prompt(job: dict) -> str:
    """Flatten a Creative Director job into the conditioning text used by NOVA."""
    director = job.get("director") or {}
    parts = [
        ("Subject", director.get("subject", "")),
        ("Action", director.get("action", "")),
        ("Environment", director.get("environment", "")),
        ("Camera", director.get("camera_movement") or job.get("camera", "auto")),
        ("Lighting", director.get("lighting", "")),
        ("Style", director.get("style", "")),
        ("Motion", director.get("motion", "natural")),
    ]
    structured = ". ".join(
        f"{label}: {value.strip()}" for label, value in parts
        if isinstance(value, str) and value.strip()
    )
    prompt = str(job.get("prompt", "")).strip()
    if structured and prompt:
        return f"{prompt}. {structured}."
    return structured or prompt


@torch.no_grad()
def generate_job(job: dict, output="outputs/nova_native.pt"):
    """Run a validated Creative Director job through the native latent pipeline."""
    prompt = build_native_prompt(job)
    if not prompt:
        raise ValueError("job prompt or director fields are required")

    duration = int(job.get("duration_seconds", 5))
    # The current native prototype operates on a compact 8-frame clip.
    frames = int(job.get("frames", 8))
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")

    seed = job.get("seed")
    seed = int(seed) if seed is not None else 0
    return generate(
        job.get("autoencoder_checkpoint", "checkpoints/nova_ae.pt"),
        job.get("latent_checkpoint", "checkpoints/nova_latent.pt"),
        prompt,
        output=output,
        frames=frames,
        height=int(job.get("height", 32)),
        width=int(job.get("width", 32)),
        timesteps=int(job.get("timesteps", 1000)),
        seed=seed,
        sampler=str(job.get("sampler", "ddpm")),
        sampling_steps=job.get("sampling_steps"),
    )


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
    parser.add_argument("--sampler", choices=["ddpm","ddim"], default="ddpm")
    parser.add_argument("--sampling-steps", type=int, default=None)
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
        args.sampler,
        args.sampling_steps,
    )
    print(f"saved {path}")


if __name__ == "__main__":
    main()
