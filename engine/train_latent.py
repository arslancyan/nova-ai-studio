"""Train NOVA latent diffusion after the autoencoder is trained."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .autoencoder import build_autoencoder
from .diffusion import GaussianDiffusion
from .latent_model import build_latent_model
from .tokenizer import encode, vocab_size


def train(
    data_root="data/synthetic",
    autoencoder_checkpoint="checkpoints/nova_ae.pt",
    epochs=5,
    batch_size=8,
    lr=2e-4,
    out="checkpoints/nova_latent.pt",
):
    root = Path(data_root)
    clips = torch.load(
        root / "clips.pt", map_location="cpu", weights_only=True
    ).float()
    captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()

    if clips.ndim != 5 or clips.shape[1] != 3:
        raise ValueError("Expected [N,3,T,H,W] clips")
    if len(captions) != clips.shape[0]:
        raise ValueError("Caption count must match clip count")
    if not torch.isfinite(clips).all():
        raise ValueError("Dataset contains NaN or infinity")

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
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        sample_latent = autoencoder.encode(clips[:1].to(device))
    latent_channels = sample_latent.shape[1]
    latent_tokens = sample_latent[0].numel() // latent_channels

    model = build_latent_model(
        latent_channels=latent_channels,
        text_vocab_size=vocab_size(),
        latent_tokens=latent_tokens,
    ).to(device)

    diffusion = GaussianDiffusion(steps=1000)
    text_ids = torch.tensor(
        [encode(caption) for caption in captions], dtype=torch.long
    )
    loader = DataLoader(
        TensorDataset(clips, text_ids),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        total = 0.0
        for video, ids in loader:
            video = video.to(device)
            ids = ids.to(device)
            with torch.no_grad():
                clean_latent = autoencoder.encode(video)

            timestep = torch.randint(
                0, diffusion.steps, (video.shape[0],), device=device
            )
            noisy, noise = diffusion.q_sample(clean_latent, timestep)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                predicted = model(noisy, ids, timestep)
                loss = torch.nn.functional.mse_loss(predicted, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss)

        mean_loss = total / max(1, len(loader))
        print(f"epoch={epoch + 1} loss={mean_loss:.6f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "latent_channels": latent_channels,
            "latent_tokens": latent_tokens,
        },
        out,
    )
    print(f"saved {out}")


if __name__ == "__main__":
    train()
