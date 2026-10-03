"""NOVA Micro: CPU-first native research profile.

Uses only tiny random-init models and synthetic data. This is for fast
engineering validation, not production video quality.
"""
from pathlib import Path

from .synthetic_dataset import build_dataset
from .train_autoencoder import train as train_autoencoder
from .train_latent import train as train_latent


def run():
    root = Path("data/micro")
    ae = Path("checkpoints/nova_micro_ae.pt")
    latent = Path("checkpoints/nova_micro_latent.pt")

    build_dataset(samples=24, frames=4, height=16, width=16, output=root)

    train_autoencoder(
        root=root,
        epochs=2,
        batch_size=4,
        lr=3e-4,
        val_fraction=0.25,
        seed=17,
        out=ae,
    )

    train_latent(
        data_root=root,
        autoencoder_checkpoint=ae,
        epochs=2,
        batch_size=4,
        lr=3e-4,
        val_fraction=0.25,
        seed=17,
        out=latent,
        history_out=Path("checkpoints/nova_micro_latent_history.jsonl"),
        model_dim=32,
        num_heads=4,
        num_layers=1,
        diffusion_steps=8,
    )

    print("NOVA Micro profile complete")
    print(f"autoencoder={ae}")
    print(f"latent={latent}")


if __name__ == "__main__":
    run()
