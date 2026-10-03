"""End-to-end CPU regression: reuse/train NOVA Micro, compare prompts, export MP4."""
from pathlib import Path
import tempfile

from .micro_profile import run
from .latent_inference import generate
from .export_video import export_video


AE_CHECKPOINT = Path("checkpoints/nova_micro_ae.pt")
LATENT_CHECKPOINT = Path("checkpoints/nova_micro_latent.pt")


def ensure_checkpoints():
    """Train once when CI/local checkpoints are missing; otherwise reuse them."""
    if AE_CHECKPOINT.exists() and LATENT_CHECKPOINT.exists():
        print("Reusing existing NOVA Micro checkpoints")
        return
    run()


def main():
    ensure_checkpoints()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        red = generate(
            str(AE_CHECKPOINT),
            str(LATENT_CHECKPOINT),
            "a red circle moving in a dark scene",
            output=root / "red.pt",
            frames=4,
            height=16,
            width=16,
            seed=23,
        )
        blue = generate(
            str(AE_CHECKPOINT),
            str(LATENT_CHECKPOINT),
            "a blue square moving in a dark scene",
            output=root / "blue.pt",
            frames=4,
            height=16,
            width=16,
            seed=23,
        )

        # Same seed + different prompt should produce different native tensors.
        red_bytes = red.read_bytes()
        blue_bytes = blue.read_bytes()
        assert red_bytes != blue_bytes, "Prompt conditioning produced identical outputs"

        mp4 = export_video(
            str(red),
            str(root / "sample.mp4"),
            fps=4,
        )
        assert mp4.exists() and mp4.stat().st_size > 0

        print("NOVA Micro prompt-conditioning + native-video regression passed")


if __name__ == "__main__":
    main()
