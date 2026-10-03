"""End-to-end CPU regression: train NOVA Micro, sample from a prompt, export MP4."""
from pathlib import Path
import tempfile
from .micro_profile import run
from .latent_inference import generate
from .export_video import export_video

def main():
    run()
    with tempfile.TemporaryDirectory() as tmp:
        tensor = generate(
            "checkpoints/nova_micro_ae.pt",
            "checkpoints/nova_micro_latent.pt",
            "a red circle moving in a dark scene",
            output=Path(tmp) / "sample.pt",
            frames=4, height=16, width=16, seed=23,
        )
        mp4 = export_video(str(tensor), str(Path(tmp) / "sample.mp4"), fps=4)
        assert mp4.exists() and mp4.stat().st_size > 0
        print("NOVA Micro prompt-to-native-video regression passed")

if __name__ == "__main__":
    main()
