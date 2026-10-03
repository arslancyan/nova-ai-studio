"""NOVA-0 sampling entry point.

This loads only a NOVA checkpoint produced by engine/train.py.
"""

import torch

from .config import NovaConfig
from .model import build_model
from .diffusion import GaussianDiffusion
from .tokenizer import encode, vocab_size

@torch.no_grad()
def generate(checkpoint, prompt, output="nova_sample.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = NovaConfig(**pack["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()

    ids = torch.tensor([encode(prompt, cfg.max_text_tokens)], device=device)
    diffusion = GaussianDiffusion(cfg.timesteps)

    x = torch.randn(
        1, cfg.channels, cfg.frames, cfg.height, cfg.width, device=device
    )

    # Research sampler: fewer steps can be used later after validation.
    for step in reversed(range(cfg.timesteps)):
        t = torch.full((1,), step, device=device, dtype=torch.long)
        x = diffusion.step(model, x, ids, t)

    torch.save(x.cpu(), output)
    return x

if __name__ == "__main__":
    generate("checkpoints/nova0.pt", "a small cinematic city at night")
