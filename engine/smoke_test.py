"""NOVA-0 architecture smoke test.

Runs a single forward/backward pass on synthetic tensors. This does not train
on external data and is safe to use for CI architecture validation.
"""
import torch
from .config import NovaConfig
from .model import build_model
from .diffusion import GaussianDiffusion
from .tokenizer import vocab_size

def main():
    cfg = NovaConfig(
        vocab_size=vocab_size(),
        model_dim=64,
        text_dim=64,
        num_heads=4,
        num_layers=2,
        frames=4,
        height=16,
        width=16,
    )
    assert cfg.vocab_size == vocab_size()
    model = build_model(cfg)
    diffusion = GaussianDiffusion(cfg.timesteps)

    video = torch.randn(2, cfg.channels, cfg.frames, cfg.height, cfg.width)
    text = torch.randint(2, cfg.vocab_size, (2, cfg.max_text_tokens))
    t = torch.randint(0, cfg.timesteps, (2,))
    noisy, noise = diffusion.q_sample(video, t)
    pred = model(noisy, text, t)
    loss = torch.nn.functional.mse_loss(pred, noise)
    loss.backward()

    assert pred.shape == video.shape
    print("NOVA-0 smoke test: PASS")
    print("input:", tuple(video.shape))
    print("output:", tuple(pred.shape))
    print("loss:", float(loss))

if __name__ == "__main__":
    main()
