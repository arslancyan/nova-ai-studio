"""Minimal NOVA-0 training entry point.

Dataset format:
data/
  clips.pt      # tensor [N, C, T, H, W], values in [-1, 1]
  captions.txt  # one caption per line

Only use video data with documented rights for the NOVA dataset.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .config import NovaConfig
from .model import build_model
from .diffusion import GaussianDiffusion
from .tokenizer import encode, vocab_size

class NovaDataset(Dataset):
    def __init__(self, root="data", max_text_tokens=96):
        root = Path(root)
        self.clips = torch.load(root / "clips.pt", map_location="cpu")
        self.captions = (root / "captions.txt").read_text(encoding="utf-8").splitlines()
        if len(self.clips) != len(self.captions):
            raise ValueError("clips.pt and captions.txt must contain the same number of samples.")
        self.max_text_tokens = max_text_tokens

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        ids = torch.tensor(encode(self.captions[i], self.max_text_tokens), dtype=torch.long)
        return self.clips[i].float(), ids

def train(epochs=1, batch_size=2, lr=2e-4, root="data", out="checkpoints/nova0.pt"):
    cfg = NovaConfig(vocab_size=vocab_size())
    model = build_model(cfg)
    diffusion = GaussianDiffusion(cfg.timesteps)
    loader = DataLoader(NovaDataset(root), batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for video, text_ids in loader:
            video, text_ids = video.to(device), text_ids.to(device)
            t = torch.randint(0, cfg.timesteps, (video.size(0),), device=device)
            noisy, noise = diffusion.q_sample(video, t)
            pred = model(noisy, text_ids, t)
            loss = torch.nn.functional.mse_loss(pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        print(f"epoch={epoch + 1} loss={loss.item():.6f}")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, out)
    print(f"saved {out}")

if __name__ == "__main__":
    train()
