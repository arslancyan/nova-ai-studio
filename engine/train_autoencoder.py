"""Train the native NOVA video autoencoder from random initialization."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from .autoencoder import build_autoencoder

def train(root="data/synthetic",epochs=5,batch_size=8,lr=2e-4,out="checkpoints/nova_ae.pt"):
    root=Path(root)
    clips=torch.load(root/"clips.pt",map_location="cpu",weights_only=True).float()
    if clips.ndim!=5 or clips.shape[1]!=3: raise ValueError("Expected [N,3,T,H,W] clips")
    if not torch.isfinite(clips).all(): raise ValueError("Dataset contains NaN or infinity")
    model=build_autoencoder(); device="cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    loader=DataLoader(TensorDataset(clips),batch_size=batch_size,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    for epoch in range(epochs):
        total=0.0
        for (video,) in loader:
            video=video.to(device); recon,_=model(video)
            loss=torch.nn.functional.l1_loss(recon,video)+0.1*torch.nn.functional.mse_loss(recon,video)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=float(loss)
        print(f"epoch={epoch+1} loss={total/max(1,len(loader)):.6f}")
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"latent_channels":8},out)
    print(f"saved {out}")

if __name__=="__main__": train()
