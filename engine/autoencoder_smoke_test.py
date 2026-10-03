"""Smoke test the native NOVA video autoencoder."""
import torch
from .autoencoder import build_autoencoder

def main():
    model=build_autoencoder()
    video=torch.randn(2,3,8,32,32)
    recon,latent=model(video)
    assert latent.shape==(2,8,2,8,8), latent.shape
    assert recon.shape==video.shape, recon.shape
    loss=torch.nn.functional.mse_loss(recon,video); loss.backward()
    print("NOVA autoencoder smoke test: PASS")
    print("latent:",tuple(latent.shape),"loss:",float(loss))

if __name__=="__main__": main()
