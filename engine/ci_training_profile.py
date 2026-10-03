from .synthetic_dataset import build_dataset
from .train_autoencoder import train

def main():
    build_dataset(samples=16, frames=8, height=32, width=32, output='data/ci_synthetic')
    train(root='data/ci_synthetic', epochs=1, batch_size=4, lr=2e-4, val_fraction=0.25, seed=123, out='checkpoints/ci_nova_ae.pt')
    print('ci native training profile complete')

if __name__ == '__main__':
    main()
