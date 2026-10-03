from dataclasses import dataclass

@dataclass
class NovaConfig:
    vocab_size: int = 96
    text_dim: int = 128
    model_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    frames: int = 8
    height: int = 32
    width: int = 32
    channels: int = 3
    timesteps: int = 1000
    max_text_tokens: int = 96
