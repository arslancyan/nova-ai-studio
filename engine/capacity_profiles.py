"""Named NOVA native capacity profiles.

All profiles are trained from random initialization. A profile only defines
architecture/training targets; it does not ship model weights.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapacityProfile:
    name: str
    model_dim: int
    text_dim: int
    num_heads: int
    num_layers: int
    diffusion_steps: int
    output_frames: int
    output_height: int
    output_width: int
    notes: str


PROFILES = {
    "micro": CapacityProfile(
        "micro", 32, 32, 4, 1, 8, 4, 16, 16,
        "CPU regression profile; not a quality target.",
    ),
    "creator-16f-base": CapacityProfile(
        "creator-16f-base", 128, 128, 4, 4, 1000, 16, 64, 64,
        "First creator-owned quality bridge profile.",
    ),
    "creator-16f-large": CapacityProfile(
        "creator-16f-large", 192, 192, 6, 6, 1000, 16, 64, 64,
        "GPU scale-up after held-out reconstruction improves.",
    ),
    "creator-24f-large": CapacityProfile(
        "creator-24f-large", 192, 192, 6, 6, 1000, 24, 64, 64,
        "Longer temporal context; requires a matching 24-frame AE checkpoint.",
    ),
}


def get_capacity_profile(name: str) -> CapacityProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(PROFILES)
        raise ValueError(f"unknown NOVA capacity profile {name!r}; choose {choices}") from exc
