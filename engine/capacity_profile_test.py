"""Regression tests for NOVA's staged native capacity profiles."""
from .capacity_profiles import get_capacity_profile
from .latent_model import build_latent_model


def test_profiles_are_monotonic_in_capacity():
    micro = get_capacity_profile("micro")
    base = get_capacity_profile("creator-16f-base")
    large = get_capacity_profile("creator-16f-large")
    assert micro.model_dim < base.model_dim < large.model_dim
    assert micro.num_layers < base.num_layers < large.num_layers
    assert micro.diffusion_steps < base.diffusion_steps == large.diffusion_steps


def test_large_profile_builds_with_small_latent_shape():
    profile = get_capacity_profile("creator-16f-large")
    model = build_latent_model(
        latent_channels=8,
        latent_tokens=32,
        latent_frames=2,
        latent_height=4,
        latent_width=4,
        model_dim=profile.model_dim,
        text_dim=profile.text_dim,
        num_heads=profile.num_heads,
        num_layers=profile.num_layers,
    )
    assert sum(p.numel() for p in model.parameters()) > 0
