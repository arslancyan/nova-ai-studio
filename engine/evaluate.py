"""Basic NOVA-0 evaluation helpers.

This intentionally starts with engineering metrics rather than pretending
that a tiny untrained model has meaningful visual quality.
"""
import torch

def tensor_health(video):
    return {
        "shape": tuple(video.shape),
        "finite": bool(torch.isfinite(video).all()),
        "mean": float(video.mean()),
        "std": float(video.std()),
        "min": float(video.min()),
        "max": float(video.max()),
    }

def compare_temporal_variation(video):
    if video.shape[2] < 2:
        return 0.0
    return float((video[:, :, 1:] - video[:, :, :-1]).abs().mean())
