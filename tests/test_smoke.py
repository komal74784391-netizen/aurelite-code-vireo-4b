"""Smoke tests for Vireo-4B."""

import torch

from src.model import Vireo4bModel


def test_forward_shape():
    model = Vireo4bModel(vocab_size=64, dim=64, n_layers=2, n_heads=4, hidden=128)
    tokens = torch.randint(0, 64, (2, 16))
    logits = model(tokens)
    assert logits.shape == (2, 16, 64)
