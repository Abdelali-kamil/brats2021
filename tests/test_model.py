"""Correctness tests for the segmentation model and its norm options.

Performance is not tested here (needs data + GPU); these check that the model
builds, runs a forward pass at the expected output shape, and that the default
configuration is unchanged so existing checkpoints still load.

Run with:  python -m pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from brats_gbm.model import WaveletUNetPlusPlus


@pytest.mark.parametrize("norm", ["batch", "instance", "group"])
def test_forward_shape(norm):
    model = WaveletUNetPlusPlus(in_channels=4, n_classes=3, norm=norm).eval()
    x = torch.randn(1, 4, 32, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 3, 32, 32, 32)


def test_default_is_batchnorm_and_keys_stable():
    """The default must stay BatchNorm so previously-saved checkpoints load."""
    model = WaveletUNetPlusPlus()
    keys = model.state_dict().keys()
    # BatchNorm tracks running_mean/running_var; their presence is the tell.
    assert any(k.endswith("running_mean") for k in keys)
    assert any(k.endswith("running_var") for k in keys)


def test_instance_norm_has_no_running_stats():
    model = WaveletUNetPlusPlus(norm="instance")
    keys = model.state_dict().keys()
    assert not any(k.endswith("running_mean") for k in keys)


def test_unknown_norm_raises():
    with pytest.raises(ValueError):
        WaveletUNetPlusPlus(norm="layernorm")
