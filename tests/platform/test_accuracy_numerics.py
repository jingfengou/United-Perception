"""Accuracy comparisons must not silently inherit a different numerical policy."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from accuracy_train import CONTROLS, configure_numerics


@pytest.fixture(autouse=True)
def restore_policy(monkeypatch):
    for key in CONTROLS:
        monkeypatch.delenv(key, raising=False)
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    yield
    threads, deterministic, warn_only, benchmark, cudnn, convolution_tf32, matmul_tf32 = previous
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = cudnn
    torch.backends.cudnn.allow_tf32 = convolution_tf32
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def test_native_explicitly_disables_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    configure_numerics()
    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.deterministic


def test_strict_determinism_keeps_fp32_policy(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    configure_numerics(True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32


def test_strict_determinism_requires_workspace_before_cuda_initialization():
    with pytest.raises(ValueError, match='CUBLAS_WORKSPACE_CONFIG'):
        configure_numerics(True)


@pytest.mark.parametrize('key,value', [('CUBLAS_WORKSPACE_CONFIG', ':4096:8'),
                                       ('MCDNN_USE_DETERMINISTIC_ALGO', '1')])
def test_native_rejects_inherited_controls(monkeypatch, key, value):
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=key):
        configure_numerics()
