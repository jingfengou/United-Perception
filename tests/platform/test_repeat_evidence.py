"""Checkpoint audit fingerprints must include values, dtype, keys and scalars."""
import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from package_native_repeat import tree_hash


def test_checkpoint_hash_ignores_dictionary_order_but_retains_values():
    a = {'weight': torch.tensor([1., 2.]), 'step': np.int64(4)}
    b = {'step': np.int64(4), 'weight': a['weight'].clone()}
    assert tree_hash(a) == tree_hash(b)
    b['weight'][0] += 1
    assert tree_hash(a) != tree_hash(b)


def test_checkpoint_hash_retains_optimizer_hyperparameters():
    a = {'state': {0: {'momentum': torch.ones(2)}}, 'param_groups': [{'lr': .1, 'params': [0]}]}
    b = copy.deepcopy(a)
    b['param_groups'][0]['lr'] = .2
    assert tree_hash(a) != tree_hash(b)
    b = copy.deepcopy(a)
    b['state'] = {'0': b['state'][0]}
    assert tree_hash(a) != tree_hash(b)


def test_checkpoint_hash_retains_tensor_dtype_and_scalar_dtype():
    assert tree_hash(torch.ones(2)) != tree_hash(torch.ones(2, dtype=torch.float64))
    assert tree_hash(np.float32(1)) != tree_hash(np.float64(1))
