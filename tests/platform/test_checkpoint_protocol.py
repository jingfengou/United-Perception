"""The fixed-input audit must distinguish metric tolerance from bitwise equality."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
import verify_checkpoint_protocol as protocol


def test_close_values_are_not_reported_as_bitwise_equal():
    value = protocol.difference(torch.tensor([1.]), torch.tensor([1.000001]))
    assert value['within_tolerance'] and not value['bitwise_equal']
    assert value['max_abs_matching_tensors'] > 0


def test_different_output_shapes_are_not_a_numeric_pass():
    value = protocol.difference(torch.ones(1), torch.ones(2))
    assert not value['within_tolerance'] and not value['bitwise_equal']
    assert value['failures'][0]['reason'] == 'shape/dtype'


def test_prediction_snapshot_is_independent_of_mutable_model_outputs():
    logits = torch.ones(2, 3)
    result = protocol.predictions({'logits': logits, 'image': torch.zeros(2, 3, 8, 8)})
    logits.add_(1)
    assert set(result) == {'logits'}
    assert torch.equal(result['logits'], torch.ones(2, 3))


def test_gradients_restore_buffers_and_rng_without_updating_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda value: None)
    torch.manual_seed(131)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2), torch.nn.Dropout(.5))
        def forward(self, batch):
            return {'actual.loss': self.layers(batch['image']).square().mean()}
    model = Model()
    initial = copy.deepcopy(model.state_dict())
    runner = SimpleNamespace(model=model, batch2device=lambda value: value)
    cases = [{'batch_index': 1, 'batch': {'image': torch.arange(12.).reshape(4, 3)}}]
    result = protocol.gradients(SimpleNamespace(output=tmp_path, trials=3), runner,
                                SimpleNamespace(purpose='training'), cases, initial)
    assert result['optimizer_updates'] == 0
    assert all(r['bitwise_equal'] and r['loss_exact'] for r in result['cases'][0]['repeats'])
    assert all(torch.equal(initial[k], value) for k,value in model.state_dict().items())
    saved = torch.load(tmp_path/'gradients-case0.pt', weights_only=True)
    assert any(torch.any(value != 0) for value in saved.values())
