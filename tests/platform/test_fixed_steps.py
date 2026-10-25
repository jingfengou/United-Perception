"""Guard fixed-step attribution against mismatched inputs and incomplete traces."""
import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def compare(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    return importlib.import_module('summarize_fixed_steps').compare_traces


@pytest.fixture
def trace():
    return [{'step': i, 'input_sha256': 'input-'+str(i), 'lr_next': [0.001],
             'loss': 2., 'optimizer_updates': i} for i in range(1, 4)]


def test_small_loss_difference_and_exceedance_are_separate(compare, trace):
    actual = copy.deepcopy(trace)
    actual[0]['loss'] += 1e-5
    actual[1]['loss'] += 0.1
    actual[2]['optimizer_updates'] -= 1
    result = compare(trace, actual, steps=3)
    assert result['first_loss_difference_step'] == 1
    assert result['first_loss_exceeding_tolerance_step'] == 2
    assert result['loss_steps_exceeding_tolerance'] == 1
    assert result['first_optimizer_update_count_difference_step'] == 3


def test_identical_trace_has_no_loss_failures(compare, trace):
    result = compare(trace, copy.deepcopy(trace), steps=3)
    assert result['first_loss_difference_step'] is None
    assert result['first_loss_exceeding_tolerance_step'] is None
    assert result['max_loss_absolute_difference'] == 0


@pytest.mark.parametrize('field,value', [('input_sha256', 'different-input'), ('lr_next', [0.002])])
def test_protocol_mismatch_is_not_reported_as_numerical_drift(compare, trace, field, value):
    actual = copy.deepcopy(trace)
    actual[1][field] = value
    with pytest.raises(ValueError, match='Different'):
        compare(trace, actual, steps=3)


def test_missing_steps_are_rejected(compare, trace):
    with pytest.raises(ValueError, match='complete requested budget'):
        compare(trace, trace[:-1], steps=3)


def test_reordered_steps_are_rejected(compare, trace):
    with pytest.raises(ValueError, match='reordered'):
        compare(trace, list(reversed(trace)), steps=3)


def test_nonfinite_loss_is_rejected(compare, trace):
    actual = copy.deepcopy(trace)
    actual[0]['loss'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        compare(trace, actual, steps=3)
