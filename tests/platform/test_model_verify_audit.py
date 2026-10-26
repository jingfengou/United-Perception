"""Ratio AP values and percentage-valued accuracy require different units."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from audit_model_verify import delta_pp


def test_ap_gap_is_converted_to_percentage_points():
    assert delta_pp(.414, .413, unit='ratio') == pytest.approx(-.1)
    assert delta_pp(.4717, .4715, unit='ratio') == pytest.approx(-.02)


def test_accuracy_is_already_in_percentage_units():
    assert delta_pp(85.54, 85.53, unit='percent') == pytest.approx(-.01)


def test_unit_is_not_inferred_from_value_range():
    with pytest.raises(ValueError, match='explicit'):
        delta_pp(.8, .7, unit='auto')
