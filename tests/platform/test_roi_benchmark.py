import hashlib
import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]/'tools'))
    return importlib.import_module('benchmark_roi_align')


def test_truncated_capture_rejected_before_loading(benchmark, tmp_path):
    expected = hashlib.sha256(b'complete capture').hexdigest()
    (tmp_path/'case-00.pt').write_bytes(b'partial')
    (tmp_path/'cases.json').write_text(json.dumps({'captures': {'case-00.pt': expected}}))
    with pytest.raises(ValueError, match='checksum mismatch'):
        benchmark.capture_paths(tmp_path)


def test_incomplete_case_set_rejected(benchmark, tmp_path):
    (tmp_path/'cases.json').write_text(json.dumps({'captures': {'case-00.pt': 'missing'}}))
    with pytest.raises(ValueError, match='file set'):
        benchmark.capture_paths(tmp_path)
