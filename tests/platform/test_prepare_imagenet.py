"""Dataset fidelity gates are independent of Torch, CUDA and parquet libraries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare_imagenet_parquet',
    Path(__file__).resolve().parents[2] / 'tools/prepare_imagenet_parquet.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestImageNetMaterialization(unittest.TestCase):
    def test_encoded_bytes_and_original_labels_are_preserved(self):
        rows = [{'image': {'bytes': b'\xff\xd8original', 'path': 'original.JPEG'}, 'label': 999},
                {'image': {'bytes': b'\x89PNGoriginal', 'path': 'second.png'}, 'label': 0}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = prepare.materialize_rows(rows, root, 'train', 'part0', root/'records.jsonl', expected_rows=2)
            records = [json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
            self.assertEqual([row['label'] for row in records], [999, 0])
            self.assertEqual(result['images'], 2)
            for source, record in zip(rows, records):
                self.assertEqual((root/'images'/record['path']).read_bytes(), source['image']['bytes'])

    def test_resuming_a_shard_does_not_overwrite_a_changed_image(self):
        rows = [{'image': {'bytes': b'original', 'path': None}, 'label': 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepare.materialize_rows(rows, root, 'train', 'part0', root/'first.jsonl', expected_rows=1)
            path = root/'images/train/part0/0000001.img'
            path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare.materialize_rows(rows, root, 'train', 'part0', root/'second.jsonl', expected_rows=1)
            self.assertEqual(path.read_bytes(), b'corrupt')

    def test_incomplete_shard_cannot_be_committed_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                prepare.materialize_rows([], root, 'train', 'part0', root/'records.jsonl', expected_rows=1)

    def test_invalid_labels_and_missing_bytes_are_rejected(self):
        for label, raw in ((-1, b'bytes'), (1000, b'bytes'), (True, b'bytes'), (0, None), (0, b'')):
            with self.subTest(label=label, raw=raw), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with self.assertRaises(ValueError):
                    prepare.materialize_rows([{'image': {'bytes': raw}, 'label': label}],
                        root, 'train', 'part0', root/'records.jsonl', expected_rows=1)
