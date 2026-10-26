"""Preparation receipts cannot replace an actual audit of transferred bytes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from full_accuracy_bind import verify_imagenet_shard


class TestFullDataIntegrity(unittest.TestCase):
    def test_copied_image_is_rehashed_not_accepted_from_metadata_alone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'images/train/part/0000001.JPEG'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'original encoded JPEG')
            manifest = root/'shard.jsonl'
            manifest.write_text(json.dumps({'path': 'train/part/0000001.JPEG', 'bytes': image.stat().st_size,
                'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'label': 7})+'\n')
            self.assertEqual(verify_imagenet_shard(root, manifest), (1, 21))
            image.write_bytes(b'changed encoded JPEG!')
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                verify_imagenet_shard(root, manifest)

    def test_manifest_cannot_audit_an_image_outside_the_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/'shard.jsonl'
            manifest.write_text(json.dumps({'path': '../../unrelated', 'bytes': 0, 'sha256': '', 'label': 0})+'\n')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                verify_imagenet_shard(root, manifest)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'Pillow data decoder required')
    def test_byte_accurate_but_undecodable_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root/'images/train/corrupt.img'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'not an encoded image')
            manifest = root/'shard.jsonl'
            manifest.write_text(json.dumps({'path': 'train/corrupt.img', 'bytes': image.stat().st_size,
                'sha256': hashlib.sha256(image.read_bytes()).hexdigest(), 'label': 7})+'\n')
            with self.assertRaises(OSError):
                verify_imagenet_shard(root, manifest, decode_images=True)
