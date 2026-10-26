"""Guards for lossless, resumable preparation of the official COCO archives."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


spec = importlib.util.spec_from_file_location('prepare_coco2017',
    Path(__file__).resolve().parents[2] / 'tools/prepare_coco2017.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestCocoPreparation(unittest.TestCase):
    def test_extraction_preserves_bytes_and_verifies_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'images.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('val2017/000000000001.jpg', b'original encoded image')
            output = root/'dataset'
            prepare.extract(archive, output)
            prepare.extract(archive, output)
            image = output/'val2017/000000000001.jpg'
            self.assertEqual(image.read_bytes(), b'original encoded image')
            image.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare.extract(archive, output)
            self.assertEqual(image.read_bytes(), b'changed')

    def test_zip_path_cannot_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('../escaped.jpg', b'bad')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                prepare.extract(archive, root/'dataset')
            self.assertFalse((root/'escaped.jpg').exists())

    def test_interrupted_extraction_can_resume_without_truncated_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'images.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('train2017/1.jpg', b'complete bytes')
            output = root/'dataset'
            with patch.object(prepare.shutil, 'copyfileobj', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    prepare.extract(archive, output)
            self.assertFalse((output/'train2017/1.jpg').exists())
            prepare.extract(archive, output)
            self.assertEqual((output/'train2017/1.jpg').read_bytes(), b'complete bytes')

    def test_existing_archive_must_match_checksum_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'file.zip'
            archive.write_bytes(b'original')
            with patch.object(prepare.urllib.request, 'urlopen', side_effect=AssertionError('network')):
                prepare.download('unused', archive, hashlib.md5(b'original').hexdigest())
                with self.assertRaisesRegex(ValueError, 'differs'):
                    prepare.download('unused', archive, '0'*32)
