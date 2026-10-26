"""Legacy SDKs do not impose requirements on the current platform boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('boundary_checker', ROOT/'tools/check_platform_boundaries.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class BoundaryTests(unittest.TestCase):
    def check_source(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'up').mkdir()
            (root/'up/example.py').write_text(source, encoding='utf8')
            with patch.object(checker, 'ROOT', root), patch.object(checker, 'collect', return_value=[]):
                return checker.inspect_boundaries()

    def test_legacy_imports_need_no_gap_ledger_or_sdk_install(self):
        self.assertEqual(self.check_source('try:\n    import spring.linklink as link\nexcept ImportError:\n    link = None\n'), [])

    def test_current_gpu_boundary_is_still_enforced(self):
        violations = self.check_source('import spring.linklink\nimport torch\n\ndef run():\n    torch.cuda.synchronize()\n')
        self.assertEqual([row[2] for row in violations], ['torch.cuda.synchronize'])

    def test_new_backend_does_not_need_a_legacy_optimizer(self):
        from up_platform.training.base import TrainingBackend
        self.assertNotIn('fused_optimizer', TrainingBackend.__abstractmethods__)
        methods = {name: lambda *a, **k: None for name in TrainingBackend.__abstractmethods__}
        backend = type('ModernTraining', (TrainingBackend,), methods)()
        backend.device = SimpleNamespace(vendor='example')
        with self.assertRaisesRegex(RuntimeError, 'Optional legacy optimizer'):
            backend.fused_optimizer('FusedFP16SGD', [])


if __name__ == '__main__':
    unittest.main()
