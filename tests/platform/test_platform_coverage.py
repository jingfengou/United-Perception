"""Keep the public GPU code map exhaustive and conservative without CUDA/Torch."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import platform_coverage as coverage


class CoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.ledger = coverage.build()

    def test_every_required_gpu_interface_has_two_code_and_test_entries(self):
        expected = {r['name'] for r in coverage.collect()} - coverage.CPU_ONLY
        self.assertEqual({r['name'] for r in self.rows}, expected)
        for row in self.rows:
            self.assertTrue(row['tests'], row['name'])
            for vendor in ('nvidia', 'metax'):
                self.assertTrue(row['vendors'][vendor]['binding'])
                self.assertTrue(row['vendors'][vendor]['body'])
                self.assertTrue(row['vendors'][vendor]['implemented'])
        self.assertEqual(len(self.rows), 155)

    def test_existing_table_matches_source_and_ledger(self):
        self.assertEqual(coverage.DOCUMENT.read_text(encoding='utf8'), coverage.render(self.rows, self.ledger))

    def test_missing_interface_or_evidence_is_rejected(self):
        ledger = copy.deepcopy(self.ledger)
        ledger['interfaces'].pop('matrix.mm')
        with self.assertRaisesRegex(ValueError, 'set drift'):
            coverage.build(ledger)
        ledger = copy.deepcopy(self.ledger)
        ledger['interfaces']['matrix.mm']['nvidia']['evidence'] = []
        with self.assertRaisesRegex(ValueError, 'requires hardware evidence'):
            coverage.build(ledger)

    def test_evidence_cannot_be_borrowed_from_the_other_vendor(self):
        ledger = copy.deepcopy(self.ledger)
        ledger['interfaces']['matrix.mm']['nvidia']['evidence'] = ['M1']
        with self.assertRaisesRegex(ValueError, 'another vendor'):
            coverage.build(ledger)

    def test_vendor_template_is_not_an_implementation(self):
        self.assertFalse(coverage.implementation('convolution.conv2d', 'ascend')['implemented'])

    def test_recorded_failures_and_unverified_cases_are_not_promoted(self):
        rows = {r['name']: r for r in self.rows}
        for vendor in ('nvidia', 'metax'):
            self.assertEqual(rows['convolution.Conv2d']['vendors'][vendor]['status'], 'partial')
            for name in ('pruning.prune', 'pruning.last_prune_step', 'pruning.set_lr_scheduler'):
                self.assertEqual(rows[name]['vendors'][vendor]['status'], 'unverified')
        self.assertEqual(rows['convolution.ConvTranspose2d']['vendors']['metax']['status'], 'passed')
        self.assertIn('M9', rows['convolution.ConvTranspose2d']['vendors']['metax']['evidence'])
        self.assertEqual(rows['training.wrap_fsdp']['vendors']['metax']['status'], 'passed')

    def test_aliases_and_passed_factories_count_as_up_references(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'up_platform').mkdir()
            (root / 'up').mkdir()
            (root / 'up_platform/layers.py').write_text(
                'def Hardswish():\n    return get_platform().category("activation").Hardswish()\n')
            (root / 'up/model.py').write_text(
                'from up_platform import get_platform, layers as platform_layers\n'
                'factory = platform_layers.Hardswish\n'
                'def run():\n'
                '    platform = get_platform()\n'
                '    comm, execution = platform.category("communication"), platform.execution\n'
                '    comm.all_gather_object([], {})\n'
                '    scaler = execution.grad_scaler\n'
                '    model.features()\n')
            names = {'activation.Hardswish', 'communication.all_gather_object', 'training.grad_scaler', 'sparse.features'}
            coverage.tree.cache_clear()
            try:
                with patch.object(coverage, 'ROOT', root):
                    uses = coverage.usage(names)
                self.assertEqual(set(uses), names - {'sparse.features'})
                self.assertEqual(uses['training.grad_scaler'][0]['line'], 7)
            finally:
                coverage.tree.cache_clear()


if __name__ == '__main__':
    unittest.main()
