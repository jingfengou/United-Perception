"""Keep owner interfaces and NVIDIA reference links consistent with actual source."""
import importlib.util
import inspect
import json
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import build_vendor_handoff as builder


class VendorHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = builder.make_spec()

    def test_generated_contracts_match_source(self):
        stored = json.loads((ROOT/'docs/platform/vendor-api-contracts.json').read_text(encoding='utf-8'))
        self.assertEqual(stored, self.spec)
        self.assertEqual(len(self.spec['operations']), 16)
        self.assertEqual(len(self.spec['services']), 33)

    def test_all_nvidia_definitions_and_binding_lines_exist(self):
        for op in self.spec['operations']:
            for ref in op['python'] + op['cpp'] + op['pybind']:
                with self.subTest(operation=op['id'], symbol=ref['symbol']):
                    line = (ROOT/ref['path']).read_text(encoding='utf-8-sig').splitlines()[ref['line']-1]
                    # Python/C++ definitions show their own name; pybind shows exported name.
                    self.assertIn(ref['symbol'].split('.')[-1], line)
                    self.assertIn(builder.REFERENCE_COMMIT, ref['url'])

    def test_template_has_exact_delivery_signatures_and_never_fake_outputs(self):
        path = ROOT / 'docs/platform/templates/native_ops.py'
        self.assertEqual(path.read_text(encoding='utf-8'), builder.template(self.spec))
        module_spec = importlib.util.spec_from_file_location('vendor_native_template', path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        instance = module.NativeOpsTemplate('metax_cuda')
        for op in self.spec['operations']:
            with self.subTest(operation=op['id']):
                method = getattr(instance, op['id'])
                signature = inspect.signature(method)
                expected = inspect.signature(eval('lambda '+op['signature_args']+': None'))
                self.assertEqual(signature, expected)
                args, kwargs = [], {}
                for parameter in signature.parameters.values():
                    if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
                        args.append(object())
                    elif parameter.kind == parameter.KEYWORD_ONLY:
                        kwargs[parameter.name] = object()
                with self.assertRaisesRegex(NotImplementedError, 'metax_cuda: '+op['id']):
                    method(*args, **kwargs)

    def test_explicit_business_flags_are_not_hidden_vendor_globals(self):
        ops = {op['id']: op for op in self.spec['operations']}
        self.assertIn('*, aligned', ops['roi_align']['vendor_signature'])
        self.assertIn('*, offset', ops['naive_nms']['vendor_signature'])

    def test_original_special_cases_remain_visible(self):
        ops = {op['id']: op for op in self.spec['operations']}
        self.assertIn('不可达', ops['cross_focal']['output'])
        self.assertIn('[0]', ops['overlap']['output'])
        self.assertIn('部分梯度', ops['iou3d']['backward'])
        self.assertIn('未实现', ops['roipoint']['backward'])
        self.assertIn('int32', ops['sigmoid_focal']['inputs'])

    def test_nvidia_sources_match_the_pinned_reference_commit(self):
        paths = {ref['path'] for op in self.spec['operations']
                 for ref in op['python'] + op['cpp'] + op['cuda']}
        for path in paths:
            with self.subTest(path=path):
                original = subprocess.check_output(['git', 'show', builder.REFERENCE_COMMIT+':'+path], cwd=ROOT)
                current = (ROOT/path).read_bytes()
                self.assertEqual(current.replace(b'\r\n', b'\n'), original.replace(b'\r\n', b'\n'))


if __name__ == '__main__':
    unittest.main()
