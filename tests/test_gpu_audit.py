"""Regression checks for audit omissions and misleading API classification."""
import ast
import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('audit_gpu_portability',
    pathlib.Path(__file__).resolve().parents[1] / 'tools/audit_gpu_portability.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def test_import_aliases_methods_matmul_and_numba_are_indexed(self):
        source = '''import torch.nn.functional as F
from torch import cat
from numba import cuda
def forward(x):
    y = F.relu(x)
    z = cat([x, y])
    return z @ x.transpose(0, 1)
@cuda.jit(device=True)
def kernel():
    return cuda.local.array((2,))
'''
        visitor = audit.Visitor('up/model.py', source)
        visitor.visit(ast.parse(source))
        apis = {row['api'] for row in visitor.calls}
        self.assertTrue({'torch.nn.functional.relu', 'torch.cat', 'Tensor.matmul',
                         'Tensor.transpose', 'numba.cuda.jit', 'numba.cuda.local.array'} <= apis)

    def test_topk_is_not_misclassified_as_device_to(self):
        self.assertEqual(audit.family('torch.topk', '', ''), 'tensor')
        self.assertEqual(audit.family('Tensor.to', '', ''), 'device')
        visitor = audit.Visitor('up/example.py', 'import torch\ntorch.cuda.empty_cache()')
        visitor.visit(ast.parse(visitor.text))
        self.assertEqual(sum(r['api'] == 'torch.cuda.empty_cache' for r in visitor.calls), 1)

    def test_demo_scope_is_separate_from_real_runtime(self):
        source = '''import torch
def execute(x):
    return x.cuda()
if __name__ == '__main__':
    x = torch.randn(4).cuda()
'''
        visitor = audit.Visitor('up/utils/example.py', source)
        visitor.visit(ast.parse(source))
        self.assertTrue(any(r['function'] == 'execute' and r['scope'] == 'model_or_runtime' for r in visitor.calls))
        self.assertTrue(any(r['scope'] == 'demo_or_test' for r in visitor.calls))

    def test_removed_optional_dependencies_are_not_vendor_blockers(self):
        for category in ('sparse', 'quantization', 'pruning', 'numba'):
            self.assertEqual(audit.vendor_decision(category, 'ascend', False)['action'], 'out_of_slim')
        self.assertEqual(audit.vendor_decision('legacy', 'metax')['action'], 'excluded')
        self.assertEqual(audit.vendor_decision('device', 'metax')['action'], 'reuse')
        self.assertEqual(audit.vendor_decision('device', 'ascend')['action'], 'adapt')


if __name__ == '__main__':
    unittest.main()
