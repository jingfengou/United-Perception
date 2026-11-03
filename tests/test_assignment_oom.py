"""Exercise the real OTA exception boundary without loading a model or GPU SDK."""
import ast
import pathlib
import types
import unittest
from unittest.mock import Mock


SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'up/tasks/det/models/postprocess/roi_supervisor.py'


class AssignmentErrorTests(unittest.TestCase):
    def run_boundary(self, first_error=None):
        tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RetinaOTASupervisor')
        boundary = next(n for n in ast.walk(cls) if isinstance(n, ast.Try) and
                        any(isinstance(x.type, ast.Name) and x.type.id == 'RuntimeError' for x in n.handlers))
        call = Mock(side_effect=[first_error, (0, None, None, None)] if first_error else None,
                    return_value=(0, None, None, None))
        cache = Mock()
        env = dict(self=types.SimpleNamespace(matcher=types.SimpleNamespace(match=call), use_centerness=False),
                   input={'image_info': [[480, 640]]}, b_ix=0,
                   get_runtime=lambda: types.SimpleNamespace(module=types.SimpleNamespace(empty_cache=cache)),
                   logger=types.SimpleNamespace(info=Mock()))
        for key in ('gts', 'preds', 'anchor_points', 'all_anchors', 'num_anchors_per_level', 'strides'):
            env[key] = object()
        module = ast.fix_missing_locations(ast.Module(body=[boundary], type_ignores=[]))
        return lambda: exec(compile(module, str(SOURCE), 'exec'), env), call, cache

    def test_missing_operator_is_not_hidden_by_cpu_retry(self):
        run, match, cache = self.run_boundary(RuntimeError('MLU operator is not implemented'))
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            run()
        self.assertEqual(match.call_count, 1)
        cache.assert_not_called()

    def test_original_oom_retry_is_preserved_on_selected_device(self):
        run, match, cache = self.run_boundary(RuntimeError('NPU out of memory'))
        run()
        self.assertEqual(match.call_count, 2)
        self.assertEqual(match.call_args.kwargs['mode'], 'cpu')
        cache.assert_called_once_with()

    def test_successful_assignment_does_not_enter_cpu_route(self):
        run, match, cache = self.run_boundary()
        run()
        match.assert_called_once()
        cache.assert_not_called()


if __name__ == '__main__':
    unittest.main()
