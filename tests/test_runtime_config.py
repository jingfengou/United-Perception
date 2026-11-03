"""Configuration contracts without installing a vendor SDK or initializing a GPU.

Run: python -m unittest discover -s tests -p 'test_runtime_config.py' -v
Hardware forward/backward validation is a separate vendor acceptance step.
"""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch


SOURCE = pathlib.Path(__file__).resolve().parents[1] / 'up/runtime/backend.py'


class DeviceAPI:
    def __init__(self):
        self.selected = None
        self.amp = object()
        self.available = True

    def current_device(self):
        return self.selected or 0

    def is_available(self):
        return self.available

    def device_count(self):
        return 2

    def set_device(self, index):
        self.selected = index

    def get_device_properties(self, index):
        return types.SimpleNamespace(total_memory=2 * 1024 ** 3)

    def memory_allocated(self, device):
        self.queried = device
        return 1024 ** 3


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        self.torch = types.ModuleType('torch')
        for name in ('cuda', 'npu', 'mlu'):
            setattr(self.torch, name, DeviceAPI())
        self.torch.device = lambda kind, index: (kind, index)
        self.torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(
            enabled=False, benchmark=True, deterministic=False, allow_tf32=True))
        spec = importlib.util.spec_from_file_location('isolated_up_runtime', SOURCE)
        self.runtime = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': self.torch}):
            spec.loader.exec_module(self.runtime)

    def test_config_selects_only_requested_vendor_plugin(self):
        for platform, kind, communication, plugin in (
                ('nvidia', 'cuda', 'nccl', None),
                ('metax_cuda', 'cuda', 'nccl', None),
                ('cambricon_mlu', 'mlu', 'cncl', 'torch_mlu'),
                ('ascend_npu', 'npu', 'hccl', 'torch_npu')):
            with self.subTest(platform=platform), patch.object(
                    self.runtime.importlib, 'import_module') as load:
                result = self.runtime.setup_runtime({'runtime': {'platform': platform}})
                self.assertEqual((result.device_type, result.dist_backend), (kind, communication))
                self.assertIs(result.amp, getattr(self.torch, kind).amp)
                self.assertEqual(result.device, (kind, 0))
                self.assertIsNone(result.module.selected)  # config parsing does not bind a card
                if plugin:
                    load.assert_called_once_with(plugin)
                else:
                    load.assert_not_called()

    def test_cli_override_is_resolved_before_backend_creation(self):
        helper = types.ModuleType('up.utils.general.cfg_helper')
        helper.merge_opts_into_cfg = lambda opts, cfg: {'runtime': {'platform': opts[1]}}
        with patch.dict(sys.modules, {'up.utils.general.cfg_helper': helper}), patch.object(
                self.runtime.importlib, 'import_module') as load:
            result = self.runtime.setup_runtime({'runtime': {'platform': 'nvidia'}},
                                                ['runtime.platform', 'ascend_npu'])
        self.assertEqual(result.platform, 'ascend_npu')
        load.assert_called_once_with('torch_npu')

    def test_unknown_vendor_and_conflicting_device_are_rejected(self):
        for config in ({'platform': 'typo'}, {'platform': 'ascend_npu', 'device': 'cuda'},
                       {'platform': 'cambricon_mlu', 'device': 'npu'}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.runtime.setup_runtime({'runtime': config})

    def test_missing_vendor_plugin_reports_vendor_and_preserves_cause(self):
        with patch.object(self.runtime.importlib, 'import_module', side_effect=ImportError('missing libhccl')):
            with self.assertRaisesRegex(RuntimeError, 'ascend_npu.*torch_npu.*libhccl') as caught:
                self.runtime.RuntimeBackend('ascend_npu')
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_missing_registered_device_api_is_reported(self):
        del self.torch.mlu
        with patch.object(self.runtime.importlib, 'import_module'), self.assertRaisesRegex(
                RuntimeError, 'torch.mlu'):
            self.runtime.RuntimeBackend('cambricon_mlu')

    def test_device_binding_checks_negative_and_out_of_range_rank(self):
        result = self.runtime.RuntimeBackend('nvidia')
        for index in (-1, 2):
            with self.assertRaises(RuntimeError):
                result.set_device(index)
        result.set_device(1)
        self.assertEqual(result.device, ('cuda', 1))
        result.module.available = False
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            result.set_device(0)

    def test_memory_budget_queries_selected_vendor_device(self):
        with patch.object(self.runtime.importlib, 'import_module'):
            result = self.runtime.RuntimeBackend('cambricon_mlu')
        device = types.SimpleNamespace(type='mlu', index=1)
        self.assertTrue(result.memory_available(device, 100))
        self.assertIs(result.module.queried, device)
        self.assertFalse(result.memory_available(device, 512 * 1024 ** 2))
        self.assertFalse(result.memory_available(types.SimpleNamespace(type='cpu'), 1))

    def test_native_dnn_defaults_and_original_nocudnn_behavior(self):
        result = self.runtime.RuntimeBackend('nvidia')
        before = vars(self.torch.backends.cudnn).copy()
        result.configure_dnn()
        self.assertEqual(vars(self.torch.backends.cudnn), before)
        result.configure_dnn(legacy_nocudnn=False)
        self.assertTrue(self.torch.backends.cudnn.enabled)
        self.assertTrue(self.torch.backends.cudnn.benchmark)
        result.configure_dnn(legacy_nocudnn=True)
        self.assertFalse(self.torch.backends.cudnn.enabled)

    def test_explicit_dnn_config_and_nocudnn_conflict(self):
        result = self.runtime.RuntimeBackend('metax_cuda')
        result.configure_dnn({'benchmark': False, 'deterministic': True, 'allow_tf32': False})
        self.assertFalse(self.torch.backends.cudnn.benchmark)
        self.assertTrue(self.torch.backends.cudnn.deterministic)
        with self.assertRaises(ValueError):
            result.configure_dnn({'enabled': True}, legacy_nocudnn=True)
        for options in ({'benchmark': 'false'}, {'unknown': False}):
            with self.assertRaises(ValueError):
                result.configure_dnn(options)

    def test_npu_and_mlu_do_not_mutate_cudnn_or_fake_dnn_policy(self):
        before = vars(self.torch.backends.cudnn).copy()
        for platform in ('ascend_npu', 'cambricon_mlu'):
            with patch.object(self.runtime.importlib, 'import_module'):
                result = self.runtime.RuntimeBackend(platform)
            result.configure_dnn(legacy_nocudnn=False)
            with self.assertRaises(NotImplementedError):
                result.configure_dnn({'deterministic': True})
            with self.assertRaises(NotImplementedError):
                result.configure_dnn(legacy_nocudnn=True)
        self.assertEqual(vars(self.torch.backends.cudnn), before)

    def test_train_eval_and_inference_keep_same_configured_device(self):
        config = {'runtime': {'platform': 'metax_cuda', 'device': 'cuda'}}
        training = self.runtime.setup_runtime(config)
        training.set_device(1)
        for model_training in (True, False, True):
            # Module.train()/eval() is not an input to platform selection.
            self.assertIs(self.runtime.get_runtime(), training)
            self.assertEqual(self.runtime.get_runtime().device, ('cuda', 1))
        self.assertIs(self.runtime.setup_runtime(config), training)


if __name__ == '__main__':
    unittest.main()
