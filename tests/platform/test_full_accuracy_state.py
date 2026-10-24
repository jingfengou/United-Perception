"""CPU contracts for the external long-run driver; no UP or vendor SDK needed."""
import importlib.util
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
    import full_accuracy_train as training


@unittest.skipUnless(HAS_TORCH, 'CPU PyTorch required')
class TestFullTrainingState(unittest.TestCase):
    def runner(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Linear(3, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9)
        return SimpleNamespace(model=model, optimizer=optimizer, lr_scheduler=scheduler,
                               ema=None, fp16=False)

    def step(self, runner):
        runner.optimizer.zero_grad()
        runner.model(torch.randn(4, 3)).square().mean().backward()
        runner.optimizer.step()
        runner.lr_scheduler.step()
        runner.cur_iter = runner.lr_scheduler.last_epoch

    def test_all_training_states_and_model_rng_restore_exactly(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([1, 2], dtype=torch.uint8)), \
                patch.object(torch.cuda, 'set_rng_state') as set_gpu:
            runner = self.runner()
            self.step(runner)
            self.step(runner)
            progress = {'completed_steps': 2, 'checkpoint_generation': 1}
            path = Path(directory)
            original = training.state_hash(runner.model.state_dict())
            training.checkpoint(runner, path, progress, 0, 1, {'source': 'frozen'})
            expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
            expected_lr = runner.optimizer.param_groups[0]['lr']
            self.step(runner)
            # The fixture's cur_iter mirrors UP's scheduler-derived property.
            runner.cur_iter = 2
            restored, _ = training.restore_checkpoint(runner, path, 0, {'source': 'frozen'})
            self.assertEqual(restored, progress)
            self.assertEqual(training.state_hash(runner.model.state_dict()), original)
            self.assertEqual(runner.optimizer.param_groups[0]['lr'], expected_lr)
            self.assertEqual(runner.lr_scheduler.last_epoch, 2)
            self.assertEqual(random.random(), expected_rng[0])
            self.assertEqual(float(np.random.random()), expected_rng[1])
            self.assertTrue(torch.equal(torch.rand(3), expected_rng[2]))
            self.assertTrue(torch.equal(set_gpu.call_args.args[0], torch.tensor([1, 2], dtype=torch.uint8)))
            with self.assertRaisesRegex(ValueError, 'does not match'):
                training.restore_checkpoint(runner, path, 0, {'source': 'changed'})

    def test_failed_next_slot_does_not_replace_last_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([1], dtype=torch.uint8)):
            runner = self.runner()
            self.step(runner)
            path = Path(directory)
            training.checkpoint(runner, path, {'completed_steps': 1, 'checkpoint_generation': 0}, 0, 1, {})
            marker = (path/'checkpoint.json').read_bytes()
            original_save = torch.save
            def fail_main_checkpoint(value, destination):
                if Path(destination).name == 'checkpoint-1.pt.writing':
                    raise OSError('interrupted write')
                return original_save(value, destination)
            with patch.object(torch, 'save', side_effect=fail_main_checkpoint):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    training.checkpoint(runner, path, {'completed_steps': 1, 'checkpoint_generation': 1}, 0, 1, {})
            self.assertEqual((path/'checkpoint.json').read_bytes(), marker)

    def test_algorithm_config_digest_ignores_machine_paths(self):
        left = {'net': [{'type': 'unchanged'}], 'dataset': {'meta': '/server/a/train.txt'}}
        right = {'net': [{'type': 'unchanged'}], 'dataset': {'meta': '/server/b/train.txt'}}
        self.assertEqual(training.canonical_config(left, {'imagenet': {'meta': '/server/a/train.txt'}}, Path('/up/a')),
                         training.canonical_config(right, {'imagenet': {'meta': '/server/b/train.txt'}}, Path('/up/b')))
        right['net'][0]['type'] = 'different'
        self.assertNotEqual(training.canonical_config(left, {'imagenet': {'meta': '/server/a/train.txt'}}, Path('/up/a')),
                            training.canonical_config(right, {'imagenet': {'meta': '/server/b/train.txt'}}, Path('/up/b')))

    def test_fresh_inference_restores_rank_local_bn_and_ema_buffers(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.cuda, 'get_rng_state', return_value=torch.tensor([1], dtype=torch.uint8)):
            runner = self.runner()
            runner.ema = SimpleNamespace(model=copy.deepcopy(runner.model).eval())
            runner.ema.state_dict = lambda: {'ema_state_dict': runner.ema.model.state_dict()}
            runner.ema.load_state_dict = lambda state: runner.ema.model.load_state_dict(state['ema_state_dict'])
            path = Path(directory)
            identity = {'world_size': 2}
            training.checkpoint(runner, path, {'completed_steps': 0, 'checkpoint_generation': 0}, 0, 1, identity)
            # Construct the second rank's independently saved buffer/RNG file.
            local = training.local_state(runner)
            local['buffers']['0.running_mean'].fill_(7.)
            local['ema_buffers']['0.running_mean'].fill_(13.)
            torch.save(local, path/'checkpoint-0.rank1.pt')
            marker = json.loads((path/'checkpoint.json').read_text())
            marker['files']['checkpoint-0.rank1.pt'] = training.digest(path/'checkpoint-0.rank1.pt')
            (path/'checkpoint.json').write_text(json.dumps(marker))
            training.load_inference_checkpoint(runner, path, identity, rank=1)
            self.assertTrue(torch.equal(runner.model[0].running_mean, torch.full((3,), 7.)))
            self.assertTrue(torch.equal(runner.ema.model[0].running_mean, torch.full((3,), 13.)))
