"""UP BaseRunner/ResNet18 FSDP2 lifecycle with tiny in-memory diagnostic batches.

Only data/logging plumbing is replaced. Model, loss, optimizer, train/eval and
checkpoint methods are the real UP code. Run explicitly on two accelerators.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from up_platform import configure_platform


def config(vendor, checkpoint=None):
    saver = dict(save_dir='checkpoints', results_dir='results', auto_resume=False, task_type='cls')
    if checkpoint is not None:
        saver['resume_model'] = str(checkpoint)
    return dict(args={'no_running_config': True, 'nocudnn': False},
        runtime={'platform': {'backend': vendor}, 'fp16': True, 'iter_base': True,
                 'distributed': {'strategy': 'fsdp2', 'shard_modules': ['backbone', 'head']}},
        dataset={}, saver=saver,
        ema={'enable': True, 'ema_type': 'exp', 'kwargs': {'decay': .99}},
        hooks=[{'type': 'auto_save_best'}, {'type': 'grad_clipper', 'kwargs': {'mode': 'pre_defined', 'max_norm': 1.}}],
        trainer={'max_iter': 3, 'test_freq': 2, 'save_freq': 3,
                 'optimizer': {'type': 'SGD', 'kwargs': {'lr': .001, 'momentum': .9}},
                 'lr_scheduler': {'type': 'MultiStepLR', 'warmup_iter': 0,
                                  'warmup_register_type': 'no_scale_lr',
                                  'kwargs': {'milestones': [10], 'gamma': .1}}},
        net=[{'name': 'backbone', 'type': 'resnet18', 'kwargs': {
                 'out_layers': [4], 'out_strides': [32], 'normalize': {'type': 'solo_bn'},
                 'initializer': {'method': 'msra'}}},
             {'name': 'head', 'type': 'base_cls_head', 'kwargs': {'num_classes': 10, 'in_plane': 512}},
             {'name': 'post_process', 'type': 'base_cls_postprocess', 'kwargs': {
                 'cls_loss': {'type': 'label_smooth_ce', 'kwargs': {'smooth_ratio': .1, 'num_classes': 10}}}}])


def main(args):
    platform = configure_platform({'backend': args.vendor}, purpose=args.purpose)
    platform.device.set_device(int(os.environ.get('LOCAL_RANK', 0)))
    from up.runner.base_runner import BaseRunner
    from up.models.model_helper import ModelHelper
    from up.data.metrics.base_evaluator import Metric
    from up.utils.env.dist_helper import env
    from up.utils.general.hook_helper import build_hooks
    comm = platform.category('communication')
    output = Path(args.output)
    image = torch.randn(2, 3, 64, 64, generator=torch.Generator().manual_seed(33))

    if args.purpose == 'inference':
        state = torch.load(output/'export.pth', map_location='cpu', weights_only=True)
        model = platform.device.move(ModelHelper(config(args.vendor)['net'])).eval()
        model.load(state['model'], strict=True)
        with torch.no_grad():
            prediction = model({'image': platform.device.move(image), 'gt': platform.device.move(torch.tensor([0, 1]))})['scores']
        torch.testing.assert_close(prediction.cpu(), state['prediction'], rtol=1e-4, atol=1e-5)
        (output/'inference.json').write_text(json.dumps({'passed': True, 'purpose': platform.purpose})+'\n')
        return

    comm.init_process_group(backend='nccl')
    torch.set_num_threads(2)
    if env.is_master():
        output.mkdir(parents=True, exist_ok=False)
    comm.barrier()

    class Dataset:
        def dump(self, prediction):
            return [{'scores': prediction['scores'].detach().cpu().tolist()}]

        def evaluate(self, *unused):
            return Metric({'AP': 1.})

    class BatchSampler:
        batch_size = 2
        def set_start_iter(self, value):
            self.start_iter = value

    class Loader:
        dataset, batch_sampler = Dataset(), BatchSampler()
        def __iter__(self):
            for _ in range(self.get_epoch_size()):
                yield {'image': image.clone(), 'gt': torch.tensor([0, 1])}
        def get_epoch_size(self):
            return 2 if env.rank == 0 else 1

    def loaders(runner):
        runner.data_loaders = {'train': Loader(), 'test': Loader()}
        runner.train_pool, runner.test_pool = ['train:train'], ['test:test']
        runner.train_epoch_size, runner.train_batch_size = 2, 2

    def hooks(runner):
        configs = copy.deepcopy(runner.config['hooks'])
        if not runner.training:
            configs = runner._remove_hooks(configs)
        runner._hooks = build_hooks(runner, configs, add_log_if_not_exists=False)

    with patch.object(BaseRunner, 'build_dataloaders', loaders), patch.object(BaseRunner, 'build_hooks', hooks):
        runner = BaseRunner(config(args.vendor), work_dir=str(output))
        runner.train()
        comm.barrier()
        assert runner.cur_iter == 3
        assert (output/'checkpoints/ckpt_best.pth').is_file()
        checkpoint = output/'checkpoints/ckpt_latest.pth'
        assert checkpoint.is_file()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['parallel_strategy'] == 'fsdp2' and saved['fsdp_scheduler_step_pending']
        assert saved['iter'] == 2
        restored = BaseRunner(config(args.vendor, checkpoint), work_dir=str(output/'restored'))
        platform.execution.load_full_state_dict(restored.model, {'model': saved['model']})
        assert restored.start_iter == 3
        assert restored.lr_scheduler.state_dict() == runner.lr_scheduler.state_dict()
        assert restored.scaler.state_dict() == runner.scaler.state_dict()
        before = restored.get_dump_dict()
        if env.is_master():
            for name, tensor in saved['model'].items():
                assert torch.equal(tensor, before['model'][name])
        restored.evaluate()
        assert restored.model.training and not restored.ema.model.training
        # An evaluation-only CLI path also keeps the training-purpose interface.
        evaluator = BaseRunner(config(args.vendor, checkpoint), work_dir=str(output/'evaluate'), training=False)
        evaluator.evaluate()
        assert platform.purpose == 'training'
        with platform.execution.validation(restored.ema.model):
            prediction = restored.ema.model({'image': platform.device.move(image),
                'gt': platform.device.move(torch.tensor([0, 1]))})['scores'].detach().cpu()
        state = platform.execution.full_state_dict(restored.ema.model)
        if env.is_master():
            torch.save({'model': state['model'], 'prediction': prediction}, output/'export.pth')
            (output/'training.json').write_text(json.dumps({'passed': True, 'model': 'UP ResNet18',
                'purpose': platform.purpose, 'steps': 3, 'amp': True, 'ema': True,
                'checks': ['BaseRunner train', 'grad_clipper', 'auto_save_best all-rank collection',
                           'train/eval/train', 'checkpoint optimizer/BN/RNG/scaler/LR resume boundary',
                           'train --evaluate path', 'full inference weights']})+'\n')
    comm.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', required=True, choices=['nvidia', 'metax'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--purpose', default='training', choices=['training', 'inference'])
    main(parser.parse_args())
