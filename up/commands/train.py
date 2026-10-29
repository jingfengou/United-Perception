from __future__ import division

# Standard Library
import argparse

# Import from third library

from up.utils.env.dist_helper import finalize
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.env.launch import launch

# Import from local
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY


__all__ = ['Train']


@SUBCOMMAND_REGISTRY.register('train')
class Train(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for training',
                                       help='train a model')

        sub_parser.add_argument('-e',
                                '--evaluate',
                                dest='evaluate',
                                action='store_true',
                                help='evaluate model on validation set')
        sub_parser.add_argument(
            '--fork-method',
            dest='fork_method',
            type=str,
            default='fork',
            choices=['spawn', 'fork'],
            help='method to fork subprocess, especially for dataloader')
        sub_parser.add_argument(
            '--nocudnn',
            dest='nocudnn',
            action='store_true',
            help='Whether to use cudnn backend or not. Please disable cudnn when running on V100'
        )
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--display',
                                dest='display',
                                type=int,
                                default=20,
                                help='display intervel')
        sub_parser.add_argument('--ng', '--num_gpus_per_machine',
                                dest='num_gpus_per_machine',
                                type=int,
                                default=1,
                                help='num_gpus_per_machine')
        sub_parser.add_argument('--no_running_config',
                                action='store_true',
                                help='disable display running config')
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)

        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    cfg = load_yaml(args.config, args.cfg_type)
    cfg['args'] = {
        'config_path': args.config,
        'nocudnn': args.nocudnn,
        'display': args.display,
        'no_running_config': args.no_running_config,
        'opts': args.opts
    }
    train_phase = 'train'
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    training = True
    if args.evaluate:
        training = False
        train_phase = "eval"
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    train_func = {"train": runner.train, "eval": runner.evaluate}
    train_func[train_phase]()
    finalize()


def _main(args):
    launch(main, args.num_gpus_per_machine, args=args, start_method=args.fork_method)
