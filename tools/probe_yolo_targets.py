"""Observe original YOLO forward indexed writes; return each real result once."""
import argparse
import collections
import collections.abc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from probe_native_operators import RepeatPure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'frozen-tools', 'data', 'initial', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--lane', choices=('original_nvidia', 'new_nvidia'), required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.frozen_tools))
    from accuracy_train import run, CONTROLS
    assert all(os.environ.get(key) is None for key in CONTROLS)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(0)
    collections.Iterable = collections.abc.Iterable
    np.float, np.int, np.bool = float, int, bool
    sys.path.insert(0, str(args.source.resolve()))
    if args.lane == 'new_nvidia':
        from up_platform import configure_platform
        configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose='training')
    from up.runner.base_runner import BaseRunner
    forward = BaseRunner.forward_train
    probe = RepeatPure(args.output)

    def observed_forward(runner, batch):
        with probe:
            return forward(runner, batch)

    BaseRunner.forward_train = observed_forward
    args.model, args.mode, args.probe_steps = 'yolov5s', 'train', 3
    run(args)
    report = {'operator_counts': probe.counts, 'findings': probe.findings,
              'observer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Forward diagnostic only; mutable indexed destinations replay on isolated copies'}
    (args.output/'forward-probe.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'findings': probe.findings}, indent=2))


if __name__ == '__main__':
    main()
