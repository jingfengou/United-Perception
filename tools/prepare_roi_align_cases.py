"""Copy trusted model captures and add CPU-generated dtype/accumulation fixtures."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from benchmark_roi_align import capture_paths, digest
from diagnose_platform_operator import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = capture_paths(args.source)
    manifest = json.loads((args.source/'cases.json').read_text())
    if not paths or 'fixture_kinds' in manifest:
        raise ValueError('Use the original real-model capture directory')
    args.output.mkdir(parents=True, exist_ok=False)
    for path in paths:
        shutil.copyfile(path, args.output/path.name)
    manifest['fixture_kinds'] = {p.name: 'real Faster R-CNN capture' for p in paths}
    generator = torch.Generator().manual_seed(131)
    for index, (dtype, channels, aligned, sampling) in enumerate(
            ((torch.float16, 17, True, 2), (torch.float64, 3, False, 0)), start=len(paths)):
        rois = torch.tensor([[0, -.8, -.5, 7.8, 6.7], [1, 1., 1., 6., 6.],
                             [0, 3, 3, 3, 3], [1, 5, 5, 2, 2]], dtype=dtype)
        top = torch.randn(4, channels, 2, 3, generator=generator).to(dtype)
        bottom = torch.full((2, channels, 7, 8), .25, dtype=dtype)
        values = (aligned, 2, 3, 1., sampling, top, rois, bottom)
        info = signature((values, {}))
        for item in info[0]:
            if isinstance(item, dict) and 'device' in item:
                item['device'] = 'cuda:0'
        path = args.output/f'case-{index:02d}.pt'
        torch.save({'operation': 'native.roi_align.backward_avg_cuda', 'args': values,
                    'kwargs': {}, 'input_signature': info}, path)
        manifest['captures'][path.name] = digest(path)
        manifest['fixture_kinds'][path.name] = 'constructed dtype/channel-tail/existing-buffer parity fixture; not AMP training'
    manifest['fixture_generator_sha256'] = digest(Path(__file__))
    (args.output/'cases.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'captures': manifest['captures']}))


if __name__ == '__main__':
    main()
