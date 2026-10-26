"""Check ordered GPU RoIAlign against an independent CPU autograd reference.

CPU computation is an audit oracle only; UP model execution stays on the GPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from up_platform import configure_platform
from up_platform.ops._native.loader import load_native
from compare_platform_models import compare, cpu
from run_platform_validation import source_manifest


def reference(features, rois, height, width, scale, ratio, aligned):
    values = []
    for roi in rois:
        batch = int(roi[0])
        start_x, start_y, end_x, end_y = roi[1:] * scale - (.5 if aligned else 0.)
        roi_w, roi_h = end_x - start_x, end_y - start_y
        if not aligned:
            roi_w, roi_h = roi_w.clamp_min(1), roi_h.clamp_min(1)
        bin_h, bin_w = roi_h/height, roi_w/width
        grid_h = ratio if ratio > 0 else math.ceil(float(roi_h/height))
        grid_w = ratio if ratio > 0 else math.ceil(float(roi_w/width))
        count = max(grid_h*grid_w, 1)
        for ph in range(height):
            for pw in range(width):
                value = features[batch, :, 0, 0] * 0
                for iy in range(grid_h):
                    y = start_y + ph*bin_h + (iy+.5)*bin_h/grid_h
                    for ix in range(grid_w):
                        x = start_x + pw*bin_w + (ix+.5)*bin_w/grid_w
                        if y < -1 or y > features.shape[2] or x < -1 or x > features.shape[3]:
                            continue
                        sy, sx = y.clamp(0, features.shape[2]-1), x.clamp(0, features.shape[3]-1)
                        yl, xl = int(sy), int(sx)
                        yh, xh = min(yl+1, features.shape[2]-1), min(xl+1, features.shape[3]-1)
                        ly, lx = sy-yl, sx-xl
                        hy, hx = 1-ly, 1-lx
                        value = value + ((hy*hx)*features[batch, :, yl, xl]
                                         + (hy*lx)*features[batch, :, yl, xh]
                                         + (ly*hx)*features[batch, :, yh, xl]
                                         + (ly*lx)*features[batch, :, yh, xh])
                values.append(value/count)
    if not values:
        return features.sum() * features.new_empty((len(rois), features.shape[1], height, width))
    return torch.stack(values).reshape(len(rois), height, width, features.shape[1]).permute(0, 3, 1, 2)


def evaluate(function, features, rois, gradient):
    x = features.detach().clone().requires_grad_()
    y = function(x, rois)
    y.backward(gradient)
    return {'output': cpu(y), 'gradient': cpu(x.grad)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    platform = configure_platform({'backend': args.vendor}, purpose='training')
    platform.device.set_device(0)
    extension = load_native(args.vendor)
    assert extension.roi_align.deterministic_backward_version == 1
    torch.manual_seed(131)
    roi_align = platform.category('detection').roi_align
    records, raw = [], {}
    rois = [[0, -.8, -.5, 7.8, 6.7], [0, .1, .2, 5.9, 5.1],
            [1, 1, 1, 6, 6], [1, 1, 1, 6, 6], [0, 0, 0, 1e-12, 1e-12],
            [1, -3, -3, -1, -1], [0, 8, 7, 10, 10],
            [1, 3, 3, 3, 3], [0, 5, 5, 2, 2]]
    for dtype in (torch.float32, torch.float64):
        for aligned in (False, True):
            for ratio in (0, 2):
                name = f'{dtype}/{aligned}/{ratio}'
                print('case', name, flush=True)
                features = torch.randn(2, 3, 7, 8, dtype=dtype)
                boxes = torch.tensor(rois, dtype=dtype)
                gradient = torch.randn(len(rois), 3, 3, 2, dtype=dtype).transpose(2, 3)
                expected = evaluate(lambda x, r: reference(x, r, 2, 3, 1., ratio, aligned), features, boxes, gradient)
                def operation(x, r):
                    return roi_align(x, r, 2, 3, 1., ratio, 'AVG', aligned=aligned)
                features, boxes, gradient = features.cuda().contiguous(), boxes.cuda().contiguous(), gradient.cuda()
                torch.use_deterministic_algorithms(True)
                actual = evaluate(operation, features, boxes, gradient)
                repeats = [evaluate(operation, features, boxes, gradient) for _ in range(2)]
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    stream.wait_stream(torch.cuda.default_stream())
                    streamed = evaluate(operation, features, boxes, gradient)
                stream.synchronize()
                torch.use_deterministic_algorithms(False)
                atomic = evaluate(operation, features, boxes, gradient)
                torch.use_deterministic_algorithms(True)
                accumulated = torch.full_like(features, .25)
                extension.roi_align.backward_avg_cuda(aligned, 2, 3, 1., ratio, gradient.contiguous(), boxes, accumulated)
                checks = {'oracle': compare(expected, actual), 'atomic': compare(atomic, actual),
                          'stream': compare(actual, streamed),
                          'existing_gradient_buffer': compare(actual['gradient']+.25, cpu(accumulated))}
                bitwise = [all(torch.equal(actual[k], other[k]) for k in actual) for other in repeats]
                records.append({'case': name, 'checks': checks, 'bitwise_repeats': bitwise,
                                'passed': all(v['passed'] for v in checks.values()) and all(bitwise)})
                raw[name] = {'oracle': expected, 'ordered': actual, 'atomic': atomic, 'stream': streamed,
                             'accumulated': cpu(accumulated), 'repeats': repeats}
                (args.output/'partial.json').write_text(json.dumps({'records': records}, indent=2)+'\n')
    torch.use_deterministic_algorithms(True)
    features = torch.randn(1, 2, 3, 4, device='cuda', dtype=torch.float64, requires_grad=True)
    boxes = torch.tensor([[0, .2, .1, 2.7, 1.8]], device='cuda', dtype=torch.float64)
    gradcheck = torch.autograd.gradcheck(lambda x: roi_align(x, boxes, 2, 2, 1., 2, 'AVG', aligned=True),
                                       (features,), eps=1e-6, rtol=1e-4, atol=1e-5)
    empty = evaluate(lambda x, r: roi_align(x, r, 2, 2, 1., 2, 'AVG', aligned=True),
                     features, boxes[:0], features.new_empty((0, 2, 2, 2)))
    records.append({'case': 'empty_rois', 'passed': empty['output'].shape == (0, 2, 2, 2) and torch.count_nonzero(empty['gradient']).item() == 0})
    path = args.output/'values.pt'
    torch.save(raw, path)
    source = source_manifest(Path(__file__).resolve().parents[1])
    (args.output/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    report = {'vendor': args.vendor, 'environment': platform.device.environment(),
              'scope': 'FP32/FP64 AVG GPU forward/backward, fixed-order reduction; AMP/FP16 not validated',
              'rtol': 1e-4, 'atol': 1e-5, 'gradcheck': gradcheck,
              'source_sha256': source['sha256'], 'binary_sha256': hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
              'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'records': records, 'passed': gradcheck and all(r['passed'] for r in records)}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'vendor': args.vendor, 'passed': report['passed'], 'gradcheck': gradcheck,
                      'failures': [r for r in records if not r['passed']]}))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
