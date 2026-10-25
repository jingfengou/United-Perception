"""Compare original UP and current platform using their separate NVIDIA binaries.

Each binary lives in its own process to avoid pybind enum/type collisions.
The existing shared-extension routing evidence is retained, not relabelled.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from compare_platform_native import cases, compare, original_module, run
from platform_operator_cases import fingerprint


def worker(args):
    from up_platform import configure_platform
    platform = configure_platform({'backend': 'nvidia', 'device': 'cuda:0'}, purpose=args.purpose)
    platform.device.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(131)
    platform.device.manual_seed(131)
    if args.route == 'original':
        binary = next((args.baseline/'up/extensions').glob('_C*.so'))
        spec = importlib.util.spec_from_file_location('_C', binary)
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    else:
        from up_platform.ops._native.loader import load_native
        extension = load_native('nvidia')
        binary = Path(extension.__file__)
    flags = SimpleNamespace(aligned=False, offset=1)
    references, values, rows = {}, {}, []
    args.output.mkdir(parents=True, exist_ok=False)
    for index, (category, method, module, entry, inputs, gradients, kwargs) in enumerate(cases(platform.device.device())):
        name = f'{category}.{method}/{index}'
        flags.aligned, flags.offset = kwargs.get('aligned', False), kwargs.get('offset', 1)
        if args.route == 'original':
            if module not in references:
                references[module] = original_module(args.baseline, module, extension, flags)
            function = references[module][entry]
            function = function.apply if isinstance(function, type) else function
            arguments = {}
        else:
            function = getattr(platform.category(category), method)
            arguments = kwargs
        row = {'interface': category+'.'+method, 'case': name,
               'inputs_sha256': fingerprint((inputs, kwargs)), 'route': args.route}
        try:
            result = run(function, inputs, arguments, gradients, platform.device, args.purpose == 'training')
            repeat = run(function, inputs, arguments, gradients, platform.device, args.purpose == 'training')
            row['repeat'] = compare(result, repeat)
            values[name] = result
            row['executed'] = True
        except Exception as error:
            row.update(executed=False, error=repr(error))
        rows.append(row)
        print(json.dumps(row), flush=True)
    torch.save(values, args.output/'values.pt')
    (args.output/'summary.json').write_text(json.dumps({'route': args.route, 'purpose': args.purpose,
        'binary': str(binary), 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'raw_sha256': hashlib.sha256((args.output/'values.pt').read_bytes()).hexdigest(),
        'rtol': 1e-4, 'atol': 1e-5, 'records': rows}, indent=2)+'\n')


def summarize(args):
    rows = []
    for purpose in ('training', 'inference'):
        paths = [args.results/(purpose+'-'+route) for route in ('original', 'routed')]
        reports = [json.loads((path/'summary.json').read_text()) for path in paths]
        values = [torch.load(path/'values.pt', weights_only=False, map_location='cpu') for path in paths]
        assert len(reports[0]['records']) == len(reports[1]['records']) == 31
        for old, new in zip(reports[0]['records'], reports[1]['records']):
            assert old['case'] == new['case'] and old['inputs_sha256'] == new['inputs_sha256']
            row = {'interface': old['interface'], 'case': old['case'], 'purpose': purpose,
                   'reference_kind': 'pinned_original_Python_and_original_NVIDIA_binary',
                   'reference_repeat': old.get('repeat'), 'routed_repeat': new.get('repeat')}
            if old['executed'] and new['executed']:
                row['comparison'] = compare(values[0][old['case']], values[1][new['case']])
                row['status'] = 'within_tolerance' if row['comparison']['passed'] else 'difference'
            else:
                row.update(status='execution_error', original_error=old.get('error'), routed_error=new.get('error'))
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump({'rtol': 1e-4, 'atol': 1e-5, 'records': rows}, handle, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('original', 'routed'))
    parser.add_argument('--purpose', choices=('training', 'inference'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', type=Path)
    args = parser.parse_args()
    summarize(args) if args.results else worker(args)
