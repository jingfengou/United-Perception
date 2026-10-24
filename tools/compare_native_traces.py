"""Compare repeated gradient observations at unchanged rtol=1e-4, atol=1e-5."""
import argparse
import json
from pathlib import Path

import torch


def compare(left, right):
    a = json.loads((left/'report.json').read_text())
    b = json.loads((right/'report.json').read_text())
    keys = ('initial_file_sha256', 'initial_state_sha256', 'data_sha256', 'algorithm_config_sha256',
            'first_batch_sha256', 'all_batches_sha256', 'learning_rates', 'driver_sha256')
    if any(a[key] != b[key] for key in keys):
        raise ValueError('Unmatched diagnostic inputs')
    if (a['status'] != 'probe_completed' or b['status'] != 'probe_completed'
            or a['training_steps'] != b['training_steps']):
        raise ValueError('Only matching complete short traces can be compared')
    oa = json.loads((left/'observations.json').read_text())
    ob = json.loads((right/'observations.json').read_text())
    if oa['observer_sha256'] != ob['observer_sha256']:
        raise ValueError('Observer changed between traces')
    steps = []
    for i, (x, y) in enumerate(zip(oa['records'], ob['records']), 1):
        ga = torch.load(left/('gradients-step'+str(i)+'.pt'), map_location='cpu', weights_only=True, mmap=True)
        gb = torch.load(right/('gradients-step'+str(i)+'.pt'), map_location='cpu', weights_only=True, mmap=True)
        if ga.keys() != gb.keys():
            raise ValueError('Gradient names differ')
        exact, failed, changed = 0, [], []
        for name in ga:
            u, v = ga[name], gb[name]
            if u.shape != v.shape or u.dtype != v.dtype:
                raise ValueError('Gradient shape/dtype changed: '+name)
            equal = torch.equal(u, v)
            if equal:
                exact += 1
                continue
            delta = float((u.double()-v.double()).abs().max())
            row = {'parameter': name, 'max_abs': delta}
            changed.append(row)
            if not torch.allclose(u, v, rtol=1e-4, atol=1e-5):
                failed.append(row)
        steps.append({'step': i, 'loss_left': x['loss'], 'loss_right': y['loss'],
            'cpu_rng_equal': x['cpu_rng'] == y['cpu_rng'], 'gpu_rng_equal': x['gpu_rng'] == y['gpu_rng'],
            'gpu_rng_after_equal': x['gpu_rng_after'] == y['gpu_rng_after'],
            'module_modes_equal': x['module_modes'] == y['module_modes'], 'gradients': len(ga),
            'bitwise_equal_gradients': exact, 'changed_gradients': changed, 'tolerance_failures': failed})
    if len(steps) != a['training_steps']:
        raise ValueError('Incomplete gradient observation')
    return {'left': str(left), 'right': str(right), 'model': a['model'], 'rtol': 1e-4, 'atol': 1e-5,
            'matching_inputs': {key: a[key] for key in keys if key != 'learning_rates'},
            'leaf_modules_equal': json.loads((left/'leaf-modules.json').read_text()) == json.loads((right/'leaf-modules.json').read_text()),
            'steps': steps, 'scope': 'Changed gradients are observations, not proof of the first causal operator'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', type=Path, required=True)
    parser.add_argument('--right', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result = compare(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'model': result['model'], 'leaf_modules_equal': result['leaf_modules_equal'],
        'steps': [{key: len(value) if key in ('changed_gradients', 'tolerance_failures') else value
                   for key, value in row.items()} for row in result['steps']]}, indent=2))
