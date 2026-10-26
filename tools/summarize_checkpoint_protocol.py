"""Compare the complete frozen-checkpoint cohort without hiding failed cases."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

import torch

from accuracy_data import digest
from accuracy_recipe import BUDGETS
from verify_checkpoint_protocol import MODELS, difference


def compact(comparison):
    value = dict(comparison)
    value['failure_count'] = len(value['failures'])
    value['failure_examples'] = value.pop('failures')[:5]
    return value


def read_cohort(root, vendor):
    controller = json.loads((root/'matrix-controller.json').read_text())
    expected = {f'inference-{model}-{profile}-{repeat}' for model in MODELS
                for profile in ('native', 'ultralytics') for repeat in ('01', '02')}
    expected |= {f'gradients-{model}-{profile}' for model in MODELS for profile in ('native', 'ultralytics', 'strict')}
    commands = {row['label']: row for row in controller['commands']}
    if not controller['completed'] or len(controller['commands']) != 28 or set(commands) != expected:
        raise ValueError('Incomplete or duplicated cohort: '+vendor)
    reports, summaries, manifests = {}, {}, {}
    for model in MODELS:
        manifests[model] = json.loads((root/'assets'/model/'manifest.json').read_text())['models'][model]
    for label in sorted(expected):
        path = root/'results'/label/'report.json'
        row = json.loads(path.read_text())
        if row['vendor'] != vendor or (row['status'] == 'completed') != (commands[label]['exit_code'] == 0):
            raise ValueError('Command/report outcome disagrees: '+label)
        if row['source']['sha256'] != controller['source_sha256']:
            raise ValueError('Runtime source drift: '+label)
        item = {key: row[key] for key in ('model', 'vendor', 'profile', 'phase', 'status', 'policy')}
        item.update(report_sha256=digest(path), warnings=dict(Counter(row['warnings'])),
                    exception=row.get('exception'), driver_sha256=row['driver_sha256'])
        if row['status'] == 'completed':
            if not row['weights_exact'] or row['model_state_sha256'] != manifests[row['model']]['state_sha256']:
                raise ValueError('Common checkpoint was not loaded: '+label)
            result = row['result']
            if row['phase'] == 'inference':
                if result['input_sha256'] != manifests[row['model']]['validation_input_sha256']:
                    raise ValueError('Input drift: '+label)
                comparisons = result['repeat_first_batch']
                item.update(metrics=result['metrics'], validation_batches=result['validation_batches'],
                    first_batch_repeats=len(comparisons),
                    exact_repeats=sum(c['bitwise_equal'] for c in comparisons),
                    within_tolerance_repeats=sum(c['within_tolerance'] for c in comparisons),
                    maximum_repeat_abs=max(c['max_abs_matching_tensors'] for c in comparisons),
                    probe_forward_median_ms=statistics.median(result['forward_batch_ms']))
            else:
                if result['optimizer_updates'] != 0:
                    raise ValueError('Unexpected optimization in backward diagnostic')
                item.update(optimizer_updates=0, cases=[])
                for case in result['cases']:
                    comparisons = case['repeats']
                    item['cases'].append({'batch_position': case['batch_position'], 'shape': case['shape'],
                        'comparisons': len(comparisons), 'baseline_loss': case['baseline_loss'],
                        'exact_losses': sum(c['loss_exact'] for c in comparisons),
                        'exact_gradients': sum(c['bitwise_equal'] for c in comparisons),
                        'within_tolerance_gradients': sum(c['within_tolerance'] for c in comparisons),
                        'maximum_abs': max(c['max_abs_matching_tensors'] for c in comparisons),
                        'first_nonexact': next((compact(c) for c in comparisons if not c['bitwise_equal']), None)})
        reports[label], summaries[label] = row, item
    return {'controller': controller, 'manifests': manifests, 'cases': summaries,
            'environment': next(iter(reports.values()))['environment']}


def load_outputs(root, label):
    return torch.load(root/'results'/label/'outputs.pt', map_location='cpu', weights_only=False)


def summarize(roots):
    cohorts = {vendor: read_cohort(root, vendor) for vendor, root in roots.items()}
    if cohorts['nvidia']['manifests'] != cohorts['metax']['manifests']:
        raise ValueError('Common asset manifests differ between vendors')
    if len({row['driver_sha256'] for cohort in cohorts.values() for row in cohort['cases'].values()}) != 1:
        raise ValueError('Different measurement drivers were used')
    rows, output_comparisons = [], []
    for model in MODELS:
        budget = BUDGETS[model]
        for profile in ('native', 'ultralytics'):
            label = f'inference-{model}-{profile}-01'
            cases = {vendor: cohorts[vendor]['cases'][label] for vendor in roots}
            if any(row['status'] != 'completed' for row in cases.values()):
                rows.append({'model': model, 'profile': profile, 'status': 'inference_failed'})
                continue
            n, m = (cases[v]['metrics'][budget['metric']] * budget['scale'] for v in ('nvidia', 'metax'))
            rows.append({'model': model, 'profile': profile, 'status': 'completed', 'metric': budget['metric'],
                         'nvidia_percent': n, 'metax_percent': m, 'delta_pp': m-n})
            output_comparisons.append({'kind': 'cross_vendor', 'model': model, 'profile': profile,
                'left': 'nvidia/'+label, 'right': 'metax/'+label,
                'all_metrics_equal': cases['nvidia']['metrics'] == cases['metax']['metrics'],
                **compact(difference(load_outputs(roots['nvidia'], label), load_outputs(roots['metax'], label)))})
        for vendor, root in roots.items():
            for left_profile, right_profile, left_repeat, right_repeat, kind in (
                ('native', 'native', '01', '02', 'fresh_process_repeat'),
                ('ultralytics', 'ultralytics', '01', '02', 'fresh_process_repeat'),
                ('native', 'ultralytics', '01', '01', 'profile_change')):
                left = f'inference-{model}-{left_profile}-{left_repeat}'
                right = f'inference-{model}-{right_profile}-{right_repeat}'
                cases = cohorts[vendor]['cases']
                if cases[left]['status'] == cases[right]['status'] == 'completed':
                    output_comparisons.append({'kind': kind, 'model': model, 'vendor': vendor, 'left': left, 'right': right,
                        'all_metrics_equal': cases[left]['metrics'] == cases[right]['metrics'],
                        'metric_delta_pp': (cases[right]['metrics'][budget['metric']]-cases[left]['metrics'][budget['metric']])*budget['scale'],
                        **compact(difference(load_outputs(root, left), load_outputs(root, right)))})
    return {'cohorts': cohorts, 'task_metrics': rows, 'output_comparisons': output_comparisons,
        'rtol': 1e-4, 'atol': 1e-5, 'scope': 'Fixed common checkpoint inference and within-vendor fixed-state/RNG backward probes, not independently trained cross-vendor accuracy',
        'performance_scope': 'Only separate first-batch CUDA-event forward probes; CPU tensor capture and full-validation audit are excluded from performance claims'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvidia', type=Path, required=True)
    parser.add_argument('--metax', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    value = summarize({'nvidia': args.nvidia, 'metax': args.metax})
    (args.output/'summary.json').write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    fields = ('model', 'profile', 'status', 'metric', 'nvidia_percent', 'metax_percent', 'delta_pp')
    with (args.output/'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(value['task_metrics'])
    print(json.dumps(value['task_metrics'], indent=2))
