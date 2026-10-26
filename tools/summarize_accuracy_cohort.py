"""Summarize all 15 native runs without an accuracy threshold or seed selection."""
import argparse
import json
from pathlib import Path

from accuracy_data import digest
from accuracy_recipe import BUDGETS

LANES = ('original_nvidia', 'new_nvidia', 'new_metax')


def summarize(roots):
    runs = {}
    for root in roots:
        for path in sorted(root.rglob('report.json')):
            report = json.loads(path.read_text())
            key = (report['model'], report['lane'])
            if key in runs:
                raise ValueError('Duplicate run; do not select a favorable retry: '+str(key))
            if report['status'] != 'completed':
                raise ValueError('A probe/incomplete run cannot count as full training')
            runs[key] = (path, report)
    if set(runs) != {(model, lane) for model in BUDGETS for lane in LANES}:
        raise ValueError('All five models and all three lanes are required')
    records = []
    for model, budget in BUDGETS.items():
        values, rows, references = {}, {}, {}
        for lane in LANES:
            path, report = runs[model, lane]
            if (report['seed'] != 131 or report['epochs'] != budget['epochs'] or report['mode'] != 'native'
                    or report['training_steps'] != budget['steps'] or report['environment']['deterministic']
                    or report['environment']['cudnn_deterministic']):
                raise ValueError('Budget/seed/numerical policy mismatch: '+str(path))
            audit = json.loads((path.parent/'metric-audit.json').read_text())
            inference = json.loads((path.parent/'standalone-inference/inference-report.json').read_text())
            if (not audit['passed'] or audit['report_sha256'] != digest(path)
                    or inference['checkpoint_sha256'] != report['checkpoint_sha256'] or not inference['weights_exact']):
                raise ValueError('Missing/stale audit or inference proof')
            references[lane] = {key: report[key] for key in ('data_sha256', 'initial_file_sha256',
                'initial_state_sha256', 'algorithm_config_sha256', 'learning_rates', 'driver_sha256', 'adapter_sha256')}
            values[lane] = report['final_metrics'][budget['metric']]*budget['scale']
            rows[lane] = {'report_sha256': digest(path), 'training_steps': report['training_steps'],
                'final_metrics': report['final_metrics'], 'restored_metrics': report['restored_metrics'],
                'independent_inference': inference['metrics'], 'training_wall_seconds': report['training_wall_seconds'],
                'peak_allocated_bytes': report['peak_allocated_bytes'], 'peak_reserved_bytes': report['peak_reserved_bytes'],
                'first_batch_sha256': report['first_batch_sha256'], 'all_batches_sha256': report['all_batches_sha256'],
                'final_state_sha256': report['final_state_sha256'], 'environment': report['environment'],
                'source_sha256': report['source']['sha256']}
        if not references['original_nvidia'] == references['new_nvidia'] == references['new_metax']:
            raise ValueError('Unmatched common inputs/recipe/LR schedule: '+model)
        if rows['new_nvidia']['source_sha256'] != rows['new_metax']['source_sha256']:
            raise ValueError('New UP source differs across vendors')
        records.append({'model': model, 'primary_metric': budget['metric'], 'values_percent': values,
            'new_nvidia_minus_original_pp': values['new_nvidia']-values['original_nvidia'],
            'new_metax_minus_new_nvidia_pp': values['new_metax']-values['new_nvidia'],
            'new_metax_minus_original_pp': values['new_metax']-values['original_nvidia'],
            'absolute_vendor_difference_pp': abs(values['new_metax']-values['new_nvidia']), 'runs': rows,
            'common_inputs': {key: value for key, value in references['original_nvidia'].items() if key != 'learning_rates'}})
    return {'complete_runs': len(runs), 'seed': 131, 'optimizer_updates': sum(b['steps'] for b in BUDGETS.values())*3,
        'scope': 'Fixed CIFAR-10 10000/2000 and PennFudan 120/50 budgets; not ImageNet/COCO convergence',
        'accuracy_threshold': None, 'accuracy_verdict': 'report differences only; no equivalence/pass claim',
        'repeat_variance': 'Not estimated by this single-run-per-lane cohort; numerical differences are retained',
        'detr_reference': 'Same pinned external official DETR adapter in all three UP Runners; original UP has no DETR recipe',
        'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete_runs': report['complete_runs'], 'optimizer_updates': report['optimizer_updates']}))
