"""Audit paired fixed-step traces separately from final validation metrics.

Read saved CPU metadata only. A fixed-step comparison is not a convergence or
statistical-equivalence test; task accuracy has no automatic passing threshold.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

RTOL, ATOL = 1e-4, 1e-5
MODELS = ('resnet50', 'yolov5s', 'vit_b16')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_traces(reference, actual, *, steps=500):
    if not reference or len(reference) != steps or len(actual) != steps:
        raise ValueError('Both traces must contain the complete requested budget')
    expected = list(range(1, steps+1))
    if [row['step'] for row in reference] != expected or [row['step'] for row in actual] != expected:
        raise ValueError('Duplicate, missing, or reordered training steps')
    for left, right in zip(reference, actual):
        if not left.get('input_sha256') or left['input_sha256'] != right.get('input_sha256'):
            raise ValueError('Different input batch at step '+str(left['step']))
        if left['lr_next'] != right['lr_next']:
            raise ValueError('Different learning-rate schedule at step '+str(left['step']))
        if not math.isfinite(left['loss']) or not math.isfinite(right['loss']):
            raise ValueError('Non-finite loss cannot establish numerical agreement')
    differences = [abs(left['loss']-right['loss']) for left, right in zip(reference, actual)]
    unequal = [i for i, value in enumerate(differences) if value != 0]
    failures = [i for i, (left, difference) in enumerate(zip(reference, differences))
                if difference > ATOL+RTOL*abs(left['loss'])]
    updates = [left['step'] for left, right in zip(reference, actual)
               if left['optimizer_updates'] != right['optimizer_updates']]
    first = unequal[0] if unequal else None
    return {'steps': steps, 'inputs_and_learning_rates_match': True,
            'first_loss_difference_step': None if first is None else first+1,
            'first_loss_difference_values': None if first is None else {
                'reference': reference[first]['loss'], 'actual': actual[first]['loss']},
            'first_loss_exceeding_tolerance_step': None if not failures else failures[0]+1,
            'loss_steps_exceeding_tolerance': len(failures),
            'max_loss_absolute_difference': max(differences),
            'first_optimizer_update_count_difference_step': next(iter(updates), None),
            'rtol': RTOL, 'atol': ATOL}


def read_case(directory, lane, model, parent, parent_sha):
    target = directory/'results'/lane/model
    report = json.loads((target/'train/report.json').read_text())
    inference = json.loads((target/'inference/inference.json').read_text())
    audit = json.loads((directory/'cpu-audits'/(lane+'-'+model+'.json')).read_text())
    identity = report['identity']
    if (report['status'] != 'preflight_completed' or report['progress']['completed_steps'] != 500
            or identity['preflight_steps'] != 500 or identity['seed'] != parent['seed']
            or identity['plan_sha256'] != parent_sha or identity['world_size'] != 2
            or identity['source_sha256'] != parent['sources'][lane]['sha256']
            or identity['data_identity'] != parent['data_identity']
            or report['initial_state_sha256'] != parent['models'][model]['initial_state_sha256']):
        raise ValueError('Unmatched source/data/initial state or wrong fixed-step budget: '+str(target))
    if (audit['status'] != 'audited_500_steps' or not audit['metrics_recompute_exact']
            or audit['scheduled_steps'] != 500 or audit['report_sha256'] != digest(target/'train/report.json')
            or audit['inference_sha256'] != digest(target/'inference/inference.json')
            or report['final_metrics'] != audit['recomputed_metrics']
            or inference['metrics'] != audit['inference_metrics'] or identity != inference['identity']
            or not inference['weights_exact'] or not inference['rank_local_buffers_restored']):
        raise ValueError('Missing or stale CPU integrity/metric/inference audit: '+str(target))
    traces = {}
    for rank in range(2):
        path = target/'train'/f'train.rank{rank}.jsonl'
        if digest(path) != audit['training_trace_sha256'][str(rank)]:
            raise ValueError('Training trace changed after its CPU audit')
        traces[str(rank)] = [json.loads(line) for line in path.read_text().splitlines()]
    return report, traces, audit


def summarize(nvidia, metax, parent_path):
    parent = json.loads(parent_path.read_text())
    parent_sha = digest(parent_path)
    cases, comparisons = {}, {}
    for model in MODELS:
        reports, traces = {}, {}
        for lane in ('original_nvidia', 'new_nvidia', 'new_metax'):
            directory = metax if lane == 'new_metax' else nvidia
            report, trace, audit = read_case(directory, lane, model, parent, parent_sha)
            reports[lane], traces[lane] = report, trace
            updates = report['progress']['optimizer_updates']
            if not 0 <= updates <= 500:
                raise ValueError('Invalid optimizer update count')
            cases[lane+'/'+model] = {'metrics': report['final_metrics'], 'scheduled_steps': 500,
                'optimizer_updates': updates, 'amp_skipped_steps': 500-updates,
                'training_seconds': report['progress']['training_seconds'],
                'evaluation_seconds': report['progress']['evaluation_seconds'],
                'rank0_peak_allocated_bytes': report['peak_allocated_bytes'],
                'train_inference_predictions_identical': audit['predictions_identical'],
                'report_sha256': audit['report_sha256'], 'environment': report['environment']}
        for lane in ('new_nvidia', 'new_metax'):
            for key in ('algorithm_config_sha256', 'driver_sha256', 'data_identity'):
                if reports[lane]['identity'][key] != reports['original_nvidia']['identity'][key]:
                    raise ValueError('Unmatched '+key)
        key, scale = ('bbox.AP', 100.) if model == 'yolov5s' else ('top1', 1.)
        values = {lane: report['final_metrics'][key]*scale for lane, report in reports.items()}
        comparisons[model] = {'metric': key, 'units': 'AP points' if scale == 100 else 'percent',
            'metrics': values, 'new_minus_original_nvidia_pp': values['new_nvidia']-values['original_nvidia'],
            'metax_minus_new_nvidia_pp': values['new_metax']-values['new_nvidia'],
            'trajectories': {name: {str(rank): compare_traces(traces[left][str(rank)], traces[right][str(rank)])
                                   for rank in range(2)} for name, left, right in (
                ('new_vs_original_nvidia', 'original_nvidia', 'new_nvidia'),
                ('metax_vs_new_nvidia', 'new_nvidia', 'new_metax'))}}
    return {'purpose': 'three_models_three_lanes_500_steps', 'audited_training_runs': 9,
            'scheduled_steps_total': 4500, 'cases': cases, 'comparisons': comparisons,
            'full_epoch_completed': False, 'convergence_claim': False, 'accuracy_pass_threshold': None,
            'limitations': ['One seed/native execution; no statistical equivalence claim',
                            'Loss differences and operator differences do not by themselves prove final-accuracy causation',
                            'Training time includes input hashing and data-loader wait; memory is rank0 allocator peak']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('nvidia', 'metax', 'parent-plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.nvidia, args.metax, args.parent_plan)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'runs': result['audited_training_runs'], 'comparisons': result['comparisons']}))


if __name__ == '__main__':
    main()
