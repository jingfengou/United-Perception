"""Require all eight same-card budgets and retain every native-policy replicate."""
import argparse
import json
import math
from pathlib import Path

from accuracy_data import digest
from accuracy_recipe import BUDGETS

VARIANTS = ('original-01', 'original-02', 'new-01', 'new-02')


def summarize(root, *, deterministic=False, native_reference=None):
    controller = json.loads((root/'controller.json').read_text())
    if not controller['completed']:
        raise ValueError('The complete controller has not finished successfully')
    if controller['mode'] != ('native_deterministic' if deterministic else 'native'):
        raise ValueError('The requested policy does not match the completed controller')
    reports = list((root/'full').rglob('report.json'))
    if len(reports) != 8:
        raise ValueError('Exactly eight complete runs are required; no retry selection')
    rows = []
    for model in ('yolov5s', 'vit_b16'):
        budget = BUDGETS[model]
        runs, inputs = {}, []
        for variant in VARIANTS:
            directory = root/'full'/model/variant/model
            path = directory/'report.json'
            report = json.loads(path.read_text())
            execution = json.loads((directory.parent/'execution.json').read_text())
            if (not execution['completed'] or execution['physical_device'] != 1
                    or report['status'] != 'completed' or report['training_steps'] != budget['steps']
                    or report['seed'] != 131 or len(report['losses']) != budget['steps']
                    or not all(math.isfinite(value) for value in report['losses'])
                    or report['mode'] != ('native_deterministic' if deterministic else 'native')):
                raise ValueError('An incomplete or unmatched run cannot be included: '+str(path))
            environment = report['environment']
            if (environment['visible_devices'] != '1' or environment['deterministic'] != deterministic
                    or environment['cudnn_deterministic'] != deterministic or environment['cudnn_benchmark']
                    or environment['matmul_tf32'] or environment['cudnn_tf32']):
                raise ValueError('Precision/device policy changed')
            expected_controls = {key: None for key in environment['controls']}
            if deterministic:
                expected_controls['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                if environment.get('deterministic_warn_only') is not False:
                    raise ValueError('Strict determinism was not enabled')
            if environment['controls'] != expected_controls:
                raise ValueError('Unexpected inherited numerics controls')
            audit = json.loads((directory/'metric-audit.json').read_text())
            inference = json.loads((directory/'standalone-inference/inference-report.json').read_text())
            if (not audit['passed'] or audit['report_sha256'] != digest(path)
                    or not inference['weights_exact'] or inference['checkpoint_sha256'] != report['checkpoint_sha256']):
                raise ValueError('Missing or stale metric/inference audit')
            inputs.append({key: report[key] for key in ('initial_file_sha256', 'initial_state_sha256',
                'data_sha256', 'algorithm_config_sha256', 'first_batch_sha256', 'all_batches_sha256',
                'learning_rates', 'driver_sha256')})
            runs[variant] = {'metric_percent': report['final_metrics'][budget['metric']]*budget['scale'],
                'final_metrics': report['final_metrics'], 'restored_metrics': report['restored_metrics'],
                'standalone_metrics': inference['metrics'], 'report_sha256': digest(path),
                'checkpoint_sha256': report['checkpoint_sha256'], 'source_sha256': report['source']['sha256'],
                'final_state_sha256': report['final_state_sha256'], 'training_steps': report['training_steps'],
                'training_wall_seconds': report['training_wall_seconds'], 'environment': environment,
                'peak_allocated_bytes': report['peak_allocated_bytes'],
                'peak_reserved_bytes': report['peak_reserved_bytes']}
        if any(value != inputs[0] for value in inputs[1:]):
            raise ValueError('Common weights/data/batches/recipe/LR do not match: '+model)
        sources = {}
        for source in ('original', 'new'):
            values = [runs[source+'-'+number]['metric_percent'] for number in ('01', '02')]
            sources[source] = {'all_values': values, 'mean': sum(values)/2, 'min': min(values),
                               'max': max(values), 'repeat_spread_pp': max(values)-min(values)}
        cross = []
        for trained, runtime in (('original', 'new'), ('new', 'original')):
            path = root/'cross-inference'/model/(trained+'-to-'+runtime)/'inference-report.json'
            inference = json.loads(path.read_text())
            reference = runs[trained+'-01']
            if not inference['weights_exact'] or inference['checkpoint_sha256'] != reference['checkpoint_sha256']:
                raise ValueError('Cross inference did not load the exact selected checkpoint')
            cross.append({'trained_with': trained, 'inferred_with': runtime, 'raw_sha256': digest(path),
                'all_metrics_exact': inference['metrics'] == reference['final_metrics'],
                'metrics': inference['metrics'], 'metric_delta_pp':
                (inference['metrics'][budget['metric']]-reference['final_metrics'][budget['metric']])*budget['scale']})
        comparisons = []
        for left, right in (('original-01', 'original-02'), ('new-01', 'new-02'), ('original-01', 'new-01')):
            a = json.loads((root/'full'/model/left/model/'report.json').read_text())
            b = json.loads((root/'full'/model/right/model/'report.json').read_text())
            differences = [abs(x-y) for x, y in zip(a['losses'], b['losses'])]
            comparisons.append({'left': left, 'right': right, 'losses_exact': a['losses'] == b['losses'],
                'first_exact_loss_difference': next((i+1 for i, value in enumerate(differences) if value), None),
                'max_abs_loss_difference': max(differences),
                'losses_exceeding_original_tolerance': sum(abs(x-y) > 1e-5+1e-4*abs(x) for x,y in zip(a['losses'],b['losses'])),
                'final_model_state_exact': a['final_state_sha256'] == b['final_state_sha256'],
                'epoch_metrics_exact': a['epoch_metrics'] == b['epoch_metrics'],
                'final_metrics_exact': a['final_metrics'] == b['final_metrics']})
        row = {'model': model, 'metric': budget['metric'], 'runs': runs, 'sources': sources,
            'mean_new_minus_original_pp': sources['new']['mean']-sources['original']['mean'],
            'all_cross_source_deltas_pp': {old+' vs '+new: runs[new]['metric_percent']-runs[old]['metric_percent']
                for old in ('original-01', 'original-02') for new in ('new-01', 'new-02')},
            'common_inputs': {key: value for key, value in inputs[0].items() if key != 'learning_rates'},
            'cross_inference': cross, 'same_source_and_cross_source_comparisons': comparisons}
        if native_reference is not None:
            reference_controller = json.loads((native_reference/'controller.json').read_text())
            if not reference_controller['completed'] or reference_controller['uuid'] != controller['uuid']:
                raise ValueError('Native performance reference is incomplete or from a different card')
            performance = {}
            for source in ('original', 'new'):
                native, controlled = [], []
                for number in ('01', '02'):
                    variant = source+'-'+number
                    reference_path = native_reference/'full'/model/variant/model/'report.json'
                    reference = json.loads(reference_path.read_text())
                    if (reference['mode'] != 'native' or reference['status'] != 'completed'
                            or reference['training_steps'] != budget['steps']
                            or any(reference[key] != inputs[0][key] for key in inputs[0] if key != 'driver_sha256')
                            or reference['source']['sha256'] != runs[variant]['source_sha256']):
                        raise ValueError('Native timing reference has different workload or source')
                    native.append({key: reference[key] for key in ('training_wall_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')})
                    controlled.append({key: runs[variant][key] for key in native[-1]})
                means = {policy: {key: sum(r[key] for r in records)/len(records) for key in records[0]}
                         for policy, records in (('native', native), ('deterministic', controlled))}
                performance[source] = {'native': native, 'deterministic': controlled, 'means': means,
                    'training_time_change_percent': (means['deterministic']['training_wall_seconds']/means['native']['training_wall_seconds']-1)*100,
                    'reference_reports': {number: digest(native_reference/'full'/model/(source+'-'+number)/model/'report.json') for number in ('01','02')}}
            row['performance_vs_prior_native'] = performance
        rows.append(row)
    return {'complete_runs': 8, 'optimizer_updates': 14960, 'seed': 131, 'physical_device': 1,
        'gpu_uuid': controller['uuid'], 'records': rows, 'accuracy_threshold': None,
        'numerics': 'native_deterministic' if deterministic else 'native',
        'scope': 'Two repeats per source/model on one physical GPU, same seed; no population equivalence claim',
        'controller_sha256': digest(root/'controller.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--native-reference', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.root, deterministic=args.deterministic, native_reference=args.native_reference)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete_runs': 8, 'optimizer_updates': 14960}))
