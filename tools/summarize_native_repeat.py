"""Require all eight same-card native budgets and retain every replicate."""
import argparse
import json
from pathlib import Path

from accuracy_data import digest
from accuracy_recipe import BUDGETS

VARIANTS = ('original-01', 'original-02', 'new-01', 'new-02')


def summarize(root):
    controller = json.loads((root/'controller.json').read_text())
    if not controller['completed']:
        raise ValueError('The complete controller has not finished successfully')
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
                    or report['seed'] != 131 or report['mode'] != 'native'):
                raise ValueError('An incomplete or unmatched run cannot be included: '+str(path))
            environment = report['environment']
            if (environment['visible_devices'] != '1' or environment['deterministic']
                    or environment['cudnn_deterministic'] or environment['cudnn_benchmark']
                    or environment['matmul_tf32'] or environment['cudnn_tf32']):
                raise ValueError('Precision/device policy changed')
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
                'training_wall_seconds': report['training_wall_seconds'], 'environment': environment}
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
        rows.append({'model': model, 'metric': budget['metric'], 'runs': runs, 'sources': sources,
            'mean_new_minus_original_pp': sources['new']['mean']-sources['original']['mean'],
            'all_cross_source_deltas_pp': {old+' vs '+new: runs[new]['metric_percent']-runs[old]['metric_percent']
                for old in ('original-01', 'original-02') for new in ('new-01', 'new-02')},
            'common_inputs': {key: value for key, value in inputs[0].items() if key != 'learning_rates'},
            'cross_inference': cross})
    return {'complete_runs': 8, 'optimizer_updates': 14960, 'seed': 131, 'physical_device': 1,
        'gpu_uuid': controller['uuid'], 'records': rows, 'accuracy_threshold': None,
        'scope': 'Two repeats per source/model on one physical GPU, same seed; no population equivalence claim',
        'controller_sha256': digest(root/'controller.json')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'complete_runs': 8, 'optimizer_updates': 14960}))
