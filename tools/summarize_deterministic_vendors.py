"""Matched complete C500 budgets against the pinned NVIDIA deterministic cohort."""
import argparse
import csv
import json
import math
from pathlib import Path

from accuracy_data import digest
from accuracy_recipe import BUDGETS
from audit_model_verify import delta_pp

VARIANTS = ('original-01', 'new-01', 'new-02', 'original-02')


def loss_comparison(reference, actual):
    if not reference or len(reference) != len(actual):
        raise ValueError('Loss traces must be nonempty and have equal complete budgets')
    if not all(math.isfinite(x) for x in (*reference, *actual)):
        raise ValueError('Non-finite loss trace')
    differences = [abs(a-b) for a,b in zip(reference, actual)]
    failures = [i+1 for i,(a,b) in enumerate(zip(reference, actual)) if abs(a-b) > 1e-5+1e-4*abs(a)]
    return {'steps':len(reference), 'values_equal':reference == actual,
        'first_difference_step':next((i+1 for i,d in enumerate(differences) if d), None),
        'first_exceeding_tolerance_step':next(iter(failures), None),
        'exceeding_tolerance_steps':len(failures), 'maximum_abs_difference':max(differences),
        'rtol':1e-4, 'atol':1e-5}


def check_policy(environment):
    if (environment.get('deterministic') is not True or environment.get('cudnn_deterministic') is not True
            or environment.get('deterministic_warn_only') is not False
            or environment.get('fill_uninitialized_memory') is not True
            or environment.get('cudnn_benchmark') is not False
            or environment.get('matmul_tf32') is not False or environment.get('cudnn_tf32') is not False):
        raise ValueError('Strict native deterministic FP32 policy was not retained')
    expected = {'CUBLAS_WORKSPACE_CONFIG':':4096:8', 'MCDNN_STABLE_RESULT_MOD':None,
        'MCDNN_USE_DETERMINISTIC_ALGO':None, 'PYTORCH_DISABLE_CUDA_CUDNN_TF32':None,
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE':None}
    if environment['controls'] != expected:
        raise ValueError('Unexpected vendor/numerics override')


def read_run(directory, expected, protocol, *, metax=False):
    path = directory/'report.json'
    report = json.loads(path.read_text())
    if (report['status'] != 'completed' or report['mode'] != 'native_deterministic'
            or report['seed'] != protocol['seed'] or report['training_steps'] != expected['expected_steps']
            or report['epochs'] != expected['epochs'] or len(report['losses']) != expected['expected_steps']):
        raise ValueError('Incomplete or wrong-budget training: '+str(path))
    for key in ('initial_file_sha256','initial_state_sha256','algorithm_config_sha256', 'driver_sha256','all_batches_sha256'):
        if report[key] != expected[key]:
            raise ValueError('Unmatched '+key+': '+str(path))
    if report['data_sha256'] != protocol['data_sha256']:
        raise ValueError('Unmatched data manifest')
    check_policy(report['environment'])
    audit = json.loads((directory/'metric-audit.json').read_text())
    inference = json.loads((directory/'standalone-inference/inference-report.json').read_text())
    if (not audit['passed'] or audit['report_sha256'] != digest(path) or audit['metrics'] != report['final_metrics']
            or not inference['weights_exact'] or inference['status'] != 'completed'
            or inference['numerics'] != 'native_deterministic'
            or inference['checkpoint_sha256'] != report['checkpoint_sha256']
            or inference['metrics'] != report['final_metrics'] or report['restored_metrics'] != report['final_metrics']
            or not report['restore_weights_exact'] or not report['checkpoint_all_states_finite']):
        raise ValueError('Failed/stale restore, metric or independent inference audit: '+str(path))
    if metax:
        if (report['lane'] != 'new_metax' or report['source']['sha256'] != protocol['runtime_source_sha256']
                or report['environment']['visible_devices'] != '2'
                or audit['predictions_sha256'] != digest(directory/'final-predictions/results.txt')):
            raise ValueError('C500 source/device/prediction mismatch')
    return report


def summarize(metax, reference, protocol):
    controller = json.loads((metax/'controller.json').read_text())
    execution = json.loads((metax/'results/execution.json').read_text())
    if (not controller['completed'] or not execution['completed'] or len(execution['jobs']) != 2
            or {j['model'] for j in execution['jobs']} != set(protocol['models'])
            or any(j['exit_code'] or j['metric_audit_exit_code'] or j['inference_exit_code'] for j in execution['jobs'])):
        raise ValueError('All two complete training, metric and inference jobs are required')
    rows = []
    for model,expected in protocol['models'].items():
        nvidia = {}
        for variant in VARIANTS:
            directory = reference/model/variant
            if digest(directory/'report.json') != expected['reference_report_sha256'][variant]:
                raise ValueError('NVIDIA reference changed: '+model+'/'+variant)
            nvidia[variant] = read_run(directory, expected, protocol)
        n = nvidia['new-01']
        for r in nvidia.values():
            if any(r[key] != n[key] for key in ('losses','learning_rates','final_state_sha256','final_metrics','first_batch_sha256')):
                raise ValueError('Pinned NVIDIA repeats disagree')
        directory = metax/'results'/model
        m = read_run(directory, expected, protocol, metax=True)
        if m['learning_rates'] != n['learning_rates'] or m['first_batch_sha256'] != n['first_batch_sha256']:
            raise ValueError('Cross-vendor inputs/LR schedule differ')
        metric_deltas = {key: delta_pp(value, m['final_metrics'][key], unit='ratio' if key.startswith('bbox.') else 'percent')
                         for key,value in n['final_metrics'].items()}
        primary = BUDGETS[model]['metric']
        rows.append({'model':model,'primary_metric':primary,'values_percent':{
            'original_nvidia':nvidia['original-01']['final_metrics'][primary]*BUDGETS[model]['scale'],
            'new_nvidia':n['final_metrics'][primary]*BUDGETS[model]['scale'],
            'new_metax':m['final_metrics'][primary]*BUDGETS[model]['scale']},
            'metax_minus_nvidia_pp':metric_deltas[primary], 'all_metric_deltas_pp':metric_deltas,
            'matched_conditions':{key:m[key] for key in ('data_sha256','initial_file_sha256','initial_state_sha256',
                'algorithm_config_sha256','driver_sha256','first_batch_sha256','all_batches_sha256')},
            'loss_comparison':loss_comparison(n['losses'],m['losses']),
            'final_model_states_bitwise_equal':n['final_state_sha256'] == m['final_state_sha256'],
            'nvidia_replicates':{v:{'report_sha256':expected['reference_report_sha256'][v],
                'final_metrics':r['final_metrics'],'final_state_sha256':r['final_state_sha256']} for v,r in nvidia.items()},
            'metax':{key:m[key] for key in ('final_metrics','initial_metrics','restored_metrics','environment','training_steps',
                'training_wall_seconds','peak_allocated_bytes','peak_reserved_bytes','checkpoint_sha256','final_state_sha256','epoch_metrics')},
            'metax_report_sha256':digest(directory/'report.json')})
    return {'records':rows,'new_complete_runs':2,'new_optimizer_updates':sum(r['metax']['training_steps'] for r in rows),
        'reference_complete_runs':8,'seed':protocol['seed'],'accuracy_threshold':None,
        'scope':'One C500 run per model vs frozen same-seed NVIDIA original/new repeats; fixed CIFAR-10/PennFudan budgets, not population equivalence or ImageNet/COCO convergence',
        'determinism':'Strict native switches enabled; not proof that every vendor kernel or full training is repeatable',
        'controller':controller, 'execution_jobs':execution['jobs']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metax','reference','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    value=summarize(args.metax,args.reference,json.loads(args.protocol.read_text()))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_bytes((json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
    with (args.output/'results.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(('model','metric','original_nvidia_percent','new_nvidia_percent','new_metax_percent','metax_minus_nvidia_pp'))
        for row in value['records']:
            writer.writerow((row['model'],row['primary_metric'],*[row['values_percent'][k] for k in ('original_nvidia','new_nvidia','new_metax')],row['metax_minus_nvidia_pp']))
    print(json.dumps([{'model':r['model'],'values_percent':r['values_percent'],'delta_pp':r['metax_minus_nvidia_pp']} for r in value['records']],indent=2))
