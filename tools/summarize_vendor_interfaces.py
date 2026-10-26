"""Review one vendor's interface audit, retaining numeric failures and rejections."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from platform_coverage import CPU_ONLY
from platform_operator_cases import ATOL, RTOL, compare_results
from summarize_interface_audit import interface_status, nonfinite_agreement


def aggregate(rows, names):
    unknown = {r['interface'] for r in rows} - set(names)
    if unknown:
        raise ValueError('Unknown interfaces: '+str(sorted(unknown)))
    result = []
    for name in sorted(names):
        cases = [r for r in rows if r['interface'] == name]
        statuses = Counter(r['status'] for r in cases)
        maxima = [leaf['max_abs'] for row in cases for leaf in row.get('comparison', {}).get('leaves', [])
                  if leaf.get('max_abs') is not None]
        result.append({'interface': name, 'status': interface_status(statuses), 'case_count': len(cases),
                       'case_statuses': dict(statuses), 'max_abs': max(maxima, default=None),
                       'repeat_difference_cases': sum(bool(r.get('repeat_difference')) for r in cases),
                       'evidence': sorted({r['evidence'] for r in cases})})
    return result


def review(results, backend, retries=()):
    results = Path(results).resolve()
    retries = [Path(root).resolve() for root in retries]
    if any(root.parent != results.parent for root in retries):
        raise ValueError('Retry results must be separate attempts in the same task directory')
    evidence_root = results.parent
    hashes, rows, pipeline_errors = {}, [], []
    def read(path):
        data = path.read_bytes()
        hashes[str(path.relative_to(evidence_root))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)
    def tensors(path):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        hashes[str(path.relative_to(evidence_root))] = digest
        receipt = path.parent/'values-sha256.json'
        if receipt.exists() and read(receipt)['sha256'] != digest:
            raise ValueError('Tensor artifact hash mismatch: '+str(path))
        return torch.load(path, weights_only=False, map_location='cpu')

    expected = {'cpu-contracts', 'basic-training', 'basic-inference', 'additional-training',
                'additional-inference', 'device', 'training', 'inference', 'distributed', 'optional-compare'}
    expected |= {case+'-'+route for case in ('sparse', 'quantization', 'qdrop', 'pruning', 'base_export', 'amba')
                 for route in ('direct', 'routed')}
    selected, history = {}, []
    for index, root in enumerate([results, *retries]):
        job = read(root/'job.json')
        if job.get('backend') != backend or job.get('status') != 'executed_awaiting_numeric_review':
            raise ValueError('Vendor job has not finished all scheduled work')
        records = {r['name']: r for r in job['records']}
        if len(records) != len(job['records']) or set(records)-expected or (not index and set(records) != expected):
            raise ValueError('Missing, duplicate or unexpected audit processes')
        if index and not job.get('retry_reasons'):
            raise ValueError('An explicit reason is required for a retry attempt')
        history.append({'attempt': root.name, 'records': job['records'], 'retry_reasons': job.get('retry_reasons', {})})
        selected.update({name: (root, record) for name, record in records.items()})
    if set(selected) != expected or any(record['exit'] for _, record in selected.values()):
        raise ValueError('Selected audit processes are incomplete or failed')

    folders = ['basic-training', 'basic-inference', 'additional-training', 'additional-inference',
               'device', 'training', 'inference', 'distributed/rank0', 'distributed/rank1']
    for folder in folders:
        root = selected[folder.split('/')[0]][0]/folder
        report = read(root/'summary.json')
        if report.get('backend') != backend or (report['rtol'], report['atol']) != (RTOL, ATOL):
            raise ValueError('Wrong backend or tolerance in '+folder)
        values = tensors(root/'values.pt')
        for entry in report['records']:
            row = dict(entry, evidence=str((root/'summary.json').relative_to(evidence_root)), purpose=report['purpose'])
            pair = values.get(entry['interface']+'/'+entry['case'], {})
            repeats = []
            if 'comparison' in entry:
                for key, left, right in [('comparison', 'reference', 'routed'),
                                         ('reference_repeat', 'reference', 'reference_repeat'),
                                         ('routed_repeat', 'routed', 'routed_repeat')]:
                    actual = compare_results(pair[left], pair[right])
                    if actual != entry[key]:
                        raise ValueError('Saved comparison does not reproduce: '+entry['interface']+'/'+key)
                    if key != 'comparison':
                        repeats.append(not actual['passed'] and not nonfinite_agreement(pair[left], pair[right]))
                if entry['status'] == 'difference':
                    agreement = nonfinite_agreement(pair['reference'], pair['routed'])
                    if agreement:
                        row.update(status='matching_nonfinite', nonfinite=agreement)
            row['repeat_difference'] = any(repeats)
            rows.append(row)

    optional_reports = {}
    for case in ('sparse', 'quantization', 'qdrop', 'pruning', 'base_export', 'amba'):
        reports, values = {}, {}
        for route in ('direct', 'routed'):
            root = selected[case+'-'+route][0]/'optional'/(case+'-'+route)
            report = read(root/'summary.json')
            if report.get('backend') != backend:
                raise ValueError('Wrong optional backend: '+case+'/'+route)
            reports[route], values[route] = report, tensors(root/'values.pt')
        optional_reports[case] = reports
        for name in sorted(set(values['direct']) | set(values['routed'])):
            interface = name.split('/')[0]
            if interface in CPU_ONLY:
                continue
            row = {'interface': interface, 'case': case+'/'+name, 'purpose': 'training',
                   'evidence': '; '.join(str((selected[case+'-'+route][0]/'optional'/(case+'-'+route)/'summary.json')
                                            .relative_to(evidence_root)) for route in ('direct', 'routed')),
                   'reference_kind': 'declared_dependency_API_pipeline_snapshot'}
            if name not in values['direct'] or name not in values['routed']:
                row['status'] = 'execution_error'
            else:
                left, right = values['direct'][name], values['routed'][name]
                row['comparison'] = compare_results(left, right)
                row['status'] = 'within_tolerance' if row['comparison']['passed'] else 'difference'
                if row['status'] == 'difference':
                    agreement = nonfinite_agreement(left, right)
                    if agreement:
                        row.update(status='matching_nonfinite', nonfinite=agreement)
            if interface == 'sparse.dense' and backend == 'metax':
                row['reference_adapter'] = 'Native N...C output normalized to declared NC... layout'
            rows.append(row)
        gpu_errors = {route: {name: error for name, error in report['errors'].items() if name not in CPU_ONLY}
                      for route, report in reports.items()}
        if any(gpu_errors.values()):
            pipeline_errors.append({'case': case, 'errors': gpu_errors})
    amba = optional_reports['amba']
    if all(report['errors'] and 'FakeSparse' in json.dumps(report['errors']) for report in amba.values()):
        for name in ('prune', 'last_prune_step', 'set_lr_scheduler'):
            rows.append({'interface': 'pruning.'+name, 'case': 'UP_Amba_policy',
                         'status': 'blocked_dependency', 'reason': 'Installed MSBench lacks UP FakeSparse policy',
                         'evidence': '; '.join(str((selected['amba-'+route][0]/'optional'/('amba-'+route)/'summary.json')
                                                  .relative_to(evidence_root)) for route in ('direct', 'routed'))})
    return rows, hashes, pipeline_errors, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--backend', choices=('nvidia', 'metax'), required=True)
    parser.add_argument('--retry-results', action='append', type=Path, default=[],
                        help='Explicit corrected attempt; previous failures remain in execution_history')
    parser.add_argument('--interfaces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    contracts = json.loads(args.interfaces.read_text())
    names = {r['name'] for r in contracts} - CPU_ONLY
    rows, hashes, errors, history = review(args.results, args.backend, args.retry_results)
    interfaces = aggregate(rows, names)
    result = {'backend': args.backend, 'rtol': RTOL, 'atol': ATOL,
              'scope': 'Same-vendor direct API versus platform; native repeats recorded separately',
              'interfaces': interfaces, 'cases': rows, 'pipeline_errors': errors, 'artifact_sha256': hashes,
              'execution_history': history,
              'interface_status_counts': dict(Counter(r['status'] for r in interfaces)),
              'case_status_counts': dict(Counter(r['status'] for r in rows))}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('backend', 'interface_status_counts', 'case_status_counts')}))


if __name__ == '__main__':
    main()
