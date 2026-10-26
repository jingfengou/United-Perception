"""Summarize actual finite interface fixtures without inventing unmeasured zeros."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten

from platform_operator_cases import ATOL, RTOL, compare_results


def nonfinite_agreement(left, right):
    """Report matching nonfinite boundaries, never mark them a finite-value pass."""
    aa, sa = tree_flatten(left)
    bb, sb = tree_flatten(right)
    count = 0
    if sa != sb:
        return None
    for a, b in zip(aa, bb):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                return None
            if a.is_floating_point():
                finite_a, finite_b = torch.isfinite(a), torch.isfinite(b)
                if not torch.equal(finite_a, finite_b) or not torch.equal(torch.isnan(a), torch.isnan(b)):
                    return None
                if not torch.equal(torch.isposinf(a), torch.isposinf(b)) or not torch.equal(torch.isneginf(a), torch.isneginf(b)):
                    return None
                if not torch.allclose(a[finite_a], b[finite_b], rtol=RTOL, atol=ATOL):
                    return None
                count += int((~finite_a).sum())
            elif not torch.equal(a, b):
                return None
        elif type(a) is not type(b) or a != b:
            return None
    return {'matching_nonfinite_elements': count, 'finite_elements_within_original_tolerance': True} if count else None


def interface_status(statuses):
    if statuses['difference'] or statuses['execution_error']:
        return 'unresolved_difference_or_execution'
    if statuses['within_tolerance']:
        return 'measured_within_tolerance'
    if statuses['intentional_geometry_correction']:
        return 'intentional_geometry_correction'
    if statuses['matching_nonfinite']:
        return 'matching_nonfinite_state'
    if statuses['blocked_dependency']:
        return 'blocked_dependency'
    return 'not_executed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--interfaces', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, sources = [], {}
    def read(path):
        path = args.evidence_root/path
        sources[str(path.relative_to(args.evidence_root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def core(stage, folder):
        base = Path(stage)/folder
        report = read(base/'summary.json')
        values = torch.load(args.evidence_root/base/'values.pt', weights_only=False, map_location='cpu')
        for entry in report['records']:
            row = dict(entry, evidence=str(base/'summary.json'), raw_status=entry['status'])
            pair = values.get(entry['interface']+'/'+entry['case'], {})
            if entry['status'] == 'difference' and 'reference' in pair and 'routed' in pair:
                agreement = nonfinite_agreement(pair['reference'], pair['routed'])
                if agreement:
                    row.update(status='matching_nonfinite', nonfinite=agreement)
            if entry['status'] == 'execution_error' and len(entry.get('errors', {})) == 4:
                errors = entry['errors'].values()
                if len({(value['type'], value['message']) for value in errors}) == 1:
                    row['status'] = 'matching_error'
            rows.append(row)
    for folder in ('basic-training', 'basic-inference', 'training-training', 'inference-inference'):
        core('interface-audit-02', folder)
    core('interface-audit-03', 'device-training')
    for folder in ('additional-training', 'additional-inference'):
        core('interface-audit-04', folder)
    native = read(Path('interface-audit-03/native-comparison.json'))
    rows.extend(dict(row, evidence='interface-audit-03/native-comparison.json') for row in native['records'])
    for rank in (0, 1):
        core('interface-audit-06', f'distributed/rank{rank}')
    optional = read(Path('interface-audit-03/optional-comparison.json'))
    for entry in optional['records']:
        if entry['interface'].startswith('optional.'):
            continue
        row = dict(entry, evidence='interface-audit-03/optional-comparison.json', raw_status=entry['status'])
        if entry['status'] == 'difference':
            case, name = entry['case'].split('/', 1)
            values = [torch.load(args.evidence_root/'interface-audit-03/optional'/(case+'-'+route)/'values.pt',
                                 weights_only=False, map_location='cpu')[name] for route in ('direct', 'routed')]
            agreement = nonfinite_agreement(*values)
            if agreement:
                row.update(status='matching_nonfinite', nonfinite=agreement)
        rows.append(row)
    # Corrected fixture for the available DefaultFakeSparse export policy.
    for route in ('direct', 'routed'):
        report = read(Path(f'interface-audit-07/optional/base_export-{route}/summary.json'))
        if report['errors']:
            raise ValueError('Export fixture did not execute: '+str(report['errors']))
    exports = [torch.load(args.evidence_root/'interface-audit-07/optional'/('base_export-'+route)/'values.pt',
                          weights_only=False, map_location='cpu')['pruning.export'] for route in ('direct', 'routed')]
    comparison = compare_results(*exports)
    rows.append({'interface': 'pruning.export', 'case': 'BaseScheduler_DefaultFakeSparse',
                 'status': 'within_tolerance' if comparison['passed'] else 'difference',
                 'comparison': comparison, 'reference_kind': 'direct_dependency_API',
                 'evidence': 'interface-audit-07/optional/base_export-{direct,routed}/summary.json'})
    kitti = read(Path('interface-audit-03/kitti/summary.json'))
    assert all(record['new_within_original_tolerance'] for record in kitti['records'])
    rows.append({'interface': kitti['interface'], 'case': 'four_criteria_7100_pairs',
                 'status': 'intentional_geometry_correction', 'evidence': 'interface-audit-03/kitti/summary.json',
                 'records': kitti['records'], 'examples': kitti['examples']})
    # These interfaces require the original Amba policy, which the actual SDK
    # attempts rejected. Ampere's inherited no-op methods are not substitutes.
    amba = [read(Path(f'interface-audit-03/optional/amba-{route}/summary.json')) for route in ('direct', 'routed')]
    assert all(value['errors'] for value in amba)
    for method in ('prune', 'last_prune_step', 'set_lr_scheduler'):
        rows.append({'interface': 'pruning.'+method, 'case': 'UP_Amba_policy', 'status': 'blocked_dependency',
                     'reason': 'Installed MSBench lacks the original FakeSparse policy; no replacement or no-op used',
                     'evidence': 'interface-audit-03/optional/amba-{direct,routed}/summary.json'})
    contracts = json.loads(args.interfaces.read_text())
    interfaces = []
    known = {value['name'] for value in contracts}
    assert len(known) == 156
    assert not {row['interface'] for row in rows}-known
    for contract in contracts:
        cases = [row for row in rows if row['interface'] == contract['name']]
        statuses = collections.Counter(row['status'] for row in cases)
        status = interface_status(statuses)
        maxima = [leaf.get('max_abs', leaf.get('abs_max')) for row in cases
                  for leaf in row.get('comparison', {}).get('leaves', row.get('comparison', {}).get('comparisons', []))]
        maxima = [value for value in maxima if value is not None]
        interfaces.append({'interface': contract['name'], 'status': status, 'case_count': len(cases),
                           'case_statuses': dict(statuses), 'max_abs_compared_numeric_fields': max(maxima, default=None),
                           'evidence': sorted({row['evidence'] for row in cases})})
    result = {'scope': 'Finite NVIDIA same-platform fixtures; independent original native binaries and declared Torch/dependency APIs',
              'rtol': RTOL, 'atol': ATOL, 'interface_count': len(interfaces),
              'interface_status_counts': dict(collections.Counter(row['status'] for row in interfaces)),
              'case_status_counts': dict(collections.Counter(row['status'] for row in rows)),
              'interfaces': interfaces, 'cases': rows, 'source_sha256': sources}
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    with (args.output/'interfaces.csv').open('w', newline='') as handle:
        fields = ['interface', 'status', 'case_count', 'max_abs_compared_numeric_fields']
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in interfaces)
    print(json.dumps({key: result[key] for key in ('interface_count', 'interface_status_counts', 'case_status_counts')}))


if __name__ == '__main__':
    main()
