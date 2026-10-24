"""Audit completed repeats and package reports, retaining large evidence in place."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import torch

from accuracy_data import digest, verify
from run_platform_validation import source_manifest
from summarize_native_repeat import summarize, VARIANTS


def write(path, value):
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def tree_hash(value):
    def describe(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            return ['array', str(item.dtype), list(item.shape), hashlib.sha256(item.tobytes()).hexdigest()]
        if isinstance(item, dict):
            return ['dict', [[repr(key), describe(child)] for key, child in sorted(item.items(), key=lambda pair: repr(pair[0]))]]
        if isinstance(item, (list, tuple)):
            return [type(item).__name__, [describe(child) for child in item]]
        if isinstance(item, np.generic):
            return [str(item.dtype), item.item()]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        raise TypeError(type(item))
    return hashlib.sha256(json.dumps(describe(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'base', 'output', 'cpu-result'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--native-reference', type=Path)
    args = parser.parse_args()
    root, base, output = args.root.resolve(), args.base.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    summary = summarize(root, deterministic=args.deterministic, native_reference=args.native_reference)
    plan = json.loads((base/'plan.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    assert all(digest(root/name) == sha for name, sha in protocol['tools'].items())
    sources = {name: source_manifest(base/folder) for name, folder in (('original', 'original'), ('new', 'source-03'))}
    assert sources['original']['sha256'] == plan['original_source_sha256']
    assert sources['new']['sha256'] == plan['new_source_sha256']
    assert digest(base/'data/data.json') == plan['data_sha256']
    data_verification = verify(json.loads((base/'data/data.json').read_text()))
    cpu = ET.parse(args.cpu_result).getroot().find('testsuite').attrib
    assert int(cpu['tests']) > 0 and cpu['failures'] == cpu['errors'] == cpu['skipped'] == '0'
    binaries = {str(p): digest(p) for directory in (base/'source-03/up_platform/backends', base/'original/up/extensions')
                for p in directory.glob('*.so')}
    assert set(binaries.values()) == {plan['native_binaries'][key] for key in ('new_nvidia', 'original_nvidia')}
    checkpoint_states = {}
    for model in ('yolov5s', 'vit_b16'):
        records = {}
        for variant in VARIANTS:
            directory = root/'full'/model/variant/model
            report = json.loads((directory/'report.json').read_text())
            assert digest(directory/'final.pt') == report['checkpoint_sha256']
            assert digest(base/'initial'/model/'initial.pt') == report['initial_file_sha256']
            # These are this experiment's own hash-checked files, not arbitrary downloads.
            checkpoint = torch.load(directory/'final.pt', map_location='cpu', weights_only=False)
            records[variant] = {key: tree_hash(value) for key, value in checkpoint.items()}
            del checkpoint
        checkpoint_states[model] = {'hashes': records, 'all_four_exact':
            {key: len({value[key] for value in records.values()}) == 1 for key in ('model', 'optimizer', 'scheduler')}}
    output.mkdir()
    items = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if (not path.is_file() or path.is_symlink() or path.is_relative_to(output)
                or any(part in ('__pycache__', '.pytest_cache', 'build') for part in relative.parts)):
            continue
        items.append({'path': relative.as_posix(), 'bytes': path.stat().st_size, 'sha256': digest(path)})
    index = {'raw_root': str(root), 'reference_root': str(base), 'created_utc': datetime.now(timezone.utc).isoformat(),
        'sources': sources, 'binaries': binaries, 'data_verification': data_verification, 'artifacts': items,
        'cpu': cpu, 'python': sys.version, 'packages': {d.metadata['Name']: d.version for d in metadata.distributions()},
        'analysis_tools': {name: digest(Path(__file__).with_name(name)) for name in ('package_native_repeat.py', 'summarize_native_repeat.py')}}
    write(output/'index.json', index)
    write(output/'summary.json', summary)
    write(output/'checkpoint-states.json', checkpoint_states)
    (output/'runs').mkdir()
    for path in sorted((root/'full').rglob('report.json')):
        report = json.loads(path.read_text())
        compact = {key: value for key, value in report.items() if key not in ('source', 'losses', 'learning_rates', 'epoch_metrics')}
        compact.update(source_sha256=report['source']['sha256'], raw_report_sha256=digest(path), raw_index_sha256=digest(output/'index.json'))
        write(output/'runs'/(report['model']+'-'+path.parents[1].name+'.json'), compact)
    archive = output/'reports.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for item in items:
            path = root/item['path']
            if path.suffix in ('.json', '.txt', '.xml', '.yaml', '.py'):
                bundle.add(path, arcname=item['path'], recursive=False)
        if args.native_reference is not None:
            for path in [args.native_reference/'controller.json', *sorted((args.native_reference/'full').rglob('report.json'))]:
                bundle.add(path, arcname='native-reference/'+path.relative_to(args.native_reference).as_posix(), recursive=False)
    receipt = {'complete_runs': 8, 'optimizer_updates': 14960, 'raw_files': len(items),
        'raw_bytes': sum(item['bytes'] for item in items), 'index_sha256': digest(output/'index.json'),
        'summary_sha256': digest(output/'summary.json'), 'checkpoint_states_sha256': digest(output/'checkpoint-states.json'),
        'reports_archive_sha256': digest(archive), 'reports_archive_bytes': archive.stat().st_size,
        'raw_root': str(root), 'cpu': cpu, 'large_artifacts': 'Retained in the original experiment directory; not included in reports.tar.gz'}
    write(output/'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
