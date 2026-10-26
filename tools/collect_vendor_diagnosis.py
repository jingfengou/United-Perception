"""Package small diagnostic reports and hash large retained tensor evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--vendor', choices=('nvidia', 'metax'), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    labels = ('prefix', 'replay', 'writes', 'writes02', 'write-replay', 'yolo-deep', 'transfer', 'first-divergence')
    jobs = {label: json.loads((a.root / (label + '-job.json')).read_text()) for label in labels}
    if any(job['status'] != 'completed' or job['exit_code'] != 0 for job in jobs.values()):
        raise ValueError('A required diagnostic job is incomplete')
    folders = [a.root / name for name in ('replay', 'writes', 'writes02', 'writes-replay', 'yolo-deep', 'first-divergence',
               'vit_b16-observed', 'vit_b16-control', 'yolov5s-observed', 'yolov5s-control')]
    if a.vendor == 'nvidia':
        folders += [a.root / name for name in ('pilot', 'vit_b16-original', 'yolov5s-original')]
    else:
        folders += [a.root / name for name in ('pilot', 'pilot02')]
    paths = {path for folder in folders for path in folder.rglob('*')
             if path.is_file() and path.suffix in ('.json', '.pt', '.log')}
    paths.update(path for path in a.root.iterdir() if path.is_file() and path.suffix in ('.json', '.log', '.xml'))
    paths.update(a.root.glob('*.pt'))
    paths.update(path for path in (a.root / 'tools').glob('*vendor*.py'))
    paths.update(a.root / 'tools' / name for name in ('accuracy_train.py', 'accuracy_recipe.py',
                  'accuracy_data.py', 'run_platform_validation.py'))
    paths.add(a.root / 'tests/platform/test_vendor_diagnosis.py')
    paths.update((a.root / 'verification-current/tools').glob('*vendor*.py'))
    paths.add(a.root / 'verification-current/tests/platform/test_vendor_diagnosis.py')
    # Include exact executed helpers, without collecting dependency trees/archives.
    paths.update(path for path in a.root.glob('*.py'))
    entries = {}
    for path in sorted(paths):
        relative = path.relative_to(a.root)
        entries[relative.as_posix()] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        if path.suffix != '.pt':
            destination = a.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    index = {'vendor': a.vendor, 'root': str(a.root), 'created_utc': datetime.now(timezone.utc).isoformat(),
             'scope': 'Raw tensor evidence retained on the assigned host; small reports copied',
             'jobs': jobs, 'files': entries, 'large_files_publicly_downloadable': False}
    (a.output / 'evidence-index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    with tarfile.open(a.output.with_suffix('.tar.gz'), 'x:gz') as bundle:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(a.output))
    print(json.dumps({'vendor': a.vendor, 'indexed_files': len(entries),
                      'tensor_bytes': sum(row['bytes'] for name, row in entries.items() if name.endswith('.pt'))}))


if __name__ == '__main__':
    main()
