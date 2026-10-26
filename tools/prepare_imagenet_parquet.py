"""Materialize an existing ImageNet parquet copy without changing image bytes.

Creates ordinary image files and zero-based UP classification manifests in a
separate directory. Shared source data is read-only. Readiness requires the
complete 1,281,167/50,000 split counts; an interrupted shard can be resumed by
verifying its already-written image bytes. No model/GPU package is imported.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


EXPECTED = {'train': 1281167, 'validation': 50000}


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def materialize_rows(rows, output, split, part, manifest, *, expected_rows):
    """Keep encoded bytes and source labels; reject corruption instead of overwriting."""
    folder = output / 'images' / split / part
    folder.mkdir(parents=True, exist_ok=True)
    labels = [0] * 1000
    count = 0
    with manifest.open('x', encoding='utf8', newline='\n') as records:
        for count, row in enumerate(rows, 1):
            label = row['label']
            if type(label) is not int or not 0 <= label < 1000:
                raise ValueError('ImageNet requires original integer labels in [0,999]')
            raw = row['image']['bytes']
            if not isinstance(raw, bytes) or not raw:
                raise ValueError('Embedded nonempty image bytes required; no external path fallback')
            suffix = '.JPEG' if raw.startswith(b'\xff\xd8') else '.png' if raw.startswith(b'\x89PNG') else '.img'
            path = folder / ('%07d' % count + suffix)
            checksum = hashlib.sha256(raw).hexdigest()
            if path.exists():
                if digest(path) != checksum:
                    raise ValueError('Previously written image changed: ' + str(path))
            else:
                with path.open('xb') as stream:
                    stream.write(raw)
            labels[label] += 1
            records.write(json.dumps({'path': path.relative_to(output / 'images').as_posix(),
                'label': label, 'bytes': len(raw), 'sha256': checksum,
                'source_image_path': row['image'].get('path')}, sort_keys=True) + '\n')
        records.flush()
        os.fsync(records.fileno())
    if count != expected_rows:
        raise ValueError('Truncated or unexpected parquet rows: %s != %s' % (count, expected_rows))
    return {'images': count, 'class_counts': labels, 'manifest_sha256': digest(manifest)}


def prepare(source, output, *, resume=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(2)
    source = source.resolve()
    output = output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Write a separate output directory, outside the shared source')
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    if (output / 'READY.json').exists():
        raise FileExistsError('Already complete; use the committed dataset, do not rematerialize')
    sources = {split: sorted((source / 'data').glob(split + '-*.parquet')) for split in EXPECTED}
    # Validate complete split counts before creating image files.
    counts = {split: sum(pq.ParquetFile(path).metadata.num_rows for path in paths)
              for split, paths in sources.items()}
    if counts != EXPECTED:
        raise ValueError('Not a complete ImageNet-1K source: ' + repr(counts))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid())
    attempt = output / 'attempts' / stamp
    attempt.mkdir(parents=True)
    started = time.monotonic()
    totals = {}
    for split, files in sources.items():
        split_records, histogram = [], [0] * 1000
        for index, path in enumerate(files):
            marker = output / 'shards' / split / (path.stem + '.json')
            manifest = output / 'shards' / split / (path.stem + '.jsonl')
            info = path.stat()
            if marker.exists():
                saved = json.loads(marker.read_text())
                if (saved['source_bytes'], saved['source_mtime_ns']) != (info.st_size, info.st_mtime_ns):
                    raise ValueError('Source shard identity changed: ' + str(path))
                if saved['manifest_sha256'] != digest(manifest):
                    raise ValueError('Completed shard manifest changed')
            else:
                parquet = pq.ParquetFile(path)
                rows = (row for batch in parquet.iter_batches(batch_size=64, columns=['image', 'label'], use_threads=False)
                        for row in batch.to_pylist())
                draft = attempt / (split + '-' + path.stem + '.jsonl')
                saved = materialize_rows(rows, output, split, path.stem, draft,
                                         expected_rows=parquet.metadata.num_rows)
                saved.update(source_name=path.name, source_bytes=info.st_size, source_mtime_ns=info.st_mtime_ns,
                             source_sha256=digest(path))
                manifest.parent.mkdir(parents=True, exist_ok=True)
                with manifest.open('xb') as destination, draft.open('rb') as stream:
                    shutil.copyfileobj(stream, destination)
                write_json(marker, saved)
            split_records.append((manifest, saved))
            histogram = [left + right for left, right in zip(histogram, saved['class_counts'])]
            print(json.dumps({'split': split, 'completed_shards': index + 1, 'total_shards': len(files),
                              'images': sum(s['images'] for _, s in split_records),
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
        if sum(histogram) != EXPECTED[split] or any(count == 0 for count in histogram):
            raise ValueError('Incomplete split or absent ImageNet class')
        meta = output / ('train.txt' if split == 'train' else 'val.txt')
        draft_meta = attempt / meta.name
        with draft_meta.open('x', encoding='utf8', newline='\n') as stream:
            for manifest, saved in split_records:
                for line in manifest.read_text().splitlines():
                    row = json.loads(line)
                    stream.write(row['path'] + ' ' + str(row['label']) + '\n')
        if meta.exists():
            if digest(meta) != digest(draft_meta):
                raise ValueError('Existing final manifest changed')
        else:
            with meta.open('xb') as stream, draft_meta.open('rb') as original:
                shutil.copyfileobj(original, stream)
        totals[split] = {'count': sum(histogram), 'class_counts': histogram,
                         'meta': meta.name, 'meta_sha256': digest(meta),
                         'shards': [{'name': path.name, 'manifest_sha256': saved['manifest_sha256'],
                                     'source_sha256': saved['source_sha256']} for path, saved in split_records]}
    ready = {'dataset': 'ImageNet-1K', 'image_bytes_unchanged': True, 'label_base': 0,
             'classes': 1000, 'image_root': 'images', 'splits': totals,
             'source': str(source), 'pyarrow': pa.__version__,
             'tool_sha256': digest(Path(__file__)), 'created_utc': datetime.now(timezone.utc).isoformat()}
    write_json(output / 'READY.json', ready)
    return ready


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.output, resume=args.resume)
