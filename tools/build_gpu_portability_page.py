"""Build the offline vendor checklist from the reproducible source-audit JSON."""
import argparse
import collections
import csv
import hashlib
import json
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--software-tests', type=int, required=True,
                        help='Actual passing count from this source stage; not a GPU pass count.')
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    data = {key: audit[key] for key in ('revisions', 'generated', 'summary', 'sources', 'native_sources_identical')}
    data['families'] = [{k: v for k, v in f.items() if k not in ('sites', 'consumers')}
                        for f in audit['families']]
    data['functions'] = []
    for function in audit['function_inventory']:
        item = {k: v for k, v in function.items() if k != 'sites'}
        item['sites'] = {v: [dict(line=r['line'], api=r['api'], scope=r['scope']) for r in rows]
                         for v, rows in function['sites'].items()}
        data['functions'].append(item)
    data['software_tests'] = args.software_tests
    data['assignment_counts'] = {v: dict(collections.Counter(f['vendors'][v]['action']
                                  for f in audit['families'])) for v in ('metax', 'cambricon', 'ascend')}
    data['examples'] = {
        'yaml': 'runtime:\n  platform: metax_cuda\n\n# nvidia / metax_cuda / cambricon_mlu / ascend_npu',
        'install': '# 保留环境中的厂商 Torch，先装应用依赖\npython -m pip install -r requirements/runtime.txt\n\n'
                   '# 非 NVIDIA 环境不编译 CUDA 扩展\n'
                   'UP_BUILD_PLATFORM=metax_cuda python -m pip install --no-deps --no-build-isolation -e .\n\n'
                   '# NVIDIA 环境保留原加速扩展；需匹配的 CUDA 编译器\n'
                   'python setup.py build_ext -i\npython -m pip install --no-deps --no-build-isolation -e .',
        'commands': 'python -m up train --launch pytorch --ng 1 \\\n'
                    '  --config configs/det/yolox/yolox_tiny.yaml \\\n'
                    '  --opts runtime.platform metax_cuda\n\n'
                    'python -m up train --evaluate --launch pytorch --ng 1 \\\n'
                    '  --config configs/det/yolox/yolox_tiny.yaml \\\n'
                    '  --opts runtime.platform metax_cuda\n\n'
                    'python -m up inference --launch pytorch --ng 1 \\\n'
                    '  --config configs/det/yolox/yolox_tiny.yaml \\\n'
                    '  -i /data/images -c /data/model.pth -v /data/predictions \\\n'
                    '  --opts runtime.platform metax_cuda',
    }
    shell = (ROOT / 'tools/templates/gpu_portability.html').read_text(encoding='utf-8')
    assert shell.count('__AUDIT_DATA__') == 1
    embedded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(shell.replace('__AUDIT_DATA__', embedded), encoding='utf-8', newline='\n')
    worklist = args.output.with_name('vendor-worklist.csv')
    with worklist.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'category', 'interface', 'scope', 'contract', 'vendor', 'action', 'task', 'pending', 'hardware'])
        for family in audit['families']:
            for vendor, state in family['vendors'].items():
                writer.writerow([family['id'], family['title'], family['interface'], family['scope'],
                    family['contract'], vendor, state['action'], state['text'], state['pending'], family['hardware']])
    print(json.dumps(dict(html=str(args.output), bytes=args.output.stat().st_size,
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        functions=len(data['functions']), work_packages=len(data['families']),
        software_tests=args.software_tests), ensure_ascii=False))


if __name__ == '__main__':
    main()
