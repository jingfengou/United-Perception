"""Generate the GPU implementation/test map without importing vendor runtimes.

Implementation and UP references come from source. Hardware verdicts come only
from the reviewed evidence ledger; missing evidence is never a passing result.
"""
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path

from platform_contracts import ROOT, collect


LEDGER = ROOT / 'tests/platform/fixtures/gpu_verification.json'
DOCUMENT = ROOT / 'docs/platform/GPU_COVERAGE.md'
CPU_ONLY = {'detection.softer_nms', 'pointcloud.boxes_bev_iou_cpu',
            'pointcloud.points_in_boxes_cpu', 'sparse.voxelize_cpu'}
VERDICTS = {'passed': '通过', 'partial': '部分通过', 'unverified': '未验证'}


@lru_cache(None)
def tree(path):
    return ast.parse((ROOT / path).read_text(encoding='utf8'))


def definition(path, name, owner=None):
    nodes = tree(path).body
    if owner:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == owner).body
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
            return {'path': path, 'line': node.lineno}
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return {'path': path, 'line': node.lineno}
    raise ValueError(f'Missing source definition: {path}:{owner or ""}.{name}')


def has_body(site):
    node = next(n for n in ast.walk(tree(site['path'])) if getattr(n, 'lineno', None) == site['line']
                and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign)))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                           and isinstance(n.value.value, str))]
        return bool(body) and not all(isinstance(n, (ast.Pass, ast.Raise)) for n in body)
    return True


def implementation(name, vendor):
    category, method = name.split('.')
    prefix = vendor.capitalize()
    if category == 'device':
        binding = definition(f'up_platform/backends/{vendor}.py', prefix + 'Device')
        body = definition('up_platform/backends/torch_device.py', method, 'TorchDevice')
    elif category in ('training', 'inference'):
        binding = definition(f'up_platform/{category}/{vendor}.py', method, prefix + category.capitalize())
        body = definition(f'up_platform/{category}/torch_impl.py', method, 'Torch' + category.capitalize())
    else:
        path = f'up_platform/ops/{category}/{vendor}.py'
        owner = next(n.name for n in tree(path).body if isinstance(n, ast.ClassDef))
        binding = definition(path, method, owner)
        node = next(n for n in tree(path).body if isinstance(n, ast.ClassDef) and n.name == owner)
        entry = next(n for n in node.body if n.lineno == binding['line'])
        if isinstance(entry, ast.Assign):
            body = definition(f'up_platform/ops/{category}/torch_impl.py', method)
        else:
            body = binding
            # Native bindings contain a local import of the semantic operation.
            for imported in ast.walk(entry):
                if isinstance(imported, ast.ImportFrom) and imported.level == 1 and imported.module:
                    path = f'up_platform/ops/{category}/{imported.module}.py'
                    body = definition(path, imported.names[0].name)
                    break
    return {'binding': binding, 'body': body, 'implemented': has_body(binding) and has_body(body)}


def usage(names):
    """Resolve explicit calls plus local aliases and passed layer factories.

    This is source-reference coverage, not dynamic model coverage. Arbitrary
    getattr, user plugins and third-party internals are deliberately not inferred.
    """
    methods = defaultdict(set)
    for name in names:
        category, method = name.split('.')
        methods[category].add(method)
    layers = {}
    for node in tree('up_platform/layers.py').body:
        if isinstance(node, ast.FunctionDef):
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Call):
                    call = child.value
                    if (isinstance(call.func, ast.Attribute) and call.func.attr == 'category'
                            and call.args and isinstance(call.args[0], ast.Constant)):
                        target = str(call.args[0].value) + '.' + child.attr
                        if target in names:
                            layers[node.name] = target
    references = defaultdict(set)

    class References(ast.NodeVisitor):
        def __init__(self, path):
            self.path, self.aliases = path, {}

        def resolve(self, node):
            if isinstance(node, ast.Name):
                return self.aliases.get(node.id, set())
            if isinstance(node, ast.IfExp):
                return self.resolve(node.body) | self.resolve(node.orelse)
            if isinstance(node, ast.Call):
                if self.resolve(node.func) == {'@get_platform'}:
                    return {'@platform'}
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'category'
                        and self.resolve(node.func.value) == {'@platform'} and len(node.args) == 1
                        and isinstance(node.args[0], ast.Constant)):
                    return {'@' + str(node.args[0].value)}
            if isinstance(node, ast.Attribute):
                parents = self.resolve(node.value)
                if parents == {'@platform'} and node.attr in ('device', 'execution'):
                    return {'@' + node.attr}
                if parents == {'@layers'}:
                    return {layers[node.attr]} if node.attr in layers else set()
                result = set()
                for parent in parents:
                    categories = ('training', 'inference') if parent == '@execution' else (parent[1:],)
                    for category in categories:
                        name = category + '.' + node.attr
                        if name in names:
                            result.add(name)
                return result
            return set()

        def visit_ImportFrom(self, node):
            if node.module == 'up_platform':
                for alias in node.names:
                    token = {'get_platform': '@get_platform', 'layers': '@layers'}.get(alias.name)
                    if token:
                        self.aliases[alias.asname or alias.name] = {token}

        def visit_Assign(self, node):
            value = self.resolve(node.value)
            self.visit(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.aliases[target.id] = value
                elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(node.value, (ast.Tuple, ast.List)):
                    for left, right in zip(target.elts, node.value.elts):
                        if isinstance(left, ast.Name):
                            self.aliases[left.id] = self.resolve(right)

        def visit_Attribute(self, node):
            for name in self.resolve(node) & names:
                references[name].add((self.path, node.lineno))
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            previous = self.aliases
            self.aliases = dict(previous)
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                self.aliases.pop(arg.arg, None)
            self.generic_visit(node)
            self.aliases = previous

        visit_AsyncFunctionDef = visit_FunctionDef

    for path in sorted((ROOT / 'up').rglob('*.py')):
        relative = path.relative_to(ROOT).as_posix()
        References(relative).visit(tree(relative))
    return {name: [{'path': path, 'line': line} for path, line in sorted(values)]
            for name, values in references.items()}


def build(ledger=None):
    ledger = ledger or json.loads(LEDGER.read_text(encoding='utf8'))
    contracts = [item for item in collect() if item['name'] not in CPU_ONLY]
    names = {item['name'] for item in contracts}
    if set(ledger['interfaces']) != names:
        raise ValueError('GPU evidence ledger/interface set drift; review new and removed interfaces')
    uses = usage(names)
    rows = []
    for contract in contracts:
        name = contract['name']
        record = ledger['interfaces'][name]
        row = {'name': name, 'definition': {'path': contract['source'], 'line': contract['line']},
               'usage': uses.get(name, []), 'vendors': {}, 'tests': record['tests']}
        for test in row['tests']:
            definition(test['path'], test['symbol'])
        for vendor in ('nvidia', 'metax'):
            verdict = record[vendor]
            if verdict['status'] not in VERDICTS:
                raise ValueError('Unknown verdict: ' + name)
            if verdict['status'] != 'unverified' and not verdict['evidence']:
                raise ValueError('Passing/partial verdict requires hardware evidence: ' + name)
            for evidence in verdict['evidence']:
                reference = ledger['evidence'][evidence]
                if reference['vendor'] != vendor:
                    raise ValueError('Evidence belongs to another vendor: ' + name)
                for artifact in reference['artifacts']:
                    if len(artifact['sha256']) != 64:
                        raise ValueError('Missing evidence hash: ' + evidence)
            provider = implementation(name, vendor)
            if not provider['implemented'] and verdict['status'] != 'unverified':
                raise ValueError('Unimplemented interface cannot pass: ' + name)
            row['vendors'][vendor] = dict(verdict, **provider)
        rows.append(row)
    return rows, ledger


def link(site, label):
    return f'[{label}](../../{site["path"]}#L{site["line"]})'


def render(rows, ledger):
    counts = {v: Counter(row['vendors'][v]['status'] for row in rows) for v in ('nvidia', 'metax')}
    used = sum(bool(row['usage']) for row in rows)
    lines = ['# GPU 接口、厂商实现与评测代码对照', '',
             '本表由 `python tools/platform_coverage.py --write` 生成；接口和代码位置取自当前源码，'
             '真机状态取自已审核的[证据索引](../../tests/platform/fixtures/gpu_verification.json)。', '',
             f'共 **{len(rows)} 项 GPU 相关必需接口**，其中 **{used} 项在 UP 源码中有明确引用**。'
             '两家均有显式实现或绑定；“已实现”不等于“真机通过”。', '',
             '这些接口包括设备、AMP、通信和工具链入口，不是 GPU 内部全部计算内核。'
             '不计 4 项纯 CPU 接口；基础算术、索引、拼接、归约、随机数和 autograd 仍由厂商 PyTorch 提供。', '',
             '## 怎么看', '',
             '- **UP 使用**：显示静态引用总数和前两处可点击位置；包括调用、局部别名和传入工厂。'
             '“无明确引用”表示当前是预留接口，不代表某次模型执行过。',
             '- **已实现**：厂商文件有具体方法或显式绑定；点击“实现”可查看复用的代码。'
             '第三方库和编译扩展仍需按环境指南安装。',
             '- **通过**：所注明真机用例通过；数值、梯度、状态或流程按对应评测检查。'
             '浮点仍用 `rtol=1e-4, atol=1e-5`，整数/索引精确比较。',
             '- **部分通过**：保留已通过用例，同时有尚未通过的数值记录。'
             '**未验证**：没有可核实的逐接口真机通过记录，或相应依赖/策略尚不可执行。',
             '- NVIDIA、沐曦两列各自描述同平台验证；**不表示两家结果相同，也不表示模型最终精度相同**。'
             '只有新增测试并审核结果后才更新真机状态。', '',
             '## 当前统计', '']
    for vendor, label in [('nvidia', 'NVIDIA'), ('metax', '沐曦 C500')]:
        c = counts[vendor]
        lines.append(f'- **{label}**：通过 {c["passed"]} 项，部分通过 {c["partial"]} 项，未验证 {c["unverified"]} 项。')
    lines += ['', '## 逐接口对照', '',
              '同一行的评测链接列出已存在的用例位置；有评测代码但没有该厂商结果时，仍记“未验证”。'
              '每个状态后的编号对应下方证据来源。NVIDIA 专用执行器标有“（NVIDIA）”；其通用输入可复用，不能当作沐曦已经执行。', '',
              '| GPU 接口（定义） | UP 是否使用、引用位置 | NVIDIA 适配代码 | 沐曦适配代码 | 对应评测代码 / 用例 | NVIDIA 真机 | 沐曦真机 |',
              '| --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        uses = row['usage']
        use = '无明确引用（预留）'
        if uses:
            items = [link(site, site['path'].removeprefix('up/') + ':' + str(site['line'])) for site in uses[:2]]
            use = f'有 · {len(uses)} 处<br>' + '<br>'.join(items)
        codes, statuses = [], []
        for vendor in ('nvidia', 'metax'):
            record = row['vendors'][vendor]
            code = ('已实现' if record['implemented'] else '未实现') + ' · ' + link(record['binding'], '厂商入口')
            if record['binding'] != record['body']:
                code += ' / ' + link(record['body'], '实现')
            codes.append(code)
            sources = ' '.join(f'[{key}](#e-{key.lower()})' for key in record['evidence'])
            statuses.append(VERDICTS[record['status']] + (' · ' + sources if sources else '') + '<br>' + record['detail'])
        tests = '<br>'.join(link(definition(test['path'], test['symbol']), test['label']) for test in row['tests']) or '无对应用例'
        lines.append('| ' + ' | '.join([link(row['definition'], '`' + row['name'] + '`'), use, *codes, tests, *statuses]) + ' |')
    lines += ['', '## 需要特别注意的结果', '',
              '- **Conv2d**：小样本逐接口对照通过；真实批次中，NVIDIA 2/15、沐曦 5/15 样本的权重梯度超差，'
              '各自原生重复也有超差。前向均匹配各自直接 Torch。故本表两家均记“部分通过”，不把它直接归因为平台路由。',
              '- **KITTI `rotate_iou_eval`**：新版修正原版旋转框几何边界错误，两家均通过独立几何参考和解析用例。'
              '相同框 IoU 修正为 1；这是评估器修正，不是与原版保持相同错误数值。',
              '- **跨厂商**：98 个真实模型算子样本中有 19 个训练样本超原容差，64 个 eval/独立推理样本在原容差内。'
              '这些是跨厂商数值结果，与表中各家“自身适配是否通过”分开理解；随机掩码差异也单独记录。',
              '- **FSDP2**：两家均完成双卡 FP32/AMP、分片、EMA、验证、溢出同步、保存恢复和新进程推理的短程验证。'
              '它是新增训练能力，没有原版 UP 的 FSDP2 对照。', '',
              '## 证据来源', '',
              '整理日期：2026-10-27。复用 2026-10-23 至 26 日的已完成真机记录，本次未新增 GPU 训练。'
              '原始日志、张量及失败阶段保存在项目证据归档，不放入代码仓库；索引保留报告路径、SHA256 和检查范围。', '']
    for key, evidence in ledger['evidence'].items():
        lines += [f'<a id="e-{key.lower()}"></a>', '',
                  f'**{key} · {evidence["label"]}** — {evidence["hardware"]}；`{evidence["torch"]}`。{evidence["scope"]}', '']
    lines += ['## 更新与检查', '', '```bash', 'python tools/platform_coverage.py --write',
              'python tools/platform_coverage.py', 'python -m unittest discover -s tests/platform -p test_platform_coverage.py -v',
              '```', '', '增加/删除接口、改变代码位置或 UP 引用后重新生成。真机状态不能由注册、导入或 CPU 测试自动提升；'
              '应先审核真实结果，再更新证据索引。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    rows, ledger = build()
    expected = render(rows, ledger)
    if args.write:
        DOCUMENT.write_text(expected, encoding='utf8', newline='\n')
    elif not DOCUMENT.exists() or DOCUMENT.read_text(encoding='utf8') != expected:
        raise SystemExit('GPU coverage map drift; run --write and review')
    print(json.dumps({'gpu_interfaces': len(rows), 'used_by_up': sum(bool(r['usage']) for r in rows),
                      'verdicts': {v: dict(Counter(r['vendors'][v]['status'] for r in rows))
                                   for v in ('nvidia', 'metax')}}))


if __name__ == '__main__':
    main()
