"""Generate concrete owner task sheets and interface stubs from NVIDIA source.

No runtime, model or kernel is changed. Proposed vendor paths are delivery paths,
not automatically registered implementations. The framework owner connects the
completed operation before the original Function.apply boundary.
"""
import ast
import copy
import csv
import hashlib
import json
import pathlib
import re

from vendor_api_spec import REFERENCE_COMMIT, VENDORS, OPS, DEVICE, TRAINING, COMMUNICATION

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/platform'
URL = 'https://github.com/jingfengou/United-Perception/blob/' + REFERENCE_COMMIT + '/'


def reference(path, symbol=None, language='python'):
    """Locate actual definitions, not similarly named calls or C++ declarations."""
    text = (ROOT / path).read_text(encoding='utf-8-sig')
    result = dict(path=path, symbol=symbol or pathlib.Path(path).name,
                  sha256=hashlib.sha256((ROOT / path).read_bytes()).hexdigest())
    if not symbol:
        result.update(line=1, url=URL + path + '#L1')
        return result
    if language == 'python':
        parts = symbol.split('.')
        nodes = ast.parse(text).body
        node = None
        for name in parts:
            node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name)
            nodes = node.body
        result['line'] = node.lineno
        if isinstance(node, ast.FunctionDef):
            result['signature'] = symbol + '(' + ast.unparse(node.args) + ')'
            result['_args'] = node.args
        result['snippet'] = '\n'.join(text.splitlines()[node.lineno-1:min(node.end_lineno, node.lineno+23)])
    else:
        for match in re.finditer(r'\b' + re.escape(symbol) + r'\s*\(', text):
            start = text.index('(', match.start())
            depth, end = 1, start + 1
            while depth:
                depth += (text[end] == '(') - (text[end] == ')')
                end += 1
            if not text[end:].lstrip().startswith('{'):
                continue
            result['line'] = text[:match.start()].count('\n') + 1
            result['signature'] = re.sub(r'\s+', ' ', text[match.start():end])
            result['snippet'] = '\n'.join(text.splitlines()[result['line']-1:result['line']+10])
            break
        else:
            raise ValueError('Missing C++ definition: {} {}'.format(path, symbol))
    result['url'] = URL + path + '#L' + str(result['line'])
    return result


def op_contract(spec):
    result = copy.deepcopy(spec)
    path = 'up/extensions/python/' + spec['file']
    is_function = '.' not in spec['symbol'] and not spec['symbol'].endswith('Function')
    forward = reference(path, spec['symbol'] if is_function else spec['symbol'] + '.forward')
    args = copy.deepcopy(forward.pop('_args'))
    if not is_function:
        assert args.args[0].arg in ('self', 'ctx')
        args.args.pop(0)
    result['public_signature'] = (spec['symbol'] + ('' if is_function else '.apply')) + '(' + ast.unparse(args) + ')'
    if spec.get('extra'):
        args.kwonlyargs.append(ast.arg(arg=spec['extra']))
        args.kw_defaults.append(None)
    result['vendor_signature'] = spec['id'] + '(' + ast.unparse(args) + ')'
    result['signature_args'] = ast.unparse(args)
    result['python'] = [forward]
    if not is_function:
        backward = reference(path, spec['symbol'] + '.backward')
        backward.pop('_args')
        result['python'].append(backward)
        tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == spec['symbol'])
        if any(isinstance(n, ast.FunctionDef) and n.name == 'symbolic' for n in cls.body):
            symbolic = reference(path, spec['symbol'] + '.symbolic')
            symbolic.pop('_args')
            result['python'].append(symbolic)
    result['cpp'] = [reference('up/extensions/csrc/' + path, name, 'cpp') for path, name in spec['cpp']]
    result['cuda'] = [reference('up/extensions/csrc/' + path, language='cpp') for path in spec['cuda']]
    binding_text = (ROOT / 'up/extensions/csrc/pybind.cpp').read_text(encoding='utf-8-sig')
    result['pybind'] = []
    for name in spec['bindings']:
        exposed = name.split('.')[-1]
        lines = binding_text.splitlines()
        hits = [i+1 for i, line in enumerate(lines) if '.def("' + exposed + '"' in line and not line.lstrip().startswith('//')]
        assert hits, 'Missing pybind export: ' + name
        # Same export name exists in different namespaces; pair it with the C++ symbol.
        expected = {r['symbol'] for r in result['cpp']}
        hits = [line for line in hits if any('&' + target in '\n'.join(lines[line-1:line+3]) for target in expected)]
        assert len(hits) == 1, 'Ambiguous pybind export: ' + name
        result['pybind'].append(dict(symbol='up.extensions._C.' + name, path='up/extensions/csrc/pybind.cpp',
            line=hits[0], url=URL+'up/extensions/csrc/pybind.cpp#L'+str(hits[0])))
    return result


def md_link(ref):
    return '[{}:{} · {}]({})'.format(ref['path'], ref['line'], ref['symbol'], ref['url'])


def service_rows():
    rows = []
    for signature, output, meaning in DEVICE:
        name = signature.split('(')[0]
        rows.append(dict(id='device.'+name, section='设备服务', signature=signature, output=output, meaning=meaning,
            nvidia='torch.cuda.'+name, reference='https://docs.pytorch.org/docs/2.10/cuda.html',
            target='RuntimeBackend.module；API 不兼容才在 up/platform/{vendor}/device.py 写适配器',
            action='现有映射已接入；复用厂商设备 API 并逐项验证'))
    for signature, output, nvidia, meaning in TRAINING:
        rows.append(dict(id='amp.'+signature.split('(')[0], section='AMP', signature=signature, output=output, meaning=meaning,
            nvidia=nvidia.replace('runtime.amp', 'torch.cuda.amp'), reference='https://docs.pytorch.org/docs/2.10/notes/amp_examples.html',
            target='RuntimeBackend.amp；工厂返回厂商 AMP，名字/行为不兼容才写适配器', action='复用并验证缩放、溢出跳步、状态恢复'))
    for signature, output, meaning in COMMUNICATION:
        name = signature.split('(')[0]
        nvidia = 'torch.distributed.' + name
        if name == 'DistributedDataParallel':
            nvidia = 'torch.nn.parallel.DistributedDataParallel'
        elif name == 'SyncBatchNorm.forward':
            nvidia = 'torch.nn.SyncBatchNorm.forward / torch.nn.modules._functions.SyncBatchNorm.apply'
        rows.append(dict(id='comm.'+name, section='通信/分布式', signature=signature, output=output, meaning=meaning,
            nvidia=nvidia, reference='https://docs.pytorch.org/docs/2.10/distributed.html',
            target='RuntimeBackend.dist_backend + up/utils/env/{launch,dist_helper}.py；SyncBN 见 up/utils/model/bn_helper.py',
            action='复用厂商通信插件；有缺口则在平台侧接入同语义实现，双卡验收'))
    rows.extend([
        dict(id='runtime.memory_available', section='平台策略',
             signature='memory_available(device, needed_memory, reserve_bytes=536870912)', output='bool',
             meaning='按 byte 比较 total_memory - memory_allocated - reserve_bytes；保留原 UP 的显存估计和比较规则。',
             nvidia='RuntimeBackend.memory_available → torch.cuda.get_device_properties / memory_allocated',
             reference=URL+'up/runtime/backend.py#L51', target='RuntimeBackend.memory_available；底层名称不兼容时写厂商 device.py 适配器',
             action='共享策略已实现，验证设备属性和 allocator 口径'),
        dict(id='runtime.configure_dnn', section='计算库策略',
             signature='configure_dnn(options=None, legacy_nocudnn=None)', output='None；未支持项抛 NotImplementedError',
             meaning='options 允许 enabled/benchmark/deterministic/allow_tf32 的 bool；只接真实有效的厂商控制项。None 保留当前默认；legacy_nocudnn 处理原 CLI。',
             nvidia='RuntimeBackend.configure_dnn → torch.backends.cudnn',
             reference=URL+'up/runtime/backend.py#L59', target='RuntimeBackend.configure_dnn；有厂商差异时由 device.py 提供策略实现',
             action='C500 核 mcDNN 兼容开关；MLU/NPU 只实现有效 CNNL/CANN 控制项，未支持的明确报错'),
    ])
    # Reference implementation is PyTorch for standard services, UP for policies.
    # A vendor with a working Torch backend reuses these APIs rather than porting
    # CUDA internals out of PyTorch into this repository.
    for row in rows:
        if row['id'].startswith('device.'):
            path = 'torch/cuda/memory.py' if row['id'].split('.')[1].startswith(('memory_', 'max_memory_', 'reset_peak_', 'empty_cache')) else 'torch/cuda/__init__.py'
        elif row['id'].startswith('amp.'):
            path = 'torch/amp/autocast_mode.py' if row['id'] == 'amp.autocast' else 'torch/amp/grad_scaler.py'
        elif row['id'] == 'comm.DistributedDataParallel':
            path = 'torch/nn/parallel/distributed.py'
        elif row['id'] == 'comm.SyncBatchNorm.forward':
            path = 'torch/nn/modules/batchnorm.py'
        elif row['id'].startswith('comm.'):
            path = 'torch/distributed/distributed_c10d.py'
        else:
            row['nvidia_source'] = row['reference']
            continue
        row['nvidia_source'] = 'https://github.com/pytorch/pytorch/blob/v2.10.0/' + path
    return rows


def make_spec():
    result = dict(reference_commit=REFERENCE_COMMIT, vendors=VENDORS, services=service_rows(),
        operations=[op_contract(op) for op in OPS],
        shared=[dict(signature='box_overlap(b1, b2, mode="IoU")', ref=reference('up/platform/ops.py', 'box_overlap')),
                dict(signature='nms(boxes, scores, threshold)', ref=reference('up/platform/ops.py', 'nms')),
                dict(signature='sigmoid_focal_loss(input, target, normalizer, gamma, alpha, num_channels, reduction)',
                     ref=reference('up/platform/ops.py', 'sigmoid_focal_loss'))])
    for shared in result['shared']:
        shared['ref'].pop('_args', None)
    return result


def template(spec):
    lines = ['"""Vendor delivery interface; not automatically loaded by UP.',
             '', 'Copy into up/platform/<vendor>/native_ops.py and implement the required methods.',
             'Each method returns the complete operation, including autograd where specified.',
             'The framework maintainer wires completed methods before the legacy Function.apply.',
             'See VENDOR_IMPLEMENTATION.md for NVIDIA references and contracts.', '"""', '', '',
             'class NativeOpsTemplate:', '    def __init__(self, platform):', '        self.platform = platform', '',
             '    def _missing(self, operation):',
             '        raise NotImplementedError("{}: {} is not implemented".format(self.platform, operation))', '']
    for op in spec['operations']:
        args = op['signature_args']
        lines += ['    def {}(self{}):'.format(op['id'], ', '+args if args else ''),
                  '        """{}\n'.format(op['title']),
                  '        输入: '+op['inputs'], '        输出: '+op['output'],
                  '        反向: '+op['backward'], '        '+op['details'],
                  '        NVIDIA: '+op['python'][0]['url'], '        """',
                  '        return self._missing("{}")'.format(op['id']), '']
    return '\n'.join(lines) + '\n'


def document(spec):
    doc = ['# 三家厂商 GPU 函数交付清单', '',
           '这份文件直接交给 C500、MLU590、910B 的负责人。每项都有函数接口、作用、输入输出、反向要求、交付位置和 NVIDIA 参考实现。', '',
           '## 先确定分工', '',
           '| 工作 | 厂商负责人交付 | UP 维护者负责 |', '|---|---|---|',
           '| 设备、AMP、通信 | 验证已有厂商 API；遇到不兼容项提供适配器 | 在 RuntimeBackend 中选择适配器，保留训练/推理调用 |',
           '| 专用 GPU 操作 | 按下面的 16 个方法交付完整操作及测试；可调用厂商已有库 | 接到原包装/.apply 入口，维护配置与导出兼容 |',
           '| 卷积、Linear、普通激活等标准 Torch | 验证原配方实际用到的 dtype/shape，提供异常清单 | 保留原模型算法，不重复包装这些操作 |', '',
           '16 个是本仓库保留的专用 GPU 操作入口，不是要求每家重写全部 PyTorch。标为点云可选的条目，只有恢复该任务时才需要接入。', '',
           '## 实现文件放在哪里', '', '```text', 'up/platform/<vendor>/',
           '  __init__.py      # 导出 NativeOps；普通导入不初始化设备',
           '  native_ops.py    # class NativeOps：实现下面同名方法，内部含所需 autograd',
           '  device.py        # 只有设备/AMP API 不兼容时才新增适配器',
           '  csrc/            # 可选：厂商 SDK 内核与编译代码',
           'tests/<vendor>/   # 前向、梯度、状态、布局、边界和实际运行测试', '```', '',
           '`<vendor>` 分别为 `metax_cuda`、`cambricon_mlu`、`ascend_npu`。这些是**交付路径约定**，当前还没有注册这三套 NativeOps 类；现有共享实现和 NVIDIA 原扩展继续按原路运行。', '',
           '复制 [完整接口模板](templates/native_ops.py) 后，将类命名为 `NativeOps`。未实现的方法明确抛出 NotImplementedError，允许按模型先交付一部分。模板本身不会让算子自动可用。', '',
           '**接入必须发生在原 `Function.apply()` 之前。** 厂商方法返回带所需梯度图的完整操作；不要在旧 `Function.forward()` 内套另一个完整 autograd Function，这会进入 no_grad 前向并切断预期的图。框架侧可在外层模块/兼容入口选厂商；直接使用旧 `.apply()` 的路径也要一起接入。', '',
           '## 三家分别做什么', '', '| 厂商 | config 值 | 设备/AMP API | 通信 | 交付目录 |', '|---|---|---|---|---|']
    for key, v in VENDORS.items():
        doc.append('| {} | `{}` | `torch.{}`，{} | {} | `up/platform/{}/` |'.format(
            v['name'], key, v['device'], v['plugin'], v['communication'], key))
    doc += ['', 'C500 的 CUDA 兼容服务优先复用；MLU/NPU 的主入口已映射到 torch.mlu/torch.npu。三家都先运行下面的接口检查，有真实差异才写适配器。厂商环境需在启动前安装好，config 不替换 Torch。', '',
            '按负责人下载任务表：[C500](assignments/metax_cuda.csv) · [MLU590](assignments/cambricon_mlu.csv) · [910B](assignments/ascend_npu.csv)。每张表都包含具体签名和 NVIDIA 源码链接。', '',
            '## A. 设备、AMP、通信接口：兼容则复用', '',
            '这里列 UP 实际需要的调用形式。完整标准库签名由所安装的 Torch 版本决定，不要求重写 Torch 整套接口。设备总量和已分配显存使用 byte。', '',
            '| 接口 | 返回 | 做什么 | NVIDIA 对应函数/实现 | 修改位置 |', '|---|---|---|---|---|']
    for s in spec['services']:
        doc.append('| `{}` | {} | {} | [{}]({}) · [实现源码]({}) | {} |'.format(s['signature'], s['output'], s['meaning'], s['nvidia'], s['reference'], s['nvidia_source'], s['target']))
    doc += ['', '另外，现有平台类提供 `device`、`amp`、`dist_backend`、`set_device(local_rank)`、`memory_available(device, needed_memory, reserve_bytes)`、`configure_dnn(options, legacy_nocudnn=None)`；参考 [RuntimeBackend]('+URL+'up/runtime/backend.py)。', '',
            'DNN 策略逐项处理 `enabled`、`benchmark`、`deterministic`、`allow_tf32`：NVIDIA 参考 `torch.backends.cudnn`；C500 核 mcDNN 的兼容控制项；MLU/910B 核 CNNL/CANN 实际能力。有对应控制项才接入，未提供的项明确报不支持；默认沿厂商原生策略。', '',
            'AMP 的调用顺序保持 `autocast → scale(loss).backward() → 必要时 unscale_/裁剪 → step → update`，缩放器随 checkpoint 保存恢复。参考 [PyTorch AMP 示例](https://docs.pytorch.org/docs/2.10/notes/amp_examples.html)。', '',
            '通信还需核 rank/world_size、组状态和异步 Work.wait()；不重写参数更新算法。TaskSyncBN 使用了 PyTorch 私有 SyncBatchNorm.apply，升级 Torch 时应单独检查签名与行为。', '',
            '## B. 专用 GPU 操作：逐方法交付', '',
            '以下 16 个方法属于 NativeOps。公共输入 Tensor 应在同一目标设备；索引输出和辅助状态也放在正确设备，不写死第0张卡。先完成 FP32 的 NVIDIA 配对；AMP/dtype 按原可执行配置补齐，不能仅凭 kernel 的模板声明认定整个 Python 包装支持某个 dtype。', '',
            '有梯度要求的方法必须返回可反向的完整操作；内部可用厂商已有 autograd 操作，或自写 torch.autograd.Function 并保存状态。NMS 等不要求梯度；RoIPointPool3d 的原反向未实现。', '',
            '| 方法 | 作用 | 当前任务状态 | NVIDIA Python 入口 |', '|---|---|---|---|']
    for op in spec['operations']:
        status = '共享路线可复用，公开入口/变体待验' if op.get('shared') else '需要厂商绑定或移植'
        doc.append('| [`{}`](#{}) | {} | {}；{} | {} |'.format(op['id'], op['id'].replace('_','-'),
            op['title'], status, op['scope'], md_link(op['python'][0])))
    for op in spec['operations']:
        doc += ['', '### '+op['id'].replace('_','-'), '', '**'+op['title']+'**', '',
                '厂商需要交付的方法：', '', '```python', op['vendor_signature'], '```', '',
                '对应原 NVIDIA 的完整入口：', '', '```python', op['public_signature'], '```', '',
                '- **输入与含义：** '+op['inputs'], '- **返回：** '+op['output'],
                '- **反向与状态：** '+op['backward'], '- **必须保留：** '+op['details'],
                '- **实现位置：** `up/platform/<vendor>/native_ops.py::NativeOps.'+op['id']+'`。',
                '- **使用范围：** '+op['scope'], '', 'NVIDIA 实现链：', '',
                '1. Python / autograd：'+'；'.join(md_link(r) for r in op['python']),
                '2. 扩展导出：'+'；'.join(md_link(r) for r in op['pybind']),
                '3. C++ 实现：'+'；'.join(md_link(r) for r in op['cpp']),
                '4. CUDA 算法：'+'；'.join(md_link(r) for r in op['cuda'])]
        if op.get('extra'):
            doc += ['', '`'+op['extra']+'` 是新增交付接口的显式上下文参数，由 UP 维护者从原全局标志传入；厂商代码不自行读取 UP 的全局业务变量。']
        if op.get('shared'):
            doc += ['', '已有共享参考：`'+op['shared']+'`。可复用其中适用变体，并补公共入口、设备/dtype 和反向契约测试。']
    doc += ['', '## C. 已有共享路线，不要求从零写', '']
    for shared in spec['shared']:
        doc += ['- `'+shared['signature']+'`：'+md_link(shared['ref'])]
    doc += ['', '这里的 nms 是业务中的 TorchVision 连续坐标路线；UP Native NMS 另有 offset 参数，不能直接拿前者替换后者。IoU 共享函数目前不覆盖 IoS。', '',
            '## D. 验收与交付', '',
            '每个方法交付：实现文件 + 实际依赖版本 + 固定输入/权重/上游梯度 + NVIDIA 配对结果 + 本卡重复结果。比较输出、所需梯度、dtype、shape、索引、状态和恢复；浮点保持 `rtol=1e-4, atol=1e-5`，整数/标志精确比较。', '',
            '空输入、并列分数、非连续布局和极值按对应原 NVIDIA 入口先建立参考；原版也报错/未定义的变体单列，不自行换一种输出后标为兼容。RoIAlign MAX、Cross Focal reduction、IoU3D 部分梯度和 RoIPoint 反向的原行为已在各条目写明。', '',
            'NVIDIA/C500/MLU/NPU 支持声明按函数实测记录；模板和导入成功只表示接口已定义。Softer NMS 在 CPU 执行，spring/linklink 属历史路线，都不列入这16项 GPU 内核任务。', '',
            '可选 GPU 内核放厂商 csrc，构建只针对该厂商；保留原 `up/extensions/csrc`。现有 setup.py 仅为 NVIDIA 构建 _C，厂商新增内核须提供自己的构建/安装命令。ONNX symbolic 的原节点和属性保持；不支持的导出单独报错。', '',
            '最后跑训练 → 训练内 eval → checkpoint 恢复 → 独立推理，以及双卡 DDP/SyncBN。未使用的点云操作可暂缓，状态写清楚；不改模型算法和训练配方。']
    return '\n'.join(doc) + '\n'


def main():
    spec = make_spec()
    (DOC / 'vendor-api-contracts.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    (DOC / 'VENDOR_IMPLEMENTATION.md').write_text(document(spec), encoding='utf-8', newline='\n')
    (DOC / 'templates').mkdir(exist_ok=True)
    (DOC / 'templates/native_ops.py').write_text(template(spec), encoding='utf-8', newline='\n')
    (DOC / 'assignments').mkdir(exist_ok=True)
    for vendor, info in VENDORS.items():
        with (DOC / 'assignments' / (vendor + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['厂商','类别','交付函数','含义/输入','返回','反向/状态','实现位置','当前状态','NVIDIA参考','C++实现','CUDA算法'])
            for s in spec['services']:
                writer.writerow([info['name'],s['section'],s['signature'],s['meaning'],s['output'],'按厂商原状态语义',s['target'].replace('{vendor}',vendor),s['action'],s['nvidia']+' '+s['reference'],s['nvidia_source'],'厂商 Torch/SDK 实现'])
            for op in spec['operations']:
                writer.writerow([info['name'],'专用GPU操作',op['vendor_signature'],op['inputs'],op['output'],op['backward'],
                    'up/platform/'+vendor+'/native_ops.py::NativeOps.'+op['id'],
                    ('共享实现可复用，变体待验' if op.get('shared') else '厂商绑定/移植')+'；'+op['scope'],
                    op['public_signature']+' '+op['python'][0]['url'],
                    ' ; '.join(r['signature']+' '+r['url'] for r in op['cpp']),
                    ' ; '.join(r['url'] for r in op['cuda'])])
    print(json.dumps(dict(services=len(spec['services']), complete_gpu_operations=len(spec['operations']),
        assignment_rows_per_vendor=len(spec['services'])+len(spec['operations'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
