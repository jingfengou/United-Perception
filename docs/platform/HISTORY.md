# Fork 与提交说明

本仓库直接 fork 自 [ModelTC/United-Perception](https://github.com/ModelTC/United-Perception)。
`main` 保留上游 `52aaeb9ed27a490820696d7754d19f98cd6ecd11`，平台改造发布在 `codex/up-platform-clean`。
[旧实验仓库](https://github.com/jingfengou/United-Perception-Hardware) 的分支、Release 和证据没有删除或强制改写。

## 统一维护分支

2026-10-26 起，平台层、通用镜像和宿主机启动脚本统一在 `codex/up-platform-clean` 维护，作为仓库默认入口。
新增功能在此分支按责任拆分 commit。

原 `codex/unified-vendor-image`、`codex/up-docker-launcher` 的提交已全部快进并入；清理分支名不改变提交历史或文件。
合并前的功能节点仍可通过固定提交查看：

- [通用镜像与两套隔离环境：9145d0a](https://github.com/jingfengou/United-Perception/tree/9145d0a6b00b93a82063899241f3acc201a5598d)
- [启动脚本、完整示例与验证：cc2f3d9](https://github.com/jingfengou/United-Perception/tree/cc2f3d9a730e217b789b5209b50623027c5902f8)

历史报告中的分支字段保留执行时的记录；其源码、提交和证据仍能在统一分支历史中追溯。`main` 继续保留原版 UP。

## 原子提交

每个提交对应一个责任边界。先加入不影响 UP 入口的平台能力，再接入执行流程与各任务，最后发布证据和 CI。
原生源码移动、构建和旧算子兼容门面放在同一提交，避免只改加载路径却没有对应构建入口。
可在 GitHub [逐个审阅提交](https://github.com/jingfengou/United-Perception/commits/codex/up-platform-clean)。

| 提交 | 单一目的 |
| --- | --- |
| `ae74423` | 固定上游基线和原始硬件调用清单 |
| `4ca5a4c` | 平台、设备、算子父类及显式选择契约 |
| `7ae6512` | 公共 PyTorch 实现、CPU 参考和用途隔离 |
| `8463d43` | 原生操作的完整 autograd 与 ONNX 语义 |
| `ec7b067` | NVIDIA 设备、训练/推理和算子绑定 |
| `2816228` | 沐曦显式绑定与必要布局适配 |
| `8c5b8cf` | 昇腾/寒武纪模板、新厂商生成器和 SDK 反向示例 |
| `a006fb9` | UP 命令、设备运行时、训练及训练内 eval 接入 |
| `415fecd` | 原生源码移动、厂商显式构建和旧导入门面 |
| `3a5dbdf` | 公共骨干网络和层工具改为平台调用 |
| `b7d9daf` | 分类、蒸馏和自监督操作接入 |
| `2f74b52` | 2D 检测操作接入 |
| `9932ef4` | 点云和稀疏张量操作接入 |
| `780f2f6` | 分割、NAS、多任务和可选工具链接入 |
| `ddd06b4` | 边界检查及真机复现工具 |
| `c7123d2` | NVIDIA 的成功、失败与源码校验证据 |
| `c2e4b97` | 沐曦的成功、失败与源码校验证据 |
| `78f0db6` | 接口文档、验收边界和 CPU CI |

测试随相应接口/厂商提交逐步加入，硬件复现工具和结果单列，便于逐项审查。
上述提交之外的 fork 发布说明只更改仓库地址、提交索引及历史检查报告，不修改运行时。

## 与已测试版本的对应

旧仓库的已测试快照为 [a7508ac](https://github.com/jingfengou/United-Perception-Hardware/commit/a7508ac7464185d2fa3b869b9dbce6ed4a0d42e0)。
重新拆分后的 `78f0db6f8e922b3f7d44334c281175430c138459` 与该快照具有完全相同的 Git tree：

```text
330da029db475202ea5ae4357951c72f7155ec68
```

这是逐文件字节、路径和文件模式相同，不是仅凭“代码看起来相似”。原始模型、配方、容差和证据没有因重排提交而变化。
本次不重复真机训练，也不提升已有验收状态；仍以 [验证报告](VALIDATION.md) 中的通过、失败和未验证项为准。

新增的 [逐提交 CPU 检查](validation/20261023/atomic-history.json) 隐藏全部 GPU，逐个导出中间提交，执行 Python 语法检查和当时已加入的契约测试。
它用于检查提交依赖是否完整，不代表新增的硬件或数据集精度验证。

17 个重排后的快照均通过，最后一个快照执行完整 22 项契约测试。
环境为 Python 3.12.3、PyTorch 2.10.0+cu126，GPU 不可见。
原始归档、JUnit 和日志保留在 NVIDIA 验证服务器的 `atomic-publication-20261023/payload/` 子目录，报告记录归档及日志的 SHA256。

复查某个中间提交时，在匹配的 CPU 测试环境中使用一个新的目录：

```bash
git worktree add --detach /absolute/new/up-check 2816228
cd /absolute/new/up-check
python -m compileall -q up up_platform tools tests
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONHASHSEED=0 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/platform -q --junitxml=tests.xml
```

将示例 SHA 换成上表中相应提交；早期提交只执行当时已加入的测试。
