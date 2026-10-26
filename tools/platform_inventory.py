"""Audit the pinned upstream, not a previous platform implementation."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

UPSTREAM = "52aaeb9ed27a490820696d7754d19f98cd6ecd11"
ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ("torch", "torchvision", "spconv", "spring", "mqbench", "msbench", "numba")
TENSOR_METHODS = {
    "cuda", "cpu", "to", "type", "type_as", "mm", "bmm", "matmul", "softmax",
    "log_softmax", "sigmoid", "tanh", "relu", "sum", "mean", "max", "min", "norm",
    "sqrt", "pow", "exp", "log", "topk", "sort", "argsort", "gather", "scatter_",
    "scatter_add_", "index_select", "index_add_", "masked_fill", "masked_fill_",
    "reshape", "view", "permute", "transpose", "contiguous", "flatten", "expand",
    "repeat", "squeeze", "unsqueeze", "new_zeros", "new_ones", "new_empty",
    "new_tensor", "bernoulli_", "normal_", "uniform_", "random_", "backward",
}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted(node.value)
        return prefix + "." + node.attr if prefix else None
    return None


def inspect_source(path, source):
    tree = ast.parse(source, filename=path)
    aliases = {}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for entry in node.names:
                aliases[entry.asname or entry.name.split(".")[0]] = entry.name if entry.asname else entry.name.split(".")[0]
                if entry.name.split(".")[0] in EXTERNAL:
                    imports.append({"line": node.lineno, "module": entry.name})
        elif isinstance(node, ast.ImportFrom) and node.module:
            for entry in node.names:
                aliases[entry.asname or entry.name] = node.module + "." + entry.name
            if node.module.split(".")[0] in EXTERNAL or "extensions" in node.module:
                imports.append({"line": node.lineno, "module": node.module})
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = dotted(node.func)
            canonical = None
            if name:
                head, *tail = name.split(".")
                canonical = ".".join([aliases.get(head, head), *tail])
            if canonical and canonical.split(".")[0] in EXTERNAL:
                calls.append({"line": node.lineno, "kind": "resolved_call", "operation": canonical})
            elif isinstance(node.func, ast.Attribute) and node.func.attr in TENSOR_METHODS:
                calls.append({"line": node.lineno, "kind": "tensor_candidate", "operation": "." + node.func.attr})
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            calls.append({"line": node.lineno, "kind": "operator", "operation": "@"})
        elif isinstance(node, ast.Attribute) and node.attr in ("is_cuda", "cuda", "cudnn"):
            calls.append({"line": node.lineno, "kind": "device_attribute", "operation": dotted(node) or node.attr})
    return {"imports": sorted(imports, key=lambda x: (x["line"], x["module"])),
            "calls": sorted(calls, key=lambda x: (x["line"], x["kind"], x["operation"]))}


def inventory():
    paths = git("ls-tree", "-r", "--name-only", UPSTREAM, "--", "up", "setup.py").decode().splitlines()
    files = {}
    for path in paths:
        if Path(path).suffix not in (".py", ".cpp", ".cu", ".h"):
            continue
        raw = git("show", UPSTREAM + ":" + path)
        entry = {"sha256": hashlib.sha256(raw).hexdigest()}
        if path.endswith(".py"):
            entry.update(inspect_source(path, raw.decode("utf-8-sig")))
        files[path] = entry
    return {"upstream": UPSTREAM, "lightx2v_reference": "19c19614b067f1746ac24350dfd097e1c0b48591",
            "limitations": ["Tensor-method candidates require call-site review; Python AST is not a tensor type checker.",
                            "External package calls are inventoried, not evidence that their kernels work.",
                            "CPU transforms and framework metadata are classified separately from accelerator operations."],
            "files": files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = ROOT / "tests/platform/fixtures/upstream_inventory.json"
    expected = json.dumps(inventory(), ensure_ascii=False, indent=2) + "\n"
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8", newline="\n")
    elif not path.exists() or path.read_text(encoding="utf-8") != expected:
        raise SystemExit("Upstream inventory drift; run --write and review the diff")
    data = json.loads(expected)
    print(json.dumps({"upstream": UPSTREAM, "files": len(data["files"]),
                      "calls": sum(len(f.get("calls", [])) for f in data["files"].values())}))


if __name__ == "__main__":
    main()
