"""Short UP/native-extension environment regression, not an accuracy benchmark."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys


def cpu(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(item) for item in value)
    return value


def tensors(value, prefix=""):
    import torch
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(tensors(item, prefix + "/" + str(key)))
    elif isinstance(value, (tuple, list)):
        for key, item in enumerate(value):
            result.update(tensors(item, prefix + "/" + str(key)))
    return result


def compare(actual, expected):
    import torch
    actual, expected = tensors(actual), tensors(expected)
    failures, maximum = [], 0.0
    for name in sorted(actual.keys() | expected.keys()):
        a, b = actual.get(name), expected.get(name)
        if a is None or b is None or a.shape != b.shape or a.dtype != b.dtype:
            failures.append({"tensor": name, "reason": "missing/shape/dtype"})
            continue
        difference = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
        maximum = max(maximum, difference)
        passed = torch.allclose(a, b, rtol=1e-4, atol=1e-5) if a.is_floating_point() else torch.equal(a, b)
        if not passed:
            failures.append({"tensor": name, "max_abs": difference})
    return {"passed": not failures, "tensors": len(actual), "max_abs": maximum, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", required=True, choices=("nvidia", "metax"))
    parser.add_argument("--phase", required=True, choices=("training", "inference"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--deterministic", action="store_true", help="Use native deterministic algorithms for this regression only")
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["DEFAULT_TASKS"] = "cls:det"
    import numpy as np
    import torch
    from up_platform import configure_platform
    from up_platform.ops._native.loader import load_native
    platform = configure_platform({"backend": args.vendor}, purpose=args.phase)
    from up.models.model_helper import ModelHelper
    from up.utils.general.yaml_loader import load_yaml
    from up.utils.model.optimizer_helper import BaseOptimizer
    platform.device.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = args.deterministic
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(131)
    np.random.seed(131)
    torch.manual_seed(131)
    torch.cuda.manual_seed_all(131)
    source = args.source_root.resolve()
    config_path = source / "configs/cls/resnet/res18.yaml"
    config = load_yaml(str(config_path))
    os.chdir(config_path.parent)
    model = ModelHelper(copy.deepcopy(config["net"]))
    initial = cpu(model.state_dict())
    model = platform.device.move(model)
    image = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(500))
    original_batch = {"image": image, "gt": torch.tensor([1, 2])}
    def batch():
        # ModelHelper appends outputs to its input dictionary. Use a new batch
        # for each forward, as the UP data loader does; never retain loss keys.
        return platform.device.move(copy.deepcopy(original_batch))
    record = {"initial": initial}
    if args.phase == "training":
        optimizer = BaseOptimizer(copy.deepcopy(config["trainer"]["optimizer"]), model).build_optimizer()
        losses = []
        for step in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(batch())
            loss = sum(value for key, value in output.items() if "loss" in key and isinstance(value, torch.Tensor))
            loss.backward()
            optimizer.step()
            losses.append(cpu(loss))
        before = cpu(model.state_dict())
        modes = [child.training for child in model.modules()]
        with platform.execution.validation(model):
            record["eval"] = cpu(model(batch()))
            assert platform.purpose == "training"
        assert modes == [child.training for child in model.modules()]
        assert compare(before, cpu(model.state_dict()))["max_abs"] == 0
        record.update(losses=losses, final=cpu(model.state_dict()))
        torch.save({"model": record["final"], "optimizer": cpu(optimizer.state_dict()), "eval": record["eval"]},
                   args.output / "checkpoint.pt")
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        with torch.no_grad():
            record["inference"] = cpu(model(batch()))
        restoration = compare(record["inference"], state["eval"])
        record["restored"] = cpu(model.state_dict())
    extension = load_native(args.vendor)
    features = torch.arange(128, dtype=torch.float32, device="cuda").reshape(1, 2, 8, 8).requires_grad_()
    rois = torch.tensor([[0., 1., 1., 6., 6.]], device="cuda")
    aligned = platform.category("detection").roi_align(features, rois, 3, 3, 1., 2, "AVG", aligned=True)
    aligned.sum().backward()
    record["roi_align"] = cpu(aligned)
    record["roi_align_gradient"] = cpu(features.grad)
    dets = torch.tensor([[0., 0., 4., 4., .9], [1., 1., 5., 5., .8], [10., 10., 12., 12., .7]], device="cuda")
    record["nms"] = cpu(platform.category("detection").naive_nms(dets, .3, offset=1))
    torch.cuda.synchronize()
    torch.save(record, args.output / "tensors.pt")
    result = {"vendor": args.vendor, "phase": args.phase, "model": "UP ResNet18", "steps": 2 if args.phase == "training" else 0,
        "seed": 131, "input": [2, 3, 224, 224], "purpose": platform.purpose,
        "python": sys.executable, "torch": torch.__version__, "torch_path": torch.__file__,
        "device": torch.cuda.get_device_name(), "extension": str(extension.__file__),
        "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
        "rtol": 1e-4, "atol": 1e-5, "scope": "synthetic functional/environment regression, not dataset accuracy"}
    result["deterministic"] = args.deterministic
    result["loaded_vendor_libraries"] = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in ("libcublas", "libcudnn", "libcudart", "libcuda.", "libmcblas", "libmcdnn", "libmcruntime"))})
    if args.phase == "inference":
        result["checkpoint_eval_comparison"] = restoration
    if args.reference:
        result["comparison"] = compare(record, torch.load(args.reference / "tensors.pt", map_location="cpu", weights_only=False))
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if (args.reference and not result["comparison"]["passed"]) or (args.phase == "inference" and not restoration["passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
