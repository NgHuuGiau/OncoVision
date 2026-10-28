"""Chuyen checkpoint pretrained ben thu ba (kho E:) sang hop dong classifier cua du an.

Pham vi hien tai: backbone ho ResNet (resnet18/resnet50) voi checkpoint kieu torchvision/timm
(keys conv1..layer4 + fc). Head ngoai duoc dua vao contract duoi dang head="linear".

shortcut: chi ho ResNet; convnext/efficientnet/timm-effnet/MONAI/nnUNet can adapter rieng.
Vi du:
    python scripts/import_foreign_model.py \
        --source-dir "E:\\Model\\models\\pretrained\\cancers\\thyroid\\classification\\agentic_resnet18" \
        --target thyroid --modality ultrasound
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from medical.cnn_classifier import MedicalCNNClassifier

CANCERS_DIR = REPO_ROOT / "models" / "pretrained" / "cancers"
STATE_KEYS = ("model_state_dict", "model_state", "state_dict")


def _load_checkpoint(path: Path) -> dict:
    """Load checkpoint from .pt/.pth/.safetensors, return state dict."""
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    # .pt, .pth, etc.
    return torch.load(path, map_location="cpu", weights_only=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state(obj: object) -> dict:
    if isinstance(obj, dict):
        for key in STATE_KEYS:
            candidate = obj.get(key)
            if isinstance(candidate, dict) and candidate and all(torch.is_tensor(v) for v in candidate.values()):
                return candidate
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise ValueError("Không tìm thấy state_dict (model_state_dict/model_state/state_dict) trong checkpoint.")


def _detect_backbone(state: dict) -> str:
    keys = set(state)
    if "layer1.0.conv1.weight" in keys:
        bottleneck = "layer1.0.conv3.weight" in keys
        if bottleneck:
            return "resnet50"
        blocks = sum(1 for k in keys if k.startswith("layer4.") and k.endswith(".conv1.weight"))
        if blocks <= 2:
            return "resnet18"
        if blocks == 3:
            return "resnet34"
        raise ValueError(f"ResNet cơ bản có {blocks} block ở layer4 (chưa hỗ trợ resnet152).")
    # timm efficientnet_b0: keys like 'backbone.conv_stem', 'backbone.blocks.*', 'backbone.conv_head', 'head.*'
    if any(k.startswith("backbone.conv_stem") for k in keys) and any(k.startswith("backbone.blocks") for k in keys):
        return "timm:efficientnet_b0"
    raise ValueError("Chỉ hỗ trợ ResNet (resnet18/resnet34/resnet50) và timm efficientnet_b0 ở bước này.")


def _read_image_size(source_dir: Path, override: int | None) -> int:
    if override:
        return int(override)
    for name in ("configs/preprocess.json", "preprocess.json"):
        path = source_dir / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("image_size"):
                return int(data["image_size"])
    raise ValueError("Không xác định được image_size; truyền --image-size.")


def _read_preprocess(source_dir: Path) -> dict:
    for name in ("configs/preprocess.json", "preprocess.json"):
        path = source_dir / name
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_temperature(manifest: dict, override: float | None) -> float:
    if override is not None:
        return float(override)
    text = str(manifest.get("threshold", ""))
    match = re.search(r"T\s*=\s*([0-9.]+)", text)
    return float(match.group(1)) if match else 1.0


def _convert_resnet(state: dict, labels: list, backbone_name: str) -> tuple[str, dict, int]:
    if "fc.weight" not in state or "fc.bias" not in state:
        raise ValueError("ResNet checkpoint thiếu 'fc.*'.")
    fc_weight = state["fc.weight"]
    out_features = int(fc_weight.shape[0])
    backbone_state = {f"backbone.{k}": v for k, v in state.items() if not k.startswith("fc.")}
    if out_features == 1:
        num_classes = 2
        if len(labels) != 2:
            raise ValueError(f"1-logit cần 2 lớp, có {labels!r}.")
        zeros = torch.zeros_like(fc_weight)
        head_weight = torch.cat([zeros, fc_weight], dim=0)
        head_bias = torch.cat([torch.zeros_like(state["fc.bias"]), state["fc.bias"]], dim=0)
    else:
        num_classes = out_features
        if len(labels) != num_classes:
            raise ValueError(f"Manifest {len(labels)} lớp, head {num_classes} logit.")
        head_weight = fc_weight
        head_bias = state["fc.bias"]
    backbone_state["classifier.weight"] = head_weight
    backbone_state["classifier.bias"] = head_bias
    return backbone_name, backbone_state, num_classes


def _convert_timm_efficientnet(state: dict, labels: list) -> tuple[str, dict, int]:
    # state keys: 'backbone.conv_stem', 'backbone.blocks.*', 'backbone.conv_head', 'head.1.*', 'head.4.*'
    # build feature_state with 'backbone.' prefix to match MedicalCNNClassifier structure
    feature_state = {k: v for k, v in state.items() if k.startswith("backbone.")}
    # compose head.1 (1280->256) + head.4 (256->5) into single Linear(1280->5)
    if "head.1.weight" not in state or "head.4.weight" not in state:
        raise ValueError("timm efficientnet checkpoint thiếu 'head.1.*' hoặc 'head.4.*'.")
    W1 = state["head.1.weight"]      # [256, 1280]
    b1 = state["head.1.bias"]        # [256]
    W4 = state["head.4.weight"]      # [5, 256]
    b4 = state["head.4.bias"]        # [5]
    head_weight = W4 @ W1            # [5, 1280]
    head_bias = W4 @ b1 + b4         # [5]
    out_features = int(head_weight.shape[0])
    num_classes = out_features
    if len(labels) != num_classes:
        raise ValueError(f"Manifest {len(labels)} lớp, head cuối {num_classes} logit.")
    feature_state["classifier.weight"] = head_weight
    feature_state["classifier.bias"] = head_bias
    return "timm:efficientnet_b0", feature_state, num_classes


def convert(source_dir: Path, target: str, modality: str | None, dest_dir: Path,
            image_size: int | None, temperature: float | None, dry_run: bool) -> Path:
    manifest_path = source_dir / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    checkpoint_name = manifest.get("checkpoint") or "final_model.pt"
    checkpoint_path = source_dir / checkpoint_name
    if not checkpoint_path.is_file():
        # Try .safetensors fallback
        for alt in ("model.safetensors", "pytorch_model.bin"):
            alt_path = source_dir / alt
            if alt_path.is_file():
                checkpoint_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Không thấy checkpoint {checkpoint_path}.")

    obj = _load_checkpoint(checkpoint_path)
    state = _extract_state(obj)
    backbone = _detect_backbone(state)
    labels = list(manifest.get("classes") or [])

    if backbone.startswith("timm:"):
        backbone_name, backbone_state, num_classes = _convert_timm_efficientnet(state, labels)
    else:
        backbone_name, backbone_state, num_classes = _convert_resnet(state, labels, backbone)

    size = _read_image_size(source_dir, image_size)
    pre = _read_preprocess(source_dir)
    interp = pre.get("interpolation", "bilinear")
    temp = _read_temperature(manifest, temperature)

    model = MedicalCNNClassifier(num_classes=num_classes, backbone=backbone_name, pretrained=False,
                                 dropout=0.0, head="linear")
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    # Bỏ qua num_batches_tracked (BN tracking buffers, không ảnh hưởng inference)
    unexpected = [u for u in unexpected if "num_batches_tracked" not in u]
    if missing or unexpected:
        raise ValueError(f"Không map khớp contract: missing={missing[:5]} unexpected={unexpected[:5]}")

    contract = {
        "num_classes": num_classes,
        "backbone": backbone_name,
        "head": "linear",
        "dropout": 0.0,
        "model_state_dict": backbone_state,
        "class_labels": labels,
        "image_size": size,
        "interpolation": interp,
        "temperature": temp,
        "threshold": manifest.get("threshold"),
        "source_model": manifest.get("model_name") or source_dir.name,
    }

    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / "classifier.pt"
    if dry_run:
        print(f"[dry-run] {backbone_name} num_classes={num_classes} labels={labels} image_size={size} T={temp}")
        print(f"[dry-run] sẽ ghi {out_path}")
        return out_path

    torch.save(contract, out_path)
    sidecar = {
        "repo": manifest.get("source") or str(source_dir),
        "repo_type": "foreign-converted",
        "file": checkpoint_name,
        "sha256": _sha256(checkpoint_path),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probe_kind": "classifier",
        "probe_ok": True,
        "converted_from": str(source_dir),
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "backbone": backbone_name,
        "head": "linear",
        "image_size": size,
        "interpolation": interp,
        "temperature": temp,
        "class_labels": labels,
        "source_model": contract["source_model"],
        "license": manifest.get("license"),
        "status": manifest.get("status"),
        "limitations": manifest.get("known_limitations"),
    }
    (out_path.with_suffix(".source.json")).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {out_path} ({backbone_name}, {num_classes} lớp: {labels})")
    print(f"Sidecar: {out_path.with_suffix('.source.json')}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert checkpoint ngoài → contract classifier dự án.")
    parser.add_argument("--source-dir", required=True, help="Thư mục model E: (có model_manifest.json).")
    parser.add_argument("--target", required=True, help="Key nhóm ung thư (vd thyroid).")
    parser.add_argument("--modality", default=None, help="Thư mục modality con (vd ultrasound).")
    parser.add_argument("--dest", default=None, help="Thư mục đích (mặc định models/pretrained/cancers/<target>/<modality>).")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    dest = Path(args.dest) if args.dest else (CANCERS_DIR / args.target / args.modality if args.modality else CANCERS_DIR / args.target)
    convert(Path(args.source_dir), args.target, args.modality, dest,
            args.image_size, args.temperature, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())