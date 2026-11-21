"""
Re-export Volume 3D model từ .pt (TorchScript/checkpoint cũ) sang .pt2 (torch.export format).

Sử dụng khi:
- File .pt2 hiện tại không load được do PyTorch version mismatch
- Có file .pt (TorchScript) hoặc checkpoint cũ cùng tên

Cách dùng:
    python scripts/reexport_volume_3d.py --source models/pretrained/cancers/brain/mri/volume.pt
    python scripts/reexport_volume_3d.py --source models/pretrained/cancers/brain/mri/volume.pt --output models/pretrained/cancers/brain/mri/volume.pt2

Script sẽ:
1. Load model từ file .pt nguồn
2. Tạo example input dựa trên volume.json manifest (nếu có)
3. Export model sang .pt2 format bằng torch.export
4. Lưu file .pt2 mới tương thích với PyTorch hiện tại
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _find_manifest(source_path: Path) -> Path | None:
    """Tìm volume.json manifest cùng thư mục với file nguồn."""
    directory = source_path.parent
    candidates = [
        "volume.json",
        "volume_model.json",
        f"{source_path.stem}.json",
    ]
    for name in candidates:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _get_input_shape(manifest_path: Path | None) -> tuple[int, int, int]:
    """Đọc input_shape từ manifest, mặc định (1, 64, 64, 64)."""
    if manifest_path is None or not manifest_path.is_file():
        return (1, 64, 64, 64)
    try:
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        shape = data.get("input_shape", [1, 64, 64, 64])
        if len(shape) == 3:
            # (D, H, W) → thêm channel
            return (1, *shape)
        return tuple(shape)
    except Exception as exc:
        logger.warning("Không đọc được manifest: %s", exc)
        return (1, 64, 64, 64)


def _get_channels(manifest_path: Path | None) -> int:
    """Đọc số channel từ manifest, mặc định 1 (grayscale CT/MRI)."""
    if manifest_path is None or not manifest_path.is_file():
        return 1
    try:
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        modalities = data.get("input_modalities", ["image"])
        # RGB cho ảnh 2D slice, grayscale cho CT/MRI volume
        if "rgb" in modalities:
            return 3
        return 1
    except Exception:
        return 1


def load_source_model(source_path: Path) -> torch.nn.Module:
    """Load model từ file .pt nguồn (TorchScript hoặc checkpoint)."""
    logger.info("Đang load model từ %s", source_path)

    # Thử TorchScript trước
    jit_err_name: str | None = None
    try:
        model = torch.jit.load(str(source_path), map_location="cpu")
        logger.info("Load thành công qua TorchScript")
        return model
    except Exception as jit_err:
        jit_err_name = type(jit_err).__name__
        logger.warning("TorchScript load thất bại: %s", jit_err_name)

    # Thử regular checkpoint
    try:
        checkpoint = torch.load(str(source_path), map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Cần rebuild model architecture — không thể tự động
                raise TypeError(
                    "Checkpoint có model_state_dict nhưng không có model architecture. "
                    "Cần cung cấp training script để rebuild model."
                )
            elif "state_dict" in checkpoint:
                raise TypeError(
                    "Checkpoint có state_dict nhưng không có model architecture. "
                    "Cần cung cấp training script để rebuild model."
                )
        elif isinstance(checkpoint, torch.nn.Module):
            logger.info("Load thành công qua raw model checkpoint")
            return checkpoint
        else:
            raise TypeError(f"Unknown checkpoint format: {type(checkpoint)}")
    except Exception as ckpt_err:
        raise RuntimeError(
            f"Không load được model từ {source_path}. "
            f"TorchScript: {jit_err_name or 'N/A'}, Checkpoint: {type(ckpt_err).__name__}. "
            f"Cần file TorchScript (.pt) hoặc training script để rebuild."
        ) from ckpt_err


def export_to_pt2(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple[int, int, int],
) -> None:
    """Export model sang .pt2 format bằng torch.export."""
    logger.info("Đang export model sang .pt2 với input_shape=%s", input_shape)

    # Tạo example input
    example = torch.randn(*input_shape, dtype=torch.float32)

    # TorchScript model cần unwrap trước khi export
    if hasattr(model, "forward") and hasattr(model, "_c"):
        # TorchScript module — cần convert về nn.Module
        # Trường hợp này không thể export trực tiếp, cần model gốc
        logger.warning(
            "Model là TorchScript module — không thể export trực tiếp sang torch.export. "
            "Cần model nn.Module gốc (trước khi TorchScript trace/jit)."
        )
        # Fallback: save as TorchScript với tên .pt2
        # Một số phiên bản PyTorch có thể load TorchScript qua fallback
        torch.jit.save(model, str(output_path))
        logger.info(
            "Đã lưu model dưới dạng TorchScript tại %s (fallback, không phải torch.export native). "
            "Hệ thống sẽ dùng TorchScript fallback khi load.",
            output_path,
        )
        return

    # Export với torch.export
    try:
        exported = torch.export.export(model, (example,))
        torch.export.save(exported, str(output_path))
        logger.info("Export thành công sang .pt2 tại %s", output_path)
    except Exception as export_err:
        logger.error(
            "torch.export thất bại: %s. Thử TorchScript fallback.",
            type(export_err).__name__,
        )
        # Fallback: TorchScript trace
        try:
            traced = torch.jit.trace(model, example)
            torch.jit.save(traced, str(output_path))
            logger.info(
                "Đã lưu model qua TorchScript trace tại %s (fallback). "
                "Hệ thống sẽ dùng TorchScript fallback khi load.",
                output_path,
            )
        except Exception as trace_err:
            raise RuntimeError(
                f"Tất cả phương pháp export thất bại. "
                f"torch.export: {type(export_err).__name__}, "
                f"TorchScript trace: {type(trace_err).__name__}. "
                f"Cần kiểm tra model architecture và PyTorch version."
            ) from trace_err


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-export Volume 3D model sang .pt2 format tương thích PyTorch"
    )
    parser.add_argument(
        "--source", "-s",
        required=True,
        help="Đường dẫn file model nguồn (.pt TorchScript hoặc checkpoint)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Đường dẫn file .pt2 đầu ra (mặc định: cùng tên, đổi suffix sang .pt2)",
    )
    parser.add_argument(
        "--input-shape",
        nargs=3,
        type=int,
        help="Kích thước input (channels depth height width), ví dụ: 1 64 64 64",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị log chi tiết",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    source_path = Path(args.source)
    if not source_path.is_file():
        print(f"LỖI: File nguồn không tồn tại: {source_path}")
        return 1

    # Xác định output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = source_path.with_suffix(".pt2")

    # Tìm manifest
    manifest_path = _find_manifest(source_path)
    if manifest_path:
        logger.info("Tìm thấy manifest: %s", manifest_path)

    # Xác định input shape
    if args.input_shape:
        input_shape = tuple(args.input_shape)
        if len(input_shape) == 3:
            input_shape = (1, *input_shape)  # Thêm channel
    else:
        channels = _get_channels(manifest_path)
        shape_from_manifest = _get_input_shape(manifest_path)
        if len(shape_from_manifest) == 3:
            input_shape = (channels, *shape_from_manifest)
        else:
            input_shape = shape_from_manifest

    logger.info("Input shape: %s", input_shape)

    # Load model nguồn
    try:
        model = load_source_model(source_path)
    except RuntimeError as exc:
        print(f"\nLỖI: {exc}")
        return 1

    # Đặt model về chế độ eval
    if hasattr(model, "eval"):
        model = model.eval()

    # Export sang .pt2
    try:
        export_to_pt2(model, output_path, input_shape)
    except RuntimeError as exc:
        print(f"\nLỖI: {exc}")
        return 1

    # Xác nhận file đầu ra
    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print("\n✅ Thành công! File .pt2 đã được tạo:")
        print(f"   Đường dẫn: {output_path}")
        print(f"   Kích thước: {size_mb:.2f} MB")
        print("\nLưu ý:")
        print("   - Đảm bảo volume.json manifest ở cùng thư mục với .pt2")
        print(f"   - Chạy test: python -c \"from medical.volume_inference import load_volume_model; load_volume_model('{output_path}')\"")
        print("   - Nếu hệ thống vẫn báo lỗi, cần re-export từ training script gốc")
    else:
        print(f"\nLỖI: File đầu ra không được tạo: {output_path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
