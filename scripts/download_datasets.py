#!/usr/bin/env python3
"""
Tải dataset ảnh y khoa cho 11 loại ung thư từ HuggingFace.

Mỗi loại ung thư sẽ tải khoảng 100 ảnh (vừa đủ để test) vào data/datasets/<cancer_type>/.

Sử dụng:
    python scripts/download_datasets.py                  # Tải tất cả 11 loại
    python scripts/download_datasets.py --cancer brain   # Chỉ tải brain
    python scripts/download_datasets.py --cancer brain,breast,lung  # Tải nhiều loại
    python scripts/download_datasets.py --dry-run        # Xem trước không tải
    python scripts/download_datasets.py --max-images 100 # Giới hạn 100 ảnh mỗi loại (mặc định)

Dữ liệu sẽ được lưu vào:
    data/datasets/<cancer_type>/<class>/image.jpg

Rate limit handling:
    - Tự động chờ khi bị HuggingFace rate limit (429)
    - Delay 0.2 giây giữa mỗi request để tránh bị chặn
    - Default 100 ảnh/mỗi loại để tải nhanh và không bị rate limit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Force flush stdout trên Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

# Dataset definitions for 11 cancer types
# Each entry has:
#   - repo_id: HuggingFace dataset repo
#   - classes: danh sách lớp/phân loại
#   - class_dirs: thư mục tương ứng trong dataset
#   - modality: loại ảnh y tế
#   - description: mô tả ngắn

DATASETS: list[dict[str, Any]] = [
    {
        "cancer_type": "brain",
        "repo_id": "BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset",
        "modality": "mri",
        "description": "Ung thư não — MRI (glioma, meningioma, pituitary, no_tumor) — 7,200 ảnh",
        "classes": ["glioma", "meningioma", "pituitary", "no_tumor"],
        "class_dirs": ["glioma", "meningioma", "pituitary", "notumor"],
        "image_extensions": [".jpg", ".jpeg", ".png"],
        "download_method": "direct",  # Co file anh truc tiep
    },
    {
        "cancer_type": "breast",
        "repo_id": "spsahikotha/Breast-Cancer-Detection",
        "modality": "ultrasound",
        "description": "Ung thư vú — Siêu âm (benign, malignant, normal)",
        "classes": ["benign", "malignant", "normal"],
        "class_dirs": ["benign", "malignant", "normal"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "lung",
        "repo_id": "dorsar/lung-cancer",
        "modality": "ct",
        "description": "Ung thư phổi — CT (benign, malignant, normal)",
        "classes": ["benign", "malignant", "normal"],
        "class_dirs": ["benign", "malignant", "normal"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "thyroid",
        "repo_id": "Johnyquest7/TN5000-thyroid-nodule-classification",
        "modality": "ultrasound",
        "description": "Ung thư tuyến giáp — Siêu âm (benign, malignant) — 5,000 ảnh",
        "classes": ["benign", "malignant"],
        "class_dirs": ["Benign", "Malignant"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "kidney",
        "repo_id": "spsahikotha/Kidney-Cancer-Detection",
        "modality": "ct",
        "description": "Ung thư thận — CT (cyst, tumor, normal)",
        "classes": ["cyst", "tumor", "normal"],
        "class_dirs": ["cyst", "tumor", "normal"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "liver",
        "repo_id": "seongun/liver-tumor-classification",
        "modality": "ct",
        "description": "Ung thư gan — CT (normal, tumor)",
        "classes": ["normal", "tumor"],
        "class_dirs": ["normal", "tumor"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "parquet",
        "image_column": "sample",
        "label_column": "label",
        "label_map": {0: "normal", 1: "tumor"},
    },
    {
        "cancer_type": "stomach",
        "repo_id": "spsahikotha/Stomach-Cancer-Detection",
        "modality": "endoscopy",
        "description": "Ung thư dạ dày — Nội soi (carcinoma, normal, ulcer, erosion)",
        "classes": ["carcinoma", "normal", "ulcer", "erosion"],
        "class_dirs": ["carcinoma", "normal", "ulcer", "erosion"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "colorectal",
        "repo_id": "spsahikotha/Colorectal-Cancer-Detection",
        "modality": "colonoscopy",
        "description": "Ung thư đại trực tràng — Nội soi (polyp, normal)",
        "classes": ["polyp", "normal"],
        "class_dirs": ["polyp", "normal"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "cervical",
        "repo_id": "spsahikotha/Cervical-Cancer-Detection",
        "modality": "colposcopy",
        "description": "Ung thư cổ tử cung — Soi cổ tử cung (normal, dysplastic, squamous_cell_carcinoma)",
        "classes": ["normal", "dysplastic", "squamous_cell_carcinoma"],
        "class_dirs": ["normal", "dysplastic", "squamous_cell_carcinoma"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "prostate",
        "repo_id": "spsahikotha/Prostate-Cancer-Detection",
        "modality": "mri",
        "description": "Ung thư tuyến tiền liệt — MRI (malignant, benign)",
        "classes": ["malignant", "benign"],
        "class_dirs": ["malignant", "benign"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
    {
        "cancer_type": "pancreas",
        "repo_id": "spsahikotha/Pancreas-Cancer-Detection",
        "modality": "ct",
        "description": "Ung thư tụy — CT (normal, pdac, neuroendocrine)",
        "classes": ["normal", "pdac", "neuroendocrine"],
        "class_dirs": ["normal", "pdac", "neuroendocrine"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "download_method": "direct",
    },
]


def is_image_file(filename: str, extensions: list[str]) -> bool:
    """Kiểm tra xem file có phải là ảnh không."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in extensions)


def count_images_in_dir(directory: Path, extensions: list[str]) -> int:
    """Đếm số ảnh trong thư mục."""
    if not directory.is_dir():
        return 0
    count = 0
    for f in directory.iterdir():
        if f.is_file() and is_image_file(f.name, extensions):
            count += 1
    return count


def copy_images(src_dir: Path, dst_dir: Path, extensions: list[str], max_per_class: int) -> int:
    """Copy ảnh từ src vào dst, giới hạn max_per_class ảnh mỗi lớp."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in sorted(src_dir.iterdir()):
        if copied >= max_per_class:
            break
        if f.is_file() and is_image_file(f.name, extensions):
            dst = dst_dir / f.name
            try:
                shutil.copy2(f, dst)
                copied += 1
            except (OSError, shutil.Error) as e:
                print(f"  ⚠️  Lỗi copy {f.name}: {e}")
    return copied


def download_with_retry(
    repo_id: str,
    filename: str,
    repo_type: str = "dataset",
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> str | None:
    """Tải file từ HuggingFace với retry logic khi bị rate limit (429)."""
    from huggingface_hub import hf_hub_download

    for attempt in range(max_retries):
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
            )
            return str(cached_path)
        except Exception as e:
            error_msg = str(e).lower()
            # Kiểm tra rate limit
            if "429" in error_msg or "rate limit" in error_msg or "to many requests" in error_msg:
                wait_time = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8, 16, 32
                print(f"     ⏳ Rate limit! Chờ {wait_time}s (lần thử {attempt + 1}/{max_retries})...", flush=True)
                time.sleep(wait_time)
            elif "401" in error_msg or "require login" in error_msg or "private" in error_msg:
                print(f"     🔒 Cần đăng nhập HuggingFace: {filename}", flush=True)
                return None
            else:
                print(f"     ⚠️  Lỗi tải {filename}: {e}", flush=True)
                return None

    print(f"     ❌ Hết lần thử, bỏ qua: {filename}", flush=True)
    return None


def download_from_parquet(
    dataset_info: dict[str, Any],
    output_dir: Path,
    max_images: int,
) -> dict[str, Any]:
    """Tải ảnh từ Parquet dataset dùng datasets library."""
    cancer_type = dataset_info["cancer_type"]
    repo_id = dataset_info["repo_id"]
    classes = dataset_info["classes"]
    image_column = dataset_info.get("image_column", "image")
    label_column = dataset_info.get("label_column", "label")
    label_map = dataset_info.get("label_map", {})

    result = {"per_class": {cls: 0 for cls in classes}, "total": 0}

    try:
        from datasets import load_dataset
    except ImportError:
        print("   ❌ Thiếu datasets library. Cài: pip install datasets", flush=True)
        return result

    try:
        print(f"   📋 Đang load dataset từ Parquet...", flush=True)
        ds = load_dataset(repo_id)

        # Hợp nhất tất cả splits
        all_data = []
        for split_name in ds:
            all_data.extend(ds[split_name])
            print(f"   📊 Split '{split_name}': {len(ds[split_name])} mẫu", flush=True)

        print(f"   📊 Tổng: {len(all_data)} mẫu", flush=True)

        # Giới hạn số lượng
        max_per_class = max_images // len(classes)
        class_counts = {cls: 0 for cls in classes}

        for idx, row in enumerate(all_data):
            if result["total"] >= max_images:
                break

            # Lấy label
            label_id = row.get(label_column)
            if label_id is None:
                continue

            # Map label ID -> class name
            class_name = label_map.get(label_id)
            if class_name is None or class_name not in classes:
                continue

            # Kiểm tra đã đủ chưa
            if class_counts[class_name] >= max_per_class:
                continue

            # Lấy ảnh
            image = row.get(image_column)
            if image is None:
                continue

            # Lưu ảnh
            class_dir = output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img_path = class_dir / f"img_{idx}.png"

            try:
                # PIL Image object từ datasets
                image.save(str(img_path))
                class_counts[class_name] += 1
                result["per_class"][class_name] += 1
                result["total"] += 1
            except Exception as e:
                if idx < 3:
                    print(f"   ⚠️  Lỗi lưu ảnh {idx}: {e}", flush=True)

            if (idx + 1) % 50 == 0:
                print(f"   📥 Đã xử lý {idx + 1} mẫu, {result['total']} ảnh...", flush=True)

    except Exception as e:
        print(f"   ❌ Lỗi load Parquet dataset: {e}", flush=True)

    return result


def download_dataset(
    dataset_info: dict[str, Any],
    base_dir: Path,
    max_images: int = 100,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Tải và tổ chức 1 dataset — dùng list_repo_files + hf_hub_download từng file ảnh.

    Rate limit handling:
    - Delay 0.2s giữa mỗi request để tránh bị chặn
    - Tự động retry với exponential backoff khi gặp 429
    """
    cancer_type = dataset_info["cancer_type"]
    repo_id = dataset_info["repo_id"]
    classes = dataset_info["classes"]
    class_dirs = dataset_info["class_dirs"]
    extensions = dataset_info["image_extensions"]
    requires_login = dataset_info.get("requires_login", False)

    print(f"\n{'='*60}", flush=True)
    print(f"📦 {cancer_type.upper()}", flush=True)
    print(f"   Repo: {repo_id}", flush=True)
    print(f"   Mô tả: {dataset_info['description']}", flush=True)
    print(f"   Modality: {dataset_info['modality']}", flush=True)
    print(f"   Lớp: {', '.join(classes)}", flush=True)
    if requires_login:
        print(f"   ⚠️  Cần đăng nhập HuggingFace để tải", flush=True)
    print(f"{'='*60}", flush=True)

    output_dir = base_dir / cancer_type
    result = {
        "cancer_type": cancer_type,
        "repo_id": repo_id,
        "total_images": 0,
        "per_class": {},
        "status": "pending",
        "error": None,
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"   [DRY RUN] Sẽ tải vào: {output_dir}")
        print(f"   [DRY RUN] Max {max_images} ảnh tổng cộng")
        return result

    # Tính max per class
    max_per_class = max_images // len(classes)

    # Chon phuong thuc tai: direct hoac parquet
    download_method = dataset_info.get("download_method", "direct")

    if download_method == "parquet":
        # Su dung datasets library de load Parquet
        print(f"   📋 Đang tải từ Parquet dataset...", flush=True)
        parquet_result = download_from_parquet(dataset_info, output_dir, max_images)

        if parquet_result["total"] > 0:
            result["total_images"] = parquet_result["total"]
            result["per_class"] = parquet_result["per_class"]
            result["status"] = "success"
            for cls, count in parquet_result["per_class"].items():
                print(f"   ✅ {cls}: {count} ảnh", flush=True)
            print(f"   📊 Tổng: {parquet_result['total']} ảnh", flush=True)
        else:
            result["status"] = "error"
            result["error"] = "No images extracted from parquet"
            print(f"   ❌ Không trích xuất được ảnh từ Parquet", flush=True)

        return result

    # Phuong thuc direct: tai file anh truc tiep
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        print("   ❌ Thiếu huggingface_hub. Cài đặt: pip install huggingface_hub")
        result["status"] = "error"
        result["error"] = "huggingface_hub not installed"
        return result

    try:
        # Bước 1: List tất cả file trong repo
        print(f"   📋 Đang liệt kê file trong repo...", flush=True)
        all_files = list_repo_files(repo_id, repo_type="dataset")

        # Lọc ra file ảnh
        image_files = [f for f in all_files if any(f.lower().endswith(ext) for ext in extensions)]
        print(f"   📊 Tìm thấy {len(image_files)} file ảnh trong repo", flush=True)

        if not image_files:
            result["status"] = "error"
            result["error"] = "No image files found in repo"
            print(f"   ❌ Không tìm thấy file ảnh trong repo", flush=True)
            return result

        # Bước 2: Phân loại ảnh theo class directory
        # Tạo mapping: class_name -> list of file paths
        class_files: dict[str, list[str]] = {cls: [] for cls in classes}
        unclassified: list[str] = []

        for file_path in image_files:
            classified = False
            for class_name, class_dir_name in zip(classes, class_dirs):
                # Kiểm tra nếu file path chứa tên thư mục class
                if class_dir_name.lower() in file_path.lower():
                    class_files[class_name].append(file_path)
                    classified = True
                    break
            if not classified:
                unclassified.append(file_path)

        # Nếu có file chưa phân loại, phân phối đều
        if unclassified and not any(class_files[cls] for cls in classes):
            print(f"   ⚠️  {len(unclassified)} file chưa phân loại, sẽ phân phối đều", flush=True)
            per_class_count = len(unclassified) // len(classes)
            for i, cls in enumerate(classes):
                start = i * per_class_count
                end = start + per_class_count if i < len(classes) - 1 else len(unclassified)
                class_files[cls] = unclassified[start:end]
            unclassified = []

        # Bước 3: Tải từng file ảnh với rate limit handling
        total_copied = 0
        for class_name in classes:
            files_to_download = class_files[class_name][:max_per_class]
            class_output = output_dir / class_name
            class_output.mkdir(parents=True, exist_ok=True)

            copied = 0
            for idx, file_path in enumerate(files_to_download):
                # Delay giữa mỗi request để tránh rate limit
                if idx > 0:
                    time.sleep(0.2)

                cached_path = download_with_retry(repo_id, file_path, "dataset")
                if cached_path is not None:
                    try:
                        dst = class_output / Path(file_path).name
                        shutil.copy2(cached_path, dst)
                        copied += 1
                    except (OSError, shutil.Error) as e:
                        print(f"     ⚠️  Lỗi copy {file_path}: {e}", flush=True)

                if (idx + 1) % 25 == 0:
                    print(f"   📥 {class_name}: đã tải {idx + 1}/{len(files_to_download)}...", flush=True)

            result["per_class"][class_name] = copied
            total_copied += copied
            print(f"   ✅ {class_name}: {copied} ảnh", flush=True)

            if total_copied >= max_images:
                break

        result["total_images"] = total_copied
        result["status"] = "success"
        print(f"   📊 Tổng: {total_copied} ảnh", flush=True)

    except Exception as e:
        error_msg = str(e)
        result["status"] = "error"
        result["error"] = error_msg
        print(f"   ❌ Lỗi: {error_msg}")

        if "401" in error_msg or "require login" in error_msg.lower() or "private" in error_msg.lower():
            print(f"   💡 Cần đăng nhập HuggingFace:")
            print(f"      huggingface-cli login")
            print(f"      Hoặc đặt biến môi trường HF_TOKEN")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dataset ảnh y khoa cho 11 loại ung thư")
    parser.add_argument(
        "--cancer",
        type=str,
        default=None,
        help="Chỉ tải các loại ung thư này (phân cách bằng dấu phẩy). VD: brain,breast,lung",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Thư mục đầu ra (mặc định: data/datasets)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Số ảnh tối đa mỗi loại ung thư (mặc định: 50)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Xem trước không tải",
    )
    args = parser.parse_args()

    # Xác định các loại ung thư cần tải
    if args.cancer:
        cancer_types = [c.strip() for c in args.cancer.split(",")]
        datasets_to_download = [d for d in DATASETS if d["cancer_type"] in cancer_types]
        if not datasets_to_download:
            print(f"❌ Không tìm thấy loại ung thư: {args.cancer}")
            print(f"   Các loại có sẵn: {', '.join(d['cancer_type'] for d in DATASETS)}")
            sys.exit(1)
    else:
        datasets_to_download = DATASETS

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🏥 OncoVision — Tải Dataset Ảnh Y Khoa")
    print("=" * 60)
    print(f"📁 Đầu ra: {base_dir.resolve()}")
    print(f"🔢 Max ảnh mỗi loại: {args.max_images}")
    print(f"📦 Số loại ung thư: {len(datasets_to_download)}")
    if args.dry_run:
        print("🔍 CHẾ ĐỘ: Xem trước (không tải)")
    print()

    # Tải từng dataset
    results = []
    for dataset_info in datasets_to_download:
        result = download_dataset(dataset_info, base_dir, args.max_images, args.dry_run)
        results.append(result)

    # Tổng kết
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)

    total_images = 0
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "⏭️" if r["status"] == "dry_run" else "❌"
        print(f"  {status_icon} {r['cancer_type']:15s} {r['total_images']:5d} ảnh  ({r['status']})")
        if r.get("error"):
            print(f"     Lỗi: {r['error']}")
        total_images += r["total_images"]

    print(f"\n  Tổng: {total_images} ảnh từ {len(results)} loại ung thư")

    # Lưu manifest
    manifest_path = base_dir / "dataset_manifest.json"
    manifest = {
        "generated_at": "2026-11-13",
        "total_images": total_images,
        "cancers": results,
        "max_images_per_cancer": args.max_images,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Manifest: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
