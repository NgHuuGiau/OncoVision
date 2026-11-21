#!/usr/bin/env python
"""Test modality detection và cancer classification với ảnh không liên quan và ảnh y khoa.

Chạy: python tests/test_modality_detection.py
"""
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
from PIL import Image

# Thêm project root vào path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from medical.dataset import (
    _extract_pixel_features,
    guess_cancer_from_pixels,
    guess_modality_from_pixels,
    infer_medical_upload_context,
)

output_dir = project_root / "output" / "test_images"
output_dir.mkdir(parents=True, exist_ok=True)


def create_random_noise_image(width=224, height=224):
    """Tạo ảnh noise ngẫu nhiên."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "random_noise.jpg"
    img.save(path)
    return path


def create_solid_color_image(color=(128, 128, 128), width=224, height=224):
    """Tạo ảnh màu đơn."""
    arr = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "solid_gray.jpg"
    img.save(path)
    return path


def create_gradient_image(width=224, height=224):
    """Tạo ảnh gradient."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(width):
        arr[:, i, :] = int(255 * i / width)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "gradient.jpg"
    img.save(path)
    return path


def create_cat_like_image(width=300, height=300):
    """Tạo ảnh giống ảnh chó mèo (nhiều màu, texture phức tạp)."""
    arr = np.random.randint(50, 220, (height, width, 3), dtype=np.uint8)
    # Thêm các vùng màu khác nhau (giống lông động vật)
    for _ in range(20):
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        color = np.random.randint(100, 200, 3, dtype=np.uint8)
        arr[y1:y2, x1:x2, :] = color
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "cat_like_texture.jpg"
    img.save(path, quality=85)
    return path


def create_landscape_like_image(width=400, height=300):
    """Tạo ảnh giống phong cảnh (trời xanh, đất nâu, cỏ xanh)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Trời xanh
    arr[0:height//3, :, :] = [135, 206, 235]  # Sky blue
    # Cỏ xanh
    arr[height//3:2*height//3, :, :] = [34, 139, 34]  # Forest green
    # Đất nâu
    arr[2*height//3:, :, :] = [139, 69, 19]  # Saddle brown
    # Thêm noise nhẹ
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "landscape_like.jpg"
    img.save(path, quality=85)
    return path


def create_selfie_like_image(width=300, height=400):
    """Tạo ảnh giống selfie (màu da, nền mờ)."""
    arr = np.full((height, width, 3), [200, 180, 170], dtype=np.uint8)  # Nền màu da
    # Khuôn mặt (tròn)
    cy, cx = height // 2, width // 2
    ry, rx = 80, 60
    y, x = np.ogrid[:height, :width]
    mask = ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1
    arr[mask] = [224, 180, 150]  # Màu da sáng hơn
    # Mắt
    for dx in [-20, 20]:
        eye_mask = ((y - (cy - 15)) / 8) ** 2 + ((x - (cx + dx)) / 6) ** 2 <= 1
        arr[eye_mask] = [50, 50, 50]
    # Môi
    mouth_mask = ((y - (cy + 25)) / 6) ** 2 + ((x - cx) / 15) ** 2 <= 1
    arr[mouth_mask] = [180, 80, 80]
    # Thêm noise
    noise = np.random.randint(-10, 10, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "selfie_like.jpg"
    img.save(path, quality=85)
    return path


def create_black_image(width=224, height=224):
    """Tạo ảnh đen hoàn toàn."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "black.jpg"
    img.save(path)
    return path


def create_white_image(width=224, height=224):
    """Tạo ảnh trắng hoàn toàn."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "white.jpg"
    img.save(path)
    return path


def create_ct_like_image(width=224, height=224):
    """Tạo ảnh giống CT scan (nền đen, vùng xám)."""
    arr = np.zeros((height, width), dtype=np.uint8)
    # Vùng cơ thể (tròn)
    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    body_mask = ((y - cy) / (height * 0.4)) ** 2 + ((x - cx) / (width * 0.4)) ** 2 <= 1
    arr = np.where(body_mask, np.random.randint(80, 150, (height, width)), 0).astype(np.uint8)
    # Thêm noise
    arr = np.clip(arr.astype(np.int16) + np.random.randint(-10, 10, arr.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    path = output_dir / "ct_like.jpg"
    img.save(path)
    return path


def create_ultrasound_like_image(width=224, height=224):
    """Tạo ảnh giống siêu âm (texture grainy, nhiều mức xám)."""
    arr = np.random.randint(30, 180, (height, width), dtype=np.uint8)
    # Thêm các vùng sáng/tối (giống nodule)
    for _ in range(5):
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        brightness = np.random.randint(150, 220)
        arr[y1:y2, x1:x2] = brightness
    img = Image.fromarray(arr, mode="L")
    path = output_dir / "ultrasound_like.jpg"
    img.save(path)
    return path


def create_endoscopy_like_image(width=300, height=300):
    """Tạo ảnh giống nội soi (màu đỏ hồng, texture)."""
    arr = np.random.randint(150, 220, (height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + 30, 0, 255)  # Đỏ nhiều hơn
    arr[:, :, 2] = np.clip(arr[:, :, 2] - 50, 0, 255)  # Xanh dương ít hơn
    img = Image.fromarray(arr, mode="RGB")
    path = output_dir / "endoscopy_like.jpg"
    img.save(path, quality=85)
    return path


def test_image(name, path, expected_modality=None, expected_cancer=None):
    """Test 1 ảnh và in kết quả."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"File: {path}")
    print(f"{'='*70}")

    # Pixel features
    features = _extract_pixel_features(path)
    if features:
        print("\n[Pixel Features]")
        print(f"  Mean: {features['mean']:.1f}")
        print(f"  Std: {features['std']:.1f}")
        print(f"  Black ratio: {features['black_ratio']:.3f}")
        print(f"  Local var: {features['local_var']:.1f}")
        print(f"  Channel diff: {features['channel_diff']:.1f}")
    else:
        print("\n[Pixel Features] Không trích xuất được")

    # Modality detection
    modality = guess_modality_from_pixels(path)
    print("\n[Modality Detection]")
    print(f"  Kết quả: {modality}")
    if expected_modality:
        status = "✅" if modality == expected_modality else "❌"
        print(f"  {status} Kỳ vọng: {expected_modality}")

    # Cancer detection
    cancer_key, cancer_modality = guess_cancer_from_pixels(path)
    print("\n[Cancer Detection]")
    print(f"  Cancer key: {cancer_key}")
    print(f"  Cancer modality: {cancer_modality}")
    if expected_cancer:
        status = "✅" if cancer_key == expected_cancer else "❌"
        print(f"  {status} Kỳ vọng: {expected_cancer}")

    # Full context inference
    target_key, inferred_modality = infer_medical_upload_context(path)
    print("\n[Full Context Inference]")
    print(f"  Target key: {target_key}")
    print(f"  Modality: {inferred_modality}")

    return {
        "name": name,
        "modality": modality,
        "cancer_key": cancer_key,
        "cancer_modality": cancer_modality,
        "target_key": target_key,
        "inferred_modality": inferred_modality,
        "features": features,
    }


def main():
    print("=" * 70)
    print("TEST: Modality Detection & Cancer Classification")
    print("=" * 70)

    results = []

    # === Phần 1: Ảnh KHÔNG liên quan y khoa ===
    print("\n\n" + "🔴" * 20)
    print("PHẦN 1: Ảnh KHÔNG liên quan y khoa")
    print("🔴" * 20)

    # 1. Ảnh noise ngẫu nhiên
    path = create_random_noise_image()
    results.append(test_image("Noise ngẫu nhiên", path))

    # 2. Ảnh màu đơn
    path = create_solid_color_image()
    results.append(test_image("Màu đơn (xám)", path))

    # 3. Ảnh gradient
    path = create_gradient_image()
    results.append(test_image("Gradient", path))

    # 4. Ảnh giống chó mèo (texture phức tạp)
    path = create_cat_like_image()
    results.append(test_image("Chó mèo (texture)", path))

    # 5. Ảnh giống phong cảnh
    path = create_landscape_like_image()
    results.append(test_image("Phong cảnh", path))

    # 6. Ảnh giống selfie
    path = create_selfie_like_image()
    results.append(test_image("Selfie", path))

    # 7. Ảnh đen
    path = create_black_image()
    results.append(test_image("Đen hoàn toàn", path))

    # 8. Ảnh trắng
    path = create_white_image()
    results.append(test_image("Trắng hoàn toàn", path))

    # === Phần 2: Ảnh GIỐNG y khoa (tạo bằng code) ===
    print("\n\n" + "🟢" * 20)
    print("PHẦN 2: Ảnh GIỐNG y khoa (tạo bằng code)")
    print("🟢" * 20)

    # 9. Ảnh giống CT
    path = create_ct_like_image()
    results.append(test_image("CT-like (tạo)", path, expected_modality="CT"))

    # 10. Ảnh giống siêu âm
    path = create_ultrasound_like_image()
    results.append(test_image("Siêu âm-like (tạo)", path, expected_modality="Siêu âm"))

    # 11. Ảnh giống nội soi
    path = create_endoscopy_like_image()
    results.append(test_image("Nội soi-like (tạo)", path, expected_modality="Nội soi"))

    # === Phần 3: Ảnh y khoa thực tế (nếu có) ===
    print("\n\n" + "🔵" * 20)
    print("PHẦN 3: Ảnh y khoa thực tế (từ data/ hoặc models/)")
    print("🔵" * 20)

    # Tìm ảnh mẫu trong data/
    data_dir = project_root / "data"
    sample_images = list(data_dir.rglob("*.jpg")) + list(data_dir.rglob("*.png")) + list(data_dir.rglob("*.jpeg"))
    if sample_images:
        for img_path in sample_images[:5]:
            results.append(test_image(f"Sample: {img_path.name}", img_path))
    else:
        print("Không tìm thấy ảnh mẫu trong data/")

    # === Tổng kết ===
    print("\n\n" + "=" * 70)
    print("TỔNG KẾT")
    print("=" * 70)

    non_medical = [r for r in results if "PHẦN 1" in str(r.get("name", "")) or any(x in r["name"] for x in ["Noise", "Màu đơn", "Gradient", "Chó mèo", "Phong cảnh", "Selfie", "Đen", "Trắng"])]
    medical_like = [r for r in results if "PHẦN 2" in str(r.get("name", "")) or any(x in r["name"] for x in ["CT-like", "Siêu âm-like", "Nội soi-like"])]

    print(f"\nẢnh không liên quan: {len(non_medical)}")
    for r in non_medical:
        modality = r["modality"]
        cancer = r["cancer_key"]
        target = r["target_key"]
        # Đánh giá: ảnh không liên quan nên trả về None hoặc uncertain
        if modality is None and cancer is None:
            status = "✅ IDEAL"
        elif modality is None and cancer is not None:
            status = "⚠️ WARN (cancer không None)"
        elif modality is not None and cancer is None:
            status = "⚠️ WARN (modality không None)"
        else:
            status = "❌ FAIL (cả 2 không None)"
        print(f"  {status} | {r['name']}: modality={modality}, cancer={cancer}, target={target}")

    print(f"\nẢnh giống y khoa: {len(medical_like)}")
    for r in medical_like:
        modality = r["modality"]
        cancer = r["cancer_key"]
        print(f"  modality={modality}, cancer={cancer}, target={r['target_key']}")

    # Lưu kết quả JSON
    results_file = output_dir / "test_results.json"
    serializable = []
    for r in results:
        sr = dict(r)
        if sr.get("features"):
            sr["features"] = {k: float(v) for k, v in sr["features"].items()}
        serializable.append(sr)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\nKết quả đã lưu: {results_file}")


if __name__ == "__main__":
    main()
