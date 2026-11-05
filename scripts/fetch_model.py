"""Tải model ung thư từ HuggingFace/GitHub, kiểm runtime rồi đặt đúng hợp đồng.

Chỉ chạy tay lúc phát triển — app cấm tải mạng lúc runtime (medical/network_policy.py).

Ví dụ:
    # Lấy toàn bộ model có sẵn cho các nhóm ung thư (1 lệnh cho máy mới)
    python scripts/fetch_model.py --all

    # Thêm 1 model lẻ
    python scripts/fetch_model.py amcm/brain-tumor-yolo11n --file brain_tumor_yolo11n_ft.pt ^
        --target brain --kind detector --modality mri
    python scripts/fetch_model.py owner/repo --file best.pt --target kidney --kind detector ^
        --modality ct --repo-type github
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical.cancer_model_registry import (
    KEY_TO_DIR,
    probe_runtime_capability,
    target_model_paths,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Neo vào gốc repo: chạy từ thư mục khác không tạo ra cây models/... lồng nhau.
DEFAULT_CATALOG = PROJECT_ROOT / "models" / "model_catalog.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_destination(base: Path, target: str, kind: str, modality: str | None) -> Path:
    if target not in KEY_TO_DIR:
        raise ValueError(f"Nhóm ung thư không hợp lệ: {target!r}; chọn {', '.join(sorted(KEY_TO_DIR))}")
    return target_model_paths(target, base=base, modality=modality)[kind]


def _pick_file(repo: str, repo_type: str) -> str:
    from huggingface_hub import HfApi

    files = [
        f for f in HfApi().list_repo_files(repo, repo_type=repo_type) if f.lower().endswith((".pt", ".onnx"))
    ]
    if not files:
        raise SystemExit(f"Không tìm thấy file .pt nào trong {repo} ({repo_type}).")
    if len(files) > 1:
        raise SystemExit("Repo có nhiều file .pt, hãy chỉ rõ --file:\n  " + "\n  ".join(files))
    return files[0]


def _download_github(repo: str, file: str) -> tuple[Path, str]:
    """Tải file thô từ GitHub; ưu tiên media (giải LFS) rồi raw."""
    attempts = []
    for base in (
        "https://media.githubusercontent.com/media",
        "https://raw.githubusercontent.com",
    ):
        url = f"{base}/{repo}/HEAD/{file}"
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "oncovision-dev"})
            with urllib.request.urlopen(request, timeout=180) as response:
                data = response.read()
        except Exception as exc:  # gom mọi lỗi mạng để thử base kế tiếp
            attempts.append(f"{base}: {exc}")
            continue
        if data[:128].startswith(b"version https://git-lfs"):
            attempts.append(f"{base}: còn LFS pointer")
            continue
        target = Path(tempfile.gettempdir()) / f"oncovision_fetch_{Path(file).name}"
        target.write_bytes(data)
        return target, "github"
    raise SystemExit("Tải GitHub thất bại:\n  " + "\n  ".join(attempts))


def download(repo: str, file: str | None, repo_type: str | None) -> tuple[Path, str]:
    if repo_type == "github" or repo_type == "gh":
        if not file:
            raise SystemExit("Nguồn GitHub cần chỉ rõ --file (không liệt kê được thư mục).")
        return _download_github(repo, file)

    from huggingface_hub import hf_hub_download

    attempts = []
    for kind in ((repo_type,) if repo_type else ("model", "dataset")):
        try:
            return Path(hf_hub_download(repo, file or _pick_file(repo, kind), repo_type=kind)), kind
        except Exception as exc:
            attempts.append(f"{kind}: {exc}")
    raise SystemExit("Tải thất bại:\n  " + "\n  ".join(attempts))


def _install(
    source: Path,
    *,
    repo: str,
    repo_type: str,
    file: str,
    target: str,
    kind: str,
    modality: str | None,
    base: Path,
    expected_sha: str | None = None,
    force: bool = False,
) -> Path:
    ok, reason = probe_runtime_capability(source, kind)
    if not ok:
        raise SystemExit(f"Không dùng được {source.name}: {reason}")
    if expected_sha and _sha256(source) != expected_sha:
        raise SystemExit(f"sha256 không khớp cho {source.name} (tải thiếu/hỏng) — không đặt vào repo.")

    destination = resolve_destination(base, target, kind, modality)
    # Nguồn .onnx phải giữ đuôi .onnx (ultralytics chọn backend theo phần mở rộng).
    if destination.suffix.lower() != source.suffix.lower():
        destination = destination.with_suffix(source.suffix.lower())
    if destination.exists() and not force:
        raise SystemExit(f"Đã có {destination}; dùng --force để ghi đè.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    sidecar = destination.with_suffix(".source.json")
    sidecar.write_text(
        json.dumps(
            {
                "repo": repo,
                "repo_type": repo_type,
                "file": file,
                "sha256": _sha256(source),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "probe_kind": kind,
                "probe_ok": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def _catalog_destination(base: Path, entry: dict) -> Path:
    destination = resolve_destination(base, entry["target"], entry["kind"], entry.get("modality"))
    suffix = Path(entry["file"]).suffix.lower()
    return destination.with_suffix(suffix) if destination.suffix.lower() != suffix else destination


def fetch_all(catalog_path: Path, base: Path, force: bool) -> int:
    if not catalog_path.is_file():
        raise SystemExit(f"Không thấy catalog: {catalog_path}")
    entries = json.loads(catalog_path.read_text(encoding="utf-8")).get("entries", [])
    if not entries:
        print(f"No qualified models in catalog: {catalog_path}; nothing downloaded.")
        return 0
    print(f"Catalog: {catalog_path} — {len(entries)} model\n")

    placed, skipped, failed = 0, 0, 0
    for entry in entries:
        name = f"{entry['target']}/{entry.get('modality') or '-'}/{entry['kind']}"
        destination = _catalog_destination(base, entry)
        if destination.is_file() and not force:
            if entry.get("sha256") is None or _sha256(destination) == entry["sha256"]:
                print(f"  bỏ qua  {name}: đã có {destination.name}")
                skipped += 1
                continue
            print(f"  tải lại {name}: sha cũ khác catalog")
        try:
            source, repo_type = download(entry["repo"], entry["file"], entry.get("repo_type"))
            installed = _install(
                source,
                repo=entry["repo"],
                repo_type=repo_type,
                file=entry["file"],
                target=entry["target"],
                kind=entry["kind"],
                modality=entry.get("modality"),
                base=base,
                expected_sha=entry.get("sha256"),
                force=force,
            )
        except SystemExit as exc:
            print(f"  LỖI     {name}: {exc}")
            failed += 1
            continue
        print(f"  OK      {name}: -> {installed}")
        placed += 1

    print(f"\nXong: {placed} tải mới, {skipped} bỏ qua, {failed} lỗi.")
    if failed:
        return 1
    print("Chạy: python run_medical.py status")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("repo", nargs="?", help="repo HuggingFace dạng owner/name (bỏ trống khi dùng --all)")
    parser.add_argument("--file", help="file .pt/.onnx trong repo; bỏ trống khi repo chỉ có một file")
    parser.add_argument("--target", help="nhóm ung thư, vd brain, lung, kidney")
    parser.add_argument("--kind", choices=("classifier", "detector"), help="loại model")
    parser.add_argument("--modality", default=None, help="vd mri, ct; bỏ trống = đặt ở cấp cơ quan")
    parser.add_argument("--repo-type", choices=("model", "dataset", "github"), default=None)
    parser.add_argument("--base", type=Path, default=PROJECT_ROOT / "models/pretrained/cancers")
    parser.add_argument("--force", action="store_true", help="ghi đè file đích nếu đã có")
    parser.add_argument("--all", action="store_true", help="tải mọi model trong catalog (máy mới)")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="đường dẫn catalog JSON")
    args = parser.parse_args(argv)

    if args.all:
        return fetch_all(args.catalog, args.base, args.force)

    if not args.repo or not args.target or not args.kind:
        parser.error("cần <repo> --target --kind, hoặc dùng --all")

    source, source_repo_type = download(args.repo, args.file, args.repo_type)
    destination = _install(
        source,
        repo=args.repo,
        repo_type=source_repo_type,
        file=args.file or source.name,
        target=args.target,
        kind=args.kind,
        modality=args.modality,
        base=args.base,
        force=args.force,
    )
    print(f"OK: {source.name} -> {destination}")
    print(f"Ghi nguồn: {destination.with_suffix('.source.json')}")
    if args.modality is None:
        print(
            "Lưu ý: ảnh có modality xác định sẽ bắt model trong thư mục modality "
            "(vd <target>/mri/). Nếu status báo định vị thiếu, chạy lại với --modality."
        )
    print("Chạy: python run_medical.py status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
