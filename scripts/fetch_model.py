"""Tải model ung thư từ HuggingFace, kiểm runtime rồi đặt đúng hợp đồng.

Chỉ chạy tay lúc phát triển — app cấm tải mạng lúc runtime (medical/network_policy.py).

Ví dụ:
    python scripts/fetch_model.py amcm/brain-tumor-yolo11n --file brain_tumor_yolo11n_ft.pt ^
        --target brain --kind detector --modality mri
    python scripts/fetch_model.py owner/repo --file weights.pt --target cervical --kind classifier
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medical.cancer_model_registry import (
    KEY_TO_DIR,
    probe_runtime_capability,
    target_model_paths,
)


def resolve_destination(base: Path, target: str, kind: str, modality: str | None) -> Path:
    if target not in KEY_TO_DIR:
        raise ValueError(f"Nhóm ung thư không hợp lệ: {target!r}; chọn {', '.join(sorted(KEY_TO_DIR))}")
    return target_model_paths(target, base=base, modality=modality)[kind]


def _pick_file(repo: str, repo_type: str) -> str:
    from huggingface_hub import HfApi

    files = [f for f in HfApi().list_repo_files(repo, repo_type=repo_type) if f.lower().endswith(".pt")]
    if not files:
        raise SystemExit(f"Không tìm thấy file .pt nào trong {repo} ({repo_type}).")
    if len(files) > 1:
        raise SystemExit("Repo có nhiều file .pt, hãy chỉ rõ --file:\n  " + "\n  ".join(files))
    return files[0]


def download(repo: str, file: str | None, repo_type: str | None) -> tuple[Path, str]:
    from huggingface_hub import hf_hub_download

    attempts = []
    for kind in ((repo_type,) if repo_type else ("model", "dataset")):
        try:
            return Path(hf_hub_download(repo, file or _pick_file(repo, kind), repo_type=kind)), kind
        except Exception as exc:
            attempts.append(f"{kind}: {exc}")
    raise SystemExit("Tải thất bại:\n  " + "\n  ".join(attempts))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("repo", help="repo HuggingFace dạng owner/name")
    parser.add_argument("--file", help="file .pt trong repo; bỏ trống khi repo chỉ có một file .pt")
    parser.add_argument("--target", required=True, choices=sorted(KEY_TO_DIR))
    parser.add_argument("--kind", required=True, choices=("classifier", "detector"))
    parser.add_argument("--modality", default=None, help="vd mri, ct; bỏ trống = đặt ở cấp cơ quan")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default=None)
    parser.add_argument("--base", type=Path, default=Path("models/pretrained/cancers"))
    parser.add_argument("--force", action="store_true", help="ghi đè file đích nếu đã có")
    args = parser.parse_args(argv)

    destination = resolve_destination(args.base, args.target, args.kind, args.modality)
    if destination.exists() and not args.force:
        raise SystemExit(f"Đã có {destination}; dùng --force để ghi đè.")

    source, source_repo_type = download(args.repo, args.file, args.repo_type)
    ok, reason = probe_runtime_capability(source, args.kind)
    if not ok:
        raise SystemExit(f"Không dùng được {source.name}: {reason}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    sidecar = destination.with_suffix(".source.json")
    sidecar.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "repo_type": source_repo_type,
                "file": args.file or source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "probe_kind": args.kind,
                "probe_ok": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"OK: {source.name} -> {destination}")
    print(f"Ghi nguồn: {sidecar}")
    if args.modality is None:
        print(
            "Lưu ý: ảnh có modality xác định sẽ bắt model trong thư mục modality "
            "(vd <target>/mri/). Nếu status báo định vị thiếu, chạy lại với --modality."
        )
    print("Chạy: python run_medical.py status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
