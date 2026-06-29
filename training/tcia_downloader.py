from __future__ import annotations

import argparse
import http.client
import json
import multiprocessing as mp
import re
import csv
import time
import zipfile
from io import StringIO
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

try:
    from training._training_bootstrap import ensure_project_root_on_path
except ModuleNotFoundError:
    from _training_bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from medical.cancer_dataset_registry import common_cancer_dataset_source_dicts


DEFAULT_COLLECTIONS_FILE = Path("training/tcia_collections_5.json")
TARGET_TOTAL_IMAGES = 25000
DOWNLOAD_RETRIES = 1
DOWNLOAD_BACKOFF_SECONDS = 2.0
DOWNLOAD_TIMEOUT_SECONDS = 90
DOWNLOAD_HARD_TIMEOUT_SECONDS = 75
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
MAX_COLLECTION_CONSECUTIVE_FAILURES = 10
COLLECTION_IMAGE_TARGET_OVERRIDES = {
    "CBIS-DDSM": 200,
}
SERIES_DOWNLOAD_STRATEGY = "medium_first_25_to_256"


TCIA_API_ROOT = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
TCIA_DOWNLOAD_ROOT = Path("dataset/medical/tcia")
TCIA_STATUS_PATH = TCIA_DOWNLOAD_ROOT / "download_status.json"
TCIA_LOG_ROOT = TCIA_DOWNLOAD_ROOT / "logs"


@dataclass(frozen=True)
class TciaCollectionSpec:
    cancer_type: str
    source_name: str
    collection_name: str
    notes: str


def build_tcia_collection_specs() -> list[TciaCollectionSpec]:
    specs: list[TciaCollectionSpec] = []
    for item in common_cancer_dataset_source_dicts():
        if not item["source_name"].startswith("TCIA"):
            continue
        collection_name = item["source_name"].replace("TCIA ", "").strip()
        specs.append(
            TciaCollectionSpec(
                cancer_type=item["cancer_type"],
                source_name=item["source_name"],
                collection_name=collection_name,
                notes=item["notes"],
            )
        )
    return specs


def build_collection_root(collection_name: str) -> Path:
    return TCIA_DOWNLOAD_ROOT / collection_name.replace("/", "_").replace(" ", "_")


def build_collection_log_path(collection_name: str) -> Path:
    return TCIA_LOG_ROOT / f"{collection_name.replace('/', '_').replace(' ', '_')}.log"


def read_collection_log(collection_name: str, *, tail_lines: int | None = None) -> dict[str, object]:
    log_path = build_collection_log_path(collection_name)
    if not log_path.exists():
        return {
            "collection_name": collection_name,
            "log_path": str(log_path),
            "exists": False,
            "lines": [],
        }
    lines = log_path.read_text(encoding="utf-8").splitlines()
    if tail_lines is not None and tail_lines > 0:
        lines = lines[-tail_lines:]
    return {
        "collection_name": collection_name,
        "log_path": str(log_path),
        "exists": True,
        "lines": lines,
    }


def build_collection_manifest(spec: TciaCollectionSpec) -> dict[str, object]:
    return {
        "source": spec.source_name,
        "cancer_type": spec.cancer_type,
        "collection_name": spec.collection_name,
        "api_root": TCIA_API_ROOT,
        "series_query": f"{TCIA_API_ROOT}/getSeries?Collection={quote(spec.collection_name)}",
        "notes": spec.notes,
    }


def write_tcia_manifest() -> Path:
    TCIA_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "TCIA",
        "api_root": TCIA_API_ROOT,
        "collections": [build_collection_manifest(spec) for spec in build_tcia_collection_specs()],
    }
    manifest_path = TCIA_DOWNLOAD_ROOT / "collections_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def _append_collection_log(collection_name: str, lines: list[str]) -> Path:
    TCIA_LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = build_collection_log_path(collection_name)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def load_collection_list(collections_file: Path | None = None) -> list[str]:
    if collections_file is None:
        collections_file = DEFAULT_COLLECTIONS_FILE
    if collections_file.exists():
        payload = json.loads(collections_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            values = payload.get("collections", [])
        else:
            values = payload
        result: list[str] = []
        for item in values:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                name = item.get("collection_name") or item.get("source_name")
                if isinstance(name, str):
                    result.append(name.replace("TCIA ", "").strip())
        return result
    return [spec.collection_name for spec in build_tcia_collection_specs()]


def load_collection_metadata(collections_file: Path | None = None) -> list[dict[str, str]]:
    if collections_file is None:
        collections_file = DEFAULT_COLLECTIONS_FILE
    if not collections_file.exists():
        return [
            {"collection_name": spec.collection_name, "cancer_type": spec.cancer_type}
            for spec in build_tcia_collection_specs()
        ]
    payload = json.loads(collections_file.read_text(encoding="utf-8"))
    values = payload.get("collections", []) if isinstance(payload, dict) else payload
    metadata: list[dict[str, str]] = []
    for item in values:
        if not isinstance(item, dict):
            continue
        collection_name = item.get("collection_name") or item.get("source_name")
        cancer_type = item.get("cancer_type")
        if not isinstance(collection_name, str) or not isinstance(cancer_type, str):
            continue
        metadata.append(
            {
                "collection_name": collection_name.replace("TCIA ", "").strip(),
                "cancer_type": cancer_type,
            }
        )
    return metadata


def _priority_order(collection_names: list[str]) -> list[str]:
    priority = [
        "CBIS-DDSM",
        "TCGA-BRCA",
        "NSCLC-Radiomics",
        "TCGA-LUAD",
        "TCGA-COAD",
        "TCGA-READ",
        "TCGA-LIHC",
        "TCGA-STAD",
    ]
    ordered = [name for name in priority if name in collection_names]
    ordered.extend([name for name in collection_names if name not in ordered])
    return ordered


def _parse_series_links(html: str) -> list[str]:
    links = re.findall(r"https://[^\"'\s>]+", html)
    series_uids = re.findall(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", html)
    parsed = [link for link in links if "getImage" in link or "api" in link or "download" in link]
    parsed.extend(
        f"{TCIA_API_ROOT}/query/getImage?SeriesInstanceUID={quote(series_uid)}"
        for series_uid in series_uids
    )
    deduped: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _fetch_series_records(collection_name: str) -> list[dict[str, object]]:
    query_urls = [
        f"{TCIA_API_ROOT}/getSeries?Collection={quote(collection_name)}",
        f"{TCIA_API_ROOT}/query/getSeries?Collection={quote(collection_name)}",
    ]
    last_error: str | None = None
    for url in query_urls:
        try:
            payload = urlopen(url, timeout=60).read().decode("utf-8", errors="ignore")
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            last_error = f"ERROR::{collection_name}::{exc}"
            continue
        try:
            rows = json.loads(payload)
        except json.JSONDecodeError:
            rows = None
        if isinstance(rows, list):
            records: list[dict[str, object]] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                series_uid = row.get("SeriesInstanceUID") or row.get("SeriesUID") or row.get("SeriesInstanceUid")
                if not isinstance(series_uid, str) or not series_uid:
                    continue
                image_count_value = row.get("ImageCount") or row.get("imageCount") or row.get("Count") or 1
                try:
                    image_count = int(image_count_value)
                except (TypeError, ValueError):
                    image_count = 1
                records.append(
                    {
                        "series_uid": series_uid,
                        "download_url": f"{TCIA_API_ROOT}/getImage?SeriesInstanceUID={quote(series_uid)}",
                        "image_count": image_count,
                        "modality": row.get("Modality", ""),
                        "patient_id": row.get("PatientID", ""),
                    }
                )
            if records:
                return records
        links = _parse_series_links(payload)
        if links:
            return [{"series_uid": f"parsed-{index}", "download_url": link, "image_count": 1} for index, link in enumerate(links)]
        for delimiter in ("\t", ",", ";"):
            reader = csv.DictReader(StringIO(payload), delimiter=delimiter)
            records = []
            for row in reader:
                series_uid = (
                    row.get("SeriesInstanceUID")
                    or row.get("SeriesUID")
                    or row.get("SeriesInstanceUid")
                    or row.get("Series Instance UID")
                    or row.get("Series Instance Uid")
                )
                if not series_uid:
                    continue
                image_count_value = row.get("ImageCount") or row.get("imageCount") or row.get("Count") or 1
                try:
                    image_count = int(image_count_value)
                except (TypeError, ValueError):
                    image_count = 1
                records.append(
                    {
                        "series_uid": series_uid,
                        "download_url": f"{TCIA_API_ROOT}/getImage?SeriesInstanceUID={quote(series_uid)}",
                        "image_count": image_count,
                    }
                )
            if records:
                return records
    return [{"error": last_error or f"ERROR::{collection_name}::No TCIA series links could be parsed"}]


def _fetch_series_links(collection_name: str) -> list[str]:
    records = _fetch_series_records(collection_name)
    links: list[str] = []
    for record in records:
        error = record.get("error")
        if isinstance(error, str):
            return [error]
        download_url = record.get("download_url")
        if isinstance(download_url, str):
            links.append(download_url)
    return links


def _download_once(url: str, target: Path) -> tuple[bool, str | None]:
    tmp_target = target.with_suffix(target.suffix + ".part")
    try:
        if tmp_target.exists():
            tmp_target.unlink()
        with urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, tmp_target.open("wb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
        if not tmp_target.exists() or tmp_target.stat().st_size == 0:
            if tmp_target.exists():
                tmp_target.unlink()
            return False, "Downloaded file was empty."
        tmp_target.replace(target)
        return True, None
    except (HTTPError, URLError, TimeoutError, OSError, http.client.IncompleteRead) as exc:
        if tmp_target.exists():
            tmp_target.unlink()
        return False, str(exc)


def _download_worker(url: str, target: Path, result_queue) -> None:
    result_queue.put(_download_once(url, target))


def _download_once_with_deadline(url: str, target: Path, hard_timeout_seconds: int = DOWNLOAD_HARD_TIMEOUT_SECONDS) -> tuple[bool, str | None]:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_download_worker, args=(url, target, result_queue))
    process.start()
    process.join(hard_timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        tmp_target = target.with_suffix(target.suffix + ".part")
        if tmp_target.exists():
            tmp_target.unlink()
        return False, f"hard timeout after {hard_timeout_seconds}s"
    if process.exitcode != 0:
        tmp_target = target.with_suffix(target.suffix + ".part")
        if tmp_target.exists():
            tmp_target.unlink()
        return False, f"download worker exited with code {process.exitcode}"
    try:
        return result_queue.get_nowait()
    except Exception as exc:
        return False, f"download worker returned no result: {exc}"


def _safe_download(url: str, target: Path, *, use_process: bool = True) -> tuple[bool, str | None]:
    last_error: str | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        if use_process:
            ok, error = _download_once_with_deadline(url, target)
        else:
            ok, error = _download_once(url, target)
        if ok:
            return True, None
        last_error = f"attempt {attempt}: {error or 'unknown error'}"
        time.sleep(DOWNLOAD_BACKOFF_SECONDS * attempt)
    return False, last_error


def _safe_series_file_stem(collection_name: str, series_uid: str) -> str:
    safe_collection = collection_name.replace(" ", "_").replace("/", "_")
    safe_uid = re.sub(r"[^A-Za-z0-9_.-]+", "_", series_uid).replace(".", "_")
    return f"{safe_collection}_{safe_uid}"


def _write_collection_download_report(metadata_dir: Path, reports_dir: Path, manifest: dict[str, object]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (metadata_dir / "download_manifest.json").write_text(payload, encoding="utf-8")
    (reports_dir / "download_summary.json").write_text(payload, encoding="utf-8")


def _read_collection_download_manifest(collection_name: str) -> dict[str, object]:
    manifest_path = build_collection_root(collection_name) / "metadata" / "download_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _series_uid_from_archive_path(collection_name: str, archive_path: Path) -> str:
    prefix = f"{collection_name.replace(' ', '_').replace('/', '_')}_"
    stem = archive_path.stem
    if stem.startswith(prefix):
        stem = stem[len(prefix):]
    return stem.replace("_", ".")


def _count_images_in_archive(archive_path: Path) -> int:
    try:
        with zipfile.ZipFile(archive_path) as archive:
            total = 0
            for item in archive.infolist():
                if item.is_dir():
                    continue
                name = PurePosixPath(item.filename).name
                upper_name = name.upper()
                suffix = PurePosixPath(name).suffix.lower()
                if upper_name in {"LICENSE", "LICENSE.TXT", "README", "README.TXT", "DICOMDIR"}:
                    continue
                if suffix in {".txt", ".csv", ".xml", ".json", ".md"}:
                    continue
                if suffix in {".dcm", ".dicom", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
                    total += 1
                    continue
                if "." not in name:
                    total += 1
            return total
    except (OSError, ValueError, zipfile.BadZipFile):
        return 0


def _series_from_existing_archives(collection_name: str, raw_images: Path) -> list[dict[str, object]]:
    restored: list[dict[str, object]] = []
    if not raw_images.exists():
        return restored
    for archive_path in sorted(raw_images.glob("*.zip")):
        restored.append(
            {
                "file": str(archive_path),
                "series_uid": _series_uid_from_archive_path(collection_name, archive_path),
                "image_count": _count_images_in_archive(archive_path),
            }
        )
    return restored


def _resolved_archive_image_count(archive_path: Path, fallback: int) -> int:
    actual_count = _count_images_in_archive(archive_path)
    if actual_count > 0:
        return max(actual_count, fallback)
    return fallback


def _count_collection_archived_images(raw_images: Path) -> int:
    if not raw_images.exists():
        return 0
    return sum(_count_images_in_archive(archive_path) for archive_path in raw_images.glob("*.zip"))


def _count_collection_extracted_images(collection_root: Path) -> int:
    if not collection_root.exists():
        return 0
    return sum(
        1
        for path in collection_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dcm", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    )


def _series_priority_key(record: dict[str, object]) -> tuple[int, int]:
    count = max(int(record.get("image_count") or 1), 1)
    if 25 <= count <= 256:
        return (0, count)
    if 5 <= count < 25:
        return (1, count)
    if 257 <= count <= 512:
        return (2, count)
    if count < 5:
        return (3, count)
    return (4, count)


def _downloaded_series_from_existing_files(collection_name: str, series_records: list[dict[str, object]], raw_images: Path) -> list[dict[str, object]]:
    restored: list[dict[str, object]] = []
    for record in series_records:
        series_uid = str(record.get("series_uid") or "")
        if not series_uid:
            continue
        target = raw_images / f"{_safe_series_file_stem(collection_name, series_uid)}.zip"
        if target.exists():
            fallback_count = int(record.get("image_count") or 1)
            restored.append(
                {
                    "file": str(target),
                    "series_uid": series_uid,
                    "image_count": _resolved_archive_image_count(target, fallback_count),
                }
            )
    return restored


def download_collection(collection_name: str, *, force: bool = False, limit: int | None = None, max_images: int | None = None) -> dict[str, object]:
    root = build_collection_root(collection_name)
    raw_images = root / "raw" / "images"
    metadata_dir = root / "metadata"
    reports_dir = root / "reports"
    raw_images.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_lines = [f"collection={collection_name}", f"root={root}", f"force={force}", f"limit={limit}"]
    previous_manifest = _read_collection_download_manifest(collection_name)
    offline_series = _series_from_existing_archives(collection_name, raw_images)
    series_records = _fetch_series_records(collection_name)
    if series_records and isinstance(series_records[0].get("error"), str):
        error = str(series_records[0]["error"]).split("::", 2)[-1]
        series_records = []
    else:
        error = ""
    if not series_records:
        manifest = dict(previous_manifest) if previous_manifest else {}
        if offline_series:
            manifest["files"] = [str(item["file"]) for item in offline_series]
            manifest["series"] = offline_series
            manifest["downloaded_count"] = len(offline_series)
            manifest["downloaded_image_count"] = sum(int(item.get("image_count") or 0) for item in offline_series)
        else:
            manifest.setdefault("files", [])
            manifest.setdefault("series", [])
            manifest.setdefault("downloaded_count", 0)
            manifest.setdefault("downloaded_image_count", 0)
        manifest = {
            **manifest,
            "collection_name": collection_name,
            "series_links_count": 0,
            "warning": error or "No TCIA download links could be resolved for this collection.",
        }
        log_lines.append("status=no_links")
        _append_collection_log(collection_name, log_lines)
        _write_collection_download_report(metadata_dir, reports_dir, manifest)
        return manifest
    restored_from_manifest = 0
    restored_from_existing = 0
    failed_series_uids = {
        str(item.get("series_uid"))
        for item in previous_manifest.get("failed", [])
        if isinstance(item, dict) and item.get("series_uid")
    }
    series_records = sorted(series_records, key=_series_priority_key)
    downloaded: list[str] = []
    downloaded_series: list[dict[str, object]] = []
    seen_downloaded_series: set[str] = set()
    seen_downloaded_files: set[str] = set()
    for item in previous_manifest.get("series", []):
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        series_uid = item.get("series_uid")
        if not isinstance(file_path, str) or not isinstance(series_uid, str):
            continue
        if not Path(file_path).exists():
            continue
        restored_item = dict(item)
        restored_item["image_count"] = _resolved_archive_image_count(Path(file_path), int(item.get("image_count") or 1))
        downloaded.append(file_path)
        downloaded_series.append(restored_item)
        seen_downloaded_series.add(series_uid)
        seen_downloaded_files.add(file_path)
        restored_from_manifest += 1
    for item in _downloaded_series_from_existing_files(collection_name, series_records, raw_images):
        series_uid = str(item["series_uid"])
        file_path = str(item["file"])
        if series_uid in seen_downloaded_series or file_path in seen_downloaded_files:
            continue
        downloaded.append(file_path)
        downloaded_series.append(item)
        seen_downloaded_series.add(series_uid)
        seen_downloaded_files.add(file_path)
        restored_from_existing += 1
    for item in offline_series:
        series_uid = str(item["series_uid"])
        file_path = str(item["file"])
        if series_uid in seen_downloaded_series or file_path in seen_downloaded_files:
            continue
        downloaded.append(file_path)
        downloaded_series.append(item)
        seen_downloaded_series.add(series_uid)
        seen_downloaded_files.add(file_path)
        restored_from_existing += 1
    failed: list[dict[str, str]] = [
        item
        for item in previous_manifest.get("failed", [])
        if isinstance(item, dict) and isinstance(item.get("url"), str)
    ]
    downloaded_image_count = sum(int(item.get("image_count") or 0) for item in downloaded_series)
    starting_downloaded_image_count = downloaded_image_count
    if failed and len(failed) >= MAX_COLLECTION_CONSECUTIVE_FAILURES and downloaded_image_count == 0 and not force:
        manifest = dict(previous_manifest)
        manifest.setdefault("collection_name", collection_name)
        manifest.setdefault("downloaded_count", len(downloaded))
        manifest.setdefault("downloaded_image_count", downloaded_image_count)
        manifest["failed_count"] = len(failed)
        manifest["warning"] = f"Skipped collection after {len(failed)} previous failed series."
        log_lines.append(f"status=skipped_previous_failures count={len(failed)}")
        _append_collection_log(collection_name, log_lines)
        _write_collection_download_report(metadata_dir, reports_dir, manifest)
        return manifest
    manifest = {
        "collection_name": collection_name,
        "downloaded_count": len(downloaded),
        "downloaded_image_count": downloaded_image_count,
        "series_links_count": len(series_records),
        "files": downloaded,
        "series": downloaded_series,
        "failed_count": len(failed),
        "failed": failed,
        "restored_from_manifest": restored_from_manifest,
        "restored_from_existing": restored_from_existing,
    }
    consecutive_failures = 0
    for index, record in enumerate(series_records):
        if limit is not None and index >= limit:
            break
        if max_images is not None and (downloaded_image_count - starting_downloaded_image_count) >= max_images:
            break
        link = str(record["download_url"])
        image_count = int(record.get("image_count") or 1)
        series_uid = str(record.get("series_uid") or index)
        file_name = f"{_safe_series_file_stem(collection_name, series_uid)}.zip"
        target = raw_images / file_name
        if series_uid in seen_downloaded_series and not force:
            log_lines.append(f"SKIP previous-downloaded={series_uid}")
            continue
        if series_uid in failed_series_uids and not force:
            log_lines.append(f"SKIP previous-failed={series_uid}")
            continue
        if target.exists() and not force:
            actual_image_count = _resolved_archive_image_count(target, image_count)
            downloaded.append(str(target))
            downloaded_image_count += actual_image_count
            downloaded_series.append({"file": str(target), "series_uid": series_uid, "image_count": actual_image_count})
            seen_downloaded_series.add(series_uid)
            seen_downloaded_files.add(str(target))
            log_lines.append(f"SKIP existing={target}")
            manifest["downloaded_count"] = len(downloaded)
            manifest["downloaded_image_count"] = downloaded_image_count
            _write_collection_download_report(metadata_dir, reports_dir, manifest)
            consecutive_failures = 0
            continue
        ok, error = _safe_download(link, target)
        if ok:
            actual_image_count = _resolved_archive_image_count(target, image_count)
            downloaded.append(str(target))
            downloaded_image_count += actual_image_count
            downloaded_series.append({"file": str(target), "series_uid": series_uid, "image_count": actual_image_count})
            seen_downloaded_series.add(series_uid)
            seen_downloaded_files.add(str(target))
            log_lines.append(f"OK {link} -> {target}")
            consecutive_failures = 0
        else:
            failed.append({"url": link, "series_uid": series_uid, "error": error or "unknown error"})
            log_lines.append(f"FAIL {link} | {error or 'unknown error'}")
            consecutive_failures += 1
        manifest["downloaded_count"] = len(downloaded)
        manifest["downloaded_image_count"] = downloaded_image_count
        manifest["failed_count"] = len(failed)
        manifest["files"] = downloaded
        manifest["series"] = downloaded_series
        _write_collection_download_report(metadata_dir, reports_dir, manifest)
        if consecutive_failures >= MAX_COLLECTION_CONSECUTIVE_FAILURES:
            manifest["warning"] = f"Stopped after {consecutive_failures} consecutive failed series."
            log_lines.append(f"status=stopped_after_consecutive_failures count={consecutive_failures}")
            _write_collection_download_report(metadata_dir, reports_dir, manifest)
            break
    manifest["downloaded_count"] = len(downloaded)
    manifest["downloaded_image_count"] = downloaded_image_count
    manifest["failed_count"] = len(failed)
    manifest["files"] = downloaded
    manifest["series"] = downloaded_series
    log_lines.append(f"downloaded_count={len(downloaded)}")
    log_lines.append(f"downloaded_image_count={downloaded_image_count}")
    log_lines.append(f"failed_count={len(failed)}")
    log_lines.append(f"download_strategy={SERIES_DOWNLOAD_STRATEGY}")
    log_lines.append(f"restored_from_manifest={restored_from_manifest}")
    log_lines.append(f"restored_from_existing={restored_from_existing}")
    _append_collection_log(collection_name, log_lines)
    _write_collection_download_report(metadata_dir, reports_dir, manifest)
    return manifest


def dry_run_collection(collection_name: str, *, limit: int | None = None) -> dict[str, object]:
    root = build_collection_root(collection_name)
    series_records = _fetch_series_records(collection_name)
    if series_records and isinstance(series_records[0].get("error"), str):
        error = str(series_records[0]["error"]).split("::", 2)[-1]
        _append_collection_log(collection_name, [f"collection={collection_name}", f"root={root}", "mode=dry-run", f"error={error}"])
        return {
            "collection_name": collection_name,
            "collection_root": str(root),
            "series_links_count": 0,
            "series_links_preview": [],
            "downloaded_count": 0,
            "failed_count": 0,
            "dry_run": True,
            "error": error,
        }
    if limit is not None:
        series_records = series_records[:limit]
    series_links = [str(item["download_url"]) for item in series_records]
    image_count = sum(int(item.get("image_count") or 1) for item in series_records)
    _append_collection_log(
        collection_name,
        [
            f"collection={collection_name}",
            f"root={root}",
            "mode=dry-run",
            f"series_links_count={len(series_links)}",
            f"image_count={image_count}",
        ]
        + [f"preview={item}" for item in series_links[:5]],
    )
    return {
        "collection_name": collection_name,
        "collection_root": str(root),
        "series_links_count": len(series_links),
        "series_links_preview": series_links[:5],
        "image_count": image_count,
        "downloaded_count": 0,
        "failed_count": 0,
        "dry_run": True,
    }


def count_downloaded_images(root: Path = TCIA_DOWNLOAD_ROOT) -> int:
    if not root.exists():
        return 0
    total = 0
    for collection_root in root.iterdir():
        if not collection_root.is_dir() or collection_root.name == "logs":
            continue
        raw_images = collection_root / "raw" / "images"
        archive_total = _count_collection_archived_images(raw_images)
        if archive_total:
            total += archive_total
            continue
        manifest_path = collection_root / "metadata" / "download_manifest.json"
        manifest_total = 0
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                manifest = {}
            manifest_total = int(manifest.get("downloaded_image_count", 0)) if isinstance(manifest, dict) else 0
            if manifest_total == 0 and isinstance(manifest, dict):
                series = manifest.get("series", [])
                manifest_total = sum(int(item.get("image_count") or 0) for item in series if isinstance(item, dict))
        if manifest_total:
            total += manifest_total
            continue
        total += _count_collection_extracted_images(collection_root)
    if total:
        return total
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")


def build_collection_status(collection_name: str, *, target_total: int = TARGET_TOTAL_IMAGES, root: Path = TCIA_DOWNLOAD_ROOT) -> dict[str, object]:
    collection_root = build_collection_root(collection_name)
    raw_images = collection_root / "raw" / "images"
    current_candidates: list[int] = []
    manifest_path = collection_root / "metadata" / "download_manifest.json"
    failed_count = 0
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_current = int(manifest.get("downloaded_image_count", 0))
            if manifest_current == 0:
                series = manifest.get("series", []) if isinstance(manifest, dict) else []
                manifest_current = sum(int(item.get("image_count") or 0) for item in series if isinstance(item, dict))
            current_candidates.append(manifest_current)
            failed_count = int(manifest.get("failed_count", 0))
        except (OSError, ValueError, TypeError):
            failed_count = 0
    zip_files = list(raw_images.glob("*.zip")) if raw_images.exists() else []
    archive_count = _count_collection_archived_images(raw_images)
    if archive_count:
        current_candidates.append(archive_count)
    elif zip_files:
        series_records = _fetch_series_records(collection_name)
        if not (series_records and isinstance(series_records[0].get("error"), str)):
            restored_count = sum(int(item.get("image_count") or 0) for item in _downloaded_series_from_existing_files(collection_name, series_records, raw_images))
            current_candidates.append(restored_count)
        offline_count = sum(int(item.get("image_count") or 0) for item in _series_from_existing_archives(collection_name, raw_images))
        current_candidates.append(offline_count)
    extracted_count = _count_collection_extracted_images(collection_root)
    if extracted_count:
        current_candidates.append(extracted_count)
    current = max(current_candidates) if current_candidates else 0
    if current == 0 and collection_root.exists():
        current = sum(1 for path in collection_root.rglob("*") if path.is_file() and path.suffix.lower() == ".zip")
    total = count_downloaded_images(root)
    return {
        "collection_name": collection_name,
        "downloaded_in_collection": current,
        "downloaded_total": total,
        "remaining_to_target": max(target_total - total, 0),
        "failed_count": failed_count,
        "collection_root": str(collection_root),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TCIA collections into dataset/medical/tcia/<collection>/...")
    parser.add_argument("--collections-file", type=Path, default=DEFAULT_COLLECTIONS_FILE, help="JSON file with collection list.")
    parser.add_argument("--collection", action="append", default=[], help="Single collection name; can be repeated.")
    parser.add_argument("--force", action="store_true", help="Redownload existing files.")
    parser.add_argument("--limit", type=int, default=None, help="Limit series downloads per collection.")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect collections and print what would be downloaded.")
    parser.add_argument("--manifest", action="store_true", help="Write the consolidated TCIA manifest.")
    parser.add_argument("--write-manifest", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.manifest or args.write_manifest:
        write_tcia_manifest()
    requested = list(args.collection) or load_collection_list(args.collections_file)
    requested = _priority_order(requested)
    TCIA_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    per_collection: list[dict[str, object]] = []
    for collection_name in requested:
        if args.dry_run:
            per_collection.append(dry_run_collection(collection_name, limit=args.limit))
        else:
            per_collection.append(download_collection(collection_name, force=args.force, limit=args.limit))
    total_downloaded = count_downloaded_images()
    remaining = max(TARGET_TOTAL_IMAGES - total_downloaded, 0)
    total_failed = sum(int(item.get("failed_count", 0)) for item in per_collection)
    status = {
        "downloaded_total": total_downloaded,
        "target_total": TARGET_TOTAL_IMAGES,
        "remaining_to_target": remaining,
        "failed_total": total_failed,
        "dry_run": args.dry_run,
        "collections": per_collection,
    }
    TCIA_STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


def run_from_file(
    collections_file: str | Path,
    *,
    force: bool = False,
    limit: int | None = None,
    write_manifest: bool = True,
    target_total: int = TARGET_TOTAL_IMAGES,
    collections: list[str] | None = None,
) -> list[dict[str, object]]:
    collections_path = Path(collections_file)
    if write_manifest:
        write_tcia_manifest()
    results: list[dict[str, object]] = []
    explicit_collections = collections is not None and len(collections) > 0
    collection_names = _priority_order(collections or load_collection_list(collections_path))
    for index, collection_name in enumerate(collection_names):
        remaining = max(target_total - count_downloaded_images(), 0)
        if remaining <= 0 and not explicit_collections:
            break
        collection_target = None
        if not explicit_collections:
            remaining_collections = max(len(collection_names) - index, 1)
            collection_target = max((remaining + remaining_collections - 1) // remaining_collections, 1)
            collection_target = min(collection_target, COLLECTION_IMAGE_TARGET_OVERRIDES.get(collection_name, collection_target))
        results.append(download_collection(collection_name, force=force, limit=limit, max_images=collection_target))
    return results


def dry_run_from_file(
    collections_file: str | Path,
    *,
    limit: int | None = None,
    write_manifest: bool = False,
    collections: list[str] | None = None,
) -> list[dict[str, object]]:
    collections_path = Path(collections_file)
    if write_manifest:
        write_tcia_manifest()
    results: list[dict[str, object]] = []
    for collection_name in _priority_order(collections or load_collection_list(collections_path)):
        results.append(dry_run_collection(collection_name, limit=limit))
    return results


def verify_downloads(collections_file: str | Path | None = None) -> dict[str, object]:
    collection_names = _priority_order(load_collection_list(Path(collections_file) if collections_file is not None else DEFAULT_COLLECTIONS_FILE))
    details = [build_collection_status(name) for name in collection_names]
    total = sum(int(item.get("downloaded_in_collection", 0)) for item in details)
    return {
        "target_total": TARGET_TOTAL_IMAGES,
        "downloaded_total": total,
        "remaining_to_target": max(TARGET_TOTAL_IMAGES - total, 0),
        "download_failed_total": sum(int(item.get("failed_count", 0)) for item in details),
        "collections": details,
        "canonical_root": str(TCIA_DOWNLOAD_ROOT),
    }


if __name__ == "__main__":
    main()
