from __future__ import annotations

import json
import os
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from training import tcia_downloader


class TciaDownloaderTests(unittest.TestCase):
    def test_safe_download_streams_to_zip_and_removes_part(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            target = Path(temp_dir) / "series.zip"
            response = MagicMock()
            response.__enter__.return_value = response
            response.__exit__.return_value = False
            response.read.side_effect = [b"abc", b"def", b""]

            with patch("training.tcia_downloader.urlopen", return_value=response):
                ok, error = tcia_downloader._safe_download("https://example.test/file.zip", target, use_process=False)

            self.assertTrue(ok)
            self.assertIsNone(error)
            self.assertEqual(target.read_bytes(), b"abcdef")
            self.assertFalse(target.with_suffix(".zip.part").exists())

    def test_safe_download_cleans_part_after_timeout(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            target = Path(temp_dir) / "series.zip"

            with patch("training.tcia_downloader.DOWNLOAD_RETRIES", 1), patch(
                "training.tcia_downloader.urlopen", side_effect=TimeoutError("stalled")
            ):
                ok, error = tcia_downloader._safe_download("https://example.test/file.zip", target, use_process=False)

            self.assertFalse(ok)
            self.assertIn("stalled", error or "")
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".zip.part").exists())

    def test_verify_downloads_counts_failed_manifest_items(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                collections_file = Path("collections.json")
                collections_file.write_text(json.dumps({"collections": ["TCGA-LIHC"]}), encoding="utf-8")
                manifest_path = Path("dataset/medical/tcia/TCGA-LIHC/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(json.dumps({"failed_count": 3}), encoding="utf-8")

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")):
                    report = tcia_downloader.verify_downloads(collections_file)

                self.assertEqual(report["download_failed_total"], 3)
                self.assertEqual(report["collections"][0]["failed_count"], 3)
            finally:
                os.chdir(previous_cwd)

    def test_run_from_file_accepts_explicit_collection_list(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=[{"series_uid": "x", "download_url": "https://example.test/x.zip", "image_count": 1}]
                ), patch("training.tcia_downloader.download_collection", return_value={"collection_name": "TCGA-READ", "downloaded_count": 1}) as download_mock:
                    results = tcia_downloader.run_from_file("training/tcia_collections_5.json", collections=["TCGA-READ"])

                self.assertEqual(len(results), 1)
                download_mock.assert_called_once()
            finally:
                os.chdir(previous_cwd)

    def test_run_from_file_keeps_explicit_collection_downloads_after_target_is_reached(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader.count_downloaded_images", return_value=tcia_downloader.TARGET_TOTAL_IMAGES
                ), patch("training.tcia_downloader.download_collection", return_value={"collection_name": "TCGA-LUAD"}) as download_mock:
                    results = tcia_downloader.run_from_file("training/tcia_collections_5.json", collections=["TCGA-LUAD"])

                self.assertEqual(len(results), 1)
                download_mock.assert_called_once_with("TCGA-LUAD", force=False, limit=None, max_images=None)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_max_images_applies_to_new_images_only(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                existing_zip = Path("dataset/medical/tcia/TCGA-LUAD/raw/images/existing.zip")
                existing_zip.parent.mkdir(parents=True)
                with zipfile.ZipFile(existing_zip, "w") as archive:
                    archive.writestr("00000001.dcm", b"a")
                    archive.writestr("00000002.dcm", b"b")
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "files": [str(existing_zip)],
                            "series": [{"file": str(existing_zip), "series_uid": "old-series", "image_count": 2}],
                        }
                    ),
                    encoding="utf-8",
                )
                records = [
                    {"series_uid": "old-series", "download_url": "https://example.test/old.zip", "image_count": 2},
                    {"series_uid": "new-series", "download_url": "https://example.test/new.zip", "image_count": 5},
                ]

                def fake_download(_: str, target: Path) -> tuple[bool, str | None]:
                    with zipfile.ZipFile(target, "w") as archive:
                        for index in range(5):
                            archive.writestr(f"{index:08d}.dcm", b"x")
                    return True, None

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ), patch("training.tcia_downloader._safe_download", side_effect=fake_download):
                    report = tcia_downloader.download_collection("TCGA-LUAD", max_images=5)

                self.assertEqual(report["downloaded_image_count"], 7)
                self.assertEqual(len(report["series"]), 2)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_skips_previous_failed_series(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(
                    json.dumps({"failed": [{"series_uid": "bad-series", "error": "timeout"}]}),
                    encoding="utf-8",
                )
                records = [
                    {"series_uid": "bad-series", "download_url": "https://example.test/bad.zip", "image_count": 10},
                    {"series_uid": "good-series", "download_url": "https://example.test/good.zip", "image_count": 5},
                ]
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ), patch("training.tcia_downloader._safe_download", return_value=(True, None)) as download_mock:
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertEqual(report["downloaded_image_count"], 5)
                download_mock.assert_called_once()
                self.assertIn("good.zip", download_mock.call_args.args[0])
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_prefers_medium_series_first(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                records = [
                    {"series_uid": "tiny-series", "download_url": "https://example.test/tiny.zip", "image_count": 3},
                    {"series_uid": "medium-series", "download_url": "https://example.test/medium.zip", "image_count": 50},
                    {"series_uid": "small-series", "download_url": "https://example.test/small.zip", "image_count": 3},
                    {"series_uid": "large-series", "download_url": "https://example.test/large.zip", "image_count": 600},
                ]
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ), patch("training.tcia_downloader._safe_download", return_value=(True, None)) as download_mock:
                    tcia_downloader.download_collection("TCGA-LUAD", limit=1)

                self.assertIn("medium.zip", download_mock.call_args.args[0])
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_preserves_previous_downloaded_series(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                existing_zip = Path("dataset/medical/tcia/TCGA-LUAD/raw/images/existing.zip")
                existing_zip.parent.mkdir(parents=True)
                existing_zip.write_bytes(b"zip")
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "files": [str(existing_zip)],
                            "series": [{"file": str(existing_zip), "series_uid": "old-series", "image_count": 10}],
                        }
                    ),
                    encoding="utf-8",
                )
                records = [
                    {"series_uid": "old-series", "download_url": "https://example.test/old.zip", "image_count": 10},
                    {"series_uid": "new-series", "download_url": "https://example.test/new.zip", "image_count": 5},
                ]
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ), patch("training.tcia_downloader._safe_download", return_value=(True, None)):
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertEqual(report["downloaded_image_count"], 15)
                self.assertEqual(len(report["series"]), 2)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_restores_existing_archives_when_fetch_fails(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                raw_images = Path("dataset/medical/tcia/TCGA-LUAD/raw/images")
                raw_images.mkdir(parents=True)
                archive_path = raw_images / f"{tcia_downloader._safe_series_file_stem('TCGA-LUAD', '1.2.3')}.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("a.dcm", b"a")
                    archive.writestr("b.dcm", b"b")

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=[{"error": "ERROR::TCGA-LUAD::timeout"}]
                ):
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertEqual(report["downloaded_count"], 1)
                self.assertEqual(report["downloaded_image_count"], 2)
                self.assertEqual(len(report["series"]), 1)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_uses_archive_contents_after_successful_download(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                records = [{"series_uid": "1.2.3", "download_url": "https://example.test/series.zip", "image_count": 1}]

                def fake_download(_: str, target: Path) -> tuple[bool, str | None]:
                    with zipfile.ZipFile(target, "w") as archive:
                        archive.writestr("a.dcm", b"a")
                        archive.writestr("b.dcm", b"b")
                    return True, None

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ), patch("training.tcia_downloader._safe_download", side_effect=fake_download):
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertEqual(report["downloaded_image_count"], 2)
                self.assertEqual(report["series"][0]["image_count"], 2)
            finally:
                os.chdir(previous_cwd)

    def test_build_collection_status_restores_count_from_existing_zip_files(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                series_uid = "1.2.3"
                file_name = f"{tcia_downloader._safe_series_file_stem('TCGA-LUAD', series_uid)}.zip"
                zip_path = Path("dataset/medical/tcia/TCGA-LUAD/raw/images") / file_name
                zip_path.parent.mkdir(parents=True)
                zip_path.write_bytes(b"zip")
                records = [{"series_uid": series_uid, "download_url": "https://example.test/series.zip", "image_count": 42}]

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=records
                ):
                    status = tcia_downloader.build_collection_status("TCGA-LUAD")

                self.assertEqual(status["downloaded_in_collection"], 42)
            finally:
                os.chdir(previous_cwd)

    def test_build_collection_status_counts_archive_contents_without_api(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                raw_images = Path("dataset/medical/tcia/TCGA-LUAD/raw/images")
                raw_images.mkdir(parents=True)
                archive_path = raw_images / f"{tcia_downloader._safe_series_file_stem('TCGA-LUAD', '1.2.3')}.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("a.dcm", b"a")
                    archive.writestr("b.dcm", b"b")
                    archive.writestr("c.dcm", b"c")

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=[{"error": "ERROR::TCGA-LUAD::timeout"}]
                ):
                    status = tcia_downloader.build_collection_status("TCGA-LUAD")

                self.assertEqual(status["downloaded_in_collection"], 3)
            finally:
                os.chdir(previous_cwd)

    def test_archive_counter_ignores_license_file(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            archive_path = Path(temp_dir) / "series.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("LICENSE", b"license")
                archive.writestr("00000001.dcm", b"a")

            self.assertEqual(tcia_downloader._count_images_in_archive(archive_path), 1)

    def test_count_downloaded_images_uses_series_counts_when_manifest_total_is_zero(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "downloaded_image_count": 0,
                            "series": [
                                {"file": "x.zip", "series_uid": "1.2.3", "image_count": 4},
                                {"file": "y.zip", "series_uid": "1.2.4", "image_count": 5},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")):
                    total = tcia_downloader.count_downloaded_images()

                self.assertEqual(total, 9)
            finally:
                os.chdir(previous_cwd)

    def test_count_downloaded_images_prefers_actual_archive_contents(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                raw_images = Path("dataset/medical/tcia/TCGA-LUAD/raw/images")
                raw_images.mkdir(parents=True)
                archive_path = raw_images / "series.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("LICENSE", b"license")
                    archive.writestr("00000001.dcm", b"a")
                    archive.writestr("00000002.dcm", b"b")
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(json.dumps({"downloaded_image_count": 1}), encoding="utf-8")

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")):
                    total = tcia_downloader.count_downloaded_images()

                self.assertEqual(total, 2)
            finally:
                os.chdir(previous_cwd)

    def test_build_collection_status_prefers_actual_archive_contents(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                raw_images = Path("dataset/medical/tcia/TCGA-LUAD/raw/images")
                raw_images.mkdir(parents=True)
                archive_path = raw_images / "series.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr("LICENSE", b"license")
                    archive.writestr("00000001.dcm", b"a")
                    archive.writestr("00000002.dcm", b"b")
                    archive.writestr("00000003.dcm", b"c")
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(json.dumps({"downloaded_image_count": 1}), encoding="utf-8")

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader._fetch_series_records", return_value=[{"error": "ERROR::TCGA-LUAD::timeout"}]
                ):
                    status = tcia_downloader.build_collection_status("TCGA-LUAD")

                self.assertEqual(status["downloaded_in_collection"], 3)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_stops_after_consecutive_failures(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                records = [
                    {"series_uid": f"series-{index}", "download_url": f"https://example.test/{index}.zip", "image_count": 1}
                    for index in range(5)
                ]
                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader.MAX_COLLECTION_CONSECUTIVE_FAILURES", 2
                ), patch("training.tcia_downloader._fetch_series_records", return_value=records), patch(
                    "training.tcia_downloader._safe_download", return_value=(False, "timeout")
                ) as download_mock:
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertEqual(report["failed_count"], 2)
                self.assertIn("Stopped after 2 consecutive failed series", report["warning"])
                self.assertEqual(download_mock.call_count, 2)
            finally:
                os.chdir(previous_cwd)

    def test_download_collection_skips_collection_after_previous_failures(self) -> None:
        with TemporaryDirectory(dir="D:\\YOLO") as temp_dir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                manifest_path = Path("dataset/medical/tcia/TCGA-LUAD/metadata/download_manifest.json")
                manifest_path.parent.mkdir(parents=True)
                manifest_path.write_text(
                    json.dumps(
                        {
                            "downloaded_image_count": 0,
                            "failed": [
                                {"url": f"https://example.test/{index}.zip", "series_uid": f"series-{index}", "error": "timeout"}
                                for index in range(3)
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                records = [{"series_uid": "new-series", "download_url": "https://example.test/new.zip", "image_count": 1}]

                with patch.object(tcia_downloader, "TCIA_DOWNLOAD_ROOT", Path("dataset/medical/tcia")), patch(
                    "training.tcia_downloader.MAX_COLLECTION_CONSECUTIVE_FAILURES", 3
                ), patch("training.tcia_downloader._fetch_series_records", return_value=records), patch(
                    "training.tcia_downloader._safe_download"
                ) as download_mock:
                    report = tcia_downloader.download_collection("TCGA-LUAD")

                self.assertIn("Skipped collection after 3 previous failed series", report["warning"])
                download_mock.assert_not_called()
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
