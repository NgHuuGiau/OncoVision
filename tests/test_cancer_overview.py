from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from medical.cancer_overview import build_cancer_overview


class CancerOverviewTests(unittest.TestCase):
    def test_build_cancer_overview_aggregates_local_images_by_cancer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            skin_dir = Path(temp_dir) / "skin"
            skin_dir.mkdir(parents=True)
            for index in range(3):
                (skin_dir / f"{index}.jpg").write_bytes(b"x")
            fake_report = {
                "downloaded_total": 15,
                "remaining_to_target": 0,
                "collections": [
                    {
                        "collection_name": "TCGA-BRCA",
                        "downloaded_in_collection": 5,
                        "collection_root": "dataset/medical/tcia/TCGA-BRCA",
                        "failed_count": 0,
                    },
                    {
                        "collection_name": "NSCLC-Radiomics",
                        "downloaded_in_collection": 7,
                        "collection_root": "dataset/medical/tcia/NSCLC-Radiomics",
                        "failed_count": 0,
                    },
                    {
                        "collection_name": "TCGA-READ",
                        "downloaded_in_collection": 3,
                        "collection_root": "dataset/medical/tcia/TCGA-READ",
                        "failed_count": 0,
                    },
                ],
            }
            with patch("medical.cancer_overview.SKIN_IMAGE_DIR", skin_dir), patch(
                "medical.cancer_overview.verify_downloads", return_value=fake_report
            ):
                overview = build_cancer_overview()

        summary = overview["summary"]
        self.assertEqual(summary["skin_raw_images"], 3)
        self.assertEqual(summary["tcia_total_images"], 15)
        self.assertEqual(summary["total_cancer_images"], 18)

        cancers = {item["cancer_type"]: item for item in overview["cancers"]}
        self.assertEqual(cancers["ung_thu_da"]["local_image_count"], 3)
        self.assertEqual(cancers["ung_thu_vu"]["local_image_count"], 5)
        self.assertEqual(cancers["ung_thu_phoi"]["local_image_count"], 7)
        self.assertEqual(cancers["ung_thu_dai_truc_trang"]["local_image_count"], 3)
        self.assertEqual(cancers["ung_thu_co_tu_cung"]["local_status"], "chua_co_nguon_local")


if __name__ == "__main__":
    unittest.main()
