from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.cancer_catalog import COMMON_CANCER_TARGETS
from medical.cancer_overview import build_cancer_overview


class CancerOverviewTests(unittest.TestCase):
    def test_build_cancer_overview_aggregates_local_images_by_cancer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_total = 0
            for index, target in enumerate(COMMON_CANCER_TARGETS, start=1):
                for split, count in (("train", index), ("val", 1), ("test", 1)):
                    split_dir = root / target.label / "processed" / "images" / split
                    split_dir.mkdir(parents=True, exist_ok=True)
                    for item_index in range(count):
                        (split_dir / f"{split}_{item_index}.jpg").write_bytes(b"x")
                    expected_total += count

            overview = build_cancer_overview(root)

        summary = overview["summary"]
        self.assertEqual(summary["total_cancer_images"], expected_total)
        self.assertEqual(summary["dataset_root"], str(root))

        cancers = {item["label"]: item for item in overview["cancers"]}
        self.assertEqual(len(cancers), 7)
        self.assertEqual(cancers["Ung thư gan"]["local_status"], "co_anh_local")
        self.assertGreater(cancers["Ung thư gan"]["local_image_count"], 0)
        self.assertTrue(all(item["model_ready"] for item in overview["cancers"]))


if __name__ == "__main__":
    unittest.main()
