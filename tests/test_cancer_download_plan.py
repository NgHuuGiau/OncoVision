from __future__ import annotations

from unittest import TestCase

from training.cancer_download_plan import build_download_plan


class CancerDownloadPlanTests(TestCase):
    def test_plan_includes_planned_downloads(self) -> None:
        plan = build_download_plan()
        priority_names = [item["cancer_name"] for item in plan["priority_downloads"]]
        names = [item["cancer_name"] for item in plan["planned_downloads"]]
        self.assertIn("ung_thu_da", priority_names)
        self.assertIn("ung_thu_vu", names)
        self.assertIn("ung_thu_phoi", names)
        self.assertIn("ung_thu_dai_truc_trang", names)
