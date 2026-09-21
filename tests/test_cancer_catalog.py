from __future__ import annotations

import unittest

from medical.cancer_catalog import (
    COMMON_CANCER_TARGETS,
    get_cancer_target,
    supported_cancer_labels,
    supported_cancer_modalities,
)


class CancerCatalogTests(unittest.TestCase):
    def test_supported_cancer_labels_include_expected_targets(self) -> None:
        labels = supported_cancer_labels()
        for expected in (
            "Ung thư gan",
            "Ung thư phổi",
            "Ung thư vú",
            "Ung thư dạ dày",
            "Ung thư đại trực tràng",
            "Ung thư tuyến tiền liệt",
            "Ung thư cổ tử cung",
            "Ung thư thận",
            "Ung thư tụy",
            "Ung thư tuyến giáp",
        ):
            self.assertIn(expected, labels)

    def test_catalog_has_eleven_targets_with_new_data_targets_pending(self) -> None:
        targets = list(COMMON_CANCER_TARGETS)
        self.assertEqual(len(targets), 11)
        model_ready_count = sum(1 for item in targets if item.model_ready)
        self.assertEqual(model_ready_count, 1)
        ready = {item.key for item in targets if item.model_ready}
        self.assertEqual(ready, {"brain"})
        pending = {item.key for item in targets if not item.model_ready}
        self.assertEqual(len(pending), 10)

    def test_catalog_includes_common_modalities(self) -> None:
        modalities = supported_cancer_modalities()
        for expected in ("CT", "MRI", "PET/CT", "Siêu âm", "Nội soi", "CT thận", "MRI tụy", "Siêu âm tuyến giáp"):
            self.assertIn(expected, modalities)

    def test_lookup_returns_only_known_target(self) -> None:
        self.assertEqual(get_cancer_target("BRAIN").key, "brain")
        self.assertIsNone(get_cancer_target("unknown"))


if __name__ == "__main__":
    unittest.main()
