from __future__ import annotations

import unittest

from medical.cancer_catalog import COMMON_CANCER_TARGETS, supported_cancer_labels, supported_cancer_modalities


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
        ):
            self.assertIn(expected, labels)

    def test_catalog_has_seven_supported_targets(self) -> None:
        targets = COMMON_CANCER_TARGETS
        self.assertEqual(len(targets), 7)
        self.assertTrue(all(item.model_ready for item in targets))

    def test_catalog_includes_common_modalities(self) -> None:
        modalities = supported_cancer_modalities()
        for expected in ("CT", "MRI", "PET/CT", "Siêu âm", "Nội soi"):
            self.assertIn(expected, modalities)


if __name__ == "__main__":
    unittest.main()
