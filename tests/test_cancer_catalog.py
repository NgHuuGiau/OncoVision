from __future__ import annotations

import unittest

from medical.cancer_catalog import (
    COMMON_CANCER_TARGETS,
    get_cancer_target,
    supported_cancer_labels,
    supported_cancer_modalities,
)


class CancerCatalogTests(unittest.TestCase):
    def test_catalog_has_only_brain_target_ready(self) -> None:
        targets = list(COMMON_CANCER_TARGETS)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].key, "brain")
        self.assertEqual(targets[0].label, "Ung thư não")
        self.assertTrue(targets[0].model_ready)

    def test_supported_cancer_labels_is_brain_only(self) -> None:
        self.assertEqual(supported_cancer_labels(), ["Ung thư não"])

    def test_catalog_includes_brain_modalities(self) -> None:
        modalities = supported_cancer_modalities()
        for expected in ("MRI não", "CT sọ não", "PET/CT não"):
            self.assertIn(expected, modalities)

    def test_lookup_returns_only_known_target(self) -> None:
        self.assertEqual(get_cancer_target("BRAIN").key, "brain")
        self.assertIsNone(get_cancer_target("unknown"))
        self.assertIsNone(get_cancer_target("thyroid"))


if __name__ == "__main__":
    unittest.main()
