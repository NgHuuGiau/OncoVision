from __future__ import annotations

import unittest

from medical.cancer_catalog import (
    COMMON_CANCER_TARGETS,
    get_cancer_target,
    supported_cancer_labels,
    supported_cancer_modalities,
)


class CancerCatalogTests(unittest.TestCase):
    def test_catalog_has_eleven_supported_targets(self) -> None:
        targets = list(COMMON_CANCER_TARGETS)
        self.assertEqual(len(targets), 11)
        self.assertEqual(targets[-1].key, "brain")
        self.assertTrue(all(target.model_ready for target in targets))

    def test_supported_cancer_labels_lists_all_targets(self) -> None:
        labels = supported_cancer_labels()
        self.assertEqual(len(labels), 11)
        self.assertIn("Ung thư não", labels)
        self.assertIn("Ung thư tuyến giáp", labels)

    def test_catalog_includes_brain_modalities(self) -> None:
        modalities = supported_cancer_modalities()
        for expected in ("MRI não", "CT sọ não", "PET/CT não"):
            self.assertIn(expected, modalities)

    def test_lookup_returns_known_targets(self) -> None:
        self.assertEqual(get_cancer_target("BRAIN").key, "brain")
        self.assertEqual(get_cancer_target("thyroid").key, "thyroid")
        self.assertIsNone(get_cancer_target("unknown"))


if __name__ == "__main__":
    unittest.main()
