from __future__ import annotations

import unittest

from medical.cancer_catalog import (
    COMMON_CANCER_TARGETS,
    get_cancer_target,
    supported_cancer_labels,
    supported_cancer_modalities,
)


class CancerCatalogTests(unittest.TestCase):
    def test_catalog_has_twelve_supported_targets(self) -> None:
        targets = list(COMMON_CANCER_TARGETS)
        self.assertEqual(len(targets), 12)
        keys = {t.key for t in targets}
        self.assertIn("brain", keys)
        self.assertIn("spleen", keys)
        # Most targets should have model_ready; spleen may not yet
        ready_count = sum(1 for t in targets if t.model_ready)
        self.assertGreaterEqual(ready_count, 11)

    def test_supported_cancer_labels_lists_all_targets(self) -> None:
        labels = supported_cancer_labels()
        self.assertEqual(len(labels), 12)
        self.assertIn("Ung thư não", labels)
        self.assertIn("Ung thư tuyến giáp", labels)
        self.assertIn("Ung thư lách", labels)

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
