from __future__ import annotations

import unittest

from medical.cancer_catalog import list_common_cancer_targets, supported_cancer_labels


class CancerCatalogTests(unittest.TestCase):
    def test_supported_cancer_labels_include_common_targets(self) -> None:
        labels = supported_cancer_labels()
        self.assertIn("Ung thu da", labels)
        self.assertIn("Ung thu vu", labels)
        self.assertIn("Ung thu phoi", labels)
        self.assertIn("Ung thu dai truc trang", labels)

    def test_skin_target_is_marked_ready(self) -> None:
        targets = list_common_cancer_targets()
        skin_target = next(item for item in targets if item.key == "skin")
        self.assertTrue(skin_target.model_ready)

    def test_non_skin_targets_are_marked_as_extension_only(self) -> None:
        targets = list_common_cancer_targets()
        self.assertTrue(all(not item.model_ready for item in targets if item.key != "skin"))


if __name__ == "__main__":
    unittest.main()
