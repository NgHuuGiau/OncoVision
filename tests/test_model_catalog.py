from __future__ import annotations

import unittest

from core.model_catalog import (
    DEFAULT_MODEL_FALLBACK,
    MODEL_QUALITY_SCORES,
    YOLO11_MODELS_ASC,
    YOLO11_MODELS_DESC,
    build_model_backups,
)


class ModelCatalogTests(unittest.TestCase):
    def test_model_orders_are_consistent(self) -> None:
        self.assertEqual(YOLO11_MODELS_ASC[0], DEFAULT_MODEL_FALLBACK)
        self.assertEqual(YOLO11_MODELS_DESC, tuple(reversed(YOLO11_MODELS_ASC)))

    def test_quality_scores_cover_all_known_models(self) -> None:
        self.assertEqual(set(MODEL_QUALITY_SCORES), set(YOLO11_MODELS_ASC))
        self.assertGreater(MODEL_QUALITY_SCORES["yolo11x.pt"], MODEL_QUALITY_SCORES["yolo11n.pt"])

    def test_build_model_backups_degrades_toward_lightest_model(self) -> None:
        backups = build_model_backups()
        self.assertEqual(backups["yolo11x.pt"], (None, "yolo11l.pt", "yolo11m.pt", "yolo11s.pt", "yolo11n.pt"))
        self.assertEqual(backups["yolo11s.pt"], (None, "yolo11n.pt"))
        self.assertEqual(backups["yolo11n.pt"], (None,))


if __name__ == "__main__":
    unittest.main()
