from __future__ import annotations

import unittest

import numpy as np

from core.camera_runner import DetectionRecord
from utils.draw_utils import (
    _boxes_overlap,
    _clamp_bbox_to_image,
    _color_for_label,
    _select_labels_to_show,
    draw_detection_results,
)


class VisualizationTests(unittest.TestCase):
    def test_draw_detection_results_renders_box_and_coordinate_text(self) -> None:
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        detections = [DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(20, 30, 100, 80))]

        output = draw_detection_results(image.copy(), detections, box_thickness=2, label_font_scale=0.8)

        self.assertGreater(int(output.sum()), 0)
        self.assertTrue(np.any(output[79:105, 20:120] != 0))

    def test_color_for_label_is_stable_and_varies_by_label(self) -> None:
        person_color = _color_for_label("person")
        helmet_color = _color_for_label("helmet")

        self.assertEqual(person_color, _color_for_label("person"))
        self.assertNotEqual(person_color, helmet_color)

    def test_draw_detection_results_clamps_out_of_bounds_bbox(self) -> None:
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        detections = [DetectionRecord(class_id=0, label="face", confidence=0.88, bbox=(-10, -5, 100, 60))]

        output = draw_detection_results(image.copy(), detections)

        self.assertGreater(int(output.sum()), 0)

    def test_draw_detection_results_ignores_fps_overlay_request(self) -> None:
        image = np.zeros((160, 220, 3), dtype=np.uint8)
        detections = [DetectionRecord(class_id=0, label="person", confidence=0.91, bbox=(40, 30, 140, 90))]

        output = draw_detection_results(
            image.copy(),
            detections,
            box_thickness=2,
            label_font_scale=0.8,
            fps=31.4,
            show_fps=True,
        )

        self.assertTrue(np.any(output[20:105, 20:160] != 0))
        self.assertTrue(np.any(output[120:160, 150:220] != 0))

    def test_draw_detection_results_renders_fixed_bottom_right_fps_without_detections(self) -> None:
        image = np.zeros((80, 180, 3), dtype=np.uint8)

        output = draw_detection_results(image.copy(), [], fps=28.0, show_fps=True)

        self.assertEqual(int(output[:40, :80].sum()), 0)
        self.assertGreater(int(output[40:, 100:].sum()), 0)

    def test_clamp_bbox_to_image_returns_none_for_invalid_box(self) -> None:
        self.assertIsNone(_clamp_bbox_to_image((20, 20, 20, 30), (40, 50, 3)))
        self.assertEqual(_clamp_bbox_to_image((30, 10, 5, 25), (40, 50, 3)), (5, 10, 30, 25))

    def test_boxes_overlap_detects_overlapping_boxes(self) -> None:
        # Two boxes that overlap significantly
        bbox1 = (10, 10, 50, 50)
        bbox2 = (30, 30, 70, 70)
        self.assertTrue(_boxes_overlap(bbox1, bbox2, iou_threshold=0.1))

    def test_boxes_overlap_returns_false_for_disjoint_boxes(self) -> None:
        # Two boxes that don't overlap
        bbox1 = (10, 10, 30, 30)
        bbox2 = (50, 50, 70, 70)
        self.assertFalse(_boxes_overlap(bbox1, bbox2, iou_threshold=0.1))

    def test_select_labels_to_show_picks_highest_confidence(self) -> None:
        # Two heavily overlapping boxes - should pick the one with higher confidence
        dets = [
            DetectionRecord(class_id=0, label="tumor", confidence=0.5, bbox=(10, 10, 50, 50)),
            DetectionRecord(class_id=0, label="tumor", confidence=0.9, bbox=(20, 20, 60, 60)),
        ]
        selected = _select_labels_to_show(dets, (100, 100, 3))
        self.assertIn(1, selected)  # Higher confidence should be selected
        self.assertNotIn(0, selected)  # Lower confidence should not be selected (high IoU)

    def test_select_labels_to_show_same_label_picks_best(self) -> None:
        # Two non-overlapping boxes with SAME label - only pick the best one
        dets = [
            DetectionRecord(class_id=0, label="tumor", confidence=0.9, bbox=(10, 10, 30, 30)),
            DetectionRecord(class_id=0, label="tumor", confidence=0.8, bbox=(60, 60, 80, 80)),
        ]
        selected = _select_labels_to_show(dets, (100, 100, 3))
        # Same label: only the highest confidence is selected
        self.assertEqual(len(selected), 1)
        self.assertIn(0, selected)  # Higher confidence
        self.assertNotIn(1, selected)  # Lower confidence, same label

    def test_select_labels_to_show_different_labels(self) -> None:
        # Two non-overlapping boxes with DIFFERENT labels - both should show
        dets = [
            DetectionRecord(class_id=0, label="tumor", confidence=0.9, bbox=(10, 10, 30, 30)),
            DetectionRecord(class_id=1, label="cyst", confidence=0.8, bbox=(60, 60, 80, 80)),
        ]
        selected = _select_labels_to_show(dets, (100, 100, 3))
        self.assertIn(0, selected)
        self.assertIn(1, selected)

    def test_select_labels_to_show_empty_list(self) -> None:
        selected = _select_labels_to_show([], (100, 100, 3))
        self.assertEqual(selected, set())

    def test_draw_detection_results_no_overlapping_labels(self) -> None:
        # Two overlapping boxes - only one label should be drawn
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [
            DetectionRecord(class_id=0, label="tumor", confidence=0.5, bbox=(50, 50, 100, 100)),
            DetectionRecord(class_id=0, label="tumor", confidence=0.9, bbox=(70, 70, 120, 120)),
        ]
        output = draw_detection_results(image.copy(), dets, show_all_labels=False)
        # Should have drawn boxes (non-zero pixels)
        self.assertGreater(int(output.sum()), 0)


if __name__ == "__main__":
    unittest.main()
