from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from training import validate_dataset


def _report(
    *,
    raw_image_count: int,
    eligible: list[tuple[Path, Path]] | None = None,
    missing_labels: list[Path] | None = None,
    empty_labels: list[Path] | None = None,
    invalid_labels: list[tuple[Path, str]] | None = None,
    orphan_labels: list[Path] | None = None,
):
    return SimpleNamespace(
        raw_image_count=raw_image_count,
        eligible=eligible or [],
        missing_labels=missing_labels or [],
        empty_labels=empty_labels or [],
        invalid_labels=invalid_labels or [],
        orphan_labels=orphan_labels or [],
    )


class ValidateDatasetTests(unittest.TestCase):
    @patch("builtins.print")
    @patch("training.validate_dataset.audit_raw_dataset")
    def test_main_uses_dataset_root_raw_dirs(self, audit_mock, print_mock) -> None:
        audit_mock.return_value = _report(raw_image_count=1, eligible=[(Path("a.jpg"), Path("a.txt"))])

        with patch("sys.argv", ["validate_dataset.py", "--dataset-root", "dataset/medical/skin_lesion"]):
            validate_dataset.main()

        audit_mock.assert_called_once_with(
            raw_images_dir=Path("dataset/medical/skin_lesion/raw/images"),
            raw_labels_dir=Path("dataset/medical/skin_lesion/raw/labels"),
        )

    @patch("builtins.print")
    @patch("training.validate_dataset.audit_raw_dataset")
    def test_main_exits_with_guidance_when_no_raw_images(self, audit_mock, print_mock) -> None:
        audit_mock.return_value = _report(raw_image_count=0)

        with self.assertRaises(SystemExit) as context:
            validate_dataset.main()

        self.assertEqual(context.exception.code, 1)
        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("prepare_dataset.py", output)
        self.assertIn("validate_dataset.py", output)

    @patch("builtins.print")
    @patch("training.validate_dataset.audit_raw_dataset")
    def test_main_exits_when_missing_or_invalid_labels_exist(self, audit_mock, print_mock) -> None:
        audit_mock.return_value = _report(
            raw_image_count=3,
            missing_labels=[Path("dataset/object_detection/raw/images/a.jpg")],
            invalid_labels=[(Path("dataset/object_detection/raw/labels/b.txt"), "bad format")],
        )

        with self.assertRaises(SystemExit) as context:
            validate_dataset.main()

        self.assertEqual(context.exception.code, 1)
        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("a.jpg", output)
        self.assertIn("b.txt", output)

    @patch("builtins.print")
    @patch("training.validate_dataset.audit_raw_dataset")
    def test_main_prints_next_commands_when_dataset_is_valid(self, audit_mock, print_mock) -> None:
        audit_mock.return_value = _report(
            raw_image_count=2,
            eligible=[(Path("img1.jpg"), Path("img1.txt"))],
        )

        validate_dataset.main()

        output = "\n".join(str(call.args[0]) for call in print_mock.call_args_list if call.args)
        self.assertIn("split_dataset.py", output)
        self.assertIn("run_train.py", output)
