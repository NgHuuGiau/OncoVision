from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from training import import_skin_lesion_dataset


class ImportSkinLesionDatasetTests(TestCase):
    @patch("training.import_skin_lesion_dataset.ensure_medical_dataset_structure")
    @patch("training.import_skin_lesion_dataset.create_default_medical_dataset_config")
    @patch("training.import_skin_lesion_dataset.import_isic_2016_part3b_to_yolo")
    def test_main_calls_importer_with_expected_paths(self, import_mock, config_mock, ensure_mock) -> None:
        config_mock.return_value = type(
            "Config",
            (),
            {
                "dataset_root": Path("dataset/custom_skin_lesion"),
                "metadata_dir": Path("dataset/custom_skin_lesion/metadata"),
            },
        )()
        import_mock.return_value = {"imported": 10, "skipped": 2}
        with patch("sys.argv", ["import_skin_lesion_dataset.py", "--dataset-root", "dataset/custom_skin_lesion"]), patch("sys.stdout"):
            import_skin_lesion_dataset.main()
        import_mock.assert_called_once()
        ensure_mock.assert_called_once()
        self.assertEqual(import_mock.call_args.kwargs["metadata_output_path"], Path("dataset/custom_skin_lesion/metadata/isic2016_import.csv"))
