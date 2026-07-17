from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from PIL import Image

from medical.modality_training import _collect_modality_samples, train_modality_classifier


def _make_modality_dataset(root: Path, modalities: list[str], count: int = 6) -> None:
    for modality in modalities:
        modality_dir = root / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.new("RGB", (32, 32), (index * 20 % 256, 50, 100)).save(
                modality_dir / f"{modality}_{index}.jpg"
            )


class TestModalityTraining:
    def test_collect_modality_samples_maps_folders_to_labels(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "medical_modality"
            _make_modality_dataset(root, ["ct", "mri", "xray"])
            samples = _collect_modality_samples(root)
            assert len(samples) == 18
            labels = {label for _, label in samples}
            assert labels == {0, 1, 2}

    def test_train_modality_classifier_runs_end_to_end(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "medical_modality"
            _make_modality_dataset(root, ["ct", "mri", "xray"])
            with mock.patch("medical.modality_training.train_cnn_classifier") as train_mock, \
                    mock.patch("medical.modality_training.build_modality_classifier") as build_mock:
                wrapper = mock.MagicMock()
                wrapper.save.return_value = root / "modality_classifier.pt"
                train_mock.return_value = (wrapper, {"train_loss": [0.1]})
                build_mock.return_value = mock.MagicMock(backbone_name="resnet18")
                out_path = root / "modality_classifier.pt"
                result = train_modality_classifier(
                    root,
                    out_path,
                    image_size=32,
                    batch_size=4,
                    num_epochs=1,
                    pretrained=False,
                    verbose=False,
                )
                assert result == out_path
                wrapper.save.assert_called_once()

    def test_train_modality_classifier_missing_dataset_raises(self) -> None:
        import pytest

        with TemporaryDirectory() as temp_dir:
            with pytest.raises(FileNotFoundError):
                train_modality_classifier(
                    Path(temp_dir) / "missing",
                    Path(temp_dir) / "out.pt",
                    pretrained=False,
                )
