from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VolumeData:
    array: np.ndarray
    spacing_xyz: tuple[float, float, float]
    affine: np.ndarray
    modality: str | None = None
    coordinate_system: str = "unknown"

    def voxel_to_world(self, xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        point = self.affine @ np.asarray((*xyz, 1.0), dtype=np.float64)
        return float(point[0]), float(point[1]), float(point[2])


@dataclass(frozen=True)
class VolumeModelManifest:
    task: str
    class_labels: tuple[str, ...]
    input_shape: tuple[int, int, int]
    mask_threshold: float
    target: str
    input_modalities: tuple[str, ...]
    segmentation_labels: tuple[str, ...]
    segmentation_activation: str

    @classmethod
    def load(cls, path: str | Path) -> VolumeModelManifest:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Không đọc được manifest model 3D: {source}") from exc
        task = str(data.get("task", ""))
        if task not in {"volume_classification", "volume_segmentation", "volume_multitask"}:
            raise ValueError("Manifest 3D phải khai báo task volume_classification, volume_segmentation hoặc volume_multitask.")
        raw_shape = data.get("input_shape")
        if not isinstance(raw_shape, list) or len(raw_shape) != 3:
            raise ValueError("Manifest 3D phải có input_shape=[depth,height,width].")
        input_shape = tuple(int(value) for value in raw_shape)
        if any(value <= 0 or value > 512 for value in input_shape) or int(np.prod(input_shape)) > 16_777_216:
            raise ValueError("input_shape của model 3D không hợp lệ hoặc vượt giới hạn bộ nhớ.")
        labels = tuple(str(value).strip() for value in data.get("class_labels", []) if str(value).strip())
        if task != "volume_segmentation" and not labels:
            raise ValueError("Model 3D có phân loại phải khai báo class_labels.")
        threshold = float(data.get("mask_threshold", 0.5))
        if not 0.0 < threshold < 1.0:
            raise ValueError("mask_threshold phải nằm trong khoảng (0, 1).")
        input_modalities = tuple(str(value).strip().lower() for value in data.get("input_modalities", ["image"]))
        if not input_modalities or any(not value for value in input_modalities):
            raise ValueError("input_modalities không hợp lệ.")
        segmentation_labels = tuple(
            str(value).strip() for value in data.get("segmentation_labels", []) if str(value).strip()
        )
        activation = str(data.get("segmentation_activation", "sigmoid")).strip().lower()
        if activation not in {"sigmoid", "softmax"}:
            raise ValueError("segmentation_activation phải là sigmoid hoặc softmax.")
        return cls(
            task=task,
            class_labels=labels,
            input_shape=input_shape,
            mask_threshold=threshold,
            target=str(data.get("target", "brain")).strip().lower(),
            input_modalities=input_modalities,
            segmentation_labels=segmentation_labels,
            segmentation_activation=activation,
        )


@dataclass(frozen=True)
class VolumeFinding:
    label: str
    confidence: float
    bbox_voxel: tuple[int, int, int, int, int, int]
    bbox_mm: tuple[float, float, float, float, float, float]
    voxel_count: int
    volume_mm3: float


@dataclass(frozen=True)
class VolumeInferenceResult:
    predictions: tuple[dict[str, Any], ...]
    findings: tuple[VolumeFinding, ...]
    mask: np.ndarray | None


def _sitk_volume(image: Any, *, modality: str | None = None) -> VolumeData:
    import SimpleITK as sitk  # type: ignore[import-not-found]

    array = np.asarray(sitk.GetArrayFromImage(image), dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("Volume SimpleITK phải là ảnh vô hướng 3D; dữ liệu vector/time-series chưa được hỗ trợ.")
    spacing = tuple(float(value) for value in image.GetSpacing())
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return VolumeData(
        array=array,
        spacing_xyz=spacing,
        affine=affine,
        modality=modality,
        coordinate_system="LPS",
    )


def _normalized_tokens(path: Path) -> set[str]:
    name = path.name.lower().removesuffix(".gz").removesuffix(".nii")
    return {token for token in re.split(r"[^a-z0-9]+", name) if token}


def _dicom_series_tokens(files: tuple[str, ...] | list[str]) -> set[str]:
    if not files:
        return set()
    try:
        import pydicom

        dataset = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        text = " ".join(
            str(getattr(dataset, field, ""))
            for field in ("SeriesDescription", "ProtocolName", "SequenceName", "ContrastBolusAgent")
        ).lower()
        return {token for token in re.split(r"[^a-z0-9]+", text) if token}
    except Exception:
        return set()


def load_medical_volume(
    source: str | Path,
    *,
    max_voxels: int = 256_000_000,
    input_modalities: tuple[str, ...] = ("image",),
) -> VolumeData:
    path = Path(source)
    if path.is_dir():
        nifti_files = sorted(candidate for candidate in path.rglob("*") if candidate.name.lower().endswith((".nii", ".nii.gz")))
        if nifti_files:
            if input_modalities == ("image",):
                if len(nifti_files) != 1:
                    raise ValueError("Thư mục có nhiều NIfTI; manifest phải khai báo input_modalities theo đúng thứ tự.")
                return load_medical_volume(nifti_files[0], max_voxels=max_voxels)
            channels: list[VolumeData] = []
            for modality in input_modalities:
                matches = [candidate for candidate in nifti_files if modality in _normalized_tokens(candidate)]
                if len(matches) != 1:
                    raise ValueError(f"Cần đúng một file NIfTI cho modality {modality!r}, tìm thấy {len(matches)}.")
                channels.append(load_medical_volume(matches[0], max_voxels=max_voxels))
            reference = channels[0]
            if any(item.array.shape != reference.array.shape or not np.allclose(item.affine, reference.affine) for item in channels[1:]):
                raise ValueError("Các modality MRI 3D không cùng shape/affine; cần đăng ký ảnh trước suy luận.")
            stacked = np.stack([item.array for item in channels], axis=0)
            if stacked.size > max_voxels:
                raise ValueError("Volume đa kênh vượt giới hạn voxel cho phép.")
            return VolumeData(
                array=stacked,
                spacing_xyz=reference.spacing_xyz,
                affine=reference.affine,
                modality="MRI",
                coordinate_system=reference.coordinate_system,
            )
        try:
            import SimpleITK as sitk  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Cần SimpleITK để đọc DICOM series 3D.") from exc
        series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path)) or ()
        if not series_ids:
            raise ValueError("Thư mục không chứa DICOM series 3D hợp lệ.")
        series_files = {
            series_id: tuple(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path), series_id))
            for series_id in series_ids
        }
        if input_modalities == ("image",):
            if len(series_files) != 1:
                raise ValueError("Thư mục có nhiều DICOM series; manifest phải khai báo input_modalities/phase để chọn đúng thứ tự.")
            selected = [next(iter(series_files.values()))]
        else:
            selected = []
            used: set[str] = set()
            for modality in input_modalities:
                matches = [
                    (series_id, files)
                    for series_id, files in series_files.items()
                    if modality.lower() in _dicom_series_tokens(files) and series_id not in used
                ]
                if len(matches) != 1:
                    raise ValueError(f"Cần đúng một DICOM series cho modality/phase {modality!r}, tìm thấy {len(matches)}.")
                used.add(matches[0][0])
                selected.append(matches[0][1])

        volumes: list[VolumeData] = []
        for files in selected:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(files)
            image = reader.Execute()
            if int(np.prod(image.GetSize())) > max_voxels:
                raise ValueError("Volume vượt giới hạn voxel cho phép.")
            volumes.append(_sitk_volume(image))
        reference = volumes[0]
        if len(volumes) == 1:
            return reference
        if any(item.array.shape != reference.array.shape or not np.allclose(item.affine, reference.affine) for item in volumes[1:]):
            raise ValueError("Các DICOM series không cùng shape/affine; cần đăng ký ảnh trước suy luận.")
        stacked = np.stack([item.array for item in volumes], axis=0)
        if stacked.size > max_voxels:
            raise ValueError("DICOM đa series vượt giới hạn voxel cho phép.")
        return VolumeData(stacked, reference.spacing_xyz, reference.affine, coordinate_system=reference.coordinate_system)

    lower_name = path.name.lower()
    if lower_name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Cần nibabel để đọc NIfTI 3D.") from exc
        image = nib.load(str(path))
        if len(image.shape) != 3:
            raise ValueError("Mỗi file NIfTI phải là một volume 3D; hãy tách dữ liệu 4D thành từng modality.")
        shape = tuple(int(value) for value in image.shape[:3])
        if len(shape) != 3 or int(np.prod(shape)) > max_voxels:
            raise ValueError("Volume NIfTI không phải 3D hoặc vượt giới hạn voxel cho phép.")
        array_xyz = np.asarray(image.dataobj, dtype=np.float32)
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        return VolumeData(
            array=np.transpose(array_xyz, (2, 1, 0)),
            spacing_xyz=spacing,
            affine=np.asarray(image.affine, dtype=np.float64),
            coordinate_system="RAS+",
        )

    if lower_name.endswith((".mha", ".mhd")):
        try:
            import SimpleITK as sitk  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Cần SimpleITK để đọc MHA/MHD 3D.") from exc
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 3 or int(np.prod(image.GetSize())) > max_voxels:
            raise ValueError("Volume MHA/MHD không phải 3D hoặc vượt giới hạn voxel cho phép.")
        return _sitk_volume(image)

    raise ValueError(f"Định dạng volume 3D không được hỗ trợ: {path}")


def _normalize_volume(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 4:
        return np.stack([_normalize_volume(channel) for channel in values], axis=0)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Volume không chứa voxel hữu hạn.")
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        raise ValueError("Volume không có đủ tương phản để phân tích.")
    return np.clip((np.nan_to_num(values, nan=low) - low) / (high - low), 0.0, 1.0)


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    try:
        import SimpleITK as sitk  # type: ignore[import-not-found]

        connected = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
        relabeled = sitk.RelabelComponent(connected, sortByObjectSize=True)
        labels = sitk.GetArrayFromImage(relabeled)
        return [(labels == index) for index in range(1, min(int(labels.max()), 20) + 1)]
    except Exception:
        return [mask] if np.any(mask) else []


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        return None
    z1, y1, x1 = coordinates.min(axis=0)
    z2, y2, x2 = coordinates.max(axis=0) + 1
    return int(x1), int(y1), int(z1), int(x2), int(y2), int(z2)


def _bbox_mm(volume: VolumeData, bbox: tuple[int, int, int, int, int, int]) -> tuple[float, float, float, float, float, float]:
    x1, y1, z1, x2, y2, z2 = bbox
    corners = [
        volume.voxel_to_world((x, y, z))
        for x in (x1, x2)
        for y in (y1, y2)
        for z in (z1, z2)
    ]
    points = np.asarray(corners, dtype=np.float64)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    return tuple(float(value) for value in (*minimum, *maximum))  # type: ignore[return-value]


class TorchScriptVolumeRuntime:
    def __init__(self, model_path: str | Path, manifest_path: str | Path, *, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.manifest = VolumeModelManifest.load(manifest_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device).eval()
        except Exception as exc:
            raise RuntimeError(f"Không nạp được TorchScript model 3D: {self.model_path}") from exc

    @staticmethod
    def _outputs(raw: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        logits: torch.Tensor | None = None
        mask_logits: torch.Tensor | None = None
        if isinstance(raw, dict):
            logits = raw.get("classification_logits", raw.get("logits"))
            mask_logits = raw.get("segmentation_logits", raw.get("mask_logits", raw.get("mask")))
        elif isinstance(raw, (tuple, list)):
            logits = raw[0] if raw else None
            mask_logits = raw[1] if len(raw) > 1 else None
        elif isinstance(raw, torch.Tensor):
            if raw.ndim <= 2:
                logits = raw
            else:
                mask_logits = raw
        if logits is not None and not isinstance(logits, torch.Tensor):
            raise TypeError("classification_logits phải là Tensor.")
        if mask_logits is not None and not isinstance(mask_logits, torch.Tensor):
            raise TypeError("segmentation_logits phải là Tensor.")
        if logits is None and mask_logits is None:
            raise ValueError("Model 3D không trả về classification_logits hoặc segmentation_logits.")
        return logits, mask_logits

    @torch.inference_mode()
    def predict(self, volume: VolumeData) -> VolumeInferenceResult:
        normalized = _normalize_volume(volume.array)
        if normalized.ndim == 3:
            normalized = normalized[None, ...]
        if normalized.ndim != 4 or normalized.shape[0] != len(self.manifest.input_modalities):
            raise ValueError("Số kênh volume không khớp input_modalities trong manifest 3D.")
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        tensor = F.interpolate(tensor, size=self.manifest.input_shape, mode="trilinear", align_corners=False)
        logits, mask_logits = self._outputs(self.model(tensor))

        predictions: list[dict[str, Any]] = []
        if logits is not None:
            probabilities = torch.softmax(logits.reshape(1, -1), dim=1)[0].detach().cpu().numpy()
            if probabilities.size != len(self.manifest.class_labels):
                raise ValueError("Số đầu ra classification không khớp class_labels trong manifest 3D.")
            for index in np.argsort(-probabilities):
                predictions.append({"label": self.manifest.class_labels[index], "confidence": float(probabilities[index])})

        mask: np.ndarray | None = None
        probability_channels: np.ndarray | None = None
        if mask_logits is not None:
            if mask_logits.ndim != 5:
                raise ValueError("Đầu ra segmentation 3D phải có shape [N,C,D,H,W].")
            if self.manifest.segmentation_activation == "sigmoid":
                probabilities = torch.sigmoid(mask_logits)
            else:
                if mask_logits.shape[1] < 2:
                    raise ValueError("Segmentation softmax phải có background và ít nhất một lớp tổn thương.")
                probabilities = torch.softmax(mask_logits, dim=1)[:, 1:]
            spatial_shape = volume.array.shape[-3:]
            probabilities = F.interpolate(probabilities, size=spatial_shape, mode="trilinear", align_corners=False)
            probability_channels = probabilities[0].detach().cpu().numpy()
            mask = np.any(probability_channels >= self.manifest.mask_threshold, axis=0)

        findings: list[VolumeFinding] = []
        if mask is not None and probability_channels is not None:
            channel_count = probability_channels.shape[0]
            if self.manifest.segmentation_labels and len(self.manifest.segmentation_labels) != channel_count:
                raise ValueError("Số segmentation_labels không khớp số kênh mask của model 3D.")
            voxel_volume = float(np.prod(volume.spacing_xyz))
            for channel_index, probability_mask in enumerate(probability_channels):
                channel_mask = probability_mask >= self.manifest.mask_threshold
                label = (
                    self.manifest.segmentation_labels[channel_index]
                    if self.manifest.segmentation_labels
                    else predictions[0]["label"] if channel_count == 1 and predictions else f"lesion_{channel_index + 1}"
                )
                for component in _component_masks(channel_mask):
                    bbox = _bbox_from_mask(component)
                    if bbox is None:
                        continue
                    confidence = float(np.mean(probability_mask[component]))
                    count = int(np.count_nonzero(component))
                    findings.append(
                        VolumeFinding(
                            label=str(label),
                            confidence=confidence,
                            bbox_voxel=bbox,
                            bbox_mm=_bbox_mm(volume, bbox),
                            voxel_count=count,
                            volume_mm3=count * voxel_volume,
                        )
                    )
        return VolumeInferenceResult(predictions=tuple(predictions), findings=tuple(findings), mask=mask)


def render_volume_overlay(
    volume: VolumeData,
    result: VolumeInferenceResult,
    normalized_dir: str | Path,
    overlay_dir: str | Path,
    *,
    patient_code: str,
) -> tuple[Path, Path]:
    normalized_output = Path(normalized_dir)
    overlay_output = Path(overlay_dir)
    normalized_output.mkdir(parents=True, exist_ok=True)
    overlay_output.mkdir(parents=True, exist_ok=True)
    if result.findings:
        z_index = (result.findings[0].bbox_voxel[2] + result.findings[0].bbox_voxel[5] - 1) // 2
    else:
        z_index = volume.array.shape[0] // 2
    display_volume = volume.array[0] if volume.array.ndim == 4 else volume.array
    normalized = (_normalize_volume(display_volume[z_index]) * 255).astype(np.uint8)
    base = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()
    for finding in result.findings:
        x1, y1, z1, x2, y2, z2 = finding.bbox_voxel
        if z1 <= z_index < z2:
            cv2.rectangle(overlay, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), (0, 0, 255), 2)
    safe_code = "".join(char if char.isalnum() or char in "-_" else "_" for char in patient_code)
    stamp = uuid4().hex[:8]
    normalized_path = normalized_output / f"{safe_code}_{stamp}_volume_slice_z{z_index}.png"
    overlay_path = overlay_output / f"{safe_code}_{stamp}_volume_overlay_z{z_index}.png"
    cv2.imwrite(str(normalized_path), base)
    cv2.imwrite(str(overlay_path), overlay)
    return normalized_path, overlay_path


__all__ = [
    "TorchScriptVolumeRuntime",
    "VolumeData",
    "VolumeFinding",
    "VolumeInferenceResult",
    "VolumeModelManifest",
    "load_medical_volume",
    "render_volume_overlay",
]
