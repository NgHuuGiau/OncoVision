from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Sequence

from medical.cnn_classifier import (
    MedicalCNNClassifierWrapper,
    _load_image_as_tensor,
)


# ---------------------------------------------------------------------------
# Lop ngoai le danh rieng cho Grad-CAM
# ---------------------------------------------------------------------------
class GradCAMError(Exception):
    """Ngoai le co ban cho quy trinh Grad-CAM."""


class GradCAMUnsupportedError(GradCAMError):
    """Ngoai le khi backbone khong ho tro trich xuat Grad-CAM."""


# Cac backbone duoc ho tro boi Grad-CAM
_SUPPORTED_BACKBONES: tuple[str, ...] = (
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b2",
    "efficientnet_b3",
    "convnext_tiny",
)


def _get_target_conv_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """Tim lop tich chap cuoi cung cua backbone de gan hook.

    Vi tri lop cuoi cung phu thuoc vao kien truc tung backbone trong
    torchvision, nen ta phai tra cuu rieng cho tung loai.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise GradCAMUnsupportedError(
            "Mo hinh khong co thuoc tinh 'backbone' de xac dinh lop tich chap."
        )

    if backbone_name in {"resnet18", "resnet50"}:
        # ResNet: layer4 la block tich chap cuoi cung truoc khi GAP + fc.
        target = backbone.layer4
    elif backbone_name in {"efficientnet_b0", "efficientnet_b2", "efficientnet_b3"}:
        # EfficientNet: features[-1] la khoi tich chap cuoi cung co dac trung khong gian.
        target = backbone.features[-1]
    elif backbone_name == "convnext_tiny":
        # ConvNeXt: features[-1] la stage cuoi cung, dau ra van giu nguyen khong gian.
        target = backbone.features[-1]
    else:
        raise GradCAMUnsupportedError(
            f"Backbone '{backbone_name}' khong ho tro Grad-CAM. "
            f"Cac backbone ho tro: {', '.join(_SUPPORTED_BACKBONES)}."
        )

    if not isinstance(target, nn.Module):
        raise GradCAMUnsupportedError(
            f"Khong the lay lop tich chap cho backbone '{backbone_name}'."
        )
    return target


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Ap dung colormap 'jet' cho heatmap (khong can thu vien ngoai).

    heatmap: mang float trong [0, 1] kich thuoc (H, W).
    Tra ve mang uint8 (H, W, 3) theo thu tu kenh RGB.
    """
    x = 4.0 * np.clip(heatmap, 0.0, 1.0)
    r = np.clip(np.minimum(x - 1.5, -x + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(x - 0.5, -x + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(x + 0.5, -x + 2.5), 0.0, 1.0)
    cmap = np.stack([r, g, b], axis=-1)
    return (cmap * 255.0).astype(np.uint8)


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize heatmap (H, W) ve kich thuoc moi (W, H) bang noi suy song tuyen.

    Chi dung numpy + torch de tranh them thu vien phu thuoc.
    """
    h, w = heatmap.shape
    if (h, w) == size:
        return heatmap
    tensor = torch.from_numpy(heatmap).to(torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


@dataclass(frozen=True)
class GradCAMResult:
    """Ket qua giai thich cho mot lop du doan cu the."""

    label: str
    confidence: float
    heatmap: np.ndarray  # (H, W) float trong [0, 1]
    overlay: np.ndarray  # (H, W, 3) uint8 RGB


class GradCAM:
    """Trien khai Gradient-weighted Class Activation Mapping (Grad-CAM).

    Lop nay hoat dong voi moi backbone duoc ho tro cua OncoVision bang cach
    gan forward hook len lop tich chap cuoi cung de lay dac trung kich hoat,
    va backward hook de lay gradient, tu do tao heatmap tai vi tri cua lop do.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    # --- Cac hook ----------------------------------------------------------
    def _forward_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        # Luu lai dac trung kich hoat (B, C, H, W) tai lop cuoi cung.
        self.activations = output.detach()

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        # Luu gradient theo dau ra cua lop (B, C, H, W).
        self.gradients = grad_output[0].detach()

    # --- Tinh heatmap ------------------------------------------------------
    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
    ) -> np.ndarray:
        """Tinh heatmap Grad-CAM cho mot lop cu the.

        Tra ve mang numpy (H, W) da duoc chuan hoa ve [0, 1], voi (H, W) bang
        kich thuoc khong gian cua anh dau vao (da duoc noi suy len).
        """
        if self.activations is None or self.gradients is None:
            raise GradCAMError(
                "Chua co dac trung hoac gradient; hay goi generate() sau mot lan forward+backward."
            )

        # Dua tensor len thiet bi va dam bao co dinh dang (1, 3, H, W).
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)

        self.model.zero_grad(set_to_none=True)

        # Forward de lay dac trung tai lop dich.
        output = self.model(x)
        if class_idx < 0 or class_idx >= output.shape[1]:
            raise GradCAMError(
                f"class_idx {class_idx} nam ngoai pham vi [0, {output.shape[1] - 1}]."
            )

        # Backward chi tren diem so cua lop muon phan tich.
        score = output[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        activations = self.activations  # (B, C, h, w)
        gradients = self.gradients  # (B, C, h, w)

        # Trong so: trung binh gradient theo khong gian cho tung kenh.
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, h, w)
        cam = torch.relu(cam)

        # Noi suy len dung kich thuoc anh dau vao.
        cam = F.interpolate(
            cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Chuan hoa ve [0, 1].
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)

        self.model.zero_grad(set_to_none=True)
        return cam.astype(np.float32)

    # --- Don dep -----------------------------------------------------------
    def remove_hooks(self) -> None:
        """Go bo cac hook da dang ky de tranh ro ri bo nho."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
        if self._backward_handle is not None:
            self._backward_handle.remove()
        self._forward_handle = None
        self._backward_handle = None

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    *,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """Tao anh overlay tu anh goc va heatmap Grad-CAM.

    Tham so:
        image: anh goc numpy (H, W, 3), co the la RGB hoac BGR.
        heatmap: mang numpy (H, W) trong [0, 1].
        alpha: he so pha tron (0 = chi anh goc, 1 = chi heatmap).
        colormap: ten bang mau hien tai chi ho tro 'jet'.

    Tra ve anh uint8 (H, W, 3) theo thu tu kenh RGB.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Anh dau vao phai co kich thuoc (H, W, 3).")
    if heatmap.ndim != 2:
        raise ValueError("Heatmap phai co kich thuoc (H, W).")

    h, w = image.shape[:2]

    # Resize heatmap ve dung kich thuoc anh goc neu can.
    if heatmap.shape != (h, w):
        heatmap = _resize_heatmap(heatmap, (w, h))

    if colormap == "jet":
        colored = _jet_colormap(heatmap)
    else:
        raise ValueError(f"Colormap '{colormap}' chua duoc ho tro (chi ho tro 'jet').")

    base = image.astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = (alpha * colored.astype(np.float32) + (1.0 - alpha) * base).astype(np.uint8)
    return overlay


def _load_raw_rgb(source: str | np.ndarray) -> np.ndarray:
    """Tai anh goc ve mang uint8 RGB (H, W, 3) phuc vu cho viec ve overlay.

    Voi numpy array, ta gioi thieu dau vao la BGR (cung quy uoc voi wrapper)
    nen se doi sang RGB de hien thi cho dung mau.
    """
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[:, :, ::-1]  # BGR -> RGB
        return arr.astype(np.uint8)

    from PIL import Image, ImageOps

    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img, dtype=np.uint8)


# Cac phep TTA va nghich dao tuong ung de canh heatmap ve lai toa do anh goc.
_TTA_SPECS: tuple[
    tuple[str, Callable[[torch.Tensor], torch.Tensor], Callable[[np.ndarray], np.ndarray]],
    ...,
] = (
    ("identity", lambda x: x, lambda h: h),
    ("hflip", lambda x: torch.flip(x, dims=[3]), lambda h: np.flip(h, axis=1)),
    ("vflip", lambda x: torch.flip(x, dims=[2]), lambda h: np.flip(h, axis=0)),
    ("roll_w", lambda x: torch.roll(x, shifts=10, dims=[3]), lambda h: np.roll(h, shift=-10, axis=1)),
    ("roll_h", lambda x: torch.roll(x, shifts=10, dims=[2]), lambda h: np.roll(h, shift=-10, axis=0)),
)


class MedicalGradCAMExplainer:
    """Wrapper giai thich Grad-CAM cho MedicalCNNClassifierWrapper.

    Cung cap phuong thuc explain() tra ve danh sach cac GradCAMResult cho
    top-k lop co xac suat cao nhat, ho tro TTA va tu xu ly nep neu backbone
    khong ho tro Grad-CAM.
    """

    def __init__(
        self,
        wrapper: MedicalCNNClassifierWrapper,
        image_size: int = 320,
        device: str | None = None,
    ) -> None:
        self.wrapper = wrapper
        self.image_size = image_size
        self.device = device or wrapper.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.gradcam: GradCAM | None = None
        self.unsupported_reason: str | None = None
        self._build_gradcam()

    def _build_gradcam(self) -> None:
        try:
            model = self.wrapper.model
            backbone_name = getattr(model, "backbone_name", None)
            if backbone_name is None or backbone_name not in _SUPPORTED_BACKBONES:
                raise GradCAMUnsupportedError(
                    f"Backbone '{backbone_name}' khong nam trong danh sach ho tro Grad-CAM."
                )
            target_layer = _get_target_conv_layer(model, backbone_name)
            self.gradcam = GradCAM(model, target_layer, device=self.device)
        except GradCAMUnsupportedError as exc:
            self.gradcam = None
            self.unsupported_reason = str(exc)

    @property
    def is_supported(self) -> bool:
        return self.gradcam is not None

    def explain(
        self,
        image: str | np.ndarray,
        top_k: int = 1,
        *,
        tta: bool = False,
        alpha: float = 0.5,
    ) -> list[GradCAMResult]:
        """Giai thich anh dau vao cho top-k lop du doan.

        Tra ve danh sach cac GradCAMResult voi cac truong:
        label, confidence, heatmap, overlay.
        """
        raw_image = _load_raw_rgb(image)
        h, w = raw_image.shape[:2]

        # Tinh xac suat va lay top-k lop bang chinh wrapper (dam bao nhat quan).
        preds = self.wrapper.predict(image, top_k=max(1, top_k), tta=tta)
        class_labels: Sequence[str] = self.wrapper.class_labels
        probabilities = preds[0]["probabilities"]
        probs_vec = np.array(
            [float(probabilities[label]) for label in class_labels], dtype=np.float32
        )
        top_indices = np.argsort(-probs_vec)[: max(1, top_k)]

        # Tensor dau vao cho mo hinh (tuan thu quy uoc BGR cua wrapper).
        input_tensor = _load_image_as_tensor(image, image_size=self.image_size, assume_bgr=True)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        results: list[GradCAMResult] = []
        for idx in top_indices:
            idx_int = int(idx)
            if self.gradcam is not None:
                heatmap = self._compute_heatmap(input_tensor, idx_int, tta=tta)
            else:
                # Neu khong ho tro, tra ve heatmap rong de khong lam dung luong.
                heatmap = np.zeros((h, w), dtype=np.float32)

            # Resize heatmap ve dung kich thuoc anh goc truoc khi overlay.
            if heatmap.shape != (h, w):
                heatmap = _resize_heatmap(heatmap, (w, h))

            overlay = generate_gradcam_overlay(raw_image, heatmap, alpha=alpha)
            label = class_labels[idx_int]
            confidence = float(probs_vec[idx_int])
            results.append(
                GradCAMResult(
                    label=label,
                    confidence=confidence,
                    heatmap=heatmap,
                    overlay=overlay,
                )
            )

        return results

    def _compute_heatmap(
        self,
        input_tensor: torch.Tensor,
        class_idx: int,
        *,
        tta: bool = False,
    ) -> np.ndarray:
        """Tinh heatmap cho mot lop, co the ket hop TTA bang cach trung binh.

        Voi TTA, moi phep bien doi duoc ap dung len anh dau vao de tinh heatmap,
        sau do heatmap duoc nghich dao ve toa do anh goc roi lay trung binh.
        """
        assert self.gradcam is not None, "GradCAM chua duoc khoi tao."
        if not tta:
            return self.gradcam.generate(input_tensor, class_idx)

        heatmaps: list[np.ndarray] = []
        for _, transform, inverse in _TTA_SPECS:
            aug_tensor = transform(input_tensor)
            cam = self.gradcam.generate(aug_tensor, class_idx)
            # Nghich dao phep bien doi khong gian tren chinh heatmap.
            cam = inverse(cam)
            heatmaps.append(cam)

        if not heatmaps:
            return self.gradcam.generate(input_tensor, class_idx)
        stacked = np.stack(heatmaps, axis=0)
        return np.clip(stacked.mean(axis=0), 0.0, 1.0).astype(np.float32)

    def remove_hooks(self) -> None:
        """Go bo hook cua doi tuong GradCAM ben trong (neu co)."""
        if self.gradcam is not None:
            self.gradcam.remove_hooks()


__all__ = [
    "GradCAMError",
    "GradCAMUnsupportedError",
    "GradCAM",
    "GradCAMResult",
    "generate_gradcam_overlay",
    "MedicalGradCAMExplainer",
    "_get_target_conv_layer",
    "_jet_colormap",
    "_resize_heatmap",
]
