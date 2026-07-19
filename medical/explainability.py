from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Sequence

from medical.cnn_classifier import MedicalCNNClassifierWrapper, _load_image_as_tensor


class GradCAMError(Exception):
    pass


class GradCAMUnsupportedError(GradCAMError):
    pass


_SUPPORTED_BACKBONES: tuple[str, ...] = (
    "resnet18", "resnet50",
    "efficientnet_b0", "efficientnet_b2", "efficientnet_b3",
    "convnext_tiny",
    "swin_t", "swin_s", "swin_b",
    "vit_b_16", "vit_l_16", "vit_h_14",
)


def _get_target_conv_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise GradCAMUnsupportedError("Mo hinh khong co thuoc tinh 'backbone'.")
    if backbone_name in {"resnet18", "resnet50"}:
        return backbone.layer4
    elif backbone_name in {"efficientnet_b0", "efficientnet_b2", "efficientnet_b3"}:
        return backbone.features[-1]
    elif backbone_name == "convnext_tiny":
        return backbone.features[-1]
    elif backbone_name.startswith("swin_"):
        return backbone.features[-1][-1].mlp
    elif backbone_name.startswith("vit_"):
        return backbone.encoder.layers[-1].mlp
    raise GradCAMUnsupportedError(f"Backbone '{backbone_name}' khong ho tro Grad-CAM.")


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    x = 4.0 * np.clip(heatmap, 0.0, 1.0)
    r = np.clip(np.minimum(x - 1.5, -x + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(x - 0.5, -x + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(x + 0.5, -x + 2.5), 0.0, 1.0)
    cmap = np.stack([r, g, b], axis=-1)
    return (cmap * 255.0).astype(np.uint8)


def _resize_heatmap(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = heatmap.shape
    if (h, w) == size:
        return heatmap
    tensor = torch.from_numpy(heatmap).to(torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze().numpy()


@dataclass(frozen=True)
class GradCAMResult:
    label: str
    confidence: float
    heatmap: np.ndarray
    overlay: np.ndarray


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(self, module: nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise GradCAMError("Chua co dac trung hoac gradient.")
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        if class_idx < 0 or class_idx >= output.shape[1]:
            raise GradCAMError(f"class_idx {class_idx} nam ngoai pham vi.")
        score = output[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        self.model.zero_grad(set_to_none=True)
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
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


class GradCAMPlusPlus(GradCAM):
    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise GradCAMError("Chua co dac trung hoac gradient.")
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        score = output[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)
        activations = self.activations
        gradients = self.gradients
        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients
        sum_activations = (activations * grads_power_2).sum(dim=(2, 3), keepdim=True) + 1e-6
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-6)
        weights = alpha * torch.relu(gradients)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        self.model.zero_grad(set_to_none=True)
        return cam.astype(np.float32)


class EigenCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.activations: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        if self.activations is None:
            raise GradCAMError("Chua co activations.")
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            _ = self.model(x)
        acts = self.activations.squeeze(0).cpu().numpy()
        reshaped = acts.reshape(acts.shape[0], -1)
        reshaped = reshaped - reshaped.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(reshaped, full_matrices=False)
        principal = Vt[0].reshape(acts.shape[1], acts.shape[2])
        cam = np.maximum(principal, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        cam = _resize_heatmap(cam, (x.shape[3], x.shape[2]))
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


class ScoreCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.activations: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        if self.activations is None:
            raise GradCAMError("Chua co activations.")
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            base_output = self.model(x)
            base_score = torch.softmax(base_output, dim=1)[0, class_idx].item()
        acts = self.activations.squeeze(0)
        cams = []
        for i in range(acts.shape[0]):
            mask = torch.sigmoid(acts[i].unsqueeze(0).unsqueeze(0))
            mask = F.interpolate(mask, size=x.shape[2:], mode="bilinear", align_corners=False)
            masked = x * mask
            with torch.no_grad():
                out = self.model(masked)
                score = torch.softmax(out, dim=1)[0, class_idx].item()
            cams.append(max(0.0, score - base_score) * acts[i].cpu().numpy())
        if not cams:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
        cam = np.mean(cams, axis=0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass


class AttentionRollout:
    def __init__(self, model: nn.Module, device: str | None = None) -> None:
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.attentions: list[torch.Tensor] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        _, attn_weights = output
        self.attentions.append(attn_weights.detach())

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.attentions = []
        x = input_tensor.to(self.device)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            _ = self.model(x)
        if not self.attentions:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
        rollout = self.attentions[0].mean(dim=1).cpu().numpy()[0]
        for attn in self.attentions[1:]:
            attn_mean = attn.mean(dim=1).cpu().numpy()[0]
            rollout = rollout @ attn_mean
        rollout = rollout.reshape(int(np.sqrt(rollout.shape[0])), -1)
        if rollout.max() > rollout.min():
            rollout = (rollout - rollout.min()) / (rollout.max() - rollout.min())
        rollout = _resize_heatmap(rollout, (x.shape[3], x.shape[2]))
        return rollout.astype(np.float32)

    def remove_hooks(self) -> None:
        for module in self.model.modules():
            if isinstance(module, nn.MultiheadAttention):
                for handle in module._forward_hooks.values():
                    handle.remove()


def generate_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.5, colormap: str = "jet") -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Anh dau vao phai co kich thuoc (H, W, 3).")
    if heatmap.ndim != 2:
        raise ValueError("Heatmap phai co kich thuoc (H, W).")
    h, w = image.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = _resize_heatmap(heatmap, (w, h))
    if colormap == "jet":
        colored = _jet_colormap(heatmap)
    else:
        raise ValueError(f"Colormap '{colormap}' chua duoc ho tro.")
    base = image.astype(np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    overlay = (alpha * colored.astype(np.float32) + (1.0 - alpha) * base).astype(np.uint8)
    return overlay


def _load_raw_rgb(source: str | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = arr[:, :, ::-1]
        return arr.astype(np.uint8)
    from PIL import Image, ImageOps
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img, dtype=np.uint8)


_TTA_SPECS: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor], Callable[[np.ndarray], np.ndarray]], ...] = (
    ("identity", lambda x: x, lambda h: h),
    ("hflip", lambda x: torch.flip(x, dims=[3]), lambda h: np.flip(h, axis=1)),
    ("vflip", lambda x: torch.flip(x, dims=[2]), lambda h: np.flip(h, axis=0)),
    ("roll_w", lambda x: torch.roll(x, shifts=10, dims=[3]), lambda h: np.roll(h, shift=-10, axis=1)),
    ("roll_h", lambda x: torch.roll(x, shifts=10, dims=[2]), lambda h: np.roll(h, shift=-10, axis=0)),
)


class MedicalGradCAMExplainer:
    def __init__(self, wrapper: MedicalCNNClassifierWrapper, image_size: int = 320, device: str | None = None) -> None:
        self.wrapper = wrapper
        self.image_size = image_size
        self.device = device or wrapper.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gradcam: GradCAM | None = None
        self.gradcam_pp: GradCAMPlusPlus | None = None
        self.eigen_cam: EigenCAM | None = None
        self.score_cam: ScoreCAM | None = None
        self.attention_rollout: AttentionRollout | None = None
        self.unsupported_reason: str | None = None
        self._build_explainers()

    def _build_explainers(self) -> None:
        try:
            model = self.wrapper.model
            backbone_name = getattr(model, "backbone_name", None)
            if backbone_name is None or backbone_name not in _SUPPORTED_BACKBONES:
                raise GradCAMUnsupportedError(f"Backbone '{backbone_name}' khong nam trong danh sach ho tro.")
            target_layer = _get_target_conv_layer(model, backbone_name)
            self.gradcam = GradCAM(model, target_layer, device=self.device)
            self.gradcam_pp = GradCAMPlusPlus(model, target_layer, device=self.device)
            self.eigen_cam = EigenCAM(model, target_layer, device=self.device)
            try:
                self.score_cam = ScoreCAM(model, target_layer, device=self.device)
            except Exception:
                self.score_cam = None
            if backbone_name.startswith("swin_") or backbone_name.startswith("vit_"):
                self.attention_rollout = AttentionRollout(model, device=self.device)
        except GradCAMUnsupportedError as exc:
            self.unsupported_reason = str(exc)

    @property
    def is_supported(self) -> bool:
        return self.gradcam is not None

    def explain(self, image: str | np.ndarray, top_k: int = 1, *, tta: bool = False, alpha: float = 0.5, methods: list[str] | None = None) -> list[GradCAMResult]:
        raw_image = _load_raw_rgb(image)
        h, w = raw_image.shape[:2]
        preds = self.wrapper.predict(image, top_k=max(1, top_k), tta=tta)
        class_labels: Sequence[str] = self.wrapper.class_labels
        probabilities = preds[0]["probabilities"]
        probs_vec = np.array([float(probabilities[label]) for label in class_labels], dtype=np.float32)
        top_indices = np.argsort(-probs_vec)[: max(1, top_k)]
        input_tensor = _load_image_as_tensor(image, image_size=self.image_size, assume_bgr=True)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        methods = methods or ["gradcam"]
        results = []
        for idx in top_indices:
            idx_int = int(idx)
            heatmap = np.zeros((h, w), dtype=np.float32)
            if "gradcam" in methods and self.gradcam is not None:
                heatmap = self.gradcam.generate(input_tensor, idx_int)
            elif "gradcam_pp" in methods and self.gradcam_pp is not None:
                heatmap = self.gradcam_pp.generate(input_tensor, idx_int)
            if "eigen_cam" in methods and self.eigen_cam is not None:
                heatmap = self.eigen_cam.generate(input_tensor)
            if "score_cam" in methods and self.score_cam is not None:
                heatmap = self.score_cam.generate(input_tensor, idx_int)
            if "attention_rollout" in methods and self.attention_rollout is not None:
                heatmap = self.attention_rollout.generate(input_tensor)
            if heatmap.shape != (h, w):
                heatmap = _resize_heatmap(heatmap, (w, h))
            overlay = generate_gradcam_overlay(raw_image, heatmap, alpha=alpha)
            label = class_labels[idx_int]
            confidence = float(probs_vec[idx_int])
            results.append(GradCAMResult(label=label, confidence=confidence, heatmap=heatmap, overlay=overlay))
        return results

    def remove_hooks(self) -> None:
        if self.gradcam is not None:
            self.gradcam.remove_hooks()
        if self.gradcam_pp is not None:
            self.gradcam_pp.remove_hooks()
        if self.eigen_cam is not None:
            self.eigen_cam.remove_hooks()
        if self.score_cam is not None:
            self.score_cam.remove_hooks()
        if self.attention_rollout is not None:
            self.attention_rollout.remove_hooks()


__all__ = [
    "GradCAMError", "GradCAMUnsupportedError", "GradCAM", "GradCAMPlusPlus", "EigenCAM", "ScoreCAM", "AttentionRollout",
    "GradCAMResult", "generate_gradcam_overlay", "MedicalGradCAMExplainer", "_get_target_conv_layer",
    "_jet_colormap", "_resize_heatmap",
]
