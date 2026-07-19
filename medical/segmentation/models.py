"""Segmentation models and ROI extraction for medical images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    confidence: float
    modality: str = "default"


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


class AttentionGate(nn.Module):
    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=2, stride=2, padding=0)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        theta_x = self.theta(x)
        phi_g = F.interpolate(self.phi(gating), size=theta_x.shape[2:], mode="bilinear", align_corners=False)
        psi = self.relu(theta_x + phi_g)
        psi = self.sigmoid(self.psi(psi))
        psi = self.upsample(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, num_classes: int = 1, in_channels: int = 3) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.att4 = AttentionGate(512, 1024, 256)
        self.att3 = AttentionGate(256, 512, 128)
        self.att2 = AttentionGate(128, 256, 64)
        self.att1 = AttentionGate(64, 128, 32)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        a4 = self.att4(e4, b)
        d4 = self.dec4(torch.cat([self.up4(b), a4], dim=1))
        a3 = self.att3(e3, d4)
        d3 = self.dec3(torch.cat([self.up3(d4), a3], dim=1))
        a2 = self.att2(e2, d3)
        d2 = self.dec2(torch.cat([self.up2(d3), a2], dim=1))
        a1 = self.att1(e1, d2)
        d1 = self.dec1(torch.cat([self.up1(d2), a1], dim=1))
        return self.final(d1)


class SAMROIExtractor:
    def __init__(self, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._sam = None

    def _load_sam(self) -> Any:
        if self._sam is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                model_type = "vit_b"
                checkpoint = None
                for candidate in ["sam_vit_b.pth", "sam_b.pth"]:
                    if Path(candidate).exists():
                        checkpoint = candidate
                        break
                if checkpoint is None:
                    raise FileNotFoundError("SAM checkpoint not found")
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                sam.to(self.device)
                self._sam = SamPredictor(sam)
            except Exception:
                self._sam = False
        return self._sam if self._sam is not None else None

    def extract_roi(self, image: np.ndarray, points: list[tuple[int, int]] | None = None) -> SegmentationResult | None:
        predictor = self._load_sam()
        if predictor is None:
            return self._fallback_roi(image)
        try:
            predictor.set_image(image)
            if not points:
                h, w = image.shape[:2]
                points = [(w // 2, h // 2), (w // 4, h // 4), (3 * w // 4, 3 * h // 4)]
            input_points = np.array(points)
            input_labels = np.ones(len(points))
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            if len(masks) == 0:
                return self._fallback_roi(image)
            best = int(np.argmax(scores))
            mask = masks[best].astype(np.uint8) * 255
            y_coords, x_coords = np.where(mask > 0)
            if len(y_coords) == 0:
                return self._fallback_roi(image)
            y1, y2 = int(y_coords.min()), int(y_coords.max())
            x1, x2 = int(x_coords.min()), int(x_coords.max())
            bbox = (x1, y1, x2, y2)
            area = float(mask.sum()) / (mask.shape[0] * mask.shape[1])
            return SegmentationResult(mask=mask, bbox=bbox, area=area, confidence=float(scores[best]))
        except Exception:
            return self._fallback_roi(image)

    def _fallback_roi(self, image: np.ndarray) -> SegmentationResult:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(largest)
            bbox = (x, y, x + w, y + h)
            area = float(cv2.contourArea(largest)) / (gray.shape[0] * gray.shape[1])
        else:
            h, w = gray.shape
            bbox = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
            area = 0.5
        return SegmentationResult(mask=mask, bbox=bbox, area=area, confidence=0.8 if contours else 0.5)


def crop_to_roi(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 10) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


__all__ = [
    "SegmentationResult",
    "UNet",
    "AttentionUNet",
    "SAMROIExtractor",
    "crop_to_roi",
]
