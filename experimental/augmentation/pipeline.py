from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentationConfig:
    image_size: int = 320
    rand_augment_n: int = 2
    rand_augment_m: int = 10
    enable_mixup: bool = True
    mixup_alpha: float = 0.2
    enable_cutmix: bool = True
    cutmix_alpha: float = 1.0
    enable_cutout: bool = True
    cutout_n_holes: int = 1
    cutout_length: int = 16
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    motion_blur_kernel: int = 5
    gaussian_noise_std: float = 0.02
    intensity_shift_range: float = 0.1
    bias_field_simulation: bool = True
    modality: str = "default"
    enable_gridmask: bool = True
    gridmask_d1: float = 0.1
    gridmask_d2: float = 0.3
    gridmask_rotate: int = 360
    enable_random_erasing: bool = True
    random_erasing_p: float = 0.15
    random_erasing_scale: tuple[float, float] = (0.02, 0.15)
    random_erasing_ratio: tuple[float, float] = (0.3, 3.3)


class MedicalAugmentationPipeline:
    def __init__(self, config: AugmentationConfig | None = None) -> None:
        self.config = config or AugmentationConfig()

    def __call__(self, image: np.ndarray, label: int | None = None) -> tuple[np.ndarray, int | None]:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image.astype(np.float32) / 255.0
        image = self._apply_rand_augment(image)
        if self.config.enable_gridmask:
            image = self._apply_gridmask(image)
        image = self._apply_elastic_deformation(image)
        image = self._apply_gaussian_noise(image)
        image = self._apply_intensity_shift(image)
        if self.config.bias_field_simulation:
            image = self._apply_bias_field_simulation(image)
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255).astype(np.uint8)
        if self.config.enable_random_erasing:
            image = self._apply_random_erasing(image)
        if label is None:
            return image
        return image, label

    def _apply_rand_augment(self, image: np.ndarray) -> np.ndarray:
        n = self.config.rand_augment_n
        m = self.config.rand_augment_m / 10.0
        ops = [
            lambda x: self._random_rotate(x),
            lambda x: self._random_brightness(x, m * 0.3),
            lambda x: self._random_contrast(x, 1.0 + m * 0.3),
            lambda x: self._random_gamma(x, 1.0 + m * 0.3),
            lambda x: self._random_gaussian_blur(x),
            lambda x: self._random_motion_blur(x),
            lambda x: self._random_clahe_variation(x),
            lambda x: self._random_sharpness(x, m * 0.3),
            lambda x: self._random_autocontrast(x),
            lambda x: self._random_equalize(x),
            lambda x: self._random_solarize(x, m * 0.2),
            lambda x: self._random_posterize(x, int(m * 2) + 1),
            lambda x: self._random_color_jitter(x, m * 0.2),
        ]
        import random
        selected = random.sample(ops, min(n, len(ops)))
        for op in selected:
            image = op(image)
        return image

    def _apply_gridmask(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        h, w = image.shape[:2]
        d1 = int(min(h, w) * self.config.gridmask_d1)
        d2 = int(min(h, w) * self.config.gridmask_d2)
        delta = np.random.randint(d1, d2)
        angle = np.random.randint(0, self.config.gridmask_rotate)
        mask = np.ones((h, w), dtype=np.float32)
        if angle % 90 == 0:
            for i in range(0, h, delta):
                mask[i:i + max(1, delta // 2), :] = 0
            for j in range(0, w, delta):
                mask[:, j:j + max(1, delta // 2)] = 0
        else:
            for i in range(-max(h, w), max(h, w), delta):
                cv2.line(mask, (i, 0), (i + h, h), 0, max(1, delta // 2))
        mask = np.clip(mask, 0, 1)
        if image.ndim == 3:
            mask = mask[..., np.newaxis]
        return image * mask

    def _apply_random_erasing(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.config.random_erasing_p:
            return image
        h, w = image.shape[:2]
        scale_min, scale_max = self.config.random_erasing_scale
        ratio_min, ratio_max = self.config.random_erasing_ratio
        area = h * w
        target_area = np.random.uniform(scale_min, scale_max) * area
        aspect_ratio = np.random.uniform(ratio_min, ratio_max)
        erase_h = int(np.sqrt(target_area / aspect_ratio))
        erase_w = int(np.sqrt(target_area * aspect_ratio))
        if erase_h >= h or erase_w >= w:
            return image
        x1 = np.random.randint(0, w - erase_w)
        y1 = np.random.randint(0, h - erase_h)
        image[y1:y1 + erase_h, x1:x1 + erase_w] = np.random.uniform(0, 255, (erase_h, erase_w, image.shape[-1])).astype(np.uint8) if image.ndim == 3 else np.random.randint(0, 255, (erase_h, erase_w))
        return image

    def _random_rotate(self, image: np.ndarray, angle: float = 15.0) -> np.ndarray:
        import random
        angle = random.uniform(-angle, angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _random_brightness(self, image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        import random
        factor = random.uniform(-factor, factor)
        return np.clip(image + factor, 0, 1)

    def _random_contrast(self, image: np.ndarray, factor_range: float = 1.3) -> np.ndarray:
        import random
        factor = random.uniform(1.0 / factor_range, factor_range)
        mean = np.mean(image)
        return np.clip((image - mean) * factor + mean, 0, 1)

    def _random_gamma(self, image: np.ndarray, gamma_range: float = 1.3) -> np.ndarray:
        import random
        gamma = random.uniform(1.0 / gamma_range, gamma_range)
        return np.power(image, gamma)

    def _random_sharpness(self, image: np.ndarray, factor_range: float = 0.3) -> np.ndarray:
        import random
        factor = random.uniform(0.0, factor_range)
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return np.clip(image + factor * (image - blurred), 0, 1)

    def _random_autocontrast(self, image: np.ndarray) -> np.ndarray:
        min_val, max_val = np.percentile(image, (2, 98))
        if max_val > min_val:
            return np.clip((image - min_val) / (max_val - min_val), 0, 1)
        return image

    def _random_equalize(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        result = image.copy()
        for c in range(image.shape[-1]):
            channel = (image[..., c] * 255).astype(np.uint8)
            result[..., c] = cv2.equalizeHist(channel) / 255.0
        return result

    def _random_solarize(self, image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
        if np.random.rand() > 0.3:
            return image
        threshold_val = np.random.uniform(threshold, 1.0 - threshold)
        return np.where(image < threshold_val, image, 1.0 - image)

    def _random_posterize(self, image: np.ndarray, bits: int = 4) -> np.ndarray:
        if np.random.rand() > 0.3:
            return image
        bits = max(1, min(bits, 8))
        levels = 2 ** bits
        return np.floor(image * levels) / levels

    def _random_color_jitter(self, image: np.ndarray, factor: float = 0.2) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = np.clip(hsv[..., 0] + np.random.uniform(-factor, factor), 0, 1)
        hsv[..., 1] = np.clip(hsv[..., 1] * np.random.uniform(0.8, 1.2), 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * np.random.uniform(0.8, 1.2), 0, 1)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _random_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        import random
        if np.random.rand() > 0.5:
            return image
        k = random.choice([3, 5, 7])
        sigma = random.uniform(0.1, 1.5)
        for c in range(image.shape[-1]):
            image[..., c] = cv2.GaussianBlur(image[..., c], (k, k), sigma)
        return image

    def _random_motion_blur(self, image: np.ndarray) -> np.ndarray:
        import random
        if np.random.rand() > 0.5:
            return image
        k = random.randint(3, self.config.motion_blur_kernel)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        angle = random.uniform(-45, 45)
        center = (k // 2, k // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (k, k))
        kernel = kernel / max(np.sum(kernel), 1e-6)
        for c in range(image.shape[-1]):
            image[..., c] = cv2.filter2D(image[..., c], -1, kernel)
        return image

    def _random_clahe_variation(self, image: np.ndarray) -> np.ndarray:
        import random
        if np.random.rand() > 0.5:
            return image
        clip_limit = random.uniform(1.0, 3.0)
        grid = random.choice([(4, 4), (8, 8)])
        result = image.copy()
        for c in range(image.shape[-1]):
            channel = (image[..., c] * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid)
            result[..., c] = clahe.apply(channel) / 255.0
        return result

    def _apply_elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        h, w = image.shape[:2]
        alpha = self.config.elastic_alpha
        sigma = self.config.elastic_sigma
        dx = np.random.randn(h, w).astype(np.float32) * alpha
        dy = np.random.randn(h, w).astype(np.float32) * alpha
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
        if image.ndim == 3:
            return np.stack([cv2.remap(image[..., c], map_x, map_y, cv2.INTER_LINEAR) for c in range(image.shape[-1])], axis=-1)
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        noise = np.random.normal(0, self.config.gaussian_noise_std, image.shape).astype(np.float32)
        return np.clip(image + noise, 0, 1)

    def _apply_intensity_shift(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.5:
            return image
        shift = np.random.uniform(-self.config.intensity_shift_range, self.config.intensity_shift_range)
        return np.clip(image + shift, 0, 1)

    def _apply_bias_field_simulation(self, image: np.ndarray) -> np.ndarray:
        if np.random.rand() > 0.3:
            return image
        h, w = image.shape[:2]
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xv, yv = np.meshgrid(x, y)
        coeffs = np.random.uniform(-0.3, 0.3, size=(3, 3))
        bias = np.zeros((h, w), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                bias += coeffs[i, j] * (xv ** i) * (yv ** j)
        bias = 1.0 + bias
        bias = (bias - bias.min()) / max(bias.max() - bias.min(), 1e-6) * 0.4 + 0.8
        if image.ndim == 3:
            return image * bias[..., np.newaxis]
        return image * bias

    def mixup(self, image1: np.ndarray, image2: np.ndarray, label1: int, label2: int) -> tuple[np.ndarray, float]:
        lam = np.random.beta(self.config.mixup_alpha, self.config.mixup_alpha)
        mixed = lam * image1.astype(np.float32) + (1 - lam) * image2.astype(np.float32)
        return mixed.astype(np.uint8), lam

    def cutmix(self, image1: np.ndarray, image2: np.ndarray, label1: int, label2: int) -> tuple[np.ndarray, float]:
        lam = np.random.beta(self.config.cutmix_alpha, self.config.cutmix_alpha)
        h, w = image1.shape[:2]
        cut_ratio = np.sqrt(1 - lam)
        cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
        cx, cy = np.random.randint(cut_w // 2, w - cut_w // 2), np.random.randint(cut_h // 2, h - cut_h // 2)
        x1, y1 = max(0, cx - cut_w // 2), max(0, cy - cut_h // 2)
        x2, y2 = min(w, cx + cut_w // 2), min(h, cy + cut_h // 2)
        result = image1.copy()
        result[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (h * w))
        return result, lam

    def cutout(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        for _ in range(self.config.cutout_n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.config.cutout_length // 2)
            y2 = min(h, y + self.config.cutout_length // 2)
            x1 = max(0, x - self.config.cutout_length // 2)
            x2 = min(w, x + self.config.cutout_length // 2)
            image[y1:y2, x1:x2] = 0
        return image
