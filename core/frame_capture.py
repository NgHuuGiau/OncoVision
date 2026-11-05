from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import strftime

import cv2
import numpy as np


@dataclass(frozen=True)
class CaptureResult:
    path: Path
    success: bool


class FrameCapture:
    def __init__(self, output_dir: str | Path = "output/captures") -> None:
        self.output_dir = Path(output_dir)

    def save_frame(self, frame: np.ndarray) -> CaptureResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"capture_{strftime('%Y%m%d_%H%M%S')}.jpg"
        target_path = self._unique_path(self.output_dir / filename)
        success = bool(cv2.imwrite(str(target_path), frame))
        return CaptureResult(path=target_path, success=success)

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 1
        while True:
            candidate = path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1
