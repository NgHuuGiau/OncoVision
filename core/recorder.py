from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import strftime

import cv2
import numpy as np


@dataclass(frozen=True)
class RecorderStatus:
    is_recording: bool
    path: Path | None = None


class VideoRecorder:
    def __init__(
        self,
        output_dir: str | Path = "output/recordings",
        *,
        codec: str = "mp4v",
        fps: float = 20.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.codec = codec
        self.fps = float(fps)
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None
        self.frame_size: tuple[int, int] | None = None

    @property
    def is_recording(self) -> bool:
        return self.writer is not None

    def start(self, frame_size: tuple[int, int]) -> RecorderStatus:
        self.stop()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_size = frame_size
        self.output_path = self._build_output_path()
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*self.codec),
            self.fps,
            frame_size,
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"Không tạo được file video: {self.output_path}")
        return RecorderStatus(is_recording=True, path=self.output_path)

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None or self.frame_size is None:
            return
        height, width = frame.shape[:2]
        normalized = frame
        if (width, height) != self.frame_size:
            normalized = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_LINEAR)
        self.writer.write(normalized)

    def stop(self) -> RecorderStatus:
        path = self.output_path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.output_path = None
        self.frame_size = None
        return RecorderStatus(is_recording=False, path=path)

    def _build_output_path(self) -> Path:
        base = self.output_dir / f"recording_{strftime('%Y%m%d_%H%M%S')}.mp4"
        if not base.exists():
            return base
        stem = base.stem
        suffix = base.suffix
        counter = 1
        while True:
            candidate = base.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1
