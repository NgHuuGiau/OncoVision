from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtGui import QImageReader, QPixmap
from PIL.ImageQt import ImageQt

from medical.dataset import is_medical_volume_source, load_medical_source_image, load_medical_volume_slices


def pil_image_to_pixmap(image) -> QPixmap:
    return QPixmap.fromImage(ImageQt(image))


def load_medical_volume_pixmaps(path: str | Path) -> list[QPixmap]:
    if not is_medical_volume_source(path):
        return []
    pixmaps = []
    for slice_image in load_medical_volume_slices(path):
        pixmaps.append(pil_image_to_pixmap(slice_image))
    return pixmaps


def load_preview_pixmap(path: str | Path, *, max_size: tuple[int, int]) -> QPixmap:
    source = Path(path)
    if source.is_file():
        reader = QImageReader(str(source))
        reader.setAutoTransform(True)
        reader.setScaledSize(QSize(*max_size))
        image = reader.read()
        if not image.isNull():
            return QPixmap.fromImage(image)

    try:
        preview = load_medical_source_image(source)
    except Exception:
        return QPixmap()
    preview.thumbnail(max_size)
    return pil_image_to_pixmap(preview)
