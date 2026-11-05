from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

ICONS_DIR = Path(__file__).resolve().parents[2] / "assets" / "icons"
ICON_CACHE: dict[tuple[str, str, int], QIcon] = {}


def themed_icon(name: str, color: str, size: int) -> QIcon:
    cache_key = (name, color, size)
    if cache_key in ICON_CACHE:
        return ICON_CACHE[cache_key]

    path = ICONS_DIR / name
    if not path.exists():
        return QIcon()

    svg_text = path.read_text(encoding="utf-8")
    svg_text = svg_text.replace("currentColor", color).replace("#AAB0BC", color)
    renderer = QSvgRenderer(QByteArray(svg_text.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    result = QIcon(pixmap)
    ICON_CACHE[cache_key] = result
    return result


def themed_pixmap(name: str, color: str, size: int) -> QPixmap:
    return themed_icon(name, color, size).pixmap(size, size)
