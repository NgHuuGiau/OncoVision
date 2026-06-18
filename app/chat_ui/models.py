from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ChatMessage:
    sender: Literal["user", "ai"]
    text: str
    attachment_path: str | None = None
    attachment_kind: Literal["image", "text", "camera"] | None = None
    metadata_json: str | None = None
    id: int | None = None


@dataclass
class Conversation:
    title: str
    subtitle: str
    messages: list[ChatMessage] = field(default_factory=list)
    id: int | None = None
