from __future__ import annotations

from dataclasses import dataclass

from app.chat_ui.models import ChatMessage
from medical.chat_service import MedicalChatService


@dataclass(frozen=True)
class MedicalUIState:
    active: bool
    placeholder: str
    status_text: str


class MedicalChatController:
    def __init__(self, service: MedicalChatService | None = None) -> None:
        self.service = service or MedicalChatService()
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def ensure_ready(self) -> str:
        return str(self.service.check_ready())

    def begin_analysis(self, analyzing_text: str) -> tuple[MedicalUIState, ChatMessage]:
        self._active = True
        state = MedicalUIState(active=True, placeholder=analyzing_text, status_text=analyzing_text)
        return state, ChatMessage(sender="ai", text=analyzing_text)

    def finish_analysis(self, input_placeholder: str) -> MedicalUIState:
        self._active = False
        return MedicalUIState(active=False, placeholder=input_placeholder, status_text="")
