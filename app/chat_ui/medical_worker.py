from __future__ import annotations

from medical.chat_service import MedicalChatResponse, MedicalChatService
from utils.logger import get_logger

logger = get_logger(__name__)


def build_patient_code(conversation_id: int | None, timestamp: int) -> str:
    if conversation_id is not None:
        return f"CHAT-{conversation_id:05d}"
    return f"CHAT-TEMP-{timestamp}"


def create_medical_worker_base(QThread, Signal):
    class MedicalAnalysisWorker(QThread):
        result_ready = Signal(object)
        error = Signal(str)
        progress = Signal(str, float)
        finished = Signal()

        def __init__(
            self,
            service: MedicalChatService,
            *,
            image_path: str,
            patient_code: str,
            user_prompt: str,
            target_key: str | None = None,
            modality: str | None = None,
        ):
            super().__init__()
            self.service = service
            self.image_path = image_path
            self.patient_code = patient_code
            self.user_prompt = user_prompt
            self.target_key = target_key
            self.modality = modality

        def run(self) -> None:
            try:
                def _on_progress(stage: str, pct: float) -> None:
                    self.progress.emit(stage, pct)
                response: MedicalChatResponse = self.service.analyze_attachment(
                    image_path=self.image_path,
                    patient_code=self.patient_code,
                    user_prompt=self.user_prompt,
                    target_key=self.target_key,
                    modality=self.modality,
                    progress_callback=_on_progress,
                )
                self.result_ready.emit(response)
            except Exception as exc:
                logger.exception("Medical image analysis failed for %s", self.image_path)
                self.error.emit(str(exc))
            finally:
                self.finished.emit()

    return MedicalAnalysisWorker
