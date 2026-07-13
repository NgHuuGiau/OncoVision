from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QMessageBox

from app.chat_ui.models import ChatMessage
from app.chat_ui.content import translate
from app.chat_ui.widgets import ComposerPreviewThumb
from medical.dataset import infer_medical_upload_context, is_supported_medical_upload_path


STANDARD_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VOLUME_EXTENSIONS = (".dcm", ".nii", ".nii.gz", ".mha", ".mhd")
VOLUME_MODALITIES = {
    "CT",
    "CT ngực",
    "CT ngực-bụng-chậu",
    "MRI",
    "MRI vú",
    "MRI trực tràng",
    "MRI tuyến tiền liệt",
    "PET",
    "PET/CT",
}


def is_volume_modality(modality: str) -> bool:
    return modality in VOLUME_MODALITIES


def modality_upload_extensions(modality: str) -> tuple[str, ...]:
    if is_volume_modality(modality):
        return STANDARD_IMAGE_EXTENSIONS + VOLUME_EXTENSIONS
    return STANDARD_IMAGE_EXTENSIONS + (".dcm",)


def build_modality_file_filter(language: str, modality: str) -> str:
    extensions = " ".join(f"*{ext}" for ext in modality_upload_extensions(modality))
    kind = translate(language, "medical_picker_volume" if is_volume_modality(modality) else "medical_picker_image")
    return f"{kind} - {modality} ({extensions})"


def build_modality_dialog_title(window, *, picker_kind: str, modality: str) -> str:
    return f"{window.tr(window.language, picker_kind)} - {window.current_medical_target.label} / {modality}"


def refresh_image_previews(window) -> None:
    while window.image_preview_layout.count() > 1:
        item = window.image_preview_layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()

    has_attachments = bool(window.pending_image_attachments)
    window.image_preview_area.setVisible(has_attachments)
    window.composer.setMinimumHeight(152 if has_attachments else 82)

    if not has_attachments:
        return

    for index, (path, attachment_kind) in enumerate(window.pending_image_attachments):
        thumb_widget = ComposerPreviewThumb(
            path=path,
            attachment_kind=attachment_kind,
            language=window.language,
            remove_callback=lambda _checked=False, idx=index: remove_pending_image_attachment(window, idx),
        )
        window.image_preview_layout.insertWidget(window.image_preview_layout.count() - 1, thumb_widget)


def remove_pending_image_attachment(window, index: int) -> None:
    if not (0 <= index < len(window.pending_image_attachments)):
        return
    window.pending_image_attachments.pop(index)
    refresh_image_previews(window)
    window.message_input.setFocus()


def clear_pending_image_previews(window) -> None:
    window.pending_image_attachments.clear()
    refresh_image_previews(window)


def add_pending_image_attachment(window, path: str, attachment_kind: str, *, guess_context: bool = True) -> None:
    window.pending_image_attachments.append((path, attachment_kind))
    if guess_context and attachment_kind == "image":
        apply_medical_context_guess(window, path)
    refresh_image_previews(window)


def apply_medical_context_guess(window, path: str) -> None:
    target_key, modality = infer_medical_upload_context(path)
    if not target_key and not modality:
        return

    target = None
    if target_key:
        target = next((candidate for candidate in window.medical_targets if candidate.key == target_key), None)
    if target is None and modality:
        target = next((candidate for candidate in window.medical_targets if modality in candidate.modalities), None)

    if target is not None and getattr(window, "current_medical_target", None) != target:
        window.set_medical_target(target.label)

    if modality and hasattr(window, "medical_modality_combo"):
        if modality in window.current_medical_target.modalities:
            window.set_medical_modality(modality)


def pending_attachment_prompt(window) -> str:
    if not window.pending_image_attachments:
        return ""
    if len(window.pending_image_attachments) == 1:
        _, attachment_kind = window.pending_image_attachments[0]
        return window.tr(window.language, "attach_camera_label") if attachment_kind == "camera" else window.tr(window.language, "attach_image_label")
    return f"{len(window.pending_image_attachments)} attachments"


def add_user_message(window, text: str, attachments: list[tuple[str, str]]) -> None:
    for index, (path, attachment_kind) in enumerate(attachments):
        attachment_text = text if index == 0 and text else ""
        window.add_message(
            ChatMessage(
                sender="user",
                text=attachment_text,
                attachment_path=path,
                attachment_kind=attachment_kind,
            )
        )

    if text and not attachments:
        window.add_message(ChatMessage(sender="user", text=text))

def submit_user_message(window, text: str = "", attachments: list[tuple[str, str]] | None = None) -> None:
    attachments = attachments or []
    if window.medical_controller.active:
        QMessageBox.information(window, window.tr(window.language, "info_title"), window.tr(window.language, "medical_pending"))
        return
    if not text and not attachments:
        QMessageBox.information(window, window.tr(window.language, "info_title"), window.tr(window.language, "empty_send"))
        return

    add_user_message(window, text, attachments)
    prompt = text or (pending_attachment_prompt(window) if attachments else "")
    first_attachment = attachments[0] if attachments else None
    clear_pending_image_previews(window)
    window.generate_system_response(
        prompt,
        first_attachment[0] if first_attachment else None,
        first_attachment[1] if first_attachment else None,
    )


def send_message(window) -> None:
    text = window.message_input.toPlainText().strip()
    submit_user_message(window, text, list(window.pending_image_attachments))
    window.message_input.clear()


def handle_dropped_image(window, path: str) -> None:
    if is_supported_medical_upload_path(path):
        add_pending_image_attachment(window, path, "image")


def pick_image(window) -> None:
    modality = getattr(window, "current_medical_modality", window.current_medical_target.modalities[0])
    image_filter = build_modality_file_filter(window.language, modality)
    path, _ = QFileDialog.getOpenFileName(
        window,
        build_modality_dialog_title(window, picker_kind="choose_image", modality=modality),
        "",
        image_filter,
    )
    if not path:
        return
    add_pending_image_attachment(window, path, "image")


def pick_dicom_folder(window) -> None:
    modality = getattr(window, "current_medical_modality", window.current_medical_target.modalities[0])
    folder = QFileDialog.getExistingDirectory(
        window,
        build_modality_dialog_title(window, picker_kind="choose_dicom_folder", modality=modality),
        "",
    )
    if not folder:
        return
    add_pending_image_attachment(window, folder, "image")


def handle_camera_capture(window, path: str) -> None:
    add_pending_image_attachment(window, path, "camera", guess_context=False)
