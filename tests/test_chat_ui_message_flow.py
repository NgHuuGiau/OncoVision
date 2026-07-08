from __future__ import annotations

import unittest

from app.chat_ui.message_flow import build_modality_file_filter, is_volume_modality, modality_upload_extensions


class ChatUiMessageFlowTests(unittest.TestCase):
    def test_modality_upload_extensions_prefers_volume_formats_for_ct_mri(self) -> None:
        extensions = modality_upload_extensions("MRI")

        self.assertIn(".nii", extensions)
        self.assertIn(".nii.gz", extensions)
        self.assertIn(".dcm", extensions)
        self.assertTrue(is_volume_modality("MRI"))

    def test_modality_upload_extensions_prefers_image_formats_for_ultrasound(self) -> None:
        extensions = modality_upload_extensions("Siêu âm")

        self.assertIn(".jpg", extensions)
        self.assertIn(".dcm", extensions)
        self.assertNotIn(".nii", extensions)
        self.assertFalse(is_volume_modality("Siêu âm"))

    def test_build_modality_file_filter_mentions_selected_modality(self) -> None:
        file_filter = build_modality_file_filter("vi", "CT")

        self.assertIn("Volume y khoa", file_filter)
        self.assertIn("*.dcm", file_filter)
        self.assertIn("*.nii.gz", file_filter)

    def test_build_modality_file_filter_uses_translated_kind_label(self) -> None:
        file_filter = build_modality_file_filter("en", "CT")

        self.assertIn("Medical volume", file_filter)


if __name__ == "__main__":
    unittest.main()
