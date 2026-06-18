from __future__ import annotations

import unittest

from app.chat_ui.medical_controller import MedicalChatController


class _FakeService:
    def __init__(self) -> None:
        self.checked = False

    def check_ready(self) -> str:
        self.checked = True
        return "models/trained/skin.pt"


class MedicalChatControllerTests(unittest.TestCase):
    def test_ensure_ready_delegates_to_service(self) -> None:
        service = _FakeService()
        controller = MedicalChatController(service)

        result = controller.ensure_ready()

        self.assertTrue(service.checked)
        self.assertEqual(result, "models/trained/skin.pt")

    def test_begin_and_finish_analysis_toggle_state(self) -> None:
        controller = MedicalChatController(_FakeService())

        state, message = controller.begin_analysis("Dang phan tich")
        self.assertTrue(controller.active)
        self.assertTrue(state.active)
        self.assertEqual(message.text, "Dang phan tich")

        final_state = controller.finish_analysis("Nhap tin nhan")
        self.assertFalse(controller.active)
        self.assertFalse(final_state.active)
        self.assertEqual(final_state.placeholder, "Nhap tin nhan")


if __name__ == "__main__":
    unittest.main()
