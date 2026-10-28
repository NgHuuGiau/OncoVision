from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.chat_ui.medical_worker import create_medical_worker_base


class _Signal:
    def __init__(self, *_args):
        self.values = []

    def emit(self, *values):
        self.values.append(values)


class _Thread:
    pass


class MedicalWorkerTests(unittest.TestCase):
    def test_worker_passes_target_and_modality_to_service(self) -> None:
        worker_class = create_medical_worker_base(_Thread, _Signal)
        service = Mock()
        service.analyze_attachment.return_value = object()
        worker = worker_class(
            service,
            image_path="scan.dcm",
            patient_code="P1",
            user_prompt="analyze",
            target_key="lung",
            modality="CT ngực",
        )

        worker.run()

        kwargs = service.analyze_attachment.call_args.kwargs
        self.assertEqual(kwargs["target_key"], "lung")
        self.assertEqual(kwargs["modality"], "CT ngực")
        self.assertEqual(worker.result_ready.values, [(service.analyze_attachment.return_value,)])


if __name__ == "__main__":
    unittest.main()
