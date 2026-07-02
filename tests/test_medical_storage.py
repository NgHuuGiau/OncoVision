from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.storage import MedicalCaseDatabase


class MedicalStorageTests(unittest.TestCase):
    def test_save_case_and_list_cases_roundtrip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db = MedicalCaseDatabase(Path(temp_dir) / "medical.db")
            case_id = db.save_case(
                patient_code="BN001",
                image_path="a.jpg",
                processed_image_path="b.jpg",
                report_json_path="c.json",
                report_md_path="d.md",
                suspected_malignant=True,
                risk_level="high",
                recommendation="Follow-up",
                metadata={"score": 0.92},
            )

            cases = db.list_cases()

            self.assertEqual(case_id, cases[0].case_id)
            self.assertEqual(cases[0].patient_code, "BN001")
            self.assertTrue(cases[0].suspected_malignant)
            self.assertEqual(cases[0].metadata["score"], 0.92)
            self.assertTrue(cases[0].created_at)

    def test_get_case_and_delete_case(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db = MedicalCaseDatabase(Path(temp_dir) / "medical.db")
            case_id = db.save_case(
                patient_code="BN002",
                image_path="img.jpg",
                processed_image_path="processed.jpg",
                report_json_path="report.json",
                report_md_path="report.md",
                suspected_malignant=False,
                risk_level="low",
                recommendation="Theo dĂµi",
                metadata={"score": 0.2},
            )

            item = db.get_case(case_id)

            self.assertIsNotNone(item)
            self.assertEqual(item.patient_code, "BN002")
            self.assertTrue(db.delete_case(case_id))
            self.assertIsNone(db.get_case(case_id))

    def test_delete_case_with_files_removes_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = MedicalCaseDatabase(root / "medical.db")
            image = root / "source.jpg"
            processed = root / "processed.jpg"
            report_json = root / "report.json"
            report_md = root / "report.md"
            for path in (image, processed, report_json, report_md):
                path.write_text("x", encoding="utf-8")
            case_id = db.save_case(
                patient_code="BN003",
                image_path=str(image),
                processed_image_path=str(processed),
                report_json_path=str(report_json),
                report_md_path=str(report_md),
                suspected_malignant=False,
                risk_level="low",
                recommendation="Theo dĂµi",
                metadata={},
            )

            deleted, deleted_paths = db.delete_case_with_files(case_id)

            self.assertTrue(deleted)
            self.assertEqual(len(deleted_paths), 4)
            self.assertFalse(image.exists())
            self.assertIsNone(db.get_case(case_id))

    def test_init_creates_indexes_for_lookup_and_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "medical.db"
            db = MedicalCaseDatabase(db_path)

            with db._connect() as conn:
                indexes = {
                    row[1]
                    for row in conn.execute("PRAGMA index_list(medical_cases)").fetchall()
                }

            self.assertIn("idx_medical_cases_patient_code", indexes)
            self.assertIn("idx_medical_cases_created_at", indexes)

    def test_connection_enables_wal_mode_and_busy_timeout(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db = MedicalCaseDatabase(Path(temp_dir) / "medical.db")

            with db._connect() as conn:
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

            self.assertEqual(str(journal_mode).lower(), "wal")
            self.assertEqual(busy_timeout, 5000)



