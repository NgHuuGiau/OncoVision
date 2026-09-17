from __future__ import annotations

import sqlite3
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

    def test_assign_approve_and_revoke_public_case_code(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db = MedicalCaseDatabase(Path(temp_dir) / "medical.db")
            case_id = db.save_case(
                patient_code="BN004", image_path="a.jpg", processed_image_path="b.jpg",
                report_json_path="c.json", report_md_path="d.md", suspected_malignant=False,
                risk_level="uncertain", recommendation="Review", metadata={},
            )
            self.assertTrue(db.assign_case(case_id, "clinician01"))
            self.assertEqual(db.list_cases(assigned_to="clinician01")[0].review_status, "pending")
            self.assertTrue(db.approve_case(
                case_id, reviewer="clinician01", risk_level="low", suspected_malignant=False,
                recommendation="Đã rà soát", public_code="ABCD234567",
            ))
            self.assertEqual(db.get_case_by_public_code("abcd234567").recommendation, "Đã rà soát")
            self.assertTrue(db.assign_case(case_id, "clinician02"))
            self.assertIsNone(db.get_case_by_public_code("ABCD234567"))
            self.assertEqual(db.get_case(case_id).review_status, "pending")

    def test_legacy_case_table_gets_review_columns_without_losing_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "legacy.db"
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    "CREATE TABLE medical_cases (id INTEGER PRIMARY KEY, patient_code TEXT NOT NULL, "
                    "image_path TEXT NOT NULL, processed_image_path TEXT NOT NULL, report_json_path TEXT NOT NULL, "
                    "report_md_path TEXT NOT NULL, suspected_malignant INTEGER NOT NULL, risk_level TEXT NOT NULL, "
                    "recommendation TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
                )
                conn.execute(
                    "INSERT INTO medical_cases (patient_code,image_path,processed_image_path,report_json_path,"
                    "report_md_path,suspected_malignant,risk_level,recommendation,metadata_json) "
                    "VALUES ('BN005','a','b','c','d',0,'low','Giữ nguyên','{}')"
                )
                conn.commit()
            finally:
                conn.close()
            case = MedicalCaseDatabase(db_path).get_case(1)
            self.assertEqual(case.patient_code, "BN005")
            self.assertIsNone(case.assigned_to)
            self.assertEqual(case.review_status, "pending")
