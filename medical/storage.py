from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MedicalCaseRecord:
    case_id: int
    patient_code: str
    image_path: str
    processed_image_path: str
    report_json_path: str
    report_md_path: str
    suspected_malignant: bool
    risk_level: str
    recommendation: str
    metadata: dict[str, Any]
    created_at: str


class MedicalCaseDatabase:
    def __init__(self, db_path: str | Path = "output/medical/medical_cases.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS medical_cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_code TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    processed_image_path TEXT NOT NULL,
                    report_json_path TEXT NOT NULL,
                    report_md_path TEXT NOT NULL,
                    suspected_malignant INTEGER NOT NULL,
                    risk_level TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def save_case(self, *, patient_code: str, image_path: str, processed_image_path: str, report_json_path: str, report_md_path: str, suspected_malignant: bool, risk_level: str, recommendation: str, metadata: dict[str, Any]) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO medical_cases (
                    patient_code, image_path, processed_image_path, report_json_path, report_md_path,
                    suspected_malignant, risk_level, recommendation, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_code,
                    image_path,
                    processed_image_path,
                    report_json_path,
                    report_md_path,
                    1 if suspected_malignant else 0,
                    risk_level,
                    recommendation,
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def list_cases(self) -> list[MedicalCaseRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, patient_code, image_path, processed_image_path, report_json_path, report_md_path,
                       suspected_malignant, risk_level, recommendation, metadata_json, created_at
                FROM medical_cases
                ORDER BY id DESC
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_case(self, case_id: int) -> MedicalCaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, patient_code, image_path, processed_image_path, report_json_path, report_md_path,
                       suspected_malignant, risk_level, recommendation, metadata_json, created_at
                FROM medical_cases
                WHERE id = ?
                """,
                (case_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def delete_case(self, case_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM medical_cases WHERE id = ?", (case_id,))
            return cursor.rowcount > 0

    def delete_case_with_files(self, case_id: int) -> tuple[bool, list[str]]:
        item = self.get_case(case_id)
        if item is None:
            return False, []
        deleted_paths: list[str] = []
        for path_str in (
            item.image_path,
            item.processed_image_path,
            item.report_json_path,
            item.report_md_path,
        ):
            path = Path(path_str)
            if path.exists() and path.is_file():
                try:
                    os.remove(path)
                    deleted_paths.append(str(path))
                except OSError:
                    pass
        deleted = self.delete_case(case_id)
        return deleted, deleted_paths

    def _row_to_record(self, row) -> MedicalCaseRecord:
        return MedicalCaseRecord(
            case_id=row[0],
            patient_code=row[1],
            image_path=row[2],
            processed_image_path=row[3],
            report_json_path=row[4],
            report_md_path=row[5],
            suspected_malignant=bool(row[6]),
            risk_level=row[7],
            recommendation=row[8],
            metadata=json.loads(row[9]),
            created_at=row[10],
        )
