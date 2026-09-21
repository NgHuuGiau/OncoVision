from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.sqlite_utils import DEFAULT_SQLITE_TIMEOUT_SECONDS, create_sqlite_connection

logger = get_logger(__name__)


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
    assigned_to: str | None
    review_status: str
    reviewed_by: str | None
    reviewed_at: str | None
    public_code: str | None


class MedicalCaseDatabase:
    CONNECT_TIMEOUT_SECONDS = DEFAULT_SQLITE_TIMEOUT_SECONDS

    def __init__(self, db_path: str | Path = "output/onco.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _create_connection(self) -> sqlite3.Connection:
        return create_sqlite_connection(
            self.db_path,
            timeout_seconds=self.CONNECT_TIMEOUT_SECONDS,
        )

    @contextmanager
    def _connect(self):
        conn = self._create_connection()
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
            columns = {row[1] for row in conn.execute("PRAGMA table_info(medical_cases)")}
            for name, definition in (
                ("assigned_to", "TEXT"),
                ("review_status", "TEXT NOT NULL DEFAULT 'pending'"),
                ("reviewed_by", "TEXT"),
                ("reviewed_at", "TEXT"),
                ("public_code", "TEXT"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE medical_cases ADD COLUMN {name} {definition}")
            self._ensure_indexes(conn)

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_medical_cases_patient_code
            ON medical_cases (patient_code)
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_medical_cases_assigned_review "
            "ON medical_cases (assigned_to, review_status, id DESC)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_medical_cases_public_code "
            "ON medical_cases (public_code) WHERE public_code IS NOT NULL"
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_medical_cases_created_at
            ON medical_cases (created_at DESC, id DESC)
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

    def list_cases(self, *, assigned_to: str | None = None, limit: int = 50, offset: int = 0) -> list[MedicalCaseRecord]:
        limit = max(1, min(int(limit), 200))
        offset = max(0, int(offset))
        with self._connect() as conn:
            query = """
                SELECT id, patient_code, image_path, processed_image_path, report_json_path, report_md_path,
                       suspected_malignant, risk_level, recommendation, metadata_json, created_at,
                       assigned_to, review_status, reviewed_by, reviewed_at, public_code
                FROM medical_cases
            """
            rows = (
                conn.execute(query + " WHERE assigned_to = ? ORDER BY id DESC LIMIT ? OFFSET ?", (assigned_to, limit, offset)).fetchall()
                if assigned_to is not None
                else conn.execute(query + " ORDER BY id DESC LIMIT ? OFFSET ? ", (limit, offset)).fetchall()
            )
        return [self._row_to_record(row) for row in rows]

    def get_case(self, case_id: int) -> MedicalCaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, patient_code, image_path, processed_image_path, report_json_path, report_md_path,
                       suspected_malignant, risk_level, recommendation, metadata_json, created_at,
                       assigned_to, review_status, reviewed_by, reviewed_at, public_code
                FROM medical_cases
                WHERE id = ?
                """,
                (case_id,),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_case_by_public_code(self, public_code: str) -> MedicalCaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, patient_code, image_path, processed_image_path, report_json_path, report_md_path, "
                "suspected_malignant, risk_level, recommendation, metadata_json, created_at, assigned_to, "
                "review_status, reviewed_by, reviewed_at, public_code FROM medical_cases "
                "WHERE public_code = ? AND review_status = 'approved'",
                (public_code.upper(),),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def assign_case(self, case_id: int, username: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE medical_cases SET assigned_to = ?, review_status = 'pending', reviewed_by = NULL, "
                "reviewed_at = NULL, public_code = NULL WHERE id = ?",
                (username, case_id),
            )
        return cursor.rowcount == 1

    def approve_case(
        self,
        case_id: int,
        *,
        reviewer: str,
        risk_level: str,
        suspected_malignant: bool,
        recommendation: str,
        public_code: str,
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE medical_cases SET risk_level = ?, suspected_malignant = ?, recommendation = ?, "
                "review_status = 'approved', reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, public_code = ? "
                "WHERE id = ?",
                (risk_level, int(suspected_malignant), recommendation, reviewer, public_code.upper(), case_id),
            )
        return cursor.rowcount == 1

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
                    logger.warning("Failed to delete medical artifact: %s", path)
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
            assigned_to=row[11],
            review_status=row[12],
            reviewed_by=row[13],
            reviewed_at=row[14],
            public_code=row[15],
        )
