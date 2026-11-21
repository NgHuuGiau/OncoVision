"""001_initial — Tạo các bảng cơ bản từ schema hiện có.

Revision ID: 001_initial
Create Date: 2026-11-13
"""
import sqlalchemy as sa

from alembic import op

revision = "001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # medical_cases table
    op.create_table(
        "medical_cases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("patient_code", sa.Text(), nullable=False),
        sa.Column("image_path", sa.Text(), nullable=False),
        sa.Column("processed_image_path", sa.Text(), nullable=False),
        sa.Column("report_json_path", sa.Text(), nullable=False),
        sa.Column("report_md_path", sa.Text(), nullable=False),
        sa.Column("suspected_malignant", sa.Integer(), nullable=False),
        sa.Column("risk_level", sa.Text(), nullable=False),
        sa.Column("recommendation", sa.Text(), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(), server_default=sa.func.current_timestamp()),
        sa.Column("assigned_to", sa.Text(), nullable=True),
        sa.Column("review_status", sa.Text(), server_default="pending", nullable=False),
        sa.Column("reviewed_by", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.Text(), nullable=True),
        sa.Column("public_code", sa.Text(), nullable=True),
        sa.Column("patient_username", sa.Text(), nullable=True),
    )
    op.create_index("idx_medical_cases_patient_code", "medical_cases", ["patient_code"])
    op.create_index(
        "idx_medical_cases_assigned_review", "medical_cases",
        ["assigned_to", "review_status", "id"], unique=False
    )
    op.create_index(
        "idx_medical_cases_public_code", "medical_cases", ["public_code"],
        unique=True, postgresql_where=sa.text("public_code IS NOT NULL")
    )
    op.create_index(
        "idx_medical_cases_patient", "medical_cases",
        ["patient_username", "review_status", "id"], unique=False
    )


def downgrade() -> None:
    op.drop_index("idx_medical_cases_patient")
    op.drop_index("idx_medical_cases_public_code")
    op.drop_index("idx_medical_cases_assigned_review")
    op.drop_index("idx_medical_cases_patient_code")
    op.drop_table("medical_cases")
