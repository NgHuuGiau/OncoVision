"""Tests for newly added modules: audit trail, model caching, environment config, JSON logger."""
from __future__ import annotations

import json
from pathlib import Path


class TestAuditTrail:
    """Tests for medical.compliance.AuditTrail."""

    def test_log_inference(self, tmp_path: Path) -> None:
        from medical.compliance import AuditTrail

        audit = AuditTrail(log_file=tmp_path / "audit.json.log")
        audit.log_inference(
            target_key="brain",
            model_path="models/test.pt",
            classification="glioma",
            confidence=0.85,
            detections_count=2,
            image_path="test.jpg",
        )

        log_file = tmp_path / "audit.json.log"
        assert log_file.exists()
        line = log_file.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert entry["action"] == "inference"
        assert entry["target_key"] == "brain"
        assert "glioma" in entry["result_summary"]
        assert "confidence=0.850" in entry["result_summary"]

    def test_log_login(self, tmp_path: Path) -> None:
        from medical.compliance import AuditTrail

        audit = AuditTrail(log_file=tmp_path / "audit.json.log")
        audit.log_login(user="admin", success=True, ip_address="127.0.0.1")

        log_file = tmp_path / "audit.json.log"
        line = log_file.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert entry["action"] == "login"
        assert entry["user"] == "admin"
        assert entry["result_summary"] == "success"

    def test_log_export(self, tmp_path: Path) -> None:
        from medical.compliance import AuditTrail

        audit = AuditTrail(log_file=tmp_path / "audit.json.log")
        audit.log_export(user="doctor", case_id=42, format="pdf")

        log_file = tmp_path / "audit.json.log"
        line = log_file.read_text(encoding="utf-8").strip()
        entry = json.loads(line)
        assert entry["action"] == "export"
        assert entry["metadata"]["case_id"] == 42
        assert entry["metadata"]["format"] == "pdf"

    def test_get_audit_trail_singleton(self) -> None:
        from medical.compliance import get_audit_trail

        trail1 = get_audit_trail()
        trail2 = get_audit_trail()
        assert trail1 is trail2


class TestModelCaching:
    """Tests for medical.cnn_classifier model caching."""

    def test_clear_model_cache(self) -> None:
        from medical.cnn_classifier import clear_model_cache, get_model_cache_info

        clear_model_cache()
        info = get_model_cache_info()
        assert info == {}

    def test_get_model_cache_info(self) -> None:
        from medical.cnn_classifier import _model_cache, get_model_cache_info

        # Simulate cached model
        _model_cache["/test/path.pt"] = (None, "cpu")  # type: ignore[arg-type]
        info = get_model_cache_info()
        assert "/test/path.pt" in info
        assert info["/test/path.pt"] == "cpu"
        _model_cache.clear()


class TestEnvironmentConfig:
    """Tests for utils.file_utils environment config loading."""

    def test_load_environment_config_default(self, tmp_path: Path) -> None:
        from utils.file_utils import _deep_merge

        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 3, "d": 4}}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": {"c": 3, "d": 4}}

    def test_deep_merge_nested(self) -> None:
        from utils.file_utils import _deep_merge

        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"y": 10}}
        merged = _deep_merge(base, override)
        assert merged == {"x": {"y": 10, "z": 2}}

    def test_deep_merge_new_keys(self) -> None:
        from utils.file_utils import _deep_merge

        base = {"a": 1}
        override = {"b": 2}
        merged = _deep_merge(base, override)
        assert merged == {"a": 1, "b": 2}


class TestJsonFormatter:
    """Tests for utils.logger JsonFormatter."""

    def test_json_formatter_output(self) -> None:
        import logging

        from utils.logger import JsonFormatter

        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        entry = json.loads(output)
        assert entry["level"] == "INFO"
        assert entry["message"] == "Test message"
        assert entry["logger"] == "test"
        assert "timestamp" in entry

    def test_json_formatter_with_exception(self) -> None:
        import logging

        from utils.logger import JsonFormatter

        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        entry = json.loads(output)
        assert "exception" in entry
        assert entry["exception"]["type"] == "ValueError"
