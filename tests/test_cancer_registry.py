import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from medical.cancer_model_registry import find_cancer_model, list_cancer_models


class CancerRegistryTests(unittest.TestCase):
    def test_lists_only_brain_group(self):
        with TemporaryDirectory() as temp_dir:
            models = list_cancer_models(base=Path(temp_dir))
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["key"], "brain")

    def test_brain_artifact_detected_when_present(self):
        with TemporaryDirectory() as temp_dir:
            brain_dir = Path(temp_dir) / "brain"
            brain_dir.mkdir()
            (brain_dir / "brain_classifier.pt").write_bytes(b"weights")
            m = find_cancer_model("brain", base=Path(temp_dir))
        self.assertTrue(m["artifact_ready"])
        self.assertFalse(m["runtime_ready"])
        self.assertEqual(m["kind"], "torch-state")

    def test_unknown_key_missing(self):
        m = find_cancer_model("unknown-xyz")
        self.assertFalse(m["artifact_ready"])


if __name__ == "__main__":
    unittest.main()
