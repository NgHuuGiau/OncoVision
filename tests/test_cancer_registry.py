import unittest

from medical.cancer_model_registry import find_cancer_model, list_cancer_models


class CancerRegistryTests(unittest.TestCase):
    def test_lists_11_groups(self):
        self.assertEqual(len(list_cancer_models()), 11)

    def test_thyroid_detected(self):
        m = find_cancer_model("thyroid")
        self.assertTrue(m["artifact_ready"])
        self.assertFalse(m["runtime_ready"])
        self.assertIn(m["kind"], {"hf-transformers", "torch-state", "safetensors", "ckpt", "keras"})

    def test_unknown_key_missing(self):
        m = find_cancer_model("unknown-xyz")
        self.assertFalse(m["artifact_ready"])


if __name__ == "__main__":
    unittest.main()
