from __future__ import annotations

import io
import unittest

import run_tests


class DummyPassingTest(unittest.TestCase):
    def test_ok(self) -> None:
        self.assertTrue(True)


class RunTestsDashboardTests(unittest.TestCase):
    def test_meter_uses_unicode_characters(self) -> None:
        text = run_tests.meter(3, 5, width=5)
        self.assertIn("█", text)
        self.assertIn("·", text)
        self.assertNotIn("â", text)
        self.assertNotIn("Â", text)

    def test_pretty_runner_renders_vietnamese_summary(self) -> None:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(DummyPassingTest)
        stream = io.StringIO()
        runner = run_tests.PrettyTestRunner(verbosity=0, total_tests=suite.countTestCases(), stream=stream)

        result = runner.run(suite)
        output = stream.getvalue()

        self.assertTrue(result.wasSuccessful())
        self.assertIn("TỔNG QUAN", output)
        self.assertIn("KẾT QUẢ CUỐI", output)
        self.assertIn("TOÀN BỘ TEST PASS", output)

