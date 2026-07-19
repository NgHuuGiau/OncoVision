from __future__ import annotations

import unittest
from unittest.mock import patch

from medical import network_policy


class NetworkPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        network_policy._warned_contexts.clear()

    def test_download_blocked_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(network_policy.weight_download_allowed())

    def test_download_allowed_when_env_set(self) -> None:
        for value in ("1", "true", "yes", "on", "TRUE"):
            with patch.dict("os.environ", {"ONCOVISION_ALLOW_WEIGHT_DOWNLOAD": value}, clear=True):
                self.assertTrue(network_policy.weight_download_allowed(), value)

    def test_resolve_pretrained_false_stays_false(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(network_policy.resolve_pretrained(False))

    def test_resolve_pretrained_true_downgraded_when_offline(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(network_policy.resolve_pretrained(True, context="test-backbone"))

    def test_resolve_pretrained_true_kept_when_allowed(self) -> None:
        with patch.dict("os.environ", {"ONCOVISION_ALLOW_WEIGHT_DOWNLOAD": "1"}, clear=True):
            self.assertTrue(network_policy.resolve_pretrained(True, context="test-backbone"))

    def test_warns_only_once_per_context(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with patch("builtins.print") as print_mock:
                network_policy.resolve_pretrained(True, context="ctx-a")
                network_policy.resolve_pretrained(True, context="ctx-a")
                network_policy.resolve_pretrained(True, context="ctx-b")
            self.assertEqual(print_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
