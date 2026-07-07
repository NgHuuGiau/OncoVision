from __future__ import annotations

from unittest import TestCase

from medical.cancer_dataset_registry import COMMON_CANCER_DATASET_SOURCES


class CancerDatasetRegistryTests(TestCase):
    def test_common_sources_include_isic_and_tcia_sources(self) -> None:
        sources = COMMON_CANCER_DATASET_SOURCES
        ids = [source.source_id for source in sources]
        self.assertIn("isic_skin", ids)
        self.assertIn("tcia_breast", ids)
        self.assertIn("tcia_lung", ids)
        self.assertIn("tcia_colorectal", ids)
        self.assertIn("tcia_liver", ids)
        self.assertIn("tcia_stomach", ids)
