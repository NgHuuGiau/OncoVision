import sys
sys.path.insert(0, '.')

from medical.cancer_catalog import COMMON_CANCER_TARGETS

for t in COMMON_CANCER_TARGETS:
    print(f'{t.key:12s}: {t.modalities}')