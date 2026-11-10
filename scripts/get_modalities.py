import sys
sys.path.insert(0, '.')

from medical.cancer_catalog import COMMON_CANCER_TARGETS

with open('modalities_output.txt', 'w', encoding='utf-8') as f:
    for t in COMMON_CANCER_TARGETS:
        f.write(f'{t.key:12s}: {t.modalities}\n')