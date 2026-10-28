import sys
sys.path.insert(0, '.')
from medical.cancer_model_registry import probe_runtime_capability
from pathlib import Path

p = Path('models/pretrained/cancers/liver/liver_swin.pt')
ok, reason = probe_runtime_capability(p, 'classifier')
with open('probe_result.txt', 'w', encoding='utf-8') as f:
    f.write(f'Liver SWIN: ok={ok}, reason={reason}')
print('Done')