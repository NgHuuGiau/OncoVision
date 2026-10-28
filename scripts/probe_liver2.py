import sys
sys.path.insert(0, '.')
from medical.cancer_model_registry import probe_runtime_capability, target_model_paths
from pathlib import Path

p = Path('models/pretrained/cancers/liver/liver_swin.pt')
print('File exists:', p.exists())

# Check target_model_paths
paths = target_model_paths('liver')
print('Target paths:', paths)

ok, reason = probe_runtime_capability(p, 'classifier')
print(f'Liver SWIN: ok={ok}, reason={reason}')