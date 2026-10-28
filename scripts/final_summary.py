import sys
sys.path.insert(0, 'D:\\\\Project\\\\OncoVision')
from medical.cancer_model_registry import list_cancer_models

print('=== FINAL MODEL SUMMARY ===')
for m in list_cancer_models():
    ok = m['runtime_ready']
    cl = m['classification_ready']
    files = [f for f in m['files'] if f.endswith(('.pt','.safetensors','.bin','.pth'))]
    status = 'OK' if ok else 'MISSING'
    print(m['key'].upper() + ': ' + status + ' | files=' + str(len(files)))
    for f in sorted(files):
        print('  - ' + f)
    print()

working = sum(1 for m in list_cancer_models() if m['runtime_ready'])
print('SUMMARY: ' + str(working) + '/11 groups working')
print('Stomach: MISSING (24ch architecture mismatch)')