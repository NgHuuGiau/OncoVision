import sys
sys.path.insert(0, '.')

try:
    from medical.cancer_model_registry import probe_runtime_capability
    targets = ['brain', 'lung', 'breast', 'liver', 'stomach', 'colorectal', 'prostate', 'cervical', 'kidney', 'pancreas', 'thyroid']
    kinds = ['classifier', 'detector', 'volume']
    for target in targets:
        for kind in kinds:
            result = probe_runtime_capability(target, kind)
            status = "OK" if result else "MISSING"
            print(f"{target} {kind}: {status}")
except Exception as e:
    import traceback
    traceback.print_exc()