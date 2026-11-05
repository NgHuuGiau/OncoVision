import sys
sys.path.insert(0, '.')

from medical.system_status import get_medical_system_status

try:
    s = get_medical_system_status()
    print('model_ready:', s.model_ready)
    print('classification_ready:', s.classification_ready)
    print('localization_ready:', s.localization_ready)
    print('volume_ready:', s.volume_ready)
    print('clinical_validation_ready:', s.clinical_validation_ready)
    print('all_targets_ready:', s.all_targets_ready)
    print('analyzed_cancers:', s.analyzed_cancers)
    print('case_count:', s.case_count)
    print('model_message:', s.model_message[:200])
except Exception as e:
    import traceback
    traceback.print_exc()