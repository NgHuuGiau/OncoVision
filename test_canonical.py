import sys
import io
sys.path.insert(0, '.')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from medical.validator import _canonical_modality

tests = ['MRI não', 'CT gan', 'X-quang ngực', 'Siêu âm gan', 'Nội soi', 'mri', 'ct', 'xray', 'ultrasound']
for t in tests:
    print(t, '->', _canonical_modality(t))