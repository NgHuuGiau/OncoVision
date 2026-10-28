from pathlib import Path

CANCERS_DIR = Path('models/pretrained/cancers')

GROUPS = [
    'liver', 'lung', 'breast', 'stomach', 'colorectal',
    'prostate', 'cervical', 'kidney', 'pancreas', 'thyroid', 'brain'
]

print('=== CAP NHAT MODEL 11 NHOM UNG THU ===')
total_ready = 0
total_needs_fix = 0
total_convertible = 0

for g in GROUPS:
    gdir = CANCERS_DIR / g
    print(f'\n--- {g.upper()} ---')
    if gdir.exists():
        models = [f for f in gdir.iterdir() if f.is_file() and f.suffix.lower() in {'.pt', '.pth', '.bin', '.safetensors'}]
        if models:
            ready = 0
            needs_fix = 0
            convertible = 0
            for f in sorted(models):
                sz = f.stat().st_size / 1e6
                if '_wrapped.pt' in f.name:
                    print(f'  [OK] {f.name:50s} {sz:>8.1f} MB')
                    ready += 1
                elif f.suffix.lower() == '.safetensors' and f.stat().st_size > 1_000_000:
                    print(f'  [CONV] {f.name:50s} {sz:>8.1f} MB')
                    convertible += 1
                elif f.suffix.lower() in {'.pt', '.pth', '.bin'} and f.stat().st_size > 1_000_000 and 'checkpoint' not in f.name.lower() and '_wrapped' not in f.name:
                    print(f'  [FIX] {f.name:50s} {sz:>8.1f} MB')
                    needs_fix += 1
            total_ready += ready
            total_needs_fix += needs_fix
            total_convertible += convertible
            print(f'  -> Ready: {ready}, Needs fix: {needs_fix}, Convertible: {convertible}')
        else:
            print('  (khong co model)')
    else:
        print('  (khong co thu muc)')

print(f'\n=== TONG KET ===')
print(f'Ready (wrapped): {total_ready}')
print(f'Needs prefix fix: {total_needs_fix}')
print(f'Convertible (.safetensors): {total_convertible}')
print(f'Tong model co the chay sau fix: {total_ready + total_needs_fix + total_convertible}')