#!/usr/bin/env python3
import os
models = ['lung', 'colorectal', 'breast', 'thyroid']
for m in models:
    path = 'models/pretrained/cancers/' + m + '/volume.pt'
    status = 'OK' if os.path.exists(path) else 'MISSING'
    print(m + ' volume: ' + status)
