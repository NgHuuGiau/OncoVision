"""Tai tiep (resume) 1 URL bang Range requests. Dung: python training/resume_download.py URL DEST"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CHUNK = 1024 * 1024


def main(url: str, dest: str) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    d = Path(dest)
    d.parent.mkdir(parents=True, exist_ok=True)
    have = d.stat().st_size if d.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
    try:
        resp = urllib.request.urlopen(req)
    except Exception as exc:
        print(f"ERR open: {exc}", flush=True)
        return 1
    total = resp.getheader("Content-Range", "")
    print(f"resume from {have / 1e9:.2f}GB ({resp.status} {total})", flush=True)
    mode = "ab" if resp.status == 206 else "wb"
    if resp.status != 206 and have:
        print("server khong ho tro resume - tai lai tu dau", flush=True)
    with open(d, mode) as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
    print(f"done {d.stat().st_size / 1e9:.2f}GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
