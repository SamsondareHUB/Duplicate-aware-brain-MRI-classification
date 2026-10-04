"""Download the official Figshare release (article 1512427, v8), verify MD5, write provenance.

Raw files go to data/raw/figshare/ (git-ignored). Only the provenance JSON is committed.
Usage: python scripts/download_figshare.py [--extract]
"""
import argparse, datetime as dt, hashlib, json, os, urllib.request, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "figshare"
EXTRACTED = ROOT / "data" / "raw" / "extracted"
ARTICLE = {"article_id": 1512427, "version": 8, "published": "2024-12-21", "license": "CC BY 4.0",
           "doi": "https://doi.org/10.6084/m9.figshare.1512427",
           "api": "https://api.figshare.com/v2/articles/1512427"}
# (filename, figshare file id, expected md5 from the Figshare API record)
FILES = [
    ("brainTumorDataPublic_1-766.zip", 3381290, "74b949ad33f042e6e103523091cd1428"),
    ("brainTumorDataPublic_767-1532.zip", 3381296, "7e8a875500d2c8a346f270538e29890e"),
    ("brainTumorDataPublic_1533-2298.zip", 3381293, "8227bf6080cb71f15a88be8d25c79ae7"),
    ("brainTumorDataPublic_2299-3064.zip", 3381302, "b378a80d6174e5317d59eb28430c6652"),
    ("cvind.mat", 7005344, "5ea82eb3212c9fcb84290ccf0beb486d"),
    ("README_2024.txt", 51340418, "4188f7c30957e9673c3611613071bd6e"),  # published as "README 2024.txt"
]


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    for name, fid, expected in FILES:
        p = RAW / name
        if not p.exists():
            urllib.request.urlretrieve(f"https://ndownloader.figshare.com/files/{fid}", p)
        obs = md5(p)
        ok &= obs == expected
        rows.append({"filename": name, "figshare_file_id": fid, "size_bytes": p.stat().st_size,
                     "expected_md5": expected, "observed_md5": obs, "md5_match": obs == expected,
                     "file_mtime_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc).isoformat(),
                     "source_url": f"https://ndownloader.figshare.com/files/{fid}"})
    out = {**ARTICLE, "all_md5_match": ok, "manifest_written_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
           "files": rows}
    (ROOT / "reports" / "manifests").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "manifests" / "provenance.json").write_text(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit("MD5 MISMATCH - not extracting")
    if a.extract:
        EXTRACTED.mkdir(parents=True, exist_ok=True)
        for name, _, _ in FILES:
            if name.endswith(".zip"):
                with zipfile.ZipFile(RAW / name) as z:
                    z.extractall(EXTRACTED)
    print("all MD5 match:", ok)


if __name__ == "__main__":
    main()
