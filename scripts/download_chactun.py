#!/usr/bin/env python
"""Download the Chactun ML-ready dataset (Kokalj et al. 2023) from Figshare.

Uses the public Figshare API, so no login is needed. Lists the files first;
by default downloads only the LiDAR visualizations and masks (what the baseline
needs). Pass --all to also get Sentinel-1/2 and the canopy height model.

    python scripts/download_chactun.py --list           # just show the files
    python scripts/download_chactun.py                  # lidar + masks
    python scripts/download_chactun.py --all            # everything
    python scripts/download_chactun.py --match lidar    # any name containing "lidar"

Zips go to data/raw/chactun/; lidar tiles and masks are extracted to data/chactun/.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from mayalidar.data import extract_chactun

ARTICLE_ID = 22202395  # doi:10.6084/m9.figshare.22202395
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "chactun"
OUT = ROOT / "data" / "chactun"

# Name fragments that mark the files the ALS baseline needs. Checked against
# the real names at run time; if nothing matches you'll get the full list.
DEFAULT_KEYS = ("lidar", "als", "mask", "annot")


def list_files() -> list[dict]:
    files, page = [], 1
    while True:
        r = requests.get(f"{API}/files", params={"page": page, "page_size": 100}, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        files += batch
        page += 1
    return files


def md5(path: Path, chunk=1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(f: dict) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / f["name"]
    expected = f.get("computed_md5") or f.get("supplied_md5")
    if dest.exists() and dest.stat().st_size == f["size"]:
        if not expected or md5(dest) == expected:
            print(f"  already have {dest.name}")
            return dest
    with requests.get(f["download_url"], stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh, tqdm(total=f["size"], unit="B", unit_scale=True, desc=f["name"]) as bar:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                bar.update(len(block))
        tmp.rename(dest)
    if expected and md5(dest) != expected:
        sys.exit(f"MD5 mismatch for {dest.name}; delete it and rerun.")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="only list files")
    ap.add_argument("--all", action="store_true", help="download every file")
    ap.add_argument("--match", nargs="*", help="download files whose name contains any of these")
    args = ap.parse_args()

    files = list_files()
    print(f"{len(files)} files in figshare article {ARTICLE_ID}:")
    for f in files:
        print(f"  {f['size']/1e9:7.2f} GB  {f['name']}")
    if args.list:
        return

    keys = tuple(k.lower() for k in (args.match or DEFAULT_KEYS))
    chosen = files if args.all else [f for f in files if any(k in f["name"].lower() for k in keys)]
    if not chosen:
        sys.exit(f"No file names contain {keys}. Pick from the list above with --match, or use --all.")
    print(f"\nDownloading {len(chosen)} file(s), {sum(f['size'] for f in chosen)/1e9:.2f} GB")
    for f in chosen:
        path = download(f)
        if path.suffix == ".zip":
            extract_chactun(path, OUT)
    print(f"\nDone. Data is under {OUT.relative_to(ROOT)}/ ; next, notebooks/01_explore_chactun.ipynb")


if __name__ == "__main__":
    main()
