"""Extract the Chactun lidar tiles and masks into data/chactun/.

Accepts the Figshare bundle (22202395.zip, which contains further zips) or any zip
holding the tile files directly. Safe to rerun; finished files are skipped.

    python scripts/extract_chactun.py /path/to/22202395.zip
"""
import sys
from pathlib import Path

from mayalidar.data import extract_chactun

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    extract_chactun(sys.argv[1], Path(__file__).resolve().parents[1] / "data" / "chactun")
