"""Download the bank-additional variant from the official UCI dataset archive."""
import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip"
EXPECTED_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
ROOT = Path(__file__).resolve().parents[1]


def csv_from_archive(blob):
    with ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if name.endswith("bank-additional-full.csv"):
                return archive.read(name)
        for name in archive.namelist():
            if name.endswith("bank-additional.zip"):
                return csv_from_archive(archive.read(name))
    raise ValueError("UCI archive does not contain bank-additional-full.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/bank-additional-full.csv")
    parser.add_argument("--force", action="store_true", help="Replace an existing CSV")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        print(f"Already exists: {args.out}; use --force to download again")
        return
    with urlopen(URL, timeout=60) as response:
        csv = csv_from_archive(response.read())
    if hashlib.sha256(csv).hexdigest() != EXPECTED_SHA256:
        raise ValueError("UCI dataset bytes changed; verify the new dataset before training")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(csv)
    print(f"Saved {args.out}; SHA256 {hashlib.sha256(csv).hexdigest()}")


if __name__ == "__main__":
    main()
