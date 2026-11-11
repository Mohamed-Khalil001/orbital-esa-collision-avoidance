"""Download the ESA Collision Avoidance Challenge dataset from Zenodo into data/.

Usage (from the project root):
    python tools/download_data.py
    python tools/download_data.py --keep-archive

The archive is checked against the MD5 published on the Zenodo record page. Only the files this
project uses are copied out of the archive; their exact location inside the archive is found by name.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RECORD_URL = "https://zenodo.org/records/4463683"
ARCHIVE_URL = RECORD_URL + "/files/Collision%20Avoidance%20Challenge%20-%20Dataset.zip?download=1"
ARCHIVE_MD5 = "d19dc8875229f2f6893253c38adddc87"  # as shown on the Zenodo record page
WANTED = ("train_data.csv", "test_data.csv", "test_data_private.csv")


def md5_of(path, chunk=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target):
    def progress(blocks, block_size, total):
        if total > 0:
            done = min(blocks * block_size, total)
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, target, reporthook=progress)
    print()


def main():
    parser = argparse.ArgumentParser(description="Download the ESA dataset into data/")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--keep-archive", action="store_true", help="keep the downloaded zip file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "_zenodo_download.zip"

    print(f"Downloading from {RECORD_URL} ...")
    try:
        download(ARCHIVE_URL, archive)
    except Exception as err:
        sys.exit(f"Download failed ({type(err).__name__}: {err}).\nDownload the archive by hand from {RECORD_URL} and unzip it into data/.")

    checksum = md5_of(archive)
    if checksum != ARCHIVE_MD5:
        print(f"WARNING: MD5 {checksum} does not match the published {ARCHIVE_MD5}. The file may be incomplete or changed.")
    else:
        print("MD5 checksum OK.")

    found = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base in WANTED and base not in found:
                found[base] = name
        for base, name in found.items():
            with zf.open(name) as src, open(data_dir / base, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  extracted {base}")
        missing = [w for w in WANTED if w not in found]
        if missing:
            print(f"Not found inside the archive: {missing}")
            print("Files in the archive:")
            for name in zf.namelist()[:40]:
                print("   ", name)

    if not args.keep_archive:
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
