#!/usr/bin/env python3
"""Download and extract pre-trained PyTorch checkpoints for Ataraxos-UNO.

Downloads the official v1.0.0 model checkpoints release archive and extracts
them into the `runs/` directory.

Usage:
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --force
"""

import argparse
import os
import sys
import tarfile
import urllib.request

RELEASE_TAG = "v1.0.0"
ASSET_NAME = "ataraxos-checkpoints-v1.0.0.tar.gz"
DOWNLOAD_URL = (
    f"https://github.com/TanishkBansode/uno-ataraxos/releases/download/{RELEASE_TAG}/{ASSET_NAME}"
)

EXPECTED_FILES = [
    "runs/blueprint_100m/latest.pt",
    "runs/blueprint_100m/best.pt",
    "runs/belief_20m/best.pt",
    "runs/belief_20m/latest.pt",
    "runs/exploiter_20m/best.pt",
    "runs/exploiter_20m/latest.pt",
    "runs/search_config.json",
]


def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100.0)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 30
        filled = int(bar_len * downloaded / total_size)
        bar = "=" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {pct:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB)")
        sys.stdout.flush()
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloaded {mb_down:6.1f} MB...")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download pre-trained Ataraxos-UNO checkpoints")
    parser.add_argument("--force", action="store_true", help="Overwrite existing checkpoints if present")
    parser.add_argument("--dest", type=str, default=".", help="Root directory to extract checkpoints into")
    args = parser.parse_args()

    root_dir = os.path.abspath(args.dest)
    all_exist = all(os.path.isfile(os.path.join(root_dir, f)) for f in EXPECTED_FILES)

    if all_exist and not args.force:
        print("✓ All pre-trained checkpoints already exist in runs/:")
        for f in EXPECTED_FILES:
            sz = os.path.getsize(os.path.join(root_dir, f)) / (1024 * 1024)
            print(f"  - {f} ({sz:.1f} MB)")
        print("\nUse --force to re-download and overwrite.")
        return

    tar_path = os.path.join(root_dir, ASSET_NAME)
    print(f"Downloading {ASSET_NAME} from GitHub Releases ({RELEASE_TAG})...")
    print(f"URL: {DOWNLOAD_URL}\n")

    try:
        urllib.request.urlretrieve(DOWNLOAD_URL, tar_path, reporthook=progress_hook)
        print("\n\n✓ Download complete. Extracting archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=root_dir)
        print(f"✓ Checkpoints successfully extracted into {os.path.join(root_dir, 'runs')}/")
    except Exception as e:
        print(f"\n❌ Error during download/extraction: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        if os.path.exists(tar_path):
            os.remove(tar_path)

    print("\nVerified extracted files:")
    for f in EXPECTED_FILES:
        fp = os.path.join(root_dir, f)
        if os.path.exists(fp):
            sz = os.path.getsize(fp) / (1024 * 1024)
            print(f"  ✓ {f} ({sz:.1f} MB)")
        else:
            print(f"  ⚠️ Missing: {f}")


if __name__ == "__main__":
    main()
