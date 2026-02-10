from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(
        description="Convert ImageNet-style dataset to LMDB"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of images (e.g., data/raw/imagenet-val)",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        required=True,
        help="Output path for LMDB (e.g., data/imagenet/val_lmdb)",
    )
    parser.add_argument("--write-freq", type=int, default=500, help="Commit frequency")
    return parser.parse_args()


def get_image_list(data_root: Path) -> list[tuple[str, int]]:
    """Generate data list from dataset root.

    Returns:
        list[(path, label)]

    """
    print(f"Scan from {data_root}")
    exts = {".jpg", ".jpeg", ".png", ".JPEG"}
    image_paths: list[Path] = []

    image_paths = [p for p in data_root.glob("*/*") if p.is_file() and p.suffix in exts]
    image_paths.sort()

    classes = sorted({p.parent.name for p in image_paths})
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    data_list = [(str(p), class_to_idx[p.parent.name]) for p in image_paths]
    print(f"Found {len(data_list)} images, {len(classes)} classes.")
    return data_list


def main() -> None:
    """Convert raw jpeg file to LMDB data."""
    args = parse_args()
    data_list = get_image_list(args.data_root)

    map_size = 1099511627776  # 1TB
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)

    env = lmdb.open(args.out_path, map_size=map_size)
    txn = env.begin(write=True)

    print(f"Converting to LMDB at {args.out_path}.")
    for idx, (path, label) in enumerate(tqdm(data_list)):
        with open(path, "rb") as f:
            img_bytes = f.read()
        key = f"{idx:08}"
        value = {
            "img_bytes": img_bytes,
            "label": label,
            "filename": Path(path).name,
        }
        txn.put(key.encode("ascii"), pickle.dumps(value))
        if idx % args.write_freq == 0:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b"length", str(len(data_list)).encode("ascii"))
    txn.commit()
    env.close()

    print(f"Saved to {args.out_path}")

if __name__ == "__main__":
    main()
