"""Extract the SMS spam dataset into the local data folder."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILE = "spam.csv"


def extract_dataset(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    DATA_DIR.mkdir(exist_ok=True)

    with ZipFile(zip_path) as archive:
        matching_files = [
            info for info in archive.infolist() if Path(info.filename).name == DATASET_FILE
        ]

        if not matching_files:
            raise ValueError(f"{DATASET_FILE} was not found inside {zip_path}")

        target_path = DATA_DIR / DATASET_FILE
        print(f"Extracting {DATASET_FILE} -> {target_path}")
        with archive.open(matching_files[0]) as source, target_path.open("wb") as target:
            target.write(source.read())

    print("Dataset setup complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SMS spam dataset.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to the downloaded dataset zip file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_dataset(args.zip_path)
