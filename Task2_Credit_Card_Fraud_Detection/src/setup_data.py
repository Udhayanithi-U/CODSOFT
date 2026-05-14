"""Extract the credit card fraud dataset into the local data directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REQUIRED_FILES = {"fraudTrain.csv", "fraudTest.csv"}


def extract_dataset(zip_path: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    DATA_DIR.mkdir(exist_ok=True)

    with ZipFile(zip_path) as archive:
        names = {Path(info.filename).name for info in archive.infolist()}
        missing = REQUIRED_FILES - names
        if missing:
            raise ValueError(
                f"Dataset zip is missing required file(s): {', '.join(sorted(missing))}"
            )

        for info in archive.infolist():
            filename = Path(info.filename).name
            if filename in REQUIRED_FILES:
                target = DATA_DIR / filename
                print(f"Extracting {filename} -> {target}")
                with archive.open(info) as source, target.open("wb") as destination:
                    destination.write(source.read())

    print("Dataset setup complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fraud dataset CSV files.")
    parser.add_argument(
        "--zip-path",
        type=Path,
        required=True,
        help="Path to the downloaded archive zip file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_dataset(args.zip_path)
