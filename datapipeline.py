from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests


class DataPipeline:
    DATASETS = {
        "PROTAC-8K": "https://zenodo.org/records/14715718/files/PROTAC-8K.zip?download=1"
    }

    # Raw structural folders expected from dataset archives.
    RAW_REQUIRED_SUBDIRS = [
        "protac",
        "target_pocket",
        "ligase_pocket",
        "target_ligand",
        "ligase_ligand",
    ]

    # Metadata required by main.py/prepare_data.py in final data/PROTAC.
    METADATA_FILES = [
        "name.json",
        "features/protac_feature.npy",
        "features/target_feature.npy",
        "features/e3_feature.npy",
    ]

    @staticmethod
    def _filename_from_url(dataset_url: str, fallback: str = "dataset.zip") -> str:
        parsed = urlparse(dataset_url)
        name = os.path.basename(parsed.path)
        return name if name else fallback

    @classmethod
    def download_dataset(cls, dataset_name: str, directory: str | Path, chunk_size: int = 1024 * 1024) -> Path:
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Invalid dataset name: {dataset_name}. Available: {list(cls.DATASETS)}")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        dataset_url = cls.DATASETS[dataset_name]
        file_name = cls._filename_from_url(dataset_url, fallback=f"{dataset_name}.zip")
        save_path = directory / file_name
        if save_path.exists() and save_path.stat().st_size > 0:
            print(f"[download] Using cached archive: {save_path}")
            return save_path

        print(f"[download] Downloading {dataset_name} from {dataset_url}")
        with requests.get(dataset_url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(save_path, "wb") as fout:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        fout.write(chunk)

        print(f"[download] Saved archive to: {save_path}")
        return save_path

    @classmethod
    def extract_dataset(cls, archive_path: str | Path, directory: str | Path) -> Path:
        archive_path = Path(archive_path)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        print(f"[extract] Extracting {archive_path.name} -> {directory}")
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(directory)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as tar_ref:
                tar_ref.extractall(directory)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

        return directory

    @classmethod
    def _is_raw_dataset_root(cls, path: Path) -> bool:
        return all((path / subdir).exists() for subdir in cls.RAW_REQUIRED_SUBDIRS)

    @classmethod
    def _raw_file_score(cls, path: Path) -> int:
        score = 0
        for subdir in cls.RAW_REQUIRED_SUBDIRS:
            d = path / subdir
            if d.exists():
                score += sum(1 for x in d.iterdir() if x.is_file())
        return score

    @classmethod
    def find_dataset_root(cls, extracted_dir: str | Path) -> Path:
        extracted_dir = Path(extracted_dir)

        candidates = []
        if cls._is_raw_dataset_root(extracted_dir) and "__MACOSX" not in extracted_dir.parts:
            candidates.append(extracted_dir)

        for candidate in extracted_dir.rglob("*"):
            if not candidate.is_dir():
                continue
            if "__MACOSX" in candidate.parts:
                continue
            if cls._is_raw_dataset_root(candidate):
                candidates.append(candidate)

        if not candidates:
            raise FileNotFoundError(
                "Could not find a valid raw dataset root after extraction. "
                f"Expected folder containing raw directories: {cls.RAW_REQUIRED_SUBDIRS}"
            )

        candidates.sort(key=cls._raw_file_score, reverse=True)
        return candidates[0]

    @classmethod
    def sync_to_project_layout(cls, source_root: str | Path, target_root: str | Path) -> Path:
        source_root = Path(source_root)
        target_root = Path(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

        print(f"[sync] Copying dataset into expected path: {target_root}")

        # Always sync raw folders first.
        for subdir in cls.RAW_REQUIRED_SUBDIRS:
            src = source_root / subdir
            if not src.exists():
                continue
            dest = target_root / subdir
            shutil.copytree(src, dest, dirs_exist_ok=True)

        # Optional metadata sync if archive provides it.
        for item in source_root.iterdir():
            if item.name in cls.RAW_REQUIRED_SUBDIRS:
                continue
            if item.name.startswith('.'):
                continue
            dest = target_root / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

        return target_root

    @classmethod
    def validate_project_layout(cls, target_root: str | Path, check_metadata: bool = True) -> None:
        target_root = Path(target_root)

        missing_raw = [d for d in cls.RAW_REQUIRED_SUBDIRS if not (target_root / d).exists()]
        if missing_raw:
            raise FileNotFoundError(
                f"Missing raw folders in {target_root}: {missing_raw}. "
                "Download/extract likely incomplete."
            )

        if check_metadata:
            missing_meta = [f for f in cls.METADATA_FILES if not (target_root / f).exists()]
            if missing_meta:
                raise FileNotFoundError(
                    "Raw folders are ready, but required metadata is missing in data/PROTAC: "
                    f"{missing_meta}. "
                    "This project needs name.json and features/*.npy to run main.py."
                )

    @classmethod
    def prepare_protac8k(cls, project_root: str | Path = ".", check_metadata: bool = True) -> Path:
        project_root = Path(project_root).resolve()
        download_dir = project_root / "data" / "_downloads"
        extract_dir = download_dir / "PROTAC-8K_extracted"

        archive = cls.download_dataset("PROTAC-8K", download_dir)
        try:
            source_root = cls.find_dataset_root(extract_dir)
            print(f"[extract] Reusing existing extraction at: {source_root}")
        except FileNotFoundError:
            cls.extract_dataset(archive, extract_dir)
            source_root = cls.find_dataset_root(extract_dir)

        target_root = project_root / "data" / "PROTAC"
        cls.sync_to_project_layout(source_root, target_root)
        cls.validate_project_layout(target_root, check_metadata=check_metadata)
        print(f"[done] Dataset prepared at: {target_root}")
        return target_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare PROTAC-8K for DegradeMaster_FT")
    parser.add_argument("--dataset", type=str, default="PROTAC-8K", choices=["PROTAC-8K"])
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root containing data/PROTAC (default: current directory)",
    )
    parser.add_argument(
        "--skip-metadata-check",
        action="store_true",
        help="Skip validation of name.json and features/*.npy in data/PROTAC",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dataset != "PROTAC-8K":
        raise ValueError("Only PROTAC-8K is currently configured")
    DataPipeline.prepare_protac8k(project_root=args.project_root, check_metadata=not args.skip_metadata_check)
