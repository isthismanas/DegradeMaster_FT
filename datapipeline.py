from __future__ import annotations

import argparse
import json
import re
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
import pandas as pd


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



class RegressionPipeline:
    """
    Classmethod-based utilities for regression data exploration/preparation.

    This class does not require instantiation; all methods are classmethods
    to support notebook/script usage for preprocessing pipelines.
    """

    # 1) Empty class-level distribution store (populated by inspect_label_distribution).
    label_distribution = {}
    cleaned_distribution = {}
    alignment_report = {}

    _numeric_pattern = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")

    @staticmethod
    def _strict_numeric_or_none(value):
        if pd.isna(value):
            return None
        text = str(value).strip().replace(',', '')
        if text == '' or text.lower() in {'n.d.', 'nd', 'nan', 'none', 'n/a'}:
            return None
        if RegressionPipeline._numeric_pattern.fullmatch(text):
            return float(text)
        return None

    @staticmethod
    def _target_from_tar_path(tar_path: str) -> str:
        tar_path = str(tar_path or '')
        return tar_path.split('_')[0] if '_' in tar_path else tar_path.replace('.pdb', '')

    @staticmethod
    def _e3_from_path(e3_path: str) -> str:
        e3_path = str(e3_path or '')
        return e3_path.split('_')[0] if '_' in e3_path else e3_path.replace('.pdb', '')

    # 4) Check label distribution, print it, and update class variable.
    @classmethod
    def inspect_label_distribution(cls, name_json_path: str | Path):
        name_json_path = Path(name_json_path)
        with open(name_json_path, 'r') as f:
            data = json.load(f)

        labels = [meta.get('label') for meta in data.values()]
        s = pd.Series(labels, dtype='object')
        dist = s.value_counts(dropna=False).to_dict()

        cls.label_distribution = {str(k): int(v) for k, v in dist.items()}
        print(f'[regression] Label distribution from {name_json_path}: {cls.label_distribution}')
        return cls.label_distribution

    # 5) Load protac.csv into DataFrame.
    @classmethod
    def load_protac_csv(cls, protac_csv_path: str | Path) -> pd.DataFrame:
        protac_csv_path = Path(protac_csv_path)
        df = pd.read_csv(protac_csv_path, low_memory=False)
        print(f'[regression] Loaded CSV: {protac_csv_path} | shape={df.shape}')
        return df

    # 6) Keep only rows where DC50 and Dmax are both strictly numeric.
    @classmethod
    def clean_regression_dataframe(
        cls,
        df: pd.DataFrame,
        dc50_col: str = 'DC50 (nM)',
        dmax_col: str = 'Dmax (%)',
    ) -> pd.DataFrame:
        required = ['Compound ID', 'Target', 'E3 ligase', 'Smiles', dc50_col, dmax_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f'Missing required columns in CSV: {missing}')

        work = df.copy()
        work['dc50_nm'] = work[dc50_col].apply(cls._strict_numeric_or_none)
        work['dmax_pct'] = work[dmax_col].apply(cls._strict_numeric_or_none)

        cleaned = work[work['dc50_nm'].notna() & work['dmax_pct'].notna()].copy()
        cleaned['compound_id'] = cleaned['Compound ID'].astype(str).str.strip()
        cleaned['target'] = cleaned['Target'].astype(str).str.strip()
        cleaned['e3_ligase'] = cleaned['E3 ligase'].astype(str).str.strip()

        # Aggregate duplicate assay rows by median so each key has a single numeric target pair.
        grouped = (
            cleaned
            .groupby(['compound_id', 'target', 'e3_ligase'], as_index=False)
            .agg(
                dc50_nm=('dc50_nm', 'median'),
                dmax_pct=('dmax_pct', 'median'),
                source_row_count=('dc50_nm', 'size'),
                smiles=('Smiles', 'first'),
            )
        )

        cls.cleaned_distribution = {
            'raw_rows': int(len(df)),
            'strict_numeric_rows': int(len(cleaned)),
            'unique_key_rows': int(len(grouped)),
        }
        print(f"[regression] Cleaned distribution: {cls.cleaned_distribution}")
        return grouped

    # 7) Explicitly align cleaned regression rows to name.json to avoid index mismatches.
    @classmethod
    def align_with_name_json(
        cls,
        cleaned_df: pd.DataFrame,
        name_json_path: str | Path,
    ) -> pd.DataFrame:
        name_json_path = Path(name_json_path)
        with open(name_json_path, 'r') as f:
            data = json.load(f)

        rows = []
        for sample_id, meta in data.items():
            rows.append({
                'sample_id': sample_id,
                'compound_id': str(meta.get('pro_comp_id', '')).strip(),
                'target': cls._target_from_tar_path(meta.get('tar_path', '')),
                'e3_ligase': cls._e3_from_path(meta.get('e3_ligase_path', '')),
                'label': meta.get('label'),
            })

        name_df = pd.DataFrame(rows)
        merged = name_df.merge(
            cleaned_df,
            how='left',
            on=['compound_id', 'target', 'e3_ligase'],
            validate='m:1',
        )
        merged['has_regression_target'] = merged['dc50_nm'].notna() & merged['dmax_pct'].notna()

        cls.alignment_report = {
            'name_json_rows': int(len(name_df)),
            'aligned_with_regression_targets': int(merged['has_regression_target'].sum()),
            'without_regression_targets': int((~merged['has_regression_target']).sum()),
        }
        print(f'[regression] Alignment report: {cls.alignment_report}')
        return merged

    # 3) Build candidate updated name-json dict including dc50/dmax fields.
    @classmethod
    def build_regression_name_dict(
        cls,
        aligned_df: pd.DataFrame,
        original_name_json_path: str | Path,
    ) -> dict:
        original_name_json_path = Path(original_name_json_path)
        with open(original_name_json_path, 'r') as f:
            original = json.load(f)

        out = {}
        for _, row in aligned_df.iterrows():
            sid = str(row['sample_id'])
            rec = dict(original[sid])
            rec['dc50_nm'] = None if pd.isna(row.get('dc50_nm')) else float(row['dc50_nm'])
            rec['dmax_pct'] = None if pd.isna(row.get('dmax_pct')) else float(row['dmax_pct'])
            rec['has_regression_target'] = bool(row.get('has_regression_target', False))
            out[sid] = rec

        return out

    @classmethod
    def write_regression_name_json(
        cls,
        regression_name_dict: dict,
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(regression_name_dict, f, indent=2)
        print(f'[regression] Saved regression-aware name json: {output_path}')
        return output_path


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
