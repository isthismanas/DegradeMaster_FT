from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from datapipeline import RegressionPipeline


RAW_SUBDIRS = [
    "protac",
    "target_pocket",
    "ligase_pocket",
    "target_ligand",
    "ligase_ligand",
    "features",
]
OPTIONAL_SUBDIRS = ["selected_target", "selected_e3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fully separate regression data workspace for DegradeMaster_FT"
    )
    parser.add_argument("--project-root", type=str, default=".")

    parser.add_argument("--source-data-root", type=str, default="data/PROTAC")
    parser.add_argument("--regression-data-root", type=str, default="data/PROTAC_regression")

    parser.add_argument("--name-json", type=str, default="data/PROTAC/name.json")
    parser.add_argument(
        "--protac-csv",
        type=str,
        default="data/_downloads/PROTAC-8K_extracted/PROTAC-8K/protac.csv",
    )

    parser.add_argument(
        "--regression-name-json",
        type=str,
        default="name_regression.json",
        help="Filename written under regression-data-root",
    )
    parser.add_argument(
        "--trainable-name-json",
        type=str,
        default="name_regression_trainable.json",
        help="Filtered filename (only rows with numeric DC50 and Dmax) under regression-data-root",
    )

    parser.add_argument(
        "--config-template",
        type=str,
        default="config/config.yml",
        help="Base config used to generate regression preprocessing config",
    )
    parser.add_argument(
        "--generated-config",
        type=str,
        default="config/config_regression_prepare.yml",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="name_regression_trainable",
        help="dataset_type value used by GraphData preprocessing",
    )

    parser.add_argument(
        "--skip-graph-preprocess",
        action="store_true",
        help="Only prepare/copy regression files and JSON labels; do not build graph .pt files",
    )
    return parser.parse_args()


def copy_subtree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source directory: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def prepare_regression_workspace(source_root: Path, regression_root: Path) -> None:
    regression_root.mkdir(parents=True, exist_ok=True)

    for subdir in RAW_SUBDIRS:
        copy_subtree(source_root / subdir, regression_root / subdir)

    for subdir in OPTIONAL_SUBDIRS:
        src = source_root / subdir
        if src.exists():
            copy_subtree(src, regression_root / subdir)

    (regression_root / "processed").mkdir(parents=True, exist_ok=True)


def build_trainable_subset(regression_dict: dict) -> dict:
    out = {}
    for key, val in regression_dict.items():
        if not val.get("has_regression_target", False):
            continue
        if val.get("dc50_nm") is None or val.get("dmax_pct") is None:
            continue
        out[key] = val
    return out




def filter_subset_by_feature_index(trainable_dict: dict, max_feature_rows: int) -> tuple[dict, int]:
    kept = {}
    dropped = 0
    for key, val in trainable_dict.items():
        try:
            idx = int(key)
        except ValueError:
            dropped += 1
            continue
        if 0 <= idx < max_feature_rows:
            kept[key] = val
        else:
            dropped += 1
    return kept, dropped

def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def generate_regression_config(
    template_path: Path,
    generated_path: Path,
    dataset_type: str,
) -> Path:
    with open(template_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset_type"] = dataset_type
    cfg["mode"] = "Test"
    cfg["feature"] = True

    generated_path.parent.mkdir(parents=True, exist_ok=True)
    with open(generated_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return generated_path


def run_graph_preprocess(
    project_root: Path,
    regression_root: Path,
    config_path: Path,
    python_exe: str,
) -> None:
    runner = f"""
import sys
sys.argv = ['prepare_regression_graphs', '--config', r'{config_path}']
from prepare_data import GraphData, args
root = r'{regression_root}'
GraphData('protac', root=root,
          select_pocket_war=args.select_pocket_war,
          select_pocket_e3=args.select_pocket_e3,
          conv_name=args.conv_name)
GraphData('ligase_pocket', root=root,
          select_pocket_war=args.select_pocket_war,
          select_pocket_e3=args.select_pocket_e3,
          conv_name=args.conv_name)
GraphData('target_pocket', root=root,
          select_pocket_war=args.select_pocket_war,
          select_pocket_e3=args.select_pocket_e3,
          conv_name=args.conv_name)
print('regression_graph_preprocessing_done')
""".strip()

    subprocess.run([python_exe, "-c", runner], cwd=project_root, check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    source_data_root = project_root / args.source_data_root
    regression_data_root = project_root / args.regression_data_root

    name_json = project_root / args.name_json
    protac_csv = project_root / args.protac_csv

    print("[regression] Step 1: inspect + clean labels")
    RegressionPipeline.inspect_label_distribution(name_json)
    df = RegressionPipeline.load_protac_csv(protac_csv)
    clean_df = RegressionPipeline.clean_regression_dataframe(df)
    aligned_df = RegressionPipeline.align_with_name_json(clean_df, name_json)

    reg_name_dict = RegressionPipeline.build_regression_name_dict(aligned_df, name_json)
    trainable_dict = build_trainable_subset(reg_name_dict)

    print("[regression] Step 2: prepare separate regression workspace")
    prepare_regression_workspace(source_data_root, regression_data_root)

    # Enforce feature/index alignment for trainable subset.
    feature_rows = None
    try:
        import numpy as np
        protac_feat = np.load(regression_data_root / 'features' / 'protac_feature.npy', allow_pickle=True)
        feature_rows = int(protac_feat.shape[0])
    except Exception:
        feature_rows = None

    if feature_rows is not None:
        trainable_dict, dropped = filter_subset_by_feature_index(trainable_dict, feature_rows)
        print(f"[regression] Trainable subset filtered by feature index range [0,{feature_rows - 1}] | dropped={dropped}")

    reg_json_path = regression_data_root / args.regression_name_json
    trainable_json_path = regression_data_root / args.trainable_name_json
    write_json(reg_name_dict, reg_json_path)
    write_json(trainable_dict, trainable_json_path)

    # Keep a copy of original classification name.json in regression workspace for traceability.
    shutil.copy2(name_json, regression_data_root / "name_classification_reference.json")

    print(f"[regression] Wrote: {reg_json_path}")
    print(f"[regression] Wrote: {trainable_json_path} | entries={len(trainable_dict)}")

    print("[regression] Step 3: create regression preprocessing config")
    generated_cfg = generate_regression_config(
        template_path=project_root / args.config_template,
        generated_path=project_root / args.generated_config,
        dataset_type=args.dataset_type,
    )
    print(f"[regression] Generated config: {generated_cfg}")

    if not args.skip_graph_preprocess:
        print("[regression] Step 4: build regression graph/feature/label tensors")
        run_graph_preprocess(
            project_root=project_root,
            regression_root=regression_data_root,
            config_path=generated_cfg,
            python_exe=sys.executable,
        )

    print("[regression] Done")


if __name__ == "__main__":
    main()
