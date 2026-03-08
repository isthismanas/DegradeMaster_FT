import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from config.config import get_args
from model import EGNNConv, GATTConv, GraphConv, ProtacModel
from nn_utils import setup_seed
from regression_models import (
    build_cross_attention_gnn,
    build_pdc50_bounded_gnn,
    build_two_head_gnn,
    regression_metrics,
    TabularRegressionBaseline,
)


def collater_regression(data_list):
    batch = {}
    batch["name"] = [x["name"] for x in data_list]
    batch["protac_graphs"] = Batch.from_data_list([x["protac_graphs"] for x in data_list])
    batch["ligase_pocket"] = Batch.from_data_list([x["ligase_pocket"] for x in data_list])
    batch["target_pocket"] = Batch.from_data_list([x["target_pocket"] for x in data_list])
    batch["feature"] = torch.as_tensor(np.asarray([x["feature"] for x in data_list], dtype=np.float32))
    batch["label"] = torch.as_tensor(np.asarray([x["label"] for x in data_list], dtype=np.float32))
    return batch


def build_base_model(args):
    if args.conv_name == "GCN":
        ligase_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.e3_dim, hidden_size=args.hidden_size)
        target_pocket_model = GraphConv(num_embeddings=118, graph_dim=args.tar_dim, hidden_size=args.hidden_size)
        protac_model = GraphConv(num_embeddings=118, graph_dim=args.protac_dim, hidden_size=args.hidden_size)
    elif args.conv_name == "GAT":
        ligase_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        target_pocket_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
        protac_model = GATTConv(num_embeddings=118, hidden_size=args.hidden_size)
    elif args.conv_name == "EGNN":
        ligase_pocket_model = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.e3_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
        protac_model = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.protac_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
        target_pocket_model = EGNNConv(
            num_embeddings=118,
            in_node_nf=1,
            in_edge_nf=1,
            graph_nf=args.tar_dim,
            hidden_nf=args.hidden_size,
            n_layers=args.n_layers,
            node_attr=0,
            attention=args.attention,
        )
    else:
        raise ValueError(f"Unsupported conv_name: {args.conv_name}")

    return ProtacModel(
        protac_model=protac_model,
        ligase_pocket_model=ligase_pocket_model,
        target_pocket_model=target_pocket_model,
        hidden_size=args.hidden_size,
    )


def resolve_variant(args):
    return str(getattr(args, "regression_variant", "base")).lower().strip()


def build_model_for_variant(args, variant):
    if variant == "base":
        return build_base_model(args)
    if variant == "two_head":
        return build_two_head_gnn(args)
    if variant == "pdc50_bounded":
        return build_pdc50_bounded_gnn(args)
    if variant == "cross_attention":
        return build_cross_attention_gnn(args)
    raise ValueError(
        f"Unsupported regression_variant: {variant}. Supported: base, two_head, pdc50_bounded, cross_attention, tabular"
    )


def split_indices(total_size, train_rate, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(total_size)
    rng.shuffle(indices)
    train_size = int(total_size * train_rate)
    train_idx = indices[:train_size].tolist()
    test_idx = indices[train_size:].tolist()
    return train_idx, test_idx


@dataclass
class RegressionTargetTransform:
    mean: torch.Tensor
    std: torch.Tensor
    log_dc50: bool
    normalize: bool

    @classmethod
    def fit(cls, y_train, log_dc50=True, normalize=True):
        y = torch.as_tensor(y_train, dtype=torch.float32).clone()
        if log_dc50:
            y[:, 0] = torch.log10(torch.clamp(y[:, 0], min=1e-8))
        mean = y.mean(dim=0)
        std = y.std(dim=0)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)
        return cls(mean=mean, std=std, log_dc50=log_dc50, normalize=normalize)

    def transform(self, y_raw):
        y = y_raw.clone()
        if self.log_dc50:
            y[:, 0] = torch.log10(torch.clamp(y[:, 0], min=1e-8))
        if self.normalize:
            mean = self.mean.to(y.device)
            std = self.std.to(y.device)
            y = (y - mean) / std
        return y

    def inverse_transform(self, y_scaled):
        y = y_scaled.clone()
        if self.normalize:
            mean = self.mean.to(y.device)
            std = self.std.to(y.device)
            y = y * std + mean
        if self.log_dc50:
            y[:, 0] = torch.pow(10.0, y[:, 0])
        return y

    def to_serializable(self):
        return {
            "kind": "log10_normalized",
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "log_dc50": bool(self.log_dc50),
            "normalize": bool(self.normalize),
        }

    @classmethod
    def from_serializable(cls, payload):
        return cls(
            mean=torch.tensor(payload["mean"], dtype=torch.float32),
            std=torch.tensor(payload["std"], dtype=torch.float32),
            log_dc50=bool(payload.get("log_dc50", True)),
            normalize=bool(payload.get("normalize", True)),
        )


@dataclass
class PDC50DmaxTransform:
    dmax_upper: float = 100.0

    def transform(self, y_raw):
        y = y_raw.clone()
        dc50_nm = torch.clamp(y[:, 0], min=1e-8)
        pdc50 = 9.0 - torch.log10(dc50_nm)
        dmax = torch.clamp(y[:, 1], min=0.0, max=self.dmax_upper)
        return torch.stack((pdc50, dmax), dim=1)

    def inverse_transform(self, y_model):
        y = y_model.clone()
        pdc50 = y[:, 0]
        dmax = torch.clamp(y[:, 1], min=0.0, max=self.dmax_upper)
        dc50_nm = torch.pow(10.0, 9.0 - pdc50)
        return torch.stack((dc50_nm, dmax), dim=1)

    def to_serializable(self):
        return {"kind": "pdc50_bounded", "dmax_upper": float(self.dmax_upper)}

    @classmethod
    def from_serializable(cls, payload):
        return cls(dmax_upper=float(payload.get("dmax_upper", 100.0)))


def deserialize_target_transform(payload):
    kind = payload.get("kind", "log10_normalized")
    if kind == "pdc50_bounded":
        return PDC50DmaxTransform.from_serializable(payload)
    return RegressionTargetTransform.from_serializable(payload)


def build_target_transform(args, y_train, variant):
    if variant == "pdc50_bounded":
        if bool(getattr(args, "normalize_targets", False)):
            print("[warn] normalize_targets ignored for pdc50_bounded variant")
        return PDC50DmaxTransform(dmax_upper=float(getattr(args, "dmax_upper", 100.0)))

    return RegressionTargetTransform.fit(
        y_train,
        log_dc50=bool(getattr(args, "log_dc50_target", True)),
        normalize=bool(getattr(args, "normalize_targets", True)),
    )


def load_dataset_arrays(dataset_root, dataset_type):
    json_path = dataset_root / f"{dataset_type}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Dataset json not found: {json_path}")

    with open(json_path, "r") as f:
        name_dict = json.load(f)
    name_list = list(name_dict.keys())

    label = np.asarray(torch.load(dataset_root / "processed" / "label.pt"), dtype=np.float32)
    feature = np.asarray(torch.load(dataset_root / "processed" / "feature.pt"), dtype=np.float32)

    if label.ndim != 2 or label.shape[1] != 2:
        raise ValueError(
            f"Regression labels must be shaped [N,2] for [DC50,Dmax], got shape {label.shape}"
        )

    if len(name_list) != len(label) or len(name_list) != len(feature):
        raise ValueError(
            f"Length mismatch: name_list={len(name_list)}, labels={len(label)}, features={len(feature)}"
        )

    return name_list, label, feature


def build_graph_dataloaders(args, dataset_root, name_list, label, feature, train_indices, test_indices):
    from prepare_data import GraphData
    from protacloader import PROTACSet

    protac_graphs = GraphData(
        "protac",
        root=str(dataset_root),
        select_pocket_war=args.select_pocket_war,
        select_pocket_e3=args.select_pocket_e3,
        conv_name=args.conv_name,
    )
    ligase_pocket = GraphData(
        "ligase_pocket",
        root=str(dataset_root),
        select_pocket_war=args.select_pocket_war,
        select_pocket_e3=args.select_pocket_e3,
        conv_name=args.conv_name,
    )
    target_pocket = GraphData(
        "target_pocket",
        root=str(dataset_root),
        select_pocket_war=args.select_pocket_war,
        select_pocket_e3=args.select_pocket_e3,
        conv_name=args.conv_name,
    )

    dataset = PROTACSet(
        name=name_list,
        protac_graphs=protac_graphs,
        ligase_pocket=ligase_pocket,
        target_pocket=target_pocket,
        feature=feature,
        label=label,
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    num_workers = int(getattr(args, "num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collater_regression,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collater_regression,
        drop_last=False,
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device, target_tf):
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        y_raw = batch["label"].to(device).float()

        preds, _ = model(
            batch["protac_graphs"].to(device),
            batch["ligase_pocket"].to(device),
            batch["target_pocket"].to(device),
            batch["feature"].to(device),
        )
        y_scaled = target_tf.transform(y_raw)
        loss = criterion(preds, y_scaled)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(model, loader, device, target_tf):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_batches = 0
    all_true = []
    all_pred = []
    all_names = []

    with torch.no_grad():
        for batch in loader:
            y_raw = batch["label"].to(device).float()
            preds_scaled, _ = model(
                batch["protac_graphs"].to(device),
                batch["ligase_pocket"].to(device),
                batch["target_pocket"].to(device),
                batch["feature"].to(device),
            )
            y_scaled = target_tf.transform(y_raw)
            loss = criterion(preds_scaled, y_scaled)
            total_loss += loss.item()
            total_batches += 1

            preds_raw = target_tf.inverse_transform(preds_scaled).cpu().numpy()
            all_true.append(y_raw.cpu().numpy())
            all_pred.append(preds_raw)
            all_names.extend(batch["name"])

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)

    metrics = regression_metrics(y_true, y_pred)
    metrics["loss_scaled"] = float(total_loss / max(total_batches, 1))
    return metrics, y_true, y_pred, all_names


def print_metrics(prefix, metrics):
    print(
        f"{prefix} | Loss(scaled): {metrics.get('loss_scaled', float('nan')):.6f} | "
        f"RMSE(DC50): {metrics['rmse_dc50_nm']:.4f} | RMSE(Dmax): {metrics['rmse_dmax_pct']:.4f} | "
        f"MAE(DC50): {metrics['mae_dc50_nm']:.4f} | MAE(Dmax): {metrics['mae_dmax_pct']:.4f} | "
        f"R2(DC50): {metrics['r2_dc50_nm']:.4f} | R2(Dmax): {metrics['r2_dmax_pct']:.4f}"
    )


def run_gnn_variant(args, variant, dataset_root, name_list, label, feature, save_dir, split_payload):
    train_indices = split_payload["train_indices"]
    test_indices = split_payload["test_indices"]

    train_loader, test_loader = build_graph_dataloaders(
        args=args,
        dataset_root=dataset_root,
        name_list=name_list,
        label=label,
        feature=feature,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_variant(args, variant).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(getattr(args, "lr_patience", 20)),
    )

    y_train = label[train_indices]
    target_tf = build_target_transform(args, y_train=y_train, variant=variant)

    best_ckpt = save_dir / "best_model.pth"
    last_ckpt = save_dir / "last_model.pth"

    if str(args.mode).lower() == "train":
        best_metric = float("inf")
        best_epoch = -1

        for epoch in range(1, args.epoch + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, target_tf)
            test_metrics, _, _, _ = evaluate(model, test_loader, device, target_tf)
            scheduler.step(test_metrics["mean_rmse"])

            log_every = int(getattr(args, "log_every", 10))
            if epoch % log_every == 0 or epoch == 1 or epoch == args.epoch:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch}/{args.epoch} | Train Loss(scaled): {train_loss:.6f} | "
                    f"LR: {lr_now:.6e} | Mean RMSE: {test_metrics['mean_rmse']:.4f}"
                )
                print_metrics("Test", test_metrics)

            if test_metrics["mean_rmse"] < best_metric:
                best_metric = test_metrics["mean_rmse"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "variant": variant,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": test_metrics,
                        "target_transform": target_tf.to_serializable(),
                        "config": vars(args),
                    },
                    best_ckpt,
                )

        torch.save(
            {
                "epoch": args.epoch,
                "variant": variant,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "target_transform": target_tf.to_serializable(),
                "config": vars(args),
            },
            last_ckpt,
        )
        print(f"Training done. Best epoch={best_epoch}, best mean RMSE={best_metric:.4f}")

    ckpt_to_load = str(getattr(args, "checkpoint_path", "")).strip()
    ckpt_path = Path(ckpt_to_load).resolve() if ckpt_to_load else best_ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    loaded = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(loaded["model_state_dict"])
    target_tf = deserialize_target_transform(loaded["target_transform"])
    print(f"Loaded checkpoint: {ckpt_path}")

    final_metrics, y_true, y_pred, names = evaluate(model, test_loader, device, target_tf)
    print_metrics("Final Test", final_metrics)

    if bool(getattr(args, "save_predictions", True)):
        out_path = save_dir / "test_predictions.npz"
        np.savez(
            out_path,
            names=np.asarray(names),
            y_true=y_true,
            y_pred=y_pred,
        )
        print(f"Saved predictions: {out_path}")

    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics: {save_dir / 'final_metrics.json'}")


def run_tabular_variant(args, variant, label, feature, save_dir, split_payload):
    del variant

    train_indices = split_payload["train_indices"]
    test_indices = split_payload["test_indices"]

    x_train = feature[train_indices]
    x_test = feature[test_indices]
    y_train_raw = label[train_indices]
    y_test_raw = label[test_indices]

    target_tf = build_target_transform(args, y_train=y_train_raw, variant="base")
    y_train_target = target_tf.transform(torch.as_tensor(y_train_raw, dtype=torch.float32)).numpy()

    backend = str(getattr(args, "tabular_backend", "hist_gb"))
    model = TabularRegressionBaseline(
        backend=backend,
        random_state=int(args.seed),
        learning_rate=float(getattr(args, "tabular_learning_rate", 0.05)),
        max_depth=int(getattr(args, "tabular_max_depth", 6)),
        max_iter=int(getattr(args, "tabular_max_iter", 500)),
        n_estimators=int(getattr(args, "tabular_n_estimators", 500)),
        n_jobs=int(getattr(args, "tabular_n_jobs", -1)),
    )

    model_path = save_dir / "best_model.pkl"
    payload_path = save_dir / "model_payload.pkl"

    if str(args.mode).lower() == "train":
        model.fit(x_train, y_train_target)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(payload_path, "wb") as f:
            pickle.dump(
                {
                    "variant": "tabular",
                    "target_transform": target_tf.to_serializable(),
                    "backend": backend,
                },
                f,
            )

    if not model_path.exists() or not payload_path.exists():
        raise FileNotFoundError(f"Tabular checkpoint missing under {save_dir}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(payload_path, "rb") as f:
        payload = pickle.load(f)
    target_tf = deserialize_target_transform(payload["target_transform"])

    y_pred_target = model.predict(x_test)
    y_pred_raw = target_tf.inverse_transform(torch.as_tensor(y_pred_target, dtype=torch.float32)).numpy()

    final_metrics = regression_metrics(y_test_raw, y_pred_raw)
    final_metrics["loss_scaled"] = float("nan")
    print_metrics("Final Test", final_metrics)

    if bool(getattr(args, "save_predictions", True)):
        out_path = save_dir / "test_predictions.npz"
        test_names = np.asarray(split_payload["test_names"])
        np.savez(
            out_path,
            names=test_names,
            y_true=y_test_raw,
            y_pred=y_pred_raw,
        )
        print(f"Saved predictions: {out_path}")

    with open(save_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics: {save_dir / 'final_metrics.json'}")


def main():
    args = get_args()
    setup_seed(args.seed)

    project_root = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root
    dataset_root = dataset_root.resolve()

    name_list, label, feature = load_dataset_arrays(
        dataset_root=dataset_root,
        dataset_type=args.dataset_type,
    )

    if not bool(getattr(args, "feature", True)):
        feature = np.random.rand(feature.shape[0], feature.shape[1]).astype(np.float32)

    train_indices, test_indices = split_indices(len(name_list), args.train_rate, args.seed)

    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = project_root / save_dir
    save_dir = (save_dir / args.run_name).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    split_payload = {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_names": [name_list[i] for i in train_indices],
        "test_names": [name_list[i] for i in test_indices],
        "dataset_type": args.dataset_type,
        "dataset_root": str(dataset_root),
        "num_samples": len(name_list),
    }
    with open(save_dir / "split_indices.json", "w") as f:
        json.dump(split_payload, f, indent=2)

    variant = resolve_variant(args)
    print(f"[regression] variant={variant}")

    if variant == "tabular":
        run_tabular_variant(args, variant, label, feature, save_dir, split_payload)
    else:
        run_gnn_variant(args, variant, dataset_root, name_list, label, feature, save_dir, split_payload)


if __name__ == "__main__":
    os.makedirs("log", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)
    main()
