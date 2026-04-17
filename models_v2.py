"""
models.py  –  GIRL Project
===========================
Compliant with GIRL Experimental Guideline and GIRL paper (Section 2).

Paper compliance (Geometric Invariant Risk Learning):
  • GNN models trained with mini-batch NeighborLoader (batch_size=256)
  • GeoReg penalty operates on batch induced subgraph representations Z
    per paper Section 2.7: L_geo = (1/2) Σ_{(i,j)∈E_B} a_ij ||z_i - z_j||²
  • IRM penalty computed per-batch by splitting batch seed nodes by environment
  • Val/test evaluation uses the full graph (standard transductive setup)
  • GNN-GeoReg baseline applies smoothness to batch output probs (distinct from GIRL)

Architecture decisions fixed by the guideline:
  • GNN baselines: SAGEConv-1layer encoder + linear head, 1-hop NeighborLoader
  • GIRL: SAGEConv-2layer encoder + linear head, 2-hop NeighborLoader

Calibration (Step 6):
  • Temperature scaling is a SEPARATE post-hoc step exposed via
    temperature_scale() — it is NOT applied inside any run_* runner.
  • All run_* functions report raw model probabilities.

Step 4 requirements:
  • 5 seeds, mean ± std reported
  • All hyperparameters logged to JSON
  • Hyperparameter tuning on validation set ONLY

Reproducibility:
  • set_seed() called at the start of every run_* call
  • All random sources fixed (numpy, torch, cuda, cudnn)
  • Checkpoints saved: .pt for torch models, .pkl for sklearn/XGB
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import json
import random
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, balanced_accuracy_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    xgb = None
    _XGB_AVAILABLE = False

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch helper
# ─────────────────────────────────────────────────────────────────────────────

def _make_neighbor_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    edge_index: torch.Tensor,
    train_idx: np.ndarray,
    num_neighbors: list[int],
    batch_size: int = 256,
    seed: int = 0,
) -> "NeighborLoader":
    """
    Build a NeighborLoader for mini-batch GNN training.

    num_neighbors: list of neighbours per hop, one entry per GNN layer.
                   e.g. [20] for 1-layer GNN, [20, 20] for GIRL (2-layer encoder).
    Only train nodes are used as seed nodes; val/test are evaluated with the
    full graph forward pass (standard transductive setup).
    """
    from torch_geometric.data import Data
    data = Data(x=x, y=y, edge_index=edge_index, num_nodes=x.shape[0])
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=torch.tensor(train_idx, dtype=torch.long),
        shuffle=True,
        num_workers=0,
    )
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random sources for full reproducibility (seeds 0–4)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# JSON logging  (Step 4: "Log hyperparameters and results in JSON format")
# ─────────────────────────────────────────────────────────────────────────────

def _save_json(data: dict, protocol: str, model_name: str, seed: int) -> str:
    """Save log to results/<protocol>/<model_name>/seed_<seed>.json"""
    out_dir = os.path.join("results", protocol, model_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def _save_checkpoint(
    state_dict: dict,
    protocol: str,
    model_name: str,
    seed: int,
) -> str:
    """
    Save model weights to results/<protocol>/<model_name>/seed_<seed>.pt
    Called automatically by every run_* function after training completes.
    """
    out_dir = os.path.join("results", protocol, model_name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"seed_{seed}.pt")
    torch.save(state_dict, path)
    return path


def load_checkpoint(
    protocol: str,
    model_name: str,
    seed: int,
) -> dict | None:
    """
    Load saved model weights from results/<protocol>/<model_name>/seed_<seed>.pt
    Returns the state_dict or None if the file does not exist.

    Usage example:
        from models import load_checkpoint, GNN
        state = load_checkpoint("temporal_ood", "gnn_erm", seed=0)
        if state is not None:
            model = GNN(in_dim, 64, n_classes)
            model.load_state_dict(state)
    """
    path = os.path.join("results", protocol, model_name, f"seed_{seed}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    return None


def checkpoint_exists(protocol: str, model_name: str, seed: int) -> bool:
    """Return True if a checkpoint has already been saved for this run."""
    pt_path  = os.path.join("results", protocol, model_name, f"seed_{seed}.pt")
    pkl_path = os.path.join("results", protocol, model_name, f"seed_{seed}.pkl")
    return os.path.exists(pt_path) or os.path.exists(pkl_path)


def load_sklearn_checkpoint(protocol: str, model_name: str, seed: int):
    """
    Load a saved sklearn model from results/<protocol>/<model_name>/seed_<seed>.pkl
    Returns the model or None if not found.
    """
    import pickle as _pkl
    path = os.path.join("results", protocol, model_name, f"seed_{seed}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return _pkl.load(f)
    return None


def _make_log(model_name, protocol, seed, hyperparameters, metrics, extra=None):
    log = {
        "model":           model_name,
        "protocol":        protocol,
        "seed":            seed,
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "hyperparameters": hyperparameters,
        "test_metrics":    metrics,
    }
    if extra:
        log.update(extra)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# Data / split / graph loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    path: str = "processed_dataset.csv",
    label_col: str = "label",
    drop_cols: list | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load feature matrix and labels from the processed dataset."""
    df = pd.read_csv(path)
    to_drop = [label_col] + (drop_cols or [])
    to_drop = [c for c in to_drop if c in df.columns]
    X_all = df.drop(columns=to_drop).select_dtypes(include=[np.number])
    y_all = df[label_col].values.astype(int)
    return X_all, y_all


def load_graph(path: str = "graph.pkl") -> dict:
    """Load kNN graph (k=20, Gaussian weights) saved by data_splits.py."""
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_tensors(graph: dict, device: torch.device):
    """Return (edge_index_t, edge_weight_t) for graph runners."""
    ei = torch.tensor(graph["edge_index"], dtype=torch.long).to(device)
    ew = torch.tensor(graph["edge_weight"], dtype=torch.float32).to(device)
    return ei, ew


def load_temporal_split(path: str = "splits/temporal_split.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_policy_split(path: str = "splits/policy_split.json") -> dict:
    with open(path) as f:
        return json.load(f)


def load_city_splits(seed: int, splits_dir: str = "splits") -> list[dict]:
    pattern = os.path.join(splits_dir, f"city_seed{seed}_split*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No city splits found for '{pattern}'. Run data_splits.py first."
        )
    # FIX-9: use context managers so file descriptors are closed immediately.
    # The previous list comprehension left handles open for the process lifetime.
    splits = []
    for fp in files:
        with open(fp) as f:
            splits.append(json.load(f))
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# Metrics  (Section 4 of guideline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """ECE for binary classification."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() > 0:
            ece += np.abs(
                np.mean(y_true[mask] == (y_prob[mask] > 0.5)) - np.mean(y_prob[mask])
            ) * mask.sum() / len(y_true)
    return float(ece)


def compute_ece_multiclass(
    y_true: np.ndarray, conf: np.ndarray, preds: np.ndarray, n_bins: int = 15,
) -> float:
    """ECE for multi-class using max-confidence."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(conf)) * abs(
                (preds[mask] == y_true[mask]).mean() - conf[mask].mean()
            )
    return float(ece)


def compute_reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_accs, bin_confs, bin_counts) for reliability diagrams (Protocol C)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(float(np.mean(y_true[mask] == (y_prob[mask] > 0.5))))
            bin_confs.append(float(np.mean(y_prob[mask])))
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(float((bins[i] + bins[i + 1]) / 2))
            bin_counts.append(0)
    return np.array(bin_accs), np.array(bin_confs), np.array(bin_counts)


def compute_metrics_binary(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10,
) -> dict:
    """
    Full metric set for binary classifiers (LR, MLP, XGB, IRM, VREx, GroupDRO).
    Keys: acc, macro_f1, bal_acc, nll, ece, brier
    Probabilities are clamped to [1e-7, 1-1e-7] before log operations.
    """
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    preds  = (y_prob > 0.5).astype(int)
    probs2 = np.stack([1 - y_prob, y_prob], axis=1)
    return {
        "acc":      float(accuracy_score(y_true, preds)),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "bal_acc":  float(balanced_accuracy_score(y_true, preds)),
        "nll":      float(log_loss(y_true, probs2)),
        "ece":      compute_ece(y_true, y_prob, n_bins),
        "brier":    float(np.mean((y_prob - y_true) ** 2)),
    }


def compute_metrics_gnn(
    logits: torch.Tensor, y_true: torch.Tensor, n_bins: int = 10,
) -> dict:
    """
    Full metric set for GNN outputs (SAGEConv, GIRL).
    For binary tasks (n_classes=2): extracts probs[:,1] for NLL/ECE to match
    the binary metric path and avoid softmax collapse inflating NLL.
    Keys: acc, macro_f1, bal_acc, nll, ece, brier
    """
    probs        = torch.softmax(logits, dim=1)
    conf, preds  = probs.max(dim=1)

    y_np     = y_true.cpu().numpy()
    preds_np = preds.cpu().numpy()
    conf_np  = conf.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    # Clamp + sanitise + renormalise before any log operation
    probs_np = np.nan_to_num(probs_np, nan=1.0 / probs_np.shape[1],
                              posinf=1.0, neginf=0.0)
    probs_np = np.clip(probs_np, 1e-7, 1.0 - 1e-7)
    probs_np = probs_np / probs_np.sum(axis=1, keepdims=True)

    n_classes = probs_np.shape[1]

    if n_classes == 2:
        p1  = probs_np[:, 1]
        nll = float(log_loss(y_np, p1))
        ece = compute_ece(y_np, p1, n_bins)
    else:
        nll = float(log_loss(y_np, probs_np))
        ece = compute_ece_multiclass(y_np, conf_np, preds_np, n_bins)

    brier = float(
        sum(np.mean((probs_np[:, c] - (y_np == c).astype(float)) ** 2)
            for c in range(n_classes)) / n_classes
    )

    return {
        "acc":      float(accuracy_score(y_np, preds_np)),
        "macro_f1": float(f1_score(y_np, preds_np, average="macro")),
        "bal_acc":  float(balanced_accuracy_score(y_np, preds_np)),
        "nll":      float(nll),
        "ece":      float(ece),
        "brier":    brier,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scaling  (Step 6 — calibration analysis, SEPARATE from runners)
# ─────────────────────────────────────────────────────────────────────────────

def temperature_scale(
    val_logits: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    n_iter: int = 50,
) -> float:
    """
    Post-hoc temperature scaling (Guo et al., 2017).  Step 6 of guideline.

    This is intentionally NOT called inside any run_* function.
    Call it explicitly in the calibration analysis section of each notebook
    after collecting raw model outputs.

    Parameters
    ----------
    val_logits : [N_val, C] logits on the validation set (already sliced)
    y_val      : [N_val]    integer labels for the validation set
    device     : torch device

    Returns
    -------
    T : float  — divide logits by T before softmax to get calibrated probs
    """
    val_logits = val_logits.detach()
    T = nn.Parameter(torch.ones(1, device=device) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=n_iter)

    def eval_fn():
        opt.zero_grad()
        loss = F.cross_entropy(val_logits / T.clamp(min=0.01), y_val)
        loss.backward()
        return loss

    opt.step(eval_fn)
    return float(T.clamp(min=0.01).item())


# ─────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(vals: list[float]) -> str:
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def summarise(results: dict, keys: list[str] | None = None) -> pd.DataFrame:
    keys = keys or [
        k for k in results
        if isinstance(results[k], list) and len(results[k]) > 0
        and isinstance(results[k][0], float)
    ]
    return pd.DataFrame([{k: fmt(results[k]) for k in keys}])


def aggregate_city_metrics(env_metrics: list[dict]) -> dict:
    """Aggregate per-city metrics into MeanEnv / WorstEnv (Protocol A)."""
    if not env_metrics:
        raise ValueError(
            "env_metrics is empty. Check that load_city_splits() returned "
            "results and each run_* call completed successfully."
        )
    accs     = [m["acc"]      for m in env_metrics]
    f1s      = [m["macro_f1"] for m in env_metrics]
    bal_accs = [m["bal_acc"]  for m in env_metrics]
    return {
        "mean_acc":      float(np.mean(accs)),
        "worst_acc":     float(np.min(accs)),
        "mean_f1":       float(np.mean(f1s)),
        "worst_f1":      float(np.min(f1s)),
        "mean_bal_acc":  float(np.mean(bal_accs)),
        "worst_bal_acc": float(np.min(bal_accs)),
    }


def fmt_city(results: dict) -> pd.DataFrame:
    keys = ["mean_acc", "worst_acc", "mean_f1", "worst_f1",
            "mean_bal_acc", "worst_bal_acc"]
    return pd.DataFrame([{k: fmt(results[k]) for k in keys if k in results}])


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions  — architectures fixed by the guideline
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """MLP (ERM) — non-graph baseline. Binary classification (single logit)."""
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class IRM_MLP(nn.Module):
    """MLP with 2-class output for IRM / VREx / GroupDRO invariant baselines."""
    def __init__(self, input_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)

VREx_MLP = IRM_MLP
GDRO_MLP = IRM_MLP


class GNN(nn.Module):
    """
    GNN with GraphSAGE encoder + linear prediction head — graph baseline.

    Architecture: 1-layer SAGEConv encoder + linear head.
      encode: SAGEConv(in, hidden) → ReLU → z
      head:   Linear(hidden, out)

    SAGEConv concatenates [h_i || MEAN(h_j)] before projection, preserving
    the node's own features regardless of neighbourhood distribution. This
    reduces the aggregation-smoothing problem under cross-city OOD shift,
    where minority-class nodes are surrounded by majority-class neighbours.

    Edge weights are NOT used in SAGEConv aggregation (SAGEConv does not
    support them natively). Gaussian weights a_ij are used exclusively in
    the GeoReg penalty — a clean separation between representation learning
    (graph structure) and geometric regularisation (proximity weighting).

    Linear head matches paper spec (Section 2.3): hw(z) = softmax(Wz + b).
    hidden_dim=64, dropout=0.5.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, out_dim)

    def encode(self, x, edge_index, edge_weight=None):
        """Return hidden representations z. edge_weight unused — SAGEConv is unweighted."""
        return F.relu(self.conv1(x, edge_index))

    def forward(self, x, edge_index, edge_weight=None):
        x = self.encode(x, edge_index, edge_weight)
        x = F.dropout(x, p=0.5, training=self.training)
        return self.head(x)


class GNN_GeoReg(GNN):
    """GNN baseline with geographic regularisation — same backbone as GNN."""
    pass


class GNN_IRM(GNN):
    """GNN baseline with IRM penalty — same backbone as GNN."""
    pass


class GIRL(nn.Module):
    """
    Geometric Invariant Risk Learning — proposed method.

    Architecture: 2-layer SAGEConv encoder + linear head.
      encode: SAGEConv(in, hidden) → ReLU → Dropout → SAGEConv(hidden, hidden) → ReLU → z
      head:   Linear(hidden, out)

    SAGEConv concatenates [h_i || MEAN(h_j)] before projection, preserving
    the node's own features across both encoding layers. This directly
    addresses the aggregation-smoothing problem under cross-city OOD shift.

    Edge weights are NOT used in SAGEConv aggregation. Gaussian weights a_ij
    are passed through to georeg_penalty() only, where they weight the
    Laplacian smoothness term: Σ a_ij ||z_i - z_j||². This separates
    representation learning (topology) from geometric regularisation (proximity).

    Linear head matches paper spec (Section 2.3): hw(z) = softmax(Wz + b).
    Training objective: ERM + λ·IRM_penalty + μ·GeoReg_penalty

    Note: λ (lam) and μ (mu) are loss hyperparameters, NOT model parameters.
    They are NOT stored on this module — pass them explicitly to girl_loss()
    so that the same trained model can be evaluated under different penalty
    strengths without reinstantiation, and so that state_dict() contains
    only learnable weights.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.head  = nn.Linear(hidden_dim, out_dim)

    def encode(self, x, edge_index, edge_weight=None):
        """Return hidden representations z. edge_weight unused in SAGEConv aggregation."""
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.5, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        return h

    def forward(self, x, edge_index, edge_weight=None):
        h = self.encode(x, edge_index, edge_weight)
        h = F.dropout(h, p=0.5, training=self.training)
        return self.head(h)


# ─────────────────────────────────────────────────────────────────────────────
# Loss / penalty helpers
# ─────────────────────────────────────────────────────────────────────────────

def irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gradient-norm IRM penalty (Arjovsky et al., 2019): ||∇_{w=1} L_e(f·w)||²"""
    scale = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss  = F.cross_entropy(logits * scale, y)
    grad  = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def georeg_penalty(
    Z: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Geographic smoothness penalty: tr(Z^T L Z)  (Section 2.5 of GIRL paper).

    Penalises high-frequency variation of the hidden REPRESENTATIONS Z
    on the spatial graph — not the output logits or probabilities.

    Z          : [N, d] hidden node representations from model.encode()
    edge_index : [2, E] graph edges
    edge_weight: [E]    Gaussian edge weights a_ij

    Expanded form: (1/2) Σ_{(i,j)∈E} a_ij ||z_i - z_j||²
    which equals tr(Z^T L Z) for the unnormalised Laplacian L = D - A.

    NOTE: Uses 0.5 * .sum() to faithfully implement the paper's formulation
    Σ_{(i,j)∈E} a_ij ||z_i - z_j||² = tr(Z^T L Z).
    Because the sum scales with |E_train|, μ grids must be chosen per-protocol
    to account for differences in training set size.  The same formulation is
    applied consistently in run_gnn_georeg so λ/μ scales are comparable.
    """
    src, dst = edge_index[0], edge_index[1]
    diff     = Z[src] - Z[dst]                   # [E, d]
    sq       = (diff ** 2).sum(dim=-1)            # [E]
    if edge_weight is not None:
        sq = sq * edge_weight
    return 0.5 * sq.sum()


def girl_loss(
    model: GIRL,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    y: torch.Tensor,
    node_envs: list[np.ndarray],
    train_idx: np.ndarray,
    device: torch.device,
    lam: float = 1.0,
    mu: float = 1.0,
    ei_train: torch.Tensor | None = None,
    ew_train: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    GIRL objective: ERM + λ·IRM_penalty + μ·GeoReg_penalty

    Matches paper Section 2.6:
      L_GIRL = Σ_e R_e(θ,w)  +  λ Σ_e ||∇_w R_e||²  +  μ tr(Z^T L Z)

    GeoReg is applied to hidden representations Z = encode(x, G),
    NOT to output logits or probabilities — per Section 2.5.

    GeoReg is computed on the induced subgraph of training nodes only
    (paper Section 2.7: "geometric term is computed on the induced subgraph
    of batch nodes"). Computing it over all edges would allow val/test node
    representations to influence the geometry penalty during training.

    lam, mu are passed explicitly (not stored on model) so that the same
    trained model can be evaluated without reinstantiation.

    ei_train / ew_train: precomputed train-node induced subgraph edges.
    If not provided they are computed here (slower — only for one-off calls).
    In run_girl they are precomputed once before the epoch loop for efficiency.
    """
    # Get hidden representations Z and final logits separately
    Z      = model.encode(x, edge_index, edge_weight)   # [N, hidden]
    logits = model.head(Z)                               # [N, n_classes] — linear head, no aggregation

    tidx   = torch.tensor(train_idx, dtype=torch.long, device=device)
    erm    = F.cross_entropy(logits[tidx], y[tidx])

    # IRM penalty on logits per environment (IRM-v1 scalar surrogate)
    irm_pen = torch.zeros(1, device=device)
    for env_nodes in node_envs:
        eidx    = torch.tensor(env_nodes, dtype=torch.long, device=device)
        irm_pen = irm_pen + irm_penalty(logits[eidx], y[eidx])
    irm_pen = irm_pen / len(node_envs)

    # GeoReg penalty on representations Z — restricted to training-node
    # induced subgraph to avoid val/test representation leakage.
    # Use precomputed ei_train/ew_train if provided (fast path),
    # otherwise compute the mask here (slow path, for one-off calls only).
    if ei_train is None:
        train_set     = set(train_idx.tolist())
        src_cpu       = edge_index[0].cpu().numpy()
        dst_cpu       = edge_index[1].cpu().numpy()
        train_mask_np = np.array(
            [s in train_set and d in train_set
             for s, d in zip(src_cpu, dst_cpu)],
            dtype=bool,
        )
        train_mask = torch.tensor(train_mask_np, dtype=torch.bool, device=device)
        ei_train   = edge_index[:, train_mask]
        ew_train   = edge_weight[train_mask] if edge_weight is not None else None

    geo_pen = georeg_penalty(Z, ei_train, ew_train)

    return erm + lam * irm_pen + mu * geo_pen


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = [0, 1, 2, 3, 4]


def _ensure_keys(results: dict) -> dict:
    """Ensure all metric keys exist regardless of how results was initialised."""
    for k in ("acc", "macro_f1", "bal_acc", "nll", "ece", "brier"):
        results.setdefault(k, [])
    results.setdefault("probs", [])   # test probabilities for reliability diagrams
    return results


def _standardise(X_train, X_val, X_test):
    """Fit StandardScaler on train only, apply to val and test."""
    def _arr(x): return x.values if hasattr(x, "values") else np.array(x)
    scaler = StandardScaler().fit(_arr(X_train))
    return scaler.transform(_arr(X_train)), \
           scaler.transform(_arr(X_val)), \
           scaler.transform(_arr(X_test))


def _scale_graph_features(
    x,
    train_idx: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Scale graph node features for GNN runners.

    Accepts either:
      - a raw numpy array or pandas DataFrame (unscaled CSV features)
      - an already-scaled float32 torch.Tensor (legacy path — returned as-is)

    When raw features are passed (numpy / DataFrame), a StandardScaler is
    fit on train_idx rows only and applied to the full matrix.  This ensures
    no val/test statistics leak into the scaler, consistent with the tabular
    runner contract in _standardise().

    IMPORTANT: The returned tensor is kept on CPU intentionally.
    NeighborLoader requires CPU tensors when building the Data object —
    it handles device transfer itself via batch.to(device) inside the
    training loop.  Moving to GPU here caused 'numpy.int64 not callable'
    errors inside GCNConv because PyG's batching lost tensor typing.
    Full-graph val/test forward passes move x to device explicitly in
    each runner when needed.
    """
    if isinstance(x, torch.Tensor):
        # Already a tensor — return on CPU so NeighborLoader can batch it.
        return x.cpu()

    X_np = x.values if hasattr(x, "values") else np.array(x)
    scaler   = StandardScaler().fit(X_np[train_idx])
    X_scaled = scaler.transform(X_np)
    # CPU tensor — device transfer happens inside each runner via batch.to(device)
    return torch.tensor(X_scaled, dtype=torch.float32)


def _gnn_init(in_dim: int, n_classes: int, device: torch.device,
              hidden: int = 64):
    """Initialise GNN (SAGEConv backbone) + Adam optimiser."""
    model = GNN(in_dim, hidden, n_classes).to(device)
    opt   = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    return model, opt


def _make_envs_tabular(X_all, y_all, train_idx: np.ndarray,
                       X_scaled: np.ndarray, device, n_envs: int = 3):
    """Split scaled training data into n_envs temporal chunks."""
    size = len(train_idx) // n_envs
    envs = []
    for i in range(n_envs):
        s   = i * size
        e   = (i + 1) * size if i < n_envs - 1 else len(train_idx)
        xi  = torch.tensor(X_scaled[s:e], dtype=torch.float32).to(device)
        yi  = torch.tensor(y_all[train_idx[s:e]], dtype=torch.long).to(device)
        envs.append((xi, yi))
    return envs


def _make_envs_graph(train_idx: np.ndarray, n_envs: int = 3):
    """Split training node indices into n_envs temporal chunks."""
    size = len(train_idx) // n_envs
    return [
        train_idx[i*size : (i+1)*size if i < n_envs-1 else len(train_idx)]
        for i in range(n_envs)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Non-graph baseline runners  (Step 4)
# ─────────────────────────────────────────────────────────────────────────────

def run_lr(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed: int,
    C_grid: list[float] | None = None,
    results: dict | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    Logistic Regression baseline.  Hyperparameter: C (tuned on val Macro-F1).
    Features are standardised (fit on train only) — LR is sensitive to feature
    scale and produces degenerate results on raw features.
    Saves to results/<protocol>/logistic_regression/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if C_grid is None:
        C_grid = [0.01, 0.1, 1.0, 10.0]

    set_seed(seed)
    best_f1, best_model, best_C = -1, None, None

    # Standardise — fit on train only, apply to val and test
    Xtr, Xv, Xte = _standardise(X_train, X_val, X_test)
    ytr = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
    yv  = y_val   if isinstance(y_val,   np.ndarray) else np.array(y_val)
    yte = y_test  if isinstance(y_test,  np.ndarray) else np.array(y_test)

    for C in C_grid:
        m = LogisticRegression(C=C, max_iter=1000, class_weight="balanced",
                               random_state=seed)
        m.fit(Xtr, ytr)
        vf = f1_score(yv, m.predict(Xv), average="macro")
        if vf > best_f1:
            best_f1, best_model, best_C = vf, m, C

    probs = best_model.predict_proba(Xte)[:, 1]
    met   = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    # Save sklearn model via pickle (not torch state_dict)
    import pickle as _pkl
    _ckpt_dir = os.path.join("results", protocol, "logistic_regression")
    os.makedirs(_ckpt_dir, exist_ok=True)
    with open(os.path.join(_ckpt_dir, f"seed_{seed}.pkl"), "wb") as _f:
        _pkl.dump(best_model, _f)

    _save_json(_make_log("LogisticRegression", protocol, seed,
                         {"C_grid": C_grid, "best_C": best_C,
                          "max_iter": 1000, "class_weight": "balanced",
                          "standardised": True},
                         met, extra={"test_probs": probs.tolist()}),
               protocol, "logistic_regression", seed)
    return results


def run_mlp(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed: int,
    lr_grid: list[float] | None = None,
    epochs: int = 100,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    MLP (ERM) baseline.  Hyperparameter: lr (tuned on val Macro-F1).
    Features are standardised (fit on train only).
    pos_weight applied for class imbalance.
    Saves to results/<protocol>/mlp/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lr_grid is None:
        lr_grid = [1e-4, 1e-3, 1e-2]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    Xtr_np, Xv_np, Xte_np = _standardise(X_train, X_val, X_test)
    ytr = y_train if isinstance(y_train, np.ndarray) else np.array(y_train)
    yv  = y_val   if isinstance(y_val,   np.ndarray) else np.array(y_val)
    yte = y_test  if isinstance(y_test,  np.ndarray) else np.array(y_test)

    def _t(a, dtype=torch.float32):
        return torch.tensor(a, dtype=dtype).to(device)

    Xtr_t = _t(Xtr_np); ytr_t = _t(ytr).unsqueeze(1)
    Xv_t  = _t(Xv_np);  Xte_t = _t(Xte_np)

    pos_w     = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], dtype=torch.float32).to(device)
    )

    best_f1, best_model, best_lr = -1, None, None

    for lr in lr_grid:
        set_seed(seed)
        model = MLP(Xtr_t.shape[1]).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            criterion(model(Xtr_t), ytr_t).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(Xv_t)).cpu().numpy().flatten()
        vf = f1_score(yv, (vp > 0.5).astype(int), average="macro")
        if vf > best_f1:
            best_f1, best_model, best_lr = vf, model, lr

    best_model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(best_model(Xte_t)).cpu().numpy().flatten()

    met = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    _save_checkpoint(best_model.state_dict(), protocol, "mlp", seed)

    _save_json(_make_log("MLP", protocol, seed,
                         {"architecture": "128-ReLU-Drop(0.3)-64-ReLU-1",
                          "lr_grid": lr_grid, "best_lr": best_lr,
                          "epochs": epochs, "optimizer": "Adam",
                          "weight_decay": 1e-4, "pos_weight": round(pos_w, 4),
                          "standardised": True},
                         met, extra={"test_probs": probs.tolist()}), protocol, "mlp", seed)
    return results


def run_xgb(
    X_train, y_train, X_val, y_val, X_test, y_test,
    seed: int,
    param_grid: dict | None = None,
    results: dict | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    XGBoost baseline.  Hyperparameters: max_depth, lr, n_estimators
    (tuned on val Macro-F1).
    Saves to results/<protocol>/xgboost/seed_<n>.json
    """
    if not _XGB_AVAILABLE:
        raise ImportError("Run: pip install xgboost")
    results = _ensure_keys(results if results is not None else {})
    if param_grid is None:
        param_grid = {"max_depth": [4, 6, 8],
                      "learning_rate": [0.01, 0.05, 0.1],
                      "n_estimators": [200, 300]}

    set_seed(seed)
    def _a(x): return x.values if hasattr(x, "values") else np.array(x)
    Xtr, ytr = _a(X_train), np.array(y_train)
    Xv,  yv  = _a(X_val),   np.array(y_val)
    Xte, yte = _a(X_test),  np.array(y_test)

    pos = (ytr == 1).sum(); neg = (ytr == 0).sum()
    spw = float(neg / pos) if pos > 0 else 1.0

    best_f1, best_model, best_params = -1, None, {}
    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for ne in param_grid["n_estimators"]:
                m = xgb.XGBClassifier(
                    max_depth=md, learning_rate=lr, n_estimators=ne,
                    scale_pos_weight=spw, subsample=0.8,
                    colsample_bytree=0.8, eval_metric="logloss",
                    random_state=seed, verbosity=0,
                )
                m.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
                vf = f1_score(yv, m.predict(Xv), average="macro")
                if vf > best_f1:
                    best_f1, best_model = vf, m
                    best_params = {"max_depth": md, "learning_rate": lr,
                                   "n_estimators": ne}

    probs = best_model.predict_proba(Xte)[:, 1]
    met   = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    # Save XGBoost model via pickle
    import pickle as _pkl
    _ckpt_dir = os.path.join("results", protocol, "xgboost")
    os.makedirs(_ckpt_dir, exist_ok=True)
    with open(os.path.join(_ckpt_dir, f"seed_{seed}.pkl"), "wb") as _f:
        _pkl.dump(best_model, _f)

    _save_json(_make_log("XGBoost", protocol, seed,
                         {"param_grid": param_grid, "best_params": best_params,
                          "scale_pos_weight": round(spw, 4),
                          "subsample": 0.8, "colsample_bytree": 0.8},
                         met, extra={"test_probs": probs.tolist()}), protocol, "xgboost", seed)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Graph baseline runners  (Step 4 — SAGEConv backbone)
# ─────────────────────────────────────────────────────────────────────────────

def run_gnn_erm(
    x, y, edge_index,
    train_idx, val_idx, test_idx,
    seed: int,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int | None = None,
    lr_grid: list[float] | None = None,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    GNN-ERM baseline (SAGEConv-1layer encoder + linear head).
    Mini-batch training via NeighborLoader (2-hop, batch_size=256).
    Val/test evaluation uses the full graph (standard transductive setup).
    Hyperparameter: lr (tuned on val Macro-F1).
    Early stopping: disabled by default (patience=None).
    Saves to results/<protocol>/gnn_erm/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lr_grid is None:
        lr_grid = [1e-3, 5e-3, 1e-2]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scale to CPU tensor (NeighborLoader requires CPU); move to device for full-graph passes.
    x_cpu = _scale_graph_features(x, np.array(train_idx), device)  # CPU tensor
    x     = x_cpu.to(device)                                        # GPU tensor for val/test
    y = y.to(device) if isinstance(y, torch.Tensor) else \
        torch.tensor(y, dtype=torch.long).to(device)
    edge_index = edge_index.to(device) if isinstance(edge_index, torch.Tensor) else \
        torch.tensor(edge_index, dtype=torch.long).to(device)

    n_classes = int(y.max().item()) + 1
    vidx    = torch.tensor(val_idx,   dtype=torch.long, device=device)
    testidx = torch.tensor(test_idx,  dtype=torch.long, device=device)

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    _ytr     = y[torch.tensor(train_idx, dtype=torch.long, device=device)].cpu().numpy()
    _n_neg   = float((_ytr == 0).sum())
    _n_pos   = float((_ytr == 1).sum())
    # Square-root damping: full inverse-frequency (neg/pos ~3.8x on policy shift)
    # over-penalises negatives and collapses accuracy. sqrt gives a balanced
    # compromise that improves minority-class recall without overcorrecting.
    _pos_w   = (_n_neg / max(_n_pos, 1.0)) ** 0.5
    cw_gnn   = torch.tensor([1.0, _pos_w], dtype=torch.float32).to(device)

    best_val_f1, best_state, best_lr = -1, None, None

    for lr in lr_grid:
        set_seed(seed)
        model = GNN(x.shape[1], 64, n_classes).to(device)
        opt   = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        # NeighborLoader: 1 hop matching 1-layer GCN encoder (linear head
        # requires no additional hop — reduces batch size significantly).
        loader = _make_neighbor_loader(
            x_cpu, y.cpu(), edge_index.cpu(), train_idx,
            num_neighbors=[20], batch_size=batch_size, seed=seed)

        best_inner_loss, best_inner_state = float("inf"), None
        no_improve = 0

        for _ in range(epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                # batch.n_id maps batch nodes back to global ids
                # out[:batch.batch_size] are the seed node logits
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size], weight=cw_gnn)
                loss.backward()
                opt.step()

            # Full-graph val evaluation
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(x, edge_index)[vidx], y[vidx]).item()
            if vl < best_inner_loss:
                best_inner_loss  = vl
                best_inner_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                if patience is not None:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        model.load_state_dict(best_inner_state); model.eval()
        with torch.no_grad():
            vf1 = f1_score(y[vidx].cpu().numpy(),
                           model(x, edge_index)[vidx].argmax(1).cpu().numpy(),
                           average="macro")
        if vf1 > best_val_f1:
            best_val_f1, best_state, best_lr = vf1, best_inner_state, lr

    model = GNN(x.shape[1], 64, n_classes).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits_test = model(x, edge_index)[testidx]
        met  = compute_metrics_gnn(logits_test, y[testidx])
        p1   = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

    for k in met:
        results[k].append(met[k])
    results["probs"].append(p1.tolist())
    _save_checkpoint(best_state, protocol, "gnn_erm", seed)

    _save_json(_make_log("GNN_ERM", protocol, seed,
                         {"architecture": "SAGEConv-1layer+LinearHead", "hidden_dim": 64,
                          "dropout": 0.5, "optimizer": "Adam",
                          "lr_grid": lr_grid, "best_lr": best_lr,
                          "weight_decay": 5e-4, "epochs": epochs,
                          **({} if patience is None else {"patience": patience}),
                          "batch_size": batch_size, "num_neighbors": [20],
                          "pos_weight": round(float(_pos_w), 4), "weighting": "sqrt_inv_freq"},
                         met, extra={"test_probs": p1.tolist()}), protocol, "gnn_erm", seed)
    return results


def run_gnn_georeg(
    x, y, edge_index, edge_weight,
    train_idx, val_idx, test_idx,
    seed: int,
    lambda_grid: list[float] | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int | None = None,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    GNN + GeoReg baseline (SAGEConv-1layer encoder + linear head + geographic smoothness penalty).
    Mini-batch training via NeighborLoader (1-hop, batch_size=256).
    GeoReg penalty is applied to the batch induced subgraph (per paper Sec 2.7).
    Val/test evaluation uses the full graph.
    Hyperparameter: λ (tuned on val Macro-F1).
    Early stopping: disabled by default (patience=None).
    Saves to results/<protocol>/gnn_georeg/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lambda_grid is None:
        lambda_grid = [0.01, 0.1, 1.0, 5.0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_cpu = _scale_graph_features(x, np.array(train_idx), device)
    x     = x_cpu.to(device)
    y = y.to(device) if isinstance(y, torch.Tensor) else \
        torch.tensor(y, dtype=torch.long).to(device)
    edge_index  = edge_index.to(device) if isinstance(edge_index, torch.Tensor) else \
        torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_weight = edge_weight.to(device) if isinstance(edge_weight, torch.Tensor) else \
        torch.tensor(edge_weight, dtype=torch.float32).to(device)

    n_classes = int(y.max().item()) + 1
    vidx    = torch.tensor(val_idx,   dtype=torch.long, device=device)
    testidx = torch.tensor(test_idx,  dtype=torch.long, device=device)

    # ── Geo penalty normalisation ─────────────────────────────────────────
    # geo_loss scales with n_edges * ||prob_diff||^2 — orders of magnitude
    # larger than ERM at init. Normalise so lambda is a fraction of ERM.
    _tmp_model, _ = _gnn_init(x.shape[1], n_classes, device)
    _tmp_model.eval()
    with torch.no_grad():
        _log_tmp = _tmp_model(x_cpu.to(device), edge_index, edge_weight)
        _pr_tmp  = torch.softmax(_log_tmp, dim=-1)
        _ew      = edge_weight if edge_weight is not None else \
                   torch.ones(edge_index.shape[1], device=device)
        _diff_tmp = _pr_tmp[edge_index[0]] - _pr_tmp[edge_index[1]]
        _geo_init = float(0.5 * (_diff_tmp ** 2).sum(dim=-1).mul(_ew).sum().item())
    scale_geo = max(_geo_init, 1e-8)
    del _tmp_model, _log_tmp, _pr_tmp, _diff_tmp, _ew
    # ─────────────────────────────────────────────────────────────────────

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    _ytr     = y[torch.tensor(train_idx, dtype=torch.long, device=device)].cpu().numpy()
    _n_neg   = float((_ytr == 0).sum())
    _n_pos   = float((_ytr == 1).sum())
    # Square-root damping: full inverse-frequency (neg/pos ~3.8x on policy shift)
    # over-penalises negatives and collapses accuracy. sqrt gives a balanced
    # compromise that improves minority-class recall without overcorrecting.
    _pos_w   = (_n_neg / max(_n_pos, 1.0)) ** 0.5
    cw_gnn   = torch.tensor([1.0, _pos_w], dtype=torch.float32).to(device)

    best_val_f1, best_state, best_lam = -1, None, None

    for lam in lambda_grid:
        set_seed(seed)
        model, opt = _gnn_init(x.shape[1], n_classes, device)

        loader = _make_neighbor_loader(
            x_cpu, y.cpu(), edge_index.cpu(), train_idx,
            num_neighbors=[20], batch_size=batch_size, seed=seed)

        best_inner_loss, best_inner_state = float("inf"), None
        no_improve = 0

        for _ in range(epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                logits   = model(batch.x, batch.edge_index, batch.edge_attr)
                erm_loss = F.cross_entropy(logits[:batch.batch_size],
                                           batch.y[:batch.batch_size], weight=cw_gnn)
                # GeoReg on batch output probabilities (all batch nodes)
                _probs = torch.softmax(logits, dim=-1)
                ei_b   = batch.edge_index
                ew_b   = batch.edge_attr
                _diff  = _probs[ei_b[0]] - _probs[ei_b[1]]
                _sq    = (_diff ** 2).sum(dim=-1)
                if ew_b is not None:
                    _sq = _sq * ew_b
                geo_loss = 0.5 * _sq.sum()
                (erm_loss + lam * (geo_loss / scale_geo)).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(
                    model(x, edge_index, edge_weight)[vidx], y[vidx]
                ).item()
            if vl < best_inner_loss:
                best_inner_loss  = vl
                best_inner_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                if patience is not None:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        model.load_state_dict(best_inner_state); model.eval()
        with torch.no_grad():
            vf1 = f1_score(y[vidx].cpu().numpy(),
                           model(x, edge_index, edge_weight)[vidx].argmax(1).cpu().numpy(),
                           average="macro")
        if vf1 > best_val_f1:
            best_val_f1, best_state, best_lam = vf1, best_inner_state, lam

    model = GNN(x.shape[1], 64, n_classes).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits_test = model(x, edge_index, edge_weight)[testidx]
        met  = compute_metrics_gnn(logits_test, y[testidx])
        p1   = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

    for k in met:
        results.setdefault(k, []).append(met[k])
    results.setdefault("probs", []).append(p1.tolist())
    _save_checkpoint(best_state, protocol, "gnn_georeg", seed)

    _save_json(_make_log("GNN_GeoReg", protocol, seed,
                         {"architecture": "SAGEConv-1layer+LinearHead", "hidden_dim": 64,
                          "dropout": 0.5, "optimizer": "Adam", "lr": 0.01,
                          "weight_decay": 5e-4, "epochs": epochs,
                          **({} if patience is None else {"patience": patience}),
                          "batch_size": batch_size, "num_neighbors": [20],
                          "lambda_grid": lambda_grid, "best_lambda": best_lam,
                          "scale_geo": round(scale_geo, 2),
                          "pos_weight": round(float(_pos_w), 4), "weighting": "sqrt_inv_freq"},
                         met, extra={"test_probs": p1.tolist()}), protocol, "gnn_georeg", seed)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Invariant baseline runners  (Step 4 — tabular MLP backbone)
# ─────────────────────────────────────────────────────────────────────────────

def run_irm(
    X_all, y_all,
    train_idx, val_idx, test_idx,
    seed: int,
    lambda_grid: list[float] | None = None,
    epochs: int = 100,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    IRM baseline (tabular MLP + gradient-norm invariance penalty).
    Environments: 3 temporal thirds of train_idx.
    Hyperparameter: λ (tuned on val Macro-F1).
    Features standardised on train. pos_weight for class balance.
    Saves to results/<protocol>/irm/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lambda_grid is None:
        lambda_grid = [1e-3, 1e-1, 1.0, 10.0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.array(train_idx)
    Xtr_raw = X_all.iloc[train_idx].values if hasattr(X_all, "iloc") else X_all[train_idx]
    Xv_raw  = X_all.iloc[val_idx].values   if hasattr(X_all, "iloc") else X_all[val_idx]
    Xte_raw = X_all.iloc[test_idx].values  if hasattr(X_all, "iloc") else X_all[test_idx]

    scaler  = StandardScaler().fit(Xtr_raw)
    Xtr_np  = scaler.transform(Xtr_raw)
    Xv_np   = scaler.transform(Xv_raw)
    Xte_np  = scaler.transform(Xte_raw)

    ytr = y_all[train_idx]; yv = y_all[val_idx]; yte = y_all[test_idx]

    Xv_t  = torch.tensor(Xv_np,  dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte_np, dtype=torch.float32).to(device)
    in_dim = Xv_t.shape[1]

    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1)
    cw    = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)

    envs = _make_envs_tabular(X_all, y_all, train_idx, Xtr_np, device)

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    best_val_f1, best_state, best_lam = -1, None, None

    for lam in lambda_grid:
        set_seed(seed)
        model = IRM_MLP(in_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(epochs):
            model.train(); opt.zero_grad()
            erm = sum(F.cross_entropy(model(xi), yi, weight=cw)
                      for xi, yi in envs) / len(envs)
            pen = sum(irm_penalty(model(xi), yi)
                      for xi, yi in envs) / len(envs)
            (erm + lam * pen).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp  = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
        vf1 = f1_score(yv, (vp > 0.5).astype(int), average="macro")
        if vf1 > best_val_f1:
            best_val_f1 = vf1; best_lam = lam
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model = IRM_MLP(in_dim).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()

    met = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    _save_checkpoint({k: v.clone() for k, v in model.state_dict().items()}, protocol, "irm", seed)

    _save_json(_make_log("IRM", protocol, seed,
                         {"architecture": "MLP-128-Drop-64-2",
                          "optimizer": "Adam", "lr": 1e-3,
                          "weight_decay": 1e-4, "epochs": epochs,
                          "n_envs": len(envs), "pos_weight": round(pos_w, 4),
                          "lambda_grid": lambda_grid, "best_lambda": best_lam,
                          "standardised": True},
                         met, extra={"test_probs": probs.tolist()}), protocol, "irm", seed)
    return results


def run_vrex(
    X_all, y_all,
    train_idx, val_idx, test_idx,
    seed: int,
    lambda_grid: list[float] | None = None,
    epochs: int = 100,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    VREx baseline (tabular MLP + variance risk extrapolation).
    Loss = mean(L_e) + λ·Var(L_e).
    Saves to results/<protocol>/vrex/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lambda_grid is None:
        lambda_grid = [1e-2, 1.0, 10.0, 100.0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.array(train_idx)
    Xtr_raw = X_all.iloc[train_idx].values if hasattr(X_all, "iloc") else X_all[train_idx]
    Xv_raw  = X_all.iloc[val_idx].values   if hasattr(X_all, "iloc") else X_all[val_idx]
    Xte_raw = X_all.iloc[test_idx].values  if hasattr(X_all, "iloc") else X_all[test_idx]

    scaler  = StandardScaler().fit(Xtr_raw)
    Xtr_np  = scaler.transform(Xtr_raw)
    Xv_np   = scaler.transform(Xv_raw)
    Xte_np  = scaler.transform(Xte_raw)

    ytr = y_all[train_idx]; yv = y_all[val_idx]; yte = y_all[test_idx]

    Xv_t  = torch.tensor(Xv_np,  dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte_np, dtype=torch.float32).to(device)
    in_dim = Xv_t.shape[1]

    pos_w = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1)
    cw    = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)

    envs = _make_envs_tabular(X_all, y_all, train_idx, Xtr_np, device)

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    best_val_f1, best_state, best_lam = -1, None, None

    for lam in lambda_grid:
        set_seed(seed)
        model = VREx_MLP(in_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(epochs):
            model.train(); opt.zero_grad()
            losses_e = torch.stack([F.cross_entropy(model(xi), yi, weight=cw)
                                    for xi, yi in envs])
            (losses_e.mean() + lam * losses_e.var()).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
        vf1 = f1_score(yv, (vp > 0.5).astype(int), average="macro")
        if vf1 > best_val_f1:
            best_val_f1 = vf1; best_lam = lam
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model = VREx_MLP(in_dim).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()

    met = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    _save_checkpoint({k: v.clone() for k, v in model.state_dict().items()}, protocol, "vrex", seed)

    _save_json(_make_log("VREx", protocol, seed,
                         {"architecture": "MLP-128-Drop-64-2",
                          "optimizer": "Adam", "lr": 1e-3,
                          "weight_decay": 1e-4, "epochs": epochs,
                          "n_envs": len(envs), "pos_weight": round(pos_w, 4),
                          "lambda_grid": lambda_grid, "best_lambda": best_lam,
                          "standardised": True},
                         met, extra={"test_probs": probs.tolist()}), protocol, "vrex", seed)
    return results


def run_groupdro(
    X_all, y_all,
    train_idx, val_idx, test_idx,
    seed: int,
    eta_grid: list[float] | None = None,
    epochs: int = 100,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    GroupDRO baseline (tabular MLP + exponentiated group weight update).
    q_e ∝ exp(η·L_e);  training loss = Σ q_e·L_e.
    Saves to results/<protocol>/groupdro/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if eta_grid is None:
        eta_grid = [0.001, 0.01, 0.1]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.array(train_idx)
    Xtr_raw = X_all.iloc[train_idx].values if hasattr(X_all, "iloc") else X_all[train_idx]
    Xv_raw  = X_all.iloc[val_idx].values   if hasattr(X_all, "iloc") else X_all[val_idx]
    Xte_raw = X_all.iloc[test_idx].values  if hasattr(X_all, "iloc") else X_all[test_idx]

    scaler  = StandardScaler().fit(Xtr_raw)
    Xtr_np  = scaler.transform(Xtr_raw)
    Xv_np   = scaler.transform(Xv_raw)
    Xte_np  = scaler.transform(Xte_raw)

    ytr = y_all[train_idx]; yv = y_all[val_idx]; yte = y_all[test_idx]

    Xv_t  = torch.tensor(Xv_np,  dtype=torch.float32).to(device)
    Xte_t = torch.tensor(Xte_np, dtype=torch.float32).to(device)
    in_dim = Xv_t.shape[1]

    pos_w  = float((ytr == 0).sum()) / max(float((ytr == 1).sum()), 1)
    cw     = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
    envs   = _make_envs_tabular(X_all, y_all, train_idx, Xtr_np, device)
    n_envs = len(envs)

    best_val_f1, best_state, best_eta = -1, None, None

    for eta_q in eta_grid:
        set_seed(seed)
        model = GDRO_MLP(in_dim).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        q     = torch.ones(n_envs, device=device) / n_envs

        for _ in range(epochs):
            model.train(); opt.zero_grad()
            losses_e = torch.stack([F.cross_entropy(model(xi), yi, weight=cw)
                                    for xi, yi in envs])
            with torch.no_grad():
                # Numerically stable log-space update
                log_q = torch.log(q + 1e-12) + eta_q * losses_e.detach()
                log_q = log_q - log_q.max()
                q     = torch.exp(log_q) / torch.exp(log_q).sum()
            (q * losses_e).sum().backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = torch.softmax(model(Xv_t), dim=1)[:, 1].cpu().numpy()
        vf1 = f1_score(yv, (vp > 0.5).astype(int), average="macro")
        if vf1 > best_val_f1:
            best_val_f1 = vf1; best_eta = eta_q
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    model = GDRO_MLP(in_dim).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(Xte_t), dim=1)[:, 1].cpu().numpy()

    met = compute_metrics_binary(yte, probs)
    for k in met:
        results[k].append(met[k])
    results["probs"].append(probs.tolist())
    _save_checkpoint({k: v.clone() for k, v in model.state_dict().items()}, protocol, "groupdro", seed)

    _save_json(_make_log("GroupDRO", protocol, seed,
                         {"architecture": "MLP-128-Drop-64-2",
                          "optimizer": "Adam", "lr": 1e-3,
                          "weight_decay": 1e-4, "epochs": epochs,
                          "n_envs": n_envs, "pos_weight": round(pos_w, 4),
                          "eta_grid": eta_grid, "best_eta": best_eta,
                          "standardised": True},
                         met, extra={"test_probs": probs.tolist()}), protocol, "groupdro", seed)
    return results


def run_gnn_irm(
    x, y, edge_index,
    train_idx, val_idx, test_idx,
    seed: int,
    node_envs: list[np.ndarray] | None = None,
    lambda_grid: list[float] | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int | None = None,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    GNN + IRM baseline (SAGEConv-1layer encoder + linear head + IRM penalty on graph node subsets).
    Mini-batch training via NeighborLoader (1-hop, batch_size=256).
    IRM penalty computed per-batch by splitting batch nodes into environments.
    Val/test evaluation uses the full graph.
    Hyperparameter: λ (tuned on val Macro-F1).
    Early stopping: disabled by default (patience=None).
    Saves to results/<protocol>/gnn_irm/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lambda_grid is None:
        lambda_grid = [1e-2, 1e-1, 1.0, 10.0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.array(train_idx)
    x_cpu = _scale_graph_features(x, train_idx, device)
    x     = x_cpu.to(device)
    y = y.to(device) if isinstance(y, torch.Tensor) else \
        torch.tensor(y, dtype=torch.long).to(device)
    edge_index = edge_index.to(device) if isinstance(edge_index, torch.Tensor) else \
        torch.tensor(edge_index, dtype=torch.long).to(device)

    if node_envs is None:
        node_envs = _make_envs_graph(train_idx)

    # Build per-node environment label for fast batch splitting
    env_label = np.full(x.shape[0], -1, dtype=np.int64)
    for e_idx, env in enumerate(node_envs):
        env_label[env] = e_idx
    env_label_t = torch.tensor(env_label, dtype=torch.long, device=device)

    n_classes = int(y.max().item()) + 1
    vidx    = torch.tensor(val_idx,   dtype=torch.long, device=device)
    testidx = torch.tensor(test_idx,  dtype=torch.long, device=device)

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    _ytr     = y[torch.tensor(train_idx, dtype=torch.long, device=device)].cpu().numpy()
    _n_neg   = float((_ytr == 0).sum())
    _n_pos   = float((_ytr == 1).sum())
    # Square-root damping: full inverse-frequency (neg/pos ~3.8x on policy shift)
    # over-penalises negatives and collapses accuracy. sqrt gives a balanced
    # compromise that improves minority-class recall without overcorrecting.
    _pos_w   = (_n_neg / max(_n_pos, 1.0)) ** 0.5
    cw_gnn   = torch.tensor([1.0, _pos_w], dtype=torch.float32).to(device)

    best_val_f1, best_state, best_lam = -1, None, None

    for lam in lambda_grid:
        set_seed(seed)
        model, opt = _gnn_init(x.shape[1], n_classes, device)

        loader = _make_neighbor_loader(
            x_cpu, y.cpu(), edge_index.cpu(), train_idx,
            num_neighbors=[20], batch_size=batch_size, seed=seed)

        best_inner_loss, best_inner_state = float("inf"), None
        no_improve = 0

        for _ in range(epochs):
            model.train()
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                logits = model(batch.x, batch.edge_index)
                # Only seed nodes (first batch_size entries) are train nodes
                seed_logits = logits[:batch.batch_size]
                seed_y      = batch.y[:batch.batch_size]
                seed_nids   = batch.n_id[:batch.batch_size]
                erm_loss    = F.cross_entropy(seed_logits, seed_y, weight=cw_gnn)

                # IRM penalty: split batch seed nodes by environment
                batch_env   = env_label_t[seed_nids]
                irm_pen     = torch.zeros(1, device=device)
                n_valid_envs = 0
                for e in range(len(node_envs)):
                    mask = batch_env == e
                    if mask.sum() < 2:   # skip envs with too few nodes
                        continue
                    irm_pen = irm_pen + irm_penalty(seed_logits[mask], seed_y[mask])
                    n_valid_envs += 1
                if n_valid_envs > 0:
                    irm_pen = irm_pen / n_valid_envs

                (erm_loss + lam * irm_pen).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(model(x, edge_index)[vidx], y[vidx]).item()
            if vl < best_inner_loss:
                best_inner_loss  = vl
                best_inner_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                if patience is not None:
                    no_improve += 1
                    if no_improve >= patience:
                        break

        model.load_state_dict(best_inner_state); model.eval()
        with torch.no_grad():
            vf1 = f1_score(y[vidx].cpu().numpy(),
                           model(x, edge_index)[vidx].argmax(1).cpu().numpy(),
                           average="macro")
        if vf1 > best_val_f1:
            best_val_f1, best_state, best_lam = vf1, best_inner_state, lam

    model = GNN(x.shape[1], 64, n_classes).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits_test = model(x, edge_index)[testidx]
        met  = compute_metrics_gnn(logits_test, y[testidx])
        p1   = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

    for k in met:
        results.setdefault(k, []).append(met[k])
    results.setdefault("probs", []).append(p1.tolist())
    _save_checkpoint(best_state, protocol, "gnn_irm", seed)

    _save_json(_make_log("GNN_IRM", protocol, seed,
                         {"architecture": "SAGEConv-1layer+LinearHead", "hidden_dim": 64,
                          "dropout": 0.5, "optimizer": "Adam", "lr": 0.01,
                          "weight_decay": 5e-4, "epochs": epochs,
                          **({} if patience is None else {"patience": patience}),
                          "batch_size": batch_size, "num_neighbors": [20],
                          "n_envs": len(node_envs),
                          "lambda_grid": lambda_grid, "best_lambda": best_lam,
                          "pos_weight": round(float(_pos_w), 4), "weighting": "sqrt_inv_freq"},
                         met, extra={"test_probs": p1.tolist()}), protocol, "gnn_irm", seed)
    return results


def run_girl(
    x, y, edge_index, edge_weight,
    train_idx, val_idx, test_idx,
    seed: int,
    node_envs: list[np.ndarray] | None = None,
    lam_grid:  list[float] | None = None,
    mu_grid:   list[float] | None = None,
    epochs: int = 200,
    batch_size: int = 256,
    patience: int | None = None,
    results: dict | None = None,
    device: torch.device | None = None,
    protocol: str = "unknown",
) -> dict:
    """
    GIRL — proposed method (SAGEConv-2layer encoder + linear head + IRM penalty + GeoReg penalty).
    Mini-batch training via NeighborLoader (2-hop, batch_size=256).

    Linear prediction head matches paper spec (Section 2.3): hw(z) = softmax(Wz + b).
    Reduces NeighborLoader from 3-hop to 2-hop vs prior GCNConv implementation;
    significantly reducing batch size and training time.

    Per paper Section 2.7:
      - IRM penalty computed on batch seed nodes split by environment
      - GeoReg penalty computed on batch induced subgraph edges
      - Val/test evaluation uses the full graph (standard transductive setup)

    Joint grid search over (λ, μ) tuned on val Macro-F1.
    Early stopping: set patience to a positive integer to halt training when
    val loss does not improve for that many consecutive epochs (disabled by default).
    Saves to results/<protocol>/girl/seed_<n>.json
    """
    results = _ensure_keys(results if results is not None else {})
    if lam_grid is None:
        lam_grid = [0.1, 1.0, 10.0]
    if mu_grid is None:
        mu_grid  = [0.01, 0.1, 1.0]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_idx = np.array(train_idx)
    x_cpu = _scale_graph_features(x, train_idx, device)
    x     = x_cpu.to(device)
    y = y.to(device) if isinstance(y, torch.Tensor) else \
        torch.tensor(y, dtype=torch.long).to(device)
    edge_index  = edge_index.to(device) if isinstance(edge_index, torch.Tensor) else \
        torch.tensor(edge_index, dtype=torch.long).to(device)
    edge_weight = edge_weight.to(device) if isinstance(edge_weight, torch.Tensor) else \
        torch.tensor(edge_weight, dtype=torch.float32).to(device)

    if node_envs is None:
        node_envs = _make_envs_graph(train_idx)

    # Build per-node environment label for fast batch splitting
    env_label = np.full(x.shape[0], -1, dtype=np.int64)
    for e_idx, env in enumerate(node_envs):
        env_label[env] = e_idx
    env_label_t = torch.tensor(env_label, dtype=torch.long, device=device)

    n_classes = int(y.max().item()) + 1
    vidx    = torch.tensor(val_idx,  dtype=torch.long, device=device)
    testidx = torch.tensor(test_idx, dtype=torch.long, device=device)

    # Inverse-frequency class weighting — computed from training labels only.
    # Corrects for class imbalance (positive class ~23% of incidents).
    # Matches the weighting applied in tabular runners (run_lr, run_mlp, etc.)
    _ytr     = y[torch.tensor(train_idx, dtype=torch.long, device=device)].cpu().numpy()
    _n_neg   = float((_ytr == 0).sum())
    _n_pos   = float((_ytr == 1).sum())
    # Square-root damping: full ratio overcorrects; sqrt balances recall vs accuracy.
    _pos_w   = (_n_neg / max(_n_pos, 1.0)) ** 0.5
    cw_gnn   = torch.tensor([1.0, _pos_w], dtype=torch.float32).to(device)

    best_val_f1, best_state = -1, None
    best_lam, best_mu       = None, None

    # ── Geometry penalty normalisation (computed once, seeded for reproducibility)
    # Lgeo scales with n_edges * ||z||^2 — typically 4,000-6,000x larger than
    # ERM loss. We normalise by the geo penalty at init so mu operates as a
    # fraction of ERM loss, making the same grid meaningful across protocols.
    #
    # IRM penalty is NOT normalised here. Its scale at random init is highly
    # sensitive to initialisation (varies 4 orders of magnitude across seeds)
    # making init-based normalisation unstable. Instead, IRM is introduced via
    # warmup: the first irm_warmup_epochs train ERM+Geo only, then lam is
    # applied. By warmup completion the model is partially converged and IRM
    # gradients are more stable. This follows standard IRM practice.
    torch.manual_seed(seed)  # seed temporary model for reproducible scale
    _need_geo_norm = any(m != 0 for m in mu_grid)
    if _need_geo_norm:
        _tmp_model = GIRL(x.shape[1], 64, n_classes).to(device)
        _tmp_model.eval()
        with torch.no_grad():
            _Z_tmp = _tmp_model.encode(x, edge_index, edge_weight)
        _train_set  = set(train_idx.tolist())
        _src_cpu    = edge_index[0].cpu().numpy()
        _dst_cpu    = edge_index[1].cpu().numpy()
        _train_mask = torch.tensor(
            [s in _train_set and d in _train_set
             for s, d in zip(_src_cpu, _dst_cpu)],
            dtype=torch.bool, device=device,
        )
        _ei_tr = edge_index[:, _train_mask]
        _ew_tr = edge_weight[_train_mask] if edge_weight is not None else None
        with torch.no_grad():
            _geo_init = georeg_penalty(_Z_tmp, _ei_tr, _ew_tr).item()
        scale_geo = max(_geo_init, 1e-8)
        del _tmp_model, _Z_tmp
    else:
        scale_geo = 1.0
    scale_irm = 1.0  # IRM not normalised — controlled by warmup instead

    irm_warmup_epochs = max(1, epochs // 10)  # first 10% of epochs = ERM+Geo only
    # ─────────────────────────────────────────────────────────────────────

    for lam in lam_grid:
        for mu in mu_grid:
            set_seed(seed)
            model = GIRL(x.shape[1], 64, n_classes).to(device)
            opt   = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

            # 2-hop neighbours matching 2-layer GIRL encoder (linear head needs no extra hop)
            loader = _make_neighbor_loader(
                x_cpu, y.cpu(), edge_index.cpu(), train_idx,
                num_neighbors=[20, 20], batch_size=batch_size, seed=seed)

            best_inner_loss, best_inner_state = float("inf"), None
            no_improve = 0

            for epoch in range(epochs):
                # IRM warmup: apply lam only after irm_warmup_epochs
                lam_eff = lam if epoch >= irm_warmup_epochs else 0.0
                model.train()
                for batch in loader:
                    batch = batch.to(device)
                    opt.zero_grad()

                    # Encode full batch for GeoReg, then classify seed nodes
                    Z_batch  = model.encode(batch.x, batch.edge_index,
                                            batch.edge_attr)
                    logits   = model.head(Z_batch)   # linear head — no aggregation

                    seed_logits = logits[:batch.batch_size]
                    seed_y      = batch.y[:batch.batch_size]
                    seed_nids   = batch.n_id[:batch.batch_size]
                    erm_loss    = F.cross_entropy(seed_logits, seed_y, weight=cw_gnn)

                    # IRM penalty: split batch seed nodes by environment
                    batch_env    = env_label_t[seed_nids]
                    irm_pen      = torch.zeros(1, device=device)
                    n_valid_envs = 0
                    for e in range(len(node_envs)):
                        mask = batch_env == e
                        if mask.sum() < 2:
                            continue
                        irm_pen = irm_pen + irm_penalty(
                            seed_logits[mask], seed_y[mask])
                        n_valid_envs += 1
                    if n_valid_envs > 0:
                        irm_pen = irm_pen / n_valid_envs

                    # GeoReg penalty on batch induced subgraph representations
                    ei_b   = batch.edge_index
                    ew_b   = batch.edge_attr
                    geo_pen = georeg_penalty(Z_batch, ei_b, ew_b)

                    loss = erm_loss + lam_eff * irm_pen + mu * (geo_pen / scale_geo)
                    loss.backward()
                    opt.step()

                # Full-graph val evaluation
                model.eval()
                with torch.no_grad():
                    vl = F.cross_entropy(
                        model(x, edge_index, edge_weight)[vidx], y[vidx]
                    ).item()
                if vl < best_inner_loss:
                    best_inner_loss  = vl
                    best_inner_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    if patience is not None:
                        no_improve += 1
                        if no_improve >= patience:
                            break

            model.load_state_dict(best_inner_state); model.eval()
            with torch.no_grad():
                vf1 = f1_score(
                    y[vidx].cpu().numpy(),
                    model(x, edge_index, edge_weight)[vidx].argmax(1).cpu().numpy(),
                    average="macro",
                )
            if vf1 > best_val_f1:
                best_val_f1 = vf1
                best_lam, best_mu = lam, mu
                best_state = best_inner_state

    model = GIRL(x.shape[1], 64, n_classes).to(device)
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        logits_test = model(x, edge_index, edge_weight)[testidx]
        met  = compute_metrics_gnn(logits_test, y[testidx])
        p1   = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()

    for k in met:
        results.setdefault(k, []).append(met[k])
    results.setdefault("probs", []).append(p1.tolist())
    _save_checkpoint(best_state, protocol, "girl", seed)

    _save_json(_make_log("GIRL", protocol, seed,
                         {"architecture": "SAGEConv-2layer+LinearHead", "hidden_dim": 64,
                          "dropout": 0.5, "optimizer": "Adam", "lr": 0.01,
                          "weight_decay": 5e-4, "epochs": epochs,
                          "batch_size": batch_size, "num_neighbors": [20, 20],
                          **({} if patience is None else {"patience": patience}),
                          "n_envs": len(node_envs),
                          "lam_grid": lam_grid, "mu_grid": mu_grid,
                          "best_lambda": best_lam, "best_mu": best_mu,
                          "scale_irm": "warmup",
                          "scale_geo": round(scale_geo, 2),
                          "irm_warmup_epochs": irm_warmup_epochs,
                          "pos_weight": round(float(_pos_w), 4), "weighting": "sqrt_inv_freq"},
                         met, extra={"test_probs": p1.tolist()}), protocol, "girl", seed)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-city reliability diagram helper  (Step 6 – calibration analysis)
# ─────────────────────────────────────────────────────────────────────────────

def collect_city_probs(
    runner_fn,
    splits: list[dict],
    y_all: np.ndarray,
    X_all,
    seed: int,
    device: torch.device,
    x_full_unscaled: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_weight: torch.Tensor | None = None,
    runner_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run one seed of any run_* function across all city folds and return
    pooled (probs, labels) suitable for a reliability diagram.

    Used exclusively for Step 6 calibration analysis — NOT for reporting
    metrics (use the main city-loop runners for that).

    Parameters
    ----------
    runner_fn        : one of run_lr, run_mlp, run_xgb,
                       run_gnn_erm, run_gnn_georeg, run_gnn_irm, run_girl
    splits           : list of city-fold dicts from load_city_splits(seed)
    y_all            : full label array [N]
    X_all            : full feature DataFrame [N, F]  (tabular runners)
    seed             : which seed to use
    device           : torch device
    x_full_unscaled  : raw (unscaled) node feature tensor — the function
                       refits a StandardScaler per fold and passes the
                       rescaled tensor to graph runners
    edge_index       : graph edge index tensor  (graph runners only)
    edge_weight      : graph edge weight tensor (georeg / girl only)
    runner_kwargs    : extra kwargs forwarded to runner_fn
                       e.g. {"epochs": 100, "lambda_grid": [0.1, 1.0]}

    Returns
    -------
    probs  : np.ndarray [N_test_total]  — pooled positive-class probabilities
    labels : np.ndarray [N_test_total]  — corresponding true labels
    """
    if runner_kwargs is None:
        runner_kwargs = {}

    is_graph = runner_fn in (run_gnn_erm, run_gnn_georeg, run_gnn_irm, run_girl)

    # FIX-5: The CSV produced by Step 1 now contains RAW (unscaled) features.
    # For graph runners, we must fit a per-fold StandardScaler on train nodes
    # and apply it to the full feature matrix before passing to the runner.
    # For tabular runners (run_lr, run_mlp, etc.) the runner's own internal
    # _standardise() call handles this — do NOT pre-scale here.
    # Previously: Step 1 wrote a pre-scaled CSV AND this function re-scaled it,
    # producing double z-scoring that collapsed feature variance to near-zero.
    X_np = X_all.values if hasattr(X_all, "iloc") else np.array(X_all)

    all_probs, all_labels = [], []

    for sp in splits:
        ti     = np.array(sp["train_idx"])
        vi     = sp["val_idx"]
        tei    = sp["test_idx"]
        kwargs = dict(runner_kwargs)   # available in both branches

        res = {}

        if is_graph:
            # Refit scaler on this fold's train nodes
            scaler   = StandardScaler().fit(X_np[ti])
            X_scaled = scaler.transform(X_np)
            x_fold   = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            if runner_fn in (run_gnn_georeg, run_girl) and edge_weight is not None:
                res = runner_fn(x_fold, torch.tensor(y_all, dtype=torch.long).to(device),
                                edge_index, edge_weight,
                                ti, vi, tei, seed=seed, device=device,
                                protocol="calibration", **kwargs)
            else:
                res = runner_fn(x_fold, torch.tensor(y_all, dtype=torch.long).to(device),
                                edge_index,
                                ti, vi, tei, seed=seed, device=device,
                                protocol="calibration", **kwargs)
        else:
            # Three calling conventions for tabular runners:
            #   run_lr, run_xgb       → sliced arrays, no device
            #   run_mlp               → sliced arrays, with device
            #   run_irm/vrex/groupdro → full X_all + indices, with device
            if runner_fn in (run_lr, run_xgb):
                res = runner_fn(
                    X_all.iloc[ti] if hasattr(X_all, "iloc") else X_all[ti],
                    y_all[ti],
                    X_all.iloc[vi] if hasattr(X_all, "iloc") else X_all[vi],
                    y_all[vi],
                    X_all.iloc[tei] if hasattr(X_all, "iloc") else X_all[tei],
                    y_all[tei],
                    seed=seed, protocol="calibration", **kwargs,
                )
            elif runner_fn in (run_irm, run_vrex, run_groupdro):
                res = runner_fn(
                    X_all, y_all, ti, vi, tei,
                    seed=seed, device=device,
                    protocol="calibration", **kwargs,
                )
            else:
                # run_mlp and other slice-based torch runners
                res = runner_fn(
                    X_all.iloc[ti] if hasattr(X_all, "iloc") else X_all[ti],
                    y_all[ti],
                    X_all.iloc[vi] if hasattr(X_all, "iloc") else X_all[vi],
                    y_all[vi],
                    X_all.iloc[tei] if hasattr(X_all, "iloc") else X_all[tei],
                    y_all[tei],
                    seed=seed, device=device,
                    protocol="calibration", **kwargs,
                )

        all_probs.append(np.array(res["probs"][-1]))
        all_labels.append(y_all[tei])

    return np.concatenate(all_probs), np.concatenate(all_labels)
