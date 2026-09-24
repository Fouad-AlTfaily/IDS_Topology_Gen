# ============================================================
# STEP 5 -- GNN-IDS for LEVEL 3 (utility validation)
#
# The repository contained NO existing GNN / IDS implementation (checked), so
# Step 5 defines one simple, reproducible, well-documented architecture and
# uses it IDENTICALLY for the three training conditions (real / synthetic /
# real+synthetic). The ONLY experimental variable is the training data.
#
# ARCHITECTURE -- E-GraphSAGE-style edge classifier (Lo et al., 2022, the
# standard GNN-NIDS baseline for NetFlow datasets incl. NF-ToN-IoT):
#   * graph  : nodes = IP addresses, one DIRECTED edge per flow
#              (IPV4_SRC_ADDR -> IPV4_DST_ADDR). Edge feature = the 9
#              preprocessed CTGAN flow features. Node init feature = all-ones
#              vector of length 9 (E-GraphSAGE convention -- structure, not
#              node identity, carries the signal; keeps the model INDUCTIVE so
#              unseen test IPs are fine).
#   * 2 message-passing layers. Per layer:
#         m_e   = ReLU(W_msg^l [ h_src(e) || edge_attr(e) ])
#         agg_v = mean of m_e over every edge incident to v (in + out)
#         h_v   = L2norm( ReLU(W_upd^l [ h_v || agg_v ]) )
#   * edge classifier: MLP( [ h_src || h_dst || edge_attr ] ) -> 9 logits
#   * loss = class-weighted cross-entropy, weights = inverse class frequency
#     of THAT training set only (never the real TEST distribution).
#
# PREPROCESSING (fit on the relevant TRAINING set only, transform-only on TEST):
#   log1p on {IN_BYTES,OUT_BYTES,IN_PKTS,OUT_PKTS,FLOW_DURATION_MILLISECONDS}
#   then per-feature standardisation (mean/std from the training edges).
#
# This module has NO side effects on import. torch is imported at module load
# (only step5_3_utility.py imports this module).
# ============================================================

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.validation import step5_common as S5

FEATURE_ORDER = list(S5.FLOW_FEATURES)                       # 9, fixed order
LOG_IDX = [FEATURE_ORDER.index(c) for c in S5.LOG_FEATURES]  # heavy-tailed cols


# ------------------------------------------------------------------
def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# preprocessing  (fit on TRAIN only)
# ------------------------------------------------------------------
def fit_preprocessor(train_df: pd.DataFrame) -> dict:
    X = train_df[FEATURE_ORDER].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    X[:, LOG_IDX] = np.log1p(np.clip(X[:, LOG_IDX], 0.0, None))
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return {"feature_order": FEATURE_ORDER, "log_idx": LOG_IDX,
            "mean": mean.tolist(), "std": std.tolist(),
            "fit_rows": int(len(train_df)),
            "fit_note": "log1p on heavy-tailed cols then per-feature standardise; "
                        "statistics from the TRAINING edges of this condition only"}


def transform_features(df: pd.DataFrame, prep: dict) -> np.ndarray:
    X = df[prep["feature_order"]].to_numpy(dtype=np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    X[:, prep["log_idx"]] = np.log1p(np.clip(X[:, prep["log_idx"]], 0.0, None))
    X = (X - np.asarray(prep["mean"])) / np.asarray(prep["std"])
    return X.astype(np.float32)


# ------------------------------------------------------------------
# graph tensorisation
# ------------------------------------------------------------------
@dataclass
class GraphTensors:
    edge_index: torch.Tensor       # [2, E] long
    edge_attr: torch.Tensor        # [E, F] float32
    y: torch.Tensor                # [E] long
    num_nodes: int
    num_features: int
    num_classes: int
    n_edges: int
    class_counts: dict


def build_graph_tensors(df: pd.DataFrame, prep: dict) -> GraphTensors:
    df = df.reset_index(drop=True)
    ips = pd.Index(pd.unique(pd.concat([df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"]],
                                       ignore_index=True)))
    id_of = {ip: i for i, ip in enumerate(ips)}
    src = df["IPV4_SRC_ADDR"].map(id_of).to_numpy()
    dst = df["IPV4_DST_ADDR"].map(id_of).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    edge_attr = torch.tensor(transform_features(df, prep), dtype=torch.float32)
    yy = df["Attack"].map(S5.LABEL_MAP)
    if yy.isna().any():
        bad = sorted(df.loc[yy.isna(), "Attack"].astype(str).unique())
        raise SystemExit(f"ABORT: unmapped Attack label(s) while building the graph: {bad}")
    y = torch.tensor(yy.to_numpy(), dtype=torch.long)
    cc = {S5.INV_LABEL_MAP[int(k)]: int(v) for k, v in
          zip(*np.unique(y.numpy(), return_counts=True))}
    return GraphTensors(edge_index, edge_attr, y, len(ips),
                        edge_attr.shape[1], len(S5.CLASSES), len(df), cc)


def class_weights_from_labels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=num_classes).float()
    counts[counts == 0] = 1.0
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def stratified_subsample(df: pd.DataFrame, max_edges: int, min_per_class: int,
                         seed: int) -> tuple[pd.DataFrame, dict]:
    n = len(df)
    if n <= max_edges:
        return df.reset_index(drop=True), {"subsampled": False, "rows": n}
    frac = max_edges / n
    parts = []
    for cls, g in df.groupby("Attack", sort=True):
        k = min(len(g), max(min_per_class, int(np.ceil(len(g) * frac))))
        parts.append(g.sample(n=k, random_state=seed))
    out = (pd.concat(parts).sample(frac=1.0, random_state=seed)
                          .reset_index(drop=True))
    return out, {"subsampled": True, "rows_before": n, "rows_after": len(out),
                 "target": max_edges, "min_per_class": min_per_class,
                 "per_class": {c: int((out["Attack"] == c).sum()) for c in S5.CLASSES}}


# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------
def _scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_zeros((dim_size, src.shape[1]))
    out.index_add_(0, index, src)
    cnt = src.new_zeros(dim_size)
    cnt.index_add_(0, index, torch.ones(index.shape[0], device=src.device, dtype=src.dtype))
    return out / cnt.clamp_(min=1.0).unsqueeze(1)


class EdgeSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.layers = layers
        self.dropout = dropout
        self.msg = nn.ModuleList()
        self.upd = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.msg.append(nn.Linear(d + in_dim, hidden))
            self.upd.append(nn.Linear(d + hidden, hidden))
            d = hidden
        self.cls = nn.Sequential(
            nn.Linear(2 * hidden + in_dim, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        inc_idx = torch.cat([dst, src], dim=0)          # aggregate over in + out edges
        h = edge_attr.new_ones((num_nodes, self.in_dim))
        for l in range(self.layers):
            m = F.relu(self.msg[l](torch.cat([h[src], edge_attr], dim=1)))
            agg = _scatter_mean(torch.cat([m, m], dim=0), inc_idx, num_nodes)
            h = F.relu(self.upd[l](torch.cat([h, agg], dim=1)))
            h = F.normalize(h, p=2, dim=1)
            h = F.dropout(h, p=self.dropout, training=self.training)
        z = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.cls(z)


# ------------------------------------------------------------------
# train / evaluate
# ------------------------------------------------------------------
def train_model(train_t: GraphTensors, cfg: dict, seed: int,
                device: torch.device, log_fn=print):
    set_determinism(seed)
    model = EdgeSAGE(train_t.num_features, cfg["hidden_dim"], train_t.num_classes,
                     cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"])
    w = (class_weights_from_labels(train_t.y, train_t.num_classes).to(device)
         if cfg["class_weighted_loss"] else None)

    ei = train_t.edge_index.to(device)
    ea = train_t.edge_attr.to(device)
    y = train_t.y.to(device)

    history = []
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(ei, ea, train_t.num_nodes)
        loss = F.cross_entropy(logits, y, weight=w)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 10 == 0 or ep == cfg["epochs"]:
            with torch.no_grad():
                acc = (logits.argmax(1) == y).float().mean().item()
            history.append({"epoch": ep, "loss": float(loss.item()),
                            "train_acc": float(acc)})
            log_fn(f"    epoch {ep:3d}/{cfg['epochs']}  loss {loss.item():.4f}  "
                   f"train_acc {acc:.4f}  elapsed {time.time()-t0:.1f}s")
    return model, {"history": history, "train_seconds": round(time.time() - t0, 1),
                   "class_weights": (w.detach().cpu().tolist() if w is not None else None)}


@torch.no_grad()
def evaluate(model: nn.Module, test_t: GraphTensors, device: torch.device):
    model.eval()
    logits = model(test_t.edge_index.to(device), test_t.edge_attr.to(device),
                   test_t.num_nodes)
    y_pred = logits.argmax(1).cpu().numpy()
    y_true = test_t.y.cpu().numpy()
    return y_true, y_pred


def save_checkpoint(model: nn.Module, path: str, meta: dict):
    S5._assert_under_s5(path)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)
