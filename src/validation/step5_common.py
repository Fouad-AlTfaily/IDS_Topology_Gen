# ============================================================
# STEP 5 -- SHARED CONFIG + UTILITIES  (final validation, 3 levels)
#
# Step 5 is VALIDATION ONLY. Everything from Steps 1-4 is READ-ONLY:
#   * Step 1 graph fingerprints
#   * Step 2 synthetic graphs
#   * Step 3 synthetic attack flows
#   * Step 4 synthetic benign flows
#   * the final synthetic dataset for the selected SCALE
#     (SCALE="smaller" -> results/small_scale/final_synthetic/synthetic_NF_ToN_IoT.csv)
#
# NOTHING here trains a CTGAN, regenerates data, or modifies any earlier
# output. Step 5 only ever writes under its own validation tree
# (SCALE="smaller" -> results/small_scale/step5_validation/).
#
# Three levels:
#   LEVEL 1  structural   -- graph degree distributions + modularity
#   LEVEL 2  statistical  -- real TEST vs final synthetic, per class/feature
#   LEVEL 3  utility      -- GNN-IDS trained on real / synthetic / real+synth,
#                            all evaluated on the SAME untouched real TEST set
#
# Hard rules enforced here:
#   * ransomware never appears (it was dropped upstream at Step 1)
#   * the real TEST split is reconstructed deterministically and is NEVER used
#     to fit anything (checked in assert_no_leakage / the Level-3 code paths)
#   * the train/val/test split is the EXACT project split (step3_2 stratified +
#     step4_1 benign), re-derived on the enriched file so IPs are retained,
#     then verified row-for-row against the existing split-statistics CSVs.
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

# re-use Step-3 / Step-4 infrastructure (atomic IO, seeds, env probe, JSON
# helpers) AND their SCALE-aware path layout, so Step 5 automatically validates
# whichever scale Steps 3-4 produced.
from src.attack_flows import step3_common as C3
from src.benign_flows import step4_common as C4
from src.config import CFG

# ------------------------------------------------------------------
# DATASET / GRAPH SCALE  (inherited from step3_common; run_pipeline.py --scale)
# ------------------------------------------------------------------
SCALE = C3.SCALE
OUT_PREFIX = C3.OUT_PREFIX

# ------------------------------------------------------------------
# PATHS  (all relative to the project root == cwd when scripts are run)
# ------------------------------------------------------------------
# Scale-INDEPENDENT: the enriched real-flow table + the Step-1 real graph.
ENRICHED_REAL = "results/step3/step3_1_enriched_real_flows.csv"
STEP1_EDGES = "results/step1/step1_edge_list.csv"
STEP1_NODES = "results/step1/step1_node_fingerprints.csv"

# Scale-DEPENDENT: the final synthetic dataset for THIS scale (smaller-scale ->
# results/small_scale/final_synthetic/).
FINAL_DIR = C4.FINAL_DIR
FINAL_SYNTH = C4.FINAL_CSV
FINAL_SYNTH_DONE = os.path.join(FINAL_DIR, "DONE")
FINAL_MERGE_MANIFEST = os.path.join(FINAL_DIR, "merge_manifest.json")

# Step-2 synthetic graph for THIS scale (results/step2/<SCALE>/). The historical
# names keep the *_SAME_* spelling for backwards compatibility with existing
# call sites, but they point at the selected scale.
STEP2_EDGES = C3.STEP2_EDGES
STEP2_NODES = C3.STEP2_NODES
STEP2_SAME_EDGES = STEP2_EDGES
STEP2_SAME_NODES = STEP2_NODES

# Step-3 attack side (generation is scale-dependent; the split-stats CSV is
# emitted into this scale's reports/ by step3_2_prepare).
STEP3_GEN_VALIDATION = os.path.join(C3.S3_GENERATION, "VALIDATION_DONE")
STEP3_DELIVERABLE = os.path.join(C3.S3_GENERATION, "synthetic_attack_flows.csv")
STEP3_SPLIT_STATS = os.path.join(C3.S3_REPORTS, "data_split_statistics.csv")

# Step-4 benign side (scale-dependent generation; split-stats emitted by
# step4_1_prepare into this scale's reports/).
STEP4_GEN_VALIDATION = os.path.join(C4.S4_GENERATION, "VALIDATION_DONE")
STEP4_DELIVERABLE = C4.SYNTHETIC_BENIGN_FLOWS
STEP4_SPLIT_STATS = os.path.join(C4.S4_REPORTS, "data_split_statistics.csv")

# ---- Step-5 output tree (scale-dependent) ---------------------------------
S5 = (os.path.join(OUT_PREFIX, "step5_validation") if OUT_PREFIX
      else "results/step5_validation")
S5_STRUCTURAL = os.path.join(S5, "structural")
S5_STATISTICAL = os.path.join(S5, "statistical")
S5_UTILITY = os.path.join(S5, "utility")
S5_FIGURES = os.path.join(S5, "figures")
S5_FIG_STRUCT = os.path.join(S5_FIGURES, "structural")
S5_FIG_STAT = os.path.join(S5_FIGURES, "statistical")
S5_FIG_UTIL = os.path.join(S5_FIGURES, "utility")
S5_REPORTS = os.path.join(S5, "reports")
S5_LOGS = os.path.join(S5, "logs")

ALL_S5_DIRS = [S5, S5_STRUCTURAL, S5_STATISTICAL, S5_UTILITY,
               S5_FIGURES, S5_FIG_STRUCT, S5_FIG_STAT, S5_FIG_UTIL,
               S5_REPORTS, S5_LOGS]

LEVEL1_DONE = os.path.join(S5_STRUCTURAL, "LEVEL1_DONE")
LEVEL2_DONE = os.path.join(S5_STATISTICAL, "LEVEL2_DONE")
LEVEL3_DONE = os.path.join(S5_UTILITY, "LEVEL3_DONE")
S5_DONE = os.path.join(S5, "DONE")

# ------------------------------------------------------------------
# CLASSES / LABELS  (established project scope -- ransomware EXCLUDED)
# ------------------------------------------------------------------
CLASSES = ["backdoor", "benign", "ddos", "dos", "injection",
           "mitm", "password", "scanning", "xss"]          # fixed sorted order
ATTACK_CLASSES = [c for c in CLASSES if c != "benign"]
LABEL_MAP = {c: i for i, c in enumerate(CLASSES)}
INV_LABEL_MAP = {i: c for c, i in LABEL_MAP.items()}
FORBIDDEN_CLASSES = ["ransomware"]

# ------------------------------------------------------------------
# FLOW FEATURES  (the 9 CTGAN-generated flow features -- nothing else)
# ------------------------------------------------------------------
FLOW_FEATURES = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT", "TCP_FLAGS",
]
# ports are integer-valued network fields -> compared with KS as numeric
# (documented in the Level-2 report); PROTOCOL / TCP_FLAGS are discrete codes
# -> compared with a categorical test (chi-square + JSD + TV distance).
NUMERIC_KS_FEATURES = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "L4_SRC_PORT", "L4_DST_PORT",
]
CATEGORICAL_FEATURES = ["PROTOCOL", "TCP_FLAGS"]
# heavy-tailed columns -> log1p before the GNN scaler (Level 3 only)
LOG_FEATURES = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
                "FLOW_DURATION_MILLISECONDS"]

IP_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]

# ------------------------------------------------------------------
# DETERMINISM / CONFIG
# ------------------------------------------------------------------
SPLIT_SEED = CFG["seeds"]["split"]       # 42, same value step3_common / step4_common use
RANDOM_SEED = CFG["seeds"]["random"]     # 42
LOUVAIN_SEED = CFG["seeds"]["louvain"]   # 42, same value step1_fingerprint.py uses

FORCE_RERUN = os.environ.get("STEP5_FORCE_RERUN", "0").strip().lower() in ("1", "true", "yes")

# ---- Level 3 (GNN utility) config -- IDENTICAL across the 3 training sets ---
# Architecture is unchanged (2-layer E-GraphSAGE edge classifier, all-ones node
# init). Only the training BUDGET was raised after the first run under-fit the
# real-only baseline (train_acc ~0.57, loss still falling at epoch 100):
#   max_train_edges 200k -> 800k   (was throwing away ~82% of real TRAIN)
#   epochs          100  -> 300    (real-only had not converged)
#   seeds           [42] -> [42,1,2]  (multi-seed mean +/- std; env still overrides)
# 800k (not "no cap") keeps the real+synthetic full-batch graph within T4 memory;
# every rare class is still taken in full via min_edges_per_class.
# Values: config.yaml -> validation.level3_egraphsage (hidden 128, 2 layers,
# dropout 0.2, Adam lr 1e-3, wd 5e-4, 300 epochs, cap 800k / min 5k per class,
# seeds [42, 1, 2]; STEP5_GNN_SEEDS overrides the seeds).
GNN_CONFIG = dict(CFG["validation"]["level3_egraphsage"])
GNN_CONFIG["seeds"] = list(GNN_CONFIG["seeds"])
_env_seeds = os.environ.get("STEP5_GNN_SEEDS", "").strip()
if _env_seeds:
    GNN_CONFIG["seeds"] = [int(s) for s in _env_seeds.split(",") if s.strip()]
# quick knobs if the full-batch graph OOMs on a smaller GPU (no code edit needed):
#   STEP5_MAX_TRAIN_EDGES=500000   STEP5_GNN_EPOCHS=200
_env_cap = os.environ.get("STEP5_MAX_TRAIN_EDGES", "").strip()
if _env_cap:
    GNN_CONFIG["max_train_edges"] = int(_env_cap)
_env_epochs = os.environ.get("STEP5_GNN_EPOCHS", "").strip()
if _env_epochs:
    GNN_CONFIG["epochs"] = int(_env_epochs)

UTILITY_CONDITIONS = ["real_only", "synthetic_only", "real_plus_synthetic"]

# ------------------------------------------------------------------
# FS / CHECKPOINT HELPERS  (writes are FORCED to stay under S5, i.e.
# results/small_scale/step5_validation/ for the smaller-scale experiment)
# ------------------------------------------------------------------
def _assert_under_s5(path: str) -> None:
    ap = os.path.abspath(path)
    root = os.path.abspath(S5)
    if not (ap == root or ap.startswith(root + os.sep)):
        raise RuntimeError(
            f"REFUSING to write outside {S5}/: {path} "
            f"(Step 5 must not modify Step 1-4 outputs)."
        )


def ensure_dirs(*extra):
    for d in list(ALL_S5_DIRS) + list(extra):
        os.makedirs(d, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: str, **kw):
    _assert_under_s5(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str):
    _assert_under_s5(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(obj, path: str):
    _assert_under_s5(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {} if default is None else default


def write_marker(path: str, payload: dict | None = None):
    _assert_under_s5(path)
    payload = dict(payload or {})
    payload.setdefault("completed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(payload, path)


def marker_ok(path: str) -> bool:
    return (not FORCE_RERUN) and os.path.exists(path)


def files_present(paths: list[str]) -> bool:
    return all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths)


def stage_done(marker: str, required: list[str]) -> bool:
    return marker_ok(marker) and files_present(required)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_global_seeds(seed: int = RANDOM_SEED):
    C3.set_global_seeds(seed)


@contextmanager
def stage_banner(name: str):
    bar = "=" * 70
    print(f"\n{bar}\n[STEP 5] {name}\n{bar}")
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[STEP 5] {name} -- finished in {time.time() - t0:,.1f}s")


# ------------------------------------------------------------------
# ENVIRONMENT REPORT
# ------------------------------------------------------------------
def _try_version(mod_name: str):
    try:
        return getattr(__import__(mod_name), "__version__", "unknown")
    except Exception:
        return "not installed"


def detect_environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _try_version("numpy"),
        "pandas": _try_version("pandas"),
        "scipy": _try_version("scipy"),
        "networkx": _try_version("networkx"),
        "sklearn": _try_version("sklearn"),
        "matplotlib": _try_version("matplotlib"),
        "python_louvain": _try_version("community"),
        "torch": _try_version("torch"),
        "torch_geometric": _try_version("torch_geometric"),
        "gpu_available": False,
        "gpu_name": "none",
        "cuda": "none",
    }
    try:
        import torch
        env["gpu_available"] = bool(torch.cuda.is_available())
        env["cuda"] = getattr(torch.version, "cuda", "unknown")
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return env


def write_environment_report() -> dict:
    ensure_dirs()
    env = detect_environment()
    lines = ["STEP 5 ENVIRONMENT", "=" * 40]
    for k, v in env.items():
        lines.append(f"{k:<18}: {v}")
    lines += ["", "config:",
              f"  SPLIT_SEED   : {SPLIT_SEED}",
              f"  RANDOM_SEED  : {RANDOM_SEED}",
              f"  LOUVAIN_SEED : {LOUVAIN_SEED}",
              f"  GNN_CONFIG   : {json.dumps(GNN_CONFIG)}",
              f"  FORCE_RERUN  : {FORCE_RERUN}"]
    atomic_write_text("\n".join(lines) + "\n", os.path.join(S5_REPORTS, "environment.txt"))
    return env


# ------------------------------------------------------------------
# INPUT VERIFICATION  (STOP instead of inventing a substitute)
# ------------------------------------------------------------------
def verify_inputs(require_synth: bool = True) -> dict:
    """Confirm every Step 1-4 artifact Step 5 needs exists and generation
    finished BEFORE now.  Returns a small provenance dict."""
    missing = []
    need = [ENRICHED_REAL, STEP1_EDGES, STEP1_NODES,
            STEP2_SAME_EDGES, STEP2_SAME_NODES,
            STEP3_SPLIT_STATS, STEP4_SPLIT_STATS]
    if require_synth:
        need += [FINAL_SYNTH, FINAL_SYNTH_DONE,
                 STEP3_GEN_VALIDATION, STEP4_GEN_VALIDATION]
    for p in need:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise SystemExit(
            "ABORT: required Step 1-4 input(s) missing -- Step 5 will NOT invent "
            "substitutes:\n  " + "\n  ".join(missing)
        )

    prov = {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "enriched_real_flows": {"path": ENRICHED_REAL, "sha256": sha256(ENRICHED_REAL)},
    }
    if require_synth:
        prov["final_synthetic"] = {
            "path": FINAL_SYNTH,
            "sha256": sha256(FINAL_SYNTH),
            "done": read_json(FINAL_SYNTH_DONE, {}),
            "merge_manifest": read_json(FINAL_MERGE_MANIFEST, {}),
        }
        prov["step3_generation_validation"] = read_json(STEP3_GEN_VALIDATION, {})
        prov["step4_generation_validation"] = read_json(STEP4_GEN_VALIDATION, {})
        # generation must have completed before Step 5 started
        now = time.time()
        for tag, p in (("final_synthetic", FINAL_SYNTH_DONE),
                       ("step3_gen", STEP3_GEN_VALIDATION),
                       ("step4_gen", STEP4_GEN_VALIDATION)):
            if os.path.getmtime(p) > now + 1:
                raise SystemExit(f"ABORT: {tag} marker is newer than 'now' -- clock/order problem")
    return prov


# ------------------------------------------------------------------
# THE PROJECT TRAIN / VAL / TEST SPLIT  (re-derived WITH IPs, then verified)
# ------------------------------------------------------------------
_CACHED_SPLIT: dict | None = None


def _read_enriched() -> pd.DataFrame:
    df = pd.read_csv(ENRICHED_REAL, low_memory=False)
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()
    # ransomware must already be gone from the enriched file
    n_ransom = int((df["Attack"] == "ransomware").sum())
    if n_ransom:
        raise SystemExit(f"ABORT: {n_ransom} ransomware rows in the enriched file -- scope broken")
    unexpected = sorted(set(df["Attack"].unique()) - set(CLASSES))
    if unexpected:
        raise SystemExit(f"ABORT: unexpected class(es) in enriched file: {unexpected}")
    return df


def load_real_split(verbose: bool = True) -> dict:
    """Reproduce the EXACT project split so real graphs carry IP endpoints.

      attacks -> step3_2_prepare.stratified_split(attack_df, 42)
      benign  -> step4_1_prepare.benign_split(benign_df.reset_index(drop=True), 42)

    Row membership is identical to results/step3_ctgan/splits/ and
    results/step4_benign/splits/ (those were produced by the same calls on the
    same rows, just with fewer columns). Verified here against the split-stats
    CSVs; raises on any mismatch.
    """
    global _CACHED_SPLIT
    if _CACHED_SPLIT is not None:
        return _CACHED_SPLIT

    from src.attack_flows import step3_2_prepare as s3p
    from src.benign_flows import step4_1_prepare as s4p

    df = _read_enriched()
    df["_row_id"] = np.arange(len(df), dtype=np.int64)          # global, stable row id
    keep = IP_COLS + FLOW_FEATURES + ["Attack", "Label", "_row_id",
                                      "src_out_degree", "dst_in_degree",
                                      "src_out_degree_cat", "dst_in_degree_cat",
                                      "src_role", "dst_role"]
    df = df[keep].copy()

    atk = df[df["Attack"].isin(ATTACK_CLASSES)].copy()          # gapped index preserved
    ben = df[df["Attack"] == "benign"].reset_index(drop=True)   # RangeIndex like step4_1

    a_tr, a_va, a_te = s3p.stratified_split(atk, SPLIT_SEED)
    b_tr, b_va, b_te = s4p.benign_split(ben, SPLIT_SEED)

    split = {
        "train": pd.concat([a_tr, b_tr], ignore_index=True),
        "val":   pd.concat([a_va, b_va], ignore_index=True),
        "test":  pd.concat([a_te, b_te], ignore_index=True),
    }

    # ---- verify against the official split-statistics CSVs -----------------
    s3 = pd.read_csv(STEP3_SPLIT_STATS).set_index("Attack")
    s4 = pd.read_csv(STEP4_SPLIT_STATS).set_index("split")
    problems = []
    for cls in ATTACK_CLASSES:
        for part, want_col in (("train", "train"), ("val", "val"), ("test", "test")):
            got = int((split[part]["Attack"] == cls).sum())
            want = int(s3.loc[cls, want_col])
            if got != want:
                problems.append(f"{cls}/{part}: re-derived {got} != official {want}")
    for part in ("train", "val", "test"):
        got = int((split[part]["Attack"] == "benign").sum())
        want = int(s4.loc[part, "rows"])
        if got != want:
            problems.append(f"benign/{part}: re-derived {got} != official {want}")
    if problems:
        raise SystemExit("ABORT: re-derived split does not match the official project "
                         "split:\n  " + "\n  ".join(problems))

    # ---- ROW-LEVEL disjointness (the real "no shared rows" guarantee) -----
    rid = {part: set(split[part]["_row_id"].tolist()) for part in ("train", "val", "test")}
    if rid["train"] & rid["test"]:
        raise SystemExit("ABORT: real TRAIN and real TEST share row ids -- split bug / leakage")
    if rid["val"] & rid["test"]:
        raise SystemExit("ABORT: real VAL and real TEST share row ids -- split bug / leakage")
    if rid["train"] & rid["val"]:
        raise SystemExit("ABORT: real TRAIN and real VAL share row ids -- split bug")
    total_ids = len(rid["train"]) + len(rid["val"]) + len(rid["test"])
    if total_ids != len(df):
        raise SystemExit(f"ABORT: split covers {total_ids} rows but enriched has {len(df)}")
    split["_row_ids"] = rid

    # ---- value-level coincidences (INFORMATIONAL -- not leakage) ----------
    # Distinct NetFlow records can share an identical (IPs + 9 features + Attack)
    # tuple (e.g. flood packets). That is expected and is NOT leakage: the row
    # partition above is exact. We only count it for transparency.
    def _keys(d):
        return set(d[IP_COLS + FLOW_FEATURES + ["Attack"]].astype(str).agg("|".join, axis=1))
    ktr, kte = _keys(split["train"]), _keys(split["test"])
    split["_value_key_overlap_train_test"] = len(ktr & kte)

    if verbose:
        print(f"  real split re-derived & verified (row-level disjoint): "
              f"train={len(split['train']):,}  val={len(split['val']):,}  "
              f"test={len(split['test']):,}")
        print(f"  identical-looking flow tuples shared train<->test "
              f"(expected, not leakage): {split['_value_key_overlap_train_test']:,}")
    _CACHED_SPLIT = split
    return split


def flow_key(df: pd.DataFrame) -> pd.Series:
    return df[IP_COLS + FLOW_FEATURES + ["Attack"]].astype(str).agg("|".join, axis=1)


# ------------------------------------------------------------------
# FINAL SYNTHETIC DATASET  (read-only)
# ------------------------------------------------------------------
def load_final_synth(verbose: bool = True) -> pd.DataFrame:
    df = pd.read_csv(FINAL_SYNTH, low_memory=False)
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()
    for lbl in FORBIDDEN_CLASSES:
        n = int((df["Attack"] == lbl).sum())
        if n:
            raise SystemExit(f"ABORT: {n} '{lbl}' rows in the final synthetic dataset")
    unexpected = sorted(set(df["Attack"].unique()) - set(CLASSES))
    if unexpected:
        raise SystemExit(f"ABORT: unexpected class(es) in final synthetic dataset: {unexpected}")
    missing = [c for c in IP_COLS + FLOW_FEATURES + ["Attack"] if c not in df.columns]
    if missing:
        raise SystemExit(f"ABORT: final synthetic dataset missing columns: {missing}")
    if verbose:
        print(f"  final synthetic dataset: {len(df):,} rows, "
              f"classes {sorted(df['Attack'].unique())}")
    return df


# ------------------------------------------------------------------
# LEAKAGE CHECKS  (hard requirement -- section 34)
# ------------------------------------------------------------------
def assert_no_leakage(train_frames: dict, real_test: pd.DataFrame,
                      split: dict, log_path: str | None = None) -> list[str]:
    """HARD check: no real TEST *row* is used for training.

    train_frames : {name -> DataFrame used for training}. A frame that carries a
                   "_row_id" column (the real frames) is checked at ROW level
                   against the real TEST row ids; a frame without it (pure
                   synthetic) is checked at IP level (synthetic uses reservoir
                   10.x IPs, disjoint from the real hosts).
    split        : the dict from load_real_split() (provides "_row_ids").
    Value-level tuple coincidences are reported but are NOT treated as leakage.
    """
    lines = ["STEP 5 LEAKAGE CHECKS", "=" * 40,
             f"checked_at: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    test_row_ids = set(split["_row_ids"]["test"])
    test_keys = set(flow_key(real_test))
    test_ips = set(real_test["IPV4_SRC_ADDR"]) | set(real_test["IPV4_DST_ADDR"])
    lines.append(f"real TEST: {len(real_test):,} rows  "
                 f"({len(test_row_ids):,} row ids, {len(test_keys):,} distinct flow tuples)")
    lines.append("")
    bad = []
    for name, fr in train_frames.items():
        n = len(fr)
        val_overlap = len(test_keys & set(flow_key(fr)))
        if "_row_id" in fr.columns:
            rid = set(pd.to_numeric(fr["_row_id"], errors="coerce").dropna().astype("int64"))
            row_overlap = len(test_row_ids & rid)
            lines.append(f"  {name:<22}: {n:,} rows | real-TEST ROW ids in training: "
                         f"{row_overlap}  | identical-looking tuples (informational): {val_overlap}")
            if row_overlap:
                bad.append(f"{name}: {row_overlap} real TEST row(s) present in training")
        else:
            ip_overlap = len((set(fr["IPV4_SRC_ADDR"]) | set(fr["IPV4_DST_ADDR"])) & test_ips)
            lines.append(f"  {name:<22}: {n:,} rows | shares {ip_overlap} IP(s) with real TEST "
                         f"(synthetic uses reservoir IPs) | identical tuples: {val_overlap}")
            if ip_overlap:
                bad.append(f"{name}: synthetic frame shares {ip_overlap} IP(s) with real TEST "
                           f"(unexpected -- investigate)")
    lines.append("")
    lines.append(f"train<->test identical-looking flow tuples are EXPECTED for NetFlow")
    lines.append(f"(flood/scan packets repeat) and are NOT leakage: the row partition is exact.")
    lines.append("")
    lines.append("preprocessing rule: every scaler is fit on the relevant TRAINING frame")
    lines.append("only; real TEST is transform-only. Class weights use TRAINING labels only.")
    lines.append("Fixed epochs, no early stopping -> no tuning on held-out real data.")
    if bad:
        lines += ["", "RESULT: FAIL -- " + "; ".join(bad)]
        text = "\n".join(lines) + "\n"
        if log_path:
            atomic_write_text(text, log_path)
        raise SystemExit("ABORT: data leakage detected:\n  " + "\n  ".join(bad))
    lines += ["", "RESULT: PASS -- no real TEST row is used for training."]
    text = "\n".join(lines) + "\n"
    if log_path:
        atomic_write_text(text, log_path)
    print(text)
    return lines


# ------------------------------------------------------------------
# WEIGHTED PER-CLASS GRAPH  (identical construction for real & synthetic --
# mirrors step1_fingerprint.build_weighted_graph)
# ------------------------------------------------------------------
def build_weighted_digraph(sub: pd.DataFrame):
    """One directed edge per ordered (src,dst) IP pair; weight = #flow rows.
    Same rule as Step 1. Returns a networkx.DiGraph."""
    import networkx as nx
    w = (sub.groupby(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
            .size().reset_index(name="weight"))
    G = nx.DiGraph()
    for s, d, wt in zip(w["IPV4_SRC_ADDR"], w["IPV4_DST_ADDR"], w["weight"]):
        if s == d:
            continue  # Step 1 drops self-loops for the synthetic side; keep parity
        G.add_edge(s, d, weight=int(wt))
    return G


def louvain_partition_and_modularity(G):
    """Weighted Louvain on the undirected projection, random_state=LOUVAIN_SEED
    -- exactly Step 1's methodology. Falls back to networkx if python-louvain
    is unavailable. Returns (partition dict, modularity float, n_communities,
    method str)."""
    import networkx as nx
    if G.number_of_nodes() <= 1 or G.number_of_edges() == 0:
        return {}, float("nan"), G.number_of_nodes(), "n/a"
    UG = G.to_undirected()
    try:
        import community as community_louvain
        part = community_louvain.best_partition(UG, weight="weight",
                                                random_state=LOUVAIN_SEED)
        mod = community_louvain.modularity(part, UG, weight="weight")
        method = "python-louvain"
    except Exception:
        communities = nx.community.louvain_communities(UG, weight="weight",
                                                       seed=LOUVAIN_SEED)
        part = {n: i for i, com in enumerate(communities) for n in com}
        mod = nx.community.modularity(UG, communities, weight="weight")
        method = "networkx-louvain"
    n_comm = len(set(part.values()))
    return part, float(mod), n_comm, method
