# ============================================================
# STEP 3 -- SHARED CONFIG + UTILITIES  (CTGAN attack-flow synthesis)
#
# Every Step-3 script imports from here. NOTHING in this file runs the
# pipeline; it only defines paths, constants, checkpoint / status helpers,
# deterministic-seed helpers, the physical-plausibility predicate (copied
# from data_cleaning.py rule 3.2.e so generated flows are held to the same
# standard as the real ones), and a defensive SDV/CTGAN import + API probe.
#
# Design rules enforced here (see the project spec):
#   * Step 3 = ATTACK traffic only. Benign + ransomware are hard-excluded.
#   * Degree categories are FINAL 2-bin (from results/step3/degree_binning_*).
#       src_out_degree <= 11 -> "low"   ; > 11 -> "high"
#       dst_in_degree  <=  5 -> "low"   ; >  5 -> "high"
#     DO NOT re-fit these thresholds and DO NOT introduce a 3rd bin.
#   * MIN_SUPPORT = 10 ; balancing weight cap MAX_WEIGHT = 5.0.
#   * CTGAN never generates IP addresses; IPs come from the Step-2 graph.
#   * Support is computed on TRAIN only; TEST is never used for any
#     modelling decision.
# ============================================================

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.config import CFG, SCALES, active_scale

# ------------------------------------------------------------------
# PATHS  (all relative to the project root == cwd when scripts are run)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.getcwd())

ENRICHED_REAL_FLOWS = "results/step3/step3_1_enriched_real_flows.csv"
DEGREE_BINNING_REPORT = "results/step3/degree_binning_report.txt"

# ------------------------------------------------------------------
# DATASET / GRAPH SCALE
# ------------------------------------------------------------------
# Step 2 produced three synthetic graphs, one per scale:
#     results/step2/same/     (x1    -- same scale as the real data)
#     results/step2/larger/   (x2    -- the LARGER-scale experiment)
#     results/step2/smaller/  (x0.5  -- the SMALLER-scale experiment)
#
# The scale is selected with run_pipeline.py --scale; the whole Step 3 ->
# Step 5 chain then consumes results/step2/<SCALE>/ and writes every
# *generated* artifact under that scale's out_prefix (config.yaml), so no
# scale's results are ever touched by another scale's run.
SCALE = active_scale()

# Every *generated* (scale-dependent) artifact is written under this prefix.
# "same" keeps the original un-prefixed layout; "larger"/"smaller" each get
# their own fully-isolated output root (results/large_scale, results/small_scale).
SCALE_OUT_PREFIX = {s: CFG["scales"][s]["out_prefix"] for s in SCALES}
OUT_PREFIX = SCALE_OUT_PREFIX[SCALE]

STEP2_SCALE = SCALE                        # Step 3 consumes the SCALE graph
STEP2_DIR = os.path.join("results", "step2", STEP2_SCALE)
STEP2_EDGES = os.path.join(STEP2_DIR, "synthetic_edge_list.csv")
STEP2_NODES = os.path.join(STEP2_DIR, "synthetic_nodes.csv")
STEP2_SUMMARY = os.path.join(STEP2_DIR, "synthetic_summary.csv")

# ------------------------------------------------------------------
# OUTPUT LAYOUT
# ------------------------------------------------------------------
# Scale-INDEPENDENT Step-3 assets: the trained CTGAN models and the real-data
# splits / support tables / balanced frames they were trained on. These do NOT
# depend on the Step-2 graph scale -- they are REUSED verbatim for every scale
# (the larger-scale experiment changes the graph, not the flow model).
S3 = "results/step3_ctgan"
S3_MODELS      = os.path.join(S3, "models")
S3_SUPPORT     = os.path.join(S3, "support")
S3_BALANCING   = os.path.join(S3, "balancing")
S3_SPLITS      = os.path.join(S3, "splits")

# Scale-DEPENDENT Step-3 outputs: everything derived from the Step-2 graph
# (generation, its logs, its reports). Each non-"same" scale gets its own
# isolated output root (results/large_scale/ or results/small_scale/) so the
# same-scale deliverables -- and every other scale's deliverables -- are
# preserved untouched.
S3_OUT_ROOT = os.path.join(OUT_PREFIX, "step3_ctgan") if OUT_PREFIX else S3
S3_GENERATION  = os.path.join(S3_OUT_ROOT, "generation", STEP2_SCALE)
S3_LOGS        = os.path.join(S3_OUT_ROOT, "logs")
S3_REPORTS     = os.path.join(S3_OUT_ROOT, "reports")

STATUS_JSON = os.path.join(S3_LOGS, "step3_status.json")
GEN_LOG_CSV = os.path.join(S3_LOGS, "generation_log.csv")

ALL_S3_DIRS = [S3, S3_MODELS, S3_SUPPORT, S3_BALANCING, S3_SPLITS,
               S3_OUT_ROOT, S3_GENERATION, S3_LOGS, S3_REPORTS]


def assert_scale_inputs() -> None:
    """Fail fast (with a clear message) if the Step-2 graph for the selected
    SCALE is not on disk. Call this from any script that consumes the graph."""
    if os.path.basename(STEP2_DIR) != SCALE:
        raise SystemExit(f"ABORT: STEP2_DIR {STEP2_DIR!r} does not match SCALE {SCALE!r}")
    for p in (STEP2_EDGES, STEP2_NODES):
        if not os.path.exists(p):
            raise SystemExit(
                f"ABORT: Step-2 '{SCALE}'-scale graph missing: {p}\n"
                f"       Expected the '{SCALE}'-scale graph in results/step2/{SCALE}/. "
                f"Do NOT regenerate Step 2 -- restore the file."
            )

# ------------------------------------------------------------------
# SCOPE
# ------------------------------------------------------------------
# Step-1/Step-2 store labels lower-case. Keep that convention everywhere.
ATTACKS_STEP3 = ["ddos", "dos", "scanning", "password",
                 "backdoor", "injection", "xss", "mitm"]
EXCLUDED_CLASSES = ["benign", "ransomware"]

# ------------------------------------------------------------------
# DEGREE CATEGORIES  -- FINAL, do not change (results/step3/degree_binning_report.txt)
# ------------------------------------------------------------------
SRC_OUT_DEGREE_LOW_MAX = 11      # <= 11 -> low   ; > 11 -> high   (global Q50)
DST_IN_DEGREE_LOW_MAX = 5        # <=  5 -> low   ; >  5 -> high   (global Q50)
DEGREE_CATEGORIES = ("low", "high")
ROLE_CATEGORIES = ("low", "medium", "high")


def bin_src_out_degree(value) -> str:
    """Apply the FINAL Step-3.1 threshold to a source out-degree."""
    return "low" if float(value) <= SRC_OUT_DEGREE_LOW_MAX else "high"


def bin_dst_in_degree(value) -> str:
    """Apply the FINAL Step-3.1 threshold to a destination in-degree."""
    return "low" if float(value) <= DST_IN_DEGREE_LOW_MAX else "high"


# ------------------------------------------------------------------
# MODELLING CONSTANTS
# ------------------------------------------------------------------
RANDOM_SEED = CFG["seeds"]["random"]                     # 42
SPLIT_SEED = CFG["seeds"]["split"]                       # 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = (CFG["split"]["train"], CFG["split"]["val"],
                                   CFG["split"]["test"])  # 0.80 / 0.10 / 0.10

MIN_SUPPORT = CFG["balancing"]["min_support"]   # a condition is "supported" iff train rows >= 10
MAX_WEIGHT = CFG["balancing"]["max_weight"]     # balancing: max oversampling factor (5.0)
MIN_WEIGHT = 1.0 / MAX_WEIGHT    # ... and the symmetric moderate-undersampling floor

# CTGAN targets -- the ONLY columns the models generate. Identical for every
# variant (fair ablation). L7_PROTO, FEAT_0..19 and Label are NOT used in Step 3.
TARGET_FEATURES = [
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "PROTOCOL", "L4_SRC_PORT", "L4_DST_PORT", "TCP_FLAGS",
]
# discrete-valued targets -> force categorical in the CTGAN metadata
CATEGORICAL_TARGETS = ["PROTOCOL", "TCP_FLAGS"]

# non-negative integer targets (used by the physical-plausibility predicate
# and for post-sampling rounding sanity)
COUNT_TARGETS = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]
PORT_TARGETS = ["L4_SRC_PORT", "L4_DST_PORT"]

# Four variants -- only the conditioning information differs.
VARIANT_CONDITIONS = {
    "A": ["Attack"],
    "B": ["Attack", "src_out_degree_cat", "dst_in_degree_cat"],
    "C": ["Attack", "src_role", "dst_role"],
    "D": ["Attack", "src_out_degree_cat", "dst_in_degree_cat",
          "src_role", "dst_role"],
}
VARIANTS = ["A", "B", "C", "D"]
VARIANT_TITLE = {
    "A": "BASELINE (Attack only)",
    "B": "DEGREE (Attack + degree categories)",
    "C": "PAGERANK ROLE (Attack + roles)",
    "D": "COMPLETE PROPOSED MODEL (Attack + degree categories + roles)",
}

# back-off ladder: level -> (variant, condition columns minus 'Attack' shown for docs)
BACKOFF_LADDER = [
    (0, "D", ["Attack", "src_out_degree_cat", "dst_in_degree_cat", "src_role", "dst_role"]),
    (1, "B", ["Attack", "src_out_degree_cat", "dst_in_degree_cat"]),
    (2, "C", ["Attack", "src_role", "dst_role"]),
    (3, "A", ["Attack"]),
]

# CTGAN hyper-parameters -- same for every variant. Anything not accepted by
# the installed CTGANSynthesizer signature is dropped at call time (logged).
CTGAN_CONFIG = {
    "epochs": CFG["ctgan"]["epochs"],                               # 300
    "batch_size": CFG["ctgan"]["batch_size"],                       # 500
    "verbose": CFG["ctgan"]["verbose_attack"],                      # True
    "enforce_rounding": CFG["ctgan"]["enforce_rounding"],           # True
    "enforce_min_max_values": CFG["ctgan"]["enforce_min_max_values"],  # True
}

# generation batching / crash-recovery granularity
EDGES_PER_BATCH = CFG["generation"]["attack"]["edges_per_batch"]        # flush a batch every N synthetic edges (25) ...
MAX_ROWS_PER_BATCH = CFG["generation"]["attack"]["max_rows_per_batch"]  # ... or sooner if it would exceed this many rows (100k)
RESAMPLE_ROUNDS = CFG["generation"]["resample_rounds"]                  # per-batch re-draw rounds for physically-invalid rows (6)
SAMPLE_MAX_TRIES_PER_BATCH = CFG["generation"]["sample_max_tries_per_batch"]  # 500

FORCE_RERUN = os.environ.get("STEP3_FORCE_RERUN", "0").strip().lower() in ("1", "true", "yes")


# ------------------------------------------------------------------
# FILE-SYSTEM / CHECKPOINT HELPERS
# ------------------------------------------------------------------
def ensure_dirs(*extra):
    for d in list(ALL_S3_DIRS) + list(extra):
        os.makedirs(d, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: str, **kw):
    """Write to <path>.tmp then os.replace -- a crash mid-write never leaves a
    half-file that a later run would mistake for complete."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json(obj, path: str):
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


def done_path(marker_dir: str, name: str = "DONE") -> str:
    return os.path.join(marker_dir, name)


def write_done(marker_dir: str, payload: dict | None = None, name: str = "DONE"):
    """Write a DONE marker -- ALWAYS the last action after generate+validate+save."""
    os.makedirs(marker_dir, exist_ok=True)
    payload = dict(payload or {})
    payload.setdefault("completed_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(payload, done_path(marker_dir, name))


def is_done(marker_dir: str, name: str = "DONE") -> bool:
    return os.path.exists(done_path(marker_dir, name))


def component_complete(directory: str, required_files: list[str]) -> bool:
    """A component is COMPLETE only when: dir exists, DONE marker exists, and
    every expected output file exists and is non-empty and readable."""
    if FORCE_RERUN:
        return False
    if not os.path.isdir(directory) or not is_done(directory):
        return False
    for rel in required_files:
        p = os.path.join(directory, rel)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
        try:
            with open(p, "rb") as fh:
                fh.read(1)
        except OSError:
            return False
    return True


# ------------------------------------------------------------------
# STATUS FILE
# ------------------------------------------------------------------
def update_status(key: str, value):
    ensure_dirs()
    st = read_json(STATUS_JSON, {})
    st.setdefault("environment", detect_environment())
    st[key] = value
    st["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(st, STATUS_JSON)


def get_status(key: str, default=None):
    return read_json(STATUS_JSON, {}).get(key, default)


# ------------------------------------------------------------------
# DETERMINISM
# ------------------------------------------------------------------
def stable_hash(s) -> int:
    """Process-independent hash (Python's built-in hash() is salted per run and
    must never be used for anything that has to reproduce across sessions)."""
    return int.from_bytes(
        hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest(), "big")


def stable_seed(base: int, *parts) -> int:
    return (int(base) + stable_hash("|".join(map(str, parts)))) % (2 ** 32 - 1)


def set_global_seeds(seed: int = RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # keep cuDNN deterministic where possible (best-effort)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


# ------------------------------------------------------------------
# PHYSICAL PLAUSIBILITY  -- mirrors data_cleaning.py rule 3.2.e
# ------------------------------------------------------------------
def physical_validity_mask(df: pd.DataFrame) -> pd.Series:
    """True where a generated flow row is physically possible, using the SAME
    rules the real data was cleaned with (data_cleaning.py 3.2.e):
      * finite numeric cells
      * ports in 0..65535
      * bytes / packets / duration >= 0
      * PROTOCOL in 0..255
      * total packets > 0 and total bytes > 0
      * IN_BYTES >= IN_PKTS and OUT_BYTES >= OUT_PKTS
    """
    num = df[TARGET_FEATURES].apply(pd.to_numeric, errors="coerce")
    ok = np.isfinite(num.to_numpy(dtype="float64")).all(axis=1)

    for c in PORT_TARGETS:
        ok &= (num[c] >= 0).to_numpy() & (num[c] <= 65535).to_numpy()

    for c in COUNT_TARGETS + ["FLOW_DURATION_MILLISECONDS"]:
        ok &= (num[c] >= 0).to_numpy()

    ok &= (num["PROTOCOL"] >= 0).to_numpy() & (num["PROTOCOL"] <= 255).to_numpy()
    ok &= ((num["IN_PKTS"] + num["OUT_PKTS"]) > 0).to_numpy()
    ok &= ((num["IN_BYTES"] + num["OUT_BYTES"]) > 0).to_numpy()
    ok &= (num["IN_BYTES"] >= num["IN_PKTS"]).to_numpy()
    ok &= (num["OUT_BYTES"] >= num["OUT_PKTS"]).to_numpy()
    return pd.Series(ok, index=df.index)


# ------------------------------------------------------------------
# ENVIRONMENT PROBE
# ------------------------------------------------------------------
def _try_version(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "not installed"


def detect_environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": _try_version("numpy"),
        "pandas": _try_version("pandas"),
        "scipy": _try_version("scipy"),
        "networkx": _try_version("networkx"),
        "sklearn": _try_version("sklearn"),
        "sdv": _try_version("sdv"),
        "ctgan": _try_version("ctgan"),
        "torch": _try_version("torch"),
        "gpu_available": False,
        "gpu_name": "none",
        "cuda": "none",
    }
    try:
        import torch
        env["gpu_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["cuda"] = getattr(torch.version, "cuda", "unknown")
    except Exception:
        pass
    return env


def write_environment_report():
    ensure_dirs()
    env = detect_environment()
    lines = ["STEP 3 ENVIRONMENT", "=" * 40]
    for k, v in env.items():
        lines.append(f"{k:<16}: {v}")
    atomic_write_text("\n".join(lines) + "\n",
                      os.path.join(S3_REPORTS, "environment.txt"))
    return env


# ------------------------------------------------------------------
# SDV / CTGAN  -- defensive import + API probe (never invent parameters)
# ------------------------------------------------------------------
class SDVApi:
    """Thin adapter over whatever SDV version is installed. Only the pieces
    Step 3 needs, resolved once against the real installed API."""

    def __init__(self):
        import sdv
        self.sdv_version = getattr(sdv, "__version__", "unknown")
        major = int(str(self.sdv_version).split(".")[0]) if self.sdv_version[0].isdigit() else 1
        if major < 1:
            raise RuntimeError(
                f"SDV {self.sdv_version} is pre-1.0 and uses an incompatible API. "
                f"Install sdv>=1.0 (Colab: pip install -U sdv)."
            )
        from sdv.single_table import CTGANSynthesizer
        self.CTGANSynthesizer = CTGANSynthesizer

        # metadata class -- prefer SingleTableMetadata, fall back to unified Metadata
        self._single_table_meta = True
        try:
            from sdv.metadata import SingleTableMetadata
            self.MetadataClass = SingleTableMetadata
        except Exception:
            from sdv.metadata import Metadata
            self.MetadataClass = Metadata
            self._single_table_meta = False

        from sdv.sampling import Condition
        self.Condition = Condition

        import inspect
        self._ctgan_init_params = set(
            inspect.signature(CTGANSynthesizer.__init__).parameters.keys()
        )
        self._sample_cond_params = set(
            inspect.signature(CTGANSynthesizer.sample_from_conditions).parameters.keys()
        )

    # -- metadata -------------------------------------------------
    def build_metadata(self, df: pd.DataFrame, categorical_cols: list[str]):
        md = self.MetadataClass()
        # detect_from_dataframe is an instance method on SingleTableMetadata and
        # a classmethod on the unified Metadata -- handle both.
        try:
            md.detect_from_dataframe(df)
        except TypeError:
            md = self.MetadataClass.detect_from_dataframe(df)

        for col in categorical_cols:
            if col not in df.columns:
                continue
            try:
                md.update_column(col, sdtype="categorical")
            except TypeError:
                # unified Metadata wants table_name
                tbl = list(md.tables.keys())[0] if getattr(md, "tables", None) else "table"
                md.update_column(column_name=col, sdtype="categorical", table_name=tbl)
            except Exception:
                pass
        return md

    # -- synthesizer construction ------------------------------
    def filter_ctgan_kwargs(self, cfg: dict, want_cuda: bool):
        kept, dropped = {}, {}
        candidate = dict(cfg)
        # cuda toggle name differs across versions
        for name in ("cuda", "use_cuda", "device"):
            if name in self._ctgan_init_params:
                candidate[name] = want_cuda if name != "device" else ("cuda" if want_cuda else "cpu")
                break
        for k, v in candidate.items():
            if k in self._ctgan_init_params:
                kept[k] = v
            else:
                dropped[k] = v
        return kept, dropped

    def make_synthesizer(self, metadata, cfg: dict, want_cuda: bool):
        kept, dropped = self.filter_ctgan_kwargs(cfg, want_cuda)
        synth = self.CTGANSynthesizer(metadata, **kept)
        return synth, kept, dropped

    # -- conditional sampling ---------------------------------
    def sample_from_conditions(self, synth, conditions: list):
        kw = {}
        if "batch_size" in self._sample_cond_params:
            kw["batch_size"] = 10_000
        if "max_tries_per_batch" in self._sample_cond_params:
            kw["max_tries_per_batch"] = SAMPLE_MAX_TRIES_PER_BATCH
        try:
            return synth.sample_from_conditions(conditions=conditions, **kw)
        except TypeError:
            return synth.sample_from_conditions(conditions)


_SDV_API = None


def get_sdv_api() -> "SDVApi":
    global _SDV_API
    if _SDV_API is None:
        _SDV_API = SDVApi()
    return _SDV_API


# ------------------------------------------------------------------
# small logging helper
# ------------------------------------------------------------------
@contextmanager
def stage_banner(name: str):
    bar = "=" * 70
    print(f"\n{bar}\n[STEP 3] {name}\n{bar}")
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[STEP 3] {name} -- finished in {time.time() - t0:,.1f}s")


def load_real_attack_frame(usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the Step-3.1 enriched real flows, normalise the label, and hard-drop
    every non-Step-3 class. Aborts if benign / ransomware survive the filter."""
    need = set(TARGET_FEATURES) | {"Attack", "src_out_degree", "dst_in_degree",
                                   "src_out_degree_cat", "dst_in_degree_cat",
                                   "src_role", "dst_role"}
    if usecols is not None:
        need = set(usecols)
    df = pd.read_csv(ENRICHED_REAL_FLOWS, usecols=sorted(need), low_memory=False)
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[df["Attack"].isin(ATTACKS_STEP3)].copy()
    print(f"  filtered attack rows: {before:,} -> {len(df):,} "
          f"(kept {sorted(df['Attack'].unique())})")

    benign_ct = int((df["Attack"] == "benign").sum())
    ransom_ct = int((df["Attack"] == "ransomware").sum())
    assert benign_ct == 0, f"benign leaked into Step-3 data ({benign_ct} rows) -- ABORT"
    assert ransom_ct == 0, f"ransomware leaked into Step-3 data ({ransom_ct} rows) -- ABORT"

    unexpected = sorted(set(df["Attack"].unique()) - set(ATTACKS_STEP3))
    if unexpected:
        raise SystemExit(f"ABORT: unexpected attack class(es) in Step-3 data: {unexpected}")
    return df
