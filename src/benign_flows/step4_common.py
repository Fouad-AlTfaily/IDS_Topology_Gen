# ============================================================
# STEP 4 -- SHARED CONFIG + UTILITIES  (CTGAN *benign* flow synthesis)
#
# Step 4 mirrors Step 3 but for BENIGN traffic only. It re-uses the Step-3
# infrastructure wherever possible (step3_common.SDVApi, the physical-validity
# predicate, atomic writes, deterministic-seed helpers, the FINAL degree bins).
#
# NOTHING in this file runs the pipeline. It only defines paths, constants and
# checkpoint helpers.
#
# Design rules enforced here (see the project spec, "STEP 4"):
#   * Step 4 = BENIGN traffic only.  Every attack class AND ransomware are
#     hard-excluded with assertions (see load_real_benign_frame).
#   * Benign is a single class, so Attack is NEVER a CTGAN conditioning
#     variable for Step-4 models.  The four variants differ only by the
#     STRUCTURAL conditioning they use:
#         A  none                       (baseline)
#         B  src_out_degree_cat + dst_in_degree_cat
#         C  src_role + dst_role
#         D  B + C                      (complete / default generation model)
#   * Degree categories are the FINAL 2-bin scheme from Step 3
#     (step3_common.bin_src_out_degree / bin_dst_in_degree). DO NOT re-fit
#     the thresholds and DO NOT add a 3rd bin.
#   * MIN_SUPPORT = 10 ; balancing weight cap MAX_WEIGHT = 5.0  (== Step 3).
#   * CTGAN never generates IP addresses; IPs come from the Step-2 *benign*
#     graph (results/step2/<SCALE>/, attack == "benign").
#   * Support + balancing use the TRAIN split ONLY. TEST is never used for any
#     modelling / balancing / back-off / generation decision.
#   * Graph scale follows step3_common.SCALE (run_pipeline.py --scale). The
#     trained benign CTGAN models are REUSED as-is -- a different scale
#     changes the graph, not the flow model.
# ============================================================

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C3  # re-use Step-3 infrastructure (SDVApi, seeds, IO, ...)
from src.config import CFG

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.getcwd())

# Step-4 input: the SAME enriched real-flow table Step 3 used.
ENRICHED_REAL_FLOWS = C3.ENRICHED_REAL_FLOWS               # results/step3/step3_1_enriched_real_flows.csv

# Step-2 synthetic BENIGN graph -- follows the SCALE chosen in step3_common
# (results/step2/<SCALE>/).
SCALE = C3.SCALE
STEP2_SCALE = C3.STEP2_SCALE
STEP2_DIR = C3.STEP2_DIR
STEP2_EDGES = C3.STEP2_EDGES                                # attack,src,dst,weight
STEP2_NODES = C3.STEP2_NODES                                # attack,ip,in_degree,out_degree,role
assert_scale_inputs = C3.assert_scale_inputs

# Step-3 attack deliverable (consumed read-only by the final merge) -- same SCALE.
STEP3_DELIVERABLE = os.path.join(C3.S3_GENERATION, "synthetic_attack_flows.csv")
STEP3_VALIDATION_DONE = os.path.join(C3.S3_GENERATION, "VALIDATION_DONE")

# Scale-INDEPENDENT Step-4 assets: the trained benign CTGAN models + the
# real-benign splits / support / balanced frames they were trained on. REUSED
# verbatim for every scale (never retrained when the graph scale changes).
S4 = "results/step4_benign"
S4_MODELS     = os.path.join(S4, "models")
S4_SUPPORT    = os.path.join(S4, "support")
S4_BALANCING  = os.path.join(S4, "balancing")
S4_SPLITS     = os.path.join(S4, "splits")

# Scale-DEPENDENT Step-4 outputs: benign generation (+ logs/reports) and the
# final merged dataset. Each non-"same" scale gets its own isolated output
# root (results/large_scale/ or results/small_scale/) so the same-scale
# deliverables -- and every other scale's deliverables -- are preserved
# untouched.
S4_OUT_ROOT   = os.path.join(C3.OUT_PREFIX, "step4_benign") if C3.OUT_PREFIX else S4
S4_GENERATION = os.path.join(S4_OUT_ROOT, "generation")
S4_GEN_BATCHES = os.path.join(S4_GENERATION, "batches")
S4_LOGS       = os.path.join(S4_OUT_ROOT, "logs")
S4_REPORTS    = os.path.join(S4_OUT_ROOT, "reports")

STATUS_JSON = os.path.join(S4_LOGS, "step4_status.json")
GEN_LOG_CSV = os.path.join(S4_LOGS, "generation_log.csv")

SYNTHETIC_BENIGN_FLOWS = os.path.join(S4_GENERATION, "synthetic_benign_flows.csv")

# Final merged dataset (Step 3 attacks + Step 4 benign) for THIS scale.
FINAL_DIR = (os.path.join(C3.OUT_PREFIX, "final_synthetic") if C3.OUT_PREFIX
             else "results/final_synthetic")
FINAL_CSV = os.path.join(FINAL_DIR, "synthetic_NF_ToN_IoT.csv")

ALL_S4_DIRS = [S4, S4_MODELS, S4_SUPPORT, S4_BALANCING, S4_SPLITS,
               S4_OUT_ROOT, S4_GENERATION, S4_GEN_BATCHES, S4_LOGS, S4_REPORTS]

# ------------------------------------------------------------------
# SCOPE
# ------------------------------------------------------------------
# Step-1/Step-2 store labels lower-case; keep that convention.
BENIGN_LABEL = "benign"
# Every label that MUST NOT appear anywhere in Step 4.
FORBIDDEN_LABELS = list(C3.ATTACKS_STEP3) + ["ransomware"]

# ------------------------------------------------------------------
# DEGREE CATEGORIES  -- FINAL, taken verbatim from Step 3 (do not change)
# ------------------------------------------------------------------
SRC_OUT_DEGREE_LOW_MAX = C3.SRC_OUT_DEGREE_LOW_MAX   # 11 : <= 11 -> low ; > 11 -> high
DST_IN_DEGREE_LOW_MAX = C3.DST_IN_DEGREE_LOW_MAX     #  5 : <=  5 -> low ; >  5 -> high
DEGREE_CATEGORIES = C3.DEGREE_CATEGORIES             # ("low", "high")
ROLE_CATEGORIES = C3.ROLE_CATEGORIES                 # ("low", "medium", "high")

bin_src_out_degree = C3.bin_src_out_degree
bin_dst_in_degree = C3.bin_dst_in_degree

# ------------------------------------------------------------------
# MODELLING CONSTANTS  (kept identical to Step 3 for a fair ablation)
# ------------------------------------------------------------------
RANDOM_SEED = C3.RANDOM_SEED         # 42
SPLIT_SEED = C3.SPLIT_SEED           # 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = C3.TRAIN_FRAC, C3.VAL_FRAC, C3.TEST_FRAC  # 0.80 / 0.10 / 0.10

MIN_SUPPORT = C3.MIN_SUPPORT         # 10
MAX_WEIGHT = C3.MAX_WEIGHT           # 5.0
MIN_WEIGHT = C3.MIN_WEIGHT           # 0.2

# The 9 flow features every benign variant generates -- identical to Step 3.
TARGET_FEATURES = list(C3.TARGET_FEATURES)
CATEGORICAL_TARGETS = list(C3.CATEGORICAL_TARGETS)   # PROTOCOL, TCP_FLAGS
COUNT_TARGETS = list(C3.COUNT_TARGETS)
PORT_TARGETS = list(C3.PORT_TARGETS)

# Four benign variants -- ONLY the structural conditioning differs.
# (No "Attack" column: benign is a single class.)
VARIANT_CONDITIONS = {
    "A": [],
    "B": ["src_out_degree_cat", "dst_in_degree_cat"],
    "C": ["src_role", "dst_role"],
    "D": ["src_out_degree_cat", "dst_in_degree_cat", "src_role", "dst_role"],
}
VARIANTS = ["A", "B", "C", "D"]
VARIANT_TITLE = {
    "A": "BENIGN BASELINE (no conditioning)",
    "B": "BENIGN DEGREE (src_out_degree_cat + dst_in_degree_cat)",
    "C": "BENIGN ROLE (src_role + dst_role)",
    "D": "BENIGN COMPLETE (degree categories + roles)",
}

# Back-off ladder for benign generation (spec section 18):
#   level 0  D  degree + role
#   level 1  B  degree
#   level 2  C  role
#   level 3  A  baseline
BACKOFF_LADDER = [
    (0, "D", ["src_out_degree_cat", "dst_in_degree_cat", "src_role", "dst_role"]),
    (1, "B", ["src_out_degree_cat", "dst_in_degree_cat"]),
    (2, "C", ["src_role", "dst_role"]),
    (3, "A", []),
]

# CTGAN hyper-parameters -- SAME epochs / batch size / seed policy for every
# variant.  verbose is False on purpose: step4_progress.py renders its own
# live epoch bar driven by the synthesizer's real per-epoch loss history
# (synth._model.loss_values). Anything the installed CTGANSynthesizer does not
# accept is dropped at call time by step3_common.SDVApi.filter_ctgan_kwargs.
CTGAN_CONFIG = {
    "epochs": CFG["ctgan"]["epochs"],                               # 300
    "batch_size": CFG["ctgan"]["batch_size"],                       # 500
    "verbose": CFG["ctgan"]["verbose_benign"],                      # False
    "enforce_rounding": CFG["ctgan"]["enforce_rounding"],           # True
    "enforce_min_max_values": CFG["ctgan"]["enforce_min_max_values"],  # True
}

# generation batching / crash-recovery granularity
EDGES_PER_BATCH = CFG["generation"]["benign"]["edges_per_batch"]        # 200
MAX_ROWS_PER_BATCH = CFG["generation"]["benign"]["max_rows_per_batch"]  # 50k
RESAMPLE_ROUNDS = CFG["generation"]["resample_rounds"]                  # 6
SAMPLE_MAX_TRIES_PER_BATCH = C3.SAMPLE_MAX_TRIES_PER_BATCH

FORCE_RERUN = os.environ.get("STEP4_FORCE_RERUN", "0").strip().lower() in ("1", "true", "yes")

# ------------------------------------------------------------------
# FILE-SYSTEM / CHECKPOINT HELPERS  (thin wrappers over step3_common)
# ------------------------------------------------------------------
atomic_write_csv = C3.atomic_write_csv
atomic_write_text = C3.atomic_write_text
write_json = C3.write_json
read_json = C3.read_json
write_done = C3.write_done
is_done = C3.is_done
done_path = C3.done_path
stable_hash = C3.stable_hash
stable_seed = C3.stable_seed
set_global_seeds = C3.set_global_seeds
detect_environment = C3.detect_environment
physical_validity_mask = C3.physical_validity_mask
get_sdv_api = C3.get_sdv_api


@contextmanager
def stage_banner(name: str):
    bar = "=" * 70
    print(f"\n{bar}\n[STEP 4] {name}\n{bar}")
    t0 = time.time()
    try:
        yield
    finally:
        print(f"[STEP 4] {name} -- finished in {time.time() - t0:,.1f}s")


def ensure_dirs(*extra):
    for d in list(ALL_S4_DIRS) + list(extra):
        os.makedirs(d, exist_ok=True)


def component_complete(directory: str, required_files: list[str]) -> bool:
    """COMPLETE iff: directory exists, a DONE marker exists, and every expected
    output file exists, is non-empty and is readable.  Honours STEP4_FORCE_RERUN."""
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


def write_environment_report() -> dict:
    ensure_dirs()
    env = detect_environment()
    lines = ["STEP 4 ENVIRONMENT", "=" * 40]
    for k, v in env.items():
        lines.append(f"{k:<16}: {v}")
    atomic_write_text("\n".join(lines) + "\n", os.path.join(S4_REPORTS, "environment.txt"))
    return env


# ------------------------------------------------------------------
# FINAL degree-category consistency check  (mirrors step3_2_prepare._verify_degree_cats)
# ------------------------------------------------------------------
def verify_degree_categories(df: pd.DataFrame) -> None:
    """Assert the enriched *_cat columns already match the FINAL Step-3 thresholds.
    We never recompute or overwrite them -- a corrupted upstream file is caught
    here rather than silently trained on."""
    exp_src = np.where(df["src_out_degree"].to_numpy(dtype=float)
                       <= SRC_OUT_DEGREE_LOW_MAX, "low", "high")
    exp_dst = np.where(df["dst_in_degree"].to_numpy(dtype=float)
                       <= DST_IN_DEGREE_LOW_MAX, "low", "high")
    bad_src = int((df["src_out_degree_cat"].astype(str).to_numpy() != exp_src).sum())
    bad_dst = int((df["dst_in_degree_cat"].astype(str).to_numpy() != exp_dst).sum())
    if bad_src or bad_dst:
        raise SystemExit(
            f"ABORT: enriched degree categories do not match the FINAL Step-3 thresholds "
            f"(src mismatches={bad_src}, dst mismatches={bad_dst}). Re-run Step 3.1."
        )
    assert set(df["src_out_degree_cat"].unique()) <= set(DEGREE_CATEGORIES)
    assert set(df["dst_in_degree_cat"].unique()) <= set(DEGREE_CATEGORIES)
    assert set(df["src_role"].unique()) <= set(ROLE_CATEGORIES)
    assert set(df["dst_role"].unique()) <= set(ROLE_CATEGORIES)


# ------------------------------------------------------------------
# LOAD REAL BENIGN FLOWS  (attack + ransomware hard-excluded with assertions)
# ------------------------------------------------------------------
BENIGN_KEEP_COLS = (TARGET_FEATURES
                    + ["Attack", "Label",
                       "src_out_degree", "dst_in_degree",
                       "src_out_degree_cat", "dst_in_degree_cat",
                       "src_role", "dst_role"])


def load_real_benign_frame(usecols: list[str] | None = None) -> pd.DataFrame:
    """Load the Step-3.1 enriched real flows, normalise the label, and keep ONLY
    the canonical benign class.  Aborts if any attack / ransomware row survives."""
    need = sorted(set(usecols if usecols is not None else BENIGN_KEEP_COLS))
    df = pd.read_csv(ENRICHED_REAL_FLOWS, usecols=need, low_memory=False)
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()

    before = len(df)
    df = df[df["Attack"] == BENIGN_LABEL].copy()
    print(f"  benign filter: {before:,} -> {len(df):,} rows "
          f"(kept {sorted(df['Attack'].unique())})")

    # ---- HARD scope assertions -------------------------------------------
    for lbl in FORBIDDEN_LABELS:
        n = int((df["Attack"] == lbl).sum())
        assert n == 0, f"'{lbl}' leaked into Step-4 benign data ({n} rows) -- ABORT"
    unexpected = sorted(set(df["Attack"].unique()) - {BENIGN_LABEL})
    if unexpected:
        raise SystemExit(f"ABORT: unexpected class(es) in Step-4 benign data: {unexpected}")
    if "Label" in df.columns:
        bad_label = int((pd.to_numeric(df["Label"], errors="coerce") != 0).sum())
        assert bad_label == 0, (
            f"ABORT: {bad_label} benign rows carry a non-zero binary Label "
            f"-- canonical benign must be Label==0"
        )
    return df.reset_index(drop=True)
