# ============================================================
# STEP 4.1 -- BENIGN DATA PREPARATION  (benign-only filter + 80/10/10 split)
#
# Reads results/step3/step3_1_enriched_real_flows.csv, keeps ONLY the canonical
# benign class (every attack class AND ransomware are hard-excluded with
# assertions), keeps the 9 CTGAN target features + the raw degree columns + the
# FINAL degree categories + the roles, and produces a deterministic 80/10/10
# split with a fixed seed.
#
# Benign is a single class, so no stratification by Attack is needed.
#
# TEST is written out but MUST NOT be used for support analysis, balancing,
# back-off or training -- only the final held-out check.
#
# Outputs (results/step4_benign/):
#   splits/train.csv  splits/val.csv  splits/test.csv  splits/DONE
#   reports/data_split_statistics.csv
#
# Idempotent: re-running with a valid splits/DONE is a no-op unless
# STEP4_FORCE_RERUN=1.
# ============================================================

import os

import numpy as np
import pandas as pd

from src.benign_flows import step4_common as S4

REQUIRED = ["train.csv", "val.csv", "test.csv"]

# raw degree columns are PRESERVED in the splits but are NEVER used as CTGAN
# conditions (only the *_cat versions are).
KEEP_COLS = (S4.TARGET_FEATURES
             + ["Attack", "Label",
                "src_out_degree", "dst_in_degree",
                "src_out_degree_cat", "dst_in_degree_cat",
                "src_role", "dst_role"])

CAT_COLS = ["src_out_degree_cat", "dst_in_degree_cat", "src_role", "dst_role"]


def benign_split(df: pd.DataFrame, seed: int):
    """Deterministic single-class shuffle then 80/10/10 slice."""
    idx = df.index.to_numpy()
    rng = np.random.default_rng(S4.stable_seed(seed, S4.BENIGN_LABEL))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(round(S4.TRAIN_FRAC * n))
    n_va = int(round(S4.VAL_FRAC * n))
    n_tr = min(max(n_tr, 1), n)
    n_va = min(n_va, n - n_tr)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    # keep original file order within each split
    return (df.loc[np.sort(tr)].copy(),
            df.loc[np.sort(va)].copy(),
            df.loc[np.sort(te)].copy())


def _category_counts(df: pd.DataFrame, split_name: str, total: int) -> dict:
    row = {"split": split_name, "rows": len(df),
           "fraction": round(len(df) / max(total, 1), 6)}
    for c in CAT_COLS:
        vc = df[c].astype(str).value_counts().to_dict()
        levels = (S4.DEGREE_CATEGORIES if c.endswith("_cat") else S4.ROLE_CATEGORIES)
        for lv in levels:
            row[f"{c}={lv}"] = int(vc.get(lv, 0))
    return row


def _write_split_statistics(df, train_df, val_df, test_df):
    n_total = len(df)
    stats = pd.DataFrame([
        _category_counts(df, "ALL", n_total),
        _category_counts(train_df, "train", n_total),
        _category_counts(val_df, "val", n_total),
        _category_counts(test_df, "test", n_total),
    ])
    S4.atomic_write_csv(stats, os.path.join(S4.S4_REPORTS, "data_split_statistics.csv"))
    return stats


def _ensure_stats_from_existing_splits():
    """Skip-path helper: the shared benign splits/ is already complete, but this
    scale's reports/ (e.g. results/small_scale/... for the smaller scale) may not
    carry data_split_statistics.csv yet. Rebuild it from the existing split files
    without rewriting a single split row (Step 5 verifies the benign split against
    this CSV)."""
    dss = os.path.join(S4.S4_REPORTS, "data_split_statistics.csv")
    if os.path.exists(dss) and os.path.getsize(dss) > 0:
        return
    df = S4.load_real_benign_frame(usecols=KEEP_COLS)
    df = df[KEEP_COLS].copy()
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()
    train_df = pd.read_csv(os.path.join(S4.S4_SPLITS, "train.csv"), low_memory=False)
    val_df = pd.read_csv(os.path.join(S4.S4_SPLITS, "val.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(S4.S4_SPLITS, "test.csv"), low_memory=False)
    _write_split_statistics(df, train_df, val_df, test_df)
    print(f"[STEP 4.1] (re)wrote benign split statistics under {S4.S4_REPORTS}")


def main():
    S4.ensure_dirs()
    S4.write_environment_report()

    if S4.component_complete(S4.S4_SPLITS, REQUIRED):
        print(f"[STEP 4.1] benign splits already complete at {S4.S4_SPLITS} -- SKIPPING "
              f"(reusing the scale-independent real-benign split)")
        _ensure_stats_from_existing_splits()
        S4.update_status("data_preparation_status", "complete")
        return

    with S4.stage_banner("4.1 BENIGN DATA PREPARATION"):
        S4.set_global_seeds(S4.SPLIT_SEED)

        df = S4.load_real_benign_frame(usecols=KEEP_COLS)
        df = df[KEEP_COLS].copy()
        df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()

        # FINAL degree categories must already be consistent -- assert, never re-fit
        S4.verify_degree_categories(df)

        n_total = len(df)
        train_df, val_df, test_df = benign_split(df, S4.SPLIT_SEED)
        assert len(train_df) + len(val_df) + len(test_df) == n_total, \
            "split lost/gained rows -- ABORT"

        # hard re-assert scope on every split
        for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
            assert set(part["Attack"].unique()) <= {S4.BENIGN_LABEL}, \
                f"non-benign row in {name} split -- ABORT"

        S4.atomic_write_csv(train_df, os.path.join(S4.S4_SPLITS, "train.csv"))
        S4.atomic_write_csv(val_df, os.path.join(S4.S4_SPLITS, "val.csv"))
        S4.atomic_write_csv(test_df, os.path.join(S4.S4_SPLITS, "test.csv"))

        stats = _write_split_statistics(df, train_df, val_df, test_df)
        print("\n" + stats.to_string(index=False))

        S4.write_done(S4.S4_SPLITS, {
            "rows_benign_total": n_total,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "split_seed": S4.SPLIT_SEED,
            "fractions": [S4.TRAIN_FRAC, S4.VAL_FRAC, S4.TEST_FRAC],
            "class": S4.BENIGN_LABEL,
            "forbidden_labels": S4.FORBIDDEN_LABELS,
            "note": "single class -> no Attack stratification; TEST is held out "
                    "from support / balancing / back-off / training.",
        })
        S4.update_status("data_preparation_status", "complete")
        print(f"\n[STEP 4.1] wrote benign splits to {S4.S4_SPLITS}")


if __name__ == "__main__":
    main()
