# ============================================================
# STEP 3.2 -- DATA PREPARATION  (attack-only filter + train/val/test split)
#
# Reads results/step3/step3_1_enriched_real_flows.csv, keeps ONLY the eight
# Step-3 attack classes (benign + ransomware hard-excluded with assertions),
# keeps the 9 CTGAN target features + the 5 structural conditioning columns,
# and produces a deterministic 80 / 10 / 10 split stratified by Attack.
#
# TEST is written out but MUST NOT be used for support analysis, balancing,
# back-off, threshold selection or training -- only the final held-out check.
#
# Outputs (results/step3_ctgan/):
#   splits/train.csv  splits/val.csv  splits/test.csv
#   splits/DONE
#   reports/attack_dataset_statistics.csv
#   reports/data_split_statistics.csv
#
# Idempotent: re-running with a valid splits/DONE is a no-op unless
# STEP3_FORCE_RERUN=1.
# ============================================================

import os

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C

KEEP_COLS = (C.TARGET_FEATURES
             + ["Attack", "src_out_degree", "dst_in_degree",
                "src_out_degree_cat", "dst_in_degree_cat",
                "src_role", "dst_role"])

REQUIRED = ["train.csv", "val.csv", "test.csv"]


def _verify_degree_cats(df: pd.DataFrame):
    """The categorical columns must already match the FINAL thresholds. We do
    NOT recompute or overwrite them -- we only assert they are consistent, so a
    corrupted upstream file is caught here rather than silently trained on."""
    exp_src = np.where(df["src_out_degree"].to_numpy(dtype=float)
                       <= C.SRC_OUT_DEGREE_LOW_MAX, "low", "high")
    exp_dst = np.where(df["dst_in_degree"].to_numpy(dtype=float)
                       <= C.DST_IN_DEGREE_LOW_MAX, "low", "high")
    bad_src = int((df["src_out_degree_cat"].to_numpy() != exp_src).sum())
    bad_dst = int((df["dst_in_degree_cat"].to_numpy() != exp_dst).sum())
    if bad_src or bad_dst:
        raise SystemExit(
            f"ABORT: enriched degree categories do not match the FINAL thresholds "
            f"(src mismatches={bad_src}, dst mismatches={bad_dst}). "
            f"Re-run Step 3.1 degree categorization."
        )
    assert set(df["src_out_degree_cat"].unique()) <= set(C.DEGREE_CATEGORIES)
    assert set(df["dst_in_degree_cat"].unique()) <= set(C.DEGREE_CATEGORIES)
    assert set(df["src_role"].unique()) <= set(C.ROLE_CATEGORIES)
    assert set(df["dst_role"].unique()) <= set(C.ROLE_CATEGORIES)


def stratified_split(df: pd.DataFrame, seed: int):
    """Deterministic per-class shuffle then 80/10/10 slice. No sklearn dependency."""
    tr, va, te = [], [], []
    for atk, g in df.groupby("Attack", sort=True):
        idx = g.index.to_numpy()
        rng = np.random.default_rng(C.stable_seed(seed, atk))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(C.TRAIN_FRAC * n))
        n_va = int(round(C.VAL_FRAC * n))
        # guarantee every non-empty class contributes at least one train row
        n_tr = min(max(n_tr, 1), n)
        n_va = min(n_va, n - n_tr)
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    tr = np.concatenate(tr) if tr else np.array([], dtype=int)
    va = np.concatenate(va) if va else np.array([], dtype=int)
    te = np.concatenate(te) if te else np.array([], dtype=int)
    # keep rows in original file order within each split (no unnecessary reorder)
    return (df.loc[np.sort(tr)].copy(),
            df.loc[np.sort(va)].copy(),
            df.loc[np.sort(te)].copy())


def _write_split_statistics(raw_counts, attack_df, train_df, val_df, test_df):
    """Emit the two lightweight statistics CSVs into C.S3_REPORTS. Split membership
    is NOT touched here -- only the derived summary tables are (re)written, so a
    scale that reuses the shared splits still gets its own reports/ populated
    (Step 5 verifies the split against data_split_statistics.csv)."""
    stat_rows = []
    for atk in sorted(set(list(raw_counts) + C.ATTACKS_STEP3)):
        orig = int(raw_counts.get(atk, 0))
        kept = int((attack_df["Attack"] == atk).sum())
        stat_rows.append({
            "Attack": atk,
            "original_row_count": orig,
            "rows_retained": kept,
            "rows_excluded": orig - kept,
            "in_step3_scope": atk in C.ATTACKS_STEP3,
        })
    stats = pd.DataFrame(stat_rows).sort_values("Attack")
    C.atomic_write_csv(stats, os.path.join(C.S3_REPORTS,
                                           "attack_dataset_statistics.csv"))

    split_rows = []
    for atk in C.ATTACKS_STEP3:
        split_rows.append({
            "Attack": atk,
            "total": int((attack_df["Attack"] == atk).sum()),
            "train": int((train_df["Attack"] == atk).sum()),
            "val": int((val_df["Attack"] == atk).sum()),
            "test": int((test_df["Attack"] == atk).sum()),
        })
    split_rows.append({
        "Attack": "ALL",
        "total": len(attack_df), "train": len(train_df),
        "val": len(val_df), "test": len(test_df),
    })
    split_stats = pd.DataFrame(split_rows)
    C.atomic_write_csv(split_stats,
                       os.path.join(C.S3_REPORTS, "data_split_statistics.csv"))
    return stats, split_stats


def _ensure_stats_from_existing_splits():
    """Skip-path helper: the shared splits/ is already complete, but this scale's
    reports/ (e.g. results/small_scale/... for the smaller scale) may not carry
    the statistics CSVs yet. Rebuild them from the existing split files without
    rewriting a single split row."""
    ads = os.path.join(C.S3_REPORTS, "attack_dataset_statistics.csv")
    dss = os.path.join(C.S3_REPORTS, "data_split_statistics.csv")
    if os.path.exists(ads) and os.path.getsize(ads) > 0 \
            and os.path.exists(dss) and os.path.getsize(dss) > 0:
        return
    df_full = pd.read_csv(C.ENRICHED_REAL_FLOWS,
                          usecols=sorted(set(KEEP_COLS)), low_memory=False)
    df_full["Attack"] = df_full["Attack"].astype(str).str.strip().str.lower()
    raw_counts = df_full["Attack"].value_counts().to_dict()
    attack_df = df_full[df_full["Attack"].isin(C.ATTACKS_STEP3)].copy()
    train_df = pd.read_csv(os.path.join(C.S3_SPLITS, "train.csv"), low_memory=False)
    val_df = pd.read_csv(os.path.join(C.S3_SPLITS, "val.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(C.S3_SPLITS, "test.csv"), low_memory=False)
    for d in (train_df, val_df, test_df):
        d["Attack"] = d["Attack"].astype(str).str.strip().str.lower()
    _write_split_statistics(raw_counts, attack_df, train_df, val_df, test_df)
    print(f"[STEP 3.2] (re)wrote split statistics under {C.S3_REPORTS}")


def main():
    C.ensure_dirs()
    C.write_environment_report()

    if C.component_complete(C.S3_SPLITS, REQUIRED):
        print(f"[STEP 3.2] splits already complete at {C.S3_SPLITS} -- SKIPPING "
              f"(reusing the scale-independent real-data split)")
        _ensure_stats_from_existing_splits()
        C.update_status("data_preparation_status", "complete")
        return

    with C.stage_banner("3.2 DATA PREPARATION"):
        C.set_global_seeds(C.SPLIT_SEED)

        df_full = pd.read_csv(C.ENRICHED_REAL_FLOWS,
                              usecols=sorted(set(KEEP_COLS)), low_memory=False)
        n_raw_total = len(df_full)
        df_full["Attack"] = df_full["Attack"].astype(str).str.strip().str.lower()

        raw_counts = df_full["Attack"].value_counts().to_dict()

        # ---- attack-only filter + hard assertions -------------------------
        attack_df = df_full[df_full["Attack"].isin(C.ATTACKS_STEP3)].copy()
        assert int((attack_df["Attack"] == "benign").sum()) == 0, "benign leaked -- ABORT"
        assert int((attack_df["Attack"] == "ransomware").sum()) == 0, "ransomware leaked -- ABORT"
        unexpected = sorted(set(attack_df["Attack"].unique()) - set(C.ATTACKS_STEP3))
        if unexpected:
            raise SystemExit(f"ABORT: unexpected class(es) present: {unexpected}")

        _verify_degree_cats(attack_df)

        # ---- split ---------------------------------------------------
        train_df, val_df, test_df = stratified_split(attack_df, C.SPLIT_SEED)
        assert len(train_df) + len(val_df) + len(test_df) == len(attack_df), \
            "split lost/gained rows -- ABORT"

        C.atomic_write_csv(train_df, os.path.join(C.S3_SPLITS, "train.csv"))
        C.atomic_write_csv(val_df, os.path.join(C.S3_SPLITS, "val.csv"))
        C.atomic_write_csv(test_df, os.path.join(C.S3_SPLITS, "test.csv"))

        # ---- statistics (attack dataset + train/val/test split) -----
        stats, split_stats = _write_split_statistics(
            raw_counts, attack_df, train_df, val_df, test_df)
        print(stats.to_string(index=False))
        print("\n" + split_stats.to_string(index=False))

        C.write_done(C.S3_SPLITS, {
            "rows_total_enriched": n_raw_total,
            "rows_attack_only": len(attack_df),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "split_seed": C.SPLIT_SEED,
            "fractions": [C.TRAIN_FRAC, C.VAL_FRAC, C.TEST_FRAC],
            "classes": C.ATTACKS_STEP3,
            "excluded": C.EXCLUDED_CLASSES,
        })
        C.update_status("data_preparation_status", "complete")
        print(f"\n[STEP 3.2] wrote splits to {C.S3_SPLITS}")


if __name__ == "__main__":
    main()
