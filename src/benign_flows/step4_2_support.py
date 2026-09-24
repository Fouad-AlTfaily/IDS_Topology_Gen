# ============================================================
# STEP 4.2 -- BENIGN CONDITION SUPPORT ANALYSIS  (TRAIN split ONLY)
#
# For each of the 4 benign variants, count how many TRAIN rows back every
# observed conditioning combination and classify it SUPPORTED (>= MIN_SUPPORT)
# or INSUFFICIENT (< MIN_SUPPORT).
#
#   Variant A : (no structural condition)          -> 1 group  (all benign train)
#   Variant B : src_out_degree_cat + dst_in_degree_cat
#   Variant C : src_role + dst_role
#   Variant D : src_out_degree_cat + dst_in_degree_cat + src_role + dst_role
#
# Reads ONLY splits/train.csv. Never touches val / test.
#
# Outputs (results/step4_benign/support/ AND results/step4_benign/reports/):
#   support/support_variant_<X>.csv            every observed combo + count + flag
#   support/support_variant_<X>_by_condition.csv  per single condition column
#   support/variant_support_summary.csv        (also copied to reports/)
#   support/variant_support_by_condition.csv   (also copied to reports/)
#   reports/variant_support_summary.csv
#   reports/variant_support_by_condition.csv
#   support/DONE
# ============================================================

import os

import numpy as np
import pandas as pd

from src.benign_flows import step4_common as S4

REQUIRED = ["variant_support_summary.csv", "variant_support_by_condition.csv"]

# finite vocabulary of each conditioning column (for "possible combinations")
LEVELS = {
    "src_out_degree_cat": list(S4.DEGREE_CATEGORIES),
    "dst_in_degree_cat": list(S4.DEGREE_CATEGORIES),
    "src_role": list(S4.ROLE_CATEGORIES),
    "dst_role": list(S4.ROLE_CATEGORIES),
}


def _summ(values: np.ndarray) -> dict:
    if len(values) == 0:
        return dict(min=0, median=0.0, mean=0.0, max=0)
    return dict(min=int(values.min()),
                median=float(np.median(values)),
                mean=float(values.mean()),
                max=int(values.max()))


def _support_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:  # Variant A -- one implicit group
        g = pd.DataFrame([{"condition": "ALL", "support": len(train)}])
    else:
        g = (train.groupby(cols, observed=True).size()
                  .reset_index(name="support").sort_values(cols).reset_index(drop=True))
    g["supported"] = g["support"] >= S4.MIN_SUPPORT
    return g


def main():
    S4.ensure_dirs()
    if S4.component_complete(S4.S4_SUPPORT, REQUIRED):
        print("[STEP 4.2] benign support analysis already complete -- SKIPPING")
        S4.update_status("support_analysis_status", "complete")
        return

    with S4.stage_banner("4.2 BENIGN CONDITION SUPPORT ANALYSIS (train only)"):
        train_path = os.path.join(S4.S4_SPLITS, "train.csv")
        if not os.path.exists(train_path):
            raise SystemExit("ABORT: splits/train.csv missing -- run step4_1_prepare.py first")
        train = pd.read_csv(train_path, low_memory=False)
        train["Attack"] = train["Attack"].astype(str).str.strip().str.lower()
        assert set(train["Attack"].unique()) <= {S4.BENIGN_LABEL}, "non-benign in train -- ABORT"
        n_train = len(train)
        print(f"  benign train rows: {n_train:,}   MIN_SUPPORT={S4.MIN_SUPPORT}")

        summary_rows, by_cond_rows = [], []

        for v in S4.VARIANTS:
            cols = S4.VARIANT_CONDITIONS[v]
            tbl = _support_table(train, cols)
            S4.atomic_write_csv(tbl, os.path.join(S4.S4_SUPPORT, f"support_variant_{v}.csv"))

            n_possible = int(np.prod([len(LEVELS[c]) for c in cols])) if cols else 1
            n_observed = len(tbl)
            n_supported = int(tbl["supported"].sum())
            n_insuff = n_observed - n_supported
            s = _summ(tbl["support"].to_numpy())
            summary_rows.append({
                "variant": v,
                "conditioning": " + ".join(cols) if cols else "(none)",
                "total_training_rows": n_train,
                "possible_combinations": n_possible,
                "observed_combinations": n_observed,
                "unobserved_combinations": n_possible - n_observed,
                "supported_combinations": n_supported,
                "unsupported_combinations": n_insuff,
                "min_support": s["min"],
                "median_support": s["median"],
                "mean_support": round(s["mean"], 2),
                "max_support": s["max"],
                "pct_observed_supported": round(100 * n_supported / max(n_observed, 1), 2),
            })
            print(f"  variant {v}: {n_observed}/{n_possible} combos observed, "
                  f"{n_supported} supported, {n_insuff} insufficient, "
                  f"median support {s['median']:.0f}")

            # ---- per single conditioning column, per category ----
            per_col = []
            for c in cols:
                vc = train.groupby(c, observed=True).size().to_dict()
                for lv in LEVELS[c]:
                    cnt = int(vc.get(lv, 0))
                    rec = {"variant": v, "condition_column": c, "category": lv,
                           "train_rows": cnt, "supported": cnt >= S4.MIN_SUPPORT}
                    per_col.append(rec)
                    by_cond_rows.append(rec)
            if not cols:
                rec = {"variant": v, "condition_column": "(none)", "category": "ALL",
                       "train_rows": n_train, "supported": n_train >= S4.MIN_SUPPORT}
                per_col.append(rec)
                by_cond_rows.append(rec)
            S4.atomic_write_csv(pd.DataFrame(per_col),
                                os.path.join(S4.S4_SUPPORT,
                                             f"support_variant_{v}_by_condition.csv"))

        summary = pd.DataFrame(summary_rows)
        by_cond = pd.DataFrame(by_cond_rows)
        for d in (S4.S4_SUPPORT, S4.S4_REPORTS):
            S4.atomic_write_csv(summary, os.path.join(d, "variant_support_summary.csv"))
            S4.atomic_write_csv(by_cond, os.path.join(d, "variant_support_by_condition.csv"))

        print("\n" + summary.to_string(index=False))

        S4.write_done(S4.S4_SUPPORT, {
            "min_support": S4.MIN_SUPPORT,
            "train_rows": n_train,
            "source": "splits/train.csv (val/test untouched)",
        })
        S4.update_status("support_analysis_status", "complete")
        print("\n[STEP 4.2] benign support tables written to", S4.S4_SUPPORT)


if __name__ == "__main__":
    main()
