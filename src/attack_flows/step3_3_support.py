# ============================================================
# STEP 3.3 -- CONDITION SUPPORT ANALYSIS  (TRAIN split ONLY)
#
# For each of the 4 variants, count how many TRAIN rows back every observed
# conditioning combination and classify it SUPPORTED (>= MIN_SUPPORT) or
# INSUFFICIENT (< MIN_SUPPORT).
#
#   Variant A : Attack
#   Variant B : Attack + src_out_degree_cat + dst_in_degree_cat
#   Variant C : Attack + src_role + dst_role
#   Variant D : Attack + src_out_degree_cat + dst_in_degree_cat
#                      + src_role + dst_role
#
# Outputs (results/step3_ctgan/support/):
#   support_variant_<X>.csv           every observed combo + count + flag
#   support_variant_<X>_by_attack.csv per-attack rollup
#   variant_support_summary.csv       one row per variant (global)
#   variant_support_by_attack.csv     one row per (variant, attack)
#   DONE
#
# Reads ONLY splits/train.csv. Never touches val/test.
# ============================================================

import os

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C

REQUIRED = ["variant_support_summary.csv", "variant_support_by_attack.csv"]

# the finite vocabulary each conditioning column can take (for "possible combos")
LEVELS = {
    "Attack": C.ATTACKS_STEP3,
    "src_out_degree_cat": list(C.DEGREE_CATEGORIES),
    "dst_in_degree_cat": list(C.DEGREE_CATEGORIES),
    "src_role": list(C.ROLE_CATEGORIES),
    "dst_role": list(C.ROLE_CATEGORIES),
}


def _support_table(train: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    g = (train.groupby(cols, observed=True)
              .size().reset_index(name="support"))
    g["supported"] = g["support"] >= C.MIN_SUPPORT
    return g.sort_values(cols).reset_index(drop=True)


def _summ(values: np.ndarray) -> dict:
    if len(values) == 0:
        return dict(min=0, median=0.0, mean=0.0, max=0)
    return dict(min=int(values.min()),
                median=float(np.median(values)),
                mean=float(values.mean()),
                max=int(values.max()))


def main():
    C.ensure_dirs()
    if C.component_complete(C.S3_SUPPORT, REQUIRED):
        print(f"[STEP 3.3] support analysis already complete -- SKIPPING")
        C.update_status("support_analysis_status", "complete")
        return

    with C.stage_banner("3.3 CONDITION SUPPORT ANALYSIS (train only)"):
        train_path = os.path.join(C.S3_SPLITS, "train.csv")
        if not os.path.exists(train_path):
            raise SystemExit("ABORT: splits/train.csv missing -- run step3_2_prepare.py first")
        train = pd.read_csv(train_path, low_memory=False)
        train["Attack"] = train["Attack"].astype(str).str.strip().str.lower()
        print(f"  train rows: {len(train):,}")

        global_rows, per_attack_rows = [], []

        for v in C.VARIANTS:
            cols = C.VARIANT_CONDITIONS[v]
            tbl = _support_table(train, cols)
            C.atomic_write_csv(tbl, os.path.join(C.S3_SUPPORT,
                                                 f"support_variant_{v}.csv"))

            n_possible = int(np.prod([len(LEVELS[c]) for c in cols]))
            n_observed = len(tbl)
            n_supported = int(tbl["supported"].sum())
            n_insuff = n_observed - n_supported
            s = _summ(tbl["support"].to_numpy())
            global_rows.append({
                "variant": v,
                "conditioning": " + ".join(cols),
                "possible_combos": n_possible,
                "observed_combos": n_observed,
                "unobserved_combos": n_possible - n_observed,
                "supported_combos": n_supported,
                "insufficient_combos": n_insuff,
                "min_support": s["min"], "median_support": s["median"],
                "mean_support": round(s["mean"], 2), "max_support": s["max"],
                "pct_observed_supported": round(100 * n_supported / max(n_observed, 1), 2),
            })
            print(f"  variant {v}: {n_observed}/{n_possible} combos observed, "
                  f"{n_supported} supported, {n_insuff} insufficient, "
                  f"median support {s['median']:.0f}")

            # per-attack rollup
            by = []
            for atk in C.ATTACKS_STEP3:
                sub = tbl[tbl["Attack"] == atk]
                non_attack_levels = [len(LEVELS[c]) for c in cols if c != "Attack"]
                atk_possible = int(np.prod(non_attack_levels)) if non_attack_levels else 1
                ss = _summ(sub["support"].to_numpy())
                by.append({
                    "variant": v, "Attack": atk,
                    "possible_combos": atk_possible,
                    "observed_combos": len(sub),
                    "supported_combos": int(sub["supported"].sum()),
                    "insufficient_combos": int((~sub["supported"]).sum()),
                    "unobserved_combos": atk_possible - len(sub),
                    "min_support": ss["min"], "median_support": ss["median"],
                    "mean_support": round(ss["mean"], 2), "max_support": ss["max"],
                })
            per_attack_rows.extend(by)
            C.atomic_write_csv(pd.DataFrame(by),
                               os.path.join(C.S3_SUPPORT,
                                            f"support_variant_{v}_by_attack.csv"))

        C.atomic_write_csv(pd.DataFrame(global_rows),
                           os.path.join(C.S3_SUPPORT, "variant_support_summary.csv"))
        C.atomic_write_csv(pd.DataFrame(per_attack_rows),
                           os.path.join(C.S3_SUPPORT, "variant_support_by_attack.csv"))

        C.write_done(C.S3_SUPPORT, {
            "min_support": C.MIN_SUPPORT,
            "train_rows": len(train),
            "source": "splits/train.csv (val/test untouched)",
        })
        C.update_status("support_analysis_status", "complete")
        print("\n[STEP 3.3] support tables written to", C.S3_SUPPORT)


if __name__ == "__main__":
    main()
