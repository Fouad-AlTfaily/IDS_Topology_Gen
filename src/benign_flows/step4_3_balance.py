# ============================================================
# STEP 4.3 -- MODERATE BENIGN CONDITION-LEVEL BALANCING  (TRAIN split ONLY)
#
# Same methodological policy as Step 3.4 (MAX_WEIGHT = 5.0):
#   * group TRAIN by that variant's structural conditioning columns
#   * conditions with support < MIN_SUPPORT are NOT oversampled (weight = 1.0)
#     -- they rely on the support-aware back-off at generation time
#   * supported conditions get weight = median_supported_count / count,
#     clipped to [1/MAX_WEIGHT, MAX_WEIGHT]  (moderate both ways)
#   * effective_count = round(count * capped_weight)
#   * build the balanced frame by per-group resampling with a fixed seed
#     (with replacement only where weight > 1)
#
# SDV's CTGANSynthesizer.fit() has NO per-row sample-weight argument (verified
# against the installed signature in step3_common.SDVApi -- the same check
# Step 3 relied on), so balancing is a reproducible resample. No row is
# replicated more than 5x. Variant A has a single group -> it is passed through
# unchanged (weight 1.0).
#
# Outputs (results/step4_benign/balancing/):
#   balanced_train_variant_<X>.csv
#   balancing_report.csv   (variant, Condition, Original support, Sampling weight,
#                           Capped weight, Post-balancing effective support, supported)
#   DONE
# ============================================================

import os

import numpy as np
import pandas as pd

from src.benign_flows import step4_common as S4

REQUIRED = ["balancing_report.csv"] + [f"balanced_train_variant_{v}.csv" for v in S4.VARIANTS]


def balance_variant(train: pd.DataFrame, v: str, rng: np.random.Generator):
    cols = S4.VARIANT_CONDITIONS[v]

    if not cols:  # Variant A -- one group, pass through unchanged
        rep = pd.DataFrame([{
            "variant": v, "Condition": "(none)", "Original support": len(train),
            "Sampling weight": 1.0, "Capped weight": 1.0,
            "Post-balancing effective support": len(train), "supported": len(train) >= S4.MIN_SUPPORT,
        }])
        return train.copy(), rep

    grp = train.groupby(cols, observed=True)
    counts = grp.size()
    supported = counts[counts >= S4.MIN_SUPPORT]
    med = float(np.median(supported.to_numpy())) if len(supported) else float(counts.median())

    report_rows, pieces = [], []
    for key, idx in grp.groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        n = int(counts.loc[key])
        if n >= S4.MIN_SUPPORT:
            raw_w = med / n if n > 0 else 1.0
            capped_w = float(np.clip(raw_w, S4.MIN_WEIGHT, S4.MAX_WEIGHT))
        else:
            raw_w = 1.0           # rare conditions: keep as-is, never inflate
            capped_w = 1.0
        eff = max(int(round(n * capped_w)), 1)

        sub = train.loc[idx]
        take = sub.sample(n=eff, replace=(eff > n),
                          random_state=int(rng.integers(1 << 31)))
        pieces.append(take)

        cond_str = ", ".join(f"{c}={val}" for c, val in zip(cols, key_tuple))
        report_rows.append({
            "variant": v,
            "Condition": cond_str,
            "Original support": n,
            "Sampling weight": round(raw_w, 4),
            "Capped weight": round(capped_w, 4),
            "Post-balancing effective support": eff,
            "supported": n >= S4.MIN_SUPPORT,
        })

    balanced = pd.concat(pieces, ignore_index=True) if pieces else train.iloc[0:0].copy()
    return balanced, pd.DataFrame(report_rows)


def main():
    S4.ensure_dirs()
    if S4.component_complete(S4.S4_BALANCING, REQUIRED):
        print("[STEP 4.3] benign balancing already complete -- SKIPPING")
        S4.update_status("balancing_status", "complete")
        return

    with S4.stage_banner("4.3 MODERATE BENIGN CONDITION-LEVEL BALANCING (train only)"):
        S4.set_global_seeds(S4.RANDOM_SEED)
        train_path = os.path.join(S4.S4_SPLITS, "train.csv")
        if not os.path.exists(train_path):
            raise SystemExit("ABORT: splits/train.csv missing -- run step4_1_prepare.py first")
        train = pd.read_csv(train_path, low_memory=False)
        train["Attack"] = train["Attack"].astype(str).str.strip().str.lower()
        assert set(train["Attack"].unique()) <= {S4.BENIGN_LABEL}, "non-benign in train -- ABORT"
        print(f"  benign train rows: {len(train):,}   "
              f"MAX_WEIGHT={S4.MAX_WEIGHT}  MIN_SUPPORT={S4.MIN_SUPPORT}")

        all_reports = []
        for v in S4.VARIANTS:
            rng = np.random.default_rng(S4.RANDOM_SEED + ord(v))
            balanced, rep = balance_variant(train, v, rng)
            S4.atomic_write_csv(balanced,
                                os.path.join(S4.S4_BALANCING,
                                             f"balanced_train_variant_{v}.csv"))
            all_reports.append(rep)
            max_w = rep["Capped weight"].max() if len(rep) else 0
            print(f"  variant {v}: {len(train):,} -> {len(balanced):,} rows "
                  f"({len(rep)} condition(s), max capped weight {max_w:.2f})")
            assert max_w <= S4.MAX_WEIGHT + 1e-9, "weight cap violated -- ABORT"

        report = pd.concat(all_reports, ignore_index=True)
        S4.atomic_write_csv(report, os.path.join(S4.S4_BALANCING, "balancing_report.csv"))

        S4.write_done(S4.S4_BALANCING, {
            "max_weight": S4.MAX_WEIGHT,
            "min_weight": S4.MIN_WEIGHT,
            "min_support": S4.MIN_SUPPORT,
            "note": "SDV CTGANSynthesizer.fit has no sample_weight; balancing done by "
                    "reproducible per-condition resample, capped at 5x, rare "
                    "conditions (<10) never oversampled. Variant A passed through.",
        })
        S4.update_status("balancing_status", "complete")
        print("\n[STEP 4.3] benign balancing written to", S4.S4_BALANCING)


if __name__ == "__main__":
    main()
