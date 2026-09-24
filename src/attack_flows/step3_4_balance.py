# ============================================================
# STEP 3.4 -- MODERATE CONDITION-LEVEL BALANCING  (TRAIN split ONLY)
#
# Goal: stop very common conditioning combinations from dominating CTGAN
# training -- NOT to equalise them. Per variant:
#
#   * group TRAIN by that variant's conditioning columns
#   * conditions with support < MIN_SUPPORT are NOT oversampled (weight = 1.0)
#   * supported conditions get weight = median_supported_count / count,
#     clipped to [1/MAX_WEIGHT, MAX_WEIGHT]  (moderate both ways)
#   * effective_count = round(count * capped_weight)
#   * build the balanced frame by per-group resampling with a fixed seed
#     (with replacement only where weight > 1)
#
# SDV's CTGANSynthesizer.fit() has no per-row sample-weight argument (checked
# against the installed signature in step3_common.SDVApi), so balancing is
# done by this reproducible resample. No row is replicated more than 5x.
#
# Outputs (results/step3_ctgan/balancing/):
#   balanced_train_variant_<X>.csv
#   balancing_report.csv          (Attack, variant, Condition, Original support,
#                                  Sampling weight, Capped weight,
#                                  Post-balancing effective support)
#   DONE
# ============================================================

import os

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C

REQUIRED = ["balancing_report.csv"] + [f"balanced_train_variant_{v}.csv" for v in C.VARIANTS]


def balance_variant(train: pd.DataFrame, v: str, rng: np.random.Generator):
    cols = C.VARIANT_CONDITIONS[v]
    grp = train.groupby(cols, observed=True)
    counts = grp.size()

    supported = counts[counts >= C.MIN_SUPPORT]
    med = float(np.median(supported.to_numpy())) if len(supported) else float(counts.median())

    report_rows, pieces = [], []
    for key, idx in grp.groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        n = int(counts.loc[key])
        if n >= C.MIN_SUPPORT:
            raw_w = med / n if n > 0 else 1.0
            capped_w = float(np.clip(raw_w, C.MIN_WEIGHT, C.MAX_WEIGHT))
        else:
            raw_w = 1.0            # rare conditions: keep as-is, never inflate
            capped_w = 1.0
        eff = int(round(n * capped_w))
        eff = max(eff, 1)

        sub = train.loc[idx]
        if eff <= n:
            take = sub.sample(n=eff, replace=False, random_state=int(rng.integers(1 << 31)))
        else:
            take = sub.sample(n=eff, replace=True, random_state=int(rng.integers(1 << 31)))
        pieces.append(take)

        cond_str = ", ".join(f"{c}={val}" for c, val in zip(cols, key_tuple))
        report_rows.append({
            "variant": v,
            "Attack": key_tuple[0],
            "Condition": cond_str,
            "Original support": n,
            "Sampling weight": round(raw_w, 4),
            "Capped weight": round(capped_w, 4),
            "Post-balancing effective support": eff,
            "supported": n >= C.MIN_SUPPORT,
        })

    balanced = (pd.concat(pieces, ignore_index=True)
                if pieces else train.iloc[0:0].copy())
    return balanced, pd.DataFrame(report_rows)


def main():
    C.ensure_dirs()
    if C.component_complete(C.S3_BALANCING, REQUIRED):
        print("[STEP 3.4] balancing already complete -- SKIPPING")
        C.update_status("balancing_status", "complete")
        return

    with C.stage_banner("3.4 MODERATE CONDITION-LEVEL BALANCING (train only)"):
        C.set_global_seeds(C.RANDOM_SEED)
        train_path = os.path.join(C.S3_SPLITS, "train.csv")
        if not os.path.exists(train_path):
            raise SystemExit("ABORT: splits/train.csv missing -- run step3_2_prepare.py first")
        train = pd.read_csv(train_path, low_memory=False)
        train["Attack"] = train["Attack"].astype(str).str.strip().str.lower()
        print(f"  train rows: {len(train):,}   MAX_WEIGHT={C.MAX_WEIGHT}  MIN_SUPPORT={C.MIN_SUPPORT}")

        all_reports = []
        for v in C.VARIANTS:
            rng = np.random.default_rng(C.RANDOM_SEED + ord(v))
            balanced, rep = balance_variant(train, v, rng)
            C.atomic_write_csv(balanced,
                               os.path.join(C.S3_BALANCING,
                                            f"balanced_train_variant_{v}.csv"))
            all_reports.append(rep)
            max_rep = rep["Capped weight"].max() if len(rep) else 0
            print(f"  variant {v}: {len(train):,} -> {len(balanced):,} rows "
                  f"({len(rep)} conditions, max capped weight {max_rep:.2f})")
            assert max_rep <= C.MAX_WEIGHT + 1e-9, "weight cap violated -- ABORT"

        report = pd.concat(all_reports, ignore_index=True)
        C.atomic_write_csv(report, os.path.join(C.S3_BALANCING, "balancing_report.csv"))

        C.write_done(C.S3_BALANCING, {
            "max_weight": C.MAX_WEIGHT,
            "min_weight": C.MIN_WEIGHT,
            "min_support": C.MIN_SUPPORT,
            "note": "SDV CTGANSynthesizer.fit has no sample_weight; "
                    "balancing done by reproducible per-condition resample, "
                    "capped at 5x, rare conditions (<10) never oversampled.",
        })
        C.update_status("balancing_status", "complete")
        print("\n[STEP 3.4] balancing written to", C.S3_BALANCING)


if __name__ == "__main__":
    main()
