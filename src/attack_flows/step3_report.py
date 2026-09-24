# ============================================================
# STEP 3 -- FINAL REPORTING  (assembles the human-readable Step-3 summary)
#
# Pure aggregation of artifacts already produced by steps 3.2 - 3.7. Writes:
#   results/step3_ctgan/reports/training_configuration.txt
#   results/step3_ctgan/reports/environment.txt        (refreshed)
#   results/step3_ctgan/reports/step3_ctgan_final_report.txt
#
# Safe to re-run at any time; it never trains or generates.
# ============================================================

import os

import pandas as pd

from src.attack_flows import step3_common as C


def _read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _variant_status():
    out = {}
    for v in C.VARIANTS:
        vdir = os.path.join(C.S3_MODELS, f"variant_{v}")
        cfg = C.read_json(os.path.join(vdir, "training_config.json"), {})
        out[v] = {
            "complete": C.is_done(vdir),
            "rows": cfg.get("training_rows"),
            "seconds": cfg.get("training_seconds"),
            "conditioning": cfg.get("conditioning_columns"),
        }
    return out


def write_training_configuration():
    env = C.detect_environment()
    lines = ["STEP 3 TRAINING CONFIGURATION", "=" * 50, ""]
    lines += [
        f"random_seed              : {C.RANDOM_SEED}",
        f"split_seed               : {C.SPLIT_SEED}",
        f"train/val/test fractions : {C.TRAIN_FRAC} / {C.VAL_FRAC} / {C.TEST_FRAC}",
        f"MIN_SUPPORT              : {C.MIN_SUPPORT}",
        f"MAX_WEIGHT (balancing)   : {C.MAX_WEIGHT}",
        f"MIN_WEIGHT (balancing)   : {round(C.MIN_WEIGHT, 4)}",
        f"CTGAN config (requested) : {C.CTGAN_CONFIG}",
        f"target features          : {C.TARGET_FEATURES}",
        f"categorical targets      : {C.CATEGORICAL_TARGETS}",
        "",
        "variant conditioning columns:",
    ]
    for v in C.VARIANTS:
        lines.append(f"  {v} ({C.VARIANT_TITLE[v]}): {C.VARIANT_CONDITIONS[v]}")
    lines += ["",
              "degree categories (FINAL 2-bin, global Q50 -- results/step3/degree_binning_report.txt):",
              f"  src_out_degree <= {C.SRC_OUT_DEGREE_LOW_MAX} -> low ; > {C.SRC_OUT_DEGREE_LOW_MAX} -> high",
              f"  dst_in_degree  <= {C.DST_IN_DEGREE_LOW_MAX} -> low ; > {C.DST_IN_DEGREE_LOW_MAX} -> high",
              "",
              "environment:"]
    for k, val in env.items():
        lines.append(f"  {k:<16}: {val}")
    for v in C.VARIANTS:
        cfg = C.read_json(os.path.join(C.S3_MODELS, f"variant_{v}", "training_config.json"), {})
        if cfg:
            lines += ["", f"variant {v} actual run:",
                      f"  ctgan_kwargs_used   : {cfg.get('ctgan_kwargs_used')}",
                      f"  ctgan_kwargs_dropped: {cfg.get('ctgan_kwargs_dropped')}",
                      f"  training_rows       : {cfg.get('training_rows')}",
                      f"  training_seconds    : {cfg.get('training_seconds')}",
                      f"  sdv_version         : {cfg.get('sdv_version')}"]
    C.atomic_write_text("\n".join(lines) + "\n",
                        os.path.join(C.S3_REPORTS, "training_configuration.txt"))


def main():
    C.ensure_dirs()
    C.write_environment_report()
    write_training_configuration()

    L = []
    def p(s=""): L.append(s)

    p("=" * 70)
    p("STEP 3 -- CTGAN ATTACK-FLOW GENERATION -- FINAL REPORT")
    p("=" * 70)
    p("")
    p("STEP 3 INCLUDED (8 attack classes):")
    for a in C.ATTACKS_STEP3:
        p(f"  - {a}")
    p("")
    p("STEP 3 EXCLUDED:")
    p("  - benign     : processed separately in Step 4 (own graph/enrich/CTGAN/generation)")
    p("  - ransomware : real sample size too small (142 flows) for reliable CTGAN training;")
    p("                 already dropped upstream at Step 1.")
    p("")

    ads = _read_csv(os.path.join(C.S3_REPORTS, "attack_dataset_statistics.csv"))
    dss = _read_csv(os.path.join(C.S3_REPORTS, "data_split_statistics.csv"))
    if ads is not None:
        p("-" * 70); p("ATTACK DATASET (enriched real flows, attack-only):"); p("-" * 70)
        p(ads.to_string(index=False))
        p("")
    if dss is not None:
        p("-" * 70); p("TRAIN / VAL / TEST SPLIT (stratified by Attack, seed "
                       f"{C.SPLIT_SEED}):"); p("-" * 70)
        p(dss.to_string(index=False))
        p("  NOTE: TEST was NOT used for support, balancing, back-off or training.")
        p("")

    vss = _read_csv(os.path.join(C.S3_SUPPORT, "variant_support_summary.csv"))
    if vss is not None:
        p("-" * 70); p(f"CONDITION SUPPORT (train only, MIN_SUPPORT={C.MIN_SUPPORT}):"); p("-" * 70)
        p(vss.to_string(index=False)); p("")

    brep = _read_csv(os.path.join(C.S3_BALANCING, "balancing_report.csv"))
    if brep is not None:
        p("-" * 70); p(f"MODERATE BALANCING (cap {C.MAX_WEIGHT}x; rare <"
                       f"{C.MIN_SUPPORT} never oversampled):"); p("-" * 70)
        agg = (brep.groupby("variant")
                   .agg(conditions=("Condition", "count"),
                        max_capped_weight=("Capped weight", "max"),
                        mean_capped_weight=("Capped weight", "mean"),
                        eff_rows=("Post-balancing effective support", "sum"))
                   .reset_index())
        p(agg.to_string(index=False)); p("")

    p("-" * 70); p("CTGAN VARIANTS:"); p("-" * 70)
    for v, s in _variant_status().items():
        p(f"  Variant {v} ({C.VARIANT_TITLE[v]})")
        p(f"    status       : {'COMPLETE' if s['complete'] else 'NOT COMPLETE'}")
        p(f"    conditioning : {s['conditioning']}")
        p(f"    train rows   : {s['rows']}    training seconds: {s['seconds']}")
    p("")

    gsum = _read_csv(os.path.join(C.S3_REPORTS, "generation_summary.csv"))
    bsum = _read_csv(os.path.join(C.S3_REPORTS, "backoff_summary.csv"))
    if gsum is not None:
        p("-" * 70); p("GENERATION SUMMARY (per attack):"); p("-" * 70)
        p(gsum.to_string(index=False)); p("")
        p(f"  total synthetic edges  : {int(gsum['synthetic_edges'].sum())}")
        p(f"  total requested flows  : {int(gsum['requested_flows'].sum())}")
        p(f"  total generated flows  : {int(gsum['generated_flows'].sum())}")
        p(f"  total failed requests  : {int(gsum['failed_requests'].sum())}")
        p("")
    if bsum is not None:
        p("-" * 70); p("SUPPORT-AWARE BACK-OFF METRICS:"); p("-" * 70)
        p(bsum.to_string(index=False)); p("")
        g = bsum[bsum["scope"] == "GLOBAL"]
        if len(g):
            r = g.iloc[0]
            p(f"  GLOBAL full-condition coverage : {r['full_condition_coverage']}")
            p(f"  GLOBAL back-off rate           : {r['backoff_rate']}")
            p(f"  GLOBAL failure rate            : {r['failure_rate']}")
            p(f"  Variant D usage rate           : {r['variant_D_usage_rate']}")
            p(f"  Variant B fallback rate        : {r['variant_B_fallback_rate']}")
            p(f"  Variant C fallback rate        : {r['variant_C_fallback_rate']}")
            p(f"  Variant A fallback rate        : {r['variant_A_fallback_rate']}")
        p("")

    combined = os.path.join(C.S3_GENERATION, "synthetic_attack_flows.csv")
    p("-" * 70); p("DELIVERABLE:"); p("-" * 70)
    p(f"  {combined}"
      + (f"   ({os.path.getsize(combined):,} bytes)" if os.path.exists(combined) else "   (not yet produced)"))
    p(f"  validation: {'PASSED (13/13)' if C.is_done(C.S3_GENERATION, 'VALIDATION_DONE') else 'not complete'}")
    p("")
    p("Step 4 (benign) and the final merged dataset are OUT OF SCOPE for Step 3.")
    p("=" * 70)

    C.atomic_write_text("\n".join(L) + "\n",
                        os.path.join(C.S3_REPORTS, "step3_ctgan_final_report.txt"))
    print("\n".join(L))
    print("\n[STEP 3] wrote", os.path.join(C.S3_REPORTS, "step3_ctgan_final_report.txt"))


if __name__ == "__main__":
    main()
