# ============================================================
# STEP 4 -- FINAL REPORTING  (human-readable Step-4 benign summary)
#
# Pure aggregation of artifacts already produced by steps 4.1 - 4.6. Writes:
#   results/step4_benign/reports/training_configuration.txt
#   results/step4_benign/reports/environment.txt        (refreshed)
#   results/step4_benign/reports/step4_benign_final_report.txt
#
# Safe to re-run at any time; it never trains or generates.
# ============================================================

import os

import pandas as pd

from src.benign_flows import step4_common as S4


def _read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _variant_status():
    out = {}
    for v in S4.VARIANTS:
        vdir = os.path.join(S4.S4_MODELS, f"benign_variant_{v}")
        cfg = S4.read_json(os.path.join(vdir, "training_config.json"), {})
        out[v] = {
            "complete": S4.is_done(vdir),
            "rows": cfg.get("training_rows"),
            "seconds": cfg.get("training_seconds"),
            "epochs": cfg.get("epochs"),
            "final_epoch": cfg.get("final_epoch_recorded"),
            "conditioning": cfg.get("conditioning_columns"),
            "mechanism": cfg.get("progress_mechanism"),
            "gpu": cfg.get("gpu_name"),
            "cuda": cfg.get("cuda_available"),
        }
    return out


def write_training_configuration():
    env = S4.detect_environment()
    lines = ["STEP 4 (BENIGN) TRAINING CONFIGURATION", "=" * 50, ""]
    lines += [
        f"random_seed              : {S4.RANDOM_SEED}",
        f"split_seed               : {S4.SPLIT_SEED}",
        f"train/val/test fractions : {S4.TRAIN_FRAC} / {S4.VAL_FRAC} / {S4.TEST_FRAC}",
        f"MIN_SUPPORT              : {S4.MIN_SUPPORT}",
        f"MAX_WEIGHT (balancing)   : {S4.MAX_WEIGHT}",
        f"MIN_WEIGHT (balancing)   : {round(S4.MIN_WEIGHT, 4)}",
        f"CTGAN config (requested) : {S4.CTGAN_CONFIG}",
        f"target features          : {S4.TARGET_FEATURES}",
        f"categorical targets      : {S4.CATEGORICAL_TARGETS}",
        f"graph scale              : {S4.STEP2_SCALE}"
        + ({
            "larger": "  (x2 -- larger-scale experiment; results under results/large_scale/)",
            "smaller": "  (x0.5 -- smaller-scale experiment; results under results/small_scale/)",
            "same": "  (same scale as real data)",
        }.get(S4.SCALE, "")),
        "",
        "benign variant conditioning columns (Attack is NEVER a condition):",
    ]
    for v in S4.VARIANTS:
        lines.append(f"  {v} ({S4.VARIANT_TITLE[v]}): "
                     f"{S4.VARIANT_CONDITIONS[v] or '(none)'}")
    lines += ["",
              "degree categories (FINAL 2-bin, taken verbatim from Step 3):",
              f"  src_out_degree <= {S4.SRC_OUT_DEGREE_LOW_MAX} -> low ; > {S4.SRC_OUT_DEGREE_LOW_MAX} -> high",
              f"  dst_in_degree  <= {S4.DST_IN_DEGREE_LOW_MAX} -> low ; > {S4.DST_IN_DEGREE_LOW_MAX} -> high",
              "",
              "support-aware back-off ladder (benign generation):",
              "  LEVEL 0  Variant D  degree + role",
              "  LEVEL 1  Variant B  degree",
              "  LEVEL 2  Variant C  role   (if B and C both supported -> greater train support wins)",
              "  LEVEL 3  Variant A  baseline",
              "  else     request FAILED and LOGGED (never silently dropped)",
              "",
              "environment:"]
    for k, val in env.items():
        lines.append(f"  {k:<16}: {val}")
    for v in S4.VARIANTS:
        cfg = S4.read_json(os.path.join(S4.S4_MODELS, f"benign_variant_{v}",
                                        "training_config.json"), {})
        if cfg:
            lines += ["", f"benign variant {v} actual run:",
                      f"  ctgan_kwargs_used   : {cfg.get('ctgan_kwargs_used')}",
                      f"  ctgan_kwargs_dropped: {cfg.get('ctgan_kwargs_dropped')}",
                      f"  training_rows       : {cfg.get('training_rows')}",
                      f"  training_seconds    : {cfg.get('training_seconds')}",
                      f"  epochs / final      : {cfg.get('epochs')} / {cfg.get('final_epoch_recorded')}",
                      f"  progress_mechanism  : {cfg.get('progress_mechanism')}",
                      f"  sdv_version         : {cfg.get('sdv_version')}",
                      f"  torch / cuda / gpu  : {cfg.get('torch_version')} / "
                      f"{cfg.get('cuda_version')} / {cfg.get('gpu_name')}"]
    S4.atomic_write_text("\n".join(lines) + "\n",
                         os.path.join(S4.S4_REPORTS, "training_configuration.txt"))


def main():
    S4.ensure_dirs()
    S4.write_environment_report()
    write_training_configuration()

    L = []
    def p(s=""): L.append(s)

    p("=" * 70)
    p("STEP 4 -- CTGAN BENIGN-FLOW GENERATION -- FINAL REPORT")
    p("=" * 70)
    p("")
    p("SCOPE: benign traffic only.")
    p("EXCLUDED (hard assertions): " + ", ".join(S4.FORBIDDEN_LABELS))
    p("")
    p("The four benign variants are an ablation study. Only the STRUCTURAL")
    p("conditioning differs; epochs / batch size / seed / target features /")
    p("SDV configuration are identical across variants.")
    p("")

    dss = _read_csv(os.path.join(S4.S4_REPORTS, "data_split_statistics.csv"))
    if dss is not None:
        p("-" * 70); p("BENIGN TRAIN / VAL / TEST SPLIT "
                       f"(seed {S4.SPLIT_SEED}, single class):"); p("-" * 70)
        p(dss.to_string(index=False))
        p("  NOTE: TEST was NOT used for support, balancing, back-off or training.")
        p("")

    vss = _read_csv(os.path.join(S4.S4_SUPPORT, "variant_support_summary.csv"))
    if vss is not None:
        p("-" * 70); p(f"CONDITION SUPPORT (train only, MIN_SUPPORT={S4.MIN_SUPPORT}):")
        p("-" * 70); p(vss.to_string(index=False)); p("")

    vsc = _read_csv(os.path.join(S4.S4_SUPPORT, "variant_support_by_condition.csv"))
    if vsc is not None:
        p("-" * 70); p("PER-CONDITION SUPPORT (train only):"); p("-" * 70)
        p(vsc.to_string(index=False)); p("")

    brep = _read_csv(os.path.join(S4.S4_BALANCING, "balancing_report.csv"))
    if brep is not None:
        p("-" * 70); p(f"MODERATE BALANCING (cap {S4.MAX_WEIGHT}x; rare <"
                       f"{S4.MIN_SUPPORT} never oversampled):"); p("-" * 70)
        agg = (brep.groupby("variant")
                   .agg(conditions=("Condition", "count"),
                        max_capped_weight=("Capped weight", "max"),
                        mean_capped_weight=("Capped weight", "mean"),
                        eff_rows=("Post-balancing effective support", "sum"))
                   .reset_index())
        p(agg.to_string(index=False)); p("")

    p("-" * 70); p("BENIGN CTGAN VARIANTS:"); p("-" * 70)
    for v, s in _variant_status().items():
        p(f"  Benign Variant {v} ({S4.VARIANT_TITLE[v]})")
        p(f"    status       : {'COMPLETE' if s['complete'] else 'NOT COMPLETE'}")
        p(f"    conditioning : {s['conditioning'] if s['conditioning'] else '(none)'}")
        p(f"    train rows   : {s['rows']}    training seconds: {s['seconds']}")
        p(f"    epochs       : {s['epochs']} (final recorded {s['final_epoch']})")
        p(f"    progress     : {s['mechanism']}")
        p(f"    gpu / cuda   : {s['gpu']} / {s['cuda']}")
    p("")

    gsum = _read_csv(os.path.join(S4.S4_REPORTS, "generation_summary.csv"))
    bsum = _read_csv(os.path.join(S4.S4_REPORTS, "backoff_summary.csv"))
    if gsum is not None:
        p("-" * 70); p("GENERATION SUMMARY:"); p("-" * 70)
        p(gsum.to_string(index=False)); p("")
    if bsum is not None:
        p("-" * 70); p("SUPPORT-AWARE BACK-OFF METRICS:"); p("-" * 70)
        p(bsum.to_string(index=False)); p("")

    final = S4.SYNTHETIC_BENIGN_FLOWS
    p("-" * 70); p("DELIVERABLE:"); p("-" * 70)
    p(f"  {final}"
      + (f"   ({os.path.getsize(final):,} bytes)" if os.path.exists(final) else "   (not yet produced)"))
    p(f"  validation: {'PASSED (16/16)' if S4.is_done(S4.S4_GENERATION, 'VALIDATION_DONE') else 'not complete'}")
    p("")
    p("The final benign generation uses the support-aware Variant D -> B/C -> A")
    p("strategy and produces ONE benign dataset. Variants A/B/C/D are kept as")
    p("trained ablation models. The final merged NF-ToN-IoT dataset is produced")
    p("by step4_7_merge_final.py.")
    p("=" * 70)

    S4.atomic_write_text("\n".join(L) + "\n",
                         os.path.join(S4.S4_REPORTS, "step4_benign_final_report.txt"))
    print("\n".join(L))
    print("\n[STEP 4] wrote", os.path.join(S4.S4_REPORTS, "step4_benign_final_report.txt"))


if __name__ == "__main__":
    main()
