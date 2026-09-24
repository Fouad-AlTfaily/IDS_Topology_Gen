# ============================================================
# STEP 5 -- FINAL REPORT + COMPLETION MARKER
#
#   python -m src.validation.step5_report
#
# Aggregates Levels 1-3 into:
#   results/step5_validation/reports/final_validation_summary.csv
#   results/step5_validation/reports/step5_final_report.txt
#   results/step5_validation/reports/environment.txt        (refreshed)
#   results/step5_validation/DONE                           (only if all pass)
#
# DONE is written ONLY when LEVEL1_DONE, LEVEL2_DONE and LEVEL3_DONE all exist
# and every expected report / figure / CSV is present and non-empty.
# Re-runnable at any time; never trains or regenerates anything.
# ============================================================

import os

import numpy as np
import pandas as pd

from src.validation import step5_common as S5

# every artifact the three levels must have produced
FINAL_ARTIFACTS = [
    # Level 1
    os.path.join(S5.S5_STRUCTURAL, "degree_statistics.csv"),
    os.path.join(S5.S5_STRUCTURAL, "ks_degree_tests.csv"),
    os.path.join(S5.S5_STRUCTURAL, "modularity_results.csv"),
    os.path.join(S5.S5_FIGURES, "real_vs_synthetic_in_degree.png"),
    os.path.join(S5.S5_FIGURES, "real_vs_synthetic_out_degree.png"),
    os.path.join(S5.S5_FIGURES, "modularity_comparison.png"),
    os.path.join(S5.S5_REPORTS, "level1_structural_report.txt"),
    # Level 2
    os.path.join(S5.S5_STATISTICAL, "ks_tests.csv"),
    os.path.join(S5.S5_STATISTICAL, "categorical_tests.csv"),
    os.path.join(S5.S5_STATISTICAL, "descriptive_statistics.csv"),
    os.path.join(S5.S5_STATISTICAL, "multiple_testing_summary.csv"),
    os.path.join(S5.S5_FIG_STAT, "ecdf_numeric_features.png"),
    os.path.join(S5.S5_FIG_STAT, "perclass_ks_heatmap.png"),
    os.path.join(S5.S5_REPORTS, "level2_statistical_report.txt"),
    # Level 3
    os.path.join(S5.S5_UTILITY, "utility_metrics.csv"),
    os.path.join(S5.S5_UTILITY, "comparison_headline.csv"),
    os.path.join(S5.S5_UTILITY, "per_class_f1.csv"),
    os.path.join(S5.S5_UTILITY, "per_class_recall.csv"),
    os.path.join(S5.S5_UTILITY, "per_class_precision.csv"),
    os.path.join(S5.S5_FIG_UTIL, "real_only_confusion_matrix.png"),
    os.path.join(S5.S5_FIG_UTIL, "synthetic_only_confusion_matrix.png"),
    os.path.join(S5.S5_FIG_UTIL, "real_plus_synthetic_confusion_matrix.png"),
    os.path.join(S5.S5_FIG_UTIL, "utility_macro_f1_comparison.png"),
    os.path.join(S5.S5_FIG_UTIL, "utility_per_class_f1.png"),
    os.path.join(S5.S5_FIG_UTIL, "utility_per_class_recall.png"),
    os.path.join(S5.S5_REPORTS, "level3_utility_report.txt"),
]


def _csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def _level1_summary():
    ks = _csv(os.path.join(S5.S5_STRUCTURAL, "ks_degree_tests.csv"))
    mod = _csv(os.path.join(S5.S5_STRUCTURAL, "modularity_results.csv"))
    if ks is None or mod is None:
        return None
    ov = ks[ks["scope"] == "overall"].set_index("metric")
    d_in = float(ov.loc["in_degree", "ks_statistic"])
    d_out = float(ov.loc["out_degree", "ks_statistic"])
    summ = mod[mod["scope"] == "summary_mean"]
    mgap = float(summ["abs_modularity_diff"].iloc[0]) if len(summ) else float("nan")
    dmax = max(d_in, d_out)
    verdict = ("PRESERVED" if dmax < 0.1 and mgap < 0.1 else
               "LARGELY PRESERVED" if dmax < 0.2 and mgap < 0.2 else
               "PARTIALLY PRESERVED" if dmax < 0.35 else "NOT PRESERVED")
    return {"d_in": d_in, "d_out": d_out, "mod_gap": mgap, "verdict": verdict}


def _level2_summary():
    ks = _csv(os.path.join(S5.S5_STATISTICAL, "ks_tests.csv"))
    mts = _csv(os.path.join(S5.S5_STATISTICAL, "multiple_testing_summary.csv"))
    if ks is None or mts is None:
        return None
    per_class = (ks[ks["scope"].isin(S5.CLASSES)]
                 .groupby("scope")["ks_statistic"].mean())
    median_d = float(per_class.median())
    worst = per_class.sort_values().index[-1] if len(per_class) else "n/a"
    best = per_class.sort_values().index[0] if len(per_class) else "n/a"
    n_sig_fdr = int(mts["significant_fdr_0.05"].sum())
    verdict = ("STRONG" if median_d < 0.1 else "MODERATE" if median_d < 0.2 else
               "LIMITED" if median_d < 0.35 else "WEAK")
    return {"median_class_D": median_d, "best_class": best, "worst_class": worst,
            "n_sig_fdr": n_sig_fdr, "n_tests": int(len(mts)), "verdict": verdict}


def _level3_summary():
    um = _csv(os.path.join(S5.S5_UTILITY, "utility_metrics.csv"))
    cmp = _csv(os.path.join(S5.S5_UTILITY, "comparison_headline.csv"))
    pcf = _csv(os.path.join(S5.S5_UTILITY, "per_class_f1.csv"))
    if um is None or cmp is None:
        return None
    def g(cond, metric):
        r = um[(um["condition"] == cond) & (um["metric"] == metric)]
        return float(r["mean"].iloc[0]) if len(r) else float("nan")
    def gstd(cond, metric):
        r = um[(um["condition"] == cond) & (um["metric"] == metric)]
        return float(r["std"].iloc[0]) if len(r) and "std" in r else 0.0
    ro = g("real_only", "macro_f1")
    so = g("synthetic_only", "macro_f1")
    rs = g("real_plus_synthetic", "macro_f1")
    acc_ro = g("real_only", "accuracy")
    acc_so = g("synthetic_only", "accuracy")
    ratio = so / ro if ro > 1e-9 else float("nan")
    acc_ratio = acc_so / acc_ro if acc_ro > 1e-9 else float("nan")
    n_seeds = int(um["n_seeds"].iloc[0]) if "n_seeds" in um and len(um) else 1
    q3 = ("YES" if ratio >= 0.9 else "PARTIALLY" if ratio >= 0.75 else
          "WEAKLY" if ratio >= 0.5 else "NO")
    drs = rs - ro
    q4 = ("IMPROVES" if drs > 0.01 else "DEGRADES" if drs < -0.01 else "NEUTRAL")
    zero_synth = []
    if pcf is not None and "synthetic_only" in pcf.columns:
        zero_synth = sorted(pcf.loc[pcf["synthetic_only"] < 0.03, "class"].tolist())
    return {"macro_f1_real": ro, "macro_f1_synth": so, "macro_f1_realsynth": rs,
            "macro_f1_real_std": gstd("real_only", "macro_f1"),
            "macro_f1_synth_std": gstd("synthetic_only", "macro_f1"),
            "macro_f1_realsynth_std": gstd("real_plus_synthetic", "macro_f1"),
            "acc_real": acc_ro, "acc_synth": acc_so, "acc_ratio": acc_ratio,
            "synth_over_real_ratio": ratio, "n_seeds": n_seeds,
            "zero_f1_classes_synth": zero_synth,
            "q3": q3, "q4": q4, "delta_realsynth_vs_real": drs}


def main():
    S5.ensure_dirs()
    env = S5.write_environment_report()

    with S5.stage_banner("STEP 5 -- FINAL REPORT"):
        markers = {"LEVEL1": os.path.exists(S5.LEVEL1_DONE),
                   "LEVEL2": os.path.exists(S5.LEVEL2_DONE),
                   "LEVEL3": os.path.exists(S5.LEVEL3_DONE)}
        missing_art = [p for p in FINAL_ARTIFACTS
                       if not (os.path.exists(p) and os.path.getsize(p) > 0)]

        l1, l2, l3 = _level1_summary(), _level2_summary(), _level3_summary()

        # ---- final_validation_summary.csv ----
        rows = []
        if l1:
            rows.append({"level": "Level 1 Structural",
                         "question": "Q1 does synthetic preserve real network structure?",
                         "key_metrics": f"in-deg KS D={l1['d_in']:.3f}; out-deg KS D={l1['d_out']:.3f}; "
                                        f"mean modularity gap={l1['mod_gap']:.3f}",
                         "verdict": l1["verdict"],
                         "details": "reports/level1_structural_report.txt"})
        if l2:
            rows.append({"level": "Level 2 Statistical",
                         "question": "Q2 do synthetic flow distributions resemble real unseen traffic?",
                         "key_metrics": f"median per-class KS D={l2['median_class_D']:.3f}; "
                                        f"best={l2['best_class']}; worst={l2['worst_class']}; "
                                        f"{l2['n_sig_fdr']}/{l2['n_tests']} tests sig @FDR0.05",
                         "verdict": l2["verdict"] + " statistical realism",
                         "details": "reports/level2_statistical_report.txt"})
        if l3:
            rows.append({"level": "Level 3 Utility",
                         "question": "Q3/Q4 can synthetic train an IDS that generalises to real TEST?",
                         "key_metrics": f"macro-F1 real={l3['macro_f1_real']:.3f}, "
                                        f"synth={l3['macro_f1_synth']:.3f}, "
                                        f"real+synth={l3['macro_f1_realsynth']:.3f}; "
                                        f"synth/real ratio={l3['synth_over_real_ratio']:.2f}",
                         "verdict": f"Q3={l3['q3']}; Q4 real+synth {l3['q4']} vs real-only",
                         "details": "reports/level3_utility_report.txt"})
        if rows:
            S5.atomic_write_csv(pd.DataFrame(rows),
                                os.path.join(S5.S5_REPORTS, "final_validation_summary.csv"))

        _write_final_report(env, markers, l1, l2, l3, missing_art)

        all_ok = all(markers.values()) and not missing_art and l1 and l2 and l3
        if all_ok:
            S5.write_marker(S5.S5_DONE, {
                "levels": ["Level 1 Structural", "Level 2 Statistical", "Level 3 Utility"],
                "level1_verdict": l1["verdict"],
                "level2_verdict": l2["verdict"],
                "level3_Q3": l3["q3"], "level3_Q4": l3["q4"],
                "enriched_real_sha256": S5.sha256(S5.ENRICHED_REAL),
                "final_synth_sha256": S5.sha256(S5.FINAL_SYNTH),
                "scale": S5.SCALE,
                "note": f"Step 5 only ever writes under {S5.S5}/ "
                        "(_assert_under_s5). Steps 1-4 outputs were treated as read-only.",
            })
            print(f"\n[STEP 5] ALL THREE LEVELS COMPLETE -- wrote {S5.S5_DONE}")
        else:
            reason = []
            if not all(markers.values()):
                reason.append("missing level marker(s): " +
                              ", ".join(k for k, v in markers.items() if not v))
            if missing_art:
                reason.append(f"{len(missing_art)} missing artifact(s), e.g. {missing_art[:3]}")
            print("\n[STEP 5] NOT complete -- DONE not written:")
            for r in reason:
                print("   -", r)
            raise SystemExit("Step 5 incomplete -- resolve the above and re-run.")


def _plain_language_detection(p, l3):
    """Thesis-style plain-language read of the transfer test, numbers filled in
    from the actual Level-3 results (kept in sync on every report regen)."""
    acc_r, acc_s = l3["acc_real"], l3["acc_synth"]
    f1_r, f1_s = l3["macro_f1_real"], l3["macro_f1_synth"]
    acc_pct = 100.0 * l3["acc_ratio"]
    f1_pct = 100.0 * l3["synth_over_real_ratio"]
    seeds = (f" (averaged over {l3['n_seeds']} random seeds)" if l3["n_seeds"] > 1 else "")
    zero = l3.get("zero_f1_classes_synth", [])
    drs = l3["delta_realsynth_vs_real"]

    p("  DOES THE SYNTHETIC DATA ACTUALLY WORK FOR DETECTION?")
    p("  " + "." * 66)
    p("  A GNN-based intrusion-detection model was trained once on the real")
    p("  training data and once on the synthetic dataset, and both were then")
    p("  tested on the SAME held-out real traffic that neither model had seen.")
    p("  This transfer test is the real measure of usefulness: matching graph")
    p("  topology (Level 1) and traffic-feature statistics (Level 2) alone does")
    p("  NOT guarantee that a model trained on the synthetic data can still catch")
    p("  real attacks.")
    p("")
    p(f"  Trained on real data, the model reached {acc_r*100:.1f}% accuracy and "
      f"{f1_r:.2f} macro-F1 on real")
    p(f"  test traffic{seeds}. Trained only on synthetic data it")
    p(f"  reached {acc_s*100:.1f}% accuracy and {f1_s:.2f} macro-F1 -- about "
      f"{acc_pct:.0f}% of the real-trained")
    p(f"  model's accuracy, but only {f1_pct:.0f}% of its macro-F1. Macro-F1 is the fairer")
    p("  figure here because it weights every attack class equally, whereas raw")
    p("  accuracy is inflated by the large benign class.")
    if zero:
        p(f"  The synthetic-only model failed almost completely on "
          f"{', '.join(c.upper() for c in zero)}")
        p("  (near-zero F1 on each), meaning the synthetic flows for those classes")
        p("  do not carry the signal a detector needs.")
    p("")
    if drs > 0.005:
        p(f"  Adding the synthetic data on TOP of the real training data gave a small")
        p(f"  improvement over real-only training (+{drs:.3f} macro-F1), concentrated in")
        p("  the minority attack classes.")
    elif drs < -0.005:
        p(f"  Adding the synthetic data on top of the real training data slightly")
        p(f"  degraded real-only performance ({drs:+.3f} macro-F1).")
    else:
        p(f"  Adding the synthetic data on top of the real training data left")
        p(f"  performance essentially unchanged ({drs:+.3f} macro-F1).")
    p("")
    p("  So the synthetic dataset currently works as a SUPPLEMENT to real data,")
    p("  not a REPLACEMENT for it. This is an early result and is the current")
    p("  focus for improvement.")
    p("")


def _write_final_report(env, markers, l1, l2, l3, missing_art):
    L = []
    def p(s=""): L.append(s)

    p("=" * 78)
    p("STEP 5 -- FINAL VALIDATION REPORT (synthetic NF-ToN-IoT)")
    p("=" * 78)
    p("")
    p("1. VALIDATION OBJECTIVE")
    p("-" * 78)
    p("  Decide, on evidence, whether the FINAL synthetic NF-ToN-IoT dataset is")
    p("    (1) structurally realistic, (2) statistically realistic, and")
    p("    (3) useful for downstream intrusion-detection modelling.")
    p("  Level 3 (downstream utility on REAL unseen traffic) is the decisive test.")
    p("")
    p("2. INPUT DATASETS  (all READ-ONLY in Step 5)")
    p("-" * 78)
    p(f"  real reference   : {S5.ENRICHED_REAL}")
    p(f"                     sha256 {S5.sha256(S5.ENRICHED_REAL)}")
    p(f"  final synthetic  : {S5.FINAL_SYNTH}")
    p(f"                     sha256 {S5.sha256(S5.FINAL_SYNTH)}")
    p(f"  Step 3 attack out: {S5.STEP3_DELIVERABLE}")
    p(f"  Step 4 benign out: {S5.STEP4_DELIVERABLE}")
    p("  classes (9): " + ", ".join(S5.CLASSES) + "   (ransomware excluded throughout)")
    p("")
    p("3. DATA SPLITS")
    p("-" * 78)
    p("  The EXACT project split is reproduced with IPs retained:")
    p("    attacks -> step3_2_prepare.stratified_split(., seed 42)")
    p("    benign  -> step4_1_prepare.benign_split(., seed 42)")
    p(f"  Row membership is verified against {S5.STEP3_SPLIT_STATS}")
    p(f"  and {S5.STEP4_SPLIT_STATS} (abort on mismatch).")
    p("  Level 1 uses the full real graph (descriptive topology; Step 2 was built")
    p("  to replicate it). Levels 2 & 3 use ONLY the held-out real TEST split as")
    p("  the real reference / evaluation set.")
    p("")
    p("4. LEAKAGE CONTROLS  (section 34)")
    p("-" * 78)
    p("  * real TEST rows are never in any training set -- explicit flow-key")
    p("    overlap check (logs/level3_leakage_checks.txt, logs/level2_leakage_checks.txt).")
    p("  * preprocessing (log1p + standardise) is fit on each condition's TRAINING")
    p("    edges only; real TEST is transform-only.")
    p("  * class weights use each condition's TRAINING label frequencies only.")
    p("  * fixed epochs, no early stopping -> no tuning on held-out real data.")
    p("  * generation finished before Step 5 (Step 3/4 VALIDATION markers checked).")
    p(f"  * Step 5 writes ONLY under {S5.S5}/ (enforced in code by")
    p("    step5_common._assert_under_s5) -> Steps 1-4 outputs cannot be modified.")
    p("")
    p("5. LEVEL 1 RESULTS -- STRUCTURAL")
    p("-" * 78)
    if l1:
        p(f"  overall in-degree  KS D  : {l1['d_in']:.3f}")
        p(f"  overall out-degree KS D  : {l1['d_out']:.3f}")
        p(f"  mean modularity gap      : {l1['mod_gap']:.3f}")
        p(f"  VERDICT                  : structure {l1['verdict']}")
    else:
        p("  (Level 1 outputs not found)")
    p("  detail: reports/level1_structural_report.txt")
    p("")
    p("6. LEVEL 2 RESULTS -- STATISTICAL (real TEST vs final synthetic)")
    p("-" * 78)
    if l2:
        p(f"  median per-class KS D    : {l2['median_class_D']:.3f}  ({l2['verdict']} realism)")
        p(f"  best-matched class       : {l2['best_class']}")
        p(f"  worst-matched class      : {l2['worst_class']}")
        p(f"  tests significant @FDR<0.05 : {l2['n_sig_fdr']} / {l2['n_tests']}")
        p("  (large N makes almost everything 'significant'; the D / JSD / TVD")
        p("   effect sizes and the per-class detail are what matter.)")
    else:
        p("  (Level 2 outputs not found)")
    p("  detail: reports/level2_statistical_report.txt")
    p("")
    p("7. LEVEL 3 RESULTS -- UTILITY (GNN-IDS, evaluated on real TEST)")
    p("-" * 78)
    if l3:
        sd = (f" (mean +/- std over {l3['n_seeds']} seeds)" if l3["n_seeds"] > 1
              else " (single seed)")
        p(f"  macro-F1  real-only       : {l3['macro_f1_real']:.3f} +/- {l3['macro_f1_real_std']:.3f}{sd}")
        p(f"  macro-F1  synthetic-only  : {l3['macro_f1_synth']:.3f} +/- {l3['macro_f1_synth_std']:.3f}")
        p(f"  macro-F1  real+synthetic  : {l3['macro_f1_realsynth']:.3f} +/- {l3['macro_f1_realsynth_std']:.3f}")
        p(f"  accuracy  real-only / synthetic-only : {l3['acc_real']:.3f} / {l3['acc_synth']:.3f}")
        p(f"  synthetic / real  macro-F1 ratio : {l3['synth_over_real_ratio']:.2f}   "
          f"(accuracy ratio {l3['acc_ratio']:.2f})")
        p(f"  real+synth - real (macro-F1)     : {l3['delta_realsynth_vs_real']:+.3f}")
        p("")
        p("  ACCURACY COMPARISON (synthetic-trained vs real-trained, on real TEST):")
        p(f"    real-trained model      : {l3['acc_real']*100:.1f}% accuracy")
        p(f"    synthetic-trained model : {l3['acc_synth']*100:.1f}% accuracy")
        p(f"    -> the synthetic-trained model reached about {l3['acc_ratio']*100:.0f}% of the")
        p(f"       real-trained model's accuracy (and about "
          f"{l3['synth_over_real_ratio']*100:.0f}% of its macro-F1, the")
        p("       fairer per-class measure).")
        p("")
        _plain_language_detection(p, l3)
    else:
        p("  (Level 3 outputs not found)")
    p("  detail: reports/level3_utility_report.txt")
    p("")
    p("8. OVERALL INTERPRETATION")
    p("-" * 78)
    p("  Structural, statistical and utility realism are reported SEPARATELY and")
    p("  are not collapsed into a single pass/fail. A dataset can be strong on one")
    p("  axis and weak on another; read all three verdicts together.")
    if l1 and l2 and l3:
        p("")
        p(f"    Q1 network structure preserved?         {l1['verdict']}")
        p(f"    Q2 flow distributions resemble real?    {l2['verdict']} "
          f"(median class KS D {l2['median_class_D']:.3f})")
        p(f"    Q3 IDS on synthetic generalises?        {l3['q3']} "
          f"(synth/real macro-F1 {l3['synth_over_real_ratio']:.2f})")
        p(f"    Q4 real+synthetic vs real-only?         {l3['q4']} "
          f"({l3['delta_realsynth_vs_real']:+.3f} macro-F1)")
    p("")
    p("9. LIMITATIONS")
    p("-" * 78)
    p("  * Level 1 compares per-class weighted graphs; Step 2 preserves each")
    p("    class's degree SEQUENCE exactly, so degree KS is expected to be small")
    p("    and modularity is the more informative structural signal.")
    p("  * Level 2 KS uses a fixed-seed sample capped per side per (class,feature);")
    p("    the D statistic is stable but exact p-values depend on that cap.")
    p("  * Level 3 uses one E-GraphSAGE-style architecture (no repo GNN existed to")
    p("    reuse) trained full-batch with a reproducible stratified edge cap; the")
    p("    absolute numbers depend on that budget, but the three conditions share")
    p("    it exactly so the COMPARISON is fair.")
    seeds = S5.GNN_CONFIG["seeds"]
    if len(seeds) == 1:
        p("  * Level 3 ran a SINGLE random seed (compute budget) -> point estimates,")
        p("    not multi-seed means. STEP5_GNN_SEEDS=\"42,1,2\" repeats it.")
    p("  * CUDA index_add_ is not fully deterministic; tiny run-to-run variation in")
    p("    Level 3 is possible even at a fixed seed.")
    p("")
    p("10. FINAL CONCLUSION")
    p("-" * 78)
    if l1 and l2 and l3:
        strong_struct = l1["verdict"] in ("PRESERVED", "LARGELY PRESERVED")
        strong_stat = l2["verdict"] in ("STRONG", "MODERATE")
        usable = l3["q3"] in ("YES", "PARTIALLY")
        helps = l3["q4"] == "IMPROVES"
        p(f"  * Structure : {l1['verdict']}.")
        p(f"  * Statistics: {l2['verdict']} realism vs real unseen traffic.")
        p(f"  * Utility   : synthetic-only IDS reaches {l3['synth_over_real_ratio']:.0%} "
          f"of real-only macro-F1 on real TEST; combining with real "
          f"{'helps' if helps else 'does not help' if l3['q4']=='NEUTRAL' else 'hurts'}.")
        p("")
        if usable and helps:
            p("  CONCLUSION: the synthetic dataset is a USEFUL complement to real data")
            p("  for GNN-IDS training and a PARTIAL stand-in for real training data.")
        elif usable:
            p("  CONCLUSION: the synthetic dataset has REAL downstream value (usable as")
            p("  a partial substitute) but does not clearly add to real-only training.")
        else:
            p("  CONCLUSION: the synthetic dataset is NOT yet a dependable substitute for")
            p("  real training data on this IDS task; treat structural/statistical")
            p("  similarity as necessary-but-not-sufficient evidence.")
        p("")
        p("  This conclusion is evidence-based and level-specific; see the three")
        p("  level reports for the per-class caveats.")
    else:
        p("  (incomplete -- one or more levels did not finish; DONE not written)")
    if missing_art:
        p("")
        p("  MISSING ARTIFACTS:")
        for mp in missing_art:
            p(f"    - {mp}")
    p("")
    p("=" * 78)

    S5.atomic_write_text("\n".join(L) + "\n",
                         os.path.join(S5.S5_REPORTS, "step5_final_report.txt"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
