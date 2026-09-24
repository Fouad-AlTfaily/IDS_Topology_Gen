# ============================================================
# STEP 5 -- LEVEL 1: STRUCTURAL VALIDATION
#
#   python -m src.validation.step5_1_structural
#
# Does the SYNTHETIC network preserve the REAL network's graph structure?
#
# Graph construction (identical for real & synthetic -- mirrors
# step1_fingerprint.build_weighted_graph):
#   * per attack class, one directed edge per ordered (IPV4_SRC_ADDR,
#     IPV4_DST_ADDR) pair, weight = number of flow rows for that pair
#   * unweighted in-/out-degree = number of distinct partners (Step 1 convention)
#   * Louvain on the weighted undirected projection, random_state = 42
#     (python-louvain, exactly Step 1)
#
# REAL graph  : results/step3/step3_1_enriched_real_flows.csv  (all 9 classes,
#               ransomware already excluded upstream). Level 1 is descriptive
#               topology only -- it trains nothing and fits no preprocessing --
#               so the full real graph is used (this is also the graph Step 2
#               was built to replicate). The real TEST split stays untouched by
#               Levels 2 and 3.
# SYNTH graph : results/final_synthetic/synthetic_NF_ToN_IoT.csv  (the FINAL
#               dataset, rebuilt end-to-end -- catches any merge corruption).
#
# Metrics: in-degree KS, out-degree KS, modularity (per class + pooled/summary),
# plus full descriptive statistics and practical (distributional) differences.
#
# Outputs (results/step5_validation/):
#   structural/degree_statistics.csv
#   structural/ks_degree_tests.csv
#   structural/modularity_results.csv
#   figures/real_vs_synthetic_in_degree.png   (+ copy in figures/structural/)
#   figures/real_vs_synthetic_out_degree.png  (+ copy)
#   figures/modularity_comparison.png         (+ copy)
#   reports/level1_structural_report.txt
#   structural/LEVEL1_DONE
# ============================================================

import os

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.validation import step5_common as S5

REQUIRED = [
    os.path.join(S5.S5_STRUCTURAL, "degree_statistics.csv"),
    os.path.join(S5.S5_STRUCTURAL, "ks_degree_tests.csv"),
    os.path.join(S5.S5_STRUCTURAL, "modularity_results.csv"),
    os.path.join(S5.S5_FIGURES, "real_vs_synthetic_in_degree.png"),
    os.path.join(S5.S5_FIGURES, "real_vs_synthetic_out_degree.png"),
    os.path.join(S5.S5_FIGURES, "modularity_comparison.png"),
    os.path.join(S5.S5_REPORTS, "level1_structural_report.txt"),
]

QUANTILES = [5, 25, 50, 75, 90, 95, 99]


def _save_fig(fig, name: str):
    top = os.path.join(S5.S5_FIGURES, name)
    S5._assert_under_s5(top)
    fig.savefig(top, dpi=120, bbox_inches="tight")
    fig.savefig(os.path.join(S5.S5_FIG_STRUCT, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def _desc(vec: np.ndarray) -> dict:
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0:
        return {**{k: float("nan") for k in
                   ["min", "max", "mean", "median", "std"] + [f"q{q}" for q in QUANTILES]},
                "n": 0}
    d = {"min": float(vec.min()), "max": float(vec.max()),
         "mean": float(vec.mean()), "median": float(np.median(vec)),
         "std": float(vec.std(ddof=1)) if vec.size > 1 else 0.0, "n": int(vec.size)}
    for q in QUANTILES:
        d[f"q{q}"] = float(np.percentile(vec, q))
    return d


def _degrees(G):
    ind = np.array([d for _, d in G.in_degree()], dtype=float)
    outd = np.array([d for _, d in G.out_degree()], dtype=float)
    return ind, outd


def _interp_ks(d: float) -> str:
    if np.isnan(d):
        return "insufficient nodes"
    if d < 0.1:
        return "very close (D<0.1)"
    if d < 0.2:
        return "close (D<0.2)"
    if d < 0.35:
        return "moderate divergence"
    return "large divergence"


def _ks_row(scope, metric, real_vec, syn_vec):
    real_vec = np.asarray(real_vec, float)
    syn_vec = np.asarray(syn_vec, float)
    if real_vec.size >= 2 and syn_vec.size >= 2:
        ks = stats.ks_2samp(real_vec, syn_vec, method="auto")
        ks_stat, p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, p = float("nan"), float("nan")
    rmean = float(np.mean(real_vec)) if real_vec.size else float("nan")
    smean = float(np.mean(syn_vec)) if syn_vec.size else float("nan")
    rmed = float(np.median(real_vec)) if real_vec.size else float("nan")
    smed = float(np.median(syn_vec)) if syn_vec.size else float("nan")
    return {
        "scope": scope, "metric": metric,
        "ks_statistic": round(ks_stat, 5) if not np.isnan(ks_stat) else np.nan,
        "p_value": p,
        "real_n": int(real_vec.size), "synth_n": int(syn_vec.size),
        "real_mean": round(rmean, 4), "synth_mean": round(smean, 4),
        "mean_diff": round(smean - rmean, 4),
        "real_median": round(rmed, 4), "synth_median": round(smed, 4),
        "median_diff": round(smed - rmed, 4),
        "practical_interpretation": _interp_ks(ks_stat),
    }


def _distribution_figure(real_vec, syn_vec, title, fname):
    real_vec = np.asarray(real_vec, float)
    syn_vec = np.asarray(syn_vec, float)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))

    hi = int(max(real_vec.max(), syn_vec.max())) if real_vec.size and syn_vec.size else 1
    bins = np.linspace(0, max(hi, 1), min(40, max(hi, 2)))
    ax[0].hist(real_vec, bins=bins, alpha=0.55, label=f"real (n={real_vec.size})",
               color="#1f77b4", density=True)
    ax[0].hist(syn_vec, bins=bins, alpha=0.55, label=f"synthetic (n={syn_vec.size})",
               color="#ff7f0e", density=True)
    ax[0].set_title(f"{title} - density"); ax[0].set_xlabel("degree")
    ax[0].set_ylabel("density"); ax[0].legend(); ax[0].set_yscale("log")

    for vec, lab, col in ((real_vec, "real", "#1f77b4"), (syn_vec, "synthetic", "#ff7f0e")):
        if vec.size:
            xs = np.sort(vec)
            ys = np.arange(1, xs.size + 1) / xs.size
            ax[1].step(xs, ys, where="post", label=lab, color=col, lw=2)
    ax[1].set_title(f"{title} - ECDF"); ax[1].set_xlabel("degree")
    ax[1].set_ylabel("cumulative fraction"); ax[1].legend()

    fig.suptitle(title)
    _save_fig(fig, fname)


def main():
    S5.ensure_dirs()
    S5.write_environment_report()

    if S5.stage_done(S5.LEVEL1_DONE, REQUIRED):
        print("[LEVEL 1] structural validation already complete -- SKIPPING")
        return

    with S5.stage_banner("LEVEL 1 -- STRUCTURAL VALIDATION"):
        print(f"  SCALE={S5.SCALE}   synthetic dataset: {S5.FINAL_SYNTH}")
        prov = S5.verify_inputs(require_synth=True)

        real = S5._read_enriched()[S5.IP_COLS + ["Attack"]].copy()
        synth = S5.load_final_synth()[S5.IP_COLS + ["Attack"]].copy()

        deg_stat_rows, ks_rows, mod_rows = [], [], []
        pooled = {("real", "in"): [], ("real", "out"): [],
                  ("synth", "in"): [], ("synth", "out"): []}

        for cls in S5.CLASSES:
            rG = S5.build_weighted_digraph(real[real["Attack"] == cls])
            sG = S5.build_weighted_digraph(synth[synth["Attack"] == cls])
            r_in, r_out = _degrees(rG)
            s_in, s_out = _degrees(sG)
            pooled[("real", "in")].append(r_in);  pooled[("real", "out")].append(r_out)
            pooled[("synth", "in")].append(s_in); pooled[("synth", "out")].append(s_out)

            for metric, rv, sv in (("in_degree", r_in, s_in), ("out_degree", r_out, s_out)):
                for src, vec in (("real", rv), ("synthetic", sv)):
                    deg_stat_rows.append({"scope": cls, "metric": metric, "source": src,
                                          **_desc(vec)})
                ks_rows.append(_ks_row(cls, metric, rv, sv))

            r_part, r_mod, r_nc, r_method = S5.louvain_partition_and_modularity(rG)
            s_part, s_mod, s_nc, s_method = S5.louvain_partition_and_modularity(sG)
            abs_diff = (abs(r_mod - s_mod) if not (np.isnan(r_mod) or np.isnan(s_mod))
                        else float("nan"))
            rel_diff = (abs_diff / abs(r_mod) if (not np.isnan(abs_diff) and abs(r_mod) > 1e-9)
                        else float("nan"))
            mod_rows.append({
                "scope": cls,
                "real_nodes": rG.number_of_nodes(), "synth_nodes": sG.number_of_nodes(),
                "real_edges": rG.number_of_edges(), "synth_edges": sG.number_of_edges(),
                "real_modularity": round(r_mod, 5) if not np.isnan(r_mod) else np.nan,
                "synth_modularity": round(s_mod, 5) if not np.isnan(s_mod) else np.nan,
                "abs_modularity_diff": round(abs_diff, 5) if not np.isnan(abs_diff) else np.nan,
                "rel_modularity_diff": round(rel_diff, 4) if not np.isnan(rel_diff) else np.nan,
                "real_n_communities": r_nc, "synth_n_communities": s_nc,
                "louvain_method": r_method if r_method == s_method else f"{r_method}/{s_method}",
            })
            print(f"  {cls:<10} nodes r/s {rG.number_of_nodes()}/{sG.number_of_nodes()}  "
                  f"edges r/s {rG.number_of_edges()}/{sG.number_of_edges()}  "
                  f"modularity r/s {r_mod:.3f}/{s_mod:.3f}")

        # ---- pooled ("overall") degree distributions ----
        for who in ("real", "synth"):
            for m in ("in", "out"):
                pooled[(who, m)] = (np.concatenate(pooled[(who, m)])
                                    if pooled[(who, m)] else np.array([]))
        for metric, rk, sk in (("in_degree", ("real", "in"), ("synth", "in")),
                               ("out_degree", ("real", "out"), ("synth", "out"))):
            rv, sv = pooled[rk], pooled[sk]
            for src, vec in (("real", rv), ("synthetic", sv)):
                deg_stat_rows.append({"scope": "overall", "metric": metric, "source": src,
                                      **_desc(vec)})
            ks_rows.append(_ks_row("overall", metric, rv, sv))

        # ---- modularity summary row (mean over classes) ----
        md = pd.DataFrame(mod_rows)
        sub = md[md["scope"].isin(S5.CLASSES)]
        mod_rows.append({
            "scope": "summary_mean",
            "real_nodes": int(sub["real_nodes"].sum()),
            "synth_nodes": int(sub["synth_nodes"].sum()),
            "real_edges": int(sub["real_edges"].sum()),
            "synth_edges": int(sub["synth_edges"].sum()),
            "real_modularity": round(float(np.nanmean(sub["real_modularity"])), 5),
            "synth_modularity": round(float(np.nanmean(sub["synth_modularity"])), 5),
            "abs_modularity_diff": round(float(np.nanmean(sub["abs_modularity_diff"])), 5),
            "rel_modularity_diff": round(float(np.nanmean(sub["rel_modularity_diff"])), 4),
            "real_n_communities": int(sub["real_n_communities"].sum()),
            "synth_n_communities": int(sub["synth_n_communities"].sum()),
            "louvain_method": "mean over classes",
        })

        deg_df = pd.DataFrame(deg_stat_rows)
        ks_df = pd.DataFrame(ks_rows)
        mod_df = pd.DataFrame(mod_rows)
        S5.atomic_write_csv(deg_df, os.path.join(S5.S5_STRUCTURAL, "degree_statistics.csv"))
        S5.atomic_write_csv(ks_df, os.path.join(S5.S5_STRUCTURAL, "ks_degree_tests.csv"))
        S5.atomic_write_csv(mod_df, os.path.join(S5.S5_STRUCTURAL, "modularity_results.csv"))

        # ---- figures ----
        _distribution_figure(pooled[("real", "in")], pooled[("synth", "in")],
                             "In-degree: real vs synthetic (all classes pooled)",
                             "real_vs_synthetic_in_degree.png")
        _distribution_figure(pooled[("real", "out")], pooled[("synth", "out")],
                             "Out-degree: real vs synthetic (all classes pooled)",
                             "real_vs_synthetic_out_degree.png")

        fig, ax = plt.subplots(figsize=(11, 4.5))
        cls_rows = mod_df[mod_df["scope"].isin(S5.CLASSES)]
        x = np.arange(len(cls_rows))
        ax.bar(x - 0.2, cls_rows["real_modularity"], 0.4, label="real", color="#1f77b4")
        ax.bar(x + 0.2, cls_rows["synth_modularity"], 0.4, label="synthetic", color="#ff7f0e")
        ax.set_xticks(x); ax.set_xticklabels(cls_rows["scope"], rotation=30, ha="right")
        ax.set_ylabel("Louvain modularity (weighted)")
        ax.set_title("Modularity: real vs synthetic per attack class")
        ax.legend()
        _save_fig(fig, "modularity_comparison.png")

        # ---- report ----
        _write_report(deg_df, ks_df, mod_df, prov)

        S5.write_marker(S5.LEVEL1_DONE, {
            "classes": S5.CLASSES,
            "graph_rule": "per-class weighted DiGraph, edge=(src,dst) IP pair, "
                          "weight=#flows; unweighted degree; Louvain "
                          f"random_state={S5.LOUVAIN_SEED} (Step-1 methodology)",
            "real_source": S5.ENRICHED_REAL,
            "synth_source": S5.FINAL_SYNTH,
        })
        print("[LEVEL 1] done ->", S5.S5_STRUCTURAL)


def _verdict_from_ks(overall_in, overall_out):
    d = max(overall_in, overall_out)
    if d < 0.1:
        return "PRESERVED", "degree distributions match very closely (max D < 0.10)"
    if d < 0.2:
        return "LARGELY PRESERVED", "degree distributions are close (max D < 0.20)"
    if d < 0.35:
        return "PARTIALLY PRESERVED", "moderate degree-distribution divergence (0.20 <= max D < 0.35)"
    return "NOT PRESERVED", "large degree-distribution divergence (max D >= 0.35)"


def _write_report(deg_df, ks_df, mod_df, prov):
    L = []
    def p(s=""): L.append(s)

    ov = ks_df[ks_df["scope"] == "overall"].set_index("metric")
    ov_in = float(ov.loc["in_degree", "ks_statistic"])
    ov_out = float(ov.loc["out_degree", "ks_statistic"])
    verdict, why = _verdict_from_ks(ov_in, ov_out)
    summ = mod_df[mod_df["scope"] == "summary_mean"].iloc[0]

    p("=" * 74)
    p("STEP 5 -- LEVEL 1: STRUCTURAL VALIDATION REPORT")
    p("=" * 74)
    p("")
    p("OBJECTIVE: does the synthetic network preserve the real network's graph")
    p("structure (in-degree, out-degree, modularity)?")
    p("")
    p("GRAPH CONSTRUCTION (identical for real and synthetic):")
    p("  * per attack class: directed graph, one edge per ordered (src IP, dst IP)")
    p("    pair, edge weight = number of flow rows for that pair (Step-1 rule).")
    p("  * degree = unweighted in-/out-degree (distinct partner count).")
    p(f"  * modularity = weighted Louvain on the undirected projection, "
      f"random_state={S5.LOUVAIN_SEED}.")
    p(f"  REAL  graph  <- {S5.ENRICHED_REAL}  (all 9 classes; ransomware already excluded)")
    p(f"  SYNTH graph  <- {S5.FINAL_SYNTH}  (final dataset, rebuilt end-to-end)")
    p(f"  final synthetic sha256: {prov.get('final_synthetic', {}).get('sha256', 'n/a')}")
    p("")
    p("-" * 74)
    p("DEGREE DISTRIBUTION -- KS (Kolmogorov-Smirnov, 2-sample)")
    p("-" * 74)
    p(ks_df.to_string(index=False))
    p("")
    p("  Do NOT read the p-values alone: with few nodes per class the test has")
    p("  little power. The KS D statistic is the practical distance; mean_diff /")
    p("  median_diff give the direction and size of the shift.")
    p("")
    p("-" * 74)
    p("DESCRIPTIVE DEGREE STATISTICS (real vs synthetic)")
    p("-" * 74)
    p(deg_df.to_string(index=False))
    p("")
    p("-" * 74)
    p("MODULARITY")
    p("-" * 74)
    p(mod_df.to_string(index=False))
    p("")
    p(f"  mean real modularity  : {summ['real_modularity']:.4f}")
    p(f"  mean synth modularity : {summ['synth_modularity']:.4f}")
    p(f"  mean |difference|     : {summ['abs_modularity_diff']:.4f}"
      f"   (mean relative: {summ['rel_modularity_diff']})")
    p("")
    p("  Note: Step 2 preserves each class's degree SEQUENCE exactly (configuration")
    p("  model) but re-wires endpoints at random, so some modularity drift is")
    p("  expected and is the informative signal here.")
    p("")
    p("-" * 74)
    p("LEVEL 1 VERDICT")
    p("-" * 74)
    p(f"  overall in-degree  KS D = {ov_in:.3f}")
    p(f"  overall out-degree KS D = {ov_out:.3f}")
    p(f"  mean modularity gap     = {summ['abs_modularity_diff']:.3f}")
    p("")
    p(f"  STRUCTURAL SIMILARITY: {verdict}")
    p(f"  ({why}; modularity gap "
      f"{'small' if summ['abs_modularity_diff'] < 0.1 else 'moderate' if summ['abs_modularity_diff'] < 0.2 else 'large'}).")
    p("")
    p("  Q1 (does the synthetic network preserve real network structure?) is")
    p("  answered by the combination above -- see step5_final_report.txt for the")
    p("  cross-level synthesis.")
    p("=" * 74)

    S5.atomic_write_text("\n".join(L) + "\n",
                         os.path.join(S5.S5_REPORTS, "level1_structural_report.txt"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
