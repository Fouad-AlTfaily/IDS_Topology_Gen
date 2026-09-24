# ============================================================
# STEP 5 -- LEVEL 2: FLOW STATISTICAL VALIDATION
#
#   python -m src.validation.step5_2_statistical
#
# Do the synthetic flow feature distributions resemble REAL UNSEEN traffic?
#
#   PRIMARY comparison:  REAL TEST  vs  FINAL SYNTHETIC DATASET
#   (never real TRAIN as the main reference -- section 11).
#
# Scope: the 9 CTGAN-generated flow features ONLY. IP / degree / role columns
# are contextual and are NOT mixed into the primary flow comparison.
#
#   numeric / ordinal (KS, 2-sample Kolmogorov-Smirnov):
#       IN_BYTES OUT_BYTES IN_PKTS OUT_PKTS FLOW_DURATION_MILLISECONDS
#       L4_SRC_PORT L4_DST_PORT
#     -> ports are integer-valued network fields; they are compared as numeric
#        with KS (documented here). Their top-value frequencies are also
#        reported as a supplement in categorical_tests.csv.
#   categorical / discrete codes:
#       PROTOCOL TCP_FLAGS
#     -> chi-square on aligned category counts + Jensen-Shannon divergence +
#        total-variation distance; category frequencies & proportions reported.
#
# Per class: benign + the 8 attack classes (ransomware excluded). Plus a pooled
# "overall" comparison. Every test carries raw p AND Benjamini-Hochberg FDR
# adjusted p, plus the practical effect (KS D, mean/median/quantile diffs).
#
# Outputs (results/step5_validation/):
#   statistical/ks_tests.csv
#   statistical/categorical_tests.csv
#   statistical/descriptive_statistics.csv
#   statistical/multiple_testing_summary.csv
#   figures/statistical/ecdf_numeric_features.png
#   figures/statistical/box_numeric_features.png
#   figures/statistical/perclass_ks_heatmap.png
#   figures/statistical/categorical_PROTOCOL.png
#   figures/statistical/categorical_TCP_FLAGS.png
#   reports/level2_statistical_report.txt
#   statistical/LEVEL2_DONE
# ============================================================

import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.validation import step5_common as S5

REQUIRED = [
    os.path.join(S5.S5_STATISTICAL, "ks_tests.csv"),
    os.path.join(S5.S5_STATISTICAL, "categorical_tests.csv"),
    os.path.join(S5.S5_STATISTICAL, "descriptive_statistics.csv"),
    os.path.join(S5.S5_STATISTICAL, "multiple_testing_summary.csv"),
    os.path.join(S5.S5_FIG_STAT, "ecdf_numeric_features.png"),
    os.path.join(S5.S5_FIG_STAT, "box_numeric_features.png"),
    os.path.join(S5.S5_FIG_STAT, "perclass_ks_heatmap.png"),
    os.path.join(S5.S5_FIG_STAT, "categorical_PROTOCOL.png"),
    os.path.join(S5.S5_FIG_STAT, "categorical_TCP_FLAGS.png"),
    os.path.join(S5.S5_REPORTS, "level2_statistical_report.txt"),
]

MAX_KS_SAMPLE = S5.CFG["validation"]["level2_statistical"]["max_ks_sample"]  # 50_000 per side, per (class,feature) -- seeded; documented
QUANTS = [5, 25, 50, 75, 90, 95, 99]
ALPHA = S5.CFG["validation"]["level2_statistical"]["alpha"]  # 0.05


# ------------------------------------------------------------------
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjusted p-values (same as statsmodels
    multipletests(method='fdr_bh')). NaNs are ignored and returned as NaN."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    n = q.size
    if n == 0:
        return out
    order = np.argsort(q)
    ranked = q[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    res = np.empty(n)
    res[order] = adj
    out[ok] = res
    return out


def _desc(vec: np.ndarray) -> dict:
    vec = np.asarray(vec, float)
    vec = vec[np.isfinite(vec)]
    if vec.size == 0:
        d = {k: float("nan") for k in ["min", "max", "mean", "median", "std"]}
        d.update({f"q{q}": float("nan") for q in QUANTS})
        d["n"] = 0
        return d
    d = {"min": float(vec.min()), "max": float(vec.max()),
         "mean": float(vec.mean()), "median": float(np.median(vec)),
         "std": float(vec.std(ddof=1)) if vec.size > 1 else 0.0, "n": int(vec.size)}
    for q in QUANTS:
        d[f"q{q}"] = float(np.percentile(vec, q))
    return d


def _sample(vec: np.ndarray, n: int, seed: int) -> np.ndarray:
    vec = np.asarray(vec, float)
    vec = vec[np.isfinite(vec)]
    if vec.size <= n:
        return vec
    rng = np.random.default_rng(seed)
    return vec[rng.choice(vec.size, size=n, replace=False)]


def _interp_ks(d: float) -> str:
    if np.isnan(d):
        return "n/a"
    if d < 0.05:
        return "excellent match (D<0.05)"
    if d < 0.10:
        return "very good match (D<0.10)"
    if d < 0.20:
        return "good match (D<0.20)"
    if d < 0.35:
        return "moderate divergence"
    return "large divergence"


def _ks_one(scope, feature, real_vec, syn_vec, seed):
    r_full = np.asarray(real_vec, float); r_full = r_full[np.isfinite(r_full)]
    s_full = np.asarray(syn_vec, float);  s_full = s_full[np.isfinite(s_full)]
    r = _sample(r_full, MAX_KS_SAMPLE, seed)
    s = _sample(s_full, MAX_KS_SAMPLE, seed + 1)
    if r.size >= 2 and s.size >= 2:
        k = stats.ks_2samp(r, s, method="auto")
        d, p = float(k.statistic), float(k.pvalue)
    else:
        d, p = float("nan"), float("nan")
    rd, sd = _desc(r_full), _desc(s_full)
    row = {
        "scope": scope, "feature": feature, "test": "KS_2sample",
        "ks_statistic": round(d, 5), "p_value_raw": p,
        "real_n_total": rd["n"], "synth_n_total": sd["n"],
        "real_n_used": int(r.size), "synth_n_used": int(s.size),
        "real_mean": rd["mean"], "synth_mean": sd["mean"],
        "mean_diff": sd["mean"] - rd["mean"],
        "real_median": rd["median"], "synth_median": sd["median"],
        "median_diff": sd["median"] - rd["median"],
        "real_std": rd["std"], "synth_std": sd["std"],
    }
    for q in QUANTS:
        row[f"q{q}_diff"] = sd[f"q{q}"] - rd[f"q{q}"]
    row["practical_interpretation"] = _interp_ks(d)
    return row


def _categorical_one(scope, feature, real_ser, syn_ser):
    r = real_ser.dropna().astype(str)
    s = syn_ser.dropna().astype(str)
    cats = sorted(set(r.unique()) | set(s.unique()),
                  key=lambda x: (len(x), x))
    rc = r.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    sc = s.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    rp = rc / rc.sum() if rc.sum() else rc
    sp = sc / sc.sum() if sc.sum() else sc

    table = np.vstack([rc, sc])
    keep = table.sum(axis=0) > 0
    if keep.sum() >= 2 and rc.sum() > 0 and sc.sum() > 0:
        chi2, p, dof, _ = stats.chi2_contingency(table[:, keep])
    else:
        chi2, p, dof = float("nan"), float("nan"), 0
    jsd = float(jensenshannon(rp, sp, base=2)) if rp.sum() and sp.sum() else float("nan")
    tvd = float(0.5 * np.abs(rp - sp).sum())

    rows = []
    for c, a, b, pa, pb in zip(cats, rc, sc, rp, sp):
        rows.append({
            "scope": scope, "feature": feature, "category": c,
            "real_count": int(a), "synth_count": int(b),
            "real_proportion": round(float(pa), 6), "synth_proportion": round(float(pb), 6),
            "proportion_diff": round(float(pb - pa), 6),
            "test": "chi2_contingency",
            "chi2": round(float(chi2), 4) if not np.isnan(chi2) else np.nan,
            "dof": dof, "p_value_raw": p,
            "jensen_shannon_distance": round(jsd, 5) if not np.isnan(jsd) else np.nan,
            "total_variation_distance": round(tvd, 5),
            "n_categories": int(len(cats)),
        })
    summary = {
        "scope": scope, "feature": feature, "category": "__ALL__",
        "real_count": int(rc.sum()), "synth_count": int(sc.sum()),
        "real_proportion": 1.0, "synth_proportion": 1.0, "proportion_diff": 0.0,
        "test": "chi2_contingency",
        "chi2": round(float(chi2), 4) if not np.isnan(chi2) else np.nan,
        "dof": dof, "p_value_raw": p,
        "jensen_shannon_distance": round(jsd, 5) if not np.isnan(jsd) else np.nan,
        "total_variation_distance": round(tvd, 5),
        "n_categories": int(len(cats)),
    }
    return rows, summary, (cats, rp, sp)


# ------------------------------------------------------------------
def _fig_ecdf(real_df, syn_df):
    feats = S5.NUMERIC_KS_FEATURES
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, f in enumerate(feats):
        ax = axes[i]
        rv = _sample(real_df[f].to_numpy(float), 40_000, 7)
        sv = _sample(syn_df[f].to_numpy(float), 40_000, 8)
        for vec, lab, col in ((rv, "real TEST", "#1f77b4"), (sv, "synthetic", "#ff7f0e")):
            if vec.size:
                xs = np.sort(vec)
                ys = np.arange(1, xs.size + 1) / xs.size
                ax.step(xs, ys, where="post", label=lab, color=col, lw=1.8)
        ax.set_title(f); ax.set_ylabel("ECDF")
        if rv.size and rv.max() > 0 and np.percentile(rv[rv > 0] if (rv > 0).any() else rv, 95) > 100:
            ax.set_xscale("symlog")
        ax.legend(fontsize=8)
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Level 2 -- ECDF: real TEST vs final synthetic (all classes pooled)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_STAT, "ecdf_numeric_features.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_box(real_df, syn_df):
    feats = S5.NUMERIC_KS_FEATURES
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    for i, f in enumerate(feats):
        ax = axes[i]
        rv = _sample(real_df[f].to_numpy(float), 40_000, 11)
        sv = _sample(syn_df[f].to_numpy(float), 40_000, 12)
        ax.boxplot([rv, sv], showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["real TEST", "synthetic"])
        ax.set_title(f)
        if rv.size and np.nanmax(rv) > 1000:
            ax.set_yscale("symlog")
    for j in range(len(feats), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Level 2 -- boxplots (no outliers): real TEST vs final synthetic",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_STAT, "box_numeric_features.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_heatmap(ks_df):
    feats = S5.NUMERIC_KS_FEATURES
    classes = S5.CLASSES
    M = np.full((len(classes), len(feats)), np.nan)
    for i, c in enumerate(classes):
        for j, f in enumerate(feats):
            r = ks_df[(ks_df["scope"] == c) & (ks_df["feature"] == f)]
            if len(r):
                M[i, j] = r["ks_statistic"].iloc[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels(feats, rotation=40, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(feats)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="KS D statistic (lower = closer)")
    ax.set_title("Level 2 -- per-class KS D: real TEST vs final synthetic")
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_STAT, "perclass_ks_heatmap.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_categorical(feature, real_df, syn_df):
    r = real_df[feature].dropna().astype(str)
    s = syn_df[feature].dropna().astype(str)
    cats = [c for c, _ in
            sorted(((c, (r == c).sum() + (s == c).sum()) for c in set(r) | set(s)),
                   key=lambda kv: -kv[1])][:15]
    rp = np.array([(r == c).mean() for c in cats])
    sp = np.array([(s == c).mean() for c in cats])
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(8, len(cats) * 0.8), 4.5))
    ax.bar(x - 0.2, rp, 0.4, label="real TEST", color="#1f77b4")
    ax.bar(x + 0.2, sp, 0.4, label="synthetic", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=40, ha="right")
    ax.set_ylabel("proportion"); ax.set_title(f"Level 2 -- {feature} category proportions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_STAT, f"categorical_{feature}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
def main():
    S5.ensure_dirs()
    S5.write_environment_report()

    if S5.stage_done(S5.LEVEL2_DONE, REQUIRED):
        print("[LEVEL 2] statistical validation already complete -- SKIPPING")
        return

    with S5.stage_banner("LEVEL 2 -- FLOW STATISTICAL VALIDATION"):
        print(f"  SCALE={S5.SCALE}   synthetic dataset: {S5.FINAL_SYNTH}")
        S5.verify_inputs(require_synth=True)
        split = S5.load_real_split()
        real_test = split["test"].copy()
        synth = S5.load_final_synth().copy()

        # leakage guard: this comparison must use real TEST, never real TRAIN
        S5.assert_no_leakage(
            {"final_synthetic": synth}, real_test, split,
            log_path=os.path.join(S5.S5_LOGS, "level2_leakage_checks.txt"))

        for c in S5.FLOW_FEATURES:
            real_test[c] = pd.to_numeric(real_test[c], errors="coerce")
            synth[c] = pd.to_numeric(synth[c], errors="coerce")

        ks_rows, desc_rows, cat_rows, cat_summ_rows = [], [], [], []
        scopes = ["overall"] + S5.CLASSES

        for scope in scopes:
            rt = real_test if scope == "overall" else real_test[real_test["Attack"] == scope]
            st = synth if scope == "overall" else synth[synth["Attack"] == scope]
            seed = S5.RANDOM_SEED + (0 if scope == "overall" else S5.CLASSES.index(scope) + 1)

            for f in S5.FLOW_FEATURES:
                desc_rows.append({"scope": scope, "feature": f, "source": "real_test",
                                  **_desc(rt[f].to_numpy(float))})
                desc_rows.append({"scope": scope, "feature": f, "source": "synthetic",
                                  **_desc(st[f].to_numpy(float))})

            for f in S5.NUMERIC_KS_FEATURES:
                ks_rows.append(_ks_one(scope, f, rt[f].to_numpy(float),
                                       st[f].to_numpy(float), seed))

            for f in S5.CATEGORICAL_FEATURES:
                rows, summ, _ = _categorical_one(scope, f, rt[f], st[f])
                cat_rows.extend(rows)
                cat_summ_rows.append(summ)

            # supplementary: ports as top-value frequencies (documented)
            for f in ("L4_SRC_PORT", "L4_DST_PORT"):
                rows, summ, _ = _categorical_one(scope, f + "_topvalues",
                                                 rt[f].round().astype("Int64"),
                                                 st[f].round().astype("Int64"))
                # keep only the 20 most common to avoid a giant table
                rows = sorted(rows, key=lambda d: -(d["real_count"] + d["synth_count"]))[:20]
                cat_rows.extend(rows)
                summ["note"] = "supplementary: ports summarised as discrete top-values; " \
                               "primary port test is KS (see ks_tests.csv)"
                cat_summ_rows.append(summ)

        ks_df = pd.DataFrame(ks_rows)
        cat_df = pd.DataFrame(cat_rows)
        cat_summ_df = pd.DataFrame(cat_summ_rows)
        desc_df = pd.DataFrame(desc_rows)

        # ---- multiple-testing correction across the whole test collection ----
        ks_df["family"] = "KS_numeric"
        cat_summ_df["family"] = "categorical_chi2"
        ptable = pd.concat([
            ks_df[["scope", "feature", "family", "p_value_raw"]],
            cat_summ_df[["scope", "feature", "family", "p_value_raw"]],
        ], ignore_index=True)
        ptable["p_value_bh_fdr"] = _bh_fdr(ptable["p_value_raw"].to_numpy())
        ptable["significant_raw_0.05"] = ptable["p_value_raw"] < ALPHA
        ptable["significant_fdr_0.05"] = ptable["p_value_bh_fdr"] < ALPHA

        merge_keys = ["scope", "feature", "family"]
        ks_df = ks_df.merge(ptable[merge_keys + ["p_value_bh_fdr",
                                                 "significant_raw_0.05",
                                                 "significant_fdr_0.05"]],
                            on=merge_keys, how="left")
        cat_summ_df = cat_summ_df.merge(ptable[merge_keys + ["p_value_bh_fdr",
                                                            "significant_raw_0.05",
                                                            "significant_fdr_0.05"]],
                                        on=merge_keys, how="left")
        # propagate adjusted p onto the per-category rows too
        cat_df = cat_df.merge(
            cat_summ_df[["scope", "feature", "p_value_bh_fdr",
                         "significant_fdr_0.05"]].rename(
                columns={"p_value_bh_fdr": "p_value_bh_fdr_feature"}),
            on=["scope", "feature"], how="left")

        S5.atomic_write_csv(ks_df, os.path.join(S5.S5_STATISTICAL, "ks_tests.csv"))
        S5.atomic_write_csv(pd.concat([cat_df], ignore_index=True),
                            os.path.join(S5.S5_STATISTICAL, "categorical_tests.csv"))
        S5.atomic_write_csv(desc_df,
                            os.path.join(S5.S5_STATISTICAL, "descriptive_statistics.csv"))

        mts = ptable.copy()
        mts["abs_effect"] = np.nan
        eff = ks_df.set_index(["scope", "feature"])["ks_statistic"]
        for i, r in mts.iterrows():
            if r["family"] == "KS_numeric":
                try:
                    mts.at[i, "abs_effect"] = float(eff.loc[(r["scope"], r["feature"])])
                except Exception:
                    pass
            else:
                v = cat_summ_df[(cat_summ_df["scope"] == r["scope"]) &
                                (cat_summ_df["feature"] == r["feature"])]
                if len(v):
                    mts.at[i, "abs_effect"] = float(v["total_variation_distance"].iloc[0])
        S5.atomic_write_csv(mts, os.path.join(S5.S5_STATISTICAL,
                                              "multiple_testing_summary.csv"))

        # ---- figures ----
        _fig_ecdf(real_test, synth)
        _fig_box(real_test, synth)
        _fig_heatmap(ks_df)
        for f in S5.CATEGORICAL_FEATURES:
            _fig_categorical(f, real_test, synth)

        _write_report(ks_df, cat_summ_df, desc_df, mts, real_test, synth)

        S5.write_marker(S5.LEVEL2_DONE, {
            "primary_reference": "real TEST split (held out) vs FINAL synthetic dataset",
            "numeric_test": "KS 2-sample (ports included as numeric)",
            "categorical_test": "chi-square on aligned counts + JSD + total-variation distance",
            "multiple_testing": "Benjamini-Hochberg FDR across all class x feature tests",
            "ks_sample_cap_per_side": MAX_KS_SAMPLE,
        })
        print("[LEVEL 2] done ->", S5.S5_STATISTICAL)


def _write_report(ks_df, cat_summ_df, desc_df, mts, real_test, synth):
    L = []
    def p(s=""): L.append(s)

    ov = ks_df[ks_df["scope"] == "overall"].sort_values("ks_statistic")
    per_class_mean_d = (ks_df[ks_df["scope"].isin(S5.CLASSES)]
                        .groupby("scope")["ks_statistic"].mean().sort_values())
    best_feat = ov.iloc[0] if len(ov) else None
    worst_feat = ov.iloc[-1] if len(ov) else None
    n_tests = len(mts)
    n_sig_raw = int(mts["significant_raw_0.05"].sum())
    n_sig_fdr = int(mts["significant_fdr_0.05"].sum())
    median_d = float(ks_df[ks_df["scope"] != "overall"]["ks_statistic"].median())

    p("=" * 74)
    p("STEP 5 -- LEVEL 2: FLOW STATISTICAL VALIDATION REPORT")
    p("=" * 74)
    p("")
    p("OBJECTIVE: do the FINAL synthetic dataset's flow feature distributions")
    p("resemble REAL UNSEEN traffic (the held-out real TEST split)?")
    p("")
    p("REFERENCE : real TEST split (reconstructed deterministically; never used")
    p("            to fit anything).  rows = "
      f"{len(real_test):,}")
    p(f"SYNTHETIC : {S5.FINAL_SYNTH}   rows = {len(synth):,}")
    p("")
    p("TESTS:")
    p("  * numeric/ordinal (IN_BYTES OUT_BYTES IN_PKTS OUT_PKTS")
    p("    FLOW_DURATION_MILLISECONDS L4_SRC_PORT L4_DST_PORT): 2-sample KS.")
    p("    Ports are integer-valued network fields and are compared as numeric")
    p("    with KS; their discrete top-value frequencies are also tabulated in")
    p("    categorical_tests.csv for transparency.")
    p("  * categorical codes (PROTOCOL TCP_FLAGS): chi-square on aligned category")
    p("    counts, plus Jensen-Shannon distance and total-variation distance as")
    p("    scale-free effect sizes; full frequencies & proportions are reported.")
    p(f"  * KS is computed on up to {MAX_KS_SAMPLE:,} rows per side per (class,feature),")
    p("    randomly sampled with a fixed seed, to bound runtime and keep p-values")
    p("    meaningful at large N. Full real/synth sample sizes are also reported.")
    p("  * Multiple testing: Benjamini-Hochberg FDR across ALL class x feature")
    p("    tests. Raw AND adjusted p-values are both reported.")
    p("")
    p("-" * 74)
    p("OVERALL (all classes pooled) -- KS")
    p("-" * 74)
    p(ov[["feature", "ks_statistic", "p_value_raw", "p_value_bh_fdr",
          "real_n_total", "synth_n_total", "mean_diff", "median_diff",
          "practical_interpretation"]].to_string(index=False))
    p("")
    p("-" * 74)
    p("PER-CLASS mean KS D (lower = closer); full detail in ks_tests.csv")
    p("-" * 74)
    for cls, d in per_class_mean_d.items():
        p(f"  {cls:<10} mean D = {d:.3f}   ({_interp_ks(d)})")
    p("")
    p("-" * 74)
    p("CATEGORICAL (PROTOCOL / TCP_FLAGS) -- distribution distance")
    p("-" * 74)
    p(cat_summ_df[cat_summ_df["category"] == "__ALL__"][
        ["scope", "feature", "chi2", "p_value_raw", "p_value_bh_fdr",
         "jensen_shannon_distance", "total_variation_distance"]].to_string(index=False))
    p("")
    p("-" * 74)
    p("SIGNIFICANCE vs PRACTICAL SIZE")
    p("-" * 74)
    p(f"  hypothesis tests run          : {n_tests}")
    p(f"  significant at raw p<0.05     : {n_sig_raw}  ({100*n_sig_raw/max(n_tests,1):.0f}%)")
    p(f"  significant at BH-FDR p<0.05  : {n_sig_fdr}  ({100*n_sig_fdr/max(n_tests,1):.0f}%)")
    p(f"  median KS D across per-class tests : {median_d:.3f}")
    p("")
    p("  With tens of thousands of rows per class, KS/chi-square will flag even")
    p("  tiny differences as 'significant'. The D statistic, JSD, TVD and the")
    p("  mean/median/quantile differences are the practically meaningful read-out.")
    p("")
    if best_feat is not None:
        p(f"  strongest overall feature match : {best_feat['feature']} "
          f"(D={best_feat['ks_statistic']:.3f})")
    if worst_feat is not None:
        p(f"  largest overall discrepancy     : {worst_feat['feature']} "
          f"(D={worst_feat['ks_statistic']:.3f})")
    if len(per_class_mean_d):
        p(f"  best-matched class              : {per_class_mean_d.index[0]} "
          f"(mean D={per_class_mean_d.iloc[0]:.3f})")
        p(f"  worst-matched class             : {per_class_mean_d.index[-1]} "
          f"(mean D={per_class_mean_d.iloc[-1]:.3f})")
    p("")
    p("-" * 74)
    p("LEVEL 2 VERDICT")
    p("-" * 74)
    if median_d < 0.1:
        v = "STRONG statistical realism (median per-class KS D < 0.10)"
    elif median_d < 0.2:
        v = "MODERATE statistical realism (median per-class KS D < 0.20)"
    elif median_d < 0.35:
        v = "LIMITED statistical realism (0.20 <= median per-class KS D < 0.35)"
    else:
        v = "WEAK statistical realism (median per-class KS D >= 0.35)"
    p(f"  {v}.")
    p("  Per-feature and per-class detail above and in the CSVs must be read")
    p("  before using any class -- overall realism can hide a weak class.")
    p("  This answers Q2 (do synthetic flow distributions resemble real unseen")
    p("  traffic?); see step5_final_report.txt for the synthesis.")
    p("=" * 74)

    S5.atomic_write_text("\n".join(L) + "\n",
                         os.path.join(S5.S5_REPORTS, "level2_statistical_report.txt"))
    print("\n".join(L))


if __name__ == "__main__":
    main()
