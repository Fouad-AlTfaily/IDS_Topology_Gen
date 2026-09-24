# ============================================================
# STEP 3.1 (addendum): DEGREE CATEGORIZATION FOR CTGAN CONDITIONING
#
# Adds categorical versions of src_out_degree and dst_in_degree to the
# Step 3.1 enriched real-flow table, WITHOUT touching the numeric columns
# or any other column / row.
#
# 1. empirical analysis of both degree distributions
# 2. two candidate schemes:
#      A: 2 bins  low/high        split = global Q50
#      B: 3 bins  low/medium/high splits = global Q33.33 / Q66.67
#    thresholds are GLOBAL (whole enriched dataset), applied identically
#    to every attack class.
# 3. condition-support analysis for
#      Attack + src_degree_cat + dst_degree_cat + src_role + dst_role
# 4. recommendation (2 vs 3) from measured statistics
# 5. writes the chosen categorical columns back into the enriched file,
#    plus degree_binning_analysis.csv and degree_binning_report.txt
#
# Input  : results/step3/step3_1_enriched_real_flows.csv   (unchanged in place
#          except for the two added *_cat columns)
# ============================================================

import os
import numpy as np
import pandas as pd

IN_CSV      = "results/step3/step3_1_enriched_real_flows.csv"
OUT_CSV     = "results/step3/step3_1_enriched_real_flows.csv"
ANALYSIS_CSV = "results/step3/degree_binning_analysis.csv"
REPORT_TXT   = "results/step3/degree_binning_report.txt"

DEG_COLS = ["src_out_degree", "dst_in_degree"]
CAT_NAME = {"src_out_degree": "src_out_degree_cat",
            "dst_in_degree":  "dst_in_degree_cat"}

# quantile fractions (Q33.33 == 1/3, Q66.67 == 2/3 terciles)
F_LO = 1.0 / 3.0
F_HI = 2.0 / 3.0

REPORT_LINES = []


def emit(s=""):
    print(s)
    REPORT_LINES.append(s)


def q(series, frac):
    """deterministic linear-interpolation quantile"""
    return float(np.quantile(series.to_numpy(dtype=float), frac))


def describe_degree(name, s):
    emit(f"--- {name} ---")
    n = int(s.shape[0])
    nz = int((s == 0).sum())
    emit(f"  count                 : {n:,}")
    emit(f"  unique values         : {s.nunique():,}")
    emit(f"  min                   : {int(s.min())}")
    emit(f"  max                   : {int(s.max())}")
    emit(f"  mean                  : {s.mean():.6f}")
    emit(f"  median (Q50)          : {s.median():.6f}")
    emit(f"  Q25                   : {q(s, 0.25):.6f}")
    emit(f"  Q33.33 (1/3)          : {q(s, F_LO):.6f}")
    emit(f"  Q50                   : {q(s, 0.50):.6f}")
    emit(f"  Q66.67 (2/3)          : {q(s, F_HI):.6f}")
    emit(f"  Q75                   : {q(s, 0.75):.6f}")
    emit(f"  Q90                   : {q(s, 0.90):.6f}")
    emit(f"  Q95                   : {q(s, 0.95):.6f}")
    emit(f"  Q99                   : {q(s, 0.99):.6f}")
    emit(f"  zeros                 : {nz:,}  ({100.0 * nz / n:.4f}%)")
    emit("  most common degree values (value: count, pct):")
    vc = s.value_counts().sort_values(ascending=False).head(20)
    for val, cnt in vc.items():
        emit(f"    {int(val):>6} : {cnt:>10,}  ({100.0 * cnt / n:.4f}%)")
    emit()


def cats_2bin(s, t):
    return np.where(s.to_numpy(dtype=float) <= t, "low", "high")


def cats_3bin(s, t_lo, t_hi):
    x = s.to_numpy(dtype=float)
    out = np.full(x.shape, "medium", dtype=object)
    out[x <= t_lo] = "low"
    out[x > t_hi] = "high"
    return out


def support_stats(df, tag, src_cat_col, dst_cat_col):
    """condition support for Attack + src_deg_cat + dst_deg_cat + src_role + dst_role"""
    emit(f"================  CONDITION SUPPORT  --  {tag}  ================")
    keys = ["Attack", src_cat_col, dst_cat_col, "src_role", "dst_role"]

    attacks = sorted(df["Attack"].unique())
    src_deg_levels = sorted(df[src_cat_col].unique())
    dst_deg_levels = sorted(df[dst_cat_col].unique())
    roles = ["high", "low", "medium"]  # full role vocabulary present in data
    role_levels = sorted(set(df["src_role"].unique()) | set(df["dst_role"].unique()))

    total_possible = (len(attacks) * len(src_deg_levels) * len(dst_deg_levels)
                      * len(role_levels) * len(role_levels))

    grp = df.groupby(keys, observed=True).size().reset_index(name="n")
    observed = len(grp)
    zero = total_possible - observed
    supp = grp["n"].to_numpy()

    b_1_4   = int(((supp >= 1) & (supp <= 4)).sum())
    b_5_9   = int(((supp >= 5) & (supp <= 9)).sum())
    b_10_19 = int(((supp >= 10) & (supp <= 19)).sum())
    b_20p   = int((supp >= 20).sum())

    emit(f"  degree categories        : {src_deg_levels}  (src) / {dst_deg_levels}  (dst)")
    emit(f"  role vocabulary          : {role_levels}")
    emit(f"  total possible combos    : {total_possible:,}   "
         f"({len(attacks)} attack x {len(src_deg_levels)} src_deg x {len(dst_deg_levels)} dst_deg "
         f"x {len(role_levels)} src_role x {len(role_levels)} dst_role)")
    emit(f"  combos observed          : {observed:,}")
    emit(f"  combos with zero rows    : {zero:,}")
    emit(f"  combos with 1-4 rows     : {b_1_4:,}")
    emit(f"  combos with 5-9 rows     : {b_5_9:,}")
    emit(f"  combos with 10-19 rows   : {b_10_19:,}")
    emit(f"  combos with 20+ rows     : {b_20p:,}")
    emit(f"  minimum support          : {int(supp.min())}")
    emit(f"  median support           : {float(np.median(supp)):.1f}")
    emit(f"  mean support             : {supp.mean():.2f}")
    emit(f"  maximum support          : {int(supp.max())}")
    rare = b_1_4 + b_5_9  # < 10 rows == rare
    emit(f"  rare combos (<10 rows)   : {rare:,}  "
         f"({100.0 * rare / observed:.2f}% of observed, "
         f"{100.0 * rare / total_possible:.2f}% of possible)")
    emit()

    emit("  support distribution by attack:")
    emit(f"    {'attack':<12}{'observed':>10}{'min':>6}{'median':>8}{'mean':>9}"
         f"{'max':>8}{'rare<10':>9}{'zero<5':>8}")
    per_attack_rare = {}
    for atk in attacks:
        gg = grp[grp["Attack"] == atk]["n"].to_numpy()
        atk_possible = (len(src_deg_levels) * len(dst_deg_levels)
                        * len(role_levels) * len(role_levels))
        r = int((gg < 10).sum())
        z = atk_possible - len(gg)
        per_attack_rare[atk] = {"observed": int(len(gg)),
                                "rare_lt10": r,
                                "zero": int(z),
                                "min": int(gg.min()),
                                "median": float(np.median(gg)),
                                "mean": float(gg.mean()),
                                "max": int(gg.max())}
        emit(f"    {atk:<12}{len(gg):>10,}{gg.min():>6}{np.median(gg):>8.1f}"
             f"{gg.mean():>9.1f}{gg.max():>8,}{r:>9}{z:>8}")
    emit()

    emit("  rare combinations (<10 rows) per attack:")
    for atk in attacks:
        emit(f"    {atk:<12}: {per_attack_rare[atk]['rare_lt10']:>4}  "
             f"(of {per_attack_rare[atk]['observed']} observed, "
             f"{per_attack_rare[atk]['zero']} unobserved)")
    emit()

    return {
        "tag": tag,
        "total_possible": total_possible,
        "observed": observed,
        "zero": zero,
        "b_1_4": b_1_4, "b_5_9": b_5_9, "b_10_19": b_10_19, "b_20p": b_20p,
        "min": int(supp.min()), "median": float(np.median(supp)),
        "mean": float(supp.mean()), "max": int(supp.max()),
        "rare_lt10": rare,
        "per_attack": per_attack_rare,
    }


def class_level_cat_dist(df, tag, src_cat_col, dst_cat_col):
    emit(f"----------------  CLASS-LEVEL CATEGORY DISTRIBUTIONS  --  {tag}  ----------------")
    for col, label in [(src_cat_col, "Attack x src_degree_category"),
                       (dst_cat_col, "Attack x dst_degree_category")]:
        emit(f"  {label}  (row counts / row %):")
        ct = pd.crosstab(df["Attack"], df[col])
        pct = ct.div(ct.sum(axis=1), axis=0) * 100.0
        cols = list(ct.columns)
        emit(f"    {'attack':<12}" + "".join(f"{c:>12}" for c in cols) + "   |" +
             "".join(f"{c + '%':>10}" for c in cols))
        for atk in ct.index:
            counts = "".join(f"{int(ct.loc[atk, c]):>12,}" for c in cols)
            perc = "".join(f"{pct.loc[atk, c]:>10.2f}" for c in cols)
            emit(f"    {atk:<12}{counts}   |{perc}")
        emit()


def main():
    df = pd.read_csv(IN_CSV, low_memory=False)
    n_rows_in = len(df)
    emit("============================================================")
    emit("STEP 3.1 DEGREE BINNING ANALYSIS")
    emit("============================================================")
    emit(f"input file : {IN_CSV}")
    emit(f"rows       : {n_rows_in:,}")
    emit(f"columns    : {df.shape[1]}")
    emit(f"attacks    : {sorted(df['Attack'].unique())}")
    emit()

    # keep pristine copies for the final verification
    orig = {c: df[c].copy() for c in DEG_COLS}

    # ---------------- 1. empirical analysis ----------------
    emit("############  1. EMPIRICAL DEGREE DISTRIBUTIONS  ############")
    emit()
    for c in DEG_COLS:
        describe_degree(c, df[c].astype("int64"))

    # ---------------- 2. global thresholds ----------------
    emit("############  2. GLOBAL THRESHOLDS (whole enriched dataset)  ############")
    emit()
    thr = {}
    for c in DEG_COLS:
        s = df[c].astype("int64")
        med = q(s, 0.50)
        t_lo = q(s, F_LO)
        t_hi = q(s, F_HI)
        thr[c] = {"median": med, "q3333": t_lo, "q6667": t_hi}
        emit(f"  {c}:")
        emit(f"    Scheme A  split (Q50)            = {med:.6f}   "
             f"-> low: deg <= {med:g} ; high: deg > {med:g}")
        emit(f"    Scheme B  split_low  (Q33.33=1/3)= {t_lo:.6f}   "
             f"-> low: deg <= {t_lo:g}")
        emit(f"    Scheme B  split_high (Q66.67=2/3)= {t_hi:.6f}   "
             f"-> medium: {t_lo:g} < deg <= {t_hi:g} ; high: deg > {t_hi:g}")
        emit()

    # ---------------- build candidate categorical columns ----------------
    A_src = cats_2bin(df["src_out_degree"].astype("int64"), thr["src_out_degree"]["median"])
    A_dst = cats_2bin(df["dst_in_degree"].astype("int64"),  thr["dst_in_degree"]["median"])
    B_src = cats_3bin(df["src_out_degree"].astype("int64"),
                      thr["src_out_degree"]["q3333"], thr["src_out_degree"]["q6667"])
    B_dst = cats_3bin(df["dst_in_degree"].astype("int64"),
                      thr["dst_in_degree"]["q3333"], thr["dst_in_degree"]["q6667"])

    work = df.copy()
    work["A_src"] = A_src
    work["A_dst"] = A_dst
    work["B_src"] = B_src
    work["B_dst"] = B_dst

    # ---------------- overall category counts / pct ----------------
    emit("############  CATEGORY COUNTS / PERCENTAGES (global)  ############")
    emit()
    cat_summary_rows = []
    for scheme, sc, dc in [("A(2-bin)", "A_src", "A_dst"),
                           ("B(3-bin)", "B_src", "B_dst")]:
        for deg_name, col in [("src_out_degree", sc), ("dst_in_degree", dc)]:
            vc = work[col].value_counts()
            emit(f"  {scheme}  {deg_name} ({CAT_NAME[deg_name]}):")
            for k in sorted(vc.index):
                pct = 100.0 * vc[k] / len(work)
                emit(f"    {k:<8}: {vc[k]:>10,}  ({pct:6.2f}%)")
                cat_summary_rows.append({"scheme": scheme, "degree": deg_name,
                                         "category": k, "count": int(vc[k]),
                                         "pct": round(pct, 4)})
            emit()

    # ---------------- 3. condition support ----------------
    emit("############  3. CONDITION-SUPPORT ANALYSIS  ############")
    emit()
    supp_A = support_stats(work, "SCHEME A (2 degree categories: low/high)", "A_src", "A_dst")
    supp_B = support_stats(work, "SCHEME B (3 degree categories: low/medium/high)", "B_src", "B_dst")

    class_level_cat_dist(work, "SCHEME A (2-bin)", "A_src", "A_dst")
    class_level_cat_dist(work, "SCHEME B (3-bin)", "B_src", "B_dst")

    # ---------------- 4. recommendation ----------------
    emit("############  4. RECOMMENDATION  ############")
    emit()

    def dominant_share(rows):
        return max(r["pct"] for r in rows) / 100.0

    A_src_rows = [r for r in cat_summary_rows if r["scheme"] == "A(2-bin)" and r["degree"] == "src_out_degree"]
    A_dst_rows = [r for r in cat_summary_rows if r["scheme"] == "A(2-bin)" and r["degree"] == "dst_in_degree"]
    B_src_rows = [r for r in cat_summary_rows if r["scheme"] == "B(3-bin)" and r["degree"] == "src_out_degree"]
    B_dst_rows = [r for r in cat_summary_rows if r["scheme"] == "B(3-bin)" and r["degree"] == "dst_in_degree"]

    emit("  criterion 1 - meaningful empirical support per category:")
    emit(f"    A src categories min share = {min(r['pct'] for r in A_src_rows):.2f}%  "
         f"| A dst categories min share = {min(r['pct'] for r in A_dst_rows):.2f}%")
    emit(f"    B src categories min share = {min(r['pct'] for r in B_src_rows):.2f}%  "
         f"| B dst categories min share = {min(r['pct'] for r in B_dst_rows):.2f}%")
    emit()
    emit("  criterion 2 - sparsity of Attack+degree+role combos:")
    emit(f"    A: observed {supp_A['observed']}, rare(<10) {supp_A['rare_lt10']} "
         f"({100.0*supp_A['rare_lt10']/supp_A['observed']:.1f}% of observed), "
         f"min {supp_A['min']}, median {supp_A['median']:.0f}")
    emit(f"    B: observed {supp_B['observed']}, rare(<10) {supp_B['rare_lt10']} "
         f"({100.0*supp_B['rare_lt10']/supp_B['observed']:.1f}% of observed), "
         f"min {supp_B['min']}, median {supp_B['median']:.0f}")
    emit()
    emit("  criterion 3 - no category holding almost all rows:")
    emit(f"    A max category share: src {dominant_share(A_src_rows)*100:.2f}%  dst {dominant_share(A_dst_rows)*100:.2f}%")
    emit(f"    B max category share: src {dominant_share(B_src_rows)*100:.2f}%  dst {dominant_share(B_dst_rows)*100:.2f}%")
    emit()

    # decision rule (documented, deterministic)
    reasons = []
    A_ok_balance = dominant_share(A_src_rows) <= 0.92 and dominant_share(A_dst_rows) <= 0.92
    B_ok_support = (min(r['pct'] for r in B_src_rows) >= 5.0 and
                    min(r['pct'] for r in B_dst_rows) >= 5.0)
    B_worse_sparsity = (supp_B['rare_lt10'] > 1.5 * supp_A['rare_lt10'] or
                        supp_B['min'] == 0)

    emit("  decision logic:")
    emit(f"    A categories reasonably balanced (max share <= 92%): {A_ok_balance}")
    emit(f"    B every category has >= 5% global support           : {B_ok_support}")
    emit(f"    B materially worse sparsity than A                  : {B_worse_sparsity}")
    emit()

    if B_ok_support and not B_worse_sparsity:
        scheme = 3
        reasons.append("3-bin: every low/medium/high category carries >=5% of rows for "
                       "both degrees, and the extra bin does not blow up rare "
                       "Attack+degree+role support relative to the 2-bin scheme.")
    else:
        scheme = 2
        if not B_ok_support:
            reasons.append("3-bin leaves at least one degree category below 5% global "
                           "support, i.e. a near-empty conditioning level.")
        if B_worse_sparsity:
            reasons.append("3-bin materially increases sparsity of the "
                           "Attack+degree+role conditioning grid (more/!=0 rare cells).")
        reasons.append("2-bin (low/high) keeps both categories well populated and the "
                       "conditioning grid denser, which is the simplest scheme with "
                       "sufficient CTGAN conditioning support.")

    emit(f"  RECOMMENDED_SCHEME = {scheme}")
    for r in reasons:
        emit(f"    - {r}")
    emit()

    # ---------------- 5. apply chosen scheme ----------------
    if scheme == 2:
        df["src_out_degree_cat"] = cats_2bin(df["src_out_degree"].astype("int64"),
                                             thr["src_out_degree"]["median"])
        df["dst_in_degree_cat"] = cats_2bin(df["dst_in_degree"].astype("int64"),
                                            thr["dst_in_degree"]["median"])
        allowed = {"low", "high"}
        final_thr = {"src_out_degree": {"rule": "low: deg <= Q50 ; high: deg > Q50",
                                        "Q50": thr["src_out_degree"]["median"]},
                     "dst_in_degree": {"rule": "low: deg <= Q50 ; high: deg > Q50",
                                       "Q50": thr["dst_in_degree"]["median"]}}
    else:
        df["src_out_degree_cat"] = cats_3bin(df["src_out_degree"].astype("int64"),
                                             thr["src_out_degree"]["q3333"],
                                             thr["src_out_degree"]["q6667"])
        df["dst_in_degree_cat"] = cats_3bin(df["dst_in_degree"].astype("int64"),
                                            thr["dst_in_degree"]["q3333"],
                                            thr["dst_in_degree"]["q6667"])
        allowed = {"low", "medium", "high"}
        final_thr = {"src_out_degree": {"rule": "low: deg <= Q33.33 ; medium: Q33.33 < deg <= Q66.67 ; high: deg > Q66.67",
                                        "Q33.33": thr["src_out_degree"]["q3333"],
                                        "Q66.67": thr["src_out_degree"]["q6667"]},
                     "dst_in_degree": {"rule": "low: deg <= Q33.33 ; medium: Q33.33 < deg <= Q66.67 ; high: deg > Q66.67",
                                       "Q33.33": thr["dst_in_degree"]["q3333"],
                                       "Q66.67": thr["dst_in_degree"]["q6667"]}}

    # ---------------- verification ----------------
    emit("############  PRE-WRITE VERIFICATION  ############")
    emit()
    ok = True
    c1 = len(df) == n_rows_in
    emit(f"  row count unchanged ({n_rows_in:,})                : {c1}")
    ok &= c1
    for c in DEG_COLS:
        same = orig[c].equals(df[c])
        emit(f"  {c} numeric values unchanged        : {same}")
        ok &= same
    for cc in ["src_out_degree_cat", "dst_in_degree_cat"]:
        nn = df[cc].notna().all()
        emit(f"  {cc} has no missing values : {nn}")
        ok &= nn
        bad = set(df[cc].unique()) - allowed
        emit(f"  {cc} values subset of {sorted(allowed)} : {len(bad) == 0}  "
             f"{'(offenders: ' + str(bad) + ')' if bad else ''}")
        ok &= (len(bad) == 0)

    # threshold-match check: recompute categories straight from reported thresholds
    if scheme == 2:
        chk_src = cats_2bin(df["src_out_degree"].astype("int64"), final_thr["src_out_degree"]["Q50"])
        chk_dst = cats_2bin(df["dst_in_degree"].astype("int64"), final_thr["dst_in_degree"]["Q50"])
    else:
        chk_src = cats_3bin(df["src_out_degree"].astype("int64"),
                            final_thr["src_out_degree"]["Q33.33"], final_thr["src_out_degree"]["Q66.67"])
        chk_dst = cats_3bin(df["dst_in_degree"].astype("int64"),
                            final_thr["dst_in_degree"]["Q33.33"], final_thr["dst_in_degree"]["Q66.67"])
    m1 = bool((df["src_out_degree_cat"].to_numpy() == chk_src).all())
    m2 = bool((df["dst_in_degree_cat"].to_numpy() == chk_dst).all())
    emit(f"  final src_out_degree_cat matches reported thresholds : {m1}")
    emit(f"  final dst_in_degree_cat  matches reported thresholds : {m2}")
    ok &= m1 and m2
    emit()
    emit(f"  ALL CHECKS PASSED : {ok}")
    emit()
    if not ok:
        raise SystemExit("verification failed -- not writing output")

    # ---------------- final category distribution (chosen scheme) ----------------
    emit("############  FINAL CATEGORY DISTRIBUTION (chosen scheme)  ############")
    emit()
    final_cat_rows = []
    for deg_name, cc in [("src_out_degree", "src_out_degree_cat"),
                         ("dst_in_degree", "dst_in_degree_cat")]:
        vc = df[cc].value_counts()
        emit(f"  {cc}:")
        for k in sorted(vc.index):
            pct = 100.0 * vc[k] / len(df)
            emit(f"    {k:<8}: {vc[k]:>10,}  ({pct:6.2f}%)")
            final_cat_rows.append({"scheme": f"FINAL({scheme}-bin)", "degree": deg_name,
                                   "category": k, "count": int(vc[k]), "pct": round(pct, 4)})
        emit()

    # ---------------- write outputs ----------------
    # analysis csv: category summaries for A, B and FINAL + support summary
    an = pd.DataFrame(cat_summary_rows + final_cat_rows)
    supp_rows = []
    for sd in (supp_A, supp_B):
        supp_rows.append({"scheme": sd["tag"].split(" ")[0] + "_" + sd["tag"].split("(")[0].strip()[-1],
                          "metric": "support", "total_possible": sd["total_possible"],
                          "observed": sd["observed"], "zero": sd["zero"],
                          "b_1_4": sd["b_1_4"], "b_5_9": sd["b_5_9"],
                          "b_10_19": sd["b_10_19"], "b_20p": sd["b_20p"],
                          "min": sd["min"], "median": sd["median"],
                          "mean": round(sd["mean"], 3), "max": sd["max"],
                          "rare_lt10": sd["rare_lt10"]})
    supp_df = pd.DataFrame(supp_rows)

    with open(ANALYSIS_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write("# category_summary\n")
        an.to_csv(fh, index=False)
        fh.write("\n# condition_support_summary\n")
        supp_df.to_csv(fh, index=False)
        fh.write("\n# per_attack_rare_combos\n")
        pa_rows = []
        for tag, sd in [("A_2bin", supp_A), ("B_3bin", supp_B)]:
            for atk, d in sd["per_attack"].items():
                pa_rows.append({"scheme": tag, "attack": atk, **d})
        pd.DataFrame(pa_rows).to_csv(fh, index=False)
    emit(f"wrote {ANALYSIS_CSV}")

    # report txt
    header = []
    header.append("STEP 3.1 DEGREE CATEGORIZATION -- SELECTION REPORT")
    header.append("=" * 60)
    header.append(f"input / output enriched file : {OUT_CSV}")
    header.append(f"rows (unchanged)             : {n_rows_in:,}")
    header.append(f"thresholds global            : YES (computed once over the "
                  f"complete enriched dataset, applied identically to every attack class)")
    header.append(f"quantile method              : numpy linear interpolation, deterministic")
    header.append("")
    header.append(f"RECOMMENDED_SCHEME = {scheme}")
    header.append("")
    header.append("exact thresholds used in the final file:")
    for c in DEG_COLS:
        header.append(f"  {c} -> {CAT_NAME[c]}")
        header.append(f"    {final_thr[c]['rule']}")
        for kk, vv in final_thr[c].items():
            if kk != "rule":
                header.append(f"      {kk} = {vv:.6f}")
    header.append("")
    header.append("rationale:")
    for r in reasons:
        header.append(f"  - {r}")
    header.append("")
    header.append("=" * 60)
    header.append("FULL ANALYSIS LOG")
    header.append("=" * 60)

    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(header) + "\n\n")
        fh.write("\n".join(REPORT_LINES) + "\n")
    print(f"wrote {REPORT_TXT}")

    # finally write the enriched csv (numeric cols untouched, 2 new *_cat cols appended)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(df):,} rows, {df.shape[1]} cols)")
    print("added columns:", ["src_out_degree_cat", "dst_in_degree_cat"])


if __name__ == "__main__":
    main()
