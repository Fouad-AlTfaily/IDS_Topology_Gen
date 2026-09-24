# ============================================================
# STEP 5 -- LEVEL 3: UTILITY VALIDATION  (the most important level)
#
#   python -m src.validation.step5_3_utility
#
# Can synthetic data train an intrusion-detection model that generalises to
# REAL UNSEEN NF-ToN-IoT traffic?
#
# The central experiment (section 45):
#
#     TRAINING DATA                     TEST DATA
#   1 real TRAIN                        real TEST   (held out)
#   2 final synthetic dataset           real TEST   (held out)
#   3 real TRAIN + final synthetic      real TEST   (held out)
#
# Same architecture, same protocol, same real TEST set for all three
# (step5_gnn.EdgeSAGE -- E-GraphSAGE-style edge classifier). The ONLY
# difference is the training data.
#
# LEAKAGE CONTROLS (hard -- section 34):
#   * real TEST split is reconstructed deterministically and used ONLY for the
#     final evaluation -- never for training, preprocessing, or model selection.
#   * preprocessing (log1p + standardisation) is fit on each condition's
#     TRAINING edges only; real TEST is transform-only.
#   * class weights come from each condition's TRAINING labels only.
#   * explicit row-overlap check: no real TEST flow row appears in any training
#     set (results/step5_validation/logs/level3_leakage_checks.txt).
#   * fixed epochs, no early stopping -> no tuning on any held-out real data.
#
# Restart-safe: each condition has results/step5_validation/utility/<cond>/DONE.
# A finished condition is skipped; a failed one resumes on re-run. The
# aggregate/figures/report step runs only once all three conditions are done.
#
# Outputs (results/step5_validation/):
#   utility/real_only/            (per-seed metrics, confusion, checkpoint, DONE)
#   utility/synthetic_only/       (...)
#   utility/real_plus_synthetic/  (...)
#   utility/utility_metrics.csv
#   utility/comparison_headline.csv
#   utility/per_class_f1.csv  per_class_recall.csv  per_class_precision.csv
#   figures/utility/real_only_confusion_matrix.png
#   figures/utility/synthetic_only_confusion_matrix.png
#   figures/utility/real_plus_synthetic_confusion_matrix.png
#   figures/utility/utility_macro_f1_comparison.png
#   figures/utility/utility_per_class_f1.png
#   figures/utility/utility_per_class_recall.png
#   figures/utility/utility_per_class_precision.png
#   reports/level3_utility_report.txt
#   utility/LEVEL3_DONE
# ============================================================

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.validation import step5_common as S5

COND = S5.UTILITY_CONDITIONS               # ["real_only","synthetic_only","real_plus_synthetic"]
COND_TITLE = {"real_only": "Real-only", "synthetic_only": "Synthetic-only",
              "real_plus_synthetic": "Real+Synthetic"}
HEADLINE = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
            "weighted_precision", "weighted_recall", "weighted_f1"]

FLOW_COLS = S5.IP_COLS + S5.FLOW_FEATURES + ["Attack"]


def _cond_dir(cond: str) -> str:
    return os.path.join(S5.S5_UTILITY, cond)


def _cond_required(cond: str) -> list[str]:
    d = _cond_dir(cond)
    return [os.path.join(d, "metrics.json"),
            os.path.join(d, "confusion_matrix_raw.csv"),
            os.path.join(d, "confusion_matrix_normalized.csv"),
            os.path.join(d, "per_class_metrics.csv")]


LEVEL3_REQUIRED = [
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
    os.path.join(S5.S5_FIG_UTIL, "utility_per_class_precision.png"),
    os.path.join(S5.S5_REPORTS, "level3_utility_report.txt"),
]


# ------------------------------------------------------------------
def _metrics(y_true, y_pred) -> dict:
    labels = list(range(len(S5.CLASSES)))
    acc = float(accuracy_score(y_true, y_pred))
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "accuracy": acc,
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_precision": float(wp), "weighted_recall": float(wr),
        "weighted_f1": float(wf),
        "per_class": {S5.CLASSES[i]: {"precision": float(pc_p[i]),
                                     "recall": float(pc_r[i]),
                                     "f1": float(pc_f[i]),
                                     "support": int(pc_s[i])}
                      for i in labels},
    }


def _confusion(y_true, y_pred):
    labels = list(range(len(S5.CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
    cmn = np.nan_to_num(cmn)
    return cm, cmn


def _run_condition(cond: str, pool: pd.DataFrame, real_test: pd.DataFrame,
                   env: dict) -> None:
    from src.validation import step5_gnn as G
    d = _cond_dir(cond)
    os.makedirs(d, exist_ok=True)
    if S5.stage_done(os.path.join(d, "DONE"), _cond_required(cond)):
        print(f"[LEVEL 3] condition '{cond}' already complete -- SKIPPING")
        return

    cfg = S5.GNN_CONFIG
    device = G.pick_device()
    print(f"\n[LEVEL 3] === condition: {cond} ===  device={device.type}")
    print(f"           training pool rows (flows/edges): {len(pool):,}")

    seed_metrics, seed_cms, seed_cmns, seed_infos = [], [], [], []
    for seed in cfg["seeds"]:
        G.set_determinism(seed)
        pool_ss, ss_info = G.stratified_subsample(
            pool, cfg["max_train_edges"], cfg["min_edges_per_class"], seed)
        prep = G.fit_preprocessor(pool_ss)                    # TRAIN ONLY
        train_t = G.build_graph_tensors(pool_ss, prep)
        test_t = G.build_graph_tensors(real_test, prep)       # transform-only

        print(f"  seed {seed}: train edges {train_t.n_edges:,} "
              f"({train_t.num_nodes:,} nodes)  test edges {test_t.n_edges:,} "
              f"({test_t.num_nodes:,} nodes)")
        model, tr_info = G.train_model(train_t, cfg, seed, device)
        y_true, y_pred = G.evaluate(model, test_t, device)

        m = _metrics(y_true, y_pred)
        cm, cmn = _confusion(y_true, y_pred)
        seed_metrics.append(m); seed_cms.append(cm); seed_cmns.append(cmn)
        info = {
            "seed": seed,
            "train_edges": train_t.n_edges, "train_nodes": train_t.num_nodes,
            "test_edges": test_t.n_edges, "test_nodes": test_t.num_nodes,
            "train_class_counts": train_t.class_counts,
            "subsample": ss_info,
            "preprocessing": {k: prep[k] for k in ("mean", "std", "fit_rows", "fit_note")},
            "train_seconds": tr_info["train_seconds"],
            "history_tail": tr_info["history"][-3:],
            "class_weights": tr_info["class_weights"],
        }
        seed_infos.append(info)
        S5.write_json({"seed": seed, "metrics": m, "info": info},
                      os.path.join(d, f"seed_{seed}_metrics.json"))
        G.save_checkpoint(model, os.path.join(d, f"seed_{seed}_model.pt"),
                          {"cond": cond, "seed": seed, "config": cfg})

    # ---- aggregate over seeds ----
    def agg(key):
        vals = [sm[key] for sm in seed_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    headline = {k: {"mean": agg(k)[0], "std": agg(k)[1]} for k in HEADLINE}
    per_class = {}
    for c in S5.CLASSES:
        for met in ("precision", "recall", "f1"):
            vals = [sm["per_class"][c][met] for sm in seed_metrics]
            per_class.setdefault(c, {})[met] = {"mean": float(np.mean(vals)),
                                                "std": float(np.std(vals))}
        per_class[c]["support"] = int(seed_metrics[0]["per_class"][c]["support"])

    cm_mean = np.mean(seed_cms, axis=0)
    cmn_mean = np.mean(seed_cmns, axis=0)
    # confusion CSVs carry a class-name index on purpose -> written directly, not
    # via atomic_write_csv (which forces index=False); still guarded to stay under S5.
    cm_raw_path = os.path.join(d, "confusion_matrix_raw.csv")
    cm_norm_path = os.path.join(d, "confusion_matrix_normalized.csv")
    S5._assert_under_s5(cm_raw_path)
    S5._assert_under_s5(cm_norm_path)
    pd.DataFrame(cm_mean.round().astype(int), index=S5.CLASSES,
                 columns=S5.CLASSES).to_csv(cm_raw_path)
    pd.DataFrame(cmn_mean.round(5), index=S5.CLASSES,
                 columns=S5.CLASSES).to_csv(cm_norm_path)

    pc_rows = []
    for c in S5.CLASSES:
        pc_rows.append({"class": c,
                        "precision_mean": per_class[c]["precision"]["mean"],
                        "precision_std": per_class[c]["precision"]["std"],
                        "recall_mean": per_class[c]["recall"]["mean"],
                        "recall_std": per_class[c]["recall"]["std"],
                        "f1_mean": per_class[c]["f1"]["mean"],
                        "f1_std": per_class[c]["f1"]["std"],
                        "support": per_class[c]["support"]})
    S5.atomic_write_csv(pd.DataFrame(pc_rows),
                        os.path.join(d, "per_class_metrics.csv"))

    metrics_doc = {
        "condition": cond,
        "title": COND_TITLE[cond],
        "seeds": cfg["seeds"],
        "n_seeds": len(cfg["seeds"]),
        "single_seed_limitation": (len(cfg["seeds"]) == 1),
        "headline_metrics": headline,
        "per_class_metrics": per_class,
        "architecture": cfg["architecture"],
        "hidden_dim": cfg["hidden_dim"], "num_layers": cfg["num_layers"],
        "dropout": cfg["dropout"], "optimizer": cfg["optimizer"],
        "learning_rate": cfg["learning_rate"], "weight_decay": cfg["weight_decay"],
        "epochs": cfg["epochs"], "batch_size": "full-batch (all training edges)",
        "class_weighted_loss": cfg["class_weighted_loss"],
        "max_train_edges": cfg["max_train_edges"],
        "training_pool_rows": int(len(pool)),
        "per_seed_info": seed_infos,
        "real_test_edges": int(len(real_test)),
        "device": device.type,
        "environment": env,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    S5.write_json(metrics_doc, os.path.join(d, "metrics.json"))
    S5.write_marker(os.path.join(d, "DONE"),
                    {"condition": cond,
                     "macro_f1_mean": headline["macro_f1"]["mean"],
                     "accuracy_mean": headline["accuracy"]["mean"]})
    print(f"[LEVEL 3] '{cond}' done: macro-F1 {headline['macro_f1']['mean']:.4f} "
          f"acc {headline['accuracy']['mean']:.4f}")


# ------------------------------------------------------------------
def _fig_confusion(cond: str):
    d = _cond_dir(cond)
    cm = pd.read_csv(os.path.join(d, "confusion_matrix_raw.csv"), index_col=0)
    cmn = pd.read_csv(os.path.join(d, "confusion_matrix_normalized.csv"), index_col=0)
    cm = cm.reindex(index=S5.CLASSES, columns=S5.CLASSES)
    cmn = cmn.reindex(index=S5.CLASSES, columns=S5.CLASSES)
    fig, ax = plt.subplots(1, 2, figsize=(19, 8))
    for a, mat, title, fmt in ((ax[0], cm.to_numpy(), "raw counts", "d"),
                               (ax[1], cmn.to_numpy(), "row-normalized", ".2f")):
        im = a.imshow(mat, cmap="Blues")
        a.set_xticks(range(len(S5.CLASSES))); a.set_xticklabels(S5.CLASSES, rotation=45, ha="right")
        a.set_yticks(range(len(S5.CLASSES))); a.set_yticklabels(S5.CLASSES)
        a.set_xlabel("predicted"); a.set_ylabel("true (real TEST)")
        a.set_title(f"{COND_TITLE[cond]} -- {title}")
        thr = mat.max() / 2 if mat.size else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                a.text(j, i, format(int(v) if fmt == "d" else v, fmt),
                       ha="center", va="center", fontsize=7,
                       color="white" if v > thr else "black")
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"Level 3 confusion matrix -- {COND_TITLE[cond]} model on real TEST",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_UTIL, f"{cond}_confusion_matrix.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_headline(metrics_by_cond: dict):
    keys = ["macro_f1", "macro_recall", "macro_precision", "weighted_f1", "accuracy"]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cond in enumerate(COND):
        means = [metrics_by_cond[cond]["headline_metrics"][k]["mean"] for k in keys]
        stds = [metrics_by_cond[cond]["headline_metrics"][k]["std"] for k in keys]
        ax.bar(x + (i - 1) * 0.27, means, 0.27, yerr=stds, capsize=3,
               label=COND_TITLE[cond])
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=20, ha="right")
    ax.set_ylabel("score on real TEST"); ax.set_ylim(0, 1)
    ax.set_title("Level 3 -- IDS utility on real TEST (train data is the only variable)")
    ax.legend()
    for xx in x:
        ax.axvline(xx + 0.4, color="0.9", lw=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_UTIL, "utility_macro_f1_comparison.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def _fig_per_class(metric: str, metrics_by_cond: dict):
    x = np.arange(len(S5.CLASSES))
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, cond in enumerate(COND):
        pc = metrics_by_cond[cond]["per_class_metrics"]
        means = [pc[c][metric]["mean"] for c in S5.CLASSES]
        stds = [pc[c][metric]["std"] for c in S5.CLASSES]
        ax.bar(x + (i - 1) * 0.27, means, 0.27, yerr=stds, capsize=2,
               label=COND_TITLE[cond])
    ax.set_xticks(x); ax.set_xticklabels(S5.CLASSES, rotation=30, ha="right")
    ax.set_ylabel(metric); ax.set_ylim(0, 1)
    ax.set_title(f"Level 3 -- per-class {metric} on real TEST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(S5.S5_FIG_UTIL, f"utility_per_class_{metric}.png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
def _aggregate_and_report(env: dict):
    m = {c: S5.read_json(os.path.join(_cond_dir(c), "metrics.json")) for c in COND}

    # utility_metrics.csv -- one row per (condition, metric)
    rows = []
    for c in COND:
        for k in HEADLINE:
            hv = m[c]["headline_metrics"][k]
            rows.append({"condition": c, "metric": k,
                         "mean": round(hv["mean"], 5), "std": round(hv["std"], 5),
                         "n_seeds": m[c]["n_seeds"]})
    S5.atomic_write_csv(pd.DataFrame(rows),
                        os.path.join(S5.S5_UTILITY, "utility_metrics.csv"))

    # comparison_headline.csv
    comp_rows = []
    base = m["real_only"]["headline_metrics"]
    for other in ("synthetic_only", "real_plus_synthetic"):
        ov = m[other]["headline_metrics"]
        for k in ["macro_f1", "macro_recall", "macro_precision", "weighted_f1", "accuracy"]:
            b, o = base[k]["mean"], ov[k]["mean"]
            comp_rows.append({
                "comparison": f"{other} vs real_only",
                "metric": k,
                "real_only": round(b, 5),
                other: round(o, 5),
                "absolute_diff": round(o - b, 5),
                "relative_change_pct": (round(100 * (o - b) / b, 2) if abs(b) > 1e-9 else np.nan),
            })
    S5.atomic_write_csv(pd.DataFrame(comp_rows),
                        os.path.join(S5.S5_UTILITY, "comparison_headline.csv"))

    # per-class tables
    for metric, fname in (("f1", "per_class_f1.csv"), ("recall", "per_class_recall.csv"),
                          ("precision", "per_class_precision.csv")):
        pc_rows = []
        for c in S5.CLASSES:
            row = {"class": c}
            for cond in COND:
                row[cond] = round(m[cond]["per_class_metrics"][c][metric]["mean"], 5)
            row["support_real_test"] = m["real_only"]["per_class_metrics"][c]["support"]
            pc_rows.append(row)
        S5.atomic_write_csv(pd.DataFrame(pc_rows), os.path.join(S5.S5_UTILITY, fname))

    # figures
    for c in COND:
        _fig_confusion(c)
    _fig_headline(m)
    for metric in ("f1", "recall", "precision"):
        _fig_per_class(metric, m)

    _write_report(m)
    S5.write_marker(S5.LEVEL3_DONE, {
        "conditions": COND,
        "seeds": S5.GNN_CONFIG["seeds"],
        "primary_comparison": "synthetic_only vs real_only on real TEST",
        "secondary_comparison": "real_plus_synthetic vs real_only on real TEST",
    })
    print("[LEVEL 3] aggregate + figures + report done")


def _verdict(m):
    ro = m["real_only"]["headline_metrics"]["macro_f1"]["mean"]
    so = m["synthetic_only"]["headline_metrics"]["macro_f1"]["mean"]
    rs = m["real_plus_synthetic"]["headline_metrics"]["macro_f1"]["mean"]
    ratio = so / ro if ro > 1e-9 else float("nan")
    if ratio >= 0.9:
        q3 = "YES -- synthetic-only training nearly matches real-only (macro-F1 ratio >= 0.90)"
    elif ratio >= 0.75:
        q3 = "PARTIALLY -- synthetic-only training reaches 75-90% of real-only macro-F1"
    elif ratio >= 0.5:
        q3 = "WEAKLY -- synthetic-only training reaches 50-75% of real-only macro-F1"
    else:
        q3 = "NO -- synthetic-only training is far below real-only (macro-F1 ratio < 0.50)"
    drs = rs - ro
    if drs > 0.01:
        q4 = f"IMPROVES real-only by +{drs:.3f} macro-F1"
    elif drs < -0.01:
        q4 = f"DEGRADES real-only by {drs:.3f} macro-F1"
    else:
        q4 = f"is ~neutral vs real-only ({drs:+.3f} macro-F1)"
    return ratio, q3, q4


def _write_report(m):
    L = []
    def p(s=""): L.append(s)
    ratio, q3, q4 = _verdict(m)

    p("=" * 74)
    p("STEP 5 -- LEVEL 3: UTILITY VALIDATION REPORT  (most important level)")
    p("=" * 74)
    p("")
    p("QUESTION: can synthetic data train a GNN-IDS that generalises to REAL")
    p("UNSEEN NF-ToN-IoT traffic (the held-out real TEST split)?")
    p("")
    p("MODEL (identical for all three conditions):")
    p(f"  {S5.GNN_CONFIG['architecture']}")
    p(f"  hidden={S5.GNN_CONFIG['hidden_dim']}  layers={S5.GNN_CONFIG['num_layers']}  "
      f"dropout={S5.GNN_CONFIG['dropout']}  optimizer={S5.GNN_CONFIG['optimizer']}  "
      f"lr={S5.GNN_CONFIG['learning_rate']}  wd={S5.GNN_CONFIG['weight_decay']}  "
      f"epochs={S5.GNN_CONFIG['epochs']}  full-batch")
    p(f"  class-weighted CE (inverse frequency of that training set only)")
    p(f"  stratified training cap: {S5.GNN_CONFIG['max_train_edges']:,} edges "
      f"(min {S5.GNN_CONFIG['min_edges_per_class']}/class), seed(s) {S5.GNN_CONFIG['seeds']}")
    p("")
    p("  Preprocessing (log1p + standardise) is fit on each condition's TRAINING")
    p("  edges only; real TEST is transform-only. Real TEST is never used for")
    p("  training, preprocessing, class weights, or model selection.")
    p("")
    p("-" * 74)
    p("HEADLINE METRICS on real TEST  (mean" +
      (" +/- std over seeds" if m["real_only"]["n_seeds"] > 1 else "; SINGLE SEED") + ")")
    p("-" * 74)
    hdr = f"{'metric':<20}" + "".join(f"{COND_TITLE[c]:>20}" for c in COND)
    p(hdr)
    for k in HEADLINE:
        line = f"{k:<20}"
        for c in COND:
            hv = m[c]["headline_metrics"][k]
            line += f"{hv['mean']:>13.4f}+/-{hv['std']:.3f}"
        p(line)
    p("")
    p("-" * 74)
    p("PRIMARY COMPARISON  (synthetic-only vs real-only, on real TEST)")
    p("-" * 74)
    base = m["real_only"]["headline_metrics"]
    for other in ("synthetic_only", "real_plus_synthetic"):
        ov = m[other]["headline_metrics"]
        p(f"  {other} vs real_only:")
        for k in ["macro_f1", "macro_recall", "macro_precision", "weighted_f1", "accuracy"]:
            b, o = base[k]["mean"], ov[k]["mean"]
            rel = (f"{100*(o-b)/b:+.1f}%" if abs(b) > 1e-9 else "n/a")
            p(f"    {k:<20} {b:.4f} -> {o:.4f}   (abs {o-b:+.4f},  rel {rel})")
        p("")
    p("-" * 74)
    p("PER-CLASS F1 on real TEST  (mean over seeds)")
    p("-" * 74)
    p(f"{'class':<12}{'Real-only':>12}{'Synth-only':>12}{'Real+Synth':>12}{'support':>10}")
    for c in S5.CLASSES:
        r = m["real_only"]["per_class_metrics"][c]["f1"]["mean"]
        s = m["synthetic_only"]["per_class_metrics"][c]["f1"]["mean"]
        rs = m["real_plus_synthetic"]["per_class_metrics"][c]["f1"]["mean"]
        sup = m["real_only"]["per_class_metrics"][c]["support"]
        p(f"{c:<12}{r:>12.3f}{s:>12.3f}{rs:>12.3f}{sup:>10}")
    p("")
    p("  (per-class recall / precision tables: utility/per_class_recall.csv,")
    p("   utility/per_class_precision.csv; figures in figures/utility/)")
    p("")
    p("-" * 74)
    p("LEVEL 3 VERDICT")
    p("-" * 74)
    p(f"  macro-F1  real-only        : {m['real_only']['headline_metrics']['macro_f1']['mean']:.4f}")
    p(f"  macro-F1  synthetic-only   : {m['synthetic_only']['headline_metrics']['macro_f1']['mean']:.4f}")
    p(f"  macro-F1  real+synthetic   : {m['real_plus_synthetic']['headline_metrics']['macro_f1']['mean']:.4f}")
    p(f"  synthetic/real macro-F1 ratio : {ratio:.3f}")
    p("")
    p(f"  Q3 (IDS trained on synthetic generalises to real unseen traffic?): {q3}")
    p(f"  Q4 (real+synthetic vs real-only): {q4}")
    if m["real_only"]["n_seeds"] == 1:
        p("")
        p("  LIMITATION: a single random seed was used (compute budget). These are")
        p("  point estimates, not multi-seed means. Set STEP5_GNN_SEEDS=\"42,1,2\"")
        p("  to repeat with more seeds and report mean +/- std.")
    p("=" * 74)

    S5.atomic_write_text("\n".join(L) + "\n",
                         os.path.join(S5.S5_REPORTS, "level3_utility_report.txt"))
    print("\n".join(L))


# ------------------------------------------------------------------
def _load_pools_and_test():
    S5.verify_inputs(require_synth=True)
    split = S5.load_real_split()
    real_train = split["train"][FLOW_COLS + ["_row_id"]].copy()   # keep row id for leakage check
    real_test = split["test"][FLOW_COLS + ["_row_id"]].copy()
    synth = S5.load_final_synth()[FLOW_COLS].copy()
    for fr in (real_train, real_test, synth):
        fr["Attack"] = fr["Attack"].astype(str).str.strip().str.lower()
    pools = {
        "real_only": real_train,
        "synthetic_only": synth,
        "real_plus_synthetic": pd.concat([real_train, synth], ignore_index=True),
    }
    return pools, real_test, split


def class_distribution_preview() -> pd.DataFrame:
    """Section 27 -- inspect class distributions in real TRAIN / synthetic /
    real+synthetic (NEVER uses the real TEST distribution)."""
    pools, _, _ = _load_pools_and_test()
    rows = []
    for cond, df in pools.items():
        vc = df["Attack"].value_counts()
        for c in S5.CLASSES:
            rows.append({"condition": cond, "class": c, "rows": int(vc.get(c, 0)),
                         "fraction": round(float(vc.get(c, 0)) / max(len(df), 1), 5)})
    return pd.DataFrame(rows)


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            f"ABORT: PyTorch is required for Level 3 and is not importable ({exc}). "
            f"In Colab torch is preinstalled; run the Step-5 setup cell first."
        )


def run_condition(cond: str) -> None:
    """Train + evaluate ONE utility condition (restart-safe; skips if done)."""
    assert cond in COND, f"unknown condition {cond!r}"
    S5.ensure_dirs()
    env = S5.write_environment_report()
    _require_torch()
    with S5.stage_banner(f"LEVEL 3 -- condition {cond}"):
        print(f"  SCALE={S5.SCALE}   synthetic dataset: {S5.FINAL_SYNTH}")
        pools, real_test, split = _load_pools_and_test()
        S5.assert_no_leakage({cond: pools[cond]}, real_test, split,
                             log_path=os.path.join(S5.S5_LOGS,
                                                   f"level3_leakage_{cond}.txt"))
        _run_condition(cond, pools[cond], real_test, env)


def aggregate() -> None:
    """Build the cross-condition tables, figures and Level-3 report + marker.
    Runs only when all three conditions are complete."""
    S5.ensure_dirs()
    env = S5.write_environment_report()
    if S5.stage_done(S5.LEVEL3_DONE, LEVEL3_REQUIRED):
        print("[LEVEL 3] aggregate already complete -- SKIPPING")
        return
    if not all(S5.stage_done(os.path.join(_cond_dir(c), "DONE"), _cond_required(c))
               for c in COND):
        missing = [c for c in COND
                   if not S5.stage_done(os.path.join(_cond_dir(c), "DONE"), _cond_required(c))]
        raise SystemExit(f"ABORT: cannot aggregate -- conditions not complete: {missing}")
    with S5.stage_banner("LEVEL 3 -- aggregate + figures + report"):
        _aggregate_and_report(env)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Step 5 Level 3 -- GNN-IDS utility validation")
    ap.add_argument("--condition", choices=COND + ["all"], default="all",
                    help="which training condition to run (default: all three)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training; only build cross-condition tables/figures/report")
    args = ap.parse_args()

    S5.ensure_dirs()
    S5.write_environment_report()

    if S5.stage_done(S5.LEVEL3_DONE, LEVEL3_REQUIRED):
        print("[LEVEL 3] utility validation already complete -- SKIPPING")
        return

    if args.aggregate_only:
        aggregate()
        return

    _require_torch()
    conds = COND if args.condition == "all" else [args.condition]
    for cond in conds:
        run_condition(cond)

    if args.condition == "all" and all(
            S5.stage_done(os.path.join(_cond_dir(c), "DONE"), _cond_required(c))
            for c in COND):
        aggregate()
    elif args.condition == "all":
        raise SystemExit("ABORT: not all conditions completed -- re-run to resume")
    print("[LEVEL 3] done ->", S5.S5_UTILITY)


if __name__ == "__main__":
    main()
