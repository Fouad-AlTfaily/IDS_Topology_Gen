# ============================================================
# STEP 3.7 -- ASSEMBLE + VALIDATE GENERATED ATTACK FLOWS
#
# Concatenates every completed generation batch into one table, concatenates
# the per-batch generation-log fragments into logs/generation_log.csv, runs
# the hard integrity checks, and computes the support-aware metrics.
#
# Outputs (SCALE="smaller" -> results/small_scale/step3_ctgan/):
#   generation/smaller/synthetic_attack_flows.csv   the Step-3 deliverable
#   logs/generation_log.csv
#   reports/generation_summary.csv
#   reports/backoff_summary.csv
#   reports/generation_log_summary.csv
#   generation/smaller/VALIDATION_DONE
#
# HARD checks (any failure -> SystemExit, no VALIDATION_DONE):
#   1  no benign rows            2  no ransomware rows
#   3  every src IP is a graph node
#   4  every dst IP is a graph node
#   5  every (src,dst) is a synthetic edge
#   6  generated count == edge weight for every successful request
#   7  Attack label matches the synthetic-graph class
#   8  no NaN in target features 9  no inf in target features
#   10 numeric ranges valid      11 port ranges valid
#   12 byte/packet relationships valid (data_cleaning.py 3.2.e)
#   13 no IP columns among the CTGAN target features
# ============================================================

import glob
import os

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C
from src.attack_flows import step3_2_prepare as C2   # reuse its stats helper (no split rows touched)

COMBINED = os.path.join(C.S3_GENERATION, "synthetic_attack_flows.csv")
VMARK_DIR = C.S3_GENERATION


def _collect_batches():
    frames = []
    for atk in C.ATTACKS_STEP3:
        adir = os.path.join(C.S3_GENERATION, atk)
        if not os.path.isdir(adir):
            continue
        for f in sorted(glob.glob(os.path.join(adir, "batch_*.csv"))):
            done = f[:-4] + "_DONE"
            if not os.path.exists(done):
                print(f"  WARNING: {f} has no DONE marker -- excluded from assembly")
                continue
            frames.append(pd.read_csv(f, low_memory=False))
    if not frames:
        raise SystemExit("ABORT: no completed generation batches found")
    return pd.concat(frames, ignore_index=True)


def _collect_logs():
    frags = sorted(glob.glob(os.path.join(C.S3_LOGS, "genlog_*_b*.csv")))
    if not frags:
        raise SystemExit("ABORT: no generation-log fragments found")
    log = pd.concat([pd.read_csv(f) for f in frags], ignore_index=True)
    log = log.sort_values(["attack", "request_id"]).reset_index(drop=True)
    C.atomic_write_csv(log, C.GEN_LOG_CSV)
    return log


def main():
    C.ensure_dirs()
    C.assert_scale_inputs()
    # Guarantee this scale's split-stats CSVs exist even if the 3.2-3.5 prep
    # cell was skipped (e.g. RUN_FROM_STEP=3.6 in the notebook, which starts
    # directly at generation). This never touches the shared split rows --
    # it only (re)writes the two small derived report CSVs Step 5 checks for.
    C2._ensure_stats_from_existing_splits()
    with C.stage_banner("3.7 ASSEMBLE + VALIDATE GENERATED ATTACK FLOWS"):
        print(f"  SCALE={C.SCALE}  graph={C.STEP2_DIR}/  generation={C.S3_GENERATION}/")
        edges = pd.read_csv(C.STEP2_EDGES)
        nodes = pd.read_csv(C.STEP2_NODES)
        for d in (edges, nodes):
            d["attack"] = d["attack"].astype(str).str.strip().str.lower()
        edges = edges[edges["attack"].isin(C.ATTACKS_STEP3)]
        nodes = nodes[nodes["attack"].isin(C.ATTACKS_STEP3)]
        node_ips = {a: set(g["ip"]) for a, g in nodes.groupby("attack")}
        edge_set = {a: set(zip(g["src"], g["dst"])) for a, g in edges.groupby("attack")}
        weight_sum = edges.groupby("attack")["weight"].sum().to_dict()

        df = _collect_batches()
        log = _collect_logs()
        print(f"  assembled {len(df):,} generated rows / {len(log):,} log records")

        fails = []

        # 1-2 scope
        if int((df["Attack"] == "benign").sum()): fails.append("benign rows present")
        if int((df["Attack"] == "ransomware").sum()): fails.append("ransomware rows present")
        if not set(df["Attack"].unique()) <= set(C.ATTACKS_STEP3):
            fails.append(f"out-of-scope classes: {sorted(set(df['Attack'].unique()) - set(C.ATTACKS_STEP3))}")

        # 3-5 / 7 topology
        for atk, g in df.groupby("Attack"):
            bad_src = ~g["IPV4_SRC_ADDR"].isin(node_ips.get(atk, set()))
            bad_dst = ~g["IPV4_DST_ADDR"].isin(node_ips.get(atk, set()))
            if bad_src.any(): fails.append(f"{atk}: {int(bad_src.sum())} rows with unknown src IP")
            if bad_dst.any(): fails.append(f"{atk}: {int(bad_dst.sum())} rows with unknown dst IP")
            pair_ok = [(s, d) in edge_set.get(atk, set())
                       for s, d in zip(g["IPV4_SRC_ADDR"], g["IPV4_DST_ADDR"])]
            if not all(pair_ok):
                fails.append(f"{atk}: {pair_ok.count(False)} rows whose (src,dst) is not a synthetic edge")

        # 6 flow-count conservation (per successful request + global per attack)
        ok_log = log[log["generation_success"] == True]  # noqa: E712
        mism = ok_log[ok_log["generated_flow_count"] != ok_log["requested_flow_count"]]
        if len(mism):
            fails.append(f"{len(mism)} successful requests with generated != requested count")
        for atk, g in ok_log.groupby("attack"):
            gen_a = int(g["generated_flow_count"].sum())
            got_a = int((df["Attack"] == atk).sum())
            if gen_a != got_a:
                fails.append(f"{atk}: log says {gen_a} rows, table has {got_a}")
            if atk in weight_sum and gen_a != int(weight_sum[atk]):
                # only enforce when that attack is fully done
                if C.is_done(os.path.join(C.S3_GENERATION, atk)):
                    fails.append(f"{atk}: generated {gen_a} != edge-weight sum {int(weight_sum[atk])}")

        # 8-13 feature integrity
        num = df[C.TARGET_FEATURES].apply(pd.to_numeric, errors="coerce")
        if int(num.isna().sum().sum()): fails.append("NaN in target features")
        if int(np.isinf(num.to_numpy(dtype='float64')).sum()): fails.append("inf in target features")
        valid = C.physical_validity_mask(df.assign(**{c: num[c] for c in C.TARGET_FEATURES}))
        if not valid.all():
            fails.append(f"{int((~valid).sum())} rows fail physical-plausibility (ranges/ports/bytes-vs-pkts)")
        if any(c.startswith("IPV4") for c in C.TARGET_FEATURES):
            fails.append("IP column present among CTGAN target features")
        for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"):
            if df[c].isna().any(): fails.append(f"{c} has nulls")

        if fails:
            for f in fails:
                print("  FAIL:", f)
            raise SystemExit(f"ABORT: {len(fails)} hard validation failure(s) -- VALIDATION_DONE not written")
        print("  all 13 hard checks passed")

        # ---- support-aware metrics ------------------------------------
        n_req = len(log)
        n_ok = int((log["generation_success"] == True).sum())          # noqa: E712
        n_fail = n_req - n_ok
        by_model = log["selected_model"].value_counts().to_dict()
        n_backoff = int((log["backoff_level"] > 0).sum())
        full_cov = int((log["full_condition_support"] >= C.MIN_SUPPORT).sum())

        gsum_rows = []
        for atk in C.ATTACKS_STEP3:
            gl = log[log["attack"] == atk]
            if len(gl) == 0:
                continue
            gsum_rows.append({
                "Attack": atk,
                "synthetic_edges": int(len(gl)),
                "requested_flows": int(gl["requested_flow_count"].sum()),
                "generated_flows": int(gl.loc[gl["generation_success"] == True, "generated_flow_count"].sum()),  # noqa: E712
                "failed_requests": int((gl["generation_success"] != True).sum()),      # noqa: E712
                "full_condition_coverage": round(
                    (gl["full_condition_support"] >= C.MIN_SUPPORT).mean(), 4),
                "backoff_rate": round((gl["backoff_level"] > 0).mean(), 4),
                "variant_D": int((gl["selected_model"] == "variant_D").sum()),
                "variant_B": int((gl["selected_model"] == "variant_B").sum()),
                "variant_C": int((gl["selected_model"] == "variant_C").sum()),
                "variant_A": int((gl["selected_model"] == "variant_A").sum()),
            })
        C.atomic_write_csv(pd.DataFrame(gsum_rows),
                           os.path.join(C.S3_REPORTS, "generation_summary.csv"))

        backoff_rows = [{
            "scope": "GLOBAL",
            "generation_requests": n_req,
            "failed_requests": n_fail,
            "failure_rate": round(n_fail / max(n_req, 1), 4),
            "full_condition_coverage": round(full_cov / max(n_req, 1), 4),
            "backoff_rate": round(n_backoff / max(n_req, 1), 4),
            "variant_D_usage_rate": round(by_model.get("variant_D", 0) / max(n_req, 1), 4),
            "variant_B_fallback_rate": round(by_model.get("variant_B", 0) / max(n_req, 1), 4),
            "variant_C_fallback_rate": round(by_model.get("variant_C", 0) / max(n_req, 1), 4),
            "variant_A_fallback_rate": round(by_model.get("variant_A", 0) / max(n_req, 1), 4),
        }]
        for atk in C.ATTACKS_STEP3:
            gl = log[log["attack"] == atk]
            if len(gl) == 0:
                continue
            m = gl["selected_model"].value_counts().to_dict()
            k = len(gl)
            backoff_rows.append({
                "scope": atk,
                "generation_requests": k,
                "failed_requests": int((gl["generation_success"] != True).sum()),   # noqa: E712
                "failure_rate": round((gl["generation_success"] != True).mean(), 4),  # noqa: E712
                "full_condition_coverage": round((gl["full_condition_support"] >= C.MIN_SUPPORT).mean(), 4),
                "backoff_rate": round((gl["backoff_level"] > 0).mean(), 4),
                "variant_D_usage_rate": round(m.get("variant_D", 0) / k, 4),
                "variant_B_fallback_rate": round(m.get("variant_B", 0) / k, 4),
                "variant_C_fallback_rate": round(m.get("variant_C", 0) / k, 4),
                "variant_A_fallback_rate": round(m.get("variant_A", 0) / k, 4),
            })
        C.atomic_write_csv(pd.DataFrame(backoff_rows),
                           os.path.join(C.S3_REPORTS, "backoff_summary.csv"))

        log_summary = (log.groupby(["attack", "selected_model", "backoff_level"])
                          .agg(requests=("request_id", "count"),
                               generated_flows=("generated_flow_count", "sum"),
                               failed=("generation_success", lambda s: int((s != True).sum())))  # noqa: E712
                          .reset_index())
        C.atomic_write_csv(log_summary,
                           os.path.join(C.S3_REPORTS, "generation_log_summary.csv"))

        C.atomic_write_csv(df, COMBINED)
        C.write_done(VMARK_DIR, {
            "generated_rows": int(len(df)),
            "requested_rows": int(log["requested_flow_count"].sum()),
            "failed_requests": n_fail,
            "checks_passed": 13,
        }, name="VALIDATION_DONE")
        C.update_status("validation_status", "complete")
        print(f"\n[STEP 3.7] wrote {COMBINED}  ({len(df):,} rows)")
        print(f"           failure_rate={n_fail / max(n_req,1):.4f}  "
              f"backoff_rate={n_backoff / max(n_req,1):.4f}  "
              f"variant_D_usage={by_model.get('variant_D',0) / max(n_req,1):.4f}")


if __name__ == "__main__":
    main()
