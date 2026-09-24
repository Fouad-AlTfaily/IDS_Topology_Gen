# ============================================================
# STEP 4.6 -- ASSEMBLE + VALIDATE GENERATED BENIGN FLOWS
#
# Concatenates every completed generation batch into one table, concatenates
# the per-batch generation-log fragments into logs/generation_log.csv, runs the
# hard integrity checks, and computes the support-aware / back-off metrics.
#
# Outputs (results/step4_benign/):
#   generation/synthetic_benign_flows.csv       the single final benign dataset
#   logs/generation_log.csv
#   reports/generation_summary.csv
#   reports/backoff_summary.csv
#   generation/VALIDATION_DONE
#
# HARD checks (any failure -> SystemExit, no VALIDATION_DONE):
#   1  no NaN in target features         2  no infinite values
#   3  exact column schema
#   4  valid PROTOCOL values (0..255)    5  valid TCP_FLAGS (>= 0, integral)
#   6  valid L4 ports (0..65535)
#   7  non-negative bytes / packets / flow duration
#   8  byte/packet relationships valid (data_cleaning.py 3.2.e)
#   9  Attack label is benign for every row
#   10 no ransomware / attack-class rows
#   11 every src IP is a synthetic benign graph node
#   12 every dst IP is a synthetic benign graph node
#   13 every (src,dst) pair is a synthetic benign edge
#   14 generated count == edge weight for every successful edge
#   15 global generated count == sum of supported edge weights
#   16 no incomplete batches included (batch_XXXX_DONE required)
# ============================================================

import glob
import os

import numpy as np
import pandas as pd

from src.benign_flows import step4_common as S4
from src.benign_flows import step4_1_prepare as S4P  # reuse its stats helper (no split rows touched)

FINAL = S4.SYNTHETIC_BENIGN_FLOWS
OUT_SCHEMA = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"] + S4.TARGET_FEATURES


def _collect_batches() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(os.path.join(S4.S4_GEN_BATCHES, "batch_*.csv"))):
        done = f[:-4] + "_DONE"
        if not os.path.exists(done):
            print(f"  WARNING: {f} has no DONE marker -- EXCLUDED from assembly")
            continue
        frames.append(pd.read_csv(f, low_memory=False))
    if not frames:
        raise SystemExit("ABORT: no completed benign generation batches found")
    return pd.concat(frames, ignore_index=True)


def _collect_logs() -> pd.DataFrame:
    frags = sorted(glob.glob(os.path.join(S4.S4_LOGS, "genlog_benign_b*.csv")))
    if not frags:
        raise SystemExit("ABORT: no benign generation-log fragments found")
    log = pd.concat([pd.read_csv(f) for f in frags], ignore_index=True)
    log = log.sort_values(["edge_idx"]).reset_index(drop=True)
    S4.atomic_write_csv(log, S4.GEN_LOG_CSV)
    return log


def main():
    S4.ensure_dirs()
    S4.assert_scale_inputs()
    # Guarantee this scale's benign split-stats CSV exists even if the 4.1 prep
    # cell was skipped -- never touches the shared split rows, only (re)writes
    # the derived report CSV Step 5 checks for.
    S4P._ensure_stats_from_existing_splits()
    with S4.stage_banner("4.6 ASSEMBLE + VALIDATE GENERATED BENIGN FLOWS"):
        print(f"  SCALE={S4.SCALE}  graph={S4.STEP2_DIR}/  generation={S4.S4_GENERATION}/")
        edges = pd.read_csv(S4.STEP2_EDGES)
        nodes = pd.read_csv(S4.STEP2_NODES)
        for d in (edges, nodes):
            d["attack"] = d["attack"].astype(str).str.strip().str.lower()
        edges = edges[edges["attack"] == S4.BENIGN_LABEL]
        nodes = nodes[nodes["attack"] == S4.BENIGN_LABEL]
        node_ips = set(nodes["ip"])
        edge_set = set(zip(edges["src"], edges["dst"]))
        weight_by_edge = {(s, d): int(w) for s, d, w in
                          zip(edges["src"], edges["dst"], edges["weight"])}
        weight_sum = int(edges["weight"].sum())

        df = _collect_batches()
        log = _collect_logs()
        print(f"  assembled {len(df):,} generated rows / {len(log):,} log records")

        fails = []

        # 3 schema
        if list(df.columns) != OUT_SCHEMA:
            fails.append(f"column schema mismatch: {list(df.columns)}")

        # 9-10 scope
        if set(df["Attack"].astype(str).unique()) != {S4.BENIGN_LABEL}:
            fails.append(f"non-benign Attack labels present: "
                         f"{sorted(set(df['Attack'].unique()) - {S4.BENIGN_LABEL})}")
        for lbl in S4.FORBIDDEN_LABELS:
            n = int((df["Attack"] == lbl).sum())
            if n:
                fails.append(f"{n} '{lbl}' rows present")

        # 1-2, 4-8 feature integrity
        num = df[S4.TARGET_FEATURES].apply(pd.to_numeric, errors="coerce")
        if int(num.isna().sum().sum()):
            fails.append("NaN in target features")
        if int(np.isinf(num.to_numpy(dtype="float64")).sum()):
            fails.append("infinite values in target features")
        if not ((num["PROTOCOL"] >= 0) & (num["PROTOCOL"] <= 255)).all():
            fails.append("PROTOCOL outside 0..255")
        if not ((num["TCP_FLAGS"] >= 0) & (num["TCP_FLAGS"] == num["TCP_FLAGS"].round())).all():
            fails.append("TCP_FLAGS negative or non-integral")
        for c in S4.PORT_TARGETS:
            if not ((num[c] >= 0) & (num[c] <= 65535)).all():
                fails.append(f"{c} outside 0..65535")
        for c in S4.COUNT_TARGETS + ["FLOW_DURATION_MILLISECONDS"]:
            if not (num[c] >= 0).all():
                fails.append(f"{c} has negative values")
        valid = S4.physical_validity_mask(df.assign(**{c: num[c] for c in S4.TARGET_FEATURES}))
        if not valid.all():
            fails.append(f"{int((~valid).sum())} rows fail physical-plausibility "
                         f"(ranges / ports / bytes-vs-pkts)")

        # 11-13 topology
        bad_src = ~df["IPV4_SRC_ADDR"].isin(node_ips)
        bad_dst = ~df["IPV4_DST_ADDR"].isin(node_ips)
        if bad_src.any():
            fails.append(f"{int(bad_src.sum())} rows with unknown src IP")
        if bad_dst.any():
            fails.append(f"{int(bad_dst.sum())} rows with unknown dst IP")
        pair_ok = [(s, d) in edge_set for s, d in zip(df["IPV4_SRC_ADDR"], df["IPV4_DST_ADDR"])]
        if not all(pair_ok):
            fails.append(f"{pair_ok.count(False)} rows whose (src,dst) is not a synthetic benign edge")

        # 14 per-edge conservation
        ok_log = log[log["generation_success"] == True]  # noqa: E712
        mism = ok_log[ok_log["generated_flow_count"] != ok_log["requested_flow_count"]]
        if len(mism):
            fails.append(f"{len(mism)} successful edges with generated != requested count")
        gen_by_pair = (df.groupby(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).size().to_dict())
        for (s, d), _r in ok_log.groupby(["source_ip", "destination_ip"]):
            want = weight_by_edge.get((s, d))
            got = gen_by_pair.get((s, d), 0)
            if want is not None and got != want:
                fails.append(f"edge {s}->{d}: table has {got} rows, edge weight is {want}")

        # 15 global conservation
        failed_weight = int(log.loc[log["generation_success"] != True, "edge_weight"].sum())  # noqa: E712
        expected = weight_sum - failed_weight
        if len(df) != expected:
            fails.append(f"global count {len(df)} != supported edge-weight sum {expected} "
                         f"(total {weight_sum}, {failed_weight} on failed edges)")

        # 16 no partial batches
        for f in sorted(glob.glob(os.path.join(S4.S4_GEN_BATCHES, "batch_*.csv"))):
            if not os.path.exists(f[:-4] + "_DONE"):
                fails.append(f"batch without DONE marker still on disk: {os.path.basename(f)}")

        if fails:
            for f in fails:
                print("  FAIL:", f)
            raise SystemExit(f"ABORT: {len(fails)} hard validation failure(s) "
                             f"-- VALIDATION_DONE not written")
        print("  all 16 hard checks passed")

        # ---- support-aware / back-off metrics ----------------------------
        n_req = len(log)
        n_ok = int((log["generation_success"] == True).sum())         # noqa: E712
        n_fail = n_req - n_ok
        by_model = log["selected_model"].value_counts().to_dict()
        n_backoff = int((log["backoff_level"] > 0).sum())
        full_cov = int((log["full_condition_support"] >= S4.MIN_SUPPORT).sum())

        gen_summary = pd.DataFrame([{
            "class": S4.BENIGN_LABEL,
            "synthetic_edges": n_req,
            "requested_flows": int(log["requested_flow_count"].sum()),
            "generated_flows": int(ok_log["generated_flow_count"].sum()),
            "failed_edges": n_fail,
            "final_rows": int(len(df)),
            "edge_weight_sum": weight_sum,
            "full_condition_coverage": round(full_cov / max(n_req, 1), 4),
            "backoff_rate": round(n_backoff / max(n_req, 1), 4),
            "variant_D": int(by_model.get("benign_variant_D", 0)),
            "variant_B": int(by_model.get("benign_variant_B", 0)),
            "variant_C": int(by_model.get("benign_variant_C", 0)),
            "variant_A": int(by_model.get("benign_variant_A", 0)),
            "variant_NONE": int(by_model.get("NONE", 0)),
        }])
        S4.atomic_write_csv(gen_summary, os.path.join(S4.S4_REPORTS, "generation_summary.csv"))

        backoff_summary = pd.DataFrame([{
            "scope": "GLOBAL",
            "generation_requests": n_req,
            "failed_requests": n_fail,
            "failure_rate": round(n_fail / max(n_req, 1), 4),
            "full_condition_coverage": round(full_cov / max(n_req, 1), 4),
            "backoff_rate": round(n_backoff / max(n_req, 1), 4),
            "level0_D_rate": round(by_model.get("benign_variant_D", 0) / max(n_req, 1), 4),
            "level1_B_rate": round(by_model.get("benign_variant_B", 0) / max(n_req, 1), 4),
            "level2_C_rate": round(by_model.get("benign_variant_C", 0) / max(n_req, 1), 4),
            "level3_A_rate": round(by_model.get("benign_variant_A", 0) / max(n_req, 1), 4),
            "rows_level0_D": int(ok_log.loc[ok_log["selected_model"] == "benign_variant_D",
                                            "generated_flow_count"].sum()),
            "rows_level1_B": int(ok_log.loc[ok_log["selected_model"] == "benign_variant_B",
                                            "generated_flow_count"].sum()),
            "rows_level2_C": int(ok_log.loc[ok_log["selected_model"] == "benign_variant_C",
                                            "generated_flow_count"].sum()),
            "rows_level3_A": int(ok_log.loc[ok_log["selected_model"] == "benign_variant_A",
                                            "generated_flow_count"].sum()),
        }])
        S4.atomic_write_csv(backoff_summary, os.path.join(S4.S4_REPORTS, "backoff_summary.csv"))

        S4.atomic_write_csv(df, FINAL)
        S4.write_done(S4.S4_GENERATION, {
            "final_rows": int(len(df)),
            "requested_rows": int(log["requested_flow_count"].sum()),
            "failed_edges": n_fail,
            "checks_passed": 16,
            "final_file": FINAL,
        }, name="VALIDATION_DONE")
        S4.update_status("validation_status", "complete")
        print(f"\n[STEP 4.6] wrote {FINAL}  ({len(df):,} rows)")
        print(f"           failure_rate={n_fail / max(n_req,1):.4f}  "
              f"backoff_rate={n_backoff / max(n_req,1):.4f}  "
              f"variant_D_usage={by_model.get('benign_variant_D',0) / max(n_req,1):.4f}")


if __name__ == "__main__":
    main()
