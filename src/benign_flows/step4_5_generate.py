# ============================================================
# STEP 4.5 -- SUPPORT-AWARE BENIGN-FLOW GENERATION   (batched, restart-safe)
#
#   python -m src.benign_flows.step4_5_generate
#
# For every edge of the Step-2 synthetic BENIGN graph at the selected SCALE
# (this project: SCALE="smaller" -> results/step2/smaller/, attack == "benign"):
#   1. read src IP, dst IP, edge weight
#   2. derive the structural condition from the synthetic graph:
#        src_out_degree_cat = bin(src node out-degree)   (FINAL 2-bin thresholds)
#        dst_in_degree_cat  = bin(dst node in-degree)
#        src_role / dst_role = node roles straight from synthetic_nodes.csv
#   3. support-aware hierarchical back-off (TRAIN support only) picks the model:
#        LEVEL 0  degree + role     -> Variant D    (support_D >= 10)
#        LEVEL 1  degree            -> Variant B
#        LEVEL 2  role              -> Variant C
#          (if both B and C are supported, the one with GREATER train support
#           wins; ties -> degree / Variant B)
#        LEVEL 3  (none)            -> Variant A    (baseline, if the model is valid)
#        else                      -> request FAILED and LOGGED (never dropped)
#   4. request exactly `weight` benign flow rows from the chosen model+condition
#   5. attach the edge's src/dst IPs  (CTGAN NEVER makes IPs)  and Attack=benign
#   6. validate physically (same rules as data_cleaning.py 3.2.e); re-draw
#      invalid rows up to RESAMPLE_ROUNDS, else STOP the batch (no DONE marker)
#   7. flush a batch atomically, verify, write batch_XXXX_DONE
#
# Crash recovery: a batch with batch_XXXX_DONE is skipped; generation resumes
# at the first unfinished batch. generated_flow_count is checked against the
# edge weight for every edge and globally.
#
# Outputs (SCALE="smaller" -> results/small_scale/step4_benign/):
#   generation/batches/batch_XXXX.csv   (+ batch_XXXX_DONE)
#   logs/genlog_benign_bXXXX.csv         (per-edge log)
#   generation/DONE                      (all batches done)
# ============================================================

import argparse
import os

import numpy as np
import pandas as pd

from src.benign_flows import step4_common as S4

OUT_SCHEMA = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"] + S4.TARGET_FEATURES
INT_TARGETS = S4.COUNT_TARGETS + S4.PORT_TARGETS + ["FLOW_DURATION_MILLISECONDS",
                                                   "PROTOCOL", "TCP_FLAGS"]


# ------------------------------------------------------------------
# support lookups (TRAIN only) built from step4_2 output
# ------------------------------------------------------------------
def _load_support(variant: str) -> dict:
    path = os.path.join(S4.S4_SUPPORT, f"support_variant_{variant}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} missing -- run step4_2_support.py first")
    tbl = pd.read_csv(path)
    cols = S4.VARIANT_CONDITIONS[variant]
    if not cols:  # Variant A -- single implicit group
        return {(): int(tbl["support"].iloc[0])}
    out = {}
    for _, r in tbl.iterrows():
        out[tuple(str(r[c]) for c in cols)] = int(r["support"])
    return out


class BackoffDecider:
    def __init__(self, a_model_valid: bool):
        self.sup = {v: _load_support(v) for v in S4.VARIANTS}
        self.a_model_valid = a_model_valid
        self.sA = int(next(iter(self.sup["A"].values()), 0))

    def decide(self, sdc, ddc, sr, dr) -> dict:
        sD = self.sup["D"].get((sdc, ddc, sr, dr), 0)
        sB = self.sup["B"].get((sdc, ddc), 0)
        sC = self.sup["C"].get((sr, dr), 0)
        sA = self.sA

        if sD >= S4.MIN_SUPPORT:
            model, level = "D", 0
        elif sB >= S4.MIN_SUPPORT and sC >= S4.MIN_SUPPORT:
            model, level = ("B", 1) if sB >= sC else ("C", 2)
        elif sB >= S4.MIN_SUPPORT:
            model, level = "B", 1
        elif sC >= S4.MIN_SUPPORT:
            model, level = "C", 2
        elif self.a_model_valid and sA >= S4.MIN_SUPPORT:
            model, level = "A", 3
        else:
            model, level = None, -1

        cond = {}
        if model is not None:
            vals = {"src_out_degree_cat": sdc, "dst_in_degree_cat": ddc,
                    "src_role": sr, "dst_role": dr}
            cond = {c: vals[c] for c in S4.VARIANT_CONDITIONS[model]}
        return dict(model=model, level=level, condition=cond,
                    support_D=sD, support_B=sB, support_C=sC, support_A=sA)


# ------------------------------------------------------------------
# model cache
# ------------------------------------------------------------------
class ModelCache:
    def __init__(self):
        self.api = S4.get_sdv_api()
        self._cache = {}

    def valid(self, variant: str) -> bool:
        vdir = os.path.join(S4.S4_MODELS, f"benign_variant_{variant}")
        return S4.is_done(vdir) and os.path.exists(os.path.join(vdir, "model.pkl"))

    def get(self, variant: str):
        if variant not in self._cache:
            vdir = os.path.join(S4.S4_MODELS, f"benign_variant_{variant}")
            mp = os.path.join(vdir, "model.pkl")
            if not self.valid(variant):
                raise SystemExit(
                    f"ABORT: Benign Variant {variant} model not trained (missing {mp}). "
                    f"Run: python -m src.benign_flows.step4_4_train --variant {variant}"
                )
            print(f"  loading Benign Variant {variant} model ...")
            self._cache[variant] = self.api.CTGANSynthesizer.load(mp)
        return self._cache[variant]

    def sample(self, variant: str, condition: dict, n: int, seed: int) -> pd.DataFrame:
        if n <= 0:
            return pd.DataFrame(columns=S4.TARGET_FEATURES)
        S4.set_global_seeds(seed)
        synth = self.get(variant)
        if condition:
            cond_obj = self.api.Condition(num_rows=int(n), column_values=dict(condition))
            got = self.api.sample_from_conditions(synth, [cond_obj])
        else:  # Variant A -- unconditional
            got = synth.sample(num_rows=int(n))
        return got.reset_index(drop=True)


# ------------------------------------------------------------------
# benign graph loading
# ------------------------------------------------------------------
def load_benign_graph():
    edges = pd.read_csv(S4.STEP2_EDGES)
    nodes = pd.read_csv(S4.STEP2_NODES)
    for d in (edges, nodes):
        d["attack"] = d["attack"].astype(str).str.strip().str.lower()

    edges = edges[edges["attack"] == S4.BENIGN_LABEL].copy().reset_index(drop=True)
    nodes = nodes[nodes["attack"] == S4.BENIGN_LABEL].copy().reset_index(drop=True)
    if len(edges) == 0 or len(nodes) == 0:
        raise SystemExit(
            f"ABORT: no synthetic BENIGN graph found in {S4.STEP2_DIR}/ "
            "(need rows with attack == 'benign' in synthetic_edge_list.csv / synthetic_nodes.csv)."
        )

    node_attr, node_ips = {}, set()
    for _, r in nodes.iterrows():
        node_attr[r["ip"]] = (int(r["in_degree"]), int(r["out_degree"]), str(r["role"]))
        node_ips.add(r["ip"])
    valid_edges = set(zip(edges["src"], edges["dst"]))

    # every edge endpoint must be a known node
    miss = {ip for pair in valid_edges for ip in pair} - node_ips
    if miss:
        raise SystemExit(f"ABORT: {len(miss)} benign edge endpoints missing from synthetic_nodes.csv")
    return edges, node_attr, node_ips, valid_edges


# ------------------------------------------------------------------
# batching
# ------------------------------------------------------------------
def make_batches(edge_rows: list[dict]):
    batches, cur, cur_rows = [], [], 0
    for er in edge_rows:
        if cur and (len(cur) >= S4.EDGES_PER_BATCH
                    or cur_rows + er["edge_weight"] > S4.MAX_ROWS_PER_BATCH):
            batches.append(cur)
            cur, cur_rows = [], 0
        cur.append(er)
        cur_rows += er["edge_weight"]
    if cur:
        batches.append(cur)
    return batches


def _finalize_ints(df: pd.DataFrame) -> pd.DataFrame:
    for c in INT_TARGETS:
        if c in df.columns:
            df[c] = np.rint(pd.to_numeric(df[c], errors="coerce")).astype("Int64")
    return df


def _log_row(er, generated, success, reason):
    return {
        "request_id": er["request_id"],
        "edge_idx": er["edge_idx"],
        "attack": S4.BENIGN_LABEL,
        "source_ip": er["source_ip"],
        "destination_ip": er["destination_ip"],
        "edge_weight": er["edge_weight"],
        "requested_flow_count": er["requested_flow_count"],
        "requested_src_degree_cat": er["requested_src_degree_cat"],
        "requested_dst_degree_cat": er["requested_dst_degree_cat"],
        "requested_src_role": er["requested_src_role"],
        "requested_dst_role": er["requested_dst_role"],
        "full_condition_support": er["support_D"],
        "degree_condition_support": er["support_B"],
        "role_condition_support": er["support_C"],
        "baseline_support": er["support_A"],
        "variant_attempted": "variant_D",
        "selected_model": f"benign_variant_{er['model']}" if er["model"] else "NONE",
        "selected_variant": er["model"] or "NONE",
        "selected_condition": ", ".join(f"{k}={v}" for k, v in er["condition"].items()) or "(none)",
        "backoff_level": er["level"],
        "generated_flow_count": generated,
        "generation_success": success,
        "failure_reason": reason,
        "timestamp": pd.Timestamp.now().isoformat(timespec="seconds"),
    }


# ------------------------------------------------------------------
# main generation
# ------------------------------------------------------------------
def generate(models: ModelCache, decider: BackoffDecider,
             edges, node_attr, node_ips, valid_edges):
    os.makedirs(S4.S4_GEN_BATCHES, exist_ok=True)
    gen_done_dir = S4.S4_GENERATION
    if S4.is_done(gen_done_dir) and not S4.FORCE_RERUN:
        print("[GEN benign] status: COMPLETE  action: SKIPPING (STEP4_FORCE_RERUN=1 to override)")
        return

    # ---- build per-edge requests ----------------------------------------
    edge_rows = []
    for i, r in edges.iterrows():
        src, dst, w = r["src"], r["dst"], int(r["weight"])
        s_in, s_out, s_role = node_attr[src]
        d_in, d_out, d_role = node_attr[dst]
        sdc = S4.bin_src_out_degree(s_out)
        ddc = S4.bin_dst_in_degree(d_in)
        dec = decider.decide(sdc, ddc, s_role, d_role)
        edge_rows.append(dict(
            request_id=f"benign_{i:05d}", edge_idx=int(i),
            source_ip=src, destination_ip=dst, edge_weight=w,
            requested_flow_count=w,
            requested_src_degree_cat=sdc, requested_dst_degree_cat=ddc,
            requested_src_role=s_role, requested_dst_role=d_role,
            **dec,
        ))

    batches = make_batches(edge_rows)
    total_req = sum(x["edge_weight"] for x in edge_rows)
    print(f"[GEN benign] {len(edges)} edges, {total_req:,} flows, {len(batches)} batch(es)")
    n_fail_edges = sum(1 for er in edge_rows if er["model"] is None)
    if n_fail_edges:
        print(f"  WARNING: {n_fail_edges} edge(s) have NO supported generation path "
              f"(will be logged as failures, never silently dropped)")

    total_gen = 0
    for bi, batch in enumerate(batches, start=1):
        bcsv = os.path.join(S4.S4_GEN_BATCHES, f"batch_{bi:04d}.csv")
        bmark = os.path.join(S4.S4_GEN_BATCHES, f"batch_{bi:04d}_DONE")
        if (not S4.FORCE_RERUN and os.path.exists(bmark)
                and os.path.exists(bcsv) and os.path.getsize(bcsv) > 0):
            total_gen += int(S4.read_json(bmark, {}).get("generated_rows", 0))
            print(f"  batch {bi:04d}/{len(batches)}: already DONE -- skipping")
            continue

        rows_out, log_rows = [], []

        # group edges by (model, condition) so each model is queried once per condition
        groups: dict = {}
        for er in batch:
            if er["model"] is None:
                continue
            key = (er["model"], tuple(sorted(er["condition"].items())))
            groups.setdefault(key, []).append(er)

        pool: dict = {}
        for (model, cond_items), ers in groups.items():
            cond = dict(cond_items)
            need = sum(x["edge_weight"] for x in ers)
            seed = S4.stable_seed(S4.RANDOM_SEED + bi * 100_003, S4.BENIGN_LABEL, model, cond_items)
            draw = models.sample(model, cond, need, seed)
            draw = _finalize_ints(draw[S4.TARGET_FEATURES].copy()).astype("float64")
            good = draw[S4.physical_validity_mask(draw).to_numpy()]
            rounds = 0
            while len(good) < need and rounds < S4.RESAMPLE_ROUNDS:
                rounds += 1
                extra = models.sample(model, cond, (need - len(good)) * 2 + 16,
                                      seed + 7919 * rounds)
                extra = _finalize_ints(extra[S4.TARGET_FEATURES].copy()).astype("float64")
                extra = extra[S4.physical_validity_mask(extra).to_numpy()]
                good = pd.concat([good, extra], ignore_index=True)
            if len(good) < need:
                bad_path = os.path.join(S4.S4_LOGS, f"invalid_benign_b{bi:04d}_{model}.csv")
                S4.atomic_write_csv(draw, bad_path)
                raise SystemExit(
                    f"ABORT: batch {bi} (benign, model {model}, cond {cond}) could not "
                    f"produce {need} physically-valid rows after {S4.RESAMPLE_ROUNDS} "
                    f"re-draws (got {len(good)}). Sample dumped to {bad_path}. "
                    f"No DONE marker written."
                )
            pool[(model, cond_items)] = good.sample(n=need, replace=False,
                                                    random_state=seed).reset_index(drop=True)

        cursors = {k: 0 for k in pool}
        for er in batch:
            w = er["edge_weight"]
            if er["model"] is None:
                log_rows.append(_log_row(er, generated=0, success=False,
                                         reason="no supported condition even at baseline level"))
                continue
            key = (er["model"], tuple(sorted(er["condition"].items())))
            cur = cursors[key]
            chunk = pool[key].iloc[cur:cur + w].copy()
            cursors[key] = cur + w
            assert len(chunk) == w, f"row shortfall for {er['request_id']}"

            # structural guards (constructed by us -- assert they hold)
            assert er["source_ip"] in node_ips, "src IP not a graph node -- ABORT"
            assert er["destination_ip"] in node_ips, "dst IP not a graph node -- ABORT"
            assert (er["source_ip"], er["destination_ip"]) in valid_edges, \
                "src/dst pair is not a synthetic benign edge -- ABORT"

            chunk.insert(0, "Attack", S4.BENIGN_LABEL)
            chunk.insert(0, "IPV4_DST_ADDR", er["destination_ip"])
            chunk.insert(0, "IPV4_SRC_ADDR", er["source_ip"])
            rows_out.append(chunk[OUT_SCHEMA])
            log_rows.append(_log_row(er, generated=w, success=True, reason=""))
            total_gen += w

        batch_df = (pd.concat(rows_out, ignore_index=True)
                    if rows_out else pd.DataFrame(columns=OUT_SCHEMA))
        batch_df = _finalize_ints(batch_df)

        # ---- hard checks on the assembled batch ----
        for lbl in S4.FORBIDDEN_LABELS:
            assert int((batch_df["Attack"] == lbl).sum()) == 0, f"{lbl} row in benign batch -- ABORT"
        assert set(batch_df["Attack"].unique()) <= {S4.BENIGN_LABEL}, "non-benign label -- ABORT"
        assert S4.physical_validity_mask(
            batch_df.astype({c: "float64" for c in S4.TARGET_FEATURES})).all(), \
            "physically-invalid row slipped into batch -- ABORT"
        got_rows = sum(r["generated_flow_count"] for r in log_rows if r["generation_success"])
        want_rows = sum(r["requested_flow_count"] for r in log_rows if r["generation_success"])
        assert got_rows == want_rows == len(batch_df), "flow-count conservation broken -- ABORT"

        S4.atomic_write_csv(batch_df, bcsv)
        S4.atomic_write_csv(pd.DataFrame(log_rows),
                            os.path.join(S4.S4_LOGS, f"genlog_benign_b{bi:04d}.csv"))
        S4.write_done(S4.S4_GEN_BATCHES, {
            "batch": bi, "edges": len(batch),
            "generated_rows": int(len(batch_df)),
            "requested_rows": int(want_rows),
        }, name=f"batch_{bi:04d}_DONE")
        print(f"  batch {bi:04d}/{len(batches)}: {len(batch_df):,} rows written")

    # ---- per-run conservation check ------------------------------------
    want_total = int(edges["weight"].sum())
    ok_total = want_total - sum(er["edge_weight"] for er in edge_rows if er["model"] is None)
    if total_gen != ok_total:
        raise SystemExit(
            f"ABORT: benign generated {total_gen:,} flows but supported edge weights "
            f"sum to {ok_total:,} (total edge weight {want_total:,}, "
            f"{want_total - ok_total:,} on unsupported edges). No generation/DONE written."
        )
    S4.write_done(S4.S4_GENERATION, {
        "edges": int(len(edges)),
        "generated_rows": total_gen,
        "edge_weight_sum": want_total,
        "unsupported_edge_weight": want_total - ok_total,
        "failed_edges": n_fail_edges,
    })
    S4.update_status("generation_status", "complete")
    print(f"[GEN benign] status: COMPLETE  ({total_gen:,} flows generated; "
          f"{n_fail_edges} failed edge(s))")


def main():
    argparse.ArgumentParser(description="Step 4.5 benign flow generation").parse_args()
    S4.ensure_dirs()
    S4.assert_scale_inputs()
    with S4.stage_banner("4.5 SUPPORT-AWARE BENIGN-FLOW GENERATION"):
        print(f"  SCALE            : {S4.SCALE}")
        print(f"  Step-2 graph     : {S4.STEP2_DIR}/  (attack == 'benign')")
        print(f"  generation output: {S4.S4_GENERATION}/")
        print(f"  benign CTGAN     : {S4.S4_MODELS}/  (reused, never retrained)")
        S4.set_global_seeds(S4.RANDOM_SEED)
        edges, node_attr, node_ips, valid_edges = load_benign_graph()
        models = ModelCache()
        decider = BackoffDecider(a_model_valid=models.valid("A"))
        if not decider.a_model_valid:
            print("  NOTE: baseline Variant A model not valid yet -- LEVEL 3 back-off unavailable")
        generate(models, decider, edges, node_attr, node_ips, valid_edges)


if __name__ == "__main__":
    main()
