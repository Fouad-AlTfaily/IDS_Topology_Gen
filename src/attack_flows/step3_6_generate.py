# ============================================================
# STEP 3.6 -- SUPPORT-AWARE ATTACK-FLOW GENERATION   (batched, restart-safe)
#
#   python -m src.attack_flows.step3_6_generate [--attack ddos]
#
# For every edge of the Step-2 synthetic attack graphs at the selected SCALE
# (this project: SCALE="smaller" -> results/step2/smaller/):
#   1. read src IP, dst IP, Attack, edge weight
#   2. derive the structural condition:
#        src_out_degree_cat = bin(src node out-degree)   (FINAL 2-bin thresholds)
#        dst_in_degree_cat  = bin(dst node in-degree)
#        src_role / dst_role = node roles straight from synthetic_nodes.csv
#   3. support-aware hierarchical back-off (TRAIN support only) picks the model:
#        level 0  Attack+deg+role      -> Variant D   (full support)
#        level 1  Attack+deg           -> Variant B
#        level 2  Attack+role          -> Variant C
#        level 3  Attack               -> Variant A
#        else                          -> request FAILED (never silently dropped)
#      If both degree and role conditions are supported, the one with the
#      GREATER empirical support wins (ties -> degree / Variant B).
#   4. request exactly `weight` flow rows from the chosen model+condition
#   5. attach the edge's src/dst IPs  (CTGAN NEVER makes IPs)
#   6. validate physically (same rules as data_cleaning.py 3.2.e); re-draw
#      invalid rows up to RESAMPLE_ROUNDS, else STOP the batch (no DONE marker)
#   7. flush a batch, verify, write batch_XXXX_DONE
#
# Crash recovery: an attack with <attack>/DONE is skipped; within an attack a
# batch with batch_XXXX_DONE is skipped; generation resumes at the first
# unfinished batch. generated_flow_count is checked against edge weight for
# every edge and globally.
#
# Outputs (SCALE="smaller"): results/small_scale/step3_ctgan/generation/smaller/
#              <attack>/batch_XXXX.csv (+DONE)
#          results/small_scale/step3_ctgan/logs/genlog_<attack>_bXXXX.csv
# ============================================================

import argparse
import os

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C

OUT_SCHEMA = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"] + C.TARGET_FEATURES
INT_TARGETS = C.COUNT_TARGETS + C.PORT_TARGETS + ["FLOW_DURATION_MILLISECONDS",
                                                 "PROTOCOL", "TCP_FLAGS"]


# ------------------------------------------------------------------
# support lookups (TRAIN only) built from step3_3 output
# ------------------------------------------------------------------
def _load_support(variant: str) -> dict:
    path = os.path.join(C.S3_SUPPORT, f"support_variant_{variant}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"ABORT: {path} missing -- run step3_3_support.py first")
    tbl = pd.read_csv(path)
    cols = C.VARIANT_CONDITIONS[variant]
    out = {}
    for _, r in tbl.iterrows():
        key = tuple(str(r[c]) for c in cols)
        out[key] = int(r["support"])
    return out


class BackoffDecider:
    def __init__(self):
        self.sup = {v: _load_support(v) for v in C.VARIANTS}

    def decide(self, atk, sdc, ddc, sr, dr):
        sD = self.sup["D"].get((atk, sdc, ddc, sr, dr), 0)
        sB = self.sup["B"].get((atk, sdc, ddc), 0)
        sC = self.sup["C"].get((atk, sr, dr), 0)
        sA = self.sup["A"].get((atk,), 0)

        if sD >= C.MIN_SUPPORT:
            model, level = "D", 0
        elif sB >= C.MIN_SUPPORT and sC >= C.MIN_SUPPORT:
            model, level = ("B", 1) if sB >= sC else ("C", 2)
        elif sB >= C.MIN_SUPPORT:
            model, level = "B", 1
        elif sC >= C.MIN_SUPPORT:
            model, level = "C", 2
        elif sA >= C.MIN_SUPPORT:
            model, level = "A", 3
        else:
            model, level = None, -1

        cond = {}
        if model is not None:
            vals = {"Attack": atk, "src_out_degree_cat": sdc, "dst_in_degree_cat": ddc,
                    "src_role": sr, "dst_role": dr}
            cond = {c: vals[c] for c in C.VARIANT_CONDITIONS[model]}
        return dict(model=model, level=level, condition=cond,
                    support_D=sD, support_B=sB, support_C=sC, support_A=sA)


# ------------------------------------------------------------------
# model cache
# ------------------------------------------------------------------
class ModelCache:
    def __init__(self):
        self.api = C.get_sdv_api()
        self._cache = {}

    def get(self, variant: str):
        if variant not in self._cache:
            vdir = os.path.join(C.S3_MODELS, f"variant_{variant}")
            mp = os.path.join(vdir, "model.pkl")
            if not (C.is_done(vdir) and os.path.exists(mp)):
                raise SystemExit(
                    f"ABORT: Variant {variant} model not trained "
                    f"(missing {mp}). Run: python -m src.attack_flows.step3_5_train --variant {variant}"
                )
            print(f"  loading Variant {variant} model ...")
            self._cache[variant] = self.api.CTGANSynthesizer.load(mp)
        return self._cache[variant]

    def sample(self, variant: str, condition: dict, n: int, seed: int) -> pd.DataFrame:
        if n <= 0:
            return pd.DataFrame(columns=C.TARGET_FEATURES)
        C.set_global_seeds(seed)
        synth = self.get(variant)
        cond_obj = self.api.Condition(num_rows=int(n), column_values=dict(condition))
        got = self.api.sample_from_conditions(synth, [cond_obj])
        return got.reset_index(drop=True)


# ------------------------------------------------------------------
# graph loading
# ------------------------------------------------------------------
def load_graph():
    edges = pd.read_csv(C.STEP2_EDGES)
    nodes = pd.read_csv(C.STEP2_NODES)
    for d in (edges, nodes):
        d["attack"] = d["attack"].astype(str).str.strip().str.lower()

    bad = sorted(set(edges["attack"].unique()) - set(C.ATTACKS_STEP3))
    if "benign" in bad or "ransomware" in bad:
        # benign/ransomware graphs exist in step2 but are OUT OF SCOPE for step 3
        print(f"  NOTE: ignoring out-of-scope step-2 graphs: "
              f"{[b for b in bad if b in ('benign', 'ransomware')]}")
    edges = edges[edges["attack"].isin(C.ATTACKS_STEP3)].copy()
    nodes = nodes[nodes["attack"].isin(C.ATTACKS_STEP3)].copy()

    node_attr = {}
    node_ips = {a: set() for a in C.ATTACKS_STEP3}
    for _, r in nodes.iterrows():
        node_attr[(r["attack"], r["ip"])] = (int(r["in_degree"]),
                                             int(r["out_degree"]), str(r["role"]))
        node_ips[r["attack"]].add(r["ip"])
    valid_edges = {a: set() for a in C.ATTACKS_STEP3}
    for _, r in edges.iterrows():
        valid_edges[r["attack"]].add((r["src"], r["dst"]))
    return edges, node_attr, node_ips, valid_edges


# ------------------------------------------------------------------
# batching
# ------------------------------------------------------------------
def make_batches(edge_rows: list[dict]):
    batches, cur, cur_rows = [], [], 0
    for er in edge_rows:
        if cur and (len(cur) >= C.EDGES_PER_BATCH
                    or cur_rows + er["edge_weight"] > C.MAX_ROWS_PER_BATCH):
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


# ------------------------------------------------------------------
# one attack
# ------------------------------------------------------------------
def generate_attack(atk, edges, node_attr, node_ips, valid_edges, decider, models):
    adir = os.path.join(C.S3_GENERATION, atk)
    os.makedirs(adir, exist_ok=True)
    if C.is_done(adir):
        print(f"[GEN {atk}] status: COMPLETE  action: SKIPPING")
        return

    e = edges[edges["attack"] == atk].reset_index(drop=True)
    if len(e) == 0:
        print(f"[GEN {atk}] no synthetic edges -- nothing to generate")
        C.write_done(adir, {"attack": atk, "edges": 0, "generated_rows": 0})
        return

    edge_rows = []
    for i, r in e.iterrows():
        src, dst, w = r["src"], r["dst"], int(r["weight"])
        s_in, s_out, s_role = node_attr[(atk, src)]
        d_in, d_out, d_role = node_attr[(atk, dst)]
        sdc = C.bin_src_out_degree(s_out)
        ddc = C.bin_dst_in_degree(d_in)
        dec = decider.decide(atk, sdc, ddc, s_role, d_role)
        edge_rows.append(dict(
            request_id=f"{atk}_{i:05d}", edge_idx=int(i), attack=atk,
            source_ip=src, destination_ip=dst, edge_weight=w,
            requested_flow_count=w,
            requested_src_degree_cat=sdc, requested_dst_degree_cat=ddc,
            requested_src_role=s_role, requested_dst_role=d_role,
            **dec,
        ))

    batches = make_batches(edge_rows)
    print(f"[GEN {atk}] {len(e)} edges, {sum(x['edge_weight'] for x in edge_rows):,} flows, "
          f"{len(batches)} batch(es)")

    total_gen = 0
    for bi, batch in enumerate(batches, start=1):
        bcsv = os.path.join(adir, f"batch_{bi:04d}.csv")
        bmark = os.path.join(adir, f"batch_{bi:04d}_DONE")
        if os.path.exists(bmark) and os.path.exists(bcsv) and os.path.getsize(bcsv) > 0:
            done_info = C.read_json(bmark, {})
            total_gen += int(done_info.get("generated_rows", 0))
            print(f"  batch {bi:04d}/{len(batches)}: already DONE -- skipping")
            continue

        rows_out, log_rows = [], []

        # group edges of this batch by (model, condition) so each model is
        # queried once per distinct condition
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
            seed = C.stable_seed(C.RANDOM_SEED + bi * 100_003, atk, model, cond_items)
            draw = models.sample(model, cond, need, seed)

            # physical-validity rejection + moderate re-draw
            draw = draw[C.TARGET_FEATURES].copy()
            draw = _finalize_ints(draw).astype("float64")
            good = draw[C.physical_validity_mask(draw).to_numpy()]
            rounds = 0
            while len(good) < need and rounds < C.RESAMPLE_ROUNDS:
                rounds += 1
                extra = models.sample(model, cond, (need - len(good)) * 2 + 16,
                                      seed + 7919 * rounds)
                extra = _finalize_ints(extra[C.TARGET_FEATURES].copy()).astype("float64")
                extra = extra[C.physical_validity_mask(extra).to_numpy()]
                good = pd.concat([good, extra], ignore_index=True)
            if len(good) < need:
                bad_path = os.path.join(C.S3_LOGS,
                                        f"invalid_{atk}_b{bi:04d}_{model}.csv")
                C.atomic_write_csv(draw, bad_path)
                raise SystemExit(
                    f"ABORT: batch {bi} ({atk}, model {model}, cond {cond}) could not "
                    f"produce {need} physically-valid rows after {C.RESAMPLE_ROUNDS} "
                    f"re-draws (got {len(good)}). Sample dumped to {bad_path}. "
                    f"No DONE marker written."
                )
            pool[(model, cond_items)] = good.sample(n=need, replace=False,
                                                    random_state=seed).reset_index(drop=True)

        # hand rows out to edges in deterministic order, attach IPs
        cursors = {k: 0 for k in pool}
        for er in batch:
            w = er["edge_weight"]
            if er["model"] is None:
                log_rows.append(_log_row(er, generated=0, success=False,
                                         reason="no supported condition even at Attack level"))
                continue
            key = (er["model"], tuple(sorted(er["condition"].items())))
            cur = cursors[key]
            chunk = pool[key].iloc[cur:cur + w].copy()
            cursors[key] = cur + w
            assert len(chunk) == w, f"row shortfall for {er['request_id']}"

            # structural guards (constructed by us -- assert they hold)
            assert er["source_ip"] in node_ips[atk], "src IP not a graph node -- ABORT"
            assert er["destination_ip"] in node_ips[atk], "dst IP not a graph node -- ABORT"
            assert (er["source_ip"], er["destination_ip"]) in valid_edges[atk], \
                "src/dst pair is not a synthetic edge -- ABORT"

            chunk.insert(0, "Attack", atk)
            chunk.insert(0, "IPV4_DST_ADDR", er["destination_ip"])
            chunk.insert(0, "IPV4_SRC_ADDR", er["source_ip"])
            rows_out.append(chunk[OUT_SCHEMA])
            log_rows.append(_log_row(er, generated=w, success=True, reason=""))
            total_gen += w

        batch_df = (pd.concat(rows_out, ignore_index=True)
                    if rows_out else pd.DataFrame(columns=OUT_SCHEMA))
        batch_df = _finalize_ints(batch_df)

        # hard checks on the assembled batch
        assert int((batch_df["Attack"] == "benign").sum()) == 0
        assert int((batch_df["Attack"] == "ransomware").sum()) == 0
        assert batch_df["Attack"].nunique() <= 1
        assert C.physical_validity_mask(batch_df.astype({c: "float64" for c in C.TARGET_FEATURES})).all(), \
            "physically-invalid row slipped into batch -- ABORT"
        got_rows = sum(r["generated_flow_count"] for r in log_rows if r["generation_success"])
        want_rows = sum(r["requested_flow_count"] for r in log_rows if r["generation_success"])
        assert got_rows == want_rows == len(batch_df), "flow-count conservation broken -- ABORT"

        C.atomic_write_csv(batch_df, bcsv)
        C.atomic_write_csv(pd.DataFrame(log_rows),
                           os.path.join(C.S3_LOGS, f"genlog_{atk}_b{bi:04d}.csv"))
        C.write_done(adir, {
            "attack": atk, "batch": bi, "edges": len(batch),
            "generated_rows": int(len(batch_df)),
            "requested_rows": int(want_rows),
        }, name=f"batch_{bi:04d}_DONE")
        print(f"  batch {bi:04d}/{len(batches)}: {len(batch_df):,} rows written")

    # per-attack conservation check
    want_total = int(e["weight"].sum())
    if total_gen != want_total:
        raise SystemExit(
            f"ABORT: {atk} generated {total_gen:,} flows but synthetic edge weights "
            f"sum to {want_total:,}. No attack-level DONE written."
        )
    C.write_done(adir, {"attack": atk, "edges": int(len(e)),
                        "generated_rows": total_gen,
                        "edge_weight_sum": want_total})
    C.update_status("generation_status_per_attack",
                    {**(C.get_status("generation_status_per_attack") or {}),
                     atk: "complete"})
    print(f"[GEN {atk}] status: COMPLETE  ({total_gen:,} flows == edge-weight sum)")


def _log_row(er, generated, success, reason):
    return {
        "request_id": er["request_id"], "attack": er["attack"],
        "source_ip": er["source_ip"], "destination_ip": er["destination_ip"],
        "edge_weight": er["edge_weight"], "requested_flow_count": er["requested_flow_count"],
        "requested_src_degree_cat": er["requested_src_degree_cat"],
        "requested_dst_degree_cat": er["requested_dst_degree_cat"],
        "requested_src_role": er["requested_src_role"],
        "requested_dst_role": er["requested_dst_role"],
        "full_condition_support": er["support_D"],
        "degree_condition_support": er["support_B"],
        "role_condition_support": er["support_C"],
        "attack_condition_support": er["support_A"],
        "selected_model": f"variant_{er['model']}" if er["model"] else "NONE",
        "selected_condition": ", ".join(f"{k}={v}" for k, v in er["condition"].items()) or "NONE",
        "backoff_level": er["level"],
        "generated_flow_count": generated,
        "generation_success": success,
        "failure_reason": reason,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", choices=C.ATTACKS_STEP3, default=None,
                    help="generate a single attack class (default: all, in order)")
    args = ap.parse_args()

    C.ensure_dirs()
    C.assert_scale_inputs()
    os.makedirs(C.S3_GENERATION, exist_ok=True)
    with C.stage_banner("3.6 SUPPORT-AWARE ATTACK-FLOW GENERATION"):
        print(f"  SCALE            : {C.SCALE}")
        print(f"  Step-2 graph     : {C.STEP2_DIR}/")
        print(f"  generation output: {C.S3_GENERATION}/")
        print(f"  CTGAN models     : {C.S3_MODELS}/  (reused, never retrained)")
        C.set_global_seeds(C.RANDOM_SEED)
        edges, node_attr, node_ips, valid_edges = load_graph()
        decider = BackoffDecider()
        models = ModelCache()

        targets = [args.attack] if args.attack else C.ATTACKS_STEP3
        for atk in targets:
            generate_attack(atk, edges, node_attr, node_ips, valid_edges, decider, models)

        if not args.attack:
            all_done = all(C.is_done(os.path.join(C.S3_GENERATION, a))
                           for a in C.ATTACKS_STEP3)
            C.update_status("generation_status", "complete" if all_done else "partial")
            print(f"\n[STEP 3.6] generation {'COMPLETE' if all_done else 'PARTIAL'}")


if __name__ == "__main__":
    main()
