# ============================================================
# STEP 2: GENERATE A NEW GRAPH AT TARGET SCALE   (research guide sections 3 + 5)
#
# For each attack class, use the Step 1 fingerprint to synthesise a fresh
# weighted directed graph that preserves the structure:
#
#   3.1  scale option        -> new_nodes = round(FACTOR * original_nodes)
#   3.2  linear edge scaling  -> new_edges = round(new_nodes * original_edges
#                                                  / original_nodes)
#                                (average degree preserved; never quadratic)
#   5.3  directed configuration model on bootstrapped in/out-degree
#        sequences; self-loops and parallel edges removed, then the edge
#        count is topped back up to the target
#   5.4  topology first, then synthetic edge weights (flows per edge) drawn
#        by bootstrapping the real edge-weight list of that class
#   5.5  Low/Medium/High roles assigned to the new nodes in the same
#        proportions measured in the real graph (degree-ranked)
#   5.6  one shared reservoir of invented IPs (10.0.0.0/8, 10 000 addresses);
#        every class samples its nodes from it, so overlap emerges naturally
#
# ALL THREE SCALES are produced, each in its own folder:
#   results/step2/same/     FACTOR = 1.0
#   results/step2/larger/   FACTOR = LARGER_FACTOR
#   results/step2/smaller/  FACTOR = SMALLER_FACTOR
#
# Per scale folder:
#   synthetic_edge_list.csv   attack, src, dst, weight   (all classes)
#   synthetic_nodes.csv       attack, ip, in_degree, out_degree, role
#   synthetic_summary.csv     original vs synthetic, per class
#   report.txt                run log
# results/step2/reservoir.csv the shared invented-IP pool
#
# Inputs (from step1_fingerprint.py):
#   results/step1/step1_fingerprint_summary.csv
#   results/step1/step1_node_fingerprints.csv
#   results/step1/step1_edge_list.csv
# ============================================================

import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import networkx as nx

from src.config import CFG

STEP1_DIR = "results/step1"
OUT_ROOT  = "results/step2"

# scale factors + per-scale seed offsets, in config order (same, larger, smaller)
SCALES = {name: s["factor"] for name, s in CFG["scales"].items()}      # 1.0 / 2.0 / 0.5
_SCALE_ID = {name: s["seed_id"] for name, s in CFG["scales"].items()}  # 0 / 1 / 2
MIN_NODES = CFG["topology"]["min_nodes"]   # floor so a scaled-down class still forms a graph (2)

RESERVOIR_BASE   = CFG["topology"]["reservoir_base"]    # "10.0.0.0/8"
RESERVOIR_SIZE   = CFG["topology"]["reservoir_size"]    # 10_000
MASTER_SEED      = CFG["seeds"]["topology_master"]      # 42

ATTACKS = ["ddos", "dos", "scanning", "password", "backdoor",
           "injection", "xss", "mitm", "benign"]


# ------------------------------------------------------------
def build_reservoir(n, seed):
    """n distinct invented IPv4 addresses drawn from 10.0.0.0/8."""
    rng = np.random.default_rng(seed)
    # 10.0.0.0/8 has 2**24 hosts; sample distinct offsets
    offsets = rng.choice(2 ** 24, size=n, replace=False)
    ips = ["10.%d.%d.%d" % ((o >> 16) & 255, (o >> 8) & 255, o & 255)
           for o in offsets]
    return ips


def scaled_degree_sequences(in_deg, out_deg, new_nodes, new_edges, seed):
    """Bootstrap (in,out) pairs jointly, then force sum == new_edges on both."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(in_deg), size=new_nodes)
    ins  = in_deg[idx].astype(float)
    outs = out_deg[idx].astype(float)

    def rescale(seq):
        s = seq.sum()
        seq = (seq * (new_edges / s)) if s > 0 else np.zeros_like(seq)
        seq = np.rint(seq).astype(int)
        seq[seq < 0] = 0
        # fix residual by +/-1 on random positions
        diff = new_edges - int(seq.sum())
        while diff != 0:
            j = rng.integers(0, len(seq))
            if diff > 0:
                seq[j] += 1; diff -= 1
            elif seq[j] > 0:
                seq[j] -= 1; diff += 1
        return seq

    return rescale(ins), rescale(outs)


def config_model_topology(in_seq, out_seq, new_edges, seed):
    """Directed configuration model -> simple DiGraph, self-loops & parallels
    removed, edge count topped back up to new_edges."""
    G = nx.directed_configuration_model(in_seq, out_seq, seed=seed)
    G = nx.DiGraph(G)                          # collapse parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))  # drop self-loops

    rng = np.random.default_rng(seed + 1)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    guard = 0
    while G.number_of_edges() < new_edges and guard < 20 * new_edges + 1000:
        guard += 1
        u = nodes[rng.integers(0, n)]
        v = nodes[rng.integers(0, n)]
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
    # if we somehow overshot (parallels collapsed unevenly), trim random extras
    while G.number_of_edges() > new_edges:
        e = list(G.edges())[rng.integers(0, G.number_of_edges())]
        G.remove_edge(*e)
    return G


def assign_roles(G, p_low, p_med, p_high, seed):
    """Degree-ranked role assignment matching the measured proportions."""
    n = G.number_of_nodes()
    order = sorted(G.nodes(),
                   key=lambda x: (G.in_degree(x) + G.out_degree(x)),
                   reverse=True)
    n_high = int(round(p_high * n))
    n_med  = int(round(p_med * n))
    n_high = min(n_high, n)
    n_med  = min(n_med, n - n_high)
    roles = {}
    for i, node in enumerate(order):
        roles[node] = "high" if i < n_high else "medium" if i < n_high + n_med else "low"
    return roles


def generate_for_scale(scale_name, factor, summary, nodes_df, edge_df, reservoir):
    out_dir = os.path.join(OUT_ROOT, scale_name)
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 78)
    print(f"SCALE '{scale_name}'  (factor = {factor})")
    print("=" * 78)

    edge_rows, node_rows, summ_rows = [], [], []

    for ci, attack in enumerate(ATTACKS):
        srow = summary.loc[summary["Attack"] == attack].iloc[0]
        o_nodes, o_edges = int(srow["Nodes"]), int(srow["Edges"])

        nd = nodes_df[nodes_df["Attack"] == attack]
        in_deg  = nd["in_degree"].to_numpy()
        out_deg = nd["out_degree"].to_numpy()
        weights = edge_df.loc[edge_df["Attack"] == attack, "weight"].to_numpy()

        new_nodes = max(MIN_NODES, int(round(factor * o_nodes)))
        new_edges = int(round(new_nodes * o_edges / o_nodes)) if o_nodes else 0
        new_edges = min(new_edges, new_nodes * (new_nodes - 1))  # simple-graph cap
        new_edges = max(new_edges, 0)

        seed = MASTER_SEED + 1000 * (ci + 1) + _SCALE_ID[scale_name]

        in_seq, out_seq = scaled_degree_sequences(
            in_deg, out_deg, new_nodes, new_edges, seed)
        G = config_model_topology(in_seq, out_seq, new_edges, seed)

        # roles from measured proportions
        tot_roles = srow["role_low"] + srow["role_medium"] + srow["role_high"]
        if tot_roles > 0:
            p_low  = srow["role_low"]    / tot_roles
            p_med  = srow["role_medium"] / tot_roles
            p_high = srow["role_high"]   / tot_roles
        else:
            p_low, p_med, p_high = 1 / 3, 1 / 3, 1 / 3
        roles = assign_roles(G, p_low, p_med, p_high, seed)

        # map integer nodes -> invented IPs from the shared reservoir
        rng = np.random.default_rng(seed + 7)
        ip_pick = rng.choice(len(reservoir), size=G.number_of_nodes(), replace=False)
        ip_map = {node: reservoir[ip_pick[i]] for i, node in enumerate(G.nodes())}

        # synthetic edge weights: permute the real weight list (no sampling
        # with replacement -- that made syn_flows_total swing against real)
        wrng = np.random.default_rng(seed + 13)
        n_e = G.number_of_edges()
        if len(weights) == 0:
            w_draw = np.ones(n_e, dtype=int)
        elif n_e <= len(weights):
            # same / smaller: shuffle the real weight list, take what we need
            w_draw = wrng.permutation(weights)[:n_e]
        else:
            # larger: tile the list until it covers n_e, then shuffle and trim
            reps = int(np.ceil(n_e / len(weights)))
            w_draw = wrng.permutation(np.tile(weights, reps))[:n_e]

        for (u, v), w in zip(G.edges(), w_draw):
            edge_rows.append({"attack": attack, "src": ip_map[u],
                              "dst": ip_map[v], "weight": int(w)})
        for node in G.nodes():
            node_rows.append({"attack": attack, "ip": ip_map[node],
                              "in_degree": int(G.in_degree(node)),
                              "out_degree": int(G.out_degree(node)),
                              "role": roles[node]})

        rc = pd.Series(roles).value_counts()
        summ_rows.append({
            "attack": attack,
            "orig_nodes": o_nodes, "orig_edges": o_edges,
            "orig_avg_deg": round(o_edges / o_nodes, 4) if o_nodes else 0,
            "new_nodes": G.number_of_nodes(), "new_edges": G.number_of_edges(),
            "new_avg_deg": round(G.number_of_edges() / G.number_of_nodes(), 4)
                           if G.number_of_nodes() else 0,
            "edge_target": new_edges,
            "role_low": int(rc.get("low", 0)),
            "role_medium": int(rc.get("medium", 0)),
            "role_high": int(rc.get("high", 0)),
            "syn_flows_total": int(sum(w_draw)),
            "real_flows_total": int(weights.sum()) if len(weights) else 0,
        })
        print(f"  {attack:<11} nodes {o_nodes:>5} -> {G.number_of_nodes():<6} "
              f"edges {o_edges:>5} -> {G.number_of_edges():<6} "
              f"(target {new_edges}, avg deg "
              f"{o_edges / o_nodes:.2f} -> "
              f"{G.number_of_edges() / max(G.number_of_nodes(), 1):.2f})")

    pd.DataFrame(edge_rows).to_csv(os.path.join(out_dir, "synthetic_edge_list.csv"), index=False)
    pd.DataFrame(node_rows).to_csv(os.path.join(out_dir, "synthetic_nodes.csv"), index=False)
    sm = pd.DataFrame(summ_rows)
    sm.to_csv(os.path.join(out_dir, "synthetic_summary.csv"), index=False)
    print("\n" + sm.to_string(index=False))
    print(f"\n  wrote {out_dir}/synthetic_edge_list.csv")
    print(f"  wrote {out_dir}/synthetic_nodes.csv")
    print(f"  wrote {out_dir}/synthetic_summary.csv")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    summary  = pd.read_csv(os.path.join(STEP1_DIR, "step1_fingerprint_summary.csv"))
    nodes_df = pd.read_csv(os.path.join(STEP1_DIR, "step1_node_fingerprints.csv"))
    edge_df  = pd.read_csv(os.path.join(STEP1_DIR, "step1_edge_list.csv"))

    reservoir = build_reservoir(RESERVOIR_SIZE, MASTER_SEED)
    pd.DataFrame({"ip": reservoir}).to_csv(
        os.path.join(OUT_ROOT, "reservoir.csv"), index=False)
    print(f"shared reservoir: {RESERVOIR_SIZE:,} invented IPs from {RESERVOIR_BASE} "
          f"-> {OUT_ROOT}/reservoir.csv")

    for scale_name, factor in SCALES.items():
        generate_for_scale(scale_name, factor, summary, nodes_df, edge_df, reservoir)


if __name__ == "__main__":
    log_path = os.path.join(OUT_ROOT, "report.txt")
    os.makedirs(OUT_ROOT, exist_ok=True)

    class _Tee:
        def __init__(self, *s): self.s = s
        def write(self, x):
            for st in self.s: st.write(x)
        def flush(self):
            for st in self.s: st.flush()

    with open(log_path, "w", encoding="utf-8") as fh:
        with redirect_stdout(_Tee(sys.stdout, fh)):
            main()
    print(f"wrote {log_path}")
