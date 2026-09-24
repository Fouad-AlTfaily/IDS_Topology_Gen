# ============================================================
# STEP 1: FINGERPRINT EACH ATTACK TYPE   (research guide section 2)
#
# For each attack type (DDoS, DoS, Scanning, Password, Backdoor, Injection,
# XSS, MITM, Benign -- ransomware DROPPED, flagged at 142 flows < 500 in the
# class analysis) build its communication subgraph from the CLEANED dataset
# and measure a structural fingerprint:
#   M1  degree distribution  -> avg in-degree, avg out-degree
#   M2  Louvain communities  -> count + size of each community
#   M3  PageRank roles       -> low / medium / high by 33rd / 66th percentile
# plus node/edge counts and the flow-row count (class-size check).
#
# Divergence from the guide snippet (kept, see new_changes.pdf /
# Generating_Datasets_5.pdf): the graph is WEIGHTED, weight = number of flows
# between an ordered IP pair. A plain DiGraph collapses repeated flows into
# one edge and loses how much traffic passed between two hosts. The same
# weighted graph is read two ways: unweighted degree (partner count, feeds
# Step 2) and strength (volume, reported). PageRank and Louvain run weighted.
#
# Input : data/NF-ToN-IoT_clean.csv   (from data_cleaning.py)
# Output: results/step1/
#   step1_fingerprint_summary.csv   the deliverable table (guide 2.6)
#   step1_edge_list.csv            every weighted edge: attack, src, dst, weight
#   step1_community_sizes.csv       one row per (attack, community) with size
#   step1_node_fingerprints.csv     raw per-node detail -> Step 2 / Step 3
#   step1_node_fingerprints.json    same, JSON
#   histograms/hist_<attack>.png    in- and out-degree distributions
#   report.txt                      run log
# ============================================================

import os
import sys
from contextlib import redirect_stdout

import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import CFG

DATA_PATH = "data/NF-ToN-IoT_clean.csv"
OUT_DIR   = "results/step1"
HIST_DIR  = os.path.join(OUT_DIR, "histograms")
os.makedirs(HIST_DIR, exist_ok=True)

# 9 classes. Ransomware is dropped on purpose (report 3.4 flag: 142 flows,
# below the project minimum of 500 -- too small to fit a degree distribution).
ATTACKS = ["ddos", "dos", "scanning", "password", "backdoor",
           "injection", "xss", "mitm", "benign"]
LABEL_COL = "Attack"
MIN_ROWS = CFG["data"]["min_class_rows"]   # 500; class-size check; matches the class-analysis minimum
LOUVAIN_SEED = CFG["seeds"]["louvain"]      # 42


# ------------------------------------------------------------
# Build a WEIGHTED directed graph: one edge per ordered (src, dst) pair,
# weight = how many flow rows had that pair. Repeated flows are not lost.
# ------------------------------------------------------------
def build_weighted_graph(sub):
    w = (sub.groupby(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"])
            .size().reset_index(name="weight"))
    G = nx.DiGraph()
    for _, r in w.iterrows():
        G.add_edge(r["IPV4_SRC_ADDR"], r["IPV4_DST_ADDR"], weight=int(r["weight"]))
    return G


def safe_mean(values):
    return round(float(np.mean(values)), 4) if len(values) > 0 else 0.0


# ------------------------------------------------------------
# M3: PageRank -> low / medium / high. Weighted PageRank (edges carry
# 'weight', which nx.pagerank uses by default). Discretised, never raw:
# raw scores are not comparable across graphs of different sizes.
# Guards: empty graph -> {}; single node -> "medium"; degenerate case
# where the 33rd == 66th percentile -> everyone "medium".
# ------------------------------------------------------------
def pagerank_roles(G):
    if G.number_of_nodes() == 0:
        return {}
    pr = nx.pagerank(G)
    if G.number_of_nodes() == 1:
        return {n: "medium" for n in pr}
    vals = np.array(list(pr.values()))
    lo, hi = np.percentile(vals, 33), np.percentile(vals, 66)
    if lo == hi:
        return {n: "medium" for n in pr}
    return {n: ("low" if s <= lo else "high" if s > hi else "medium")
            for n, s in pr.items()}


def main():
    df = pd.read_csv(DATA_PATH)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.lower()
    print(f"loaded {DATA_PATH}: {len(df):,} rows")
    present = set(df[LABEL_COL].unique())
    if "ransomware" in present:
        n_r = int((df[LABEL_COL] == "ransomware").sum())
        print(f"ransomware present ({n_r} flows) -> DROPPED per report 3.4 flag")

    summary_rows = []
    community_rows = []
    node_fingerprints = []
    edge_rows = []

    for attack in ATTACKS:
        sub = df[df[LABEL_COL] == attack]
        n_flows = len(sub)
        if n_flows < MIN_ROWS:
            print(f"FLAG: {attack} has only {n_flows} rows (< {MIN_ROWS})")

        G = build_weighted_graph(sub)
        n_nodes = G.number_of_nodes()

        # M0 -- the weighted edge list itself (src -> dst : flow count)
        for u, v, wt in G.edges(data="weight"):
            edge_rows.append({"Attack": attack, "src": u, "dst": v, "weight": int(wt)})

        # M1 -- unweighted degree = number of distinct partners
        avg_in_deg  = safe_mean([d for _, d in G.in_degree()])
        avg_out_deg = safe_mean([d for _, d in G.out_degree()])
        # strength = weighted degree = flow volume (reported alongside)
        avg_in_str  = safe_mean([d for _, d in G.in_degree(weight="weight")])
        avg_out_str = safe_mean([d for _, d in G.out_degree(weight="weight")])

        # M2 -- Louvain on the weighted undirected projection
        if n_nodes <= 1:
            n_comm = n_nodes
            comm_sizes = [n_nodes] if n_nodes == 1 else []
        else:
            partition = community_louvain.best_partition(
                G.to_undirected(), weight="weight", random_state=LOUVAIN_SEED)
            n_comm = len(set(partition.values()))
            comm_sizes = sorted(pd.Series(partition).value_counts().tolist(),
                                reverse=True)
        for i, sz in enumerate(comm_sizes):
            community_rows.append({"Attack": attack, "community": i, "size": sz})

        # M3 -- PageRank roles
        roles = pagerank_roles(G)
        role_counts = pd.Series(roles).value_counts().to_dict() if roles else {}

        summary_rows.append({
            "Attack": attack,
            "Nodes": n_nodes,
            "Edges": G.number_of_edges(),
            "Avg In-Deg": avg_in_deg,
            "Avg Out-Deg": avg_out_deg,
            "Avg In-Strength": avg_in_str,
            "Avg Out-Strength": avg_out_str,
            "# Comm.": n_comm,
            "# Flows": n_flows,
            "role_low": role_counts.get("low", 0),
            "role_medium": role_counts.get("medium", 0),
            "role_high": role_counts.get("high", 0),
        })

        for node in G.nodes():
            node_fingerprints.append({
                "Attack": attack,
                "IP": node,
                "in_degree":   int(G.in_degree(node)),
                "out_degree":  int(G.out_degree(node)),
                "in_strength":  int(G.in_degree(node, weight="weight")),
                "out_strength": int(G.out_degree(node, weight="weight")),
                "role": roles[node],
            })

        # degree-distribution histograms (in and out)
        if n_nodes > 0:
            fig, ax = plt.subplots(1, 2, figsize=(10, 3))
            ax[0].hist([d for _, d in G.in_degree()], bins=30)
            ax[0].set_title(f"{attack} - in-degree")
            ax[0].set_xlabel("in-degree"); ax[0].set_ylabel("# nodes")
            ax[1].hist([d for _, d in G.out_degree()], bins=30)
            ax[1].set_title(f"{attack} - out-degree")
            ax[1].set_xlabel("out-degree"); ax[1].set_ylabel("# nodes")
            plt.tight_layout()
            plt.savefig(os.path.join(HIST_DIR, f"hist_{attack}.png"), dpi=110)
            plt.close()
        else:
            print(f"FLAG: {attack} produced an empty graph -- no histogram")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "step1_fingerprint_summary.csv"), index=False)

    edges = pd.DataFrame(edge_rows)
    edges.to_csv(os.path.join(OUT_DIR, "step1_edge_list.csv"), index=False)

    communities = pd.DataFrame(community_rows)
    communities.to_csv(os.path.join(OUT_DIR, "step1_community_sizes.csv"), index=False)

    nodes = pd.DataFrame(node_fingerprints)
    nodes.to_csv(os.path.join(OUT_DIR, "step1_node_fingerprints.csv"), index=False)
    nodes.to_json(os.path.join(OUT_DIR, "step1_node_fingerprints.json"),
                  orient="records", indent=2)

    # ---- deliverable table (research guide 2.6) ----
    print("\n" + "=" * 78)
    print("STEP 1 DELIVERABLE TABLE  (guide 2.6)")
    print("=" * 78)
    deliver = summary[["Attack", "Nodes", "Edges", "Avg In-Deg", "Avg Out-Deg",
                       "# Comm.", "# Flows"]]
    with pd.option_context("display.width", 200):
        print(deliver.to_string(index=False))

    print("\n" + "-" * 78)
    print("full fingerprint (incl. strength + PageRank role split):")
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False))

    print("\n" + "-" * 78)
    print("Louvain community sizes per attack:")
    for attack in ATTACKS:
        sizes = communities.loc[communities["Attack"] == attack, "size"].tolist()
        print(f"    {attack:<11} {len(sizes):>2} communities  sizes={sizes}")

    print(f"\nwrote {OUT_DIR}/step1_fingerprint_summary.csv")
    print(f"wrote {OUT_DIR}/step1_edge_list.csv  ({len(edges):,} weighted edges)")
    print(f"wrote {OUT_DIR}/step1_community_sizes.csv")
    print(f"wrote {OUT_DIR}/step1_node_fingerprints.csv / .json")
    print(f"wrote {HIST_DIR}/hist_<attack>.png  ({len(ATTACKS)} files)")


if __name__ == "__main__":
    log_path = os.path.join(OUT_DIR, "report.txt")

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
