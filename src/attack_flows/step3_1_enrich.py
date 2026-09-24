# ============================================================
# STEP 3.1: ENRICH THE REAL FLOW TABLE
#
# For every flow row in the cleaned NF-ToN-IoT table, look up its source and
# destination IP in that attack type's ORIGINAL Step 1 subgraph and append
# four columns:
#
#   src_out_degree  out-degree of the source IP in the original subgraph
#   dst_in_degree   in-degree of the destination IP in the original subgraph
#   src_role        PageRank role of the source IP  (low / medium / high)
#   dst_role        PageRank role of the destination IP
#
# NO community-ID column: Louvain labels are arbitrary and not comparable
# between the real and synthetic graphs, so it would be noise. Degree and
# PageRank role only.
#
# Base table  : data/NF-ToN-IoT_clean.csv          (from data_cleaning.py)
# Lookup      : results/step1/step1_node_fingerprints.csv  (from step1_fingerprint.py)
# Output      : results/step3/step3_1_enriched_real_flows.csv
#               results/step3/step3_1_enrich_report.txt
#
# Ransomware has no Step 1 subgraph (dropped, 142 flows < project minimum).
# Its rows cannot be enriched, so they are dropped from THIS table too --
# consistent with Step 1's 9-class scope. The cleaned CSV still holds them.
# ============================================================

import os
import sys
from contextlib import redirect_stdout

import pandas as pd

CLEAN_PATH = "data/NF-ToN-IoT_clean.csv"
NODES_PATH = "results/step1/step1_node_fingerprints.csv"
OUT_DIR    = "results/step3"
OUT_CSV    = os.path.join(OUT_DIR, "step3_1_enriched_real_flows.csv")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    df = pd.read_csv(CLEAN_PATH, low_memory=False)
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()
    print(f"base flow table : {CLEAN_PATH}  ({len(df):,} rows, {df.shape[1]} cols)")

    nodes = pd.read_csv(NODES_PATH)
    nodes["Attack"] = nodes["Attack"].astype(str).str.strip().str.lower()
    print(f"node fingerprints: {NODES_PATH}  ({len(nodes):,} nodes, "
          f"{nodes['Attack'].nunique()} classes)")

    # keep only classes that have a Step 1 subgraph (drops ransomware)
    modeled = set(nodes["Attack"].unique())
    dropped = df[~df["Attack"].isin(modeled)]
    if len(dropped):
        print(f"dropping {len(dropped):,} rows with no Step 1 subgraph: "
              f"{sorted(dropped['Attack'].unique())}")
        df = df[df["Attack"].isin(modeled)].copy()

    # source lookup: (Attack, IP) -> out_degree, role
    src_tbl = (nodes[["Attack", "IP", "out_degree", "role"]]
               .rename(columns={"IP": "IPV4_SRC_ADDR",
                                "out_degree": "src_out_degree",
                                "role": "src_role"}))
    # destination lookup: (Attack, IP) -> in_degree, role
    dst_tbl = (nodes[["Attack", "IP", "in_degree", "role"]]
               .rename(columns={"IP": "IPV4_DST_ADDR",
                                "in_degree": "dst_in_degree",
                                "role": "dst_role"}))

    n0 = len(df)
    df = df.merge(src_tbl, on=["Attack", "IPV4_SRC_ADDR"], how="left")
    df = df.merge(dst_tbl, on=["Attack", "IPV4_DST_ADDR"], how="left")
    assert len(df) == n0, "row count changed during merge -- duplicate keys in lookup"

    # nullable integer degrees (ransomware rows stay <NA>)
    df["src_out_degree"] = df["src_out_degree"].astype("Int64")
    df["dst_in_degree"]  = df["dst_in_degree"].astype("Int64")

    # ---- diagnostics ----
    print("\n" + "=" * 70)
    print("MATCH RATE per attack class")
    print("=" * 70)
    print(f"{'class':<12}{'rows':>10}{'src matched':>13}{'dst matched':>13}")
    for cls, g in df.groupby("Attack"):
        sm = g["src_role"].notna().sum()
        dm = g["dst_role"].notna().sum()
        flag = "  <- no Step 1 subgraph" if sm == 0 else ""
        print(f"{cls:<12}{len(g):>10,}{sm:>13,}{dm:>13,}{flag}")

    unmatched = df["src_role"].isna() | df["dst_role"].isna()
    print("\n" + "-" * 70)
    print(f"rows fully enriched : {(~unmatched).sum():,}")
    n_bad = int(unmatched.sum())
    print(f"rows with null cols : {n_bad:,}"
          + (f"  (classes: {sorted(df.loc[unmatched, 'Attack'].unique())})" if n_bad else ""))
    print("\nrole distribution (enriched rows):")
    print("  src_role:", df["src_role"].value_counts(dropna=False).to_dict())
    print("  dst_role:", df["dst_role"].value_counts(dropna=False).to_dict())
    print("\ndegree ranges:")
    print(f"  src_out_degree: {df['src_out_degree'].min()} .. {df['src_out_degree'].max()}")
    print(f"  dst_in_degree : {df['dst_in_degree'].min()} .. {df['dst_in_degree'].max()}")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df):,} rows, {df.shape[1]} cols)")
    print("new columns:", [c for c in df.columns
                           if c in ("src_out_degree", "dst_in_degree",
                                    "src_role", "dst_role")])


if __name__ == "__main__":
    log_path = os.path.join(OUT_DIR, "step3_1_enrich_report.txt")

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
