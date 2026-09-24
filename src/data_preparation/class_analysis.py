# ============================================================
# REPORT SECTION 3.4: PER-CLASS ANALYSIS
#
# For every attack class, report:
#   - flow count
#   - percentage of all flows
#   - unique source IPs
#   - unique destination IPs
#   - unique communication edges (distinct src -> dst ordered pairs)
#   - unique undirected pairs (extra, for context)
#
# Classes with fewer than MIN_FLOWS flows are FLAGGED, not deleted.
# MIN_FLOWS = 500 matches step1code.py's MIN_ROWS threshold.
#
# Runs on the CLEANED dataset written by data_cleaning.py.
#
# Output:
#   results/step3_cleaning/class_analysis.csv
#   results/step3_cleaning/class_analysis.txt   (this log)
# ============================================================

import os
import sys
from contextlib import redirect_stdout

import pandas as pd

from src.config import CFG

CLEAN_PATH = "data/NF-ToN-IoT_clean.csv"
RAW_PATH   = "data/NF-ToN-IoT.csv"
OUT_DIR    = "results/cleaning-analysis"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_FLOWS = CFG["data"]["min_class_rows"]   # 500; project minimum; classes below this are flagged, not dropped

# Report order requested by section 3.4.
CLASS_ORDER = ["ddos", "dos", "scanning", "password", "backdoor",
               "injection", "xss", "mitm", "ransomware", "benign"]

CORE_COLUMNS = [
    "IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
    "PROTOCOL", "L7_PROTO", "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Label", "Attack",
]

LABEL_ALIASES = {
    "normal": "benign", "man-in-the-middle": "mitm", "mitm attack": "mitm",
    "cross-site scripting": "xss", "brute force": "password",
    "bruteforce": "password", "sql injection": "injection",
}


def load_clean():
    """Prefer the cleaned CSV; fall back to cleaning the raw file on the fly."""
    if os.path.exists(CLEAN_PATH):
        df = pd.read_csv(CLEAN_PATH, low_memory=False)
        print(f"loaded cleaned dataset : {CLEAN_PATH}")
    else:
        print(f"cleaned file not found -- reading raw {RAW_PATH} instead")
        with open(RAW_PATH) as fh:
            ncols = len(fh.readline().split(","))
        names = CORE_COLUMNS + [f"FEAT_{i}" for i in range(ncols - len(CORE_COLUMNS))]
        df = pd.read_csv(RAW_PATH, header=None, names=names, low_memory=False)

    df["Attack"] = (df["Attack"].astype(str).str.strip().str.lower()
                    .replace(LABEL_ALIASES))
    print(f"rows                   : {len(df):,}")
    return df


def analyse(df):
    total = len(df)
    seen = set(df["Attack"].unique())
    rows = []

    # requested classes first, then any extra class actually present
    ordered = CLASS_ORDER + sorted(seen - set(CLASS_ORDER))

    for cls in ordered:
        sub = df[df["Attack"] == cls]
        n = len(sub)
        edges_dir = sub.groupby(["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).ngroups
        undirected = {
            frozenset((s, d)) for s, d in
            zip(sub["IPV4_SRC_ADDR"], sub["IPV4_DST_ADDR"])
        }
        rows.append({
            "class": cls,
            "flows": n,
            "pct_of_all_flows": round(100 * n / total, 4) if total else 0.0,
            "unique_src_ips": sub["IPV4_SRC_ADDR"].nunique(),
            "unique_dst_ips": sub["IPV4_DST_ADDR"].nunique(),
            "unique_edges_directed": edges_dir,
            "unique_pairs_undirected": len(undirected),
            "below_min": n < MIN_FLOWS,
            "flag": "FLAG < %d" % MIN_FLOWS if n < MIN_FLOWS else "",
        })

    out = pd.DataFrame(rows)
    return out, total


def main():
    df = load_clean()
    table, total = analyse(df)

    print("\n" + "=" * 78)
    print(f"SECTION 3.4  PER-CLASS ANALYSIS   (total flows = {total:,}, "
          f"project minimum = {MIN_FLOWS})")
    print("=" * 78)
    show = table.drop(columns=["below_min"])
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(show.to_string(index=False))

    flagged = table.loc[table["below_min"], "class"].tolist()
    print("\n" + "-" * 78)
    if flagged:
        print(f"FLAGGED (below {MIN_FLOWS} flows, KEPT not deleted): {', '.join(flagged)}")
    else:
        print(f"No class is below the project minimum of {MIN_FLOWS} flows.")
    missing = [c for c in CLASS_ORDER if c not in set(df["Attack"].unique())]
    if missing:
        print(f"NOT PRESENT in the data at all: {', '.join(missing)}")
    print("-" * 78)

    csv_path = os.path.join(OUT_DIR, "class_analysis.csv")
    table.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    log_path = os.path.join(OUT_DIR, "class_analysis.txt")

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
