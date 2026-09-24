# ============================================================
# STEP 3 (report sections 3.1 + 3.2): LOAD, INSPECT, CLEAN NF-ToN-IoT
#
# 3.1  Load the raw (headerless) NetFlow export, attach the canonical
#      NF-ToN-IoT v1 column names by position, and confirm every field
#      the pipeline needs is present: src/dst IP, label, byte counts,
#      packet counts, flow duration, protocol, ports, TCP flags.
#
# 3.2  Clean it: drop rows with missing/invalid IPs or missing core
#      fields, normalize the attack labels, force the numeric fields to
#      real numeric dtypes, and drop physically impossible flow records.
#      Every removal is counted per attack class and printed, and a final
#      guard aborts if any attack class present in the raw data has been
#      wiped out -- rare classes (mitm, ransomware) must survive cleaning.
#
# NOT removed: exact duplicate rows. NF-ToN-IoT v1 carries no timestamp or
# flow id, so two genuinely separate short flows with the same endpoints
# and size are byte-identical and cannot be told apart. The repeats are
# real recurring traffic (DNS, TLS handshakes, mDNS/SSDP), not copied
# records -- copies are scattered across the file, never adjacent blocks.
# Step 1 deliberately uses flow multiplicity as edge weight, so dedup here
# would discard signal it wants. (Duplicate count is still reported in 3.1
# for the record.) Label collisions -- the same flow stamped with several
# attack labels -- are a known NF-ToN-IoT labelling artifact; left as-is.
#
# Output:
#   data/NF-ToN-IoT_clean.csv          cleaned rows, WITH a header line
#   results/step3_cleaning/report.txt  full text log of this run
#   results/step3_cleaning/class_counts.csv  raw vs clean rows per class
#
# NOTE: the cleaned file is written WITH a header row. If you want
# step1code.py to consume it, change its read_csv to header=0 (and drop
# the positional `names=` list) -- the column names below already match.
# ============================================================

import os
import sys
import ipaddress
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

RAW_PATH   = "data/NF-ToN-IoT.csv"
CLEAN_PATH = "data/NF-ToN-IoT_clean.csv"
OUT_DIR    = "results/cleaning-analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Canonical NF-ToN-IoT v1 schema (Sarhan et al., "NetFlow Datasets").
# The raw file has NO header; these are the column meanings by position.
# Columns 0..13 are the standard 12 NetFlow features + Label + Attack.
# Columns 14..33 are 20 engineered feature columns carried in from an
# earlier pipeline step; we keep them but do not rely on them here.
# ------------------------------------------------------------
CORE_COLUMNS = [
    "IPV4_SRC_ADDR",               # 0  source IP            (string)
    "L4_SRC_PORT",                 # 1  source port          0..65535
    "IPV4_DST_ADDR",               # 2  destination IP       (string)
    "L4_DST_PORT",                 # 3  destination port     0..65535
    "PROTOCOL",                    # 4  IP protocol number   (6=TCP,17=UDP,1=ICMP,...)
    "L7_PROTO",                    # 5  nDPI L7 protocol id  (float)
    "IN_BYTES",                    # 6  bytes  src -> dst
    "OUT_BYTES",                   # 7  bytes  dst -> src
    "IN_PKTS",                     # 8  packets src -> dst
    "OUT_PKTS",                    # 9  packets dst -> src
    "TCP_FLAGS",                   # 10 cumulative OR of TCP flags
    "FLOW_DURATION_MILLISECONDS",  # 11 flow duration (ms)
    "Label",                       # 12 0 = benign, 1 = attack
    "Attack",                      # 13 attack category string
]
N_EXTRA_EXPECTED = 20

# Which raw column answers each field the assignment asks for.
REQUIRED_FIELDS = {
    "source IP":       ["IPV4_SRC_ADDR"],
    "destination IP":  ["IPV4_DST_ADDR"],
    "label":           ["Label", "Attack"],
    "byte counts":     ["IN_BYTES", "OUT_BYTES"],
    "packet counts":   ["IN_PKTS", "OUT_PKTS"],
    "flow duration":   ["FLOW_DURATION_MILLISECONDS"],
    "protocol":        ["PROTOCOL", "L7_PROTO"],
    "ports":           ["L4_SRC_PORT", "L4_DST_PORT"],
    "TCP flags":       ["TCP_FLAGS"],
}

# Numeric fields that must end up as real numbers.
NUMERIC_COLUMNS = [
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Label",
]
PORT_COLUMNS  = ["L4_SRC_PORT", "L4_DST_PORT"]
COUNT_COLUMNS = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]

# Canonical attack-label spelling. Anything not here is KEPT and flagged,
# never dropped -- normalization must not quietly delete a class.
CANONICAL_ATTACKS = {
    "benign", "backdoor", "ddos", "dos", "injection",
    "mitm", "password", "ransomware", "scanning", "xss",
}
LABEL_ALIASES = {
    "normal": "benign",
    "man-in-the-middle": "mitm",
    "mitm attack": "mitm",
    "cross-site scripting": "xss",
    "brute force": "password",
    "bruteforce": "password",
    "sql injection": "injection",
}


def rule(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def per_class(mask, labels):
    """Human-readable 'why rows were dropped', broken out by attack class."""
    if mask.sum() == 0:
        return "    (none)"
    counts = labels[mask].value_counts().sort_index()
    return "\n".join(f"    {k:<12} {v:>9,}" for k, v in counts.items())


# ============================================================
# 3.1  LOAD AND INSPECT
# ============================================================
def load_and_inspect():
    rule("3.1  LOAD AND INSPECT NF-ToN-IoT")

    with open(RAW_PATH, "r", encoding="utf-8", errors="replace") as fh:
        first_line = fh.readline().rstrip("\n")
    raw_ncols = len(first_line.split(","))
    print(f"raw file            : {RAW_PATH}")
    print(f"columns in row 1    : {raw_ncols}  (expected {len(CORE_COLUMNS) + N_EXTRA_EXPECTED})")
    print(f"header present       : no  (first token is {first_line.split(',')[0]!r} -- an IP, not a name)")

    names = list(CORE_COLUMNS)
    if raw_ncols > len(CORE_COLUMNS):
        names += [f"FEAT_{i}" for i in range(raw_ncols - len(CORE_COLUMNS))]
    elif raw_ncols < len(CORE_COLUMNS):
        sys.exit(f"ABORT: raw file has {raw_ncols} columns, fewer than the "
                 f"{len(CORE_COLUMNS)} core NF-ToN-IoT fields -- schema mismatch.")

    df = pd.read_csv(RAW_PATH, header=None, names=names, low_memory=False)
    print(f"loaded shape        : {df.shape[0]:,} rows x {df.shape[1]} cols")

    rule("column dtypes (core schema)")
    print(df[CORE_COLUMNS].dtypes.to_string())

    rule("first 3 rows (core schema)")
    print(df[CORE_COLUMNS].head(3).to_string())

    rule("required-field -> raw-column map")
    missing = []
    for field, cols in REQUIRED_FIELDS.items():
        present = [c for c in cols if c in df.columns]
        gone    = [c for c in cols if c not in df.columns]
        missing += gone
        status = "OK" if not gone else f"MISSING {gone}"
        print(f"    {field:<16} -> {', '.join(cols):<45} [{status}]")
    if missing:
        sys.exit(f"ABORT: required columns not found: {missing}")
    print("\n    all required fields present.")

    rule("missing values per core column (raw)")
    print(df[CORE_COLUMNS].isna().sum().to_string())

    rule("Label (binary) value counts")
    print(df["Label"].value_counts(dropna=False).to_string())

    rule("Attack (category) value counts -- RAW spelling")
    print(df["Attack"].value_counts(dropna=False).to_string())

    rule("numeric-field ranges (raw, coerced view only)")
    for c in NUMERIC_COLUMNS:
        s = pd.to_numeric(df[c], errors="coerce")
        n_bad = int(s.isna().sum() - df[c].isna().sum())
        print(f"    {c:<28} min={np.nanmin(s):>14.3f}  max={np.nanmax(s):>16.3f}  "
              f"non-numeric={n_bad}  neg={int((s < 0).sum())}  inf={int(np.isinf(s).sum())}")

    rule("exact duplicate rows (raw)")
    print(f"    {int(df.duplicated().sum()):,} of {len(df):,} rows are exact copies of an earlier row")

    return df


# ============================================================
# 3.2  CLEANING
# ============================================================
def clean(df):
    rule("3.2  CLEANING")
    raw_class_counts = (
        df["Attack"].astype(str).str.strip().str.lower().value_counts().sort_index()
    )
    n_start = len(df)
    print(f"starting rows: {n_start:,}")

    # ---- 3.2.a  normalize attack labels -----------------------------------
    rule("3.2.a  normalize attack labels")
    norm = df["Attack"].astype(str).str.strip().str.lower()
    norm = norm.replace(LABEL_ALIASES)
    changed = int((norm != df["Attack"]).sum())
    print(f"    labels rewritten (case/whitespace/alias): {changed:,}")
    unknown = sorted(set(norm.unique()) - CANONICAL_ATTACKS)
    if unknown:
        print(f"    WARNING: non-canonical labels kept (NOT dropped): {unknown}")
    else:
        print("    all labels map to the canonical set.")
    df["Attack"] = norm
    labels = df["Attack"]  # tracks the frame through every filter below

    # Label (binary) vs Attack (category) agreement.
    mismatch = ((labels == "benign") & (df["Label"] != 0)) | \
               ((labels != "benign") & (df["Label"] != 1))
    print(f"    Label/Attack disagreements: {int(mismatch.sum()):,}")

    # ---- 3.2.b  numeric fields -> numeric dtype --------------------------
    rule("3.2.b  coerce numeric fields")
    for c in NUMERIC_COLUMNS:
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        made_na = int(df[c].isna().sum() - before_na)
        print(f"    {c:<28} -> {df[c].dtype}   new non-numeric NaNs: {made_na}")

    # ---- 3.2.c  missing / invalid IPs ---------------------------------
    rule("3.2.c  drop rows with missing or invalid IP addresses")

    def bad_ip_series(col):
        s = df[col].astype(str).str.strip()
        uniq = pd.Series(s.unique())

        def ok(v):
            if v in ("", "nan", "None", "0.0.0.0"):
                return False
            try:
                ipaddress.ip_address(v)
                return True
            except ValueError:
                return False

        good = set(uniq[uniq.map(ok)])
        return ~s.isin(good)

    bad_ip = bad_ip_series("IPV4_SRC_ADDR") | bad_ip_series("IPV4_DST_ADDR")
    print(f"    rows with an unparseable / placeholder IP: {int(bad_ip.sum()):,}")
    print(per_class(bad_ip, labels))
    df, labels = df[~bad_ip].copy(), labels[~bad_ip]

    # ---- 3.2.d  missing core fields ----------------------------------
    rule("3.2.d  drop rows with a missing core field")
    core_na = df[CORE_COLUMNS].isna().any(axis=1)
    print(f"    rows with NaN in a core column: {int(core_na.sum()):,}")
    print(per_class(core_na, labels))
    df, labels = df[~core_na].copy(), labels[~core_na]

    # ---- 3.2.e  physically impossible flow records ------------------
    rule("3.2.e  drop impossible values")
    imp = pd.Series(False, index=df.index)

    # non-finite in a numeric field (catches +/-inf that a string "inf"
    # would slip through pd.to_numeric as, and any leftover NaN there)
    nonfinite = ~np.isfinite(df[NUMERIC_COLUMNS].to_numpy(dtype="float64")).all(axis=1)
    nonfinite = pd.Series(nonfinite, index=df.index)
    print(f"    non-finite numeric cell      : {int(nonfinite.sum()):,}")
    imp |= nonfinite

    port_bad = pd.Series(False, index=df.index)
    for c in PORT_COLUMNS:
        port_bad |= (df[c] < 0) | (df[c] > 65535)
    print(f"    port outside 0..65535        : {int(port_bad.sum()):,}")
    imp |= port_bad

    neg = pd.Series(False, index=df.index)
    for c in COUNT_COLUMNS + ["FLOW_DURATION_MILLISECONDS"]:
        neg |= df[c] < 0
    print(f"    negative bytes/pkts/duration : {int(neg.sum()):,}")
    imp |= neg

    proto_bad = (df["PROTOCOL"] < 0) | (df["PROTOCOL"] > 255)
    print(f"    IP protocol outside 0..255   : {int(proto_bad.sum()):,}")
    imp |= proto_bad

    no_pkts  = (df["IN_PKTS"] + df["OUT_PKTS"]) <= 0
    no_bytes = (df["IN_BYTES"] + df["OUT_BYTES"]) <= 0
    print(f"    flow with 0 total packets    : {int(no_pkts.sum()):,}")
    print(f"    flow with 0 total bytes      : {int(no_bytes.sum()):,}")
    imp |= no_pkts | no_bytes

    # bytes must cover at least 1 byte per observed packet in each direction
    fewer_bytes_than_pkts = (df["IN_BYTES"] < df["IN_PKTS"]) | (df["OUT_BYTES"] < df["OUT_PKTS"])
    print(f"    bytes < packets (impossible) : {int(fewer_bytes_than_pkts.sum()):,}")
    imp |= fewer_bytes_than_pkts

    print(f"    -> impossible rows total     : {int(imp.sum()):,}")
    print(per_class(imp, labels))
    df, labels = df[~imp].copy(), labels[~imp]

    # ---- 3.2.f  guard: no attack class wiped out -------------------
    rule("3.2.f  rare-class guard")
    clean_class_counts = labels.value_counts().sort_index()
    summary = pd.DataFrame({
        "raw_rows":   raw_class_counts,
        "clean_rows": clean_class_counts,
    }).fillna(0).astype("int64")
    summary["removed"] = summary["raw_rows"] - summary["clean_rows"]
    summary["pct_removed"] = (100 * summary["removed"] / summary["raw_rows"]).round(2)
    print(summary.to_string())

    wiped = summary.index[(summary["raw_rows"] > 0) & (summary["clean_rows"] == 0)].tolist()
    if wiped:
        sys.exit(f"ABORT: cleaning removed every row of attack class(es) {wiped}. "
                 f"Refusing to write output -- rare classes must be preserved.")
    print("\n    every attack class present in the raw data survives cleaning.")

    rule("cleaning totals")
    print(f"    rows in  : {n_start:,}")
    print(f"    rows out : {len(df):,}")
    print(f"    removed  : {n_start - len(df):,}  ({100 * (n_start - len(df)) / n_start:.2f}%)")

    return df, summary


def main():
    df = load_and_inspect()
    df_clean, summary = clean(df)

    df_clean.to_csv(CLEAN_PATH, index=False)
    summary.to_csv(os.path.join(OUT_DIR, "class_counts.csv"))
    print(f"\nwrote {CLEAN_PATH}  ({len(df_clean):,} rows, header included)")
    print(f"wrote {os.path.join(OUT_DIR, 'class_counts.csv')}")


if __name__ == "__main__":
    report_path = os.path.join(OUT_DIR, "report.txt")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(report_path, "w", encoding="utf-8") as fh:
        with redirect_stdout(_Tee(sys.stdout, fh)):
            main()
    print(f"wrote {report_path}")
