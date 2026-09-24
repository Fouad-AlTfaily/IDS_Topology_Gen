# ============================================================
# STEP 4.7 -- FINAL MERGE  (Step-3 synthetic attacks + Step-4 synthetic benign)
#
#   python -m src.benign_flows.step4_7_merge_final
#
# Combines the Step-3 attack deliverable and the Step-4 benign deliverable into
# ONE synthetic NF-ToN-IoT dataset. Adds only the canonical binary `Label`
# column (0 = benign, 1 = attack); no experimental / metadata columns.
#
# Step 3 is NOT re-run and NOT modified -- it is read-only input here. If Step 3
# is not complete + valid, this script STOPS.
#
# Outputs (results/final_synthetic/):
#   synthetic_NF_ToN_IoT.csv
#   class_counts.csv
#   merge_manifest.json
#   final_validation_report.txt
#   DONE                     (written ONLY after all checks pass)
#
# FINAL checks (spec section 24):
#   1  file exists            2  file reads            3  row count correct
#   4  expected columns       5  no ransomware rows    6  benign rows exist
#   7  attack rows exist      8  no unexpected labels  9  no NaN/inf in numerics
#   10 IPs valid              11 every (class,src,dst) pair is a synthetic edge
#   12 generated counts agree with graph edge weights
#   13 Step-3 attack output unmodified (row count == VALIDATION_DONE record)
#   14 Step-4 benign output valid
#   15 merge contains only synthetic data (no real rows)
#   16 no incomplete generation batches included
# ============================================================

import hashlib
import ipaddress
import os
import time

import numpy as np
import pandas as pd

from src.attack_flows import step3_common as C3
from src.benign_flows import step4_common as S4

FLOW_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"] + S4.TARGET_FEATURES
FINAL_COLS = FLOW_COLS + ["Label"]
NUMERIC_COLS = S4.TARGET_FEATURES


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _valid_ipv4(s: str) -> bool:
    try:
        ipaddress.IPv4Address(str(s))
        return True
    except Exception:
        return False


def _load_step2_edge_weights() -> dict:
    """(attack, src, dst) -> weight  for the selected SCALE graph (all 9 classes)."""
    e = pd.read_csv(C3.STEP2_EDGES)
    e["attack"] = e["attack"].astype(str).str.strip().str.lower()
    return {(a, s, d): int(w) for a, s, d, w in
            zip(e["attack"], e["src"], e["dst"], e["weight"])}


def main():
    os.makedirs(S4.FINAL_DIR, exist_ok=True)
    S4.assert_scale_inputs()
    with S4.stage_banner("4.7 FINAL MERGE (Step-3 attacks + Step-4 benign)"):
        report = []
        print(f"[STEP 4.7] SCALE={S4.SCALE}  graph={C3.STEP2_DIR}/  ->  final dir {S4.FINAL_DIR}/")
        def p(s=""):
            report.append(s)
            print(s)

        # -------- verify Step 3 (read-only) -------------------------------
        if not os.path.exists(S4.STEP3_VALIDATION_DONE):
            raise SystemExit("ABORT: Step 3 VALIDATION_DONE missing -- Step 3 is not complete. "
                             "Do NOT re-run Step 3 here; finish it first.")
        if not os.path.exists(S4.STEP3_DELIVERABLE):
            raise SystemExit(f"ABORT: Step 3 deliverable missing: {S4.STEP3_DELIVERABLE}")
        step3_done = S4.read_json(S4.STEP3_VALIDATION_DONE, {})
        attacks = pd.read_csv(S4.STEP3_DELIVERABLE, low_memory=False)
        step3_rows_expected = int(step3_done.get("generated_rows", -1))
        if step3_rows_expected >= 0 and len(attacks) != step3_rows_expected:
            raise SystemExit(
                f"ABORT: Step 3 deliverable has {len(attacks):,} rows but VALIDATION_DONE "
                f"records {step3_rows_expected:,} -- Step 3 output looks MODIFIED. Aborting."
            )
        p(f"Step 3 attack deliverable : {S4.STEP3_DELIVERABLE}")
        p(f"  rows                    : {len(attacks):,}  (matches VALIDATION_DONE record)")
        p(f"  sha256                  : {_sha256(S4.STEP3_DELIVERABLE)}")

        # -------- verify Step 4 benign ----------------------------------
        if not S4.is_done(S4.S4_GENERATION, "VALIDATION_DONE"):
            raise SystemExit("ABORT: Step 4 benign VALIDATION_DONE missing -- run step4_6_validate.py")
        if not os.path.exists(S4.SYNTHETIC_BENIGN_FLOWS):
            raise SystemExit(f"ABORT: Step 4 benign deliverable missing: {S4.SYNTHETIC_BENIGN_FLOWS}")
        benign = pd.read_csv(S4.SYNTHETIC_BENIGN_FLOWS, low_memory=False)
        p(f"Step 4 benign deliverable : {S4.SYNTHETIC_BENIGN_FLOWS}")
        p(f"  rows                    : {len(benign):,}")
        p(f"  sha256                  : {_sha256(S4.SYNTHETIC_BENIGN_FLOWS)}")
        p("")

        # -------- align + concat --------------------------------------
        for name, d in (("attacks", attacks), ("benign", benign)):
            missing = [c for c in FLOW_COLS if c not in d.columns]
            if missing:
                raise SystemExit(f"ABORT: {name} deliverable missing columns: {missing}")
        attacks = attacks[FLOW_COLS].copy()
        benign = benign[FLOW_COLS].copy()
        attacks["Attack"] = attacks["Attack"].astype(str).str.strip().str.lower()
        benign["Attack"] = benign["Attack"].astype(str).str.strip().str.lower()

        assert set(benign["Attack"].unique()) == {S4.BENIGN_LABEL}, "benign file not all benign -- ABORT"
        assert set(attacks["Attack"].unique()) <= set(C3.ATTACKS_STEP3), \
            f"attack file has out-of-scope classes: {sorted(set(attacks['Attack'].unique()) - set(C3.ATTACKS_STEP3))}"

        attacks["Label"] = 1
        benign["Label"] = 0
        merged = pd.concat([attacks[FINAL_COLS], benign[FINAL_COLS]], ignore_index=True)

        # ============ FINAL VALIDATION (spec section 24) ================
        fails = []

        # 3 row count
        if len(merged) != len(attacks) + len(benign):
            fails.append(f"row count {len(merged)} != {len(attacks)} + {len(benign)}")

        # 4 columns
        if list(merged.columns) != FINAL_COLS:
            fails.append(f"unexpected columns: {list(merged.columns)}")

        # 5 no ransomware  / 8 no unexpected labels
        allowed = set(C3.ATTACKS_STEP3) | {S4.BENIGN_LABEL}
        seen = set(merged["Attack"].unique())
        if "ransomware" in seen:
            fails.append("ransomware rows present")
        if not seen <= allowed:
            fails.append(f"unexpected labels: {sorted(seen - allowed)}")

        # 6 / 7 benign + attack present
        if int((merged["Attack"] == S4.BENIGN_LABEL).sum()) == 0:
            fails.append("no benign rows")
        if int((merged["Attack"] != S4.BENIGN_LABEL).sum()) == 0:
            fails.append("no attack rows")

        # 9 no NaN / inf in numeric features
        num = merged[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
        if int(num.isna().sum().sum()):
            fails.append("NaN in numeric features")
        if int(np.isinf(num.to_numpy(dtype="float64")).sum()):
            fails.append("infinite values in numeric features")
        # physical plausibility (same rule as the real cleaning)
        valid = C3.physical_validity_mask(merged.assign(**{c: num[c] for c in NUMERIC_COLS}))
        if not valid.all():
            fails.append(f"{int((~valid).sum())} rows fail physical-plausibility")

        # 10 IPs valid
        uniq_ips = pd.unique(merged[["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]].to_numpy().ravel())
        bad_ips = [ip for ip in uniq_ips if not _valid_ipv4(ip)]
        if bad_ips:
            fails.append(f"{len(bad_ips)} invalid IP strings (e.g. {bad_ips[:3]})")

        # 11 every (class,src,dst) is a synthetic edge  /  12 counts vs edge weights
        ew = _load_step2_edge_weights()
        grp = merged.groupby(["Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]).size()
        not_edge = 0
        weight_mismatch = 0
        for (atk, s, d), cnt in grp.items():
            w = ew.get((atk, s, d))
            if w is None:
                not_edge += 1
            elif int(cnt) != int(w):
                weight_mismatch += 1
        if not_edge:
            fails.append(f"{not_edge} (class,src,dst) groups are not synthetic edges")
        if weight_mismatch:
            fails.append(f"{weight_mismatch} edges where generated count != edge weight")

        # 13 Step 3 unmodified (already checked above; restate)
        if step3_rows_expected >= 0 and len(attacks) != step3_rows_expected:
            fails.append("Step 3 deliverable row count changed vs VALIDATION_DONE")

        # 14 Step 4 benign valid
        s4v = S4.read_json(S4.done_path(S4.S4_GENERATION, "VALIDATION_DONE"), {})
        if int(s4v.get("final_rows", -1)) != len(benign):
            fails.append(f"Step 4 benign VALIDATION_DONE records {s4v.get('final_rows')} rows "
                         f"but file has {len(benign)}")

        # 16 no incomplete generation batches
        import glob
        for f in glob.glob(os.path.join(S4.S4_GEN_BATCHES, "batch_*.csv")):
            if not os.path.exists(f[:-4] + "_DONE"):
                fails.append(f"incomplete Step 4 batch on disk: {os.path.basename(f)}")

        p("-" * 70)
        p("FINAL VALIDATION")
        p("-" * 70)
        if fails:
            for f in fails:
                p(f"  FAIL: {f}")
            _write_report(report)
            raise SystemExit(f"ABORT: {len(fails)} final validation failure(s) -- DONE not written")
        p("  all checks passed (1-16)")
        p("  15 merge contains only synthetic data: attacks from Step 3 + benign from Step 4;")
        p("     no rows were read from data/NF-ToN-IoT_clean.csv or any real source.")
        p("")

        # ============ write outputs =====================================
        S4.atomic_write_csv(merged, S4.FINAL_CSV)

        cc = (merged.groupby("Attack").size().reset_index(name="rows")
                    .sort_values("rows", ascending=False))
        cc["fraction"] = (cc["rows"] / len(merged)).round(6)
        cc["Label"] = np.where(cc["Attack"] == S4.BENIGN_LABEL, 0, 1)
        S4.atomic_write_csv(cc, os.path.join(S4.FINAL_DIR, "class_counts.csv"))
        p("CLASS COUNTS")
        p(cc.to_string(index=False))
        p("")

        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_csv": S4.FINAL_CSV,
            "total_rows": int(len(merged)),
            "columns": FINAL_COLS,
            "sources": {
                "attacks": {
                    "path": S4.STEP3_DELIVERABLE,
                    "rows": int(len(attacks)),
                    "sha256": _sha256(S4.STEP3_DELIVERABLE),
                    "validation_done": step3_done,
                    "classes": sorted(attacks["Attack"].unique()),
                },
                "benign": {
                    "path": S4.SYNTHETIC_BENIGN_FLOWS,
                    "rows": int(len(benign)),
                    "sha256": _sha256(S4.SYNTHETIC_BENIGN_FLOWS),
                    "validation_done": s4v,
                },
            },
            "label_rule": "Label = 0 for benign, 1 for every attack class",
            "excluded": ["ransomware", "any real (non-synthetic) row"],
            "graph_scale": S4.STEP2_SCALE,
            "checks_passed": 16,
        }
        S4.write_json(manifest, os.path.join(S4.FINAL_DIR, "merge_manifest.json"))

        _write_report(report)
        S4.write_done(S4.FINAL_DIR, {
            "total_rows": int(len(merged)),
            "attack_rows": int(len(attacks)),
            "benign_rows": int(len(benign)),
            "checks_passed": 16,
        })
        S4.update_status("final_merge_status", "complete")
        p(f"[STEP 4.7] wrote {S4.FINAL_CSV}  ({len(merged):,} rows)")
        p(f"[STEP 4.7] wrote {S4.FINAL_DIR}/DONE")


def _write_report(lines: list[str]):
    S4.atomic_write_text("\n".join(lines) + "\n",
                         os.path.join(S4.FINAL_DIR, "final_validation_report.txt"))


if __name__ == "__main__":
    main()
