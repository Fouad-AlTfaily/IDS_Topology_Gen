# ============================================================
# STEP 3.5 -- TRAIN ONE CTGAN VARIANT   (checkpointed, restart-safe)
#
#   python -m src.attack_flows.step3_5_train --variant A|B|C|D
#
# Trains a single CTGANSynthesizer on that variant's balanced TRAIN frame,
# using the SAME target features and SAME hyper-parameters for every variant
# (fair ablation). Only the conditioning columns differ.
#
# A variant directory results/step3_ctgan/models/variant_<X>/ is COMPLETE
# only when it contains:
#   model.pkl              (the saved synthesizer, and it re-loads cleanly)
#   metadata.json          (SDV metadata dict)
#   training_config.json   (epochs, batch_size, seed, columns, sdv version, ...)
#   DONE
# On re-run a complete variant is SKIPPED unless STEP3_FORCE_RERUN=1.
#
# CTGAN never sees IP addresses or the raw numeric degree columns -- only the
# 2-bin degree categories / roles as discrete conditions.
# ============================================================

import argparse
import os
import time

import pandas as pd

from src.attack_flows import step3_common as C

REQUIRED = ["model.pkl", "metadata.json", "training_config.json"]


def variant_dir(v: str) -> str:
    return os.path.join(C.S3_MODELS, f"variant_{v}")


def _load_ok(api, path: str) -> bool:
    try:
        api.CTGANSynthesizer.load(path)
        return True
    except Exception as exc:  # noqa: BLE001  -- explicit: report why load failed
        print(f"  WARNING: saved model at {path} failed to re-load: {exc!r}")
        return False


def train_one(v: str):
    api = C.get_sdv_api()
    vdir = variant_dir(v)
    os.makedirs(vdir, exist_ok=True)

    if C.component_complete(vdir, REQUIRED) and _load_ok(api, os.path.join(vdir, "model.pkl")):
        print(f"[VARIANT {v}] status: COMPLETE  action: SKIPPING (reusing pre-trained model)")
        C.update_status(f"variant_{v}_status", "complete")
        return

    # ---- REUSE-ONLY GUARD ------------------------------------------------
    # This project runs a different-scale (SCALE != "same") experiment. That
    # changes the Step-2 GRAPH, not the flow-generation model: the attack CTGAN
    # models were copied in and MUST be reused, never retrained. If a model is
    # missing we STOP with a clear message instead of silently kicking off a
    # 4x300-epoch training run. Set STEP3_ALLOW_TRAIN=1 only to deliberately
    # (re)train.
    if os.environ.get("STEP3_ALLOW_TRAIN", "0").strip().lower() not in ("1", "true", "yes"):
        raise SystemExit(
            f"ABORT: attack CTGAN Variant {v} is missing/incomplete at {vdir}\n"
            f"       (need {REQUIRED} + a re-loadable model.pkl).\n"
            f"       This project REUSES the pre-trained CTGAN models -- it does not\n"
            f"       retrain them for the '{C.SCALE}'-scale graph. Restore the trained\n"
            f"       model directory, or set STEP3_ALLOW_TRAIN=1 to train on purpose."
        )

    print(f"[VARIANT {v}] status: INCOMPLETE  action: STARTING TRAINING "
          f"(STEP3_ALLOW_TRAIN override in effect)")
    C.update_status(f"variant_{v}_status", "running")
    C.set_global_seeds(C.RANDOM_SEED)

    cond_cols = C.VARIANT_CONDITIONS[v]
    bal_path = os.path.join(C.S3_BALANCING, f"balanced_train_variant_{v}.csv")
    if not os.path.exists(bal_path):
        raise SystemExit(f"ABORT: {bal_path} missing -- run step3_4_balance.py first")

    use_cols = C.TARGET_FEATURES + cond_cols
    df = pd.read_csv(bal_path, usecols=lambda c: c in use_cols, low_memory=False)
    df = df[use_cols].copy()
    df["Attack"] = df["Attack"].astype(str).str.strip().str.lower()

    # hard scope guard -- benign / ransomware must never reach a synthesizer
    assert int((df["Attack"] == "benign").sum()) == 0, "benign in training data -- ABORT"
    assert int((df["Attack"] == "ransomware").sum()) == 0, "ransomware in training data -- ABORT"
    assert not any(col.startswith("IPV4") for col in df.columns), "IP column in CTGAN input -- ABORT"
    assert "src_out_degree" not in df.columns and "dst_in_degree" not in df.columns, \
        "raw numeric degree column leaked into CTGAN conditions -- ABORT"

    categorical = list(dict.fromkeys(C.CATEGORICAL_TARGETS + cond_cols))
    metadata = api.build_metadata(df, categorical)

    want_cuda = bool(C.detect_environment().get("gpu_available"))
    synth, kept_kwargs, dropped_kwargs = api.make_synthesizer(metadata, C.CTGAN_CONFIG, want_cuda)
    if dropped_kwargs:
        print(f"  NOTE: CTGAN kwargs not supported by sdv {api.sdv_version} and dropped: {dropped_kwargs}")

    t0 = time.time()
    synth.fit(df)
    train_seconds = time.time() - t0

    # ---- atomic-ish save: write model + sidecars, THEN the DONE marker ----
    model_path = os.path.join(vdir, "model.pkl")
    synth.save(model_path)
    if not _load_ok(api, model_path):
        raise SystemExit(f"ABORT: variant {v} model did not re-load after save")

    try:
        md_dict = metadata.to_dict()
    except Exception:
        md_dict = {"note": "metadata.to_dict() unavailable in this sdv version"}
    C.write_json(md_dict, os.path.join(vdir, "metadata.json"))

    cfg = {
        "variant": v,
        "title": C.VARIANT_TITLE[v],
        "target_features": C.TARGET_FEATURES,
        "conditioning_columns": cond_cols,
        "categorical_columns": categorical,
        "ctgan_kwargs_used": kept_kwargs,
        "ctgan_kwargs_dropped": dropped_kwargs,
        "random_seed": C.RANDOM_SEED,
        "training_rows": int(len(df)),
        "training_seconds": round(train_seconds, 1),
        "sdv_version": api.sdv_version,
        "gpu": C.detect_environment().get("gpu_name"),
        "min_support": C.MIN_SUPPORT,
        "max_weight": C.MAX_WEIGHT,
        "balanced_source": bal_path,
    }
    C.write_json(cfg, os.path.join(vdir, "training_config.json"))

    C.write_done(vdir, {"variant": v, "training_seconds": round(train_seconds, 1),
                        "training_rows": int(len(df))})
    C.update_status(f"variant_{v}_status", "complete")
    print(f"[VARIANT {v}] status: COMPLETE  ({train_seconds:,.1f}s, {len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=C.VARIANTS,
                    help="which CTGAN variant to train (A/B/C/D)")
    args = ap.parse_args()
    C.ensure_dirs()
    with C.stage_banner(f"3.5 TRAIN CTGAN VARIANT {args.variant} -- {C.VARIANT_TITLE[args.variant]}"):
        train_one(args.variant)


if __name__ == "__main__":
    main()
