# ============================================================
# STEP 4.4 -- TRAIN ONE BENIGN CTGAN VARIANT   (checkpointed, restart-safe)
#
#   python -m src.benign_flows.step4_4_train --variant A|B|C|D
#   # or, from the notebook:  import step4_4_train as T; T.train_one("D")
#
# Trains a single CTGANSynthesizer on that variant's balanced benign TRAIN
# frame.  SAME target features and SAME hyper-parameters (epochs / batch size /
# seed policy) for every variant -- the ONLY intended difference is the
# structural conditioning (this is an ablation study):
#
#   A  (none)
#   B  src_out_degree_cat + dst_in_degree_cat
#   C  src_role + dst_role
#   D  B + C
#
# LIVE PROGRESS: step4_progress.fit_with_live_progress() runs fit() in a worker
# thread and renders a REAL epoch bar from synthesizer._model.loss_values
# (one row per completed epoch). No fake bar; no monkey-patching.
#
# A variant directory results/step4_benign/models/benign_variant_<X>/ is
# COMPLETE only when it contains model.pkl (re-loads cleanly), metadata.json,
# training_config.json and DONE. A complete variant is SKIPPED on re-run unless
# STEP4_FORCE_RERUN=1.
#
# CTGAN never sees IP addresses or the raw numeric degree columns -- only the
# 2-bin degree categories / roles as discrete conditions.
# ============================================================

import argparse
import os
import platform
import time

import pandas as pd

from src.benign_flows import step4_common as S4
from src.benign_flows.step4_progress import describe_capability, fit_with_live_progress

REQUIRED = ["model.pkl", "metadata.json", "training_config.json"]


def variant_dir(v: str) -> str:
    return os.path.join(S4.S4_MODELS, f"benign_variant_{v}")


def _load_ok(api, path: str) -> bool:
    try:
        api.CTGANSynthesizer.load(path)
        return True
    except Exception as exc:  # noqa: BLE001 -- explicit: report why load failed
        print(f"  WARNING: saved model at {path} failed to re-load: {exc!r}")
        return False


def _cuda_report() -> dict:
    env = S4.detect_environment()
    return {
        "torch": env.get("torch"),
        "cuda_available": bool(env.get("gpu_available")),
        "cuda_version": env.get("cuda"),
        "gpu_name": env.get("gpu_name"),
    }


def train_one(v: str) -> None:
    assert v in S4.VARIANTS, f"unknown variant {v!r}"
    api = S4.get_sdv_api()
    vdir = variant_dir(v)
    os.makedirs(vdir, exist_ok=True)

    if S4.component_complete(vdir, REQUIRED) and _load_ok(api, os.path.join(vdir, "model.pkl")):
        print(f"[BENIGN VARIANT {v}] status: COMPLETE  action: SKIPPING "
              f"(reusing pre-trained model)")
        S4.update_status(f"benign_variant_{v}_status", "complete")
        return

    # ---- REUSE-ONLY GUARD ------------------------------------------------
    # A different-scale (SCALE != "same") experiment changes the Step-2 BENIGN
    # graph, not the benign flow model. The benign CTGAN models were copied in
    # and MUST be reused, never retrained. If one is missing we STOP with a
    # clear message rather than silently launching a 300-epoch training run.
    # Set STEP4_ALLOW_TRAIN=1 only to deliberately (re)train.
    if os.environ.get("STEP4_ALLOW_TRAIN", "0").strip().lower() not in ("1", "true", "yes"):
        raise SystemExit(
            f"ABORT: benign CTGAN Variant {v} is missing/incomplete at {vdir}\n"
            f"       (need {REQUIRED} + a re-loadable model.pkl).\n"
            f"       This project REUSES the pre-trained benign CTGAN models -- it\n"
            f"       does not retrain them for the '{S4.SCALE}'-scale graph. Restore the\n"
            f"       trained model directory, or set STEP4_ALLOW_TRAIN=1 to train."
        )

    print(f"[BENIGN VARIANT {v}] status: INCOMPLETE  action: STARTING TRAINING "
          f"(STEP4_ALLOW_TRAIN override in effect)")
    S4.update_status(f"benign_variant_{v}_status", "running")
    S4.set_global_seeds(S4.RANDOM_SEED)

    cond_cols = S4.VARIANT_CONDITIONS[v]
    bal_path = os.path.join(S4.S4_BALANCING, f"balanced_train_variant_{v}.csv")
    if not os.path.exists(bal_path):
        raise SystemExit(f"ABORT: {bal_path} missing -- run step4_3_balance.py first")

    use_cols = S4.TARGET_FEATURES + cond_cols
    df = pd.read_csv(bal_path, usecols=lambda c: c in use_cols, low_memory=False)
    df = df[use_cols].copy()

    # ---- HARD guards -------------------------------------------------------
    assert list(df.columns) == use_cols, f"unexpected columns in {bal_path}: {list(df.columns)}"
    assert not any(c.startswith("IPV4") for c in df.columns), "IP column in CTGAN input -- ABORT"
    assert "src_out_degree" not in df.columns and "dst_in_degree" not in df.columns, \
        "raw numeric degree column leaked into CTGAN conditions -- ABORT"
    assert "Attack" not in df.columns, "Attack must NOT be a benign CTGAN condition -- ABORT"
    for c in cond_cols:
        if c.endswith("_cat"):
            assert set(df[c].astype(str).unique()) <= set(S4.DEGREE_CATEGORIES)
        else:
            assert set(df[c].astype(str).unique()) <= set(S4.ROLE_CATEGORIES)

    categorical = list(dict.fromkeys(S4.CATEGORICAL_TARGETS + cond_cols))
    metadata = api.build_metadata(df, categorical)

    want_cuda = bool(S4.detect_environment().get("gpu_available"))
    synth, kept_kwargs, dropped_kwargs = api.make_synthesizer(metadata, S4.CTGAN_CONFIG, want_cuda)
    if dropped_kwargs:
        print(f"  NOTE: CTGAN kwargs not supported by sdv {api.sdv_version} and dropped: {dropped_kwargs}")

    cap = describe_capability()
    print("\n" + "=" * 60)
    print(f"TRAINING BENIGN VARIANT {v}  --  {S4.VARIANT_TITLE[v]}")
    print("=" * 60)
    print(f"  rows            : {len(df):,}")
    print(f"  epochs          : {S4.CTGAN_CONFIG['epochs']}")
    print(f"  batch size      : {S4.CTGAN_CONFIG['batch_size']}")
    print(f"  random seed     : {S4.RANDOM_SEED}")
    print(f"  target features : {S4.TARGET_FEATURES}")
    print(f"  conditioning    : {cond_cols if cond_cols else '(none)'}")
    print(f"  ctgan kwargs    : {kept_kwargs}")
    print(f"  cuda requested  : {want_cuda}   ({_cuda_report()})")
    print(f"  live progress   : {cap['live_epoch_progress'].splitlines()[0]}")
    print("=" * 60)

    start_wall = time.strftime("%Y-%m-%d %H:%M:%S")
    prog = fit_with_live_progress(
        synth, df, total_epochs=int(S4.CTGAN_CONFIG["epochs"]),
        label=f"Benign Variant {v}")
    end_wall = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- atomic-ish save: model + sidecars, THEN the DONE marker ---------
    model_path = os.path.join(vdir, "model.pkl")
    synth.save(model_path)
    if not _load_ok(api, model_path):
        raise SystemExit(f"ABORT: benign variant {v} model did not re-load after save")

    try:
        md_dict = metadata.to_dict()
    except Exception:
        md_dict = {"note": "metadata.to_dict() unavailable in this sdv version"}
    S4.write_json(md_dict, os.path.join(vdir, "metadata.json"))

    env = S4.detect_environment()
    cfg = {
        "variant": v,
        "title": S4.VARIANT_TITLE[v],
        "target_features": S4.TARGET_FEATURES,
        "conditioning_columns": cond_cols,
        "categorical_columns": categorical,
        "ctgan_kwargs_used": kept_kwargs,
        "ctgan_kwargs_dropped": dropped_kwargs,
        "random_seed": S4.RANDOM_SEED,
        "epochs": S4.CTGAN_CONFIG["epochs"],
        "batch_size": S4.CTGAN_CONFIG["batch_size"],
        "training_rows": int(len(df)),
        "training_seconds": prog["elapsed_seconds"],
        "training_started": start_wall,
        "training_finished": end_wall,
        "final_epoch_recorded": prog["final_epoch"],
        "progress_mechanism": prog["mechanism"],
        "last_generator_loss": prog["last_gen_loss"],
        "last_discriminator_loss": prog["last_disc_loss"],
        "python_version": env.get("python") or platform.python_version(),
        "pandas_version": env.get("pandas"),
        "numpy_version": env.get("numpy"),
        "sdv_version": api.sdv_version,
        "ctgan_version": env.get("ctgan"),
        "torch_version": env.get("torch"),
        "cuda_available": bool(env.get("gpu_available")),
        "cuda_version": env.get("cuda"),
        "gpu_name": env.get("gpu_name"),
        "min_support": S4.MIN_SUPPORT,
        "max_weight": S4.MAX_WEIGHT,
        "balanced_source": bal_path,
    }
    S4.write_json(cfg, os.path.join(vdir, "training_config.json"))

    S4.write_done(vdir, {
        "variant": v,
        "training_seconds": prog["elapsed_seconds"],
        "training_rows": int(len(df)),
        "final_epoch": prog["final_epoch"],
        "progress_mechanism": prog["mechanism"],
        "model_path": model_path,
    })
    S4.update_status(f"benign_variant_{v}_status", "complete")

    print("\n" + "-" * 60)
    print(f"TRAINING COMPLETED  --  Benign Variant {v}")
    print(f"  total training time : {prog['elapsed_seconds']:,.1f} s "
          f"({prog['elapsed_seconds'] / 60:.1f} min)")
    print(f"  final epoch         : {prog['final_epoch']} / {S4.CTGAN_CONFIG['epochs']}")
    print(f"  progress mechanism  : {prog['mechanism']}")
    print(f"  model path          : {model_path}")
    print(f"  variant name        : benign_variant_{v}  ({S4.VARIANT_TITLE[v]})")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=S4.VARIANTS,
                    help="which benign CTGAN variant to train (A/B/C/D)")
    args = ap.parse_args()
    S4.ensure_dirs()
    with S4.stage_banner(f"4.4 TRAIN BENIGN CTGAN VARIANT {args.variant} "
                         f"-- {S4.VARIANT_TITLE[args.variant]}"):
        train_one(args.variant)


if __name__ == "__main__":
    main()
