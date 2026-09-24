# ============================================================
# STEP 4 -- LIVE CTGAN TRAINING PROGRESS
#
# The Colab notebook MUST show a REAL, live, epoch-by-epoch progress indicator
# during the (long) benign CTGAN training runs -- not a frozen "Training..."
# line and NOT a faked bar.
#
# HOW IT WORKS (and why it is real):
#   * The installed CTGAN implementation (ctgan >= 0.10, used by
#     sdv.single_table.CTGANSynthesizer) records one row PER COMPLETED EPOCH in
#       synthesizer._model.loss_values     -> DataFrame[Epoch, Generator Loss,
#                                                       Discriminator Loss]
#     This list is appended every epoch regardless of the verbose flag.
#   * We run synthesizer.fit(...) in a worker thread and, from the main thread,
#     poll len(synthesizer._model.loss_values) as the completed-epoch counter,
#     advancing a tqdm bar to match and showing the latest G/D loss, elapsed
#     time and ETA.
#   * We NEVER monkey-patch CTGAN internals. We only READ a public-in-practice
#     attribute. If a future SDV/ctgan version stops exposing it, describe_
#     capability() / the runtime detector report that, and the bar degrades to
#     an honest elapsed-time display (still driven by the real fit thread).
#
# describe_capability()  -- static inspection of the installed API.
# probe_live_epochs()    -- optional 2-epoch throw-away fit on tiny random data
#                           that PROVES the epoch counter advances during fit.
# fit_with_live_progress() -- the function the training script calls.
# ============================================================

from __future__ import annotations

import inspect
import io
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Optional

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# tqdm (ctgan depends on it, so it is always available in the Colab env)
# ------------------------------------------------------------------
def _get_tqdm():
    try:
        from tqdm.auto import tqdm
        return tqdm
    except Exception:  # pragma: no cover - tqdm missing
        try:
            from tqdm import tqdm
            return tqdm
        except Exception:
            return None


# ------------------------------------------------------------------
# format helpers
# ------------------------------------------------------------------
def _fmt_hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _completed_epochs(synth) -> Optional[int]:
    """Real per-epoch counter from the CTGAN loss history, or None if the
    installed version does not expose it yet / not built yet."""
    model = getattr(synth, "_model", None)
    if model is None:
        return None
    lv = getattr(model, "loss_values", None)
    if lv is None:
        return None
    try:
        return int(len(lv))
    except Exception:
        return None


def _last_losses(synth):
    model = getattr(synth, "_model", None)
    lv = getattr(model, "loss_values", None) if model is not None else None
    try:
        if lv is not None and len(lv):
            row = lv.iloc[-1]
            g = float(row.get("Generator Loss", float("nan")))
            d = float(row.get("Discriminator Loss", float("nan")))
            return g, d
    except Exception:
        pass
    return float("nan"), float("nan")


# ------------------------------------------------------------------
# STATIC capability report
# ------------------------------------------------------------------
def describe_capability() -> dict:
    """Inspect the installed SDV / CTGAN API WITHOUT training anything."""
    info: dict = {
        "sdv_version": "not installed",
        "ctgan_version": "not installed",
        "ctgan_synthesizer_init_params": [],
        "has_verbose_param": False,
        "has_cuda_param": False,
        "has_get_loss_values_method": False,
        "loss_history_attribute": "synthesizer._model.loss_values (per-epoch)",
        "live_epoch_progress": "unknown (confirm with probe_live_epochs())",
        "notes": "",
    }
    try:
        import sdv
        info["sdv_version"] = getattr(sdv, "__version__", "unknown")
        from sdv.single_table import CTGANSynthesizer
        params = list(inspect.signature(CTGANSynthesizer.__init__).parameters.keys())
        info["ctgan_synthesizer_init_params"] = params
        info["has_verbose_param"] = "verbose" in params
        info["has_cuda_param"] = any(p in params for p in ("cuda", "use_cuda", "device"))
        info["has_get_loss_values_method"] = hasattr(CTGANSynthesizer, "get_loss_values")
    except Exception as exc:  # pragma: no cover
        info["notes"] = f"SDV import failed: {exc!r}"
        return info
    try:
        import ctgan
        info["ctgan_version"] = getattr(ctgan, "__version__", "unknown")
    except Exception:
        pass

    if info["has_get_loss_values_method"]:
        info["live_epoch_progress"] = (
            "LIKELY AVAILABLE -- CTGANSynthesizer exposes get_loss_values() and "
            "keeps a per-epoch loss history on synthesizer._model.loss_values; "
            "step4_progress polls its length live during fit(). "
            "Run probe_live_epochs() to confirm on this exact install."
        )
    else:
        info["live_epoch_progress"] = (
            "UNCERTAIN -- get_loss_values() not found; step4_progress will still "
            "try synthesizer._model.loss_values and otherwise fall back to a real "
            "elapsed-time display driven by the fit thread."
        )
    return info


def print_capability_report() -> dict:
    info = describe_capability()
    print("=" * 68)
    print("STEP 4 -- CTGAN LIVE-PROGRESS CAPABILITY (static inspection)")
    print("=" * 68)
    for k in ("sdv_version", "ctgan_version", "has_verbose_param", "has_cuda_param",
              "has_get_loss_values_method", "loss_history_attribute"):
        print(f"  {k:<28}: {info[k]}")
    print(f"  live_epoch_progress         : {info['live_epoch_progress']}")
    if info["notes"]:
        print(f"  notes                       : {info['notes']}")
    print("=" * 68)
    return info


# ------------------------------------------------------------------
# DYNAMIC probe -- proves the epoch counter advances during a real fit
# ------------------------------------------------------------------
def probe_live_epochs(n_rows: int = 64, epochs: int = 3, verbose: bool = True) -> dict:
    """Train a throw-away CTGAN for a few epochs on tiny RANDOM data and check
    that _completed_epochs() actually advances while fit() runs. This trains a
    *disposable* model only -- it is NOT part of the Step-4 pipeline."""
    result = {"ok": False, "observed_epoch_updates": [], "message": ""}
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
    except Exception as exc:
        result["message"] = f"SDV not importable: {exc!r}"
        return result

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.integers(0, 100, n_rows),
        "b": rng.integers(0, 100, n_rows),
        "c": rng.choice(["x", "y"], n_rows),
    })
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)

    try:
        synth = CTGANSynthesizer(md, epochs=epochs, batch_size=16, verbose=False)
    except TypeError:
        synth = CTGANSynthesizer(md, epochs=epochs, batch_size=16)

    err: dict = {}

    def _run():
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                synth.fit(df)
        except Exception as e:  # pragma: no cover
            err["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    seen = []
    while th.is_alive():
        ce = _completed_epochs(synth)
        if ce is not None and (not seen or ce != seen[-1]):
            seen.append(ce)
        time.sleep(0.05)
    th.join()
    ce = _completed_epochs(synth)
    if ce is not None and (not seen or ce != seen[-1]):
        seen.append(ce)

    if err:
        result["message"] = f"probe fit failed: {err['e']!r}"
        return result

    result["observed_epoch_updates"] = seen
    advanced = len([s for s in seen if s and s > 0]) > 0
    result["ok"] = bool(advanced)
    if advanced:
        result["message"] = (
            f"LIVE EPOCH PROGRESS CONFIRMED on this install: the per-epoch loss "
            f"history advanced through {seen} during a real fit(). "
            f"step4_progress will show a true epoch bar."
        )
    else:
        result["message"] = (
            "Could not observe the per-epoch counter advancing. step4_progress "
            "will fall back to a real elapsed-time display (still driven by the "
            "fit thread; no fake progress)."
        )
    return result


def print_probe_report() -> dict:
    print("Probing live epoch progress with a 3-epoch throw-away CTGAN fit ...")
    r = probe_live_epochs()
    print(f"  observed epoch updates : {r['observed_epoch_updates']}")
    print(f"  result                 : {r['message']}")
    return r


# ------------------------------------------------------------------
# THE training driver used by step4_4_train.py
# ------------------------------------------------------------------
def fit_with_live_progress(synth, data: pd.DataFrame, total_epochs: int,
                           label: str, poll_seconds: float = 2.0) -> dict:
    """Run synth.fit(data) in a worker thread and render a live epoch bar from
    the synthesizer's REAL per-epoch loss history.

    Returns a dict: elapsed_seconds, final_epoch, mechanism, last_gen_loss,
    last_disc_loss.
    Re-raises any exception from fit() in the caller's thread.
    """
    tqdm = _get_tqdm()
    holder: dict = {}

    def _run():
        try:
            synth.fit(data)
        except BaseException as e:  # noqa: BLE001 - propagate to caller
            holder["exc"] = e

    t0 = time.time()
    th = threading.Thread(target=_run, name=f"ctgan-fit-{label}", daemon=True)
    th.start()

    # give fit() a moment to build _model, then decide which mechanism we have
    time.sleep(min(poll_seconds, 1.0))
    mechanism = "epoch-history"  # optimistic; downgraded below if never seen
    saw_epoch_counter = False

    bar = None
    if tqdm is not None:
        bar = tqdm(total=total_epochs, desc=label, unit="epoch",
                   dynamic_ncols=True, leave=True, file=sys.stdout)

    last_print = 0.0
    while th.is_alive():
        done = _completed_epochs(synth)
        elapsed = time.time() - t0
        if done is not None:
            saw_epoch_counter = True
            done = max(0, min(done, total_epochs))
            g, d = _last_losses(synth)
            if bar is not None:
                if done > bar.n:
                    bar.update(done - bar.n)
                eta = (elapsed / done * (total_epochs - done)) if done else 0.0
                bar.set_postfix_str(
                    f"G={g:.3f} D={d:.3f} elapsed {_fmt_hms(elapsed)} "
                    f"ETA {_fmt_hms(eta)}", refresh=True)
            elif elapsed - last_print >= poll_seconds:
                last_print = elapsed
                pct = 100.0 * done / max(total_epochs, 1)
                eta = (elapsed / done * (total_epochs - done)) if done else 0.0
                filled = int(round(24 * done / max(total_epochs, 1)))
                barstr = "#" * filled + "." * (24 - filled)
                sys.stdout.write(
                    f"\r{label}  {done:>4}/{total_epochs} [{barstr}] {pct:5.1f}%  "
                    f"G={g:.3f} D={d:.3f}  elapsed {_fmt_hms(elapsed)}  "
                    f"ETA {_fmt_hms(eta)}   ")
                sys.stdout.flush()
        else:
            # _model / loss_values not visible yet -> preprocessing or unsupported
            msg = ("preparing data / building model ..."
                   if elapsed < 30 else
                   "epoch counter not exposed by this ctgan build -- elapsed only")
            if bar is not None:
                bar.set_postfix_str(f"{msg}  elapsed {_fmt_hms(elapsed)}", refresh=True)
            elif elapsed - last_print >= poll_seconds:
                last_print = elapsed
                sys.stdout.write(f"\r{label}  {msg}  elapsed {_fmt_hms(elapsed)}   ")
                sys.stdout.flush()
        time.sleep(poll_seconds)

    th.join()
    elapsed = time.time() - t0
    final_epoch = _completed_epochs(synth)
    g, d = _last_losses(synth)

    if bar is not None:
        if final_epoch:
            bar.update(max(0, min(final_epoch, total_epochs) - bar.n))
        bar.set_postfix_str(f"G={g:.3f} D={d:.3f} elapsed {_fmt_hms(elapsed)}",
                            refresh=True)
        bar.close()
    else:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if "exc" in holder:
        raise holder["exc"]

    if not saw_epoch_counter:
        mechanism = "elapsed-only (per-epoch loss history not exposed by this ctgan build)"

    return {
        "elapsed_seconds": round(elapsed, 1),
        "final_epoch": int(final_epoch) if final_epoch is not None else None,
        "mechanism": mechanism,
        "last_gen_loss": None if np.isnan(g) else round(g, 5),
        "last_disc_loss": None if np.isnan(d) else round(d, 5),
    }


if __name__ == "__main__":
    print_capability_report()
    print()
    print_probe_report()
