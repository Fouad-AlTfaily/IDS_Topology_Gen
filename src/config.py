# ============================================================
# CENTRAL CONFIGURATION LOADER
#
# Loads config.yaml (project root) once and exposes it as CFG. The active
# network scale is chosen by run_pipeline.py --scale, which passes it to every
# pipeline step through the PIPELINE_SCALE environment variable.
# ============================================================

from __future__ import annotations

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

SCALES = list(CFG["scales"])          # ["same", "larger", "smaller"] (config order)
SCALE_ENV = "PIPELINE_SCALE"


def active_scale() -> str:
    """The scale selected by run_pipeline.py --scale. Aborts with a clear
    message if a step is started without one."""
    scale = os.environ.get(SCALE_ENV, "").strip().lower()
    if scale not in SCALES:
        raise SystemExit(
            f"ABORT: no valid scale selected ({SCALE_ENV}={scale!r}).\n"
            f"       Run the pipeline through: python run_pipeline.py "
            f"--scale {{{'|'.join(SCALES)}|all}}"
        )
    return scale
