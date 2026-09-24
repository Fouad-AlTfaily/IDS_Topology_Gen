"""Single entry point of the IDS_Topology_Gen pipeline.

    python run_pipeline.py --scale smaller          # one scale
    python run_pipeline.py --scale all              # smaller, same and larger
    python run_pipeline.py --scale all --dry-run    # show the plan only

Stages, in order (one package per stage under src/):
    data_preparation -> fingerprinting -> topology -> attack_flows
    -> benign_flows -> validation

Shared (scale-independent) steps run once and are skipped when their outputs
already exist; --from STAGE re-runs the shared steps from that stage onward.
Scale-dependent steps run once per selected scale, each in its own Python
process (exactly as the original notebooks ran the scripts), and skip work
that is already complete via their own DONE markers. The trained CTGAN models
are reused; they are only retrained when STEP3_ALLOW_TRAIN=1 / STEP4_ALLOW_TRAIN=1
is set explicitly.

Outputs per scale (config.yaml -> scales.<scale>.out_prefix):
    same    -> results/            (step3_ctgan/generation/same, step4_benign/generation,
                                    final_synthetic, step5_validation)
    smaller -> results/small_scale/
    larger  -> results/large_scale/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from src import attack_flows, benign_flows, data_preparation, fingerprinting, topology, validation
from src.config import CFG, ROOT, SCALE_ENV, SCALES

STAGES = {
    "data_preparation": data_preparation,
    "fingerprinting": fingerprinting,
    "topology": topology,
    "attack_flows": attack_flows,
    "benign_flows": benign_flows,
    "validation": validation,
}


def _banner(text: str) -> None:
    print("\n" + "=" * 72 + f"\n{text}\n" + "=" * 72, flush=True)


def _describe(step: dict) -> str:
    return " ".join(["python -m", step["module"], *step.get("args", [])])


def build_plan(from_stage: str | None):
    """-> (shared_steps, per_scale_steps), each a list of (stage, step, action)."""
    stage_names = list(STAGES)
    forced = set(stage_names[stage_names.index(from_stage):]) if from_stage else set()

    shared, per_scale = [], []
    for stage, pkg in STAGES.items():
        for step in pkg.STEPS:
            if step.get("shared"):
                done = all(os.path.exists(os.path.join(ROOT, p))
                           for p in step.get("done_if_exists", []))
                action = "run" if (stage in forced or not done) else "skip"
                shared.append((stage, step, action))
            else:
                per_scale.append((stage, step, "run"))
    return shared, per_scale


def run_step(step: dict, scale: str | None) -> None:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONHASHSEED"] = str(CFG["seeds"]["random"])     # as set by the original notebooks
    if scale is not None:
        env[SCALE_ENV] = scale
    cmd = [sys.executable, "-m", step["module"], *step.get("args", [])]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"\nPIPELINE STOPPED: '{_describe(step)}' failed (exit {result.returncode})"
            + (f" for scale '{scale}'" if scale else "") + ".")
    print(f"[done in {time.time() - t0:,.1f}s] {_describe(step)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scale", required=True, choices=SCALES + ["all"],
                    help="network scale to generate and validate")
    ap.add_argument("--from", dest="from_stage", choices=list(STAGES),
                    help="re-run the shared steps from this stage onward even if "
                         "their outputs already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan without running anything")
    args = ap.parse_args()

    scales = SCALES if args.scale == "all" else [args.scale]
    shared, per_scale = build_plan(args.from_stage)

    _banner(f"PIPELINE PLAN  scale(s): {', '.join(scales)}   config: config.yaml")
    print("shared steps (run once):")
    for stage, step, action in shared:
        print(f"  [{action.upper():4}] {stage:<17} {_describe(step)}")
    print("per-scale steps (self-skip completed work):")
    for stage, step, _ in per_scale:
        print(f"         {stage:<17} {_describe(step)}")
    if args.dry_run:
        return

    for stage, step, action in shared:
        if action == "skip":
            print(f"[SKIP] {_describe(step)}  (outputs present; use --from {stage} to re-run)")
            continue
        _banner(f"[{stage}] {_describe(step)}")
        run_step(step, None)

    for scale in scales:
        for stage, step, _ in per_scale:
            _banner(f"[{scale}] [{stage}] {_describe(step)}")
            run_step(step, scale)

    _banner(f"PIPELINE COMPLETE  scale(s): {', '.join(scales)}")


if __name__ == "__main__":
    main()
