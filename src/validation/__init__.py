"""Stage 6 -- VALIDATION (Step 5) of the selected scale's final synthetic dataset.

  Level 1  structural   -- degree distributions + Louvain modularity
  Level 2  statistical  -- real TEST vs synthetic, per class / feature
  Level 3  utility      -- E-GraphSAGE trained on real-only / synthetic-only /
                           real+synthetic, all tested on the same real TEST set
  report   -- step5_final_report.txt

Every level self-skips once its DONE marker exists (STEP5_FORCE_RERUN=1 redoes it).
"""

STEPS = [
    {"module": "src.validation.step5_1_structural"},
    {"module": "src.validation.step5_2_statistical"},
    *[{"module": "src.validation.step5_3_utility", "args": ["--condition", c]}
      for c in ("real_only", "synthetic_only", "real_plus_synthetic")],
    {"module": "src.validation.step5_3_utility", "args": ["--aggregate-only"]},
    {"module": "src.validation.step5_report"},
]
