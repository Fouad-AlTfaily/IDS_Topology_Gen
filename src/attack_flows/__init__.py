"""Stage 4 -- ATTACK FLOW GENERATION with CTGAN (Step 3).

  3.1   enrich the real flows with node structure + FINAL degree bins  (shared)
  3.2   stratified train/val/test split of the real attack flows
  3.3   condition support on TRAIN (MIN_SUPPORT)
  3.4   balancing weights (MAX_WEIGHT)
  3.5   CTGAN variants A-D -- REUSED; never retrained unless STEP3_ALLOW_TRAIN=1
  3.6   generate flows for every edge of the selected scale's attack graphs
  3.7   assemble + validate the generated attack flows
        + report

3.2-3.5 produce scale-independent assets (results/step3_ctgan/{splits,support,
balancing,models}) and self-skip once complete; they are still run per scale
because they (re)write that scale's split-statistics reports.
"""

STEPS = [
    {"module": "src.attack_flows.step3_1_enrich", "shared": True,
     "done_if_exists": ["results/step3/step3_1_enriched_real_flows.csv"]},
    {"module": "src.attack_flows.step3_1_degree_binning", "shared": True,
     "done_if_exists": ["results/step3/degree_binning_report.txt"]},
    {"module": "src.attack_flows.step3_2_prepare"},
    {"module": "src.attack_flows.step3_3_support"},
    {"module": "src.attack_flows.step3_4_balance"},
    *[{"module": "src.attack_flows.step3_5_train", "args": ["--variant", v]}
      for v in ("A", "B", "C", "D")],
    {"module": "src.attack_flows.step3_6_generate"},
    {"module": "src.attack_flows.step3_7_validate"},
    {"module": "src.attack_flows.step3_report"},
]
