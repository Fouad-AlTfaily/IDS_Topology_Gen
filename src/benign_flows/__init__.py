"""Stage 5 -- BENIGN FLOW GENERATION with CTGAN (Step 4) + final merge.

  4.1   benign train/val/test split
  4.2   condition support on TRAIN
  4.3   balancing weights
  4.4   benign CTGAN variants A-D -- REUSED; never retrained unless STEP4_ALLOW_TRAIN=1
  4.5   generate benign flows for every edge of the selected scale's benign graph
  4.6   assemble + validate the generated benign flows
        + report
  4.7   merge Step-3 attacks + Step-4 benign into the final synthetic dataset
"""

STEPS = [
    {"module": "src.benign_flows.step4_1_prepare"},
    {"module": "src.benign_flows.step4_2_support"},
    {"module": "src.benign_flows.step4_3_balance"},
    *[{"module": "src.benign_flows.step4_4_train", "args": ["--variant", v]}
      for v in ("A", "B", "C", "D")],
    {"module": "src.benign_flows.step4_5_generate"},
    {"module": "src.benign_flows.step4_6_validate"},
    {"module": "src.benign_flows.step4_report"},
    {"module": "src.benign_flows.step4_7_merge_final"},
]
