"""Stage 1 -- DATA PREPARATION.

Cleans the raw NF-ToN-IoT CSV (data/NF-ToN-IoT.csv -> data/NF-ToN-IoT_clean.csv)
and writes the per-class size analysis. Scale-independent: runs once.

Skipped when every consumer of the cleaned CSV (Step 1 fingerprints and the
Step 3.1 enriched real flows) already has its output, so the raw dataset is
only needed to rebuild the pipeline from scratch.
"""

_DOWNSTREAM_DONE = [
    "results/step1/step1_node_fingerprints.csv",
    "results/step3/step3_1_enriched_real_flows.csv",
]

STEPS = [
    {"module": "src.data_preparation.data_cleaning", "shared": True,
     "done_if_exists": _DOWNSTREAM_DONE},
    {"module": "src.data_preparation.class_analysis", "shared": True,
     "done_if_exists": _DOWNSTREAM_DONE},
]
