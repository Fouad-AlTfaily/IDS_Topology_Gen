"""Stage 3 -- TOPOLOGY GENERATION (Step 2).

Generates one synthetic graph per scale in config.yaml (same x1, larger x2,
smaller x0.5) with a degree-preserving configuration model. All scales are
produced in a single run (each with its own seed), so this stage is shared.
"""

from src.config import SCALES

STEPS = [
    {"module": "src.topology.step2_generate", "shared": True,
     "done_if_exists": [f"results/step2/{s}/{f}" for s in SCALES
                        for f in ("synthetic_edge_list.csv", "synthetic_nodes.csv")]},
]
