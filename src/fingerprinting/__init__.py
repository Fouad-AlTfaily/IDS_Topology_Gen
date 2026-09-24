"""Stage 2 -- REAL NETWORK FINGERPRINTING (Step 1).

Builds the weighted communication graph of every class from the cleaned real
data and measures its structural fingerprint (degree distribution, Louvain
communities, PageRank roles). Scale-independent: runs once.
"""

STEPS = [
    {"module": "src.fingerprinting.step1_fingerprint", "shared": True,
     "done_if_exists": ["results/step1/step1_node_fingerprints.csv"]},
]
