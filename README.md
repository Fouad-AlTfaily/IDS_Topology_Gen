# IDS_Topology_Gen

Topology-aware generation of synthetic network-intrusion datasets from **NF-ToN-IoT**.
The pipeline learns the communication-graph structure of every traffic class in the
real data, generates a new synthetic topology at a chosen **network scale**, fills
every synthetic edge with flow records produced by **CTGAN**, and validates the
result on three levels (structural, statistical and E-GraphSAGE detection utility).

Three scales are supported. Each has its own output folder:

| `--scale` | Graph size vs. real data | Outputs |
|-----------|--------------------------|---------|
| `smaller` | x0.5 | `results/small_scale/` |
| `same`    | x1   | `results/` (original layout) |
| `larger`  | x2   | `results/large_scale/` |

---

## Pipeline steps

| Stage (package in `src/`) | Original step | What it does |
|---|---|---|
| `data_preparation` | cleaning | Cleans `data/NF-ToN-IoT.csv` into `data/NF-ToN-IoT_clean.csv` (`data_cleaning.py`) and reports class sizes (`class_analysis.py`). Ransomware (142 flows, below the 500-flow minimum) is dropped. |
| `fingerprinting` | Step 1 | For each of the 9 classes, builds the weighted communication graph (weight = number of flows between an IP pair) and measures its fingerprint: in/out-degree distribution, Louvain communities and PageRank roles (low/medium/high). Output: `results/step1/`. |
| `topology` | Step 2 | Generates a new synthetic graph per class with a degree-preserving configuration model, rescaling node and edge counts by the scale factor (`new_nodes = round(factor * original_nodes)`). IPs are drawn from an invented `10.0.0.0/8` reservoir. All three scales are generated in one run. Output: `results/step2/<scale>/`. |
| `attack_flows` | Step 3 | **3.1** enriches the real flows with node structure and the final 2-bin degree categories. **3.2** stratified 80/10/10 split. **3.3** support of each condition on TRAIN (`MIN_SUPPORT = 10`). **3.4** balancing weights (`MAX_WEIGHT = 5.0`). **3.5** four CTGAN variants A–D (A: attack only; B: + degree categories; C: + PageRank roles; D: B + C), **reused, not retrained**. **3.6** generates flows for every edge of the selected scale's attack graphs, backing off D → B → C → A for unsupported conditions. **3.7** assembles and validates them. |
| `benign_flows` | Step 4 | Same process for benign traffic (4.1 split, 4.2 support, 4.3 balancing, 4.4 benign CTGAN variants A–D, **reused**, 4.5 generation, 4.6 validation). **4.7** merges the attack and benign flows into the final dataset `.../final_synthetic/synthetic_NF_ToN_IoT.csv`. |
| `validation` | Step 5 | **Level 1 (structural):** real vs. synthetic degree distributions and Louvain modularity. **Level 2 (statistical):** real TEST vs. synthetic, per class and feature (KS, chi-square, JSD, BH-FDR). **Level 3 (utility):** an E-GraphSAGE edge classifier trained on real-only, synthetic-only and real+synthetic data, all tested on the same held-out real TEST set, over seeds 42, 1 and 2. **Report:** `step5_final_report.txt`. |

---

## Folder structure

```
IDS_Topology_Gen/
├── run_pipeline.py          # single entry point (--scale smaller|same|larger|all)
├── config.yaml              # ALL settings: scales, seeds, CTGAN, E-GraphSAGE, ...
├── requirements.txt
├── README.md
├── src/
│   ├── config.py            # loads config.yaml, resolves the selected scale
│   ├── data_preparation/    # data_cleaning.py, class_analysis.py
│   ├── fingerprinting/      # step1_fingerprint.py
│   ├── topology/            # step2_generate.py
│   ├── attack_flows/        # step3_common.py + step3_1 ... step3_7, step3_report.py
│   ├── benign_flows/        # step4_common.py + step4_1 ... step4_7, step4_progress.py, step4_report.py
│   └── validation/          # step5_common.py, step5_gnn.py (E-GraphSAGE),
│                            # step5_1_structural.py (L1), step5_2_statistical.py (L2),
│                            # step5_3_utility.py (L3), step5_report.py
├── notebooks/
│   └── run_pipeline.ipynb   # Colab wrapper: only calls run_pipeline.py
├── data/                    # put NF-ToN-IoT.csv here (only needed to rebuild from scratch)
├── results/                 # all outputs (see below)
└── legacy/                  # the three original folders, unchanged (to be removed once confirmed)
    ├── generation_same/
    ├── generation_smaller/
    └── generation_larger/
```

Each stage's `__init__.py` lists its steps in order. `run_pipeline.py` runs each
step as `python -m src.<stage>.<step>` in its own process, the same way the
original notebooks ran the scripts.

### Results layout

```
results/
├── step1/  step2/{same,larger,smaller}/  step3/        # shared: fingerprints, graphs, enriched flows
├── step3_ctgan/{splits,support,balancing,models}/      # shared: attack CTGAN models (reused by every scale)
├── step4_benign/{splits,support,balancing,models}/     # shared: benign CTGAN models (reused by every scale)
├── step3_ctgan/generation/same/, step4_benign/generation/,
│   final_synthetic/, step5_validation/                 # scale "same"
├── small_scale/{step3_ctgan,step4_benign,final_synthetic,step5_validation}/   # scale "smaller"
└── large_scale/{step3_ctgan,step4_benign,final_synthetic,step5_validation}/   # scale "larger"
```

---

## Installation

Python 3.13 is recommended (the results were produced with 3.13 on Google Colab with a Tesla T4 GPU).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`torch` is only needed for Level 3 and `sdv`/`ctgan` only for Steps 3–4.
For GPU training, install the CUDA build of torch that
matches your system (see pytorch.org).

### Dataset

Download the raw NF-ToN-IoT dataset from https://doi.org/10.48610/2fa2ed6 and
save it as `data/NF-ToN-IoT.csv`.

---

## Running the pipeline

Run all commands from the project root.

```bash
# see what would run (nothing is executed)
python run_pipeline.py --scale all --dry-run

# one scale
python run_pipeline.py --scale smaller
python run_pipeline.py --scale same
python run_pipeline.py --scale larger

# all three scales (shared steps run once, then smaller/same/larger in config order)
python run_pipeline.py --scale all
```

Options:

| Option | Effect |
|---|---|
| `--dry-run` | Print the plan only. |
| `--from STAGE` | Re-run the shared steps from `STAGE` onward even if their outputs exist (`data_preparation`, `fingerprinting`, `topology`, `attack_flows`, ...). Per-scale steps keep their own DONE markers (see below). |

### What gets skipped

- **Shared steps** (data preparation, Step 1, Step 2, Step 3.1) are skipped when
  their outputs exist. Data preparation is also skipped when everything that reads
  the cleaned CSV is already built, so `data/` can stay empty unless you rebuild
  from scratch.
- **Per-scale steps** skip completed work using their own `DONE` markers (models,
  generation batches, validation levels). Steps 3.7, 4.6, 4.7 and the reports
  always rebuild their outputs from the existing batch files. This is
  deterministic and gives the same content.

### Trained CTGAN models are reused

Steps 3.5 and 4.4 **never retrain** the CTGAN models by default. The same four
attack and four benign models (`results/step3_ctgan/models`,
`results/step4_benign/models`) are used for every scale, because a different
scale changes the graph, not the flow model. If a model is missing, the pipeline
stops with an error instead of starting a 300-epoch training run. To retrain on
purpose:

```bash
STEP3_ALLOW_TRAIN=1 STEP4_ALLOW_TRAIN=1 python run_pipeline.py --scale same
```

### Advanced environment switches (unchanged from the original scripts)

| Variable | Effect |
|---|---|
| `STEP3_FORCE_RERUN=1`, `STEP4_FORCE_RERUN=1`, `STEP5_FORCE_RERUN=1` | Ignore DONE markers of Steps 3 / 4 / 5. |
| `STEP3_ALLOW_TRAIN=1`, `STEP4_ALLOW_TRAIN=1` | Allow CTGAN (re)training. |
| `STEP5_GNN_SEEDS=42,1,2` | Override the E-GraphSAGE seeds. |
| `STEP5_MAX_TRAIN_EDGES`, `STEP5_GNN_EPOCHS` | Reduce the Level 3 budget if the GPU runs out of memory. **These change the results.** |

---

## Configuration

Every tunable value is in [`config.yaml`](config.yaml):

- **scales:** factor (0.5 / 1 / 2), seed offset and output folder
- **seeds:** random / split / Louvain / topology, all 42
- **balancing:** `MIN_SUPPORT = 10`, `MAX_WEIGHT = 5.0`
- **ctgan:** 300 epochs, batch size 500
- **generation:** batching (this determines the `batch_XXXX` file layout)
- **validation:** Level 2 sample size and alpha; Level 3 E-GraphSAGE hyperparameters
  (hidden 128, 2 layers, dropout 0.2, Adam lr 0.001, weight decay 0.0005, 300 epochs,
  full batch, 800k-edge cap with at least 5k per class, seeds [42, 1, 2])

The values are those used for the thesis results. Changing any of them changes the results.

---

## Legacy folders

`legacy/` holds the three original project folders exactly as they were
(`generation_same`, `generation_smaller`, `generation_larger`), including their
own `results/`. `src/` is based on the `generation_smaller` code, which has the
correct per-scale output mapping. The existing results were **copied** (not
moved) into the root `results/`. Once the refactored pipeline is confirmed,
`legacy/` can be deleted.
