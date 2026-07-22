# Does the Data-Selection Proxy Know Grammar?

Evaluation harness and analysis scripts for the study of whether ranking DataDecide's
pretraining data recipes by **grammatical competence** (BLiMP minimal pairs) reproduces the
ranking obtained from **downstream QA**.

**Headline result.** At 90M parameters, BLiMP grammar and DataDecide's published `olmes_10`
QA aggregate rank the 25 recipes in *opposite* directions: Spearman **rho = -0.60**
(p = 0.0015, 95% CI [-0.85, -0.20], n = 25). The cost is driven by classifier-based quality
filtering. The effect is scale-dependent (not significant at 20M).

Everything here is **inference-only over public checkpoints** (DataDecide models + eval
bundle; BLiMP). No new data or checkpoints are released.

## Layout
```
eval/            BLiMP evaluation harness for the DataDecide grid (90M)
  main.py        loads each (recipe, size, seed) checkpoint, scores 6 BLiMP paradigms
  score.py       minimal-pair scoring (fixed evaluator; do not modify)
  solve.py       the 25 recipes, sizes, seeds, paradigms
  task.yaml      compute spec + pinned environment (see notes below)
  20m/           same harness parameterized for the 20M size replication
analysis/
  qa_join.py     reproduces the headline Spearman rho at 90M and 20M
  make_figures.py regenerates the three paper figures
  parse_sweep.py parses raw sweep stdout logs into per-cell records
data/
  blimp-90m-v2-full.csv   per-recipe BLiMP macros + per-paradigm means (90M)
  blimp-20m-full.csv      same at 20M
  blimp-20m-results.json  raw per-cell per-paradigm acc/z/n (20M)
  blimp-macro-90m-recovered.csv  90M macros recovered from run logs
```

## Reproduce the analysis (no GPU needed)
```bash
pip install pandas numpy scipy pyarrow matplotlib requests
python analysis/qa_join.py        # prints rho at 90M and 20M; downloads the DataDecide
                                  # eval parquet (~140 MB) from HuggingFace once
python analysis/make_figures.py   # writes fig1/fig2/fig3 PNGs
```

## Reproduce the evaluation (GPU)
The harness runs on the Transformer Lab `lab` platform (`task.yaml`), but `main.py` is a
plain script: for each recipe/size/seed it loads the DataDecide checkpoint and scores the
six BLiMP paradigms. Environment is pinned and non-obvious:

- DataDecide checkpoints are `model_type: hf_olmo` (original `ai2-olmo` arch), not
  transformers-native `olmo`; the `ai2-olmo` package must be installed to register it.
- Working version window: `transformers==4.49.0`, `tokenizers>=0.20`; `import hf_olmo`
  **before** `AutoModelForCausalLM`. (4.44.2 -> tokenizer fails; 5.x -> API mismatch.)
- Seeds/steps are **git branches** on each model repo (`step<N>-seed-<name>`), not separate
  repo ids.

## Data sources
- Models: `allenai/DataDecide-<recipe>-<size>` (HuggingFace).
- QA: `allenai/DataDecide-eval-results`, `olmes_10_macro_avg` primary metric at final step.
- Grammar: `nyu-mll/blimp`, 6 paradigms x 1000 items.
