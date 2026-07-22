import warnings; warnings.filterwarnings("ignore")
import os, json, time, statistics, urllib.request
import torch, hf_olmo                                   # hf_olmo registers the arch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from score import score_paradigm
import solve
from lab import lab

DEV = "cuda" if torch.cuda.is_available() else "cpu"
lab.init()          # REQUIRED before lab.finish(); omitting it fails the job AFTER all compute
print(f"device={DEV} torch={torch.__version__}", flush=True)

# ---------------------------------------------------------------------------
# Results are saved as a lab ARTIFACT after every recipe.
# Job 53803a02 computed all 75 cells correctly and wrote them to stdout only;
# when the Lambda instance terminated, the per-cell data was unrecoverable.
# See SWEEP-90M-DATA-LOSS.md. Never let results live only in stdout.
# ---------------------------------------------------------------------------
OUT = "blimp_sweep_results.json"
results = {}

def persist():
    json.dump(results, open(OUT, "w"), indent=1)
    try:
        lab.save_artifact(OUT)
    except Exception as e:
        print(f"ARTIFACT SAVE FAILED {type(e).__name__}: {e}", flush=True)

def final_revision(repo, seed):
    """Seeds/checkpoints are git BRANCHES on the model repo, not separate repo ids."""
    url = f"https://huggingface.co/api/models/{repo}/refs"
    refs = json.load(urllib.request.urlopen(url, timeout=60))
    names = [b["name"] for b in refs.get("branches", [])]
    cands = [n for n in names if n.endswith(f"-seed-{seed}")]
    if not cands:
        return "main" if seed == "default" else None
    return max(cands, key=lambda n: int(n.split("step")[1].split("-")[0]))

data = {p: list(load_dataset("nyu-mll/blimp", p, split="train"))[:solve.N_ITEMS]
        for p in solve.PARADIGMS}
print(f"loaded {len(data)} paradigms x {solve.N_ITEMS} items", flush=True)

for recipe in solve.RECIPES:
    repo = f"allenai/DataDecide-{recipe}-{solve.SIZE}"
    for seed in solve.SEEDS:
        key = f"{recipe}|{solve.SIZE}|{seed}"
        rev = None
        for attempt in range(3):                       # transient HF failures are retriable
            try:
                rev = final_revision(repo, seed)
                break
            except Exception as e:
                print(f"  refs retry {attempt+1} {repo}: {e}", flush=True)
                time.sleep(10)
        if rev is None:
            print(f"SKIP {key}: no revision", flush=True); continue
        m = tok = None
        for attempt in range(3):
            try:
                tok = AutoTokenizer.from_pretrained(repo, revision=rev)
                m = AutoModelForCausalLM.from_pretrained(repo, revision=rev).to(DEV).eval()
                break
            except Exception as e:
                print(f"  load retry {attempt+1} {key}: {type(e).__name__}: {e}", flush=True)
                time.sleep(15)
        if m is None:
            print(f"SKIP {key}: load failed", flush=True); continue

        per = {}
        for p, rows in data.items():
            acc, n, z = score_paradigm(m, tok, rows, DEV)
            per[p] = {"acc": acc, "n": n, "z": z}
        accs = [v["acc"] for v in per.values() if v["acc"] is not None]
        macro = statistics.mean(accs)
        results[key] = {"recipe": recipe, "size": solve.SIZE, "seed": seed,
                        "revision": rev, "macro": macro, "paradigms": per}
        # single-line output; the wrapped multi-line format broke log regexes before
        flat = " ".join(f"{p}={per[p]['acc']:.4f}" for p in solve.PARADIGMS)
        print(f"CELL {recipe} {solve.SIZE} {seed} macro={macro:.4f} {flat}", flush=True)
        del m
        persist()
    print(f"--- {recipe} done ({len([k for k in results if k.startswith(recipe+'|')])}/3 seeds) ---", flush=True)

persist()
ranges = {}
for r in solve.RECIPES:
    ms = [v["macro"] for k, v in results.items() if v["recipe"] == r]
    if len(ms) > 1: ranges[r] = (max(ms) - min(ms)) * 100
lab.finish(
    message=f"BLiMP sweep — {len(results)} cells, {len(ranges)} recipes with >1 seed",
    score={
        "cells_evaluated": float(len(results)),
        "recipes_complete": float(sum(1 for r in solve.RECIPES
                                      if len([k for k in results if results[k]['recipe'] == r]) == 3)),
        "mean_seed_range_pp": float(statistics.mean(ranges.values())) if ranges else 0.0,
        "max_seed_range_pp": float(max(ranges.values())) if ranges else 0.0,
        "n_items": float(solve.N_ITEMS),
    },
)
