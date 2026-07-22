"""Parse mirrored sweep stdout into per-cell records.

`lab job logs` hard-wraps long lines, so a CELL record spans several physical
lines. Join any line that is not the start of a known record type back onto the
previous one. (An earlier regex listed bare paradigm stems -- `determiner_noun`
rather than `determiner_noun_agreement_1` -- and silently dropped 5 of 6
paradigms per cell.)
"""
import re, sys, json, statistics, collections

STARTS = ("CELL ", "SKIP ", "--- ", "device=", "loaded ", "  load retry",
          "  refs retry", "ARTIFACT")

def records(path):
    lines = []
    for ln in open(path).read().split("\n"):
        if ln.startswith(STARTS) or not lines:
            lines.append(ln.rstrip())
        else:
            lines[-1] = lines[-1] + " " + ln.strip()
    for ln in lines:
        if not ln.startswith("CELL "):
            continue
        parts = ln.split()
        vals = dict(re.findall(r"([a-z_0-9]+)=([\d.]+)", ln))
        yield {"recipe": parts[1], "size": parts[2], "seed": parts[3],
               "macro": float(vals.pop("macro")),
               "paradigms": {k: float(v) for k, v in vals.items()}}

if __name__ == "__main__":
    recs = list(records(sys.argv[1]))
    by = collections.defaultdict(list)
    for r in recs:
        by[r["recipe"]].append(r)
    print(f"{len(recs)} cells, {len(by)} recipes, "
          f"{sum(1 for v in by.values() if len(v) == 3)} complete")
    if recs:
        print("paradigms:", sorted(recs[0]["paradigms"]))
