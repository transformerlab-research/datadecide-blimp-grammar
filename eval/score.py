"""OFF LIMITS — fixed evaluator. Computes BLiMP accuracy and reports the score.

Do not modify to 'fix' a metric. The minimal-pair criterion is the standard one:
a model is correct on an item iff logP(sentence_good) > logP(sentence_bad).
"""
import math, torch

def sent_logprob(model, tok, text, device):
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) < 2:
        return None
    with torch.no_grad():
        lg = model(torch.tensor([ids], device=device)).logits[0]
    lp = torch.log_softmax(lg.float(), dim=-1)
    return sum(lp[i - 1, ids[i]].item() for i in range(1, len(ids)))

def score_paradigm(model, tok, rows, device):
    n = ok = 0
    for r in rows:
        a = sent_logprob(model, tok, r["sentence_good"], device)
        b = sent_logprob(model, tok, r["sentence_bad"], device)
        if a is None or b is None:
            continue
        n += 1
        ok += int(a > b)
    if not n:
        return None, 0, None
    acc = ok / n
    z = (acc - 0.5) / math.sqrt(0.25 / n)      # distance from chance
    return acc, n, z
