# ci: skip
"""A character-level GPT trained from scratch on one novel (CPU, a few minutes)."""
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import gutenberg

try:
    nltk.download("gutenberg", quiet=True)
except Exception:
    pass
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(2)

text = gutenberg.raw("austen-emma.txt").lower()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
n_val = len(data) // 10
train, val = data[:-n_val], data[-n_val:]
V = len(chars)
print("characters:", len(text), " vocabulary:", V, " train/val:", len(train), len(val))

CFG = dict(d=96, layers=3, heads=4, ctx=128, batch=32, steps=1500, lr=3e-3)


class Block(nn.Module):
    def __init__(s, d, h, ctx):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.h = h
        s.register_buffer("mask", torch.tril(torch.ones(ctx, ctx)).bool())

    def forward(s, x, return_w=False):
        B, T, D = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(D, dim=2)
        q, k, v = [t.view(B, T, s.h, D // s.h).transpose(1, 2) for t in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)
        att = att.masked_fill(~s.mask[:T, :T], float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + s.proj(y)
        x = x + s.ff(s.ln2(x))
        return (x, att) if return_w else x


class GPT(nn.Module):
    def __init__(s, V, d, layers, heads, ctx):
        super().__init__()
        s.ctx = ctx
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList([Block(d, heads, ctx) for _ in range(layers)])
        s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False)
        s.head.weight = s.tok.weight
        nn.init.normal_(s.tok.weight, std=0.02)       # GPT-2 style; default N(0,1) is too large
        nn.init.normal_(s.pos.weight, std=0.02)

    def forward(s, idx, return_w=False):
        T = idx.shape[1]
        x = s.tok(idx) + s.pos(torch.arange(T))
        ws = []
        for b in s.blocks:
            if return_w:
                x, w = b(x, True); ws.append(w)
            else:
                x = b(x)
        out = s.head(s.ln(x))
        return (out, ws) if return_w else out


def batch(src, B, T):
    ix = torch.randint(len(src) - T - 1, (B,))
    x = torch.stack([src[i:i + T] for i in ix])
    y = torch.stack([src[i + 1:i + T + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, src, n=40):
    model.eval()
    g = torch.Generator().manual_seed(1)
    L = []
    for _ in range(n):
        ix = torch.randint(len(src) - CFG["ctx"] - 1, (32,), generator=g)
        x = torch.stack([src[i:i + CFG["ctx"]] for i in ix])
        y = torch.stack([src[i + 1:i + CFG["ctx"] + 1] for i in ix])
        L.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(L))


# ---- baselines: uniform, unigram, bigram, trigram (add-one smoothing) on the same val text
def ngram_loss(order):
    tr, va = train.numpy(), val.numpy()
    if order == 0:
        return math.log(V)
    cnt = {}
    for i in range(len(tr) - order + 1):
        key = tuple(tr[i:i + order - 1])
        cnt.setdefault(key, np.zeros(V))[tr[i + order - 1]] += 1
    tot, n = 0.0, 0
    for i in range(len(va) - order + 1):
        key = tuple(va[i:i + order - 1])
        c = cnt.get(key, np.zeros(V))
        tot += -math.log((c[va[i + order - 1]] + 1) / (c.sum() + V)); n += 1
    return tot / n


base = {f"{k}-gram": ngram_loss(k) for k in (0, 1, 2, 3, 4)}
base["uniform"] = base.pop("0-gram")
print("baseline val loss (nats/char):", {k: round(v, 3) for k, v in base.items()})

model = GPT(V, CFG["d"], CFG["layers"], CFG["heads"], CFG["ctx"])
npar = sum(p.numel() for p in model.parameters())
print("parameters:", npar)
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=CFG["steps"], pct_start=0.05)
log, t0 = [], time.time()
for step in range(CFG["steps"] + 1):
    if step % 100 == 0:
        tl, vl = evaluate(model, train), evaluate(model, val)
        log.append((step, tl, vl))
        print(f"step {step:5d}  train {tl:.3f}  val {vl:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if step == CFG["steps"]:
        break
    x, y = batch(train, CFG["batch"], CFG["ctx"])
    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()


@torch.no_grad()
def generate(prompt, n=200, temp=1.0, top_k=None, greedy=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([[stoi[c] for c in prompt]])
    model.eval()
    for _ in range(n):
        lg = model(idx[:, -CFG["ctx"]:])[0, -1]
        if greedy:
            nxt = lg.argmax().view(1, 1)
        else:
            lg = lg / temp
            if top_k:
                lg[lg < lg.topk(top_k).values[-1]] = float("-inf")
            nxt = torch.multinomial(lg.softmax(-1), 1, generator=g).view(1, 1)
        idx = torch.cat([idx, nxt], 1)
    return "".join(itos[i] for i in idx[0].tolist())


samples = {}
for name, kw in {"greedy": dict(greedy=True), "T=0.5": dict(temp=0.5), "T=1.0": dict(temp=1.0),
                 "T=1.5": dict(temp=1.5), "top-k=5": dict(top_k=5)}.items():
    samples[name] = generate("emma was ", 160, **kw)
    print(f"--- {name}\n{samples[name]}")

# attention of the last layer on a short sentence
s = "the young lady was very happy"
with torch.no_grad():
    model.eval()
    _, ws = model(torch.tensor([[stoi[c] for c in s]]), return_w=True)
att = ws[-1][0].numpy()                                 # (heads, T, T)
np.savez("mini_gpt_attn.npz", att=att, text=s)
json.dump(dict(cfg=CFG, params=npar, log=log, baselines=base, samples=samples,
               final_val=log[-1][2], final_train=log[-1][1]), open("mini_gpt_results.json", "w"), indent=1)
torch.save(model.state_dict(), "mini_gpt.pt")
