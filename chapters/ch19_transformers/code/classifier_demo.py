# ci: skip
"""Transformer text classifier: the notebook's design and three fixes (Reuters, 5 topics)."""
import json
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import reuters
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    nltk.download("reuters", quiet=True)
except Exception:
    pass
torch.set_num_threads(2)
TOPICS = ["acq", "crude", "trade", "money-fx", "interest"]
docs, labels = [], []
per = Counter()
for f in reuters.fileids():
    c = reuters.categories(f)
    if len(c) == 1 and c[0] in TOPICS and per[c[0]] < 300:
        per[c[0]] += 1
        docs.append(reuters.raw(f)); labels.append(TOPICS.index(c[0]))
y = np.array(labels)
print("documents:", len(docs), dict(per))
tok = lambda t: re.findall(r"[a-z]+", t.lower())
MAXLEN, D, H = 128, 64, 4


class Clf(nn.Module):
    def __init__(s, V, pool="first", use_mask=False, ff=None, use_pos=True, layers=1):
        super().__init__()
        s.pool, s.use_mask, s.use_pos = pool, use_mask, use_pos
        s.tok, s.pos = nn.Embedding(V, D, padding_idx=0), nn.Embedding(MAXLEN, D)
        s.enc = nn.ModuleList([nn.TransformerEncoderLayer(
            D, H, ff or D, 0.1, batch_first=True) for _ in range(layers)])
        s.out = nn.Linear(D, len(TOPICS))

    def forward(s, x):
        pad = x == 0
        h = s.tok(x) + (s.pos(torch.arange(x.shape[1]))[None] if s.use_pos else 0)
        for l in s.enc:
            h = l(h, src_key_padding_mask=pad if s.use_mask else None)
        if s.pool == "first":
            z = h[:, 0]
        else:
            m = (~pad).float().unsqueeze(-1)
            z = (h * m).sum(1) / m.sum(1)
        return s.out(z)


def run(seed):
    tr, te = train_test_split(np.arange(len(docs)), test_size=0.2, random_state=seed, stratify=y)
    res = {}
    vec = TfidfVectorizer(stop_words="english", min_df=2)
    Xtr = vec.fit_transform([docs[i] for i in tr]); Xte = vec.transform([docs[i] for i in te])
    res["TF-IDF + logistic regression"] = LogisticRegression(max_iter=1000).fit(Xtr, y[tr]).score(Xte, y[te])
    cnt = Counter(w for i in tr for w in tok(docs[i]))
    vocab = {w: j + 2 for j, (w, c) in enumerate(cnt.most_common(5000)) if c >= 2}   # 0=PAD 1=UNK

    def enc(i):
        ids = [vocab.get(w, 1) for w in tok(docs[i])][:MAXLEN]
        return ids + [0] * (MAXLEN - len(ids))
    Xt = torch.tensor([enc(i) for i in tr]); Xv = torch.tensor([enc(i) for i in te])
    yt, yv = torch.tensor(y[tr]), torch.tensor(y[te])
    cfgs = {
        "notebook design (first token, no pad mask, ff=d)": dict(pool="first", use_mask=False, ff=D),
        "mean pooling + padding mask + ff=4d": dict(pool="mean", use_mask=True, ff=4 * D),
        "same, without positional embedding": dict(pool="mean", use_mask=True, ff=4 * D, use_pos=False),
    }
    for name, kw in cfgs.items():
        torch.manual_seed(seed)
        m = Clf(len(vocab) + 2, **kw)
        opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(15):
            m.train()
            perm = torch.randperm(len(Xt))
            for b in range(0, len(Xt), 32):
                ix = perm[b:b + 32]
                loss = F.cross_entropy(m(Xt[ix]), yt[ix])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            res[name] = (m(Xv).argmax(1) == yv).float().mean().item()
        res["_params"] = sum(p.numel() for p in m.parameters())
    return res


allres = [run(s) for s in range(3)]
out = {}
for k in allres[0]:
    if k.startswith("_"):
        continue
    v = [r[k] for r in allres]
    out[k] = (float(np.mean(v)), float(np.std(v)))
    print(f"{k:52s} {np.mean(v):.4f} +- {np.std(v):.4f}")
json.dump(dict(results=out, n_docs=len(docs), params=allres[0]["_params"]), open("classifier_results.json", "w"), indent=1)
