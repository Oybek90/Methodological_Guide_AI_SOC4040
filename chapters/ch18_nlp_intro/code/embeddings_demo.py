# ci: skip
"""GloVe-50 word vectors: similarity, analogies, polysemy, a toy attention layer."""
import numpy as np
import gensim.downloader as api

kv = api.load("glove-wiki-gigaword-50")        # 400k words, 50 dimensions
rng = np.random.default_rng(0)
print("vocab:", len(kv.key_to_index), " dim:", kv.vector_size)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


for a, b in [("king", "queen"), ("king", "man"), ("cat", "dog"), ("cat", "economy"), ("good", "bad")]:
    print(f"cos({a},{b}) = {cos(kv[a], kv[b]):.3f}")
print("neighbours of 'frog':", [w for w, _ in kv.most_similar("frog", topn=5)])

# analogy by hand: nearest neighbour of king - man + woman
q = kv["king"] - kv["man"] + kv["woman"]
sims = kv.cosine_similarities(q, kv.vectors)
best = np.argsort(-sims)[:5]
print("raw nearest to king-man+woman:", [(kv.index_to_key[i], round(float(sims[i]), 3)) for i in best])
print("after excluding the query words:", [w for w, _ in kv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)])
for pos, neg, tgt in [(["paris", "italy"], ["france"], "rome"), (["walking", "swam"], ["swimming"], "walked"),
                      (["doctor", "woman"], ["man"], "nurse")]:
    print(pos, "-", neg, "->", [w for w, _ in kv.most_similar(positive=pos, negative=neg, topn=3)], "(expected", tgt + ")")

# polysemy: ONE vector for 'bank'
print("neighbours of 'bank':", [w for w, _ in kv.most_similar("bank", topn=8)])
print("neighbours of 'bass':", [w for w, _ in kv.most_similar("bass", topn=6)])

# toy contextualisation: x + attention over the OTHER words
# (identity projections, no training)
def attend(words):
    X = np.array([kv[w] for w in words])
    S = X @ X.T / np.sqrt(X.shape[1])
    np.fill_diagonal(S, -np.inf)                     # a word does not attend to itself
    A = np.exp(S - S.max(1, keepdims=True)); A /= A.sum(1, keepdims=True)
    return X + A @ X, A

s1 = ["river", "bank", "water", "flows"]
s2 = ["money", "bank", "loan", "interest"]
C1, A1 = attend(s1); C2, A2 = attend(s2)
print("static   cos(bank,bank)      = 1.000 (same vector)")
print(f"context  cos(bank_1,bank_2)  = {cos(C1[1], C2[1]):.3f}")
for name, C in (("river sentence", C1), ("money sentence", C2)):
    print(f"{name}: cos(bank_ctx,'river') {cos(C[1], kv['river']):.3f}, cos(bank_ctx,'money') {cos(C[1], kv['money']):.3f}")
print("static bank: river", round(cos(kv['bank'], kv['river']), 3), " money", round(cos(kv['bank'], kv['money']), 3))
print("attention of 'bank' (river sentence):", np.round(A1[1], 2))
print("attention of 'bank' (money sentence):", np.round(A2[1], 2))
