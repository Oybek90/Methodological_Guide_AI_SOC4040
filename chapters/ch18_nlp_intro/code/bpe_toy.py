# ci: run
"""A toy byte-pair-encoding (BPE) learner: sub-word tokenisation in 30 lines."""
from collections import Counter


def learn_bpe(words, n_merges):
    """words: dict word -> frequency. Returns the list of merges."""
    corpus = {tuple(w) + ("</w>",): f for w, f in words.items()}
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for sym, f in corpus.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += f
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])   # ties: alphabetical
        merges.append(best)
        new = {}
        for sym, f in corpus.items():
            out, i = [], 0
            while i < len(sym):
                if i + 1 < len(sym) and (sym[i], sym[i + 1]) == best:
                    out.append(sym[i] + sym[i + 1]); i += 2
                else:
                    out.append(sym[i]); i += 1
            new[tuple(out)] = new.get(tuple(out), 0) + f
        corpus = new
    return merges


def apply_bpe(word, merges):
    sym = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(sym) - 1:
            if (sym[i], sym[i + 1]) == (a, b):
                sym[i:i + 2] = [a + b]
            else:
                i += 1
    return sym


if __name__ == "__main__":
    freq = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    merges = learn_bpe(freq, 10)
    for k, m in enumerate(merges, 1):
        print(k, m)
    for w in ["lowest", "newer", "wide", "xyz"]:
        print(w, "->", apply_bpe(w, merges))
    assert apply_bpe("lowest", merges)[:2] == ["low", "est</w>"]
    print("bpe_toy self-test passed")
