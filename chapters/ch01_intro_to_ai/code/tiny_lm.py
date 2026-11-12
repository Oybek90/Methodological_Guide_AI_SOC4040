# ci: run
"""A word-level bigram 'language model': the simplest next-token predictor."""
from collections import Counter, defaultdict

import numpy as np

CORPUS = (
    "the model reads the text . the model predicts the next word . "
    "the next word depends on the words before it . "
    "a large model reads a large text . a large model predicts a plausible word . "
    "the plausible word is not always the true word . "
    "the model learns patterns from the text . the model does not look up facts ."
)


def train_bigram(text):
    tokens = text.split()
    counts = defaultdict(Counter)
    for a, b in zip(tokens[:-1], tokens[1:]):
        counts[a][b] += 1
    return counts


def next_token_probs(counts, word):
    c = counts[word]
    total = sum(c.values())
    return {w: n / total for w, n in c.most_common()}


def generate(counts, start, length, seed=0):
    rng = np.random.default_rng(seed)
    out = [start]
    for _ in range(length):
        probs = next_token_probs(counts, out[-1])
        words, p = zip(*probs.items())
        out.append(str(rng.choice(words, p=p)))
    return " ".join(out)


if __name__ == "__main__":
    lm = train_bigram(CORPUS)
    print("P(next | 'the')   :", {k: round(v, 3) for k, v in next_token_probs(lm, "the").items()})
    print("P(next | 'model') :", {k: round(v, 3) for k, v in next_token_probs(lm, "model").items()})
    for seed in range(3):
        print(f"sample {seed}:", generate(lm, "the", 12, seed))
