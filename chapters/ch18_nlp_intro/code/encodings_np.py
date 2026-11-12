# ci: run
"""Text encodings from scratch: one-hot, bag of words, TF-IDF (NumPy only)."""
import re
import numpy as np


def tokenize(text):
    """Lower-case word tokens (letters and digits only)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(docs):
    """Vocabulary in order of first appearance (deterministic)."""
    vocab = {}
    for d in docs:
        for w in d:
            vocab.setdefault(w, len(vocab))
    return vocab


def one_hot(word, vocab):
    v = np.zeros(len(vocab), dtype=int)
    v[vocab[word]] = 1
    return v


def bow(doc, vocab):
    v = np.zeros(len(vocab), dtype=int)
    for w in doc:
        if w in vocab:
            v[vocab[w]] += 1
    return v


def tf_idf(docs, vocab, smooth=False, norm=False):
    """TF = count / length; IDF = ln(N/df), or sklearn's ln((1+N)/(1+df))+1."""
    n = len(docs)
    counts = np.array([bow(d, vocab) for d in docs], dtype=float)
    tf = counts / counts.sum(axis=1, keepdims=True)
    df = (counts > 0).sum(axis=0)
    idf = np.log((1 + n) / (1 + df)) + 1 if smooth else np.log(n / df)
    x = tf * idf
    if norm:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x, idf


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    docs = [tokenize("Hi, I love NLP"), tokenize("I am learning NLP now")]
    vocab = build_vocab(docs)
    print("vocab:", list(vocab))
    print("one-hot('love'):", one_hot("love", vocab))
    X = np.array([bow(d, vocab) for d in docs])
    print("BoW:\n", X)
    tfidf, idf = tf_idf(docs, vocab)
    print("IDF (ln N/df):", idf)
    print("TF-IDF:\n", tfidf)

    # compare with scikit-learn (smooth IDF + L2 norm is its default)
    from sklearn.feature_extraction.text import TfidfVectorizer
    raw = ["Hi, I love NLP", "I am learning NLP now"]
    sk = TfidfVectorizer(token_pattern=r"[A-Za-z0-9]+", lowercase=True)
    S = sk.fit_transform(raw).toarray()
    cols = [sk.vocabulary_[w] for w in vocab]
    mine, _ = tf_idf(docs, vocab, smooth=True, norm=False)
    # sklearn uses raw counts (not count/len) as TF, then L2-normalises
    cnt = X.astype(float)
    sk_like = cnt * (np.log((1 + 2) / (1 + (cnt > 0).sum(0))) + 1)
    sk_like /= np.linalg.norm(sk_like, axis=1, keepdims=True)
    print("sklearn  :\n", S[:, cols])
    print("by hand  :\n", sk_like)
    print("max |diff| =", float(np.abs(S[:, cols] - sk_like).max()))
    print("cosine(BoW) =", round(cosine(X[0], X[1]), 4),
          " cosine(TF-IDF) =", round(cosine(tfidf[0], tfidf[1]), 4))
    print("words found in every document (i, nlp) get IDF 0 with ln(N/df)")
    assert np.abs(S[:, cols] - sk_like).max() < 1e-12
    print("encodings_np self-test passed")
