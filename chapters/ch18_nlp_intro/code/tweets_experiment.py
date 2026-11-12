# ci: skip
"""NLTK twitter_samples: why 99.7% accuracy is a bug, and honest numbers.
The labels of this corpus come from emoticons (':)' vs ':(') that stay in the text."""
import re
import json
import numpy as np
import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

for pkg in ("twitter_samples", "punkt_tab"):
    nltk.download(pkg, quiet=True)
pos = twitter_samples.strings("positive_tweets.json")
neg = twitter_samples.strings("negative_tweets.json")
tweets = pos + neg
y = np.array([1] * len(pos) + [0] * len(neg))
print("tweets:", len(pos), len(neg), " unique texts:", len(set(tweets)))

EMO = re.compile("|".join([
    r"(?::|;|=)(?:-)?(?:\)|\(|D|P|p)+",   # :) :( ;) :-) :D :P =)
    r"\)+\s*$", r"\(+\s*$",             # a tweet ending with ) or (
    r":'\(", r":\w\b", r"\(:", r"\):", r"<3"]))
clean = [re.sub(r"\s+", " ", EMO.sub(" ", t)).strip() for t in tweets]
print("tweets that contain an emoticon-like token:",
      sum(a != b for a, b in zip(tweets, clean)), "of", len(tweets))
print("example:", repr(tweets[0][-30:]), "->", repr(clean[0][-30:]))
n_clean_emo = sum(bool(re.search(r"[:;=]-?[()DP]", t)) for t in clean)
print("emoticons left after cleaning:", n_clean_emo)


def multihot(tok_lists, vocab):
    X = np.zeros((len(tok_lists), len(vocab)), dtype=np.int8)
    for i, toks in enumerate(tok_lists):
        for w in toks:
            j = vocab.get(w)
            if j is not None:
                X[i, j] = 1
    return X


def run_multihot(texts, seed, vocab_on_train_only=True):
    toks = [word_tokenize(t.lower()) for t in texts]
    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    src = [toks[i] for i in (tr if vocab_on_train_only else idx)]
    vocab = {}
    for tl in src:
        for w in tl:
            vocab.setdefault(w, len(vocab))
    Xtr = multihot([toks[i] for i in tr], vocab)
    Xte = multihot([toks[i] for i in te], vocab)
    m = LogisticRegression(max_iter=1000).fit(Xtr, y[tr])
    return accuracy_score(y[te], m.predict(Xte)), len(vocab), m, vocab


def run_sparse(texts, vec, seed, model=None):
    tr, te = train_test_split(np.arange(len(texts)), test_size=0.2, random_state=seed, stratify=y)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xte = vec.transform([texts[i] for i in te])
    m = (LogisticRegression(max_iter=1000) if model is None else model).fit(Xtr, y[tr])
    return accuracy_score(y[te], m.predict(Xte))


seeds = [0, 1, 2, 3, 4]
res = {}


def rec(name, vals):
    res[name] = (float(np.mean(vals)), float(np.std(vals)))
    print(f"{name:42s} {np.mean(vals):.4f} +- {np.std(vals):.4f}")


# 1. the notebook's pipeline (word_tokenize, binary, emoticons kept)
rec("multi-hot, emoticons kept", [run_multihot(tweets, s)[0] for s in seeds])
# 2. same, emoticons removed
rec("multi-hot, emoticons removed", [run_multihot(clean, s)[0] for s in seeds])
# 3. count and TF-IDF (sklearn default token pattern drops punctuation)
rec("BoW (counts), emoticons kept", [run_sparse(tweets, CountVectorizer(stop_words="english"), s) for s in seeds])
rec("BoW (counts), emoticons removed", [run_sparse(clean, CountVectorizer(stop_words="english"), s) for s in seeds])
rec("TF-IDF, emoticons removed", [run_sparse(clean, TfidfVectorizer(stop_words="english"), s) for s in seeds])
rec("TF-IDF, keep stop words", [run_sparse(clean, TfidfVectorizer(), s) for s in seeds])
rec("TF-IDF uni+bigrams", [run_sparse(clean, TfidfVectorizer(ngram_range=(1, 2), min_df=2), s) for s in seeds])
rec("TF-IDF, min_df=2", [run_sparse(clean, TfidfVectorizer(min_df=2), s) for s in seeds])

# 4. which tokens does the leaky model use?
acc, V, m, vocab = run_multihot(tweets, 0)
inv = {j: w for w, j in vocab.items()}
order = np.argsort(m.coef_[0])
print("most negative weights:", [(inv[j], round(float(m.coef_[0][j]), 2)) for j in order[:6]])
print("most positive weights:", [(inv[j], round(float(m.coef_[0][j]), 2)) for j in order[::-1][:6]])

# 5. classical models on clean TF-IDF
models = {
    "Logistic regression": lambda: LogisticRegression(max_iter=1000),
    "Linear SVM": lambda: LinearSVC(),
    "Naive Bayes": lambda: MultinomialNB(),
    "k-NN (k=5, cosine)": lambda: KNeighborsClassifier(5, metric="cosine", algorithm="brute"),
    "Decision tree": lambda: DecisionTreeClassifier(random_state=0),
    "Random forest": lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1),
}
for name, mk in models.items():
    rec("model: " + name, [run_sparse(clean, TfidfVectorizer(stop_words="english"), s, mk()) for s in seeds[:3]])

# 6. GloVe mean vectors (needs gensim and the 50-d vectors)
try:
    import gensim.downloader as api
    kv = api.load("glove-wiki-gigaword-50")

    def avg(t):
        v = [kv[w] for w in word_tokenize(t.lower()) if w in kv]
        return np.mean(v, axis=0) if v else np.zeros(50)

    E = np.array([avg(t) for t in clean])
    accs = []
    for s in seeds:
        tr, te = train_test_split(np.arange(len(E)), test_size=0.2, random_state=s, stratify=y)
        mm = LogisticRegression(max_iter=2000).fit(E[tr], y[tr])
        accs.append(accuracy_score(y[te], mm.predict(E[te])))
    rec("GloVe-50 mean vector + LR", accs)
except Exception as e:
    print("GloVe skipped:", e)

json.dump(res, open("tweets_results.json", "w"), indent=1)
