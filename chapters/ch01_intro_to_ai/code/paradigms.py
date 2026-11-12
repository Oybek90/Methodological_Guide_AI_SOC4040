# ci: run
"""The three learning paradigms in one file: supervised, unsupervised, reinforcement."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def supervised(seed=0):
    """Labelled digits -> classifier; evaluated on unseen test images."""
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    return clf.fit(Xtr, ytr).score(Xte, yte)


def unsupervised(seed=0):
    """Same images, labels hidden -> k-means groups; compare with hidden labels afterwards."""
    X, y = load_digits(return_X_y=True)
    km = KMeans(n_clusters=10, n_init=10, random_state=seed).fit(X)
    return adjusted_rand_score(y, km.labels_)


def reinforcement(episodes=200, n_states=8, alpha=0.5, gamma=0.9, eps=0.2, seed=0):
    """Tabular Q-learning on a corridor: start at state 0, reward 1 at the last state.

    Actions: 0 = left, 1 = right. Returns the number of steps used in each episode.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, 2))
    steps_per_episode = []
    for _ in range(episodes):
        s, steps = 0, 0
        while s != n_states - 1 and steps < 1000:
            a = rng.integers(2) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(2)))
            s2 = max(0, s - 1) if a == 0 else s + 1
            r = 1.0 if s2 == n_states - 1 else 0.0
            Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
            s, steps = s2, steps + 1
        steps_per_episode.append(steps)
    return Q, steps_per_episode


if __name__ == "__main__":
    print(f"supervised   : test accuracy        = {supervised():.3f}")
    print(f"unsupervised : adjusted Rand index  = {unsupervised():.3f}  (1 = labels recovered, 0 = chance)")
    Q, st = reinforcement()
    print("reinforcement: steps in episodes 1, 5, 20, 200 =", [st[0], st[4], st[19], st[199]])
    print("               learned policy (0=left, 1=right):", Q.argmax(axis=1)[:-1].tolist())
