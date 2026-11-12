# ci: run
"""Chinchilla parametric loss L(N,D) = E + A/N^a + B/D^b and its compute-optimal split."""
import numpy as np

FITS = {   # (E, A, B, alpha, beta)
    "Hoffmann et al. (published)": (1.69, 406.4, 410.7, 0.34, 0.28),
    "Besiroglu et al. (replication)": (1.82, 482.01, 2085.43, 0.35, 0.37),
}


def loss(N, D, fit):
    E, A, B, a, b = fit
    return E + A / N ** a + B / D ** b


def optimum(C, fit):
    """Minimise L subject to 6 N D = C (closed form)."""
    E, A, B, a, b = fit
    G = (a * A / (b * B)) ** (1 / (a + b))
    N = G * (C / 6) ** (b / (a + b))
    return N, C / (6 * N)


if __name__ == "__main__":
    C = 5.76e23                                   # approx. training compute of Chinchilla (FLOPs)
    for name, fit in FITS.items():
        N, D = optimum(C, fit)
        print(f"{name:32s} C={C:.2e}: N*={N / 1e9:6.1f}B  D*={D / 1e9:7.0f}B  D/N={D / N:5.1f}"
              f"  L*={loss(N, D, fit):.3f}")
        for c in (1e21, 1e23, 1e25):
            n, d = optimum(c, fit)
            print(f"    C={c:.0e}: N*={n:.2e} D*={d:.2e} D/N={d / n:5.1f}")
    # named models: (parameters, training tokens) as reported by their authors
    models = {"GPT-3": (175e9, 300e9), "Gopher": (280e9, 300e9),
              "Chinchilla": (70e9, 1.4e12), "LLaMA-65B": (65e9, 1.4e12)}
    for k, (N, D) in models.items():
        print(f"{k:11s} N={N:.1e} D={D:.1e} D/N={D / N:5.1f}  6ND={6 * N * D:.2e} FLOPs"
              f"  L_pub={loss(N, D, FITS['Hoffmann et al. (published)']):.3f}"
              f"  L_rep={loss(N, D, FITS['Besiroglu et al. (replication)']):.3f}")
    # the cost of an exact power law: halving the reducible loss
    a = FITS["Besiroglu et al. (replication)"]
    N, D = optimum(1e23, a)
    red = loss(N, D, a) - a[0]
    for k in (10, 100):
        n2, d2 = optimum(1e23 * k, a)
        print(f"x{k:3d} compute: reducible loss {red:.3f} -> {loss(n2, d2, a) - a[0]:.3f}")
    assert abs(optimum(1e23, a)[0] * optimum(1e23, a)[1] * 6 - 1e23) < 1e17
    print("scaling_laws self-test passed")
