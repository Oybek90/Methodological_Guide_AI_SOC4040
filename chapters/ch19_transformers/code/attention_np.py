# ci: run
"""Transformer building blocks in NumPy: attention, heads, masks, positions."""
import numpy as np


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(n):
    """True where attention is FORBIDDEN (position j > i)."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def attention(Q, K, V, mask=None, scale=True):
    """softmax(Q K^T / sqrt(d_k)) V ; returns (output, weights)."""
    s = Q @ K.T
    if scale:
        s = s / np.sqrt(K.shape[-1])
    if mask is not None:
        s = np.where(mask, -np.inf, s)
    w = softmax(s, axis=-1)
    return w @ V, w


def multi_head(X, Wq, Wk, Wv, Wo, h, mask=None):
    """X: (n, d). All W: (d, d). Splits d into h heads of size d/h."""
    n, d = X.shape
    dh = d // h
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs = []
    for i in range(h):
        sl = slice(i * dh, (i + 1) * dh)
        o, _ = attention(Q[:, sl], K[:, sl], V[:, sl], mask)
        outs.append(o)
    return np.concatenate(outs, axis=1) @ Wo


def sinusoidal_pe(n, d):
    """PE[p, 2i] = sin(p / 10000^(2i/d)), PE[p, 2i+1] = cos(...)."""
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None, :]
    ang = pos / (10000.0 ** (2 * i / d))
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(ang), np.cos(ang)
    return pe


def layer_norm(x, g=None, b=None, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    y = (x - m) / np.sqrt(v + eps)
    return y if g is None else g * y + b


def ffn(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2


def init_block(d, dff, rng):
    s = 1 / np.sqrt(d)
    return dict(Wq=rng.normal(0, s, (d, d)), Wk=rng.normal(0, s, (d, d)),
                Wv=rng.normal(0, s, (d, d)), Wo=rng.normal(0, s, (d, d)),
                W1=rng.normal(0, s, (d, dff)), b1=np.zeros(dff),
                W2=rng.normal(0, 1 / np.sqrt(dff), (dff, d)), b2=np.zeros(d))


def encoder_block(x, P, h):
    """Post-LN block of the original paper (no mask)."""
    x = layer_norm(x + multi_head(x, P["Wq"], P["Wk"], P["Wv"], P["Wo"], h))
    return layer_norm(x + ffn(x, P["W1"], P["b1"], P["W2"], P["b2"]))


def decoder_block(x, P, h):
    """Pre-LN block with a causal mask (GPT style)."""
    m = causal_mask(len(x))
    y = layer_norm(x)
    x = x + multi_head(y, P["Wq"], P["Wk"], P["Wv"], P["Wo"], h, m)
    return x + ffn(layer_norm(x), P["W1"], P["b1"], P["W2"], P["b2"])


def block_params(d, dff, qkv_bias=False):
    """One block: attention 4d^2 + d (+3d if q,k,v have biases), FFN 2 d dff + dff + d, 2 LN."""
    att = 4 * d * d + d + (3 * d if qkv_bias else 0)
    ff = 2 * d * dff + dff + d
    return att + ff + 4 * d


def gpt_params(vocab, ctx, d, layers, dff=None, tied=True, qkv_bias=False):
    dff = dff or 4 * d
    n = vocab * d + ctx * d + layers * block_params(d, dff, qkv_bias) + 2 * d
    return n if tied else n + vocab * d


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    rng = np.random.default_rng(0)

    # 1. hand-checkable example (Worked example in the chapter)
    X = np.array([[1., 0.], [0., 1.], [1., 1.]])
    out, w = attention(X, X, X)
    print("weights:\n", w)
    print("output:\n", out)
    _, wc = attention(X, X, X, causal_mask(3))
    print("causal weights:\n", wc)

    # 2. causality: changing a future token does not change earlier outputs
    d, h, n = 16, 4, 6
    P = init_block(d, 4 * d, rng)
    x = rng.normal(size=(n, d))
    y1 = decoder_block(x, P, h)
    x2 = x.copy(); x2[-1] += 5.0
    y2 = decoder_block(x2, P, h)
    print("max change of positions 0..4 :", float(np.abs(y1[:-1] - y2[:-1]).max()))
    print("max change of position 5     :", float(np.abs(y1[-1] - y2[-1]).max()))
    assert np.abs(y1[:-1] - y2[:-1]).max() == 0.0

    # 3. no positional information => permutation equivariance
    perm = rng.permutation(n)
    e1 = encoder_block(x, P, h)
    e2 = encoder_block(x[perm], P, h)
    print("permutation equivariance error:", float(np.abs(e1[perm] - e2).max()))
    assert np.abs(e1[perm] - e2).max() < 1e-12
    pe = sinusoidal_pe(n, d)
    e3 = encoder_block(x + pe, P, h)
    e4 = encoder_block(x[perm] + pe, P, h)
    print("with positional encoding, the error is", float(np.abs(e3[perm] - e4).max()))

    # 4. why divide by sqrt(d_k)
    print("d_k   Var(q.k)  mean max-weight (no scale)  (with scale)")
    for dk in (4, 64, 512, 4096):
        q = rng.normal(size=(2000, dk)); k = rng.normal(size=(10, dk))
        s = q @ k.T
        v = s.var()
        m0 = softmax(s).max(1).mean()
        m1 = softmax(s / np.sqrt(dk)).max(1).mean()
        print(f"{dk:5d} {v:9.1f} {m0:14.3f} {m1:20.3f}")

    # 5. parameter counts
    print("GPT-2 small, no qkv bias :", gpt_params(50257, 1024, 768, 12))
    print("GPT-2 small, qkv bias    :", gpt_params(50257, 1024, 768, 12, qkv_bias=True))
    print("   untied output head    :", gpt_params(50257, 1024, 768, 12, tied=False))
    print("one block d=768:", block_params(768, 3072))
    assert gpt_params(50257, 1024, 768, 12) == 124_412_160
    assert gpt_params(50257, 1024, 768, 12, qkv_bias=True) == 124_439_808
    print("encoder_np self-test passed")
