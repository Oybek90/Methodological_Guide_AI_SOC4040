# ci: skip
"""Our NumPy attention against PyTorch, and parameter counts against the formula."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_np import (attention, causal_mask, multi_head, gpt_params,
                          block_params)

torch.manual_seed(0)
d, h, n = 16, 4, 6
x = torch.randn(1, n, d, dtype=torch.float64)

# 1. multi-head attention, no bias, equals nn.MultiheadAttention
mha = nn.MultiheadAttention(d, h, bias=False, batch_first=True).double()
Wq, Wk, Wv = mha.in_proj_weight.detach().chunk(3)          # each (d, d), y = x W^T
Wo = mha.out_proj.weight.detach()
mine = multi_head(x[0].numpy(), Wq.T.numpy(), Wk.T.numpy(), Wv.T.numpy(), Wo.T.numpy(), h)
ref, _ = mha(x, x, x, need_weights=False)
print("multi-head error        :", float(np.abs(mine - ref[0].detach().numpy()).max()))

# 2. causal attention equals F.scaled_dot_product_attention(is_causal=True)
Q = torch.randn(n, 8, dtype=torch.float64)
o, _ = attention(Q.numpy(), Q.numpy(), Q.numpy(), causal_mask(n))
o2 = F.scaled_dot_product_attention(Q[None], Q[None], Q[None], is_causal=True)[0]
print("causal attention error  :", float(np.abs(o - o2.numpy()).max()))

# 3. one nn.TransformerEncoderLayer has block_params(d, dff) parameters
dff = 4 * d
layer = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, batch_first=True)
cnt = sum(p.numel() for p in layer.parameters())
# PyTorch uses biases on q,k,v too:
print("TransformerEncoderLayer :", cnt, " formula (with qkv bias):", block_params(d, dff, True))
assert cnt == block_params(d, dff, True)

# 4. a small GPT: formula against a real count
class Tiny(nn.Module):
    def __init__(s, V, T, d, L, h):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        s.blocks = nn.ModuleList([nn.TransformerEncoderLayer(
            d, h, 4 * d, 0.0, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(L)])
        s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False)
        s.head.weight = s.tok.weight                       # weight tying

m = Tiny(1000, 64, 32, 4, 4)
real = sum(p.numel() for p in m.parameters())              # shared weights counted once
print("tiny GPT parameters     :", real, " formula:", gpt_params(1000, 64, 32, 4, qkv_bias=True))
assert real == gpt_params(1000, 64, 32, 4, qkv_bias=True)

# 5. the attention matrix is n x n: memory and FLOPs grow with n^2
for n in (1_000, 8_000, 128_000):
    print(f"n={n:7d}: scores per head and layer = {n * n:.2e} numbers "
          f"= {n * n * 2 / 1e9:.1f} GB in fp16")
