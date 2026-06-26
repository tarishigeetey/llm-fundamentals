import numpy as np

np.set_printoptions(precision=3, suppress=True)
np.random.seed(42)

# -----------------------------
# Input
# -----------------------------

X = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]], dtype=float)

num_heads = 2
d_model = 4
head_dim = d_model // num_heads

# -----------------------------
# Weights
# -----------------------------

Wq = np.random.randn(num_heads, d_model, head_dim)
Wk = np.random.randn(num_heads, d_model, head_dim)
Wv = np.random.randn(num_heads, d_model, head_dim)

# -----------------------------
# Softmax
# -----------------------------


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# -----------------------------
# Attention
# -----------------------------


def attention(Q, K, V):
    scores = (Q @ K.T) / np.sqrt(K.shape[-1])
    weights = softmax(scores)
    return weights @ V


# -----------------------------
# Multi-head
# -----------------------------

outputs = []

for h in range(num_heads):
    Q = X @ Wq[h]
    K = X @ Wk[h]
    V = X @ Wv[h]

    out = attention(Q, K, V)

    outputs.append(out)

multi = np.concatenate(outputs, axis=-1)

Wo = np.random.randn(d_model, d_model)

final = multi @ Wo

print("Final Output")
print(final)
