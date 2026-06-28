import numpy as np


# -------------------------
# LayerNorm
# -------------------------
def layer_norm(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    return (x - mean) / np.sqrt(var + eps)


# -------------------------
# Dropout
# -------------------------
def dropout(x, drop_prob=0.1, training=True):

    if not training:
        return x

    keep_prob = 1 - drop_prob

    mask = np.random.rand(*x.shape) < keep_prob

    return (x * mask) / keep_prob


# -------------------------
# Feed Forward Network
# -------------------------
def relu(x):
    return np.maximum(0, x)


def feed_forward(x, W1, b1, W2, b2):

    hidden = relu(x @ W1 + b1)

    output = hidden @ W2 + b2

    return output


# -------------------------
# Fake Multi-Head Attention
# (Placeholder)
# -------------------------
def multi_head_attention(x):

    # Placeholder so we can assemble the block.
    # Later we'll replace this with our real implementation.
    return x * 0.5


def transformer_block(x, W1, b1, W2, b2, drop_prob=0.1, training=True):

    # ---------- Attention ----------
    attn = multi_head_attention(x)

    attn = dropout(attn, drop_prob, training)

    x = layer_norm(x + attn)

    # ---------- FFN ----------
    ffn = feed_forward(x, W1, b1, W2, b2)

    ffn = dropout(ffn, drop_prob, training)

    x = layer_norm(x + ffn)

    return x


# Tests

np.random.seed(42)

batch_size = 2
seq_len = 3
d_model = 4
hidden_dim = 8

x = np.random.randn(batch_size, seq_len, d_model)

W1 = np.random.randn(d_model, hidden_dim)
b1 = np.random.randn(hidden_dim)

W2 = np.random.randn(hidden_dim, d_model)
b2 = np.random.randn(d_model)

output = transformer_block(x, W1, b1, W2, b2, drop_prob=0.2, training=True)

print(output.shape)
print(output)
