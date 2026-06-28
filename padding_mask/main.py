import numpy as np


# -----------------------------
# Softmax Function
# -----------------------------
def softmax(x):
    exp = np.exp(x - np.max(x))  # Numerical stability
    return exp / np.sum(exp)


# -----------------------------
# Attention Scores
# -----------------------------
scores = np.array([4.0, 3.0, 5.0, 2.0])

print("=" * 60)
print("Original Attention Scores")
print("=" * 60)
print(scores)

# -----------------------------
# Padding Mask
# -----------------------------
mask = np.array([1, 1, 1, 0])

print("\nPadding Mask")
print(mask)

# -----------------------------
# Apply Mask
# -----------------------------
masked_scores = np.where(mask == 1, scores, -1e9)

print("\nMasked Scores")
print(masked_scores)

# -----------------------------
# Softmax
# -----------------------------
attention = softmax(masked_scores)

print("\nAttention Probabilities")
print(attention)

print("\nSum of probabilities:")
print(np.sum(attention))
