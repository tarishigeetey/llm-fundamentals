import numpy as np


def layer_norm(tokens, gamma, beta, eps=1e-5):
    """
    Layer Normalization

    Args:
        tokens : shape (num_tokens, embedding_dim)
        gamma  : shape (embedding_dim,)
        beta   : shape (embedding_dim,)
        eps    : small value to avoid divide-by-zero

    Returns:
        mean
        variance
        std
        normalized_tokens
        output
    """

    # Step 1: Mean of each token
    mean = np.mean(tokens, axis=1, keepdims=True)

    # Step 2: Variance of each token
    variance = np.mean((tokens - mean) ** 2, axis=1, keepdims=True)

    # Step 3: Standard Deviation
    std = np.sqrt(variance + eps)

    # Step 4: Normalize
    normalized = (tokens - mean) / std

    # Step 5: Scale & Shift
    output = gamma * normalized + beta

    return mean, variance, std, normalized, output


# ----------------------------------------------------
# Example Input
# ----------------------------------------------------

tokens = np.array([[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0], [9.0, 8.0, 7.0, 6.0]])

embedding_dim = tokens.shape[1]

gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

# ----------------------------------------------------
# Run LayerNorm
# ----------------------------------------------------

mean, variance, std, normalized, output = layer_norm(tokens, gamma, beta)

# ----------------------------------------------------
# Print Results
# ----------------------------------------------------

print("=" * 60)
print("Original Tokens")
print("=" * 60)
print(tokens)

print("\nShape:", tokens.shape)

print("\n" + "=" * 60)
print("Mean")
print("=" * 60)
print(mean)

print("\nShape:", mean.shape)

print("\n" + "=" * 60)
print("Variance")
print("=" * 60)
print(variance)

print("\nShape:", variance.shape)

print("\n" + "=" * 60)
print("Standard Deviation")
print("=" * 60)
print(std)

print("\nShape:", std.shape)

print("\n" + "=" * 60)
print("Normalized Tokens")
print("=" * 60)
print(normalized)

print("\nShape:", normalized.shape)

print("\n" + "=" * 60)
print("Final Output")
print("=" * 60)
print(output)

# ----------------------------------------------------
# Verification
# ----------------------------------------------------

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

print("\nMean of every token:")
print(np.mean(normalized, axis=1))

print("\nStd of every token:")
print(np.std(normalized, axis=1))
