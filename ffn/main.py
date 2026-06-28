import numpy as np

# Make results reproducible
np.random.seed(42)

# ---------------------------------------
# One Token Embedding
# ---------------------------------------

# Embedding dimension = 2
x = np.array([2.0, 1.0])

print("=" * 60)
print("Input Embedding")
print("=" * 60)
print(x)

# ---------------------------------------
# Layer 1 (2 -> 4)
# ---------------------------------------

W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)

print("\n" + "=" * 60)
print("Weight Matrix W1")
print("=" * 60)
print(W1)

print("\nBias b1")
print(b1)

# Linear Transformation
hidden = x @ W1 + b1

print("\nHidden Layer (Before Activation)")
print(hidden)

# ---------------------------------------
# ReLU Activation
# ---------------------------------------


def relu(x):
    return np.maximum(0, x)


hidden_relu = relu(hidden)

print("\nAfter ReLU")
print(hidden_relu)

# ---------------------------------------
# Layer 2 (4 -> 2)
# ---------------------------------------

W2 = np.random.randn(4, 2)
b2 = np.random.randn(2)

print("\n" + "=" * 60)
print("Weight Matrix W2")
print("=" * 60)
print(W2)

print("\nBias b2")
print(b2)

# Linear Transformation
output = hidden_relu @ W2 + b2

print("\nOutput Embedding")
print(output)

# ---------------------------------------
# Shape Summary
# ---------------------------------------

print("\n" + "=" * 60)
print("Shape Summary")
print("=" * 60)

print(f"Input Shape      : {x.shape}")
print(f"W1 Shape         : {W1.shape}")
print(f"Hidden Shape     : {hidden.shape}")
print(f"ReLU Shape       : {hidden_relu.shape}")
print(f"W2 Shape         : {W2.shape}")
print(f"Output Shape     : {output.shape}")
