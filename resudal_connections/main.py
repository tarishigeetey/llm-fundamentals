import numpy as np


def residual_connection(x, fx):
    """
    Performs a residual (skip) connection.

    Output = x + F(x)
    """
    return x + fx


# -----------------------------
# Example 1
# -----------------------------

x = np.array([2.0, 4.0, 6.0, 8.0])

fx = np.array([0.5, -1.0, 0.2, 1.5])

output = residual_connection(x, fx)

print("=" * 50)
print("Example 1")
print("=" * 50)

print("Input x:")
print(x)

print("\nLayer Output F(x):")
print(fx)

print("\nResidual Output:")
print(output)

print("\nVerification (output - x):")
print(output - x)


# -----------------------------
# Example 2
# -----------------------------

print("\n" + "=" * 50)
print("Example 2")
print("=" * 50)

x = np.array([10.0, 20.0, 30.0])

fx = np.array([-10.0, -20.0, -30.0])

output = residual_connection(x, fx)

print("Input:")
print(x)

print("\nF(x):")
print(fx)

print("\nResidual Output:")
print(output)
