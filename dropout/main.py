import numpy as np


def dropout(x, drop_prob=0.2, training=True):
    """
    Applies Inverted Dropout.

    Parameters
    ----------
    x : numpy.ndarray
        Input activations

    drop_prob : float
        Probability of dropping a neuron

    training : bool
        True during training
        False during inference

    Returns
    -------
    numpy.ndarray
    """

    if not training:
        return x

    keep_prob = 1 - drop_prob

    # Random mask of True/False values
    mask = np.random.rand(*x.shape) < keep_prob

    # Inverted Dropout
    output = (x * mask) / keep_prob

    return output


# ---------------- Demo ----------------

np.random.seed(42)

x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

print("Input:")
print(x)

print("\nTraining:")
print(dropout(x, drop_prob=0.5, training=True))

print("\nInference:")
print(dropout(x, drop_prob=0.5, training=False))
