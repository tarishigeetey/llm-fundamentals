import numpy as np

# -----------------------------
# Configuration
# -----------------------------
seq_len = 6  # Number of words in sentence
d_model = 8  # Embedding size

# -----------------------------
# Create empty Positional Encoding matrix
# Shape = (seq_len, d_model)
# -----------------------------
PE = np.zeros((seq_len, d_model))

# -----------------------------
# Fill the matrix
# -----------------------------
for pos in range(seq_len):
    # Even columns: 0,2,4,6
    for i in range(0, d_model, 2):
        # Calculate angle
        angle = pos / (10000 ** (i / d_model))

        # Even dimension -> sin
        PE[pos, i] = np.sin(angle)

        # Odd dimension -> cos
        PE[pos, i + 1] = np.cos(angle)

print("Positional Encoding Matrix")
print(np.round(PE, 4))
print()

print("Shape:", PE.shape)
