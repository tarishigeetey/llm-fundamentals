# Attention(Q,K,V) = softmax(QKᵀ)V

# QK^T = similarity score softmax(QK^T) = probablity of score
# Value vectors + weighted averaging = complete attention.
# 1. Compare
# 2. Normalize
# 3. Weighted Average

import math


def dot_product(a, b):
    total = 0

    for x, y in zip(a, b):
        total += x * y

    return total


sentence = ["I", "love", "python"]

embeddings = {"I": [1, 0], "love": [0, 1], "python": [1, 1]}

query = embeddings["python"]

keys = [embeddings["I"], embeddings["love"], embeddings["python"]]

scores = []

for key in keys:
    scores.append(dot_product(query, key))

print(scores)


def softmax(scores):

    exp_scores = []

    for score in scores:
        exp_scores.append(math.exp(score))

    total = sum(exp_scores)

    probabilities = []

    for value in exp_scores:
        probabilities.append(round(value / total, 2))

    return probabilities


weights = softmax(scores)

print(weights)

values = [
    [1, 0],  # I
    [0, 2],  # love
    [3, 1],  # python
]


def weighted_sum(weights, values):

    dimensions = len(values[0])

    result = [0] * dimensions

    for weight, value_vector in zip(weights, values):
        for i in range(dimensions):
            result[i] += weight * value_vector[i]

    return result


print(weighted_sum(weights, values))

query = [1, 1]

keys = [[1, 0], [0, 1], [1, 1]]

values = [[1, 0], [0, 2], [3, 1]]

scores = []

for key in keys:
    scores.append(dot_product(query, key))

weights = softmax(scores)

output = weighted_sum(weights, values)

print("Scores:", scores)
print("Weights:", weights)
print("Output:", output)

vectors = [[1, 0], [0, 1], [1, 1]]


def self_attention(vectors):

    outputs = []

    for query in vectors:
        scores = []

        for key in vectors:
            scores.append(dot_product(query, key))

        weights = softmax(scores)

        output = weighted_sum(weights, vectors)

        outputs.append(output)

    return outputs


outputs = self_attention(vectors)

for output in outputs:
    print(output)
