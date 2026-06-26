import math

embeddings = {"I": [1, 0], "love": [0, 1], "python": [1, 1]}

sentence = ["I", "love", "python"]

# These would normally be learned during training. For our toy example, we'll define them manually.

X = [embeddings[word] for word in sentence]

Wq = [[2, 0], [0, 1]]

Wk = [[1, 0], [0, 2]]

Wv = [[1, 1], [0, 2]]


def matmul(A, B):
    result = []

    for row in A:
        new_row = []

        for col in range(len(B[0])):
            value = 0

            for k in range(len(B)):
                value += row[k] * B[k][col]

            new_row.append(value)

        result.append(new_row)

    return result


def dot_product(a, b):
    total = 0

    for x, y in zip(a, b):
        total += x * y

    return total


def softmax(scores):

    exps = []

    for score in scores:
        exps.append(math.exp(score))

    total = sum(exps)

    weights = []

    for value in exps:
        weights.append(value / total)

    return weights


def weighted_sum(weights, values):

    output = [0] * len(values[0])

    for weight, vector in zip(weights, values):
        for i in range(len(vector)):
            output[i] += weight * vector[i]

    return output


def transformer_attention(X, Wq, Wk, Wv):

    # Create Query, Key and Value matrices

    Q = matmul(X, Wq)
    K = matmul(X, Wk)
    V = matmul(X, Wv)

    outputs = []

    # Compute attention for every token

    for query in Q:
        scores = []

        # Compare query with every key
        for key in K:
            scores.append(dot_product(query, key))

        # Convert scores to probabilities
        weights = softmax(scores)

        # Mix value vectors
        output = weighted_sum(weights, V)

        outputs.append(output)

    return outputs


outputs = transformer_attention(X, Wq, Wk, Wv)

for word, output in zip(sentence, outputs):
    print(word, "->", output)
