documents = {
    "FastAPI Backend": [0.9, 0.2, 0.1],
    "Machine Learning": [0.2, 0.9, 0.8],
    "Python API Design": [0.8, 0.3, 0.2],
    "Cooking Recipe": [0.1, 0.1, 0.9],
}

documents["Huge Vector"] = [9, 2, 1]


def dot_product(a, b):
    total = 0

    for x, y in zip(a, b):
        total += x * y

    return total


def search(query_vector, k=2):
    results = []

    for title, vector in documents.items():
        score = dot_product(query_vector, vector)

        results.append((title, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results[:k]


query = [0.85, 0.25, 0.15]

print(search(query))
