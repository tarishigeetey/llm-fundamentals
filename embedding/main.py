import math

documents = [
    "i love python",
    "python is great",
    "cooking recipes are fun",
    "apple is my fav fruit",
]
vocab = {
    "i": 0,
    "love": 1,
    "python": 2,
    "is": 3,
    "great": 4,
    "cooking": 5,
    "recipes": 6,
    "are": 7,
    "fun": 8,
}


def magnitude(vector):
    total = 0

    for x in vector:
        total += x**2

    return math.sqrt(total)


def dot_product(a, b):
    total = 0

    for x, y in zip(a, b):
        total += x * y

    return total


def cosine_similarity(a, b):
    numerator = dot_product(a, b)

    denominator = magnitude(a) * magnitude(b)

    return numerator / denominator


def build_vocab(documents):
    vocab = {}

    for doc in documents:
        for word in doc.split():
            if word not in vocab:
                vocab[word] = len(vocab)

    return vocab


def vectorize(text, vocab):

    vector = [0] * len(vocab)

    for word in text.split():
        if word in vocab:
            index = vocab[word]
            vector[index] = 1

    return vector


doc_vectors = []

for doc in documents:
    doc_vectors.append(vectorize(doc, vocab))

query_vector = vectorize("python", vocab)


def search(query, documents, vocab):

    query_vector = vectorize(query, vocab)

    results = []

    for doc in documents:
        doc_vector = vectorize(doc, vocab)

        score = cosine_similarity(query_vector, doc_vector)

        results.append((doc, score))

    results.sort(key=lambda x: x[1], reverse=True)

    return results


def vectorize_tf(text, vocab):

    vector = [0] * len(vocab)

    for word in text.split():
        if word in vocab:
            index = vocab[word]
            vector[index] += 1

    return vector


def document_frequency(documents):

    df = {}

    for doc in documents:
        unique_words = set(doc.split())

        for word in unique_words:
            df[word] = df.get(word, 0) + 1

    return df


def compute_idf(documents):

    df = document_frequency(documents)

    total_docs = len(documents)

    idf = {}

    for word, count in df.items():
        idf[word] = math.log(total_docs / count)

    return idf


def vectorize_tfidf(text, vocab, idf):

    tf = {}

    for word in text.split():
        tf[word] = tf.get(word, 0) + 1

    vector = [0] * len(vocab)

    for word, count in tf.items():
        if word in vocab:
            index = vocab[word]

            vector[index] = count * idf[word]

    return vector
