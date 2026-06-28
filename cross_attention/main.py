import numpy as np


def softmax(x):
    exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def cross_attention(decoder_input, encoder_output, Wq, Wk, Wv):
    """
    Cross Attention

    Parameters
    ----------
    decoder_input : (target_seq_len, d_model)
        Hidden states from decoder

    encoder_output : (source_seq_len, d_model)
        Output from encoder

    Wq : (d_model, d_model)
    Wk : (d_model, d_model)
    Wv : (d_model, d_model)

    Returns
    -------
    output : (target_seq_len, d_model)
    attention_weights : (target_seq_len, source_seq_len)
    """

    # ---------------------
    # Query from Decoder
    # ---------------------
    Q = decoder_input @ Wq

    # ---------------------
    # Key from Encoder
    # ---------------------
    K = encoder_output @ Wk

    # ---------------------
    # Value from Encoder
    # ---------------------
    V = encoder_output @ Wv

    # ---------------------
    # Attention Scores
    # ---------------------
    scores = Q @ K.T

    # ---------------------
    # Scale
    # ---------------------
    dk = K.shape[-1]

    scores = scores / np.sqrt(dk)

    # ---------------------
    # Softmax
    # ---------------------
    attention_weights = softmax(scores)

    # ---------------------
    # Weighted Sum
    # ---------------------
    output = attention_weights @ V

    return output, attention_weights
