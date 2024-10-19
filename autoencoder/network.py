import tensorflow as tf
from typing import Tuple, List


def AutoEncoder(
        input_size: Tuple[int] = (50, 50),
        embedding_dims: int = 2,
        encoder_hidden_dims: List[int] = [100],
        decoder_hidden_dims: List[int] = [100],
):
    inp = tf.keras.layers.Input(shape=input_size, name='input')
    flatten_dims = input_size[0] * input_size[1]
    #
    encoder = [tf.keras.layers.Reshape((flatten_dims,))]
    for d in encoder_hidden_dims:
        encoder.append(tf.keras.layers.Dense(d, activation="relu"))
    encoder.append(tf.keras.layers.Dense(embedding_dims, activation="relu"))
    encoder = tf.keras.Sequential(encoder, name="encoder")
    #
    decoder = []
    for d in decoder_hidden_dims:
        decoder.append(tf.keras.layers.Dense(d, activation="relu"))
    decoder.append(tf.keras.layers.Dense(flatten_dims, activation="sigmoid"))
    decoder.append(tf.keras.layers.Reshape(input_size))
    decoder = tf.keras.Sequential(decoder, name="decoder")
    #
    z = encoder(inp)
    out = decoder(z)
    return tf.keras.Model(inputs=inp, outputs=out)


if __name__ == "__main__":
    network = AutoEncoder()
    network.summary()
