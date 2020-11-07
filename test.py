import tensorflow as tf
import sys
import numpy as np
import pickle
import tempfile
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


# Hotfix function
def make_keras_picklable():
    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
        self.__dict__ = model.__dict__


    cls = Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def test(args):
    # Run the function
    make_keras_picklable()

    file = "merged_lyrics_metalcore.txt"
    file_augmented = "merged_lyrics_metalcore.txt"
    
    lyrics = open(file, "r").read() + open(file_augmented, "r").read()
    print('Length of text: {} characters'.format(len(lyrics)))

    chars = sorted(list(set(lyrics)))
    print('{} unique characters'.format(len(chars)))

    char2idx = dict((c,i) for i, c in enumerate(chars))
    idx2char = np.array(chars)

    lyrics_as_int = np.array([char2idx[c] for c in lyrics])

    seq_length = 120
    examples_per_epoch = len(lyrics)//seq_length

    char_dataset = tf.data.Dataset.from_tensor_slices(lyrics_as_int)

    vocab_size = len(chars)
    embedding_dim = 256
    rnn_units = 1024

    def build_model(vocab_size, embedding_dim, rnn_units, batch_size, stateful=True):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=stateful,
                                recurrent_activation="sigmoid",
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=stateful,
                                recurrent_activation="sigmoid",
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size)
        ], name="GRU")
        return model

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights('best_weight_latest2.h5')
    model.build(tf.TensorShape([1, None]))
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.save('deeplyrics.h5')

    print(model.summary())

    # pred = generate_text(model, idx2char, char2idx, args[1].lower())
    # print()
    # print()
    # print(pred)


def generate_text(model, idx2char, char2idx, start_string):
    # Number of characters to generate
    num_generate = 3000

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model.predict(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


if __name__ == "__main__":
    print(sys.argv)
    test(sys.argv)