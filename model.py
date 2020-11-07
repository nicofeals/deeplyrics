import os
import numpy as np
import tensorflow as tf

file = "merged_lyrics_freeze.txt"
lyrics = open(file, "r").read()
print('Length of text: {} characters'.format(len(lyrics)))

chars = sorted(list(set(lyrics)))
print('{} unique characters'.format(len(chars)))

char2idx = dict((c,i) for i, c in enumerate(chars))
idx2char = np.array(chars)
print(char2idx)

lyrics_as_int = np.array([char2idx[c] for c in lyrics])
for char in char2idx:
    print('{}: {}'.format(repr(char), char2idx[char]))

seq_length = 100
examples_per_epoch = len(lyrics)//seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(lyrics_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for input_example, target_example in  dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

# # Batch size
# BATCH_SIZE = 64

# # Buffer size to shuffle the dataset
# # (TF data is designed to work with possibly infinite sequences,
# # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# # it maintains a buffer in which it shuffles elements).
# BUFFER_SIZE = 10000

# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# print(dataset)

# vocab_size = len(chars)
# embedding_dim = 256
# rnn_units = 1024

# def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
#     model = tf.keras.Sequential([
#         tf.keras.layers.Embedding(vocab_size, embedding_dim,
#                                   batch_input_shape=[batch_size, None]),
#         tf.keras.layers.GRU(rnn_units,
#                             return_sequences=True,
#                             stateful=True,
#                             recurrent_initializer='glorot_uniform'),
#         tf.keras.layers.Dense(vocab_size)
#     ])
#     return model

# model = build_model(
#     vocab_size=vocab_size,
#     embedding_dim=embedding_dim,
#     rnn_units=rnn_units,
#     batch_size=BATCH_SIZE)

# print(model.summary())

# for input_example_batch, target_example_batch in dataset.take(1):
#     example_batch_predictions = model(input_example_batch)
#     print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
#     sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
#     sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#     print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
#     print()
#     print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

# def loss(labels, logits):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# model.compile(optimizer="adam", loss=loss)
# checkpoint_dir = './training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
# checkpoint_callbacks = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_prefix,
#     save_weights_only=True,
#     save_best_only=True,
#     monitor="loss"
# )

# EPOCHS = 10

# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callbacks])