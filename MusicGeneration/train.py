import pickle
import numpy as np
from music21 import converter, instrument, note, chord, corpus
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from model import MusicNet
import os
from keras.optimizers import Adam
import random
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

EPOCHS = 300
INIT_LEARNING_RATE = 0.001
BATCH_SIZE = 256
SEQUENCE_LENGTH = 50
DATASET = "tracks_bach_corpus_augmented_len_200"

INIT_EPOCH = 0

# MODEL_NAME = f"model_lr-{INIT_LEARNING_RATE}_" \
#              f"bs-{BATCH_SIZE}_" \
#              f"epochs-{EPOCHS}_" \
#              f"seqlen-{SEQUENCE_LENGTH}_" \
#              f"trained_on_{DATASET}" + "_different_dropout"

MODEL_NAME = f"model_trained_on_{DATASET}_seq_{SEQUENCE_LENGTH}"

# WEIGHTS_PATH = f"./models/{MODEL_NAME}/weights/"+"weights_trained_on_tracks_bach_corpus_augmented_len_200-epoch-164-loss-0.5451-val_loss-4.0202-notes_acc-0.8682-val_notes_acc-0.4637-rhythmic_acc-0.9478-val_rhythmic_acc-0.7342.hdf5"
WEIGHTS_PATH = None
WEIGHTS_PATH_TEMPLATE = f"models/{MODEL_NAME}/weights/weights_trained_on_{DATASET}" \
                        "-epoch-{epoch:02d}" \
                        "-loss-{loss:.4f}" \
                        "-val_loss-{val_loss:.4f}" \
                        "-notes_acc-{notes_output_accuracy:.4f}" \
                        "-val_notes_acc-{val_notes_output_accuracy:.4f}" \
                        "-rhythmic_acc-{rhythmic_output_accuracy:.4f}" \
                        "-val_rhythmic_acc-{val_rhythmic_output_accuracy:.4f}.hdf5"


def train_network():
    """ Train a Neural Network to generate music """

    with open(f'data/{DATASET}', 'rb') as filepath:
        tracks = pickle.load(filepath)

    # get amount of pitch names
    all_notes = [note for track in tracks for note in track]
    n_vocab = len(set(note[0] for note in all_notes))
    d_vocab = len(set(note[1] for note in all_notes))

    network_input, network_output_notes, network_output_durations = prepare_sequences(tracks)
    print(network_input.shape)
    X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(network_input, network_output_notes,
                                                                         network_output_durations,
                                                                         test_size=0.33, random_state=42)

    model = MusicNet.build_final_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                                       notes_vocab=network_output_notes.shape[1],
                                       duration_vocab=network_output_durations.shape[1])

    plot_model(model, to_file='model.png')

    losses = {
        "notes_output": "categorical_crossentropy",
        "rhythmic_output": "categorical_crossentropy",
    }

    loss_weights = {"notes_output": 1.0, "rhythmic_output": 1.0}
    opt = Adam(learning_rate=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    if not os.path.exists(f"models/{MODEL_NAME}"):
        model.save(f"models/{MODEL_NAME}")
        os.mkdir(f"models/{MODEL_NAME}/weights")

    with open(f'models/{MODEL_NAME}/test_dataset', 'wb') as filepath:
        pickle.dump((X_test, {'notes_output': y_test, 'rhythmic_output': z_test}), filepath)

    print(model.summary())

    train(model=model,
          network_input=X_train,
          network_output={'notes_output': y_train, 'rhythmic_output': z_train},
          weights_path=WEIGHTS_PATH,
          initial_epoch=INIT_EPOCH,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_test, {'notes_output': y_test, 'rhythmic_output': z_test}))

def prepare_sequences(tracks: list):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = SEQUENCE_LENGTH
    all_notes = [note for track in tracks for note in track]
    # get all element names
    pitchnames = sorted(set(item[0] for item in all_notes))
    # get all duration values
    durations = sorted(set(item[1] for item in all_notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    # create a dictionary to map duration values to integers
    duration_to_int = dict((duration, number) for number, duration in enumerate(durations))

    network_input = []
    network_output = []
    for notes in tracks:

        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([[note_to_int[char[0]], duration_to_int[char[1]]] for char in sequence_in])
            network_output.append([note_to_int[sequence_out[0]], duration_to_int[sequence_out[1]]])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 2))
    # normalize input
    network_input_normalized = normalize_network_input(network_input=network_input)

    # change output to categorical depending on if it is duration or note
    network_output = np.array(network_output)
    network_output_notes = np_utils.to_categorical(network_output[:, 0])
    network_output_durations = np_utils.to_categorical(network_output[:, 1])

    return network_input_normalized, network_output_notes, network_output_durations


def shuffle_dataset(network_input, network_output):
    assert len(network_output) == len(network_output)
    list_to_shuffle = [[i, o] for i, o in zip(network_input, network_output)]
    random.seed(10)
    random.shuffle(list_to_shuffle)
    shuffled_input = [x[0] for x in list_to_shuffle]
    shuffled_output = [x[1] for x in list_to_shuffle]

    assert len(network_input) == len(shuffled_input)
    assert len(network_output) == len(shuffled_output)
    return shuffled_input, shuffled_output


def normalize_network_input(network_input):
    network_input = network_input.astype('float64')
    network_input_normalized = network_input.copy()
    # norm_notes = np.linalg.norm(network_input[:, :, 0])
    # norm_durations = np.linalg.norm(network_input[:, :, 1])
    network_input_normalized[:, :, 0] = network_input[:, :, 0] / network_input[:, :, 0].max()
    network_input_normalized[:, :, 1] = network_input[:, :, 1] / network_input[:, :, 1].max()
    return network_input_normalized


def train(model, network_input: np.array, network_output: np.array, epochs: int,
          initial_epoch: int, batch_size: int, weights_path: str = None, validation_data=None):

    checkpoint = ModelCheckpoint(
        WEIGHTS_PATH_TEMPLATE,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]
    if weights_path:
        model.load_weights(weights_path)
        print('weights loaded....')
    model.fit(x=network_input, y=network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
              initial_epoch=initial_epoch, validation_data=validation_data, verbose=2)


if __name__ == '__main__':
    train_network()
