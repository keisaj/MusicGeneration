import os
import pickle
from datetime import datetime

import numpy as np
from keras.utils import np_utils, plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from model import MusicNet

EPOCHS = 300
INIT_LEARNING_RATE = 0.001
BATCH_SIZE = 256
SEQUENCE_LENGTH = 100
DATASET = "all"

INIT_EPOCH = 0

MODEL_NAME = f"model_trained_on_{DATASET}_seq_{SEQUENCE_LENGTH}"

# Add weights path if you want to continue training form certain weights, else None
# WEIGHTS_PATH = f"./models/{MODEL_NAME}/weights/"\
#                "weights_trained_on_tracks_bach_corpus-epoch-283-loss-0.1609-val_loss-5.1636-notes_acc-0.9616-val_notes_acc-0.4932-rhythmic_acc-0.9844-val_rhythmic_acc-0.7466.hdf5"
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
    if not os.path.exists(f"models/{MODEL_NAME}"):
        os.mkdir(f"models/{MODEL_NAME}")
        os.mkdir(f"models/{MODEL_NAME}/weights")

    with open(f'data/{DATASET}', 'rb') as filepath:
        tracks = pickle.load(filepath)

    X_train, X_test, X_val, y_train, y_test, y_val, z_train, z_test, z_val = get_data(tracks=tracks)

    model = MusicNet.build_final_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                                       notes_vocab=y_train.shape[1],
                                       duration_vocab=z_train.shape[1])

    plot_model(model, to_file='model_architecture.png')

    losses = {
        "notes_output": "categorical_crossentropy",
        "rhythmic_output": "categorical_crossentropy",
    }

    loss_weights = {"notes_output": 1.0, "rhythmic_output": 1.0}
    opt = Adam(learning_rate=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    model.save(f"models/{MODEL_NAME}/")

    create_info(X_train, X_test, X_val, tracks, y_train, z_train)

    print(model.summary())

    train(model=model,
          network_input=X_train,
          network_output={'notes_output': y_train, 'rhythmic_output': z_train},
          weights_path=WEIGHTS_PATH,
          initial_epoch=INIT_EPOCH,
          epochs=EPOCHS,
          batch_size=BATCH_SIZE,
          validation_data=(X_test, {'notes_output': y_test, 'rhythmic_output': z_test}))


def get_data(tracks):
    if os.path.exists(f'models/{MODEL_NAME}/train_dataset') and os.path.exists(f'models/{MODEL_NAME}/test_dataset') \
            and os.path.exists(f'models/{MODEL_NAME}/val_dataset'):
        print("Loading datasets from existing files...")
        with open(f'models/{MODEL_NAME}/train_dataset', 'rb') as filepath:
            X_train, y_train, z_train = pickle.load(filepath)
        with open(f'models/{MODEL_NAME}/test_dataset', 'rb') as filepath:
            X_test, y_test, z_test = pickle.load(filepath)
        with open(f'models/{MODEL_NAME}/val_dataset', 'rb') as filepath:
            X_val, y_val, z_val = pickle.load(filepath)
    else:
        print("Creating datasets...")
        network_input, network_output_notes, network_output_durations = prepare_sequences(tracks=tracks,
                                                                                          sequence_len=SEQUENCE_LENGTH)

        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(network_input, network_output_notes,
                                                                             network_output_durations,
                                                                             test_size=0.2, random_state=42)

        X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(X_train, y_train,
                                                                          z_train,
                                                                          test_size=0.01, random_state=42)

        # save training_data
        with open(f'models/{MODEL_NAME}/train_dataset', 'wb') as filepath:
            pickle.dump((X_train, y_train, z_train), filepath)
        # save test_data
        with open(f'models/{MODEL_NAME}/test_dataset', 'wb') as filepath:
            pickle.dump((X_test, y_test, z_test), filepath)
        # save validation_data for prediction
        with open(f'models/{MODEL_NAME}/val_dataset', 'wb') as filepath:
            pickle.dump((X_val, y_val, z_val), filepath)

    return X_train, X_test, X_val, y_train, y_test, y_val, z_train, z_test, z_val


def create_info(X_train, X_test, X_val, tracks, y_train, z_train):
    info = {'n_all_samples': [sum(x) for x in zip(X_train.shape, X_test.shape, X_val.shape)][0],
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape,
            'X_val_shape': X_val.shape,
            'melodic_vocab': y_train.shape[1],
            'rhythmic_vocab': z_train.shape[1],
            'n_tracks': len(tracks)}

    # write dataset info to txt.file
    with open(f'models/{MODEL_NAME}/info.txt', 'w') as f:
        for key in info.keys():
            f.write(f"{key} : {info[key]}")
            f.write('\n')


def prepare_sequences(tracks: list, sequence_len: int):
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
        for i in range(0, len(notes) - sequence_len, 1):
            sequence_in = notes[i:i + sequence_len]
            sequence_out = notes[i + sequence_len]
            network_input.append([[note_to_int[char[0]], duration_to_int[char[1]]] for char in sequence_in])
            network_output.append([note_to_int[sequence_out[0]], duration_to_int[sequence_out[1]]])

    n_patterns = len(network_input)

    # reshape the input into a shape compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_len, 2))
    # normalize input
    network_input_normalized = normalize_network_input(network_input=network_input)

    # change output to categorical depending on if it is duration or note
    network_output = np.array(network_output)
    network_output_notes = np_utils.to_categorical(network_output[:, 0])
    network_output_durations = np_utils.to_categorical(network_output[:, 1])

    return network_input_normalized, network_output_notes, network_output_durations


def normalize_network_input(network_input):
    network_input = network_input.astype('float64')
    network_input_normalized = network_input.copy()
    network_input_normalized[:, :, 0] = network_input[:, :, 0] / network_input[:, :, 0].max()
    network_input_normalized[:, :, 1] = network_input[:, :, 1] / network_input[:, :, 1].max()
    return network_input_normalized


def train(model, network_input: np.array, network_output: np.array, epochs: int,
          initial_epoch: int, batch_size: int, weights_path: str = None, validation_data=None):
    logdir = f"logs/scalars/{MODEL_NAME}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)

    checkpoint = ModelCheckpoint(
        WEIGHTS_PATH_TEMPLATE,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint, tensorboard_callback]
    if weights_path:
        model.load_weights(weights_path)
        print('weights loaded....')
    model.fit(x=network_input, y=network_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list,
              initial_epoch=initial_epoch, validation_data=validation_data, verbose=2)


if __name__ == '__main__':
    train_network()
