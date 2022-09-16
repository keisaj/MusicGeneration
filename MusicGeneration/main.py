import glob
import pickle
import random

import numpy
from keras.optimizers import Adam
from keras.utils import np_utils
from music21 import converter, note, chord
from keras.callbacks import ModelCheckpoint
from model import MusicNet

MIDI_PATH = 'midi_songs/'
NOTES_PATH = 'data/notes_test'
WEIGHTS_PATH = 'weights_trained_on_midi_songs/' + "weights_trained_on_midi-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"

EPOCHS = 100
INIT_LR = 0.001
BS = 128


def main():
    # lista przetworzonych utworów, kazdy utwór zapisany w osobnej liście
    songs = get_songs(midi_path=MIDI_PATH, notes_path=NOTES_PATH, n_samples=10)

    # liczba wykorzystanych ogólnie nut bez powtórzeń
    notes_vocab = len(set(note[0] for note in sum(songs, [])))
    duration_vocab = len(set(note[1] for note in sum(songs, [])))
    # na podstawie dostępnych nut i liczby unikatowych nut otrzymuje input podawany do sieci
    # i output którego się spodziwam
    network_input, notes_output, rythmic_output = prepare_sequences(songs)
    notes_output = np_utils.to_categorical(notes_output)
    rythmic_output = np_utils.to_categorical(rythmic_output)

    # tworze model na podstawie długości network_input i n_vocab
    model = MusicNet.build_final_model(network_input, notes_vocab, duration_vocab)
    losses = {
        "notes_output": "categorical_crossentropy",
        "rythmic_output": "categorical_crossentropy",
    }
    loss_weights = {"notes_output": 1.0, "rythmic_output": 1.0}
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    print(model.summary())

    checkpoint = ModelCheckpoint(
        WEIGHTS_PATH,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]

    model.fit(x=network_input,
              y={'notes_output': notes_output, 'rythmic_output': rythmic_output},
              epochs=EPOCHS,
              batch_size=BS,
              callbacks=callbacks_list)


def get_songs(midi_path, notes_path, n_samples: int):
    # file_list = [random.choice(glob.glob(midi_path + "*.mid")) for i in range(n_samples)]
    file_list = glob.glob((midi_path + "*.mid"))

    songs = []

    for file in file_list:
        notes = []
        midi = converter.parse(str(file))
        print(f"Parsing {str(file)}")

        part = midi.parts[0]  # bierzemy tylko partię melodyczną

        for element in part.recurse():
            if isinstance(element, note.Note):
                notes.append([str(element.pitch), element.duration.quarterLength])
            elif isinstance(element, chord.Chord):
                notes.append(['.'.join(str(n) for n in element.pitches), element.duration.quarterLength])
            elif isinstance(element, note.Rest):
                notes.append([element.name, element.duration.quarterLength])

        songs.append(notes)
    with open(notes_path, 'wb') as filepath:
        pickle.dump(songs, filepath)

    return songs


def prepare_sequences(songs,sequence_length: int = 100):
    # weź wszystkie występujące nuty bez powtórzeń, zignoruj znak STOP
    pitchnames = sorted(set(item[0] for item in sum(songs, [])))
    durations = sorted(set(item[1] for item in sum(songs, [])))
    n_vocab = len(pitchnames)
    # stwórz słownik konwertujący nutę na liczbę
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    # stwórz słownik konwertujący czas trwania nuty na liczbę
    duration_to_int = dict((duration, number) for number, duration in enumerate(durations))
    network_input = []
    notes_output = []
    rythmic_output = []

    # create input sequences and the corresponding outputs

    for song in songs:

        for i in range(0, len(song) - sequence_length, 1):
            sequence_in = song[i:i + sequence_length]
            sequence_out = song[i + sequence_length]

            network_input.append([[note_to_int[char[0]], duration_to_int[char[1]]] for char in sequence_in])
            # network_output.append([note_to_int[sequence_out[0]], duration_to_int[sequence_out[1]]])
            notes_output.append(note_to_int[sequence_out[0]])
            rythmic_output.append(duration_to_int[sequence_out[1]])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 2))
    # normalize input
    network_input = network_input / float(n_vocab) #TODO this normalization could be done better, for example each column devided by different value

    return network_input, notes_output, rythmic_output


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = 'weights_trained_on_midi_songs/' + "weights_trained_on_midi-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]
    # model.load_weights(WEIGHTS_TRAINED_ON_JAZZ_PATH+'weights_trained_on_jazz-improvement-197-1.2607-bigger.hdf5')
    # model.fit(network_input, network_output, epochs=250, batch_size=128, callbacks=callbacks_list, initial_epoch=197)
    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


# main()
