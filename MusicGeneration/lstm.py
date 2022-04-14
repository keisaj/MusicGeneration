""" This module prepares midi file data and feeds it to the neural
    network for training """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord, exceptions21
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras import backend as k
import random

WEIGHTS_PATH = 'weights/'
WEIGHTS_TRAINED_ON_LOFI_PATH = WEIGHTS_PATH + 'weights_trained_on_lofi/'
WEIGHTS_TRAINED_ON_JAZZ_PATH = WEIGHTS_PATH + 'weights_trained_on_jazz/'
WEIGHTS_TRAINED_ON_GOLDBERG_VARIATIONS_PATH = WEIGHTS_PATH + 'weights_trained_on_goldberg_variations/'
WEIGHTS_TRAINED_ON_MIDI_SONGS_PATH = WEIGHTS_PATH + 'weights_trained_on_midi_songs/'

JAZZ_PATH = 'jazz/'
LOFI_PATH = 'lofi/'
GOLDBERG_VARIATIONS_PATH = 'goldberg_variations/'
MIDI_SONGS_PATH = 'midi_songs/'

DATA_NOTES_PATH = 'data/notes'

# CURRENTLY USED PATHS
USED_WEIGHTS_PATH = WEIGHTS_TRAINED_ON_JAZZ_PATH
USED_NOTES_PATH = DATA_NOTES_PATH + '_jazz'
USED_MIDI_PATH = JAZZ_PATH
WEIGHTS_FILE_NAME = "weights_trained_on_jazz-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"


def train_network():
    """ Train a Neural Network to generate music """
    # notes = get_notes()
    with open(USED_NOTES_PATH, 'rb') as filepath:
        notes = pickle.load(filepath)
    # get amount of pitch names
    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []
    file_list = [random.choice(glob.glob(USED_MIDI_PATH + "*.mid")) for i in range(50)]
    for file in file_list:
        try:
            midi = converter.parse(str(file))
            print("Parsing %s" % file)

            notes_to_parse = None

            piano_parts = []

            try:  # file has instrument parts
                # s2 = instrument.partitionByInstrument(midi)
                # notes_to_parse = s2.parts[0].recurse()

                instr = instrument.Piano
                for part in instrument.partitionByInstrument(midi):
                    if isinstance(part.getInstrument(), instr):
                        piano_parts.append(part.recurse())
                notes_to_parse = piano_parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
                # elif isinstance(element, note.Rest):  # ADDED
                #     notes.append(element.name)  # ADDED
        except exceptions21.StreamException:
            print('failed to find time signature for file %s' % file)

    with open(USED_NOTES_PATH, 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    # model = Sequential()
    # model.add(LSTM(
    #     512,
    #     input_shape=(network_input.shape[1], network_input.shape[2]),
    #     return_sequences=True,
    #     activation="tanh", recurrent_activation="sigmoid", unroll=False, use_bias=True
    # ))
    # model.add(LSTM(512, return_sequences=True, activation="tanh", recurrent_activation="sigmoid", unroll=False, use_bias=True))
    # model.add(LSTM(512, activation="tanh", recurrent_activation="sigmoid", unroll=False, use_bias=True))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(BatchNorm())
    # model.add(Dropout(0.3))
    # model.add(Dense(n_vocab))
    # model.add(Activation('softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = USED_WEIGHTS_PATH + WEIGHTS_FILE_NAME
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list = [checkpoint]
    model.load_weights(WEIGHTS_TRAINED_ON_JAZZ_PATH+'weights_trained_on_jazz-improvement-197-1.2607-bigger.hdf5')
    model.fit(network_input, network_output, epochs=250, batch_size=128, callbacks=callbacks_list, initial_epoch=197)
    # model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()
