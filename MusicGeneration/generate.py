import pickle

from model import MusicNet
from main import prepare_sequences, WEIGHTS_PATH
import numpy as np

USED_NOTES_PATH = 'data/notes_test'
WEIGHTS_PATH = 'weights_trained_on_midi_songs/' + 'weights_trained_on_midi-improvement-100-6.4002-bigger.hdf5'

def generate():
    with open(USED_NOTES_PATH, 'rb') as filepath:
        songs = pickle.load(filepath)

    pitchnames = sorted(set(item[0] for item in sum(songs, [])))
    durations = sorted(set(item[1] for item in sum(songs, [])))

    notes_vocab = len(pitchnames)
    duration_vocab = len(durations)

    network_input, _, _ = prepare_sequences(songs)

    model = MusicNet.build_final_model(network_input, notes_vocab, duration_vocab)
    model.load_weights(WEIGHTS_PATH)

    prediction_output = generate_notes(model, network_input, pitchnames, durations, notes_vocab, duration_vocab)
    create_midi(prediction_output)

def generate_notes(model, network_input, pitchnames, durations, notes_vocab, duration_vocab):
    start = np.random.randint(0, len(network_input) - 1)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    duration_to_int = dict((duration, number) for number, duration in enumerate(durations))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 2))
        prediction_input = prediction_input / float(notes_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        note_index = np.argmax(prediction[0])
        duration_index = np.argmax(prediction[1])

        note = note_to_int[note_index]
        duration = duration_to_int[duration_index]

        prediction_output.append([note, duration])

        pattern.append([note_index, duration_index])
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    pass

generate()