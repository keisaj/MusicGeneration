import pickle
import numpy as np
from music21 import instrument, note, stream, chord, duration
from train import prepare_sequences, INIT_LEARNING_RATE, EPOCHS

from keras.optimizers import Adam
from keras import models

MODEL_PATH = "models/" + "model_trained_on_tracks_goldberg_variations_better_seq_100"
WEIGHTS_PATH = "weights_trained_on_tracks_goldberg_variations_better-epoch-295-loss-0.0492-val_loss-3.0617-notes_acc-0.9893-val_notes_acc-0.7097-rhythmic_acc-0.9959-val_rhythmic_acc-0.8881.hdf5"
NOTES_PATH = "data/tracks_goldberg_variations_better"
N_NOTES = 500
OUTPUT_NAME = "test_output_4"


def generate_music(model_path: str, weights_path: str, notes_path: str, n_notes: int):
    """ Generate a piano midi file """
    # load the notes used to train the model
    with open(f"{notes_path}", 'rb') as filepath:
        tracks = pickle.load(filepath)

    # create list of all notes from tracks
    all_notes = [note for track in tracks for note in track]
    # get all notes names
    pitchnames = sorted(set(item[0] for item in all_notes))
    # get all duration values
    durations = sorted(set(item[1] for item in all_notes))
    n_vocab = len(pitchnames)
    d_vocab = len(durations)

    network_input, network_output_notes, network_output_durations = prepare_sequences(tracks)

    model = models.load_model(f"{model_path}")

    losses = {
        "notes_output": "categorical_crossentropy",
        "rhythmic_output": "categorical_crossentropy",
    }

    loss_weights = {"notes_output": 1.0, "rhythmic_output": 1.0}
    opt = Adam(learning_rate=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE / EPOCHS)
    model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

    model.load_weights(
        f"{model_path}/weights/{weights_path}")

    prediction_output = generate_notes(model, network_input, pitchnames, durations, n_vocab, d_vocab, n_notes)
    create_midi(prediction_output, model_path)


def generate_notes(model, network_input, pitchnames: list, durations: list, n_vocab: int, d_vocab: int, n_notes: int):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    int_to_duration = dict((number, duration) for number, duration in enumerate(durations))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(n_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern), 2))

        prediction = model.predict(prediction_input, verbose=0)

        note_index = int(np.argmax(prediction[0]))
        duration_index = int(np.argmax(prediction[1]))

        note = int_to_note[note_index]
        duration = int_to_duration[duration_index]

        prediction_output.append([note, duration])

        pattern = list(pattern)
        #                           scaling prediction to match input
        pattern.append([note_index / n_vocab, duration_index / d_vocab])
        pattern = pattern[1:len(pattern)]


    return prediction_output


def create_midi(prediction_output: list, model_path: str):
    offset = 0
    output_notes = []

    for element, element_duration in prediction_output:

        d = duration.Duration()
        d.quarterLength = element_duration

        # element is a chord
        if ('.' in element) or element.isdigit():
            notes_in_chord = element.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.duration = d
            output_notes.append(new_chord)
        # element is a rest
        elif element == 'rest':
            new_rest = note.Rest(element)
            new_rest.offset = offset
            new_rest.storedInstrument = instrument.Piano()
            new_rest.duration = d
            output_notes.append(new_rest)
        # element is a note
        else:
            new_note = note.Note(element)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = d
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += element_duration
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=f'{model_path}/{OUTPUT_NAME}.mid')

# Print iterations progress
def progressBar(iterable, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iterable    - Required  : iterable object (Iterable)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    total = len(iterable)
    # Progress Bar Printing Function
    def printProgressBar (iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Initial Call
    printProgressBar(0)
    # Update Progress Bar
    for i, item in enumerate(iterable):
        yield item
        printProgressBar(i + 1)
    # Print New Line on Complete
    print()

if __name__ == '__main__':
    generate_music(model_path=MODEL_PATH,
                   weights_path=WEIGHTS_PATH,
                   notes_path=NOTES_PATH,
                   n_notes=N_NOTES)
