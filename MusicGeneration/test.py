import fractions

from music21 import *
import random

# b = converter.parse(r'C:\Users\janbu\PycharmProjects\PracaMagisterska\MusicGeneration\goldberg_variations\988-aria.mid')

# b.show()

file = r'C:\Users\janbu\PycharmProjects\PracaMagisterska\MusicGeneration\midi_songs\BlueStone_LastDungeon.mid'

def get_notes(file) -> dict:
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = {} # dictionary for notes storage in separate parts

    midi = converter.parse(str(file)) # midi file converted to notes stream
    print("Parsing %s" % file)

    parts = []

    # for part in instrument.partitionByInstrument(midi):
    for part in midi:

        parts.append(part.recurse())

    notes_to_parse = {f'{idx}': part for idx, part in enumerate(parts)}


    for idx, part in enumerate(notes_to_parse.values()):
        notes[f'{idx}'] = []

    for idx, part in enumerate(notes_to_parse.values()):
        for element in part:
            element_duration = element.duration.quarterLength
            # if isinstance(element_duration, fractions.Fraction):
            #     element_duration = round(float(element_duration.numerator/element_duration.denominator), 2)

            if isinstance(element, note.Note):
                notes[f'{idx}'].append([str(element.pitch), element_duration])
            elif isinstance(element, chord.Chord):
                notes[f'{idx}'].append(['.'.join(str(n) for n in element.normalOrder), element_duration])
            elif isinstance(element, note.Rest):
                notes[f'{idx}'].append([element.name, element_duration])

    return notes

def create_midi(prediction_output: dict):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """

    output_notes = {f'{idx}': [] for idx in prediction_output.keys()}

    # create note and chord objects based on the values generated by the model
    for idx, part in enumerate(prediction_output.values()):
        offset = 0
        for element, element_duration in part:
            d = duration.Duration()
            d.quarterLength = element_duration
            # element is a chord
            if ('.' in element) or element.isdigit():
                notes_in_chord = element.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                new_chord.duration = d
                output_notes[f'{idx}'].append(new_chord)
            # element is a rest
            elif element == 'rest':
                new_note = note.Rest(element)
                new_note.offset = offset
                new_note.duration = d
                new_note.storedInstrument = instrument.Piano()
                output_notes[f'{idx}'].append(new_note)
            # element is a note
            else:
                new_note = note.Note(element)
                new_note.offset = offset
                new_note.duration = d
                new_note.storedInstrument = instrument.Piano()
                output_notes[f'{idx}'].append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += element_duration
    midi_stream = stream.Stream()
    # put parts into stream
    for idx, part in enumerate(output_notes.values()):
        p = stream.Part()
        p.append(part)
        midi_stream.insert(0, p)


    midi_stream.write('midi', fp='test.mid')


notes = get_notes(file)
create_midi(prediction_output=notes)
print(notes)