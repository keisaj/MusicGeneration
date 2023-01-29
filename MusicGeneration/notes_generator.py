from music21 import converter, note, chord, corpus
import glob


def get_notes():  # TODO it should now be called get_tracks
    # notes = []
    tracks = []
    # all_bach_paths = corpus.getComposer('haydn')
    path = r'C:\Users\janbu\AppData\Local\Programs\Python\Python38\Lib\site-packages\music21\corpus\bach'

    # for file in glob.glob(f"../goldberg_variations/*.mid"):

    for file in glob.glob(f'{path}/*.mxl'):

        piece = converter.parse(file)

        print("Parsing %s" % file)

        # s2 = instrument.partitionByInstrument(piece)

        for part in piece.parts:
            notes = []
            notes_to_parse = part.recurse()

            # notes_to_parse = midi.parts.activeElementList[0].recurse()

            for element in notes_to_parse:
                # check element duration and add it to note
                if isinstance(element, note.Note):
                    notes.append([str(element.pitch), element.duration.quarterLength])
                elif isinstance(element, chord.Chord):
                    notes.append(['.'.join(str(n) for n in element.pitches), element.duration.quarterLength])
                # add pause detection
                elif isinstance(element, note.Rest):
                    notes.append([element.name, element.duration.quarterLength])

            # print(notes)
            tracks.append(notes)
        print("")
    # tracks = filter_tracks(tracks)
    # with open(f'data/tracks_goldberg_variations', 'wb') as filepath:
    #     pickle.dump(tracks, filepath)

    return tracks


def filter_tracks(tracks, min_track_len=200):
    return [x for x in tracks if len(x) >= min_track_len]


get_notes()
