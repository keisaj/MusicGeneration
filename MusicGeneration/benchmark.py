from music21 import converter, instrument, note, chord, corpus
import glob
import pickle

# lst = ["tracks_classical_midi_better",
#        "tracks_goldberg_variations_better",
#        "tracks_midi_songs_better",
#        "tracks_haydn_corpus_only_one_part",
#        "tracks_bach_corpus_augmented_len_200"]
# all = []
# for notes in lst:
#
#     with open(f"data/{notes}", 'rb') as filepath:
#         tracks = pickle.load(filepath)
#         all.extend(tracks)
#
# with open(f'data/tracks_all', 'wb') as filepath:
#     pickle.dump(all, filepath)


with open(f"data/tracks_midi_songs_better", 'rb') as filepath:
        notes = pickle.load(filepath)

print("")