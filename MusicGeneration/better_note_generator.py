from operator import concat

from music21 import converter, instrument, note, chord, corpus, stream, duration
from music21 import stream, key, meter
import glob
import pickle
from functools import reduce

from typing import List, Tuple, Type

Music21_element = type('Music21_element')


def create_dataset(info_dict_list):
    """
    :param info_dict_list:
    :return:
    """
    all_elements = reduce(concat, [info_dict["elements_list"] for info_dict in info_dict_list])
    save_data(info_dict_list)
    # create dict for element names
    element_names = sorted(set(element[0] for element in all_elements))
    # create dict for element durations
    element_durations = sorted(set(element[1] for element in all_elements))
    # create dict to map names to integers
    note_to_int = dict((note, number) for number, note in enumerate(element_names))
    # create dict to map durations values to integers
    duration_to_int = dict((duration, number) for number, duration in enumerate(element_durations))

    # convert info_dicts into np.arrays -> network input and output TODO

    print("xd")


def save_data(info_dict_list):
    output_list = []
    for info_dict in info_dict_list:
        output_list.append(info_dict["elements_list"])

    with open(f'data/tracks_{DATASET}_better', 'wb') as filepath:
        pickle.dump(output_list, filepath)


def convert_files_to_dicts(path: str = "../goldberg_variations/*.mid"):
    """
    :param path:
    :return:
    """
    print("Conversion started")
    converted_elements_list = []
    # for every file in directory
    for filepath in glob.glob(f"{path}"):
        print(f"\t Converting {filepath}")
        converted_elements_list.append(get_music_info(filepath))

    return converted_elements_list


def get_music_info(filepath):
    # get list of elements: Measures, Notes, Chords, Rests and it's durations
    music21_elements_list = get_music21_elements(filepath)
    # get key of the piece
    key = get_key(music21_elements_list)
    # get time signature of the piece
    time_signature = get_time_signature(music21_elements_list)

    music_info_dict = {'key': key,
                       'time_signature': time_signature,
                       'elements_list': midi_to_list(filepath=filepath)}
    return music_info_dict


def convert_32_chords_into_notes(elemenets_list):
    temp_lst = []
    for element in elemenets_list:
        if isinstance(element, chord.Chord) and element.duration.quarterLength == 0.25:
            for n in element.pitches:
                temp_lst.append(note.Note(name=str(n), duration=duration.Duration(quarterLength=0.125)))
        else:
            temp_lst.append(element)
    return temp_lst


def midi_to_list(filepath: str):
    """
    :param filepath:
    :return:
    """
    # get list of music21 elements from file in filepath
    elements_list = get_music21_elements(filepath)
    # convert 0.25 chords into notes - music21 recognizes 32' as 0.25 chord
    converted_list = convert_32_chords_into_notes(elements_list)
    # convert music21 elements to [element, duration] -> List[str, float] and return them in separate List
    return [element_to_list(element) for element in converted_list if element_to_list(element) is not None]


def get_key(piece_elements):
    for element in piece_elements:
        if isinstance(element, key.Key):
            return element.name
    return


def get_time_signature(piece_elements):
    for element in piece_elements:
        if isinstance(element, meter.TimeSignature):
            return element.ratioString
    return


def element_to_list(element):
    """
    :param element:
    :return: List[str, float]
    """
    # if element is measure
    # if isinstance(element, stream.Measure):
    #     return ['Measure', 0.0]
    # if element is note
    if isinstance(element, note.Note):
        return [str(element.pitch), element.duration.quarterLength]
    # if elements is chord
    elif isinstance(element, chord.Chord):
        return ['.'.join(str(n) for n in element.pitches), element.duration.quarterLength]
    # if elements is pause
    elif isinstance(element, note.Rest):
        return [element.name, element.duration.quarterLength]
    return


def get_music21_elements(file_path: str, part: int = 0):
    """
    :param file_path:
    :param part:
    :return: List[Music21_element]
    """
    score = converter.parse(file_path)  # <music21.stream.Score>
    part = score.parts[part]  # <music21.stream.Part0>
    elements = part.recurse()  # <music21.stream.iterator.RecursiveIterator for Part>
    return list(elements)


if __name__ == "__main__":
    DATASET = "classical_midi"
    dict_list = convert_files_to_dicts(path=f"../{DATASET}/*.mid")
    create_dataset(dict_list)
    print("Done")
