from keras import Input, Model
from keras.layers import LSTM, BatchNormalization, Dropout, Dense, TimeDistributed


class MusicNet:

    @staticmethod
    def build_lstm_branch(inputs):
        x = LSTM(units=512, return_sequences=True)(inputs)
        # x = TimeDistributed(Dropout(0.1))(x)
        x = Dropout(0.3)(x)
        x = LSTM(units=512, return_sequences=True)(x)
        # x = TimeDistributed(Dropout(0.1))(x)
        x = Dropout(0.3)(x)
        x = LSTM(units=512, return_sequences=False)(x)
        return x

    @staticmethod
    def build_notes_branch(inputs, notes_vocab):
        x = BatchNormalization()(inputs)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(notes_vocab, activation='softmax', name='notes_output')(x)
        return x

    @staticmethod
    def build_rhythmic_branch(inputs, duration_vocab):
        x = BatchNormalization()(inputs)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(duration_vocab, activation='softmax', name='rhythmic_output')(x)
        return x

    @staticmethod
    def build_final_model(input_shape: tuple, notes_vocab, duration_vocab):
        inputs = Input(shape=(input_shape[0], input_shape[1]))
        lstm_branch = MusicNet.build_lstm_branch(inputs)
        notes_branch = MusicNet.build_notes_branch(lstm_branch, notes_vocab)
        rhythmic_branch = MusicNet.build_rhythmic_branch(lstm_branch, duration_vocab)
        model = Model(inputs=inputs,
                      outputs=[notes_branch, rhythmic_branch],
                      name='musicnet')
        return model
