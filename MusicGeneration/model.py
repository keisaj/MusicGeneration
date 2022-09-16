from keras import Input, Model
from keras.layers import LSTM, Lambda, BatchNormalization, Dropout, Dense


class MusicNet:

    @staticmethod
    def build_lstm_branch(inputs):
        x = LSTM(units=512, recurrent_dropout=0.3, return_sequences=True)(inputs)
        x = LSTM(units=512, return_sequences=True, recurrent_dropout=0.3)(x)
        x = LSTM(units=512, return_sequences=True)(x)
        return x

    # @staticmethod
    # def build_lstm_branch(inputs):
    #     x = LSTM(units=512, return_sequences=True,
    #              activation='tanh',
    #              recurrent_activation='sigmoid',
    #              unroll=False,
    #              use_bias=True)(inputs)
    #     x = LSTM(units=512, return_sequences=True,
    #              activation='tanh',
    #              recurrent_activation='sigmoid',
    #              unroll=False,
    #              use_bias=True)(x)
    #     x = LSTM(units=512, activation="tanh",
    #              recurrent_activation="sigmoid",
    #              unroll=False,
    #              use_bias=True)(x)
    #     return x

    @staticmethod
    def build_notes_branch(inputs, notes_vocab):
        x = Lambda(lambda x: x[:, 0])(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(notes_vocab, activation='softmax', name='notes_output')(x)
        return x

    @staticmethod
    def build_rythmic_branch(inputs, duration_vocab):
        x = Lambda(lambda x: x[:, 1])(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(duration_vocab, activation='softmax', name='rythmic_output')(x)
        return x

    @staticmethod
    def build_final_model(network_inuput, notes_vocab, duration_vocab):
        inputs = Input(shape=(network_inuput.shape[1], network_inuput.shape[2]))
        lstm_branch = MusicNet.build_lstm_branch(inputs)
        notes_branch = MusicNet.build_notes_branch(lstm_branch, notes_vocab)
        rythmic_branch = MusicNet.build_rythmic_branch(lstm_branch, duration_vocab)
        model = Model(inputs=inputs,
                      outputs=[notes_branch, rythmic_branch],
                      name='musicnet')
        return model


