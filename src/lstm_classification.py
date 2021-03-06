import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, LSTM, SpatialDropout1D, Embedding, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from src.dataset_preparation import DatasetPreparation
from src.model_evaluation import ModelEvaluation


class LSTMClassification:

    def __init__(self, model_save_path: str = 'models/lstm_model.p') -> None:
        self._model_save_path = model_save_path
        self._max_number_of_words = 50000  # maximum number of words to be used. (most frequent)
        self._embedding_dim = 300
        self._categories_path = 'data/categories_lstm.p'
        self._tokenizer_save_path = 'data/tokenizer_for_lstm.p'
        self._max_sequence_length = 435  # maximum number of words in each text for NN

    def get_string_tokens(self, text: str) -> list:
        """Returns numeric tokens of input string."""
        tokenizer = pickle.load(open(self._tokenizer_save_path, 'rb'))
        tokens = tokenizer.texts_to_sequences([text])
        return tokens

    def get_tokens_and_labels(self, data_df: pd.DataFrame) -> list:
        """ Returns tokenized text and one hot encoded categories as labels."""

        tokenizer = Tokenizer(num_words=self._max_number_of_words,
                              filters=r'!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                              lower=True)
        tokenizer.fit_on_texts(data_df['text'].values)
        tokens = tokenizer.texts_to_sequences(data_df['text'].values)

        if not os.path.exists(self._tokenizer_save_path):
            pickle.dump(tokenizer, open(self._tokenizer_save_path, 'wb'))

        one_hot_encoded_categories = pd.get_dummies(data_df['category'])
        labels = one_hot_encoded_categories.values
        categories = one_hot_encoded_categories.columns.tolist()

        if not os.path.exists(self._categories_path):
            pickle.dump(categories, open(self._categories_path, 'wb'))

        return [tokens, labels, categories]

    def get_padded_sequences(self, tokens: list) -> list:
        """Returns padded vectors with zeros."""
        padded_tokens = pad_sequences(tokens, maxlen=self._max_sequence_length)
        return padded_tokens

    def _get_model(self, input_length) -> Sequential:
        """Defines the model's architecture."""
        model = Sequential()
        model.add(Embedding(self._max_number_of_words, self._embedding_dim, input_length=input_length, mask_zero=True))
        model.add(SpatialDropout1D(0.2))
        model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(24, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def calculate_category_weights(data_df: pd.DataFrame, categories: list, weighted: bool = False) -> dict:
        """Calculation of weights for each category."""
        data_value_counts = data_df.category.value_counts()
        category_weights = data_value_counts.max() / data_value_counts
        class_weights = dict()

        for ind, category in enumerate(categories):

            if weighted:
                class_weights[ind] = category_weights[category]
            else:
                class_weights[ind] = 1.

        return class_weights

    def train_model(self, train_test_data_labels: list, class_weights: dict, epochs: int = 5,
                    batch_size: int = 64, save_model: bool = True) -> Sequential:
        """Trains the LSTM model on input data and labels."""
        [train_data, test_data, train_labels, test_labels] = train_test_data_labels
        model = self._get_model(train_data.shape[1])

        history = model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size,
                            validation_data=(test_data, test_labels),
                            class_weight=class_weights,
                            callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        self._plot_training_history(history)

        if save_model:
            pickle.dump(model, open(self._model_save_path, 'wb'))
            model.save(self._model_save_path)

        return model

    @staticmethod
    def _plot_training_history(history) -> None:
        """Shows plots with accuracy and loss during training."""
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()

    def evaluate_model(self, data: list, labels: list, categories: list, model: Sequential = None) -> None:
        """Evaluates the model on the input data."""

        if model is None:
            model = pickle.load(open(self._model_save_path, 'rb'))

        model_accuracy = model.evaluate(data, labels)
        print(f'Test set\n  Loss: {model_accuracy[0]}\n  Accuracy: {model_accuracy[1]}')

        predicted_output = model.predict(data)
        predicted_labels = [np.argmax(y_pred) for y_pred in predicted_output]
        labels = [np.argmax(y_test) for y_test in labels]
        evaluator = ModelEvaluation()
        evaluator.evaluate_model(labels, predicted_labels, categories)

    def classify(self, text: str) -> str:
        """Classifies input text with LSTM model and returns it's category."""
        print(f'Input: {text}')
        preparator = DatasetPreparation()
        text = preparator.clean_text(text)

        tokens = self.get_string_tokens(text)
        padded_tokens = self.get_padded_sequences(tokens)
        model = pickle.load(open(self._model_save_path, 'rb'))
        categories = pickle.load(open(self._categories_path, 'rb'))

        predicted_output = model.predict(padded_tokens)
        predicted_label = np.argmax(predicted_output)
        return categories[predicted_label]
