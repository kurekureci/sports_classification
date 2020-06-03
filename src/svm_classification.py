from nltk.tokenize.casual import casual_tokenize
import os
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.model_evaluation import ModelEvaluation
from src.dataset_preparation import DatasetPreparation
from src.fasttext_vectors import FasttextVectors


class SVMClassification:

    def __init__(self, model_save_path: str = 'svm_model.p') -> None:
        self._model_save_path = model_save_path
        self._categories_path = 'data/categories.p'

    @staticmethod
    def get_string_tokens(text: str) -> list:
        """Returns word tokens of input string."""
        return casual_tokenize(text)

    def get_tokens_and_labels(self, data_df: pd.DataFrame) -> list:
        """ Returns tokenized text and encoded categories as numbers.

        :param data_df: dataframe with columns: text, category
        """
        tokens = data_df['text'].map(lambda x: self.get_string_tokens(x))

        # encode categories to labels - text to numbers
        encoder = preprocessing.LabelEncoder()
        labels = encoder.fit_transform(data_df['category'])
        categories = list(encoder.classes_)

        if not os.path.exists(self._categories_path):
            pickle.dump(categories, open(self._categories_path, 'wb'))

        return [tokens, labels, categories]

    @staticmethod
    def get_model():
        """Defines the model's architecture."""
        return Pipeline([('std', StandardScaler()), ('svm', SVC(kernel='rbf'))])

    def train_model(self, data: list, labels: list, save_model: bool = True):
        """Trains the SVM model on input data and labels."""
        model = self.get_model()
        model.fit(data, labels)

        if save_model:
            pickle.dump(model, open(self._model_save_path, 'wb'))

        return model

    def evaluate_model(self, data: list, labels: list, categories: list, model=None) -> None:
        """Evaluates the model on the input data."""

        if model is None:
            model = pickle.load(open(self._model_save_path, 'rb'))

        predicted_labels = model.predict(data)
        evaluator = ModelEvaluation()
        evaluator.evaluate_model(labels, predicted_labels, categories)

    def classify(self, text: str) -> str:
        """Classifies input text with SVM model and returns it's category."""
        print(f'Input: {text}')
        preparator = DatasetPreparation()
        text = preparator.clean_text(text)
        tokens = self.get_string_tokens(text)

        fasttext = FasttextVectors()
        text_vector = fasttext.get_text_tokens_vector(tokens)

        model = pickle.load(open(self._model_save_path, 'rb'))
        categories = pickle.load(open(self._categories_path, 'rb'))
        predicted_label = model.predict([text_vector])
        return categories[predicted_label][0]
