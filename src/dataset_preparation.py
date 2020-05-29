import os
import pandas as pd
import pickle
from nltk.tokenize.casual import casual_tokenize
from sklearn import model_selection, preprocessing
from typing import List


class DatasetPreparation:

    def __init__(self, version: str, csv_file_name: str = 'data/sportoclanky.csv') -> None:
        self._input_file_name = csv_file_name
        self._version = version

    def prepare_dataset(self, text_column_names: List[str]) -> list:
        """Loads input csv data and prepare the for the feature extractor.

        Text is tokenized and categories encoded.
        :param text_column_names specifies the columns in csv file, that should be concatenated to the 'text' DF column
        """
        tokens_file_path = f'data/tokens_{self._version}.p'
        labels_file_path = f'data/labels_{self._version}.p'
        categories_file_path = 'data/categories.p'

        if (os.path.exists(tokens_file_path)) and (os.path.exists(labels_file_path)):
            tokens = pickle.load(open(tokens_file_path, 'rb'))
            labels = pickle.load(open(labels_file_path, 'rb'))
            categories = pickle.load(open(categories_file_path, 'rb'))

        else:
            data_df = self._load_input_data(text_column_names)
            tokens = data_df['text'].map(lambda x: casual_tokenize(x))

            # encode categories to labels
            encoder = preprocessing.LabelEncoder()
            labels = encoder.fit_transform(data_df['category'])
            categories = list(encoder.classes_)
            pickle.dump(tokens, open(tokens_file_path, 'wb'))
            pickle.dump(labels, open(labels_file_path, 'wb'))

            if not os.path.exists(categories_file_path):
                pickle.dump(categories, open(categories_file_path, 'wb'))

        return [tokens, labels, categories]

    def _load_input_data(self, text_column_names: List[str] = None) -> pd.DataFrame:
        """Loads data from csv file into DataFrame (DF) with category and text column.

        :param text_column_names specifies the columns in csv file, that should be concatenated to the 'text' DF column
        """
        input_data_df = pd.read_csv(self._input_file_name)
        data_df = input_data_df[['category']]
        data_df['text'] = ''

        for text_column in text_column_names:
            data_df.loc[:, 'text'] = data_df['text'] + ' ' + input_data_df[text_column]

        return data_df
