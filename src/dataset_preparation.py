import pandas as pd
from sklearn import model_selection, preprocessing
from typing import List


class DatasetPreparation:

    def __init__(self, csv_file_name: str = 'data/sportoclanky.csv') -> None:
        """Constructor."""
        self._input_file_name = csv_file_name

    def prepare_dataset(self, text_column_names: List[str]) -> List[pd.DataFrame]:

        input_data_df = pd.read_csv(self._input_file_name)
        data_df = input_data_df[['category']]
        data_df['text'] = ''

        for text_column in text_column_names:
            data_df.loc[:, 'text'] = data_df['text'] + ' ' + input_data_df[text_column]

        # split into training and validation data
        train_data, valid_data, train_labels, valid_labels = model_selection.train_test_split(data_df['text'],
                                                                                              data_df['category'])

        # label encode categories
        encoder = preprocessing.LabelEncoder()
        train_labels_encode = encoder.fit_transform(train_labels)
        valid_labels_encode = encoder.fit_transform(valid_labels)

        return [train_data, valid_data, train_labels_encode, valid_labels_encode]