import pandas as pd
import re
from sklearn import model_selection
from typing import List


class DatasetPreparation:

    def __init__(self, csv_file_name: str = 'data/sportoclanky.csv') -> None:
        self._input_file_name = csv_file_name

    @staticmethod
    def clean_text(input_text: str) -> str:
        """Cleans the input text from "bad" characters, numbers and converts it to lowercase.

        :return: modified string
        """
        text = input_text.lower()
        text = re.sub(r'\d+', '', text)  # replace digits with nothing
        text = re.sub(r'[/(){}\[\]\|@,;:.]', ' ', text)  # replace the matched string with space
        text = text.replace('  ', ' ').strip(' ')  # get rid of double spaces
        return text

    def prepare_dataset(self, text_column_names: List[str]) -> pd.DataFrame:
        """Loads and cleans input csv data, text is tokenized and categories encoded.

        :param text_column_names specifies the columns in csv file, that should be concatenated to the 'text' DF column
        """
        data_df = self.load_input_data(text_column_names)
        data_df['text'] = data_df['text'].apply(self.clean_text)
        return data_df

    def load_input_data(self, text_column_names: List[str] = None) -> pd.DataFrame:
        """Loads data from csv file into DataFrame with category and text column.

        :param text_column_names specifies the columns in csv file, that should be concatenated to the 'text' DF column
        :return DataFrame with category and text columns
        """
        input_data_df = pd.read_csv(self._input_file_name)
        data_df = input_data_df[['category']]
        data_df['text'] = ''

        for text_column in text_column_names:
            data_df.loc[:, 'text'] = data_df['text'] + ' ' + input_data_df[text_column]

        return data_df

    @staticmethod
    def split_train_test_data(data: pd.Series, labels: pd.Series) -> list:
        """Splits dataset into training and testing data."""
        train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size=0.2,
                                                                                            random_state=1)
        print(f'Size of train data: {len(train_data)}')
        print(f'Size of test data: {len(test_data)}')
        return [train_data, test_data, train_labels, test_labels]
