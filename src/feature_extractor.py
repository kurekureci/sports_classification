import numpy as np
import os
import pickle
import pandas as pd


class FeatureExtractor:

    def __init__(self, version: str) -> None:
        self._version = version

    def extract_fasttext_vectors(self, tokens: pd.Series) -> list:
        """Get vectors data representation using fastText."""
        data_vectors_file_path = f'data/data_vectors_{self._version}.p'

        if os.path.exists(data_vectors_file_path):
            data_vectors = pickle.load(open(data_vectors_file_path, 'rb'))
        else:
            fasttext_vectors = self._load_fasttext_vectors('data/fasttext_vectors.p', 'data/cc.cs.300.vec')
            vector_dimension = 300
            data_vectors = list()

            for text in tokens:
                vector_list = list()

                for token in text:

                    if token in fasttext_vectors.keys():
                        vector_list.append(fasttext_vectors.get(token))

                if len(vector_list) == 0:
                    vector_list.append(np.zeros(vector_dimension))

                data_vectors.append(np.average(vector_list, axis=0))

            pickle.dump(data_vectors, open(data_vectors_file_path, 'wb'))

        return data_vectors

    def _load_fasttext_vectors(self, pickle_file_path, vector_file_path):
        """Pre-trained word vectors for czech language, trained on Common Crawl and Wikipedia using fastText.

        Using CBOW with position-weights, dimension 300, character n-grams of length 5, window size 5 and 10 negatives
        """
        if os.path.exists(pickle_file_path):
            vectors = pickle.load(open(pickle_file_path, 'rb'))
        else:
            vectors = dict()

            with open(vector_file_path) as file:
                file.readline()

                for line in file.readlines():
                    items = line.strip().split()
                    vectors[items[0]] = np.array(items[1:], dtype=np.float32)

            pickle.dump(vectors, open(pickle_file_path, 'wb'))

        return vectors
