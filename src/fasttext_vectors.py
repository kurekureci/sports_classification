import numpy as np
import os
import pandas as pd
import pickle


class FasttextVectors:

    def __init__(self) -> None:
        self._vector_dim = 300
        self._vector_file_path = 'data/cc.cs.300.vec'
        self._vector_dict_file_path = 'data/fasttext_vectors.p'
        self._fasttext_vectors = self._load_fasttext_vectors_to_dict()

    def get_text_tokens_vector(self, tokens: list) -> np.array:
        """Convert input text tokens to vector representation using fastText."""
        vector_list = list()

        for token in tokens:

            if token in self._fasttext_vectors.keys():
                vector_list.append(self._fasttext_vectors.get(token))
            # else:
            #     print(token)

        if len(vector_list) == 0:
            vector_list.append(np.zeros(self._vector_dim))

        return np.average(vector_list, axis=0)

    def get_data_vectors(self, data_tokens: pd.Series) -> list:
        """Get vectors representation from input data using fastText."""
        data_vectors = list()

        for text_tokens in data_tokens:
            vector_list = self.get_text_tokens_vector(text_tokens)
            data_vectors.append(vector_list)

        return data_vectors

    def _load_fasttext_vectors_to_dict(self):
        """Pre-trained word vectors for czech language, trained on Common Crawl and Wikipedia using fastText.

        Using CBOW with position-weights, dimension 300, character n-grams of length 5, window size 5 and 10 negatives
        """
        if os.path.exists(self._vector_dict_file_path):
            vectors = pickle.load(open(self._vector_dict_file_path, 'rb'))
        else:
            vectors = dict()

            with open(self._vector_file_path) as file:
                file.readline()

                for line in file.readlines():
                    items = line.strip().split()
                    vectors[items[0]] = np.array(items[1:], dtype=np.float32)

            pickle.dump(vectors, open(self._vector_dict_file_path, 'wb'))

        return vectors


if __name__ == '__main__':
    from gensim.models import Word2Vec
    model = Word2Vec.load_word2vec_format('cc.cs.300.bin', binary=True)

