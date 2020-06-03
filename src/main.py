from imblearn.over_sampling import RandomOverSampler
import pickle

from src.dataset_preparation import DatasetPreparation
from src.fasttext_vectors import FasttextVectors
from src.svm_classification import SVMClassification
from src.lstm_classification import LSTMClassification

MODEL_TYPE = 'SVM'
TRAINING = True
CLASSIFICATION = True

if TRAINING:
    # Data preparation
    preparator = DatasetPreparation()
    input_text_column_names = ['rss_title', 'rss_perex']
    data_df = preparator.prepare_dataset(input_text_column_names)

    data_value_counts = data_df.category.value_counts()
    print(data_value_counts)
    #
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # chart = sns.countplot(data_df['category'])
    # chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
    # plt.show()

    if MODEL_TYPE == 'SVM':
        svm_classificator = SVMClassification('svm_model_resampled_data.p')
        [tokens, labels, categories] = svm_classificator.get_tokens_and_labels(data_df)

        fasttext = FasttextVectors()
        data_vectors = fasttext.get_data_vectors(tokens)
        [train_data, test_data, train_labels, test_labels] = preparator.split_train_test_data(data_vectors, labels)

        # Naive random over-sampling
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        train_data_resampled, train_labels_resampled = ros.fit_resample(train_data, train_labels)
        from collections import Counter
        print(sorted(Counter(train_labels_resampled).items()))

        model = svm_classificator.train_model(train_data_resampled, train_labels_resampled, save_model=True)
        svm_classificator.evaluate_model(test_data, test_labels, categories)

    elif MODEL_TYPE == 'LSTM':
        lstm_classificator = LSTMClassification('lstm_model_bidirectional_without_weights_masking.p')
        [tokens, labels, categories] = lstm_classificator.get_tokens_and_labels(data_df)
        padded_tokens = lstm_classificator.get_padded_sequences(tokens)

        [train_data, test_data, train_labels, test_labels] = preparator.split_train_test_data(padded_tokens, labels)
        categories_weights = lstm_classificator.calculate_category_weights(data_df, categories=categories, weighted=False)
        model = lstm_classificator.train_model([train_data, test_data, train_labels, test_labels],
                                               class_weights=categories_weights, save_model=True)
        lstm_classificator.evaluate_model(test_data, test_labels, categories)

if CLASSIFICATION:
    text = ''

    if MODEL_TYPE == 'SVN':
        svm_classificator = SVMClassification('models/svm_model.p')
        category = svm_classificator.classify(text)
    elif MODEL_TYPE == 'LSTM':
        lstm_classificator = LSTMClassification('models/lstm_model.p')
        category = lstm_classificator.classify(text)

    print(category)
