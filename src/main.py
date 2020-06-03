from imblearn.over_sampling import RandomOverSampler
import pickle

from src.dataset_preparation import DatasetPreparation
from src.fasttext_vectors import FasttextVectors
from src.svm_classification import SVMClassification
from src.lstm_classification import LSTMClassification

MODEL_TYPE = 'LSTM'
TRAINING = False
CLASSIFICATION = True

if TRAINING:
    # Data preparation
    preparator = DatasetPreparation()
    input_text_column_names = ['rss_title', 'rss_perex']
    data_df = preparator.prepare_dataset(input_text_column_names)
    data_value_counts = data_df.category.value_counts()
    print(data_value_counts)

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
    if MODEL_TYPE == 'SVN':
        classificator = SVMClassification('models/svm_model.p')
    elif MODEL_TYPE == 'LSTM':
        classificator = LSTMClassification('models/lstm_model_bidirectional_without_weights.p')

    with open('data/example_texts.txt') as f:
        example_texts = list(f)

    text = example_texts[0]
    category = classificator.classify(text)
    print(category)
