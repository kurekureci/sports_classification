import os
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import model_selection, metrics


from dataset_preparation import DatasetPreparation
from feature_extractor import FeatureExtractor

data_version = 'v0'
fe_version = 'v0'

# Data preparation
preparator = DatasetPreparation(data_version)
input_text_column_names = ['rss_title', 'rss_perex']
tokens, labels, categories = preparator.prepare_dataset(['rss_title', 'rss_perex'])

# Feature Extractor - FastText
extractor = FeatureExtractor(fe_version)
data_vectors = extractor.extract_fasttext_vectors(tokens)

# split into training and validation data
train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data_vectors, labels, test_size=0.2)
print(f'Size of train data: {len(train_data)}')
print(f'Size of test data: {len(test_data)}')

# Classification
clf_pipeline = Pipeline([('std', StandardScaler()), ('svm', SVC(kernel='rbf'))])
clf_pipeline.fit(train_data, train_labels)

pickle.dump(clf_pipeline, open('svm_model.p', 'wb'))

predicted_labels = clf_pipeline.predict(test_data)

# Classification model evaluation
print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted_labels)}")
print("Confusion matrix:")
print(metrics.confusion_matrix(test_labels, predicted_labels))
print(metrics.classification_report(test_labels, predicted_labels, target_names=categories))