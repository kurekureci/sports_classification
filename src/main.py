import pandas as pd

from src.dataset_preparation import DatasetPreparation

preparator = DatasetPreparation()
train_data, valid_data, train_labels, valid_labels = preparator.prepare_dataset(['rss_title', 'rss_perex'])
