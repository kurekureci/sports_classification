from sklearn import metrics


class ModelEvaluation:

    @staticmethod
    def evaluate_model(test_labels, predicted_labels, categories):
        """Prints model accuracy, confusion matrix and classification report."""
        print(f"Accuracy: {metrics.accuracy_score(test_labels, predicted_labels)}")
        print("Confusion matrix:")
        print(metrics.confusion_matrix(test_labels, predicted_labels))
        print(metrics.classification_report(test_labels, predicted_labels, target_names=categories))

