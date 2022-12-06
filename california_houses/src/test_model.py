"""
Test of model trained in `model.py`.
"""

import os
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error


def main():
    """Reads test dataset and testlabels and applies prediction."""

    model_path = os.path.join("..", "models", "forest_reg.pkl")
    model = joblib.load(model_path)

    test_set_labels_path = os.path.join("..", "data", "test_set.csv")
    test_set = pd.read_csv(test_set_labels_path).to_numpy()

    test_set_labels_path = os.path.join("..", "data", "test_set_labels.csv")
    test_set_labels = pd.read_csv(test_set_labels_path).to_numpy().flatten()

    final_predictions = model.predict(test_set)
    final_mse = mean_squared_error(test_set_labels, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print("FINAL RMSE:".ljust(25), final_rmse)

    confidence = 0.95
    squared_errors = (final_predictions - test_set_labels) ** 2
    confidence_interval = np.sqrt(
        stats.t.interval(
            confidence,
            len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors)
        )
    )
    print("CONFIDENCE INTERVAL:".ljust(25), confidence_interval)


if __name__ == '__main__':
    main()
