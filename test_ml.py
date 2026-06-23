import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Load the cleaned census dataset once for the whole test module."""
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, "data", "census.csv")
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def processed_train(data):
    """Process a sample of the data for training-based tests."""
    sample = data.sample(n=2000, random_state=42)
    X, y, encoder, lb = process_data(
        sample,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    return X, y, encoder, lb


def test_train_model_returns_random_forest(processed_train):
    """train_model should return a fitted RandomForestClassifier."""
    X, y, _, _ = processed_train
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    # A fitted RandomForest exposes estimators_ after .fit().
    assert hasattr(model, "estimators_")


def test_inference_returns_expected_shape_and_values(processed_train):
    """inference should return one binary prediction per input row."""
    X, y, _, _ = processed_train
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]
    # The label binarizer yields a binary classification problem.
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_returns_floats_in_range():
    """compute_model_metrics should return precision, recall, and F1 in [0, 1]."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0
