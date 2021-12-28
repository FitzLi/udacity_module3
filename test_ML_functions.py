import pytest
import numpy as np
from starter.ml.model import train_model, compute_model_metrics, inference

@pytest.fixture()
def data():
    return np.arange(10).reshape(-1, 1), np.array([0] * 5 + [1] * 5)

def test_trained_model_predict(data):
    '''
    Test whether the trained model has 'predict' attribute.
    '''
    X, y = data
    model = train_model(X, y)
    assert hasattr(model, 'predict')

def test_metrics_computation(data):
    '''
    Test whether the computed metrics are correct.
    '''
    _, y = data
    # if preds == y for all elements, all metrics should == 1
    preds = y
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == recall == fbeta == 1
    # if preds != y for all elements, all metrics should == 0
    preds = y[::-1]
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == recall == fbeta == 0

def test_inference_type(data):
    '''
    Test whether the prediction is the type of np.ndarray.
    '''
    X, y = data
    model = train_model(X, y)
    preds = inference(model, X)
    assert isinstance(preds, np.ndarray)
