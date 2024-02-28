"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import sklearn
from regression import LogisticRegressor
import numpy as np

def test_prediction():
	weights = np.array([0.1, 0.2, -0.1])
	model = LogisticRegressor(num_feats=2)
	model.W = weights 
	X_test = sklearn.preprocessing.add_dummy_feature(np.array([[2, 3], [4, 5]])) #bias
	expected_predictions = 1 / (1 + np.exp(-np.dot(X_test, weights)))
	predictions = model.make_prediction(X_test)
	assert np.all(predictions == expected_predictions), "the predictions do not match the expected values"

def test_loss_function():
	y_true = np.array([0,1,0,1])
	y_pred = np.array([0.2,0.9,0.1,0.8])
	expected = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / len(y_true)
	model = LogisticRegressor(num_feats=2)
	calculated = model.loss_function(y_true, y_pred)
	tol = 1e-6
	assert calculated-expected <= tol, "the calculated and expected losses are different"

def test_gradient():
	model = LogisticRegressor(num_feats=2)
	weights = np.array([0.1, 0.2, -0.1])
	X = sklearn.preprocessing.add_dummy_feature(np.array([[1, 2], [3, 4], [5, 6]]))
	y_true = np.array([0, 1, 0])
	y_pred = model.make_prediction(X)
	grad_exp = np.dot(X.T, (y_pred - y_true)) / len(y_true)
	grad_cal = model.calculate_gradient(y_true, X)
	tol = 1e-6
	assert np.all(grad_cal - grad_exp <= tol), "the calculated and expected gradients are different"

def test_training():
	model = LogisticRegressor(num_feats=2,learning_rate=0.01, tol=0.001, max_iter=10, batch_size=2)
	model.W = np.array([0.1, 0.2, 0.3])
	initial_w = model.W
	X_train = np.array([[1, 1], [2, 2], [3, 3]])
	y_train = np.array([0, 1, 0])
	X_val = np.array([[1, 1], [2, 2]])
	y_val = np.array([0, 1])
	model.train_model(X_train, y_train, X_val, y_val)
	assert np.all(initial_w != model.W), "weights did not update during training"
