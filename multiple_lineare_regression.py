import numpy as np
import pandas as pd


def berechne_parameter(X, y):

    # w = (X^T * X)^-1 * X^T * y
    return np.matmul(np.matmul(np.linalg.inv((np.matmul(np.transpose(X), X))), np.transpose(X)), y)


def predict(X, w):
    return np.matmul(X, w)


def r2_score(y, y_predict):
    y_mittel = (1 / len(y)) * sum(y)

    y_zip = list(zip(y, y_predict))

    dividend = [(y_i - y_predict_i)**2 for y_i, y_predict_i in y_zip]
    divisor = [(y_i - y_mittel)**2 for y_i in y]

    return 1 - (sum(dividend) / sum(divisor))


def mean_squared_error(y, y_predict):
    # L(w) = (y - X*w)^T * (y - X*w)
    return (1 / len(y)) * np.matmul(np.transpose(np.subtract(y, y_predict)), np.subtract(y, y_predict))


def pad_training_data(training_data):
    rows, cols = training_data.shape
    return np.c_[np.ones(rows), training_data]
