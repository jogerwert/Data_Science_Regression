import pandas as pd
from sklearn import linear_model, metrics
import multiple_lineare_regression as mlr
import timeit


def regression_mit_sklearn():
    boston_housing = pd.read_csv('./boston-housing.data', header=None, sep='\s+')

    y = boston_housing.iloc[:, 13].values
    X = boston_housing.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12]].values

    reg = linear_model.LinearRegression()
    reg.fit(X, y)

    koeffizienten_format = [f'w{index+1}: {round(koeffizient, 5)}' for index, koeffizient in enumerate(reg.coef_)]
    koeffizienten_concat = ', '.join(koeffizienten_format)

    print('Parameter:')
    print('Intercept')
    print(f'w0: {round(reg.intercept_, 5)}')
    print('Koeffizienten')
    print(f'{koeffizienten_concat}')
    print('Bestimmtheitsmaß')
    print(f'R2: {round(metrics.r2_score(y, reg.predict(X)), 5)}')
    print('Mittlerer quadratischer Fehler')
    print(f'MSE: {round(metrics.mean_squared_error(y, reg.predict(X)), 5)}')


def regression_selbstgemacht():
    boston_housing = pd.read_csv('./boston-housing.data', header=None, sep='\s+')

    y = boston_housing.iloc[:, 13].values
    training_data = boston_housing.iloc[:, [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12]].values
    X = mlr.pad_training_data(training_data)

    coef = mlr.berechne_parameter(X, y)
    koeffizienten_format = [f'w{index}: {round(koeffizient, 5)}' for index, koeffizient in enumerate(coef)]
    koeffizienten_concat = ', '.join(koeffizienten_format)

    print('Koeffizienten')
    print(f'{koeffizienten_concat}')
    print('Bestimmtheitsmaß')
    print(f'R2: {round(mlr.r2_score(y, mlr.predict(X, coef)), 5)}')
    print('Mittlerer quadratischer Fehler')
    print(f'MSE: {round(mlr.mean_squared_error(y, mlr.predict(X, coef)), 5)}')


def start():
    print('Regression mit SKlearn')
    start_sklearn = timeit.default_timer()
    regression_mit_sklearn()
    stop_sklearn = timeit.default_timer()
    print(f'Ausführungsdauer: {stop_sklearn - start_sklearn}')

    print()

    print('Regression selbstgemacht')
    start_selbstgemacht = timeit.default_timer()
    regression_selbstgemacht()
    stop_selbstgemacht = timeit.default_timer()
    print(f'Ausführungsdauer: {stop_selbstgemacht - start_selbstgemacht}')


if __name__ == '__main__':
    start()
