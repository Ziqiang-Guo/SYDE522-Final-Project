import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error


def cycle_life_errors(life: np.ndarray, log_pred: np.ndarray) -> tuple[float, float]:
    '''
    Metrics of Severson et al. (2019), Table 1, for a model that predicts log10(cycle life):
    RMSE in cycles and mean percent error 100 / n * sum |y_hat - y| / y, both after undoing the log10.
    >>> life = np.array([100., 1000.])
    >>> rmse, percent = cycle_life_errors(life, np.log10([110., 900.]))
    >>> round(rmse, 4), round(percent, 4)
    (71.0634, 10.0)
    '''
    pred = 10 ** np.asarray(log_pred)
    return root_mean_squared_error(life, pred), 100 * mean_absolute_percentage_error(life, pred)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
