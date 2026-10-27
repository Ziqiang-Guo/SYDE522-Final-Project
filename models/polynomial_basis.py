from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.linear_regression import LinearRegression

class PolynomialBasis(LinearRegression):
    '''
    Linear regression on polynomial basis functions of a single variable:
    Phi = [1, x_1, x_1^2, ..., x_1^degree, x_2, x_2^2, ...], with no cross-terms (x_1 * x_2).
    '''
    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        '''
        X: (N, d) -> Phi: (N, 1 + d * degree), columns [1, x1, x1^2, x1^3, x2, x2^2, x2^3]
        Note: No cross term implemented
        >>> model = PolynomialBasis(degree=3)
        >>> X = np.array([[1., 2.],
        ...               [3., 4.]])
        >>> Phi = model.design_matrix(X)
        >>> Phi.shape
        (2, 7)
        >>> Phi
        array([[ 1.,  1.,  1.,  1.,  2.,  4.,  8.],
               [ 1.,  3.,  9., 27.,  4., 16., 64.]])
        >>> PolynomialBasis(degree=1).design_matrix(X)
        array([[1., 1., 2.],
               [1., 3., 4.]])
        '''
        N, d = X.shape
        cols = [np.ones(N)] + [X[:, j]**k for j in range(d) for k in range(1, self.degree + 1)]
        return np.column_stack(cols)


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
