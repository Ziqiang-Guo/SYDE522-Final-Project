import numpy as np

from models.linear_regression import LinearRegression

class GaussianBasis(LinearRegression):
    '''
    Linear regression on Gaussian basis functions centred on chosen data points:
    phi_k(x, c_k, a) = exp(a * ||x - c_k||^2), with a < 0, and
    Phi = [1, phi_1(x), ..., phi_K(x)]. A more negative a gives narrower bumps.
    '''
    def __init__(self, X: np.ndarray, n_centers: int, a: float = -1.0, seed: int = 0):
        super().__init__()
        X = np.atleast_2d(np.asarray(X))
        perm = np.random.default_rng(seed).permutation(len(X))
        self.centers = X[perm[:n_centers]]
        self.a = float(a)

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        '''
        X: (N, d) -> Phi: (N, 1 + K), columns [1, phi_1(x), ..., phi_K(x)]
        >>> model = GaussianBasis(np.array([[0., 0.], [1., 1.]]), n_centers=2)
        >>> model.centers
        array([[0., 0.],
               [1., 1.]])
        >>> Phi = model.design_matrix(np.array([[0., 0.]]))
        >>> Phi.shape
        (1, 3)
        >>> Phi.round(4)
        array([[1.    , 1.    , 0.1353]])
        '''
        X = np.atleast_2d(np.asarray(X))[:, None, :]
        centers = self.centers[None, :, :]
        sq_distances = ((X - centers) ** 2).sum(axis=-1)
        Phi_basis = np.exp(self.a * sq_distances)
        bias_column = np.ones((X.shape[0], 1))
        Phi = np.hstack([bias_column, Phi_basis])
        return Phi


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
