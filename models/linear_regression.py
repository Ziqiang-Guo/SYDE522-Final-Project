import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


class LinearRegression:
    '''
    Linear basis function regression fitted by least squares with the pseudo-inverse:

        w = pinv(Phi) @ y
    '''
    def __init__(self):
        self.w = None

    def design_matrix(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        Phi = self.design_matrix(X)
        self.w = np.linalg.pinv(Phi) @ y
        return self

    def regularized_fit(self, X: np.ndarray, y: np.ndarray, lam: np.ndarray,
                        cross_validation: bool = False, folds=None):
        if cross_validation:
            if folds is None:
                folds = KFold(n_splits=10, shuffle=True, random_state=0)
            self.cv_train_rms, self.cv_val_rms = self.cross_validate(X, y, lam, folds)

        self._ridge_solve(X, y, lam)
        return self

    def _ridge_solve(self, X: np.ndarray, y: np.ndarray, lam: np.ndarray):
        Phi = self.design_matrix(X)
        A = Phi.T @ Phi + np.diag(lam)
        self.w = np.linalg.pinv(A) @ (Phi.T @ y)

    def cross_validate(self, X: np.ndarray, y: np.ndarray, lam: np.ndarray, folds):
        train_rms, val_rms = [], []
        for train_idx, val_idx in folds.split(X):
            self._ridge_solve(X[train_idx], y[train_idx], lam)
            train_rms.append(self.evaluate(X[train_idx], y[train_idx]))
            val_rms.append(self.evaluate(X[val_idx], y[val_idx]))
        return float(np.mean(train_rms)), float(np.mean(val_rms))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.design_matrix(X) @ self.w

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        return root_mean_squared_error(y, self.predict(X))


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
