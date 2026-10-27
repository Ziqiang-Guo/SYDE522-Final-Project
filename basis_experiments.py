import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedKFold

from features import PAPER_MODELS, compute_features, load_raw, paper_split
from metrics.cycle_life import cycle_life_errors
from models.gaussian_basis import GaussianBasis
from models.polynomial_basis import PolynomialBasis

BLUE, ORANGE, INK, GRID = '#2a78d6', '#eb6834', '#555555', '#e0e0e0'
LAMBDAS = np.logspace(-3, 2, 11)
FOLDS = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)


class Standardizer:
    '''z-score every input feature with the mean / std of the training cells only.'''
    def fit(self, X):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std


def ridge_penalty(lam, n_cols):
    '''Same lambda on every weight except the bias column.'''
    lam_vec = np.full(n_cols, lam)
    lam_vec[0] = 0.
    return lam_vec


def poly_configs(X):
    for degree, lam in itertools.product((1, 2, 3), LAMBDAS):
        yield dict(degree=degree, lam=lam), PolynomialBasis(degree)


def gauss_configs(X):
    n, d = X.shape
    for k, a, lam in itertools.product((10, 20, n), (-0.03, -0.1, -0.3, -1.), LAMBDAS):
        # a is divided by d so the bump width is comparable for any number of (standardized) features
        yield dict(n_centers=k, a=a, lam=lam), GaussianBasis(X, n_centers=k, a=a / d)


def tune(configs, X, y):
    '''Pick the config with the lowest cross-validated RMSE (log10 cycle life) on the training cells.'''
    best = None
    for params, model in configs(X):
        lam = ridge_penalty(params['lam'], model.design_matrix(X[:1]).shape[1])
        model.regularized_fit(X, y, lam, cross_validation=True, folds=FOLDS)
        if best is None or model.cv_val_rms < best[2].cv_val_rms:
            best = (params, lam, model)
    return best


def cycle_errors(model, X, life):
    '''RMSE (cycles) and mean absolute percent error, after undoing the log10 transform.'''
    return cycle_life_errors(life, model.predict(X))


def forward_selection(F, y, names, max_features=8):
    '''Greedily add the feature that lowers the polynomial-basis CV error most; stop when nothing helps.'''
    chosen, best_cv = [], np.inf
    while len(chosen) < max_features:
        trials = []
        for j in set(range(F.shape[1])) - set(chosen):
            cols = chosen + [j]
            trials.append((tune(poly_configs, F[:, cols], y)[2].cv_val_rms, j))
        cv, j = min(trials)
        if cv >= best_cv:
            break
        chosen.append(j)
        best_cv = cv
        print(f'  + {names[j]:22s} CV RMSE (log10) {cv:.4f}')
    return [names[j] for j in chosen]


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def plot_screening(screen, path='plots/feature_screening.png'):
    screen = sorted(screen, key=lambda r: -r[1])  # best at the top
    names = [r[0] for r in screen]
    pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
    ax.barh(pos + 0.2, [r[1] for r in screen], height=0.38, color=BLUE, label='Polynomial basis')
    ax.barh(pos - 0.2, [r[2] for r in screen], height=0.38, color=ORANGE, label='Gaussian basis')
    ax.set_yticks(pos, names)
    ax.set_xlabel('Cross-validated RMSE of log10(cycle life), training cells (lower is better)')
    ax.set_title('Single-feature screening: early-life features (cycles ≤ 100)')
    ax.legend(frameon=False, loc='upper right')
    style(ax)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


def plot_single_feature_fits(F_tr, y_tr, names, features, path='plots/basis_fits.png'):
    '''What each basis learns from one standardized feature: data and fitted curve.'''
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4.2), layout='constrained')
    for ax, name in zip(axes, features):
        x = F_tr[:, [names.index(name)]]
        grid = np.linspace(x.min(), x.max(), 300)[:, None]
        ax.scatter(x, y_tr, s=30, color=INK, alpha=0.6, edgecolor='white', linewidth=1, label='Training cells')
        for configs, color, label in ((poly_configs, BLUE, 'Polynomial'), (gauss_configs, ORANGE, 'Gaussian')):
            params, _, model = tune(configs, x, y_tr)
            ax.plot(grid, model.predict(grid), color=color, lw=2, label=f'{label} fit')
        ax.set_xlabel(f'{name} (standardized)')
        ax.set_ylabel('log10(cycle life)')
        ax.set_title(name)
        style(ax)
    axes[0].legend(frameon=False)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


def plot_pred_vs_obs(model, sets, title, path='plots/pred_vs_obs.png'):
    fig, axes = plt.subplots(1, len(sets), figsize=(5 * len(sets), 4.6), layout='constrained', sharey=True)
    lo, hi = 100, 2400
    for ax, (label, X, life) in zip(axes, sets):
        rmse, mape = cycle_errors(model, X, life)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls='--')
        ax.scatter(life, 10 ** model.predict(X), s=36, color=BLUE, edgecolor='white', linewidth=1)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Observed cycle life')
        ax.set_title(f'{label}\nRMSE {rmse:.0f} cycles, error {mape:.1f}%')
        style(ax)
    axes[0].set_ylabel('Predicted cycle life')
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')


if __name__ == '__main__':
    raw = load_raw()
    F, names = compute_features(raw)
    life = raw['cycle_life']
    train, test1, test2 = paper_split(raw['batch'])
    # the paper also reports the primary test without its one outlier cell (148 cycles)
    test1_clean = test1[life[test1] > 200]

    scaler = Standardizer().fit(F[train])
    Z = scaler.transform(F)  # every input normalized with training statistics
    y = np.log10(life)
    print(f'{len(train)} train, {len(test1)} primary test, {len(test2)} secondary test cells')

    # 1. single-feature screening (training CV only)
    print('\n== Single-feature screening (CV RMSE of log10 life) ==')
    screen = []
    for j, name in enumerate(names):
        x = Z[train][:, [j]]
        poly, gauss = tune(poly_configs, x, y[train]), tune(gauss_configs, x, y[train])
        corr = np.corrcoef(F[train, j], y[train])[0, 1]
        screen.append((name, poly[2].cv_val_rms, gauss[2].cv_val_rms))
        print(f'{name:22s} corr {corr:+.2f}   poly {poly[2].cv_val_rms:.4f} (deg {poly[0]["degree"]})'
              f'   gauss {gauss[2].cv_val_rms:.4f}')
    os.makedirs('plots', exist_ok=True)
    plot_screening(screen)
    top3 = [r[0] for r in sorted(screen, key=lambda r: min(r[1], r[2]))[:3]]
    plot_single_feature_fits(Z[train], y[train], names, top3)

    # 2. feature sets: the paper's three models + greedy forward selection
    print('\n== Forward selection (polynomial basis, training CV) ==')
    feature_sets = dict(PAPER_MODELS)
    feature_sets['forward selected'] = forward_selection(Z[train], y[train], names)

    print('\n== Test results (RMSE in cycles / mean % error) ==')
    print(f'{"features":17s} {"basis":10s} {"params":34s} {"CV log":>7s} {"train":>12s} '
          f'{"primary":>12s} {"prim. excl":>12s} {"secondary":>12s}')
    best = None
    for set_name, feats in feature_sets.items():
        cols = [names.index(f) for f in feats]
        for basis, configs in (('polynomial', poly_configs), ('gaussian', gauss_configs)):
            params, _, model = tune(configs, Z[train][:, cols], y[train])
            errs = [cycle_errors(model, Z[idx][:, cols], life[idx]) for idx in (train, test1, test1_clean, test2)]
            p = ', '.join(f'{k}={v:.3g}' for k, v in params.items())
            print(f'{set_name:17s} {basis:10s} {p:34s} {model.cv_val_rms:7.4f} '
                  + ' '.join(f'{r:5.0f} /{m:5.1f}%' for r, m in errs))
            if best is None or model.cv_val_rms < best[0].cv_val_rms:  # model choice uses training CV only
                best = (model, cols, f'{set_name} features, {basis} basis')

    model, cols, title = best
    print(f'\nBest by training CV: {title}')
    plot_pred_vs_obs(model, [('Train', Z[train][:, cols], life[train]),
                             ('Primary test', Z[test1][:, cols], life[test1]),
                             ('Secondary test', Z[test2][:, cols], life[test2])],
                     f'Best model by training CV: {title}')
