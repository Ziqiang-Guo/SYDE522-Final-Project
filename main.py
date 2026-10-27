import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

import battery_loader
from dataloader import DataLoader

LABELS = {
    'IR': 'Internal resistance (Ohm)',
    'QC': 'Charge capacity (Ah)',
    'QD': 'Discharge capacity (Ah)',
    'Tavg': 'Average temperature (°C)',
    'Tmin': 'Minimum temperature (°C)',
    'Tmax': 'Maximum temperature (°C)',
    'chargetime': 'Charge time (min)',
}

# one blue, light -> dark: short-lived cells light, long-lived cells dark
CMAP = LinearSegmentedColormap.from_list('life', ['#9ec5f4', '#2a78d6', '#0d366b'])


def plot_dataset(X, y, path='plots/dataset.png'):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), layout='constrained')
    norm = Normalize(y.min(), y.max())

    # one panel per summary feature, one line per cell
    for f, (key, ax) in enumerate(zip(battery_loader.SUMMARY_KEYS, axes.flat)):
        for i in np.argsort(-y):  # longest-lived first so shorter cells stay visible on top
            cell, life = X[i], y[i]
            ax.plot(np.arange(1, len(cell) + 1), cell[:, f], color=CMAP(norm(life)), lw=0.8, alpha=0.8)
        values = np.concatenate([cell[:, f] for cell in X])
        lo, hi = np.nanpercentile(values, [0.5, 99.5])  # ignore a few measurement spikes
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(LABELS[key])
        ax.set_xlabel('Cycle number')

    # last panel: distribution of cycle lives
    ax = axes.flat[-1]
    ax.hist(y, bins=20, color='#2a78d6', edgecolor='white', linewidth=2)
    ax.set_title(f'Cycle life ({len(y)} cells)')
    ax.set_xlabel('Cycle life')
    ax.set_ylabel('Number of cells')

    for ax in axes.flat:
        ax.grid(color='#e0e0e0', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, label='Cycle life (line color)',
                 shrink=0.6, pad=0.01)
    fig.suptitle('Battery dataset: summary features per cycle', fontsize=14)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f'Saved {path}')
    plt.show()


if __name__ == '__main__':
    loader = DataLoader(battery_loader.load)  # all three batches
    print(f'Loaded {len(loader.X)} cells')
    plot_dataset(loader.X, loader.y)
