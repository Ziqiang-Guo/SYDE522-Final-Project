import gc
import os
import pickle

import numpy as np
from scipy.signal import medfilt
from scipy.stats import kurtosis, skew

# Same cleaning as battery_loader.load, but one batch is unpickled at a time so the
# ~8 GB of raw data never sits in memory at once.
REMOVED = {1: ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22'],
           2: ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16'],  # carried over into batch 1 cells below
           3: ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']}
CARRIED_OVER = {'b1c0': 662, 'b1c1': 981, 'b1c2': 1060, 'b1c3': 208, 'b1c4': 482}

N_CYCLES = 100  # only the first 100 cycles are used ("early prediction")
CACHE = './Data/early_cycles.npz'


def _extract_cell(cell):
    '''Raw early-cycle arrays of one cell. Summary index i is cycle i + 1; cycles key str(i) is cycle i + 1.'''
    s, cycles = cell['summary'], cell['cycles']
    assert len(s['QD']) >= N_CYCLES, 'cell has fewer than 100 cycles'
    T_integral = np.array([np.trapezoid(cycles[str(i)]['T'], cycles[str(i)]['t'])
                           for i in range(N_CYCLES)])
    return dict(
        QD=s['QD'][:N_CYCLES], IR=s['IR'][:N_CYCLES], chargetime=s['chargetime'][:N_CYCLES],
        Tmax=s['Tmax'][:N_CYCLES], T_integral=T_integral,
        Qdlin10=cycles['9']['Qdlin'], Qdlin100=cycles['99']['Qdlin'],
    )


def build_cache(data_dir='./Data', path=CACHE):
    raw, cycle_life, batch = [], [], []
    for b in (1, 2, 3):
        data = pickle.load(open(f'{data_dir}/batch{b}.pkl', 'rb'))
        for key in REMOVED[b]:
            del data[key]
        for key, cell in data.items():
            raw.append(_extract_cell(cell))
            cycle_life.append(float(np.squeeze(cell['cycle_life'])) + CARRIED_OVER.get(key, 0))
            batch.append(b)
        del data
        gc.collect()
        print(f'batch {b}: {sum(x == b for x in batch)} cells')

    arrays = {k: np.stack([r[k] for r in raw]) for k in raw[0]}
    np.savez(path, cycle_life=np.array(cycle_life), batch=np.array(batch), **arrays)
    print(f'Saved {path}')


def load_raw(path=CACHE):
    if not os.path.exists(path):
        build_cache(path=path)
    return dict(np.load(path))


def compute_features(raw):
    '''
    Candidate early-life features from Severson et al. (2019), Nature Energy, all from cycles <= 100.
    Returns (F, names): F is (n_cells, n_features).
    '''
    dQ = raw['Qdlin100'] - raw['Qdlin10']  # Delta Q_{100-10}(V), 1000 voltage points
    # a few cells have single-cycle measurement glitches (e.g. QD = 2.9 Ah), so smooth per-cycle curves
    smooth = lambda A: np.stack([medfilt(row, 5) for row in A])
    QD, IR, T_integral = smooth(raw['QD']), raw['IR'], smooth(raw['T_integral'])
    # one cell logs time in seconds instead of minutes (integral ~60x the others)
    T_integral = np.where(np.median(T_integral, axis=1, keepdims=True) > 2e4, T_integral / 60, T_integral)
    cyc = np.arange(2, N_CYCLES + 1)  # cycles 2..100 (cycle 1 is skipped as in the paper)
    fade = np.polyfit(cyc, QD[:, 1:].T, 1)  # (2, n_cells): slope, intercept
    late = np.polyfit(np.arange(91, 101), QD[:, 90:].T, 1)

    feats = {
        # Delta Q(V) features
        'log|min dQ|': np.log10(np.abs(dQ.min(axis=1))),
        'log|mean dQ|': np.log10(np.abs(dQ.mean(axis=1))),
        'log var dQ': np.log10(dQ.var(axis=1)),
        'log|skew dQ|': np.log10(np.abs(skew(dQ, axis=1))),
        'log|kurt dQ|': np.log10(np.abs(kurtosis(dQ, axis=1))),
        'log|dQ(2V)|': np.log10(np.abs(dQ[:, -1])),
        # discharge capacity fade curve features
        'QD cycle 2': QD[:, 1],
        'QD max - QD 2': QD[:, 1:].max(axis=1) - QD[:, 1],
        'QD 100 - QD 2': QD[:, 99] - QD[:, 1],
        'fade slope 2-100': fade[0],
        'fade intercept 2-100': fade[1],
        'fade slope 91-100': late[0],
        # other features
        'charge time 1-5': raw['chargetime'][:, :5].mean(axis=1),
        'T integral 2-100': T_integral[:, 1:].sum(axis=1),
        'Tmax max 2-100': raw['Tmax'][:, 1:].max(axis=1),
        'IR min 2-100': np.where(IR[:, 1:] > 0, IR[:, 1:], np.inf).min(axis=1),
        'IR 100 - IR 2': IR[:, 99] - IR[:, 1],
    }
    return np.column_stack(list(feats.values())), list(feats)


# Feature subsets of the three models in the paper
PAPER_MODELS = {
    'variance': ['log var dQ'],
    'discharge': ['log|min dQ|', 'log var dQ', 'log|skew dQ|', 'log|kurt dQ|',
                  'QD cycle 2', 'QD max - QD 2'],
    'full': ['log|min dQ|', 'log var dQ', 'fade slope 2-100', 'fade intercept 2-100', 'QD cycle 2',
             'charge time 1-5', 'T integral 2-100', 'IR min 2-100', 'IR 100 - IR 2'],
}


def paper_split(batch):
    '''
    Train / primary test / secondary test indices of the paper (41 / 43 / 40 cells): batches 1 and 2
    are interleaved (odd -> train, even and the last cell -> primary test), batch 3 is the secondary test set.
    '''
    idx = np.flatnonzero(batch != 3)
    return idx[1:-1:2], np.append(idx[0::2], idx[-1]), np.flatnonzero(batch == 3)


if __name__ == '__main__':
    raw = load_raw()
    F, names = compute_features(raw)
    print(F.shape)
    for n, col in zip(names, F.T):
        print(f'{n:22s} min {col.min():10.4g}  max {col.max():10.4g}  finite {np.isfinite(col).all()}')
