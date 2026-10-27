import numpy as np
import pickle

SUMMARY_KEYS = ['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime']


def load(data_dir='./Data', batches=(1, 2, 3)):
    '''
    Load the battery data, same steps as "Load Data.ipynb".

    Returns X, y:
        X: list with one array per cell, shape (num_cycles, 7), the summary features in SUMMARY_KEYS
        y: array of cycle lives
    '''
    batch1 = pickle.load(open(data_dir + '/batch1.pkl', 'rb'))
    #remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    batch2 = pickle.load(open(data_dir + '/batch2.pkl','rb'))

    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482];

    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']

    batch3 = pickle.load(open(data_dir + '/batch3.pkl','rb'))
    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c42']
    del batch3['b3c43']

    bat_dict = {**batch1, **batch2, **batch3}

    # keep only the requested batches ('b1c0' -> batch 1)
    cells = [bat_dict[k] for k in bat_dict if int(k[1]) in batches]
    X = [np.stack([cell['summary'][k] for k in SUMMARY_KEYS], axis=1).astype(np.float32) for cell in cells]
    y = np.array([float(np.squeeze(cell['cycle_life'])) for cell in cells])
    return X, y
