import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split


class SequenceDataset(Dataset):
    '''One sample = (sequence of shape (L, F), target).'''
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i], dtype=torch.float32)


class Collator:
    '''Pads sequences of different lengths to the longest one in the batch.'''
    def __call__(self, batch):
        xs, ys = zip(*batch)
        lengths = torch.tensor([len(x) for x in xs])
        return pad_sequence(xs, batch_first=True), lengths, torch.stack(ys)


class DataLoader:
    '''
    Load a dataset with its dataset loader and split it into training and test sets.

    >>> def toy_load():
    ...     X = [np.ones((n, 2), dtype=np.float32) for n in range(3, 13)]
    ...     y = np.arange(10.)
    ...     return X, y
    >>> loader = DataLoader(toy_load)
    >>> len(loader.X), loader.y.shape
    (10, (10,))

    >>> loader.train_test_split(loader.X, loader.y, 6, False)
    >>> X_train, y_train = loader.get_train()
    >>> X_test, y_test = loader.get_test()
    >>> len(X_train), len(X_test)
    (6, 4)

    >>> x, lengths, y = next(iter(loader.get_test_loader(batch_size=4)))
    >>> tuple(x.shape), lengths.tolist(), y.tolist()
    ((4, 12, 2), [9, 10, 11, 12], [6.0, 7.0, 8.0, 9.0])
    '''
    def __init__(self, dataset_loader, **kwargs):
        '''dataset_loader: a function returning X, y, e.g. battery_loader.load'''
        self.X, self.y = dataset_loader(**kwargs)

    def train_test_split(self, X, y, train_size, shuffle: bool, random_state=None):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, train_size=train_size, shuffle=shuffle, random_state=random_state)

    def get_train(self):
        return self.X_train, self.y_train

    def get_test(self):
        return self.X_test, self.y_test

    def get_train_loader(self, batch_size=8):
        return TorchDataLoader(SequenceDataset(self.X_train, self.y_train),
                               batch_size=batch_size, shuffle=True, collate_fn=Collator())

    def get_test_loader(self, batch_size=8):
        return TorchDataLoader(SequenceDataset(self.X_test, self.y_test),
                               batch_size=batch_size, shuffle=False, collate_fn=Collator())


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
