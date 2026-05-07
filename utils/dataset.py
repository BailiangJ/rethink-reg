import pickle
import random
from itertools import permutations

from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, data, length=None):
        if len(data) < 2:
            raise ValueError('PairDataset requires at least two samples.')

        self.data = data
        self.selected_pairs = []

        # pair every i with a random sample from the rest
        for i in range(len(data) - 1):
            j = random.randint(i + 1, len(data) - 1)
            self.selected_pairs.append((i, j))

        self.selected_pairs.append(
            (len(data) - 1, random.randint(0, len(data) - 2))
        )

        self.pairs = list(permutations(range(len(data)), 2))
        for sel_pair in self.selected_pairs:
            self.pairs.remove(sel_pair)

        random.shuffle(self.pairs)

        self.pairs = self.selected_pairs + self.pairs
        if length is not None:
            if length < 1:
                raise ValueError(f'length must be positive, got {length}.')
            self.pairs = self.pairs[:length]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        index1, index2 = self.pairs[idx]
        # print(index1, index2)
        return self.data[index1], self.data[index2]


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


class IXIBrainDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        file_path = self.data_paths[index]
        img, seg = pkload(file_path)

        item = {'image': img, 'label': seg}

        if self.transform:
            item = self.transform(item)
        return item
