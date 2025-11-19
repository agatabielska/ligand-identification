from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch


class BlobDataset(Dataset):
    def __init__(self, path: str, transform=None, cache: bool = False):
        self.path = Path(path)
        self.cache = cache
        self.transform = transform
        self.samples = []
        for class_dir in self.path.iterdir():
            if class_dir.is_dir():
                for file in class_dir.glob("*.npz"):
                    self.samples.append((file, int(class_dir.name)))

        if self.cache:
            self.cached_data = []
            for i in tqdm(range(self.__len__()), desc="Loading dataset into cache"):
                self.cached_data.append(self._load_data(i))

    def __len__(self):
        return len(self.samples)

    def _load_data(self, idx):
        file_path, label = self.samples[idx]
        npz = np.load(file_path)

        if self.transform is None:
            idx = npz["indices"]
            vals = npz["values"]
            shape = tuple(npz["shape"])

            A = np.zeros(shape, dtype=vals.dtype)
            A[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
            A = torch.from_numpy(A)
            return A, label
        else:
            return self.transform(npz), label

    def __getitem__(self, idx):
        if self.cache:
            return self.cached_data[idx]
        else:
            return self._load_data(idx)
