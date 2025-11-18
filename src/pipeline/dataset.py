from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class BlobDataset(Dataset):
    def __init__(self, path: str, transform=None):
        self.path = Path(path)
        self.transform = transform
        self.samples = []
        for class_dir in self.path.iterdir():
            if class_dir.is_dir():
                for file in class_dir.glob("*.npz"):
                    self.samples.append((file, int(class_dir.name)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        npz = np.load(file_path)

        if self.transform is None:
            idx = npz["indices"]
            vals = npz["values"]
            shape = tuple(npz["shape"])

            A = np.zeros(shape, dtype=vals.dtype)
            A[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
            return A, label
        else:
            return self.transform(npz), label
