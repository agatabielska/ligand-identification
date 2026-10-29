from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch


class BlobDataset(Dataset):
    def __init__(
        self,
        path: str,
        transform=None,
        global_transform=None,
        normalize=False,
        cache: bool = False,
        num_workers: int = 8,
        label_to_idx: dict = None,
    ):
        self.path = Path(path)
        self.cache = cache
        self.transform = transform
        self.global_transform = global_transform
        self.normalize = normalize
        self.samples = []
        self.num_workers = num_workers

        class_dirs = sorted([d for d in self.path.iterdir() if d.is_dir()])
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
        else:
            self.label_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            label = self.label_to_idx[class_dir.name]
            for file in class_dir.glob("*.npz"):
                self.samples.append((file, label))
        if self.cache:
            self.cached_data = []
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.cached_data = list(
                    tqdm(
                        executor.map(self._load_data, range(self.__len__())),
                        total=self.__len__(),
                        desc="Loading dataset into cache",
                    )
                )

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def pc_normalize(pc: np.ndarray) -> np.ndarray:
        """Normalize the coordinates of the blob to be centered at the origin and fit within a unit sphere."""
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        max_dist_from_center = np.max(np.linalg.norm(pc, axis=1))
        # A zero-radius blob has no meaningful direction or scale. Returning
        # zero coordinates preserves its finite shape and lets the model see
        # its density values without dividing by zero.
        return pc / max_dist_from_center if max_dist_from_center > 1e-12 else np.zeros_like(pc)

    def _load_data(self, idx):
        file_path, label = self.samples[idx]
        npz = np.load(file_path)

        # Convert to dict so that we can modify it in place
        npz = dict(npz)
        
        g = self.global_transform(npz) if self.global_transform else None

        if self.normalize:
            npz["indices"] = self.pc_normalize(npz["indices"])

        if self.transform is None:
            idx = npz["indices"]
            vals = npz["values"]
            shape = tuple(npz["shape"])

            x = np.zeros(shape, dtype=vals.dtype)
            x[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
            x = torch.from_numpy(x)
            return (x, label, g) if g is not None else (x, label)
        else:
            x = self.transform(npz)
            return (x, label, g) if g is not None else (x, label)

    def __getitem__(self, idx):
        if self.cache:
            return self.cached_data[idx]
        else:
            return self._load_data(idx)
