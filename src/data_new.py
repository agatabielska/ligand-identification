import collections
import os
import random
from abc import ABC, abstractmethod
from copy import deepcopy
from random import seed
from typing import List, Tuple

import MinkowskiEngine as ME
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from pathlib import Path


class BlobDataset(Dataset):
    """ Adam compatible dataset class """
    
    def __init__(
        self,
        path: str,
        transform=None,
        normalize=False,
        cache: bool = False,
        num_workers: int = 4,
        label_to_idx: dict = None,
    ):
        self.path = Path(path)
        self.cache = cache
        self.transform = transform
        self.normalize = normalize
        self.samples = []
        self.num_workers = num_workers

        
        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
            class_dirs = sorted([d for d in self.path.iterdir() if d.is_dir() and d.name in label_to_idx])
        else:
            #TODO this is buggy part
            class_dirs = sorted([d for d in self.path.iterdir() if d.is_dir()])
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
        m = np.max(np.linalg.norm(pc, axis=1))
        return pc / m

    def _load_data(self, idx):
        file_path, label = self.samples[idx]
        npz = np.load(file_path)

        # Convert to dict so that we can modify it in place
        npz = dict(npz)

        if self.normalize:
            npz["indices"] = self.pc_normalize(npz["indices"])

        idx = npz["indices"]
        vals = npz["values"]
        shape = tuple(npz["shape"])

        blob = np.zeros(shape, dtype=np.float32)
        blob[idx[:, 0], idx[:, 1], idx[:, 2]] = vals
        blob = torch.from_numpy(blob)
        
        coordinates, features = self._get_coords_feats(blob)
        label = torch.tensor(label, dtype=torch.float32) 
        return coordinates, features, label
        
    def _get_coords_feats(self, batch: torch.Tensor) -> tuple:
        coordinates = torch.nonzero(batch).long()
        if coordinates.numel() == 0:
            return coordinates, torch.empty((0, 1), dtype=torch.float32)

        values = batch[tuple(coordinates.t())]
        features = values.unsqueeze(-1)
        return coordinates, features

    def __getitem__(self, idx):
        if self.cache:
            return self.cached_data[idx]
        else:
            return self._load_data(idx)


def collation_fn_contiguous(blobel: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements collation function for batching torch tensors and adding padding (-1, -1, -1), where needed, to make
    the CoordsDataset working with blobels of different size and `torch.utils.data.DataLoader`.

    :param blobel: tuple of blo(b) + (la)bel => blobel
    :type blobel: List[Tuple[torch.Tensor, torch.Tensor]]
    """
    np.random.seed(23)
    coordinates = []
    labels = []
    max_len = 0
    for c, l in blobel:
        max_len = max(max_len, len(c))
        coordinates.append(c)
        labels.append(l)
    for idx, c in enumerate(coordinates):
        choose = np.random.choice(len(c), max_len, replace=True)
        coordinates[idx] = coordinates[idx][choose]

    coordinates = torch.stack(coordinates)
    labels = torch.stack(labels)
    return coordinates, labels


def collation_fn_sparse(blobel):
    """
    Implements collation function for batching the LigandsDataset using
    `torch.utils.data.DataLoader`

    :param blobel: tuple (coordinates, features, labels); blo(b)+(la)bel => blobel; all credit to Witek T.
    """
    coords_batch, feats_batch, labels_batch = [], [], []

    for (coords, feats, label) in blobel:
        coords_batch.append(coords)
        feats_batch.append(feats)
        labels_batch.append(label)

    coords_batch = ME.utils.batched_coordinates(coords_batch)
    feats_batch = torch.cat(feats_batch, 0)
    labels_batch = torch.stack(labels_batch)

    return coords_batch, feats_batch, labels_batch
