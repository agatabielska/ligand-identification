# Adapted from https://github.com/jkarolczak/ligand-classification/blob/8ca4ffd678382d8084c0a162333692b74c985471/src/pipeline/transforms.py#L146
from abc import ABC, abstractmethod
from typing import Dict, Union, Callable
import numpy as np
import itertools
import math


class Transform(ABC):
    """
    Abstract class for a transformation that can be applied to a blob.
    """
    @abstractmethod
    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        """
        Abstract method for applying preprocessing methods to a blob
        :param blob: blob to be processed
        """
        pass


class RandomSelectionTransform(Transform):
    """
    A class that limit voxels in the blob by drawing random non-zero voxels.
    """
    def __init__(self, max_blob_size: int):
        self.max_blob_size = max_blob_size

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        if self.max_blob_size is None:
            raise ValueError("max_blob_size has to be specified for RandomSelectionTransform")
        non_zeros = blob.nonzero()
        if non_zeros[0].shape[0] <= self.max_blob_size:
            return blob

        indices_mask = np.array(range(non_zeros[0].shape[0]))
        indices_mask = np.random.choice(indices_mask, size=self.max_blob_size, replace=False)
        x = non_zeros[0][indices_mask]
        y = non_zeros[1][indices_mask]
        z = non_zeros[2][indices_mask]

        mask = np.zeros_like(blob)
        mask[x, y, z] = 1.0

        return blob * mask
    

class UniformSelectionTransform(Transform):
    """
    A class that limits the number of voxels in the blob by selecting one point for each cell in a 3D grid.
    The value of that point depends on the chosen method.

    :param config: configuration dictionary with integer 'max_blob_size' (maximal number of remaining voxels)
     and string 'method' (dictating the method of assigning values to sampled voxels - options: 'basic'/'average'/'max')
    """


    def __init__(self, max_blob_size: int, method: str = 'max') -> None:
        methods = {
            'basic': UniformSelectionTransform._basic_selection,
            'average': UniformSelectionTransform._average_selection,
            'max': UniformSelectionTransform._max_selection
        }

        if method not in methods:
            raise ValueError(f"Invalid method. Choose from: {list(methods.keys())}")

        self.max_blob_size = max_blob_size
        self.method = method
        self._selection = methods[method]

    @staticmethod
    def _pad_blob(blob: np.ndarray, scale: int) -> np.ndarray:
        x, y, z = blob.shape
        new_shape = [val + (scale - val % scale) % scale for val in (x, y, z)]
        padded_blob = np.zeros(new_shape, dtype=np.float32)
        padded_blob[:x, :y, :z] += blob
        return padded_blob

    @staticmethod
    def _nonzero(blob: np.ndarray) -> int:
        nonzero = np.array(np.nonzero(blob))
        return nonzero.shape[-1]

    @staticmethod
    def _basic_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        voxel_samples = blob[scale // 2::scale, scale // 2::scale, scale // 2::scale]
        processed_blob = np.zeros(blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        return processed_blob

    @staticmethod
    def _average_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for (x, y, z) in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.average(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        processed_blob = processed_blob[:blob.shape[0], :blob.shape[1], :blob.shape[2]]
        return processed_blob

    @staticmethod
    def _max_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for (x, y, z) in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.max(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[scale // 2::scale, scale // 2::scale, scale // 2::scale] = voxel_samples
        processed_blob = processed_blob[:blob.shape[0], :blob.shape[1], :blob.shape[2]]
        return processed_blob

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        nonzero = self._nonzero(blob)
        if nonzero <= self.max_blob_size:
            return blob
        scale = (nonzero / self.max_blob_size) ** (1 / 3)
        scale = math.ceil(scale)
        processed_blob = blob
        while self._nonzero(processed_blob) > self.max_blob_size:
            processed_blob = self._selection(blob, scale)
            scale += 1
        return processed_blob
