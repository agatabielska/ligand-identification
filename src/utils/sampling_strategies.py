# Adapted from https://github.com/jkarolczak/ligand-classification/blob/8ca4ffd678382d8084c0a162333692b74c985471/src/pipeline/transforms.py#L146
from numpy.lib.stride_tricks import sliding_window_view
from scipy.special import softmax
from abc import ABC, abstractmethod
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

    def __init__(self, max_blob_size: int = 2000):
        self.max_blob_size = max_blob_size

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        if self.max_blob_size is None:
            raise ValueError(
                "max_blob_size has to be specified for RandomSelectionTransform"
            )
        non_zeros = blob.nonzero()
        if non_zeros[0].shape[0] <= self.max_blob_size:
            return blob

        indices_mask = np.array(range(non_zeros[0].shape[0]))
        indices_mask = np.random.choice(
            indices_mask, size=self.max_blob_size, replace=False
        )
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
    """

    def __init__(self, max_blob_size: int = 2000, method: str = "max") -> None:
        methods = {
            "basic": UniformSelectionTransform._basic_selection,
            "average": UniformSelectionTransform._average_selection,
            "max": UniformSelectionTransform._max_selection,
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
        voxel_samples = blob[
            scale // 2 :: scale, scale // 2 :: scale, scale // 2 :: scale
        ]
        processed_blob = np.zeros(blob.shape)
        processed_blob[
            scale // 2 :: scale, scale // 2 :: scale, scale // 2 :: scale
        ] = voxel_samples
        return processed_blob

    @staticmethod
    def _average_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for x, y, z in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.average(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[
            scale // 2 :: scale, scale // 2 :: scale, scale // 2 :: scale
        ] = voxel_samples
        processed_blob = processed_blob[
            : blob.shape[0], : blob.shape[1], : blob.shape[2]
        ]
        return processed_blob

    @staticmethod
    def _max_selection(blob: np.ndarray, scale: int) -> np.ndarray:
        padded_blob = UniformSelectionTransform._pad_blob(blob, scale)
        sub_arrays = []
        for x, y, z in itertools.product(list(range(scale)), repeat=3):
            sub_array = padded_blob[x::scale, y::scale, z::scale]
            sub_arrays.append(sub_array)
        sub_arrays = np.stack(sub_arrays)
        voxel_samples = np.max(sub_arrays, axis=0)
        processed_blob = np.zeros(padded_blob.shape)
        processed_blob[
            scale // 2 :: scale, scale // 2 :: scale, scale // 2 :: scale
        ] = voxel_samples
        processed_blob = processed_blob[
            : blob.shape[0], : blob.shape[1], : blob.shape[2]
        ]
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


class PowerTransform(Transform):
    """
    A class that applies a power transformation to the blob.
    Each voxel value is raised to the specified power.
    """

    def __init__(self, power: float):
        self.power = power

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        blob = np.clip(blob, a_min=0, a_max=None)
        non_zero = blob.nonzero()
        blob[non_zero] = np.power(blob[non_zero], self.power)
        blob[non_zero] = (blob[non_zero] - np.min(blob[non_zero])) / (
            np.max(blob[non_zero]) - np.min(blob[non_zero]) + 1e-9
        )
        return blob


class ProbabilisticSelectionTransform(Transform):
    """
    A class that limits the number of voxels in the blob by selecting points based on their probabilities.
    Voxels with higher probabilities have a higher chance of being selected.
    Probabilities are adjusted using a alpha parameter to control the sharpness of the distribution.
    """

    def __init__(self, max_blob_size: int = 2000):
        self.max_blob_size = max_blob_size

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        non_zeros = blob.nonzero()
        if non_zeros[0].shape[0] <= self.max_blob_size:
            return blob

        probabilities = blob[non_zeros]
        probabilities /= probabilities.sum()

        indices_mask = np.array(range(non_zeros[0].shape[0]))
        indices_mask = np.random.choice(
            indices_mask, size=self.max_blob_size, replace=False, p=probabilities
        )
        x = non_zeros[0][indices_mask]
        y = non_zeros[1][indices_mask]
        z = non_zeros[2][indices_mask]

        mask = np.zeros_like(blob)
        mask[x, y, z] = 1.0

        return blob * mask


class SpatialNormalization(Transform):
    """
    Normalizes the blob based on local min and max values within a square_size x square_size x square_size cube around each voxel.
    This process is repeated for a specified number of iterations to enhance local contrast.
    """

    def __init__(self, iterations: int = 1, square_size: int = 3):
        if square_size % 2 == 0:
            raise ValueError("square_size must be an odd number.")
        self.iterations = iterations
        self.square_size = square_size

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        for _ in range(self.iterations):
            padded_blob = np.pad(blob, pad_width=self.square_size // 2, mode="edge")
            windows = sliding_window_view(
                padded_blob, (self.square_size, self.square_size, self.square_size)
            )

            # Compute min and max for each 3x3x3 cube
            mins = windows.min(axis=(-3, -2, -1))
            maxs = windows.max(axis=(-3, -2, -1))

            blob = (blob - mins) / (maxs - mins + 1e-9)

        return blob


class SpatialStandardization(Transform):
    """
    Standardizes the blob based on local mean and std values within a square_size x square_size x square_size cube around each voxel.
    This process is repeated for a specified number of iterations to enhance local contrast.
    """

    def __init__(self, iterations: int = 1, square_size: int = 3):
        if square_size % 2 == 0:
            raise ValueError("square_size must be an odd number.")
        self.iterations = iterations
        self.square_size = square_size

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        for _ in range(self.iterations):
            padded_blob = np.pad(blob, pad_width=self.square_size // 2, mode="edge")
            windows = sliding_window_view(
                padded_blob, (self.square_size, self.square_size, self.square_size)
            )

            # Compute mean and std for each 3x3x3 cube
            means = windows.mean(axis=(-3, -2, -1))
            stds = windows.std(axis=(-3, -2, -1))

            blob = (blob - means) / (stds + 1e-9)

        # Rescale to [0, 1]
        blob = blob - np.min(blob)
        blob = blob / (np.max(blob) + 1e-9)

        return blob


class SpatialNormalization2(Transform):
    """
    Normalizes the blob based on local values within a square_size x square_size x square_size cube around each voxel.
    This process is repeated for a specified number of iterations to enhance local contrast.
    """

    def __init__(self, iterations: int = 1, square_size: int = 3, percentile: float = 90):
        if square_size % 2 == 0:
            raise ValueError("square_size must be an odd number.")
        self.iterations = iterations
        self.square_size = square_size
        self.percentile = percentile

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        for _ in range(self.iterations):
            padded_blob = np.pad(blob, pad_width=self.square_size // 2, mode="edge")
            windows = sliding_window_view(
                padded_blob, (self.square_size, self.square_size, self.square_size)
            )

            scale = np.percentile(windows, self.percentile, axis=(-3, -2, -1))
            blob = blob / (scale + 1e-9)

        # Rescale to [0, 1]
        blob = blob - np.min(blob)
        blob = blob / (np.max(blob) + 1e-9)

        return blob


class TemperatureScaling(Transform):
    """
    A class that applies temperature scaling to the blob.
    """

    def __init__(self, temperature: float):
        self.temperature = temperature

    def preprocess(self, blob: np.ndarray) -> np.ndarray:
        blob = np.clip(blob, a_min=0, a_max=None)
        non_zero = blob.nonzero()
        blob[non_zero] = softmax(blob[non_zero] / self.temperature)
        return blob
