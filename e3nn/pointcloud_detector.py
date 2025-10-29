import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import glob
from scipy.ndimage import zoom, gaussian_filter
from e3nn.o3 import spherical_harmonics, Irreps, Linear


class PointCloudE3NNModel(nn.Module):
    """E3NN model operating on point cloud representation of density."""

    def __init__(self, num_classes: int, max_points: int = 512):
        super().__init__()
        self.max_points = max_points
        self.lmax = 2

        # Simpler irreps structure that matches our features
        # 0e (scalars): 16 features
        # 1o (vectors): 8 * 3 = 24 features (8 3D vectors)
        # 2e (rank-2): 4 * 5 = 20 features (4 5D tensors)
        self.irreps_in = Irreps("16x0e + 8x1o + 4x2e")
        self.irreps_hidden1 = Irreps("32x0e + 8x1o + 4x2e")
        self.irreps_hidden2 = Irreps("64x0e + 4x1o + 2x2e")
        self.irreps_scalar = Irreps("128x0e")

        print(f"Input irreps: {self.irreps_in}")
        print(f"Hidden1 irreps: {self.irreps_hidden1}")
        print(f"Hidden2 irreps: {self.irreps_hidden2}")
        print(f"Output irreps: {self.irreps_scalar}")

        # E3NN layers
        self.e3nn_layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.e3nn_layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.e3nn_layer3 = Linear(self.irreps_hidden2, self.irreps_scalar)

        # Classifier with dropout
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def density_to_pointcloud(self, density, method='probabilistic'):
        """
        Convert 3D density to point cloud using various sampling strategies.

        Args:
            density: (B, 1, D, H, W) tensor
            method: 'probabilistic', 'topk', 'uniform', or 'fps'

        Returns:
            points: (B, N, 3) coordinates
            features: (B, N) density values
        """
        batch_size = density.shape[0]
        all_points = []
        all_features = []

        for b in range(batch_size):
            vol = density[b, 0].cpu().numpy()

            # Get all non-zero voxels
            threshold = vol.max() * 0.01  # Only consider voxels above 1% of max
            coords = np.array(np.where(vol > threshold)).T  # (num_voxels, 3)
            values = vol[vol > threshold]

            if len(coords) == 0:
                # Fallback: use top 10% of voxels
                threshold = np.percentile(vol, 90)
                coords = np.array(np.where(vol > threshold)).T
                values = vol[vol > threshold]

            if len(coords) == 0:
                # Last resort: use all voxels
                z, y, x = np.meshgrid(
                    np.arange(vol.shape[0]),
                    np.arange(vol.shape[1]),
                    np.arange(vol.shape[2]),
                    indexing='ij'
                )
                coords = np.stack([z.flatten(), y.flatten(), x.flatten()], axis=1)
                values = vol.flatten() + 1e-6
            # sampling methods
            if method == 'probabilistic':
                sampled_coords, sampled_values = self._probabilistic_sampling(
                    coords, values, self.max_points
                )
            elif method == 'topk':
                sampled_coords, sampled_values = self._topk_sampling(
                    coords, values, self.max_points
                )
            elif method == 'uniform':
                sampled_coords, sampled_values = self._uniform_sampling(
                    coords, values, self.max_points
                )
            elif method == 'fps':
                sampled_coords, sampled_values = self._fps_sampling(
                    coords, values, self.max_points
                )
            else:
                raise ValueError(f"Unknown sampling method: {method}")

            # Normalize coordinates to [-1, 1]
            center = np.array(vol.shape) / 2
            scale = np.max(vol.shape) / 2
            sampled_coords = (sampled_coords - center) / scale

            all_points.append(torch.tensor(sampled_coords, dtype=torch.float32))
            all_features.append(torch.tensor(sampled_values, dtype=torch.float32))

        points = torch.stack(all_points).to(density.device)
        features = torch.stack(all_features).to(density.device)

        return points, features

    def _probabilistic_sampling(self, coords, values, n_points):
        """Sample points with probability proportional to density values."""
        n_available = len(coords)
        n_sample = min(n_points, n_available)

        if n_sample == n_available:
            return coords, values

        # Use density as probability (higher density = higher chance)
        probs = values / (values.sum() + 1e-10)

        # Handle edge case where all values are the same
        if np.allclose(probs, probs[0]):
            indices = np.random.choice(n_available, size=n_sample, replace=False)
        else:
            indices = np.random.choice(n_available, size=n_sample, replace=False, p=probs)

        return coords[indices], values[indices]

    def _topk_sampling(self, coords, values, n_points):
        """Sample the top-k points with highest density."""
        n_sample = min(n_points, len(coords))
        indices = np.argsort(values)[-n_sample:]
        return coords[indices], values[indices]

    def _uniform_sampling(self, coords, values, n_points):
        """Uniformly sample points."""
        n_available = len(coords)
        n_sample = min(n_points, n_available)
        indices = np.random.choice(n_available, size=n_sample, replace=False)
        return coords[indices], values[indices]

    def _fps_sampling(self, coords, values, n_points):
        """Farthest Point Sampling for better spatial coverage."""
        n_available = len(coords)
        n_sample = min(n_points, n_available)

        if n_sample == n_available:
            return coords, values

        # Start with random point
        selected_indices = [np.random.randint(n_available)]
        distances = np.full(n_available, np.inf)

        for _ in range(n_sample - 1):
            last_point = coords[selected_indices[-1]]
            dist_to_last = np.sum((coords - last_point) ** 2, axis=1)
            distances = np.minimum(distances, dist_to_last)
            selected_indices.append(np.argmax(distances))

        selected_indices = np.array(selected_indices)
        return coords[selected_indices], values[selected_indices]

    def compute_point_features(self, points, density_values):
        """
        Compute rotation-equivariant features for each point.
        Features must match irreps: 16x0e + 8x1o + 4x2e
        Total: 16 + 8*3 + 4*5 = 16 + 24 + 20 = 60 dimensions

        Args:
            points: (B, N, 3) normalized coordinates
            density_values: (B, N) scalar density at each point

        Returns:
            features: (B, N, 60) equivariant features
        """
        batch_size, n_points, _ = points.shape
        all_features = []

        for b in range(batch_size):
            pts = points[b]  # (N, 3)
            vals = density_values[b]  # (N,)

            # computing center of mass for this batch - rotation invariant reference poin
            if vals.sum() > 0:
                com = (pts * vals.unsqueeze(-1)).sum(dim=0) / vals.sum()
            else:
                com = pts.mean(dim=0)

            # Center points around COM
            centered_pts = pts - com

            # radial distance
            r = torch.norm(centered_pts, dim=1)
            r_safe = torch.clamp(r, min=1e-6)
            directions = centered_pts / r_safe.unsqueeze(-1)

            # spherical harmonics
            sh_l0 = spherical_harmonics(0, directions, normalize=True)  # (N, 1)
            sh_l1 = spherical_harmonics(1, directions, normalize=True)  # (N, 3)
            sh_l2 = spherical_harmonics(2, directions, normalize=True)  # (N, 5)

            # scalar features (16x0e = 16 scalars), rotation invariant
            scalar_list = [
                vals,
                r_safe,
                sh_l0[:, 0],
                vals * r_safe,
                vals ** 2,
                torch.log1p(r_safe),
                torch.sqrt(vals + 1e-6),
                r_safe ** 2,
                vals * sh_l0[:, 0],
                torch.exp(-r_safe),
                vals ** 3,
                r_safe ** 0.5,
                vals * torch.log1p(r_safe),
                torch.sigmoid(vals),
                torch.sigmoid(r_safe),
                vals * r_safe ** 2,
            ]
            scalar_features = torch.stack(scalar_list, dim=1)

            # vector features (8x1o = 8 vectors of 3 components = 24 features)
            vector_list = [
                centered_pts,
                centered_pts * vals.unsqueeze(-1),
                sh_l1,
                centered_pts * r_safe.unsqueeze(-1),
                directions,
                directions * vals.unsqueeze(-1),
                centered_pts * (vals ** 2).unsqueeze(-1),
                sh_l1 * r_safe.unsqueeze(-1),
            ]
            vector_features = torch.cat(vector_list, dim=1)

            # tensor features (4x2e = 4 tensors of 5 components = 20 features)
            tensor_list = [
                sh_l2,
                sh_l2 * vals.unsqueeze(-1),
                sh_l2 * r_safe.unsqueeze(-1),
                sh_l2 * torch.log1p(r_safe).unsqueeze(-1),
            ]
            tensor_features = torch.cat(tensor_list, dim=1)

            # Concatenate all features: 16 + 24 + 20 = 60
            point_features = torch.cat([
                scalar_features,
                vector_features,
                tensor_features,
            ], dim=1)

            all_features.append(point_features)

        return torch.stack(all_features)

    def forward(self, x, sampling_method='probabilistic'):
        # convert density to point cloud
        points, density_values = self.density_to_pointcloud(x, method=sampling_method)

        # compute equivariant features
        point_features = self.compute_point_features(points, density_values)

        # Pass through E3NN layers, operates on each point independently
        x = point_features
        x = self.e3nn_layer1(x)
        x = self.e3nn_layer2(x)
        x = self.e3nn_layer3(x)  # (B, N, 128)

        # global pooling (rotation invariant)
        x = torch.max(x, dim=1)[0]  # (B, 128) - max pooling over points

        # Classification head
        x = torch.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        return x, point_features


def load_and_process(filepath):
    data = np.load(filepath)
    density = data["arr_0"]
    density = density - density.min()
    if density.max() > 0:
        density = density / density.max()


    target = (24, 24, 24)  # Slightly larger for better point sampling
    zoom_factors = [t / s for t, s in zip(target, density.shape)]
    density = zoom(density, zoom_factors, order=1)
    density = gaussian_filter(density, sigma=0.7)

    return density
