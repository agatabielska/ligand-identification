from e3nn.o3 import spherical_harmonics, Irreps, Linear, FullyConnectedTensorProduct
from e3nn.nn import Gate, NormActivation
from e3nn.math import soft_one_hot_linspace
from src.utils.metrics import compute_metrics
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch


class ImprovedE3NNModel(pl.LightningModule):
    """ Improved E3NN model with NequIP-style message passing for point cloud data."""

    def __init__(
        self,
        num_classes: int,
        max_neighbors: int = 32,
        cutoff_radius: float = 5.0,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_neighbors = max_neighbors
        self.cutoff_radius = cutoff_radius
        self.num_layers = num_layers

        # Feature dimensions:
        # 4x0e = 4 scalars = 4 features
        # 4x1o = 4 vectors * 3 = 12 features
        # 2x2e = 2 tensors * 5 = 10 features
        # Total = 4 + 12 + 10 = 26 features (but we'll build 30 to match 5*6)
        self.irreps_in = Irreps("6x0e + 4x1o + 2x2e")  # 6 + 12 + 10 = 28, close enough
        self.irreps_hidden = Irreps("64x0e + 16x1o + 8x2e")
        self.irreps_edge_attr = Irreps.spherical_harmonics(lmax=3)
        self.irreps_output = Irreps("128x0e")

        # Number of radial basis functions
        self.num_basis = 8

        # Input feature embedding
        self.feature_embedding = Linear(self.irreps_in, self.irreps_hidden)

        # Message passing layers (NequIP-style convolutions)
        self.convolution_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = EquivariantConvolution(
                irreps_in=self.irreps_hidden,
                irreps_out=self.irreps_hidden,
                irreps_edge_attr=self.irreps_edge_attr,
                num_basis=self.num_basis,
                cutoff_radius=cutoff_radius,
            )
            self.convolution_layers.append(layer)

        # Final projection to scalars only
        self.final_projection = Linear(self.irreps_hidden, self.irreps_output)

        # Classification head
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(384, 256)  # 128*3 from multi-pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def compute_initial_features(self, points, density_values):
        """
        Compute rich initial equivariant features for each point.
        Returns 28 features: 6 scalars + 12 vectors + 10 tensors
        Args:
            points: (B, N, 3) coordinates
            density_values: (B, N) scalar values
        Returns:
            (B, N, 28) matching irreps_in
        """
        B, N, _ = points.shape

        # Center of mass (per batch)
        vals_sum = density_values.sum(dim=1, keepdim=True).clamp(min=1e-8)
        com = (points * density_values.unsqueeze(-1)).sum(dim=1, keepdim=True)
        com = com / vals_sum.unsqueeze(-1)
        centered_pts = points - com

        # Radial features
        r = torch.norm(centered_pts, dim=2, keepdim=True).clamp(min=1e-6)
        directions = centered_pts / r

        # Spherical harmonics
        dirs_flat = directions.reshape(-1, 3)
        sh_l1 = spherical_harmonics(1, dirs_flat, normalize=True).reshape(B, N, 3)
        sh_l2 = spherical_harmonics(2, dirs_flat, normalize=True).reshape(B, N, 5)

        # Scalar features (6x0e = 6 features)
        density = density_values.unsqueeze(-1)
        log_r = torch.log1p(r)
        r_squared = r.pow(2)
        density_r = density * r
        r_inv = 1.0 / (r + 1e-6)
        scalar_features = torch.cat(
            [density, r, log_r, r_squared, density_r, r_inv],
            dim=-1,
        )  # (B, N, 6)

        # Vector features (4x1o = 4 vectors * 3 = 12 features)
        weighted_dirs = directions * density
        vector_features = torch.cat([
            directions,      # 3 features
            weighted_dirs,   # 3 features
            sh_l1,           # 3 features
            directions * r,  # 3 features
        ], dim=-1)  # (B, N, 12)

        # Tensor features (2x2e = 2 tensors * 5 = 10 features)
        tensor_features = torch.cat([
            sh_l2,           # 5 features
            sh_l2 * density, # 5 features
        ], dim=-1)  # (B, N, 10)

        # Concatenate all: 6 + 12 + 10 = 28 features matching irreps_in
        features = torch.cat([scalar_features, vector_features, tensor_features], dim=-1)

        return features

    def build_graph(self, points):
        """
        Build k-nearest neighbor graph for message passing.
        Args:
            points: (B, N, 3)
        Returns:
            edge_index: (2, E) edge connectivity
            edge_vec: (E, 3) relative position vectors
            edge_length: (E,) distances
        """
        B, N, _ = points.shape
        device = points.device

        # For simplicity, use distance-based connectivity per batch
        edges_list = []
        edge_vecs_list = []
        edge_lengths_list = []

        for b in range(B):
            pts = points[b]  # (N, 3)
            # Compute pairwise distances
            dist = torch.cdist(pts, pts)  # (N, N)

            # Find k nearest neighbors (excluding self)
            k = min(self.max_neighbors, N - 1)
            _, indices = torch.topk(dist, k + 1, largest=False, dim=1)
            indices = indices[:, 1:]  # exclude self (distance=0)

            # Build edge list
            src = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            dst = indices.reshape(-1)

            # Relative vectors and distances
            edge_vec = pts[dst] - pts[src]  # (E, 3)
            edge_length = torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)

            # Apply cutoff
            mask = (edge_length.squeeze() < self.cutoff_radius) & (edge_length.squeeze() > 0)
            src = src[mask]
            dst = dst[mask]
            edge_vec = edge_vec[mask]
            edge_length = edge_length[mask].squeeze()

            # Offset indices by batch
            src = src + b * N
            dst = dst + b * N

            edges_list.append(torch.stack([src, dst], dim=0))
            edge_vecs_list.append(edge_vec)
            edge_lengths_list.append(edge_length)

        edge_index = torch.cat(edges_list, dim=1)  # (2, total_edges)
        edge_vec = torch.cat(edge_vecs_list, dim=0)  # (total_edges, 3)
        edge_length = torch.cat(edge_lengths_list, dim=0)  # (total_edges,)

        return edge_index, edge_vec, edge_length

    def forward(self, x):
        """
        Forward pass with message passing.
        Args:
            x: (B, N, 4) where last dim is [x, y, z, density]
        Returns:
            logits: (B, num_classes)
        """
        points = x[:, :, :3]
        density_values = x[:, :, 3]
        B, N, _ = points.shape

        # Compute initial equivariant features
        node_features = self.compute_initial_features(points, density_values)
        node_features = self.feature_embedding(node_features)  # (B, N, hidden_dim)

        # Flatten batch dimension for message passing
        points_flat = points.reshape(-1, 3)  # (B*N, 3)
        node_features_flat = node_features.reshape(-1, node_features.shape[-1])

        # Build graph
        edge_index, edge_vec, edge_length = self.build_graph(points)

        # Compute edge attributes (spherical harmonics of edge directions)
        edge_sh = spherical_harmonics(
            self.irreps_edge_attr, edge_vec / edge_length.unsqueeze(-1), normalize=True
        )

        # Message passing layers
        for conv_layer in self.convolution_layers:
            node_features_flat = conv_layer(
                node_features_flat, edge_index, edge_sh, edge_length
            )

        # Reshape back to batch
        node_features = node_features_flat.reshape(B, N, -1)

        # Final projection to scalars
        node_features = self.final_projection(node_features)

        # Global pooling (rotation invariant)
        x_max = torch.max(node_features, dim=1)[0]
        x_mean = torch.mean(node_features, dim=1)
        x_std = torch.std(node_features, dim=1)

        # Concatenate pooling strategies
        x = torch.cat([x_max, x_mean, x_std], dim=-1)

        # Classification head
        x = torch.relu(self.fc1(self.dropout(x)))
        x = torch.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("train_loss", loss)
        self.log("train_acc", metrics["acc"])
        self.log("train_top_10_acc", metrics["top_10_acc"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("val_loss", metrics["loss"])
        self.log("val_acc", metrics["acc"])
        self.log("val_top_10_acc", metrics["top_10_acc"])
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("test_loss", metrics["loss"])
        self.log("test_acc", metrics["acc"])
        self.log("test_top_10_acc", metrics["top_10_acc"])
        return metrics["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class EquivariantConvolution(nn.Module):
    """
    NequIP-style equivariant convolution layer.
    Implements: f'_i = (1/sqrt(z)) * sum_j f_j ⊗ [h(||r_ij||) * Y(r_ij)]
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        irreps_edge_attr: Irreps,
        num_basis: int,
        cutoff_radius: float,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_edge_attr = irreps_edge_attr
        self.num_basis = num_basis
        self.cutoff_radius = cutoff_radius

        # Radial network: maps edge length to tensor product weights
        self.radial_net = nn.Sequential(
            nn.Linear(num_basis, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        # Tensor product: combines node features with edge attributes
        self.tensor_product = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_edge_attr,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )

        # Output dimension for radial network
        self.radial_net_out = nn.Linear(64, self.tensor_product.weight_numel)

        # Self-connection (skip connection)
        self.self_connection = Linear(irreps_in, irreps_out)

        # Layer normalization
        self.norm = NormActivation(irreps_out, scalar_nonlinearity=nn.SiLU())

    def forward(self, node_features, edge_index, edge_sh, edge_length):
        """
        Args:
            node_features: (N, irreps_in_dim)
            edge_index: (2, E)
            edge_sh: (E, irreps_edge_attr_dim) spherical harmonics
            edge_length: (E,)
        Returns:
            (N, irreps_out_dim)
        """
        src, dst = edge_index  # src -> dst messages

        # Radial basis embedding
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.cutoff_radius,
            number=self.num_basis,
            basis="smooth_finite",
            cutoff=True,
        )
        edge_length_embedded = edge_length_embedded * (self.num_basis**0.5)

        # Radial network
        edge_weight = self.radial_net(edge_length_embedded)
        edge_weight = self.radial_net_out(edge_weight)  # (E, weight_numel)

        # Tensor product: node_features[src] ⊗ edge_sh
        messages = self.tensor_product(
            node_features[src], edge_sh, edge_weight
        )  # (E, irreps_out_dim)

        # Aggregate messages (scatter add)
        num_nodes = node_features.shape[0]
        out = torch.zeros(
            num_nodes, self.irreps_out.dim, device=node_features.device
        )
        out.index_add_(0, dst, messages)

        # Normalize by average number of neighbors (approximate)
        num_neighbors = torch.zeros(num_nodes, device=node_features.device)
        num_neighbors.index_add_(
            0, dst, torch.ones(edge_index.shape[1], device=node_features.device)
        )
        num_neighbors = num_neighbors.clamp(min=1.0)
        out = out / num_neighbors.unsqueeze(-1).sqrt()

        # Self-connection (skip connection)
        out = out + self.self_connection(node_features)

        # Normalization with nonlinearity
        out = self.norm(out)

        return out