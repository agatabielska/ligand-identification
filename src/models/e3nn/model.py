import torch
import torch.nn as nn
import pytorch_lightning as pl
from e3nn.o3 import spherical_harmonics, Irreps, Linear

class E3NNPointCloudModel(pl.LightningModule):
    """E3NN model operating on point cloud representation of density."""

    def __init__(
            self,
            num_classes: int,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Define irreps structure
        self.irreps_in = Irreps("4x0e + 2x1o + 2x2e")
        self.irreps_hidden1 = Irreps("32x0e + 8x1o + 4x2e")
        self.irreps_hidden2 = Irreps("64x0e + 8x1o + 4x2e")
        self.irreps_hidden3 = Irreps("128x0e + 4x1o + 2x2e")
        self.irreps_scalar = Irreps("256x0e")

        # E3NN layers
        self.e3nn_layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.e3nn_layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.e3nn_layer3 = Linear(self.irreps_hidden2, self.irreps_hidden3)
        self.e3nn_layer4 = Linear(self.irreps_hidden3, self.irreps_scalar)

        # Classifier with dropout
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 256)  # 768 from multi-scale pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def compute_point_features(self, points, density_values):
        """
        VECTORIZED computation of rotation-equivariant features.

        Args:
            points: (B, N, 3) normalized coordinates
            density_values: (B, N) scalar density at each point

        Returns:
            features: (B, N, 20) equivariant features
        """
        B, N, _ = points.shape

        # Compute center of mass per sample (vectorized)
        vals_sum = density_values.sum(dim=1, keepdim=True).clamp(min=1e-8)
        com = (points * density_values.unsqueeze(-1)).sum(dim=1, keepdim=True) / vals_sum.unsqueeze(-1)

        # Center points around COM
        centered_pts = points - com

        # Radial distance
        r = torch.norm(centered_pts, dim=2, keepdim=True)
        r_safe = torch.clamp(r, min=1e-6)
        directions = centered_pts / r_safe

        # Spherical harmonics
        dirs_flat = directions.reshape(-1, 3)
        sh_l1 = spherical_harmonics(1, dirs_flat, normalize=True).reshape(B, N, 3)
        sh_l2 = spherical_harmonics(2, dirs_flat, normalize=True).reshape(B, N, 5)

        # Scalar features (4x0e = 4 scalars)
        density = density_values.unsqueeze(-1)
        log_r = torch.log1p(r_safe)
        density_r = density * r_safe

        scalar_features = torch.cat([
            density,
            r_safe,
            log_r,
            density_r,
        ], dim=-1)

        # Vector features (2x1o = 2 vectors * 3 components = 6 features)
        vector_features = torch.cat([
            directions,
            sh_l1
        ], dim=-1)

        # Tensor features (2x2e = 2 tensors * 5 components = 10 features)
        tensor_features = torch.cat([
            sh_l2,
            sh_l2 * density
        ], dim=-1)

        # Concatenate: 4 + 6 + 10 = 20 features
        point_features = torch.cat([
            scalar_features,
            vector_features,
            tensor_features
        ], dim=-1)

        return point_features

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: (B, N, 4) tensor where last dim is [x, y, z, density]

        Returns:
            logits: (B, num_classes) classification logits
        """
        # Separate coordinates and density features
        points = x[:, :, :3]
        density_values = x[:, :, 3]

        # Compute equivariant features
        point_features = self.compute_point_features(points, density_values)

        # Pass through E3NN layers
        x = self.e3nn_layer1(point_features)
        x = self.e3nn_layer2(x)
        x = self.e3nn_layer3(x)
        x = self.e3nn_layer4(x)

        # Global pooling (rotation invariant)
        x_max = torch.max(x, dim=1)[0]
        x_mean = torch.mean(x, dim=1)
        x_std = torch.std(x, dim=1)

        # Concatenate all pooling strategies
        x = torch.cat([x_max, x_mean, x_std], dim=-1)

        # Classification head
        x = torch.relu(self.fc1(self.dropout(x)))
        x = torch.relu(self.fc2(self.dropout(x)))
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        # Calculate top-10 accuracy
        top_10 = output.topk(10, dim=1).indices
        top_10_acc = torch.tensor([
            1.0 if target[i] in top_10[i] else 0.0
            for i in range(target.size(0))
        ]).mean()

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        self.log('train_top_10_acc', top_10_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)

        # Calculate accuracy
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()

        # Calculate top-10 accuracy
        top_10 = output.topk(10, dim=1).indices
        top_10_acc = torch.tensor([
            1.0 if target[i] in top_10[i] else 0.0
            for i in range(target.size(0))
        ]).mean()

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_top_10_acc', top_10_acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
