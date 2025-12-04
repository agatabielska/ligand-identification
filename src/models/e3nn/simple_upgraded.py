from e3nn.o3 import spherical_harmonics, Irreps, Linear
from e3nn.nn import BatchNorm
from src.utils.metrics import compute_metrics
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torch


class SimpleUpgradedE3NN(pl.LightningModule):
    """ E3NN model operating on point cloud representation of density.
        Improved with higher-order spherical harmonics, attention pooling, batch normalization and more scalar features."""
    

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

        self.irreps_in = Irreps("8x0e + 4x1o + 4x2e + 2x3o")
        self.irreps_hidden1 = Irreps("64x0e + 16x1o + 8x2e + 4x3o")
        self.irreps_hidden2 = Irreps("96x0e + 16x1o + 8x2e + 4x3o")
        self.irreps_hidden3 = Irreps("128x0e + 8x1o + 4x2e + 2x3o")
        self.irreps_scalar = Irreps("256x0e")

        self.e3nn_layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.bn1 = BatchNorm(self.irreps_hidden1)
        
        self.e3nn_layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.bn2 = BatchNorm(self.irreps_hidden2)
        
        self.e3nn_layer3 = Linear(self.irreps_hidden2, self.irreps_hidden3)
        self.bn3 = BatchNorm(self.irreps_hidden3)
        
        self.e3nn_layer4 = Linear(self.irreps_hidden3, self.irreps_scalar)

        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.dropout = nn.Dropout(0.15)
        self.fc1 = nn.Linear(768, 384)  # 256*3 from pooling
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

        self.ln1 = nn.LayerNorm(384)
        self.ln2 = nn.LayerNorm(192)

    def compute_point_features(self, points, density_values):
        """ 
        VECTORIZED computation of rotation-equivariant features.
        Args:
            points: (B, N, 3) normalized coordinates
            density_values: (B, N) scalar density at each point
        Returns:
            features: (B, N, 54) equivariant features
        """
        B, N, _ = points.shape

        vals_sum = density_values.sum(dim=1, keepdim=True).clamp(min=1e-8)
        com = (points * density_values.unsqueeze(-1)).sum(dim=1, keepdim=True)
        com = com / vals_sum.unsqueeze(-1)
        centered_pts = points - com

        # Radial features
        r = torch.norm(centered_pts, dim=2, keepdim=True).clamp(min=1e-6)
        directions = centered_pts / r

        # Up to l=3 spherical harmonics
        dirs_flat = directions.reshape(-1, 3)
        sh_l1 = spherical_harmonics(1, dirs_flat, normalize=True).reshape(B, N, 3)
        sh_l2 = spherical_harmonics(2, dirs_flat, normalize=True).reshape(B, N, 5)
        sh_l3 = spherical_harmonics(3, dirs_flat, normalize=True).reshape(B, N, 7)

        # Scalar features (8x0e = 8 features)
        density = density_values.unsqueeze(-1)
        log_r = torch.log1p(r)
        r_squared = r.pow(2)
        r_cubed = r.pow(3)
        density_r = density * r
        density_r2 = density * r_squared
        gaussian_like = torch.exp(-r_squared / 2.0)
        scalar_features = torch.cat(
            [density, r, log_r, r_squared, density_r, density_r2, gaussian_like, r_cubed],
            dim=-1,
        )  # Shape: (B, N, 8)

        # Vector features
        weighted_dirs = directions * density
        vector_features = torch.cat([directions, sh_l1], dim=-1)
        scaled_dirs = directions * log_r
        vector_features = torch.cat([vector_features, weighted_dirs, scaled_dirs], dim=-1)  # (B, N, 12)

        # Rank-2 tensor features 
        tensor2_features = torch.cat([
            sh_l2,                    
            sh_l2 * density,          
            sh_l2 * r,                
            sh_l2 * log_r,           
        ], dim=-1)  

        # Rank-3 tensor features (2x3o = 2 tensors * 7 components = 14 features)
        tensor3_features = torch.cat([
            sh_l3,                    
            sh_l3 * density,          
        ], dim=-1)  # (B, N, 14)

        # Concatenate: 8 + 12 + 20 + 14 = 54 features
        point_features = torch.cat(
            [scalar_features, vector_features, tensor2_features, tensor3_features],
            dim=-1,
        )  # (B, N, 54)

        return point_features

    def attention_pooling(self, x):
        """
        Attention-weighted pooling: learns which points are most important.
        Args:
            x: (B, N, D)
        Returns:
            (B, D)
        """
        B, N, D = x.shape
        
        # Compute attention weights
        att_weights = self.attention(x)  # (B, N, 1)
        att_weights = F.softmax(att_weights, dim=1)
        
        # Weighted sum
        x_att = (x * att_weights).sum(dim=1)  # (B, D)
        
        return x_att

    def forward(self, x):
        """
        Forward pass with residual connections and attention.
        """
        points = x[:, :, :3]
        density_values = x[:, :, 3]

        # Compute equivariant features
        point_features = self.compute_point_features(points, density_values)

        # E3NN layers with batch normalization
        x = self.e3nn_layer1(point_features)
        x = self.bn1(x)
        
        x = self.e3nn_layer2(x)
        x = self.bn2(x)
        
        x = self.e3nn_layer3(x)
        x = self.bn3(x)
        
        x = self.e3nn_layer4(x)

        # Multi-scale pooling
        x_max = torch.max(x, dim=1)[0]
        x_mean = torch.mean(x, dim=1)
        x_att = self.attention_pooling(x)

        # Concatenate all pooling strategies
        x = torch.cat([x_max, x_mean, x_att], dim=-1)

        # Classification head with layer normalization
        x = torch.relu(self.ln1(self.fc1(self.dropout(x))))
        x = torch.relu(self.ln2(self.fc2(self.dropout(x))))
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
        self.log("train_brier_score", metrics["brier_score"])
        self.log("train_macro_recall", metrics["macro_recall"])
        self.log("train_mean_rank", metrics["mean_rank"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("val_loss", metrics["loss"])
        self.log("val_acc", metrics["acc"])
        self.log("val_top_10_acc", metrics["top_10_acc"])
        self.log("val_brier_score", metrics["brier_score"])
        self.log("val_macro_recall", metrics["macro_recall"])
        self.log("val_mean_rank", metrics["mean_rank"])
        return metrics["loss"]

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("test_loss", metrics["loss"])
        self.log("test_acc", metrics["acc"])
        self.log("test_top_10_acc", metrics["top_10_acc"])
        self.log("test_brier_score", metrics["brier_score"])
        self.log("test_macro_recall", metrics["macro_recall"])
        self.log("test_mean_rank", metrics["mean_rank"])
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