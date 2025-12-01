import torch
import torch.nn as nn
import pytorch_lightning as pl
from e3nn.o3 import spherical_harmonics, Irreps, Linear


class E3NNPointCloudModel(pl.LightningModule):

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
        self.irreps_hidden1 = Irreps("64x0e + 16x1o + 8x2e")
        self.irreps_hidden2 = Irreps("128x0e + 16x1o + 4x2e")
        self.irreps_hidden3 = Irreps("256x0e + 8x1o + 2x2e")
        self.irreps_scalar = Irreps("512x0e")

        # Simple E3NN linear layers (equivariant but no message passing yet)
        self.layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.layer3 = Linear(self.irreps_hidden2, self.irreps_hidden3)
        self.layer4 = Linear(self.irreps_hidden3, self.irreps_scalar)

        # Non-linearity on scalars only
        self.act = nn.SiLU()

        # Classification head
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(1536, 512),  # 512*3 from max/mean/std pooling
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()

    def compute_point_features(self, points, density_values):
        """
        Compute properly structured rotation-equivariant features.
        
        Args:
            points: (B, N, 3) coordinates
            density_values: (B, N) scalar density values
            
        Returns:
            features: (B, N, 20) features matching irreps_in
        """
        B, N, _ = points.shape

        # Center around center of mass
        vals_sum = density_values.sum(dim=1, keepdim=True).clamp(min=1e-8)
        com = (points * density_values.unsqueeze(-1)).sum(dim=1, keepdim=True) / vals_sum.unsqueeze(-1)
        centered_pts = points - com

        # Radial features
        r = torch.norm(centered_pts, dim=2, keepdim=True)
        r_safe = torch.clamp(r, min=1e-6)
        directions = centered_pts / r_safe

        # Spherical harmonics
        dirs_flat = directions.reshape(-1, 3)
        sh_l1 = spherical_harmonics(1, dirs_flat, normalize=True).reshape(B, N, 3)
        sh_l2 = spherical_harmonics(2, dirs_flat, normalize=True).reshape(B, N, 5)

        # PROPERLY STRUCTURED FEATURES
        # Scalars (4x0e)
        density = density_values.unsqueeze(-1)
        log_r = torch.log1p(r_safe)
        density_r = density * r_safe
        scalars = torch.cat([density, r_safe, log_r, density_r], dim=-1)

        # Vectors (2x1o = 6 components)
        vectors = torch.cat([directions, sh_l1], dim=-1)

        # Rank-2 tensors (2x2e = 10 components)
        tensors = torch.cat([sh_l2, sh_l2 * density], dim=-1)

        # Concatenate in irreps order
        return torch.cat([scalars, vectors, tensors], dim=-1)

    def apply_nonlinearity(self, x, irreps):
        """
        Apply non-linearity only to scalar components (preserves equivariance).
        
        Args:
            x: (B, N, features) input features
            irreps: Irreps object describing the structure
            
        Returns:
            x: (B, N, features) with non-linearity applied to scalars only
        """
        # Find which indices correspond to scalars (l=0)
        idx = 0
        output_chunks = []
        
        for mul, ir in irreps:
            dim = mul * ir.dim
            chunk = x[..., idx:idx+dim]
            
            if ir.l == 0:
                chunk = self.act(chunk)
            # Vectors and tensors - leave unchanged
            
            output_chunks.append(chunk)
            idx += dim
        
        return torch.cat(output_chunks, dim=-1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (B, N, 4) tensor [x, y, z, density]  
        Returns:
            logits: (B, num_classes)
        """
        B, N, _ = x.shape
        
        # Extract points and density
        points = x[:, :, :3]
        density_values = x[:, :, 3]
        
        # Compute initial node features
        node_features = self.compute_point_features(points, density_values)  # (B, N, 20)
        
        # Flatten for processing
        node_features_flat = node_features.reshape(-1, node_features.shape[-1])
        
        # Pass through E3NN layers with non-linearities
        x = self.layer1(node_features_flat)
        x = self.apply_nonlinearity(x, self.irreps_hidden1)
        
        x = self.layer2(x)
        x = self.apply_nonlinearity(x, self.irreps_hidden2)
        
        x = self.layer3(x)
        x = self.apply_nonlinearity(x, self.irreps_hidden3)
        
        # Final projection to scalars only (rotation invariant)
        x = self.layer4(x)
        x = self.act(x)  
        
        # Reshape back to (B, N, features)
        x = x.reshape(B, N, -1)
        
        # Global pooling
        x_max = torch.max(x, dim=1)[0]
        x_mean = torch.mean(x, dim=1)
        x_std = torch.std(x, dim=1)
        
        # Concatenate pooled features
        pooled = torch.cat([x_max, x_mean, x_std], dim=-1)
        
        # Classification
        return self.classifier(self.dropout(pooled))

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        
        top_10 = output.topk(10, dim=1).indices
        top_10_acc = (top_10 == target.unsqueeze(1)).any(dim=1).float().mean()
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_top_10_acc", top_10_acc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        
        top_10 = output.topk(10, dim=1).indices
        top_10_acc = (top_10 == target.unsqueeze(1)).any(dim=1).float().mean()
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_top_10_acc", top_10_acc)
        
        return loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        
        pred = output.argmax(dim=1)
        acc = (pred == target).float().mean()
        
        top_10 = output.topk(10, dim=1).indices
        top_10_acc = (top_10 == target.unsqueeze(1)).any(dim=1).float().mean()
        
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_top_10_acc", top_10_acc)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Warmup + Cosine annealing
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)
        cosine = CosineAnnealingLR(optimizer, T_max=45)
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[5]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }