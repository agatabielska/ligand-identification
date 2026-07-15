from src.utils.metrics import compute_metrics
from typing import List, Optional
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch


class ScalarShell(nn.Module):
    """
    Function 1: SCALARSHELL (Learnable version)
    Generates scalar features on a shell at distance r from origin.
    """

    def __init__(
        self, p: int, q: int, n_points: int = 64, learnable_sigma: bool = True
    ):
        """
        Args:
            p, q: Signature of Clifford algebra Cl(p,q)
            n_points: Number of sampling points on the shell
            learnable_sigma: Whether sigma is learnable
        """
        super().__init__()
        self.p = p
        self.q = q
        self.dim = p + q
        self.n_points = n_points

        # Learnable sigma parameter
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("sigma", torch.ones(1))

        # Learnable sampling points on unit sphere (normalized during forward)
        self.sampling_points = nn.Parameter(torch.randn(n_points, self.dim))

    def forward(self, eta_pq: torch.Tensor, r: float) -> torch.Tensor:
        """
        Args:
            eta_pq: Metric tensor of shape (p+q, p+q)
            r: Radius of the shell

        Returns:
            s: Scalar features on the shell (n_points,)
        """
        # Normalize sampling points to unit sphere
        v = F.normalize(self.sampling_points, dim=-1)

        # Compute metric: η^pq(v,v)
        # eta_vv = v^T @ eta @ v for each point
        eta_vv = torch.einsum("ni,ij,nj->n", v, eta_pq, v)

        # s ← sgn(η^pq(v,v)) · exp(-(|η^pq(v,v)| - r)^2 / (2σ^2))
        s = torch.sign(eta_vv) * torch.exp(
            -(torch.abs(eta_vv) - r**2) / (2 * self.sigma**2)
        )

        return s

def get_clifford_cayley(p: int, q: int) -> torch.Tensor:
    """
    Generates the 3D Cayley structure constant tensor \Lambda^c_{ab}
    for the Clifford Algebra Cl(p, q).
    Shape: (dim_mv, dim_mv, dim_mv)
    """
    dim = p + q
    dim_mv = 1 << dim
    Lambda = torch.zeros(dim_mv, dim_mv, dim_mv)
    
    # Signature: +1 for the first p basis vectors, -1 for the next q
    sig = [1] * p + [-1] * q
    
    for a in range(dim_mv):
        for b in range(dim_mv):
            sign = 1
            # Extract active basis vectors for indices a and b
            list_a = [i for i in range(dim) if (a & (1 << i))]
            list_b = [i for i in range(dim) if (b & (1 << i))]
            
            # Count swaps needed to order the combined basis vectors
            swaps = 0
            for ia in list_a:
                for ib in list_b:
                    if ia > ib:
                        swaps += 1
            if swaps % 2 == 1:
                sign *= -1
                
            # Account for squaring basis vectors according to the metric signature
            for i in range(dim):
                if (a & (1 << i)) and (b & (1 << i)):
                    sign *= sig[i]
            
            # The resulting basis blade index is the XOR of a and b
            c = a ^ b
            Lambda[a, b, c] = sign
            
    return Lambda


class CliffordSteerableKernel(nn.Module):
    def __init__(
        self, 
        p: int, 
        q: int, 
        c_in: int, 
        c_out: int, 
        kernel_size: int = 3,
        hidden_dim: int = 64
    ):
        super().__init__()
        self.p = p
        self.q = q
        self.dim = p + q
        self.dim_mv = 2 ** self.dim
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.N = kernel_size ** self.dim 
        
        # 1. Automatically generate and register the Cayley Table
        cayley_table = get_clifford_cayley(p, q)
        self.register_buffer("Lambda", cayley_table) # Shape: (dim_mv, dim_mv, dim_mv)
        
        # Metric tensor η for the Scalar Shell
        eta_diag = torch.cat([torch.ones(p), -torch.ones(q)])
        self.register_buffer("eta", torch.diag(eta_diag))

        # 2. Weighted Cayley Initialization: w^c_{oiab} ~ N(0, 1 / sqrt(c_in * N))
        std_dev = 1.0 / math.sqrt(self.c_in * self.N)
        self.w_oiab = nn.Parameter(
            torch.randn(c_out, c_in, self.dim_mv, self.dim_mv, self.dim_mv) * std_dev
        )

        # 3. Global Scalar Shell Variance
        self.sigma = nn.Parameter(torch.empty(1).uniform_(0.4, 0.6))
        
        # 4. Mask Kernel Variances (Per output/input channel pair)
        self.sigma_mask = nn.Parameter(torch.empty(c_out, c_in).uniform_(0.4, 0.6))

        # 5. Corrected CGENN (Coordinate Network / INR)
        # Input: x_n = [s_n, v_n] -> Size: 1 (scalar) + (p+q) (vector)
        # Output: Spatial kernel coefficients for every channel combination and basis blade
        self.cgenn = nn.Sequential(
            nn.Linear(1 + self.dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, c_out * c_in * self.dim_mv)
        )

    def _get_spatial_grid(self, device) -> torch.Tensor:
        """Generates the localized coordinate grid v_n."""
        coords = torch.linspace(-1.0, 1.0, self.kernel_size, device=device)
        grid = torch.stack(torch.meshgrid(*(coords,) * self.dim, indexing="ij"), dim=-1)
        return grid.reshape(-1, self.dim) # (N, p+q)

    def forward(self) -> torch.Tensor:
        device = self.w_oiab.device
        v_n = self._get_spatial_grid(device) # (N, p+q)

        # --- Step 1: SCALARSHELL & EMBED ---
        v_eta_v = torch.einsum("ni,ij,nj->n", v_n, self.eta, v_n)
        s_n = torch.sign(v_eta_v) * torch.exp(-torch.abs(v_eta_v) / (2 * self.sigma**2))
        x_n = torch.cat([s_n.unsqueeze(-1), v_n], dim=-1) # (N, 1+p+q)

        # --- Step 2: Evaluate Coordinate Network ---
        k_hat = self.cgenn(x_n) # (N, c_out * c_in * dim_mv)
        k_hat = k_hat.reshape(self.N, self.c_out, self.c_in, self.dim_mv)

        # --- Step 3: Compute & Apply Mask ---
        mask_exp = torch.exp(
            -torch.abs(v_eta_v.unsqueeze(1).unsqueeze(2)) / (2 * self.sigma_mask**2)
        )
        mask = torch.sign(v_eta_v.unsqueeze(1).unsqueeze(2)) * mask_exp # (N, c_out, c_in)
        k_hat = k_hat * mask.unsqueeze(-1) 

        # --- Step 4: Kernel Head (Weighted Cayley Product) ---
        # Combine the structural algebra constraints with the learnable weights
        W_cayley = self.w_oiab * self.Lambda.unsqueeze(0).unsqueeze(0)

        # Contract over the input basis dimension 'a'
        # k_hat: (N, c_out, c_in, a) | W_cayley: (c_out, c_in, a, b, c) -> (N, c_out, c_in, b, c)
        k = torch.einsum('noia, oiabc -> noibc', k_hat, W_cayley)

        # --- Step 5: Reshape to Standard Convolutions Matrix ---
        # Rearrange dimensions to merge channels with Clifford basis dimensions
        k_final = k.permute(1, 4, 2, 3, 0) # (c_out, c, c_in, b, N)
        spatial_dims = (self.kernel_size,) * self.dim
        
        # Flatten into standard weight shapes: (c_out * 2^d, c_in * 2^d, K, K, K)
        k_final = k_final.reshape(
            self.c_out * self.dim_mv, 
            self.c_in * self.dim_mv, 
            *spatial_dims
        )

        return k_final
    

class CliffordSteerableConvolution(nn.Module):
    """
    Function 3: CLIFFORDSTEERABLECONVOLUTION (Fully Learnable)
    Applies learnable steerable convolution to input feature map.
    """

    def __init__(
        self,
        p: int,
        q: int,
        c_in: int,
        c_out: int,
        n_shells: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        n_sampling_points: int = 64,
        hidden_dim: int = 64,
    ):
        """
        Args:
            p, q: Clifford algebra signature
            c_in: Input channels
            c_out: Output channels
            n_shells: Number of learnable radial shells
            kernel_size: Kernel size
            stride: Convolution stride
            padding: Padding size
            n_sampling_points: Sampling points per shell
            hidden_dim: Hidden dimension for networks
        """
        super().__init__()
        self.p = p
        self.q = q
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Learnable kernel generator
        self.kernel_gen = CliffordSteerableKernel(
            p, q, c_in, c_out, n_shells, kernel_size, n_sampling_points, hidden_dim
        )

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable steerable convolution.

        Args:
            F_in: Input feature map (B, c_in, D, H, W)

        Returns:
            F_out: Output feature map (B, c_out, D', H', W')
        """
        # Generate learnable steerable kernel
        k = self.kernel_gen()  # (c_out, c_in, k, k, k)

        # Apply 3D convolution
        F_out = F.conv3d(
            F_in, k, stride=self.stride, padding=self.padding, bias=self.bias
        )

        return F_out


class CliffordSteerableNetwork(pl.LightningModule):
    def __init__(
        self,
        p: int = 3,
        q: int = 0,
        in_channels: int = 1,
        hidden_channels: List[int] = [16, 32, 64],
        out_channels: int = 10,
        n_shells: int = 3,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Build learnable steerable layers
        layers = []
        prev_c = in_channels

        for hidden_c in hidden_channels:
            layers.append(
                CliffordSteerableConvolution(
                    p,
                    q,
                    prev_c,
                    hidden_c,
                    n_shells=n_shells,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.GroupNorm(min(8, hidden_c), hidden_c))
            layers.append(nn.GELU())
            prev_c = hidden_c

        self.features = nn.Sequential(*layers)

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        # TODO: Check if this additional Linear layer is necessary
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_c, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels),
        )

        # I have to use torchmetrics for macro recall, because it cannot be computed
        # from batches, and sklearn's recall_score doesn't work with accumulating the
        # statistics across batches.
        self.val_recall = torchmetrics.Recall(
            num_classes=out_channels,
            average="macro",
            task="multiclass",
            zero_division=0,
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=out_channels,
            average="macro",
            task="multiclass",
            zero_division=0,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # Computing the loss separately, since compute_metrics won't pass the gradient
        loss = F.cross_entropy(y_hat, y)
        metrics = compute_metrics(y_hat, y, self.out_channels)

        self.log("train_loss", loss)
        self.log("train_acc", metrics["acc"])
        self.log("train_top_10_acc", metrics["top_10_acc"])
        self.log("train_brier_score", metrics["brier_score"])
        self.log("train_mean_rank", metrics["mean_rank"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.out_channels)

        preds = y_hat.argmax(dim=1)
        target = y.squeeze()
        self.val_recall.update(preds, target)

        self.log("val_loss", metrics["loss"])
        self.log("val_acc", metrics["acc"])
        self.log("val_top_10_acc", metrics["top_10_acc"])
        self.log("val_brier_score", metrics["brier_score"])
        self.log("val_mean_rank", metrics["mean_rank"])
        return metrics["loss"]

    def on_validation_epoch_end(self):
        recall = self.val_recall.compute()
        self.log("val_macro_recall", recall)
        self.val_recall.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.out_channels)

        preds = y_hat.argmax(dim=1)
        target = y.squeeze()
        self.test_recall.update(preds, target)

        self.log("test_loss", metrics["loss"])
        self.log("test_acc", metrics["acc"])
        self.log("test_top_10_acc", metrics["top_10_acc"])
        self.log("test_brier_score", metrics["brier_score"])
        self.log("test_mean_rank", metrics["mean_rank"])
        return metrics["loss"]

    def on_test_epoch_end(self):
        recall = self.test_recall.compute()
        self.log("test_macro_recall", recall)
        self.test_recall.reset()

    def on_before_optimizer_step(self, optimizer):
        # Compute and log the total gradient norm
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        self.log("grad_norm", total_norm, on_step=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
