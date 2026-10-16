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
            self.sigma = nn.Parameter(torch.empty(1).uniform_(0.4, 0.6))
        else:
            self.register_buffer("sigma", torch.empty(1).uniform_(0.4, 0.6))

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
        # We use normalized v, so this formula yields unit length in the metric space
        eta_unit = torch.einsum("ni,ij,nj->n", v, eta_pq, v)
        
        eta_vv = r**2 * eta_unit  # Scale by radius squared

        # s ← sgn(η^pq(v,v)) · exp(-|η^pq(v,v)| / (2σ^2))
        s = torch.sign(eta_vv) * torch.exp(
            -torch.abs(eta_vv) / (2 * self.sigma**2)
        )

        return s


class CGENN(nn.Module):
    """
    Clifford Geometric Equivariant Neural Network.
    Learnable network that processes multivectors.
    """

    def __init__(self, dim_mv: int, c_in: int, c_out: int, hidden_dim: int = 64):
        """
        Args:
            dim_mv: Dimension of multivector space (2^{p+q})
            c_in: Input channels
            c_out: Output channels
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.dim_mv = dim_mv
        self.c_in = c_in
        self.c_out = c_out

        # Learnable weights for geometric product approximation
        # We use MLPs that respect the geometric structure
        self.mv_projection = nn.Sequential(
            nn.Linear(dim_mv, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Channel mixing with geometric awareness
        self.channel_mix = nn.Parameter(torch.randn(c_out, c_in, hidden_dim))
        nn.init.xavier_uniform_(self.channel_mix)

        # Output projection back to multivector space
        self.output_proj = nn.Linear(hidden_dim, dim_mv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input multivectors (..., c_in, dim_mv) or (c_in, dim_mv)
        Returns:
            Output multivectors (..., c_out, dim_mv)
        """
        # Process multivector structure
        # x shape: (..., c_in, dim_mv)
        x_proj = self.mv_projection(x)  # (..., c_in, hidden_dim)

        # Mix channels with learned geometric operations
        # Einstein sum over input channels and hidden dimensions
        out = torch.einsum("...ih,oih->...oh", x_proj, self.channel_mix)

        # Project back to multivector space
        out = self.output_proj(out)  # (..., c_out, dim_mv)

        return out


class CliffordSteerableKernel(nn.Module):
    """
    Function 2: CLIFFORDSTEERABLEKERNEL (Fully Learnable)
    Constructs a steerable convolution kernel using Clifford algebra.
    """

    def __init__(
        self,
        p: int,
        q: int,
        c_in: int,
        c_out: int,
        n_shells: int,
        kernel_size: int = 3,
        n_sampling_points: int = 64,
        hidden_dim: int = 64,
    ):
        """
        Args:
            p, q: Clifford algebra signature
            c_in: Input channels
            c_out: Output channels
            n_shells: Number of radial shells
            kernel_size: Spatial kernel size
            n_sampling_points: Points per shell
            hidden_dim: Hidden dimension for CGENN
        """
        super().__init__()
        self.p = p
        self.q = q
        self.c_in = c_in
        self.c_out = c_out
        self.n_shells = n_shells
        self.kernel_size = kernel_size

        # Dimension of multivector space
        self.dim_mv = 2 ** (p + q)

        # Learnable radii for each shell
        self.radii = nn.Parameter(torch.linspace(0.5, 2.0, n_shells))

        # Learnable ScalarShells (one per shell)
        self.scalar_shells = nn.ModuleList(
            [
                ScalarShell(p, q, n_sampling_points, learnable_sigma=True)
                for _ in range(n_shells)
            ]
        )

        # Learnable embedding networks (scalar + vector -> multivector)
        self.embedders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1 + p + q, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.dim_mv),
                )
                for _ in range(n_shells)
            ]
        )

        # Learnable CGENN for each shell
        self.cgenn_nets = nn.ModuleList(
            [CGENN(self.dim_mv, c_in, c_out, hidden_dim) for _ in range(n_shells)]
        )

        # Metric tensor
        eta_diag = torch.cat([torch.ones(p), -torch.ones(q)])
        self.register_buffer("eta", torch.diag(eta_diag))

        # Learnable kernel mask weights
        self.mask_weights = nn.Parameter(torch.ones(n_shells))

        total_features = c_out * c_in * self.dim_mv
        self.kernel_head = nn.Sequential(
            nn.Conv1d(total_features, total_features, 1, groups=c_out),
            nn.GroupNorm(c_out, total_features),
            nn.GELU(),
            nn.Conv1d(total_features, total_features, 1),
        )

    def forward(self) -> torch.Tensor:
        """
        Construct learnable steerable kernel.
        
        Returns:
            k: Steerable kernel (c_out, c_in, kernel_size)
        """
        device = self.eta.device

        # Create spatial grid for kernel
        spatial_grid = self._create_spatial_grid(device)

        # Initialize accumulated kernel
        k_accum = torch.zeros(
            self.c_out,
            self.c_in,
            self.dim_mv,
            self.kernel_size,
            device=device,
        )

        # Loop over learnable shells
        for shell_idx in range(self.n_shells):
            r = self.radii[shell_idx]
            s_n = self.scalar_shells[shell_idx](self.eta, r)
            v_n = F.normalize(self.scalar_shells[shell_idx].sampling_points, dim=-1)

            sv_concat = torch.cat([s_n.unsqueeze(-1), v_n], dim=-1)
            x_n = self.embedders[shell_idx](sv_concat)
            x_n_agg = x_n.mean(dim=0).unsqueeze(0).expand(self.c_in, -1)  # (c_in, dim_mv)

            k_n = self.cgenn_nets[shell_idx](x_n_agg)  # (c_out, dim_mv)

            # Ensure shape is (c_out, c_in, dim_mv)
            if len(k_n.shape) == 3:
                if k_n.shape[0] == self.c_in and k_n.shape[1] == self.c_out:
                    k_n = k_n.permute(1, 0, 2)
            else:
                k_n = k_n.unsqueeze(1).expand(-1, self.c_in, -1)  # (c_out, c_in, dim_mv)

            # Add spatial dimension: (c_out, c_in, dim_mv, kernel_size)
            k_n = k_n.unsqueeze(-1).expand(-1, -1, -1, self.kernel_size)
            mask = self._compute_learnable_mask(spatial_grid, r, shell_idx)
            k_accum += self.mask_weights[shell_idx] * k_n * mask

        # Flatten for kernel head transformation
        k_flat = k_accum.reshape(1, self.c_out * self.c_in * self.dim_mv, self.kernel_size)
        k_transformed = self.kernel_head(k_flat)
        k_out_struct = k_transformed.reshape(self.c_out, self.c_in, self.dim_mv, self.kernel_size)

        # Build full block matrix of shape (c_out * dim_mv, c_in * dim_mv, kernel_size)[cite: 1]
        k_final = torch.zeros(
            self.c_out, self.dim_mv, self.c_in, self.dim_mv, self.kernel_size,
            device=device, dtype=k_accum.dtype
        )
        for m in range(self.dim_mv):
            k_final[:, m, :, m, :] = k_out_struct[:, :, m, :]

        k_final = k_final.reshape(
            self.c_out * self.dim_mv, self.c_in * self.dim_mv, self.kernel_size
        )

        return k_final

    def _create_spatial_grid(self, device):
        """Create coordinate grid for kernel positions."""
        coords = torch.arange(self.kernel_size, device=device)
        grid = torch.stack(
            torch.meshgrid(coords, indexing="ij"), dim=-1
        )
        center = self.kernel_size // 2
        grid = grid - center  # Center at origin
        return grid.float()

    def _compute_learnable_mask(
        self, spatial_grid: torch.Tensor, r: float, shell_idx: int
    ):
        """Compute learnable geometric mask."""
        # Distance from center
        dist = torch.norm(spatial_grid, dim=-1)

        # Learnable Gaussian-like mask
        mask = torch.exp(-((dist - r) ** 2) / (2.0 * (0.5 + shell_idx * 0.2) ** 2))

        # Expand for channels and multivector dimensions
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1, 1, 1, k)

        return mask


class CliffordSteerableConvolution(nn.Module):
    """
    Function 3: CLIFFORDSTEERABLECONVOLUTION (Fully Learnable)
    Applies learnable steerable convolution to input feature maps,
    preserving multivector components.
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
        self.dim_mv = 2 ** (p + q)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Learnable kernel generator
        self.kernel_gen = CliffordSteerableKernel(
            p, q, c_in, c_out, n_shells, kernel_size, n_sampling_points, hidden_dim
        )
        self.bias = nn.Parameter(torch.zeros(c_out * self.dim_mv))

    def forward(self, F_in: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable steerable convolution.

        Args:
            F_in: Input feature map of shape (B, c_in, H) or (B, c_in * dim_mv, H)
        Returns:
            F_out: Output feature map of shape (B, c_out * dim_mv, H_out)
        """
        B, C_in, H = F_in.shape

        if C_in == self.c_in:
            F_mv = torch.zeros(
                B, self.c_in, self.dim_mv, H, device=F_in.device, dtype=F_in.dtype
            )
            F_mv[:, :, 0, :] = F_in
            F_in_flat = F_mv.reshape(B, self.c_in * self.dim_mv, H)
        elif C_in == self.c_in * self.dim_mv:
            F_in_flat = F_in
        else:
            raise ValueError(
                f"CliffordSteerableConvolution expected input channels to be either {self.c_in} "
                f"or {self.c_in * self.dim_mv}, but got {C_in}."
            )

        k = self.kernel_gen()  # (c_out * dim_mv, c_in * dim_mv, kernel_size)[cite: 1]

        F_out_flat = F.conv1d(
            F_in_flat,
            k,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias,
        )
        return F_out_flat


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
        self.p = p
        self.q = q
        self.dim_mv = 2 ** (p + q)
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
            total_hidden_c = hidden_c * self.dim_mv
            layers.append(nn.GroupNorm(min(8, total_hidden_c), total_hidden_c))
            layers.append(nn.GELU())
            prev_c = hidden_c

        self.features = nn.Sequential(*layers)

        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool1d(1)
        # TODO: Check if this additional Linear layer is necessary
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_c * self.dim_mv, 512),
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
