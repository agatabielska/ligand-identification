from src.utils.metrics import compute_metrics
from typing import List, Optional
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
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
            x: Input multivectors (..., dim_mv) or (..., c_in, dim_mv)
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

        # Learnable kernel head (final transformation)
        # Takes (1, c_out*c_in, k, k, k) and outputs (1, c_out*c_in, k, k, k)
        self.kernel_head = nn.Sequential(
            nn.Conv3d(c_out * c_in, c_out * c_in, 1, groups=c_out),
            nn.GroupNorm(c_out, c_out * c_in),
            nn.GELU(),
            nn.Conv3d(c_out * c_in, c_out * c_in, 1),
        )

    def forward(self, spatial_grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Construct learnable steerable kernel.

        Args:
            spatial_grid: Optional spatial grid for kernel positions

        Returns:
            k: Steerable kernel (c_out, c_in, kernel_size, kernel_size, kernel_size)
        """
        device = self.eta.device

        # Create spatial grid for kernel
        if spatial_grid is None:
            spatial_grid = self._create_spatial_grid(device)

        # Initialize accumulated kernel
        k_accum = torch.zeros(
            self.c_out,
            self.c_in,
            self.dim_mv,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
            device=device,
        )

        # Loop over learnable shells
        for shell_idx in range(self.n_shells):
            r = self.radii[shell_idx]

            # Compute scalar shell features
            s_n = self.scalar_shells[shell_idx](self.eta, r)  # (n_points,)

            # Get sampling points
            v_n = F.normalize(self.scalar_shells[shell_idx].sampling_points, dim=-1)

            # Embed scalars and vectors as multivectors
            sv_concat = torch.cat([s_n.unsqueeze(-1), v_n], dim=-1)  # (n_points, 1+p+q)
            x_n = self.embedders[shell_idx](sv_concat)  # (n_points, dim_mv)

            # Aggregate over sampling points first (mean pooling)
            x_n_agg = x_n.mean(dim=0)  # (dim_mv,)

            # Expand for input channels: (c_in, dim_mv)
            x_n_agg = x_n_agg.unsqueeze(0).expand(self.c_in, -1)  # (c_in, dim_mv)

            # Apply CGENN - it will handle the shape internally
            k_n = self.cgenn_nets[shell_idx](
                x_n_agg
            )  # Should output (c_in, c_out, dim_mv)

            # Check shape and permute if needed
            if len(k_n.shape) == 3:
                k_n = k_n.permute(1, 0, 2)  # (c_out, c_in, dim_mv)
            else:
                # If shape is (c_out, dim_mv), expand for c_in
                k_n = k_n.unsqueeze(1).expand(
                    -1, self.c_in, -1
                )  # (c_out, c_in, dim_mv)

            # Broadcast to spatial dimensions
            k_n = k_n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            k_n = k_n.expand(
                -1, -1, -1, self.kernel_size, self.kernel_size, self.kernel_size
            )

            # Apply learnable mask
            mask = self._compute_learnable_mask(spatial_grid, r, shell_idx)
            k_accum += self.mask_weights[shell_idx] * k_n * mask

        # Reshape for kernel head
        # k_accum shape: (c_out, c_in, dim_mv, k, k, k)
        # We need to reduce dim_mv dimension for standard convolution

        # Option 1: Average over multivector components
        k_reduced = k_accum.mean(dim=2)  # (c_out, c_in, k, k, k)

        # Apply learnable kernel head transformation
        # Input: (c_out, c_in, k, k, k)
        k_reshaped = k_reduced.reshape(
            1,
            self.c_out * self.c_in,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )

        # Use 3D conv to mix and transform kernel
        k_transformed = self.kernel_head(k_reshaped)  # (1, c_out*c_in, k, k, k)

        # Reshape to final kernel format
        k_final = k_transformed.reshape(
            self.c_out, self.c_in, self.kernel_size, self.kernel_size, self.kernel_size
        )

        return k_final

    def _create_spatial_grid(self, device):
        """Create coordinate grid for kernel positions."""
        coords = torch.arange(self.kernel_size, device=device)
        grid = torch.stack(
            torch.meshgrid(coords, coords, coords, indexing="ij"), dim=-1
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
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, k, k, k)

        return mask


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
        self.log("train_macro_recall", metrics["macro_recall"])
        self.log("train_mean_rank", metrics["mean_rank"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.out_channels)

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
        metrics = compute_metrics(y_hat, y, self.out_channels)

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
