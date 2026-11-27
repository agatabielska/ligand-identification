"""
1D CNN whose convolution kernels are generated from learnable radial "shells".

Inspired by Clifford-Steerable CNNs (Zhdanov et al., 2024) and Clifford Group
Equivariant Neural Networks (Ruhe et al., 2023). This model uses no Clifford
algebra and has no O(p, q) equivariance: it is an ordinary 1D CNN whose kernels
are generated, independently of the input, from Gaussian rings at learned radii.
"""

from src.utils.metrics import compute_metrics
from typing import List
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import torch


class ScalarShell(nn.Module):
    """
    Learnable scalar features on a shell of radius r.

    Holds `n_points` learnable points on the unit sphere of the
    (n_pos + n_neg)-dimensional feature space (not the kernel's 1-D axis; r is
    the only link between the two). For each point v it evaluates the quadratic
    form vᵀ·G·v at radius r, with G = diag(+1 x n_pos, -1 x n_neg), and maps it
    through a signed Gaussian of width sigma.
    """

    def __init__(
        self, n_pos: int, n_neg: int, n_points: int = 64, learnable_sigma: bool = True
    ):
        """
        Args:
            n_pos: Number of +1 entries on the diagonal of the metric G
            n_neg: Number of -1 entries on the diagonal of the metric G
            n_points: Number of learnable points on the unit sphere
            learnable_sigma: Whether sigma is a parameter (otherwise a fixed buffer)
        """
        super().__init__()
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.dim = n_pos + n_neg
        self.n_points = n_points

        # Learnable sigma parameter
        if learnable_sigma:
            self.sigma = nn.Parameter(torch.empty(1).uniform_(0.4, 0.6))
        else:
            self.register_buffer("sigma", torch.empty(1).uniform_(0.4, 0.6))

        # Learnable sampling points on unit sphere (normalized during forward)
        self.sampling_points = nn.Parameter(torch.randn(n_points, self.dim))

    def forward(self, metric: torch.Tensor, r: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            metric: Diagonal metric G of shape (dim, dim), dim = n_pos + n_neg
            r: Radius of the shell (a 0-d tensor taken from the learnable radii)

        Returns:
            s: Scalar feature at each point (n_points,)
            v: Unit-norm points (n_points, dim); not scaled by r
        """
        # Normalize sampling points to unit sphere
        v = F.normalize(self.sampling_points, dim=-1)

        # Quadratic form vᵀ·G·v (G = metric) for each unit-norm point.
        # Equals 1 everywhere when n_neg == 0; otherwise lies in [-1, 1].
        form_unit = torch.einsum("ni,ij,nj->n", v, metric, v)
        form_shell = r**2 * form_unit  # Scale by radius squared

        # s = sgn(form_shell) · exp(-|form_shell| / (2σ²))
        s = torch.sign(form_shell) * torch.exp(
            -torch.abs(form_shell) / (2 * self.sigma**2)
        )

        return s, v


class ChannelMixMLP(nn.Module):
    """
    Shared MLP followed by a learned gain for every (output, input) channel pair.

    A LayerNorm MLP maps a feature vector to `hidden_dim` units. Each
    (output channel, input channel) pair then rescales every hidden unit with its
    own learned gain (`channel_mix`), and a linear layer maps back to `feature_dim`.
    """

    def __init__(self, feature_dim: int, c_in: int, c_out: int, hidden_dim: int = 64):
        """
        Args:
            feature_dim: Width of the feature vectors (input and output)
            c_in: Input channels
            c_out: Output channels
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.c_in = c_in
        self.c_out = c_out

        # MLP applied to the feature vector
        self.feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # One learned gain per (output channel, input channel, hidden unit)
        self.channel_mix = nn.Parameter(torch.randn(c_out, c_in, hidden_dim))
        nn.init.xavier_uniform_(self.channel_mix)

        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature vectors (..., feature_dim)
        Returns:
            Per-channel-pair features (..., c_out, c_in)
        """
        h = self.feature_mlp(x)  # (..., hidden_dim)

        # Elementwise gain for each (o, i) pair; no sum over i, so the
        # input-channel axis is kept
        h = torch.einsum("...h,oih->...oih", h, self.channel_mix) # (..., c_out, c_in, hidden_dim)

        return self.output_proj(h).squeeze(-1) # (..., c_out, c_in)


class RadialShellKernel1d(nn.Module):
    """
    Generates a (c_out, c_in, kernel_size) Conv1d weight from learnable radial shells.

    For each of the `n_shells` shells (learned radius r_s):
      1. ScalarShell gives a scalar and a unit vector at each shell point; an MLP
         embeds each (scalar, vector) pair as a `feature_dim` vector, and the
         embeddings are averaged over the points.
      2. ChannelMixMLP turns that vector into features of shape
         (c_out, c_in, feature_dim).
      3. These are multiplied by a Gaussian ring over the kernel offsets,
         exp(-(|offset| - r_s)² / (2 w_s²)) with fixed width w_s = 0.5 + 0.2·s,
         and by a learnable per-shell weight.
    The shell contributions are summed and averaged over feature_dim, and a
    pointwise head (`kernel_head`) transforms the result. The kernel does not
    depend on the input.
    """

    def __init__(
        self,
        n_pos: int,
        n_neg: int,
        c_in: int,
        c_out: int,
        n_shells: int,
        kernel_size: int = 3,
        n_points: int = 64,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_pos: Number of +1 entries on the diagonal of the metric used by ScalarShell
            n_neg: Number of -1 entries on the diagonal of that metric
            c_in: Input channels
            c_out: Output channels
            n_shells: Number of radial shells
            kernel_size: Spatial kernel size
            n_points: Points per shell
            hidden_dim: Hidden width of the embedders and ChannelMixMLPs
        """
        super().__init__()
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.c_in = c_in
        self.c_out = c_out
        self.n_shells = n_shells
        self.kernel_size = kernel_size

        # Width of the latent feature vectors, tied to the number of +1/-1 entries in the metric
        self.feature_dim = 2 ** (n_pos + n_neg)

        # Learnable radii for each shell
        self.radii = nn.Parameter(torch.linspace(0.5, 2.0, n_shells))

        # One ScalarShell per shell
        self.scalar_shells = nn.ModuleList(
            [
                ScalarShell(n_pos, n_neg, n_points, learnable_sigma=True)
                for _ in range(n_shells)
            ]
        )

        # Embedders: (scalar, unit vector) -> feature vector, one per shell
        self.embedders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1 + n_pos + n_neg, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, self.feature_dim),
                )
                for _ in range(n_shells)
            ]
        )

        # One ChannelMixMLP per shell
        self.channel_mlps = nn.ModuleList(
            [ChannelMixMLP(self.feature_dim, c_in, c_out, hidden_dim) for _ in range(n_shells)]
        )

        # Diagonal metric G = diag(+1 x n_pos, -1 x n_neg)
        metric_diag = torch.cat([torch.ones(n_pos), -torch.ones(n_neg)])
        self.register_buffer("metric", torch.diag(metric_diag))

        # Learnable weight for each shell
        self.shell_weights = nn.Parameter(torch.ones(n_shells))

        # Pointwise head over the flattened (c_out, c_in) axis:
        # grouped 1x1 conv (mixes the c_in channels of each output channel),
        # GroupNorm, GELU, then a 1x1 conv over all c_out*c_in channels.
        # Takes (1, c_out*c_in, k) and outputs (1, c_out*c_in, k)
        self.kernel_head = nn.Sequential(
            nn.Conv1d(c_out * c_in, c_out * c_in, 1, groups=c_out),
            nn.GroupNorm(c_out, c_out * c_in),
            nn.GELU(),
            nn.Conv1d(c_out * c_in, c_out * c_in, 1),
        )

    def forward(self) -> torch.Tensor:
        """
        Generate the convolution kernel.

        Returns:
            k: Kernel of shape (c_out, c_in, kernel_size)
        """
        device = self.metric.device

        # Tap offsets from the kernel center
        offsets = self._kernel_offsets(device)

        # Initialize accumulated kernel
        k_accum = torch.zeros(
            self.c_out,
            self.c_in,
            self.kernel_size,
            device=device,
        )

        # Loop over shells
        for shell_idx in range(self.n_shells):
            r = self.radii[shell_idx]

            # Scalar and unit vector at each shell point
            s, v = self.scalar_shells[shell_idx](self.metric, r)  # (n_points,), (n_points, n_pos + n_neg)

            # Embed (scalar, unit vector) pairs as feature vectors
            sv = torch.cat([s.unsqueeze(-1), v], dim=-1)  # (n_points, 1 + n_pos + n_neg)
            point_feats = self.embedders[shell_idx](sv)  # (n_points, feature_dim)

            # Mean-pool over the shell points
            shell_feat = point_feats.mean(dim=0)  # (feature_dim,)

            # Per-(output, input) channel features for this shell
            shell_k = self.channel_mlps[shell_idx](shell_feat)  # (c_out, c_in, feature_dim)

            # Gaussian ring (only r is learnable)
            ring = self._shell_ring(offsets, r, shell_idx)
            k_accum += self.shell_weights[shell_idx] * shell_k.unsqueeze(-1) * ring

        # Flatten (c_out, c_in) into channels for the head
        k_reshaped = k_accum.reshape(
            1,
            self.c_out * self.c_in,
            self.kernel_size
        )

        k_transformed = self.kernel_head(k_reshaped)  # (1, c_out*c_in, k)

        # Back to Conv1d weight layout
        k_final = k_transformed.reshape(
            self.c_out, self.c_in, self.kernel_size
        )

        return k_final

    def _kernel_offsets(self, device):
        """Signed tap offsets from the kernel center, shape (k,)."""
        k = self.kernel_size
        return (torch.arange(k, device=device) - k // 2).float()

    def _shell_ring(
        self, offsets: torch.Tensor, r: torch.Tensor, shell_idx: int
    ):
        """Gaussian ring over kernel offsets, centered at radius r.

        The width is fixed at 0.5 + 0.2 * shell_idx; only r is learnable.
        """
        # Distance from center
        dist = offsets.abs()  # (k,)

        # Gaussian ring around radius r
        ring = torch.exp(-((dist - r) ** 2) / (2.0 * (0.5 + shell_idx * 0.2) ** 2))

        # (1, 1, k): broadcasts over c_out, c_in
        ring = ring.unsqueeze(0).unsqueeze(0)

        return ring


class RadialShellConv1d(nn.Module):
    """
    Conv1d whose weight is generated by a RadialShellKernel1d on every forward pass.
    """

    def __init__(
        self,
        n_pos: int,
        n_neg: int,
        c_in: int,
        c_out: int,
        n_shells: int = 3,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        n_points: int = 64,
        hidden_dim: int = 64,
    ):
        """
        Args:
            n_pos: Number of +1 entries on the diagonal of the metric used by ScalarShell
            n_neg: Number of -1 entries on the diagonal of that metric
            c_in: Input channels
            c_out: Output channels
            n_shells: Number of radial shells
            kernel_size: Kernel size
            stride: Convolution stride
            padding: Padding size
            n_points: Sampling points per shell
            hidden_dim: Hidden width of the kernel generator's networks
        """
        super().__init__()
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.c_in = c_in
        self.c_out = c_out
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # Kernel generator
        self.kernel_gen = RadialShellKernel1d(
            n_pos, n_neg, c_in, c_out, n_shells, kernel_size, n_points, hidden_dim
        )

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolve with the generated kernel.

        Args:
            x: Input feature map (B, c_in, L)

        Returns:
            out: Output feature map (B, c_out, L_out); L_out == L for stride 1
            and padding kernel_size // 2 with odd kernel_size
        """
        # Generate the kernel
        k = self.kernel_gen()  # (c_out, c_in, k)

        # Apply 1D convolution
        out = F.conv1d(
            x, k, stride=self.stride, padding=self.padding, bias=self.bias
        )

        return out


class RadialShellCNN1d(pl.LightningModule):
    """
    1D classifier: stacked RadialShellConv1d blocks, global average pooling and an MLP head.

    Input (B, in_channels, L) -> logits (B, num_classes).

    Args:
        n_pos, n_neg: Number of +1 / -1 entries on the diagonal of the metric
            used inside each ScalarShell
        in_channels: Channels of the input sequence
        hidden_channels: Output channels of each RadialShellConv1d block
        classifier_hidden: Hidden widths of the MLP head
        num_classes: Number of classes
        n_shells: Radial shells per kernel
        kernel_size: Kernel size of every block
        learning_rate, weight_decay: AdamW settings
        scheduler_factor, scheduler_patience: ReduceLROnPlateau settings
    """
    
    def __init__(
        self,
        n_pos: int = 3,
        n_neg: int = 0,
        in_channels: int = 1,
        hidden_channels: List[int] = [16, 32, 64],
        classifier_hidden: List[int] = [512],
        num_classes: int = 10,
        n_shells: int = 3,
        kernel_size: int = 3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience

        # Stack of RadialShellConv1d -> GroupNorm -> GELU blocks
        layers = []
        prev_c = in_channels

        for hidden_c in hidden_channels:
            layers.append(
                RadialShellConv1d(
                    n_pos,
                    n_neg,
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
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
        )
        for hidden in classifier_hidden:
            self.classifier.append(nn.Linear(prev_c, hidden))
            self.classifier.append(nn.ReLU())
            prev_c = hidden
        self.classifier.append(nn.Linear(prev_c, num_classes))

        # I have to use torchmetrics for macro recall, because it cannot be computed
        # from batches, and sklearn's recall_score doesn't work with accumulating the
        # statistics across batches.
        self.val_recall = torchmetrics.Recall(
            num_classes=num_classes,
            average="macro",
            task="multiclass",
            zero_division=0,
        )
        self.test_recall = torchmetrics.Recall(
            num_classes=num_classes,
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
        metrics = compute_metrics(y_hat, y, self.num_classes)

        self.log("train_loss", loss)
        self.log("train_acc", metrics["acc"])
        self.log("train_top_10_acc", metrics["top_10_acc"])
        self.log("train_brier_score", metrics["brier_score"])
        self.log("train_mean_rank", metrics["mean_rank"])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        metrics = compute_metrics(y_hat, y, self.num_classes)

        preds = y_hat.argmax(dim=1)
        target = y.view(-1)
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
        metrics = compute_metrics(y_hat, y, self.num_classes)

        preds = y_hat.argmax(dim=1)
        target = y.view(-1)
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
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        if grads:
            total_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
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
