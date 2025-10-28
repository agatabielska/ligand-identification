import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class ScalarShell(nn.Module):
    """
    Function 1: SCALARSHELL (Learnable version)
    Generates scalar features on a shell at distance r from origin.
    """
    def __init__(self, p: int, q: int, n_points: int = 64, learnable_sigma: bool = True):
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
            self.register_buffer('sigma', torch.ones(1))
        
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
        eta_vv = torch.einsum('ni,ij,nj->n', v, eta_pq, v)
        
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
            nn.GELU()
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
        # Handle different input shapes
        original_shape = x.shape
        
        # If input is (..., dim_mv), expand to (..., 1, dim_mv) for processing
        if x.shape[-1] == self.dim_mv and len(x.shape) == 2:
            # Shape is (batch, dim_mv) -> treat as (batch, 1, dim_mv)
            x = x.unsqueeze(-2)  # (batch, 1, dim_mv)
            expand_channels = True
        else:
            expand_channels = False
        
        # Process multivector structure
        # x shape: (..., c_in, dim_mv)
        x_proj = self.mv_projection(x)  # (..., c_in, hidden_dim)
        
        # Mix channels with learned geometric operations
        # Einstein sum over input channels and hidden dimensions
        out = torch.einsum('...ih,oih->...oh', x_proj, self.channel_mix)
        
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
        hidden_dim: int = 64
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
        self.dim_mv = 2**(p + q)
        
        # Learnable radii for each shell
        self.radii = nn.Parameter(torch.linspace(0.5, 2.0, n_shells))
        
        # Learnable ScalarShells (one per shell)
        self.scalar_shells = nn.ModuleList([
            ScalarShell(p, q, n_sampling_points, learnable_sigma=True)
            for _ in range(n_shells)
        ])
        
        # Learnable embedding networks (scalar + vector -> multivector)
        self.embedders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + p + q, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.dim_mv)
            ) for _ in range(n_shells)
        ])
        
        # Learnable CGENN for each shell
        self.cgenn_nets = nn.ModuleList([
            CGENN(self.dim_mv, c_in, c_out, hidden_dim)
            for _ in range(n_shells)
        ])
        
        # Metric tensor
        eta_diag = torch.cat([torch.ones(p), -torch.ones(q)])
        self.register_buffer('eta', torch.diag(eta_diag))
        
        # Learnable kernel mask weights
        self.mask_weights = nn.Parameter(torch.ones(n_shells))
        
        # Learnable kernel head (final transformation)
        # Takes (1, c_out*c_in, k, k, k) and outputs (1, c_out*c_in, k, k, k)
        self.kernel_head = nn.Sequential(
            nn.Conv3d(c_out * c_in, c_out * c_in, 1, groups=c_out),
            nn.GroupNorm(c_out, c_out * c_in),
            nn.GELU(),
            nn.Conv3d(c_out * c_in, c_out * c_in, 1)
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
            self.c_out, self.c_in, self.dim_mv,
            self.kernel_size, self.kernel_size, self.kernel_size,
            device=device
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
            k_n = self.cgenn_nets[shell_idx](x_n_agg)  # Should output (c_in, c_out, dim_mv)
            
            # Check shape and permute if needed
            if len(k_n.shape) == 3:
                k_n = k_n.permute(1, 0, 2)  # (c_out, c_in, dim_mv)
            else:
                # If shape is (c_out, dim_mv), expand for c_in
                k_n = k_n.unsqueeze(1).expand(-1, self.c_in, -1)  # (c_out, c_in, dim_mv)
            
            # Broadcast to spatial dimensions
            k_n = k_n.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            k_n = k_n.expand(-1, -1, -1, self.kernel_size, self.kernel_size, self.kernel_size)
            
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
        batch_size_for_conv = k_reduced.shape[0]
        k_reshaped = k_reduced.reshape(1, self.c_out * self.c_in, 
                                       self.kernel_size, self.kernel_size, self.kernel_size)
        
        # Use 3D conv to mix and transform kernel
        k_transformed = self.kernel_head(k_reshaped)  # (1, c_out*c_in, k, k, k)
        
        # Reshape to final kernel format
        k_final = k_transformed.reshape(
            self.c_out, self.c_in,
            self.kernel_size, self.kernel_size, self.kernel_size
        )
        
        return k_final
    
    def _create_spatial_grid(self, device):
        """Create coordinate grid for kernel positions."""
        coords = torch.arange(self.kernel_size, device=device)
        grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
        center = self.kernel_size // 2
        grid = grid - center  # Center at origin
        return grid.float()
    
    def _compute_learnable_mask(self, spatial_grid: torch.Tensor, r: float, shell_idx: int):
        """Compute learnable geometric mask."""
        # Distance from center
        dist = torch.norm(spatial_grid, dim=-1)
        
        # Learnable Gaussian-like mask
        mask = torch.exp(-(dist - r)**2 / (2.0 * (0.5 + shell_idx * 0.2)**2))
        
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
        hidden_dim: int = 64
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
            p, q, c_in, c_out, n_shells, kernel_size,
            n_sampling_points, hidden_dim
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
            F_in, k,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias
        )
        
        return F_out


class CliffordSteerableNetwork(nn.Module):
    """
    Full learnable steerable network with multiple layers.
    Includes scikit-learn style fit/predict methods.
    """
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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.out_channels = out_channels
        
        # Build learnable steerable layers
        layers = []
        prev_c = in_channels
        
        for hidden_c in hidden_channels:
            layers.append(
                CliffordSteerableConvolution(
                    p, q, prev_c, hidden_c,
                    n_shells=n_shells,
                    kernel_size=kernel_size,
                    padding=kernel_size//2
                )
            )
            layers.append(nn.GroupNorm(min(8, hidden_c), hidden_c))
            layers.append(nn.GELU())
            prev_c = hidden_c
        
        self.features = nn.Sequential(*layers)
        
        # Global pooling and classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(prev_c, out_channels)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 10,
        optimizer=None,
        criterion=None,
        scheduler=None,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
        checkpoint_path: Optional[str] = None
    ):
        """
        Train the model (scikit-learn style).
        
        Args:
            train_loader: PyTorch DataLoader for training data
            val_loader: Optional DataLoader for validation
            epochs: Number of training epochs
            optimizer: Optional custom optimizer (default: Adam)
            criterion: Optional custom loss function (default: CrossEntropyLoss)
            scheduler: Optional learning rate scheduler
            verbose: Whether to print training progress
            early_stopping_patience: Stop if val loss doesn't improve for N epochs
            checkpoint_path: Path to save best model checkpoint
            
        Returns:
            self: The trained model
        """
        # Setup optimizer and criterion
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion
            )
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                # Print progress
                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        
                        # Save checkpoint
                        if checkpoint_path is not None:
                            self.save(checkpoint_path)
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered after {epoch+1} epochs")
                        break
            else:
                if verbose:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        return self
    
    def _train_epoch(self, train_loader, optimizer, criterion):
        """Single training epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader, criterion):
        """Single validation epoch."""
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def predict(self, x, batch_size: int = 32, return_probs: bool = False):
        """
        Make predictions on input data.
        
        Args:
            x: Input data (numpy array or torch tensor)
            batch_size: Batch size for prediction
            return_probs: If True, return probabilities; else return class labels
            
        Returns:
            predictions: Class labels or probabilities
        """
        self.eval()
        
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        
        # Create dataset and loader
        dataset = torch.utils.data.TensorDataset(x)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        
        predictions = []
        
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                output = self(batch)
                
                if return_probs:
                    probs = F.softmax(output, dim=1)
                    predictions.append(probs.cpu())
                else:
                    pred = output.argmax(dim=1)
                    predictions.append(pred.cpu())
        
        predictions = torch.cat(predictions, dim=0)
        
        return predictions.numpy()
    
    def predict_proba(self, x, batch_size: int = 32):
        """
        Predict class probabilities.
        
        Args:
            x: Input data
            batch_size: Batch size for prediction
            
        Returns:
            probabilities: Array of shape (n_samples, n_classes)
        """
        return self.predict(x, batch_size=batch_size, return_probs=True)
    
    def evaluate(self, test_loader, criterion=None):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            metrics: Dictionary with loss and accuracy
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        test_loss, test_acc = self._validate_epoch(test_loader, criterion)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'history': self.history,
            'device': self.device
        }
        torch.save(checkpoint, path)
        
    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)
        return self
    
    def plot_history(self):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot loss
            ax1.plot(self.history['train_loss'], label='Train Loss')
            if self.history['val_loss']:
                ax1.plot(self.history['val_loss'], label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(self.history['train_acc'], label='Train Acc')
            if self.history['val_acc']:
                ax2.plot(self.history['val_acc'], label='Val Acc')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Training and Validation Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Install it to plot training history.")
    
    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print("Clifford Steerable Network Summary")
        print("=" * 70)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
        print(f"Output classes: {self.out_channels}")
        print("=" * 70)
        
        # Print layer info
        print("\nArchitecture:")
        print(self.features)
        print(self.classifier)
        print("=" * 70)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Clifford Steerable Convolution - Example Usage")
    print("=" * 70)
    
    # Setup: 3D Euclidean space -> Cl(3,0)
    p, q = 3, 0
    
    # Create learnable steerable network with 219 classes
    model = CliffordSteerableNetwork(
        p=p, q=q,
        in_channels=4,
        hidden_channels=[32, 64, 128],  # Increased capacity for more classes
        out_channels=219,  # 219 classes
        n_shells=3,
        kernel_size=3,
        learning_rate=1e-3
    )
    
    # Show model summary
    model.summary()
    
    print("\n" + "=" * 70)
    print("Creating synthetic dataset for demonstration...")
    print("=" * 70)
    
    # Create synthetic dataset
    n_train = 500  # Increased for more classes
    n_val = 100
    n_test = 100
    
    X_train = torch.randn(n_train, 1, 32, 32, 32)
    y_train = torch.randint(0, 219, (n_train,))  # 219 classes
    
    X_val = torch.randn(n_val, 1, 32, 32, 32)
    y_val = torch.randint(0, 219, (n_val,))
    
    X_test = torch.randn(n_test, 1, 32, 32, 32)
    y_test = torch.randint(0, 219, (n_test,))
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)  # Smaller batch for more classes
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: 219")
    
    # Train the model (scikit-learn style!)
    print("\n" + "=" * 70)
    print("Training model with .fit() method...")
    print("=" * 70)
    
    model.fit(
        train_loader,
        val_loader=val_loader,
        epochs=50,
        verbose=True,
        early_stopping_patience=10,
        checkpoint_path='best_model_219classes.pth'
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    metrics = model.evaluate(test_loader)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    
    # Make predictions (scikit-learn style!)
    print("\n" + "=" * 70)
    print("Making predictions with .predict() method...")
    print("=" * 70)
    
    # Predict classes
    predictions = model.predict(X_test[:5])
    print(f"Predicted classes: {predictions}")
    print(f"True classes: {y_test[:5].numpy()}")
    
    # Predict probabilities
    probabilities = model.predict_proba(X_test[:5])
    print(f"\nPredicted probabilities shape: {probabilities.shape}")
    print(f"Top-5 predicted classes for first sample:")
    top5_indices = probabilities[0].argsort()[-5:][::-1]
    for i, idx in enumerate(top5_indices):
        print(f"  {i+1}. Class {idx}: {probabilities[0][idx]:.4f}")
    
    # Uncomment to plot training history (requires matplotlib)
    # model.plot_history()