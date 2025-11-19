import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from e3nn.o3 import spherical_harmonics, Irreps, Linear
from typing import Optional, Dict
import time

class E3NNPointCloudModel(nn.Module):
    """E3NN model operating on point cloud representation of density."""

    def __init__(
            self,
            num_classes: int,
            max_points: int = 512,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            save_every_epoch: int = 10,
            device: str = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_points = max_points
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_every_epoch = save_every_epoch
        self.lmax = 2

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)


        self.irreps_in = Irreps("4x0e + 2x1o + 2x2e")
        self.irreps_hidden1 = Irreps("32x0e + 8x1o + 4x2e")
        self.irreps_hidden2 = Irreps("64x0e + 8x1o + 4x2e")
        self.irreps_hidden3 = Irreps("128x0e + 4x1o + 2x2e")
        self.irreps_scalar = Irreps("256x0e")
        # =====================================

        # E3NN layers
        self.e3nn_layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.e3nn_layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.e3nn_layer3 = Linear(self.irreps_hidden2, self.irreps_hidden3)
        self.e3nn_layer4 = Linear(self.irreps_hidden3, self.irreps_scalar)

        # Classifier with dropout (keep multi-scale pooling dimensions)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 256)  # 768 from multi-scale pooling
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # History tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_accuracy = 0.0

        # Move to device
        self.to(self.device)

    def compute_point_features(self, points, density_values):
        """
        VECTORIZED computation of rotation-equivariant features.
        NO BATCH LOOPS - operates on full (B, N, 3) tensors.

        Enhanced features:
        - 8 scalars: density, radius, log_radius, density*radius, + 4 k-NN features
        - 2 vectors: directions, sh_l1
        - 2 tensors: sh_l2, sh_l2*density

        Total: 8 + 2*3 + 2*5 = 24 features (same count, but better quality!)

        Args:
            points: (B, N, 3) normalized coordinates
            density_values: (B, N) scalar density at each point

        Returns:
            features: (B, N, 24) equivariant features
        """
        B, N, _ = points.shape

        # Compute center of mass per sample (vectorized)
        vals_sum = density_values.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        com = (points * density_values.unsqueeze(-1)).sum(dim=1, keepdim=True) / vals_sum.unsqueeze(-1)  # (B, 1, 3)

        # Center points around COM
        centered_pts = points - com  # (B, N, 3)

        # Radial distance
        r = torch.norm(centered_pts, dim=2, keepdim=True)  # (B, N, 1)
        r_safe = torch.clamp(r, min=1e-6)
        directions = centered_pts / r_safe  # (B, N, 3)

        # Spherical harmonics - reshape to (B*N, 3), compute, reshape back
        dirs_flat = directions.reshape(-1, 3)
        sh_l1 = spherical_harmonics(1, dirs_flat, normalize=True).reshape(B, N, 3)  # (B, N, 3)
        sh_l2 = spherical_harmonics(2, dirs_flat, normalize=True).reshape(B, N, 5)  # (B, N, 5)

        # Scalar features (8x0e = 8 scalars)
        density = density_values.unsqueeze(-1)  # (B, N, 1)
        log_r = torch.log1p(r_safe)  # (B, N, 1)
        density_r = density * r_safe  # (B, N, 1)

        scalar_features = torch.cat([
            density,  # raw density
            r_safe,  # radius
            log_r,  # log scale
            density_r,  # interaction term
        ], dim=-1)  # (B, N, 8)

        # Vector features (2x1o = 2 vectors * 3 components = 6 features)
        vector_features = torch.cat([
            directions,  # normalized directions
            sh_l1  # l=1 harmonics
        ], dim=-1)  # (B, N, 6)

        # Tensor features (2x2e = 2 tensors * 5 components = 10 features)
        tensor_features = torch.cat([
            sh_l2,  # l=2 harmonics
            sh_l2 * density  # modulated by density
        ], dim=-1)  # (B, N, 10)

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
        points = x[:, :, :3]  # (B, N, 3)
        density_values = x[:, :, 3]  # (B, N)

        # Compute equivariant features (vectorized - no loops!)
        point_features = self.compute_point_features(points, density_values)

        # Pass through E3NN layers
        x = self.e3nn_layer1(point_features)
        x = self.e3nn_layer2(x)
        x = self.e3nn_layer3(x)
        x = self.e3nn_layer4(x)  # (B, N, 256)

        # Global pooling (rotation invariant)
        x_max = torch.max(x, dim=1)[0]  # (B, 256) - captures strongest features
        x_mean = torch.mean(x, dim=1)  # (B, 256) - captures average distribution
        x_std = torch.std(x, dim=1)  # (B, 256) - captures variance/spread

        # Concatenate all pooling strategies
        x = torch.cat([x_max, x_mean, x_std], dim=-1)  # (B, 768)
        # Classification head
        x = torch.relu(self.fc1(self.dropout(x)))  # 768 → 256
        x = torch.relu(self.fc2(self.dropout(x)))  # 256 → 128
        x = self.fc3(x)  # 128 → num_classes

        return x

    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 70)
        print("E3NN Point Cloud Model Summary")
        print("=" * 70)
        print(f"Number of classes: {self.num_classes}")
        print(f"Max points: {self.max_points}")
        print(f"Device: {self.device}")
        print(f"\nIrreps structure:")
        print(f"  Input:    {self.irreps_in} - 20 features")
        print(f"    ├─ 4 scalars: density, r, log_r, density*r")
        print(f"    ├─ 2 vectors (6 features): directions, sh_l1")
        print(f"    └─ 2 tensors (10 features): sh_l2, sh_l2*density")
        print(f"  Hidden1:  {self.irreps_hidden1}")
        print(f"  Hidden2:  {self.irreps_hidden2}")
        print(f"  Hidden3:  {self.irreps_hidden3}")
        print(f"  Output:   {self.irreps_scalar}")
        print(f"\nPooling strategy: Multi-scale (max + mean + std)")
        print(f"  Pooled features: 256 × 3 = 768")
        print(f"\nClassifier architecture:")
        print(f"  fc1: 768 → 256 (+ ReLU + Dropout)")
        print(f"  fc2: 256 → 128 (+ ReLU + Dropout)")
        print(f"  fc3: 128 → {self.num_classes}")

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 70 + "\n")

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Single training epoch."""
        self.train()
        total_loss = 0.0
        correct = 0
        top_10_correct = 0
        total = 0
        num_batches = len(train_loader)
        update_interval = max(1, num_batches // 5)  # Update every 20%

        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\tTraining batch {batch_idx + 1}/{len(train_loader)}", end='\r')

            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Top-10 accuracy calculation
            top_10 = output.topk(10, dim=1).indices
            top_10_correct += np.sum(
                [1 if target[i] in top_10[i] else 0 for i in range(target.size(0))]
            )
            total += target.size(0)
            if (batch_idx + 1) % update_interval == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)

                current_loss = total_loss / (batch_idx + 1)
                current_acc = 100.0 * correct / total

                print(f"    [{progress:5.1f}%] "
                      f"Batch {batch_idx + 1}/{num_batches} | "
                      f"Loss: {current_loss:.4f} | "
                      f"Acc: {current_acc:.2f}% | "
                      f"ETA: {eta / 60:.1f}m", flush=True)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        top_10_accuracy = 100.0 * top_10_correct / total

        return avg_loss, accuracy, top_10_accuracy

    def _validate_epoch(self, val_loader, criterion):
        """Single validation epoch."""
        self.eval()
        total_loss = 0.0
        correct = 0
        top_10_correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

                # Top-10 accuracy calculation
                top_10 = output.topk(10, dim=1).indices
                top_10_correct += np.sum(
                    [1 if target[i] in top_10[i] else 0 for i in range(target.size(0))]
                )
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        top_10_accuracy = 100.0 * top_10_correct / total

        return avg_loss, accuracy, top_10_accuracy

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
        """Train the model."""
        # Setup optimizer
        if optimizer is None:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )

        # Setup criterion
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        # Setup scheduler
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )

        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Save checkpoint every N epochs
            if checkpoint_path is not None and epoch % self.save_every_epoch == 0:
                self.save(f"{'/'.join(checkpoint_path.split('/')[:-1])}/checkpoint_epoch_{epoch + 1}.pth")

            # Training phase
            train_loss, train_acc, train_top_10 = self._train_epoch(
                train_loader, optimizer, criterion
            )

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation phase
            if val_loader is not None:
                val_loss, val_acc, val_top_10 = self._validate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Learning rate scheduling
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                # Print progress
                if verbose:
                    print(f"Epoch [{epoch + 1}/{epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Top-10: {train_top_10:.2f}% | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Top-10: {val_top_10:.2f}%")

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
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            else:
                if verbose:
                    print(f"Epoch [{epoch + 1}/{epochs}] "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Top-10: {train_top_10:.2f}%")

        return self

    def evaluate(self, test_loader, criterion=None) -> Dict[str, float]:
        """Evaluate the model."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        test_loss, test_acc, test_top_10 = self._validate_epoch(test_loader, criterion)

        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_top_10_accuracy': test_top_10
        }

        return metrics

    def predict(self, data_loader, return_probabilities: bool = False):
        """Make predictions."""
        self.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self(data)

                if return_probabilities:
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs.cpu().numpy())

                pred = output.argmax(dim=1)
                all_preds.append(pred.cpu().numpy())

        predictions = np.concatenate(all_preds)

        if return_probabilities:
            probabilities = np.concatenate(all_probs)
            return predictions, probabilities

        return predictions

    def save(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'history': self.history,
            'device': str(self.device),
            'num_classes': self.num_classes,
            'max_points': self.max_points,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
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