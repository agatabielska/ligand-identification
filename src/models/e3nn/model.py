import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from e3nn.o3 import spherical_harmonics, Irreps, Linear
from typing import Optional, Dict, List



class E3NNPointCloudModel(nn.Module):
    """E3NN model operating on point cloud representation of density."""

    def __init__(self, num_classes: int, max_points: int = 512):
        super().__init__()
        self.max_points = max_points
        self.lmax = 2

        # Irreps structure matching features
        # 0e (scalars): 16 features
        # 1o (vectors): 8 * 3 = 24 features (8 3D vectors)
        # 2e (rank-2): 4 * 5 = 20 features (4 5D tensors)
        self.irreps_in = Irreps("16x0e + 8x1o + 4x2e")
        self.irreps_hidden1 = Irreps("32x0e + 8x1o + 4x2e")
        self.irreps_hidden2 = Irreps("64x0e + 4x1o + 2x2e")
        self.irreps_scalar = Irreps("128x0e")

        # E3NN layers
        self.e3nn_layer1 = Linear(self.irreps_in, self.irreps_hidden1)
        self.e3nn_layer2 = Linear(self.irreps_hidden1, self.irreps_hidden2)
        self.e3nn_layer3 = Linear(self.irreps_hidden2, self.irreps_scalar)

        # Classifier with dropout
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def compute_point_features(self, points, density_values):
        """
        Compute rotation-equivariant features for each point.
        Features must match irreps: 16x0e + 8x1o + 4x2e
        Total: 16 + 8*3 + 4*5 = 16 + 24 + 20 = 60 dimensions

        Args:
            points: (B, N, 3) normalized coordinates
            density_values: (B, N) scalar density at each point

        Returns:
            features: (B, N, 60) equivariant features
        """
        batch_size, n_points, _ = points.shape
        all_features = []

        for b in range(batch_size):
            pts = points[b]  # (N, 3)
            vals = density_values[b]  # (N,)

            # Compute center of mass
            if vals.sum() > 0:
                com = (pts * vals.unsqueeze(-1)).sum(dim=0) / vals.sum()
            else:
                com = pts.mean(dim=0)

            # Center points around COM
            centered_pts = pts - com

            # Radial distance
            r = torch.norm(centered_pts, dim=1)
            r_safe = torch.clamp(r, min=1e-6)
            directions = centered_pts / r_safe.unsqueeze(-1)

            # Spherical harmonics
            sh_l0 = spherical_harmonics(0, directions, normalize=True)  # (N, 1)
            sh_l1 = spherical_harmonics(1, directions, normalize=True)  # (N, 3)
            sh_l2 = spherical_harmonics(2, directions, normalize=True)  # (N, 5)

            # Scalar features (16x0e = 16 scalars)
            scalar_list = [
                vals,
                r_safe,
                sh_l0[:, 0],
                vals * r_safe,
                vals ** 2,
                torch.log1p(r_safe),
                torch.sqrt(vals + 1e-6),
                r_safe ** 2,
                vals * sh_l0[:, 0],
                torch.exp(-r_safe),
                vals ** 3,
                r_safe ** 0.5,
                vals * torch.log1p(r_safe),
                torch.sigmoid(vals),
                torch.sigmoid(r_safe),
                vals * r_safe ** 2,
            ]
            scalar_features = torch.stack(scalar_list, dim=1)

            # Vector features (8x1o = 8 vectors of 3 components = 24 features)
            vector_list = [
                centered_pts,
                centered_pts * vals.unsqueeze(-1),
                sh_l1,
                centered_pts * r_safe.unsqueeze(-1),
                directions,
                directions * vals.unsqueeze(-1),
                centered_pts * (vals ** 2).unsqueeze(-1),
                sh_l1 * r_safe.unsqueeze(-1),
            ]
            vector_features = torch.cat(vector_list, dim=1)

            # Tensor features (4x2e = 4 tensors of 5 components = 20 features)
            tensor_list = [
                sh_l2,
                sh_l2 * vals.unsqueeze(-1),
                sh_l2 * r_safe.unsqueeze(-1),
                sh_l2 * torch.log1p(r_safe).unsqueeze(-1),
            ]
            tensor_features = torch.cat(tensor_list, dim=1)

            # Concatenate all features: 16 + 24 + 20 = 60
            point_features = torch.cat([
                scalar_features,
                vector_features,
                tensor_features,
            ], dim=1)

            all_features.append(point_features)

        return torch.stack(all_features)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: (B, N, 4) tensor where last dim is [x, y, z, density]

        Returns:
            logits: (B, num_classes) classification logits
        """
        # Separate coordinates and density features
        # x is assumed to be (B, N, 4)
        points = x[:, :, :3]       # (B, N, 3)
        density_values = x[:, :, 3] # (B, N)

        # Compute equivariant features
        point_features = self.compute_point_features(points, density_values)

        # Pass through E3NN layers
        x = self.e3nn_layer1(point_features)
        x = self.e3nn_layer2(x)
        x = self.e3nn_layer3(x)  # (B, N, 128)

        # Global pooling (rotation invariant)
        x = torch.max(x, dim=1)[0]  # (B, 128) - max pooling over points

        # Classification head
        x = torch.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))

        return x


class E3NNModelWrapper:
    """Wrapper class to match the interface of CliffordSteerableNetwork."""

    def __init__(
            self,
            num_classes: int,
            max_points: int = 512,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            save_every_epoch: int = 10,
            device: str = None
    ):
        self.num_classes = num_classes
        self.max_points = max_points
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_every_epoch = save_every_epoch

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = E3NNPointCloudModel(num_classes, max_points).to(self.device)

        # History tracking - adapted to match Clifford model
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_accuracy = 0.0

    def summary(self):
        """Print model summary."""
        print("\n" + "=" * 70)
        print("E3NN Point Cloud Model Summary")
        print("=" * 70)
        print(f"Number of classes: {self.num_classes}")
        print(f"Max points: {self.max_points}")
        print(f"Device: {self.device}")
        print(f"\nIrreps structure:")
        print(f"  Input:    {self.model.irreps_in}")
        print(f"  Hidden1:  {self.model.irreps_hidden1}")
        print(f"  Hidden2:  {self.model.irreps_hidden2}")
        print(f"  Output:   {self.model.irreps_scalar}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("=" * 70 + "\n")

    def _train_epoch(self, train_loader, optimizer, criterion):
        """Single training epoch - adapted from Clifford model."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        top_10_correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"\tTraining batch {batch_idx + 1}/{len(train_loader)}", end='\r')

            # Move data to device
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Top-10 accuracy calculation (adapted from Clifford model)
            top_10 = output.topk(10, dim=1).indices
            top_10_correct += np.sum(
                [1 if target[i] in top_10[i] else 0 for i in range(target.size(0))]
            )
            total += target.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        top_10_accuracy = 100.0 * top_10_correct / total

        return avg_loss, accuracy, top_10_accuracy

    def _validate_epoch(self, val_loader, criterion):
        """Single validation epoch - adapted from Clifford model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        top_10_correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

                # Top-10 accuracy calculation (adapted from Clifford model)
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
        """Train the model - adapted from Clifford model."""
        # Setup optimizer
        if optimizer is None:
            optimizer = optim.AdamW(
                self.model.parameters(),
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

                # Print progress (adapted from Clifford model)
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
        """Evaluate the model - adapted from Clifford model."""
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
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)

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
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'device': str(self.device)
        }

        torch.save(checkpoint, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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