import os
import sys

import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.pipeline import Pipeline
from pipeline.data_loader import NPZDataLoader
from clifford.model import CliffordSteerableNetwork
from utils.sampling_strategies import ProbabilisticSelectionTransform

if __name__ == "__main__":
    print("=" * 70)
    print("Clifford Steerable Convolution - Example Usage")
    print("=" * 70)
    
    transformer = ProbabilisticSelectionTransform(2048)
    def preprocess(blob: np.ndarray) -> np.ndarray:
        """ Create point cloud from voxel grid blob. """
        blob = transformer.preprocess(blob)    
        # blob consists of 0s and 1s, get 1s coordinates
        points = np.argwhere(blob > 0)
        # pad points to 2048
        points = np.pad(points, ((0, 2048 - points.shape[0]), (0, 0)), mode='constant', constant_values=0)
        points = points.reshape(8, 16, 16, 3).transpose(3, 0, 1, 2)  # (3, 8, 16, 16) (c_in, D, H, W)
        return points.astype(np.float32)
    
    # Create dataloader with point cloud preprocessing
    data_loader = NPZDataLoader(
        root_dir="../../data/cryoem_blobs/grouped_blobs",  # Change this to your data directory
        preprocess_fn=preprocess,
        npz_key=None,  # Use first key in NPZ file
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
        batch_size=16,
        num_workers=4,
        cache_data=False  # Set True if you have enough RAM
    )
    
    # Setup: 3D Euclidean space -> Cl(3,0)
    p, q = 3, 0
    
    model = CliffordSteerableNetwork(
        p=p, q=q,
        in_channels=3,
        hidden_channels=[32, 64],
        out_channels=569,  # 569 classes
        n_shells=3,
        kernel_size=3,
        learning_rate=1e-3
    )
    
    # Show model summary
    model.summary()
    
    # Build a pipeline
    pipeline = Pipeline(data_loader, model)
    
    # Train the model (scikit-learn style!)
    print("\n" + "=" * 70)
    print("Training model with .fit() method...")
    print("=" * 70)
    
    pipeline.fit(
        epochs=50,
        verbose=True,
        early_stopping_patience=10,
        checkpoint_path='best_model.pth'
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    metrics = pipeline.evaluate()
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2f}%")
    
    # Uncomment to plot training history (requires matplotlib)
    # pipeline.model.plot_history()
    