import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline.pipeline import Pipeline
from pipeline.data_loader import NPZDataLoader
from pipeline.samplers import StochasticSampler
from models.e3nn.model import E3NNPointCloudModel
from utils.sampling_strategies import ProbabilisticSelectionTransform
import numpy as np

# Configuration
MAX_POINTS = 2000
BATCH_SIZE = 16
NUM_EPOCHS = 100

# Initialize the transformer
transformer = ProbabilisticSelectionTransform(max_blob_size=MAX_POINTS)


def preprocess_pointcloud(blob: np.ndarray) -> np.ndarray:
    """Enhanced preprocessing for X-ray density data."""
    sampled_blob = transformer.preprocess(blob)
    coords = np.argwhere(sampled_blob > 0)
    values = sampled_blob[sampled_blob > 0]

    if len(coords) == 0:
        return np.zeros((MAX_POINTS, 4), dtype=np.float32)

    # Normalize by actual point cloud extent
    center = coords.mean(axis=0)  # Use actual center of mass
    centered_coords = coords - center
    scale = np.percentile(np.linalg.norm(centered_coords, axis=1), 95)  # Robust scaling
    scale = max(scale, 1.0)  # Prevent division by zero
    norm_coords = centered_coords / scale
    points = np.column_stack([norm_coords, values])

    current_points = points.shape[0]
    if current_points < MAX_POINTS:
        padding = np.zeros((MAX_POINTS - current_points, 4), dtype=np.float32)

        points = np.vstack([points, padding])

    elif current_points > MAX_POINTS:
        points = points[:MAX_POINTS, :]


    return points.astype(np.float32)



if __name__ == "__main__":
    scripts_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(scripts_path, '..', '..'))

    print("=" * 70)
    print("E3NN Model - Training on X-ray Data")
    print("=" * 70)

    # Create sampler
    sampler = StochasticSampler(
        num_samples=BATCH_SIZE * 5000,  # Number of samples per epoch
        random_seed=42,
        replacement=False
    )

    # Create dataloader
    data_loader = NPZDataLoader(
        root_dir=os.path.join(project_root, 'data/xray_blobs/'),
        train_folder='xray_train',
        val_folder='xray_holdout',
        test_folder=None,
        preprocess_fn=preprocess_pointcloud,
        npz_key=None,  # Use first key in NPZ file
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        random_seed=42,
        batch_size=BATCH_SIZE,
        num_workers=4,
        cache_data=False,
        sampler=sampler
    )

    print(f"\nDataset loaded:")
    print(f"  Train samples: {len(data_loader.train_dataset)}")
    print(f"  Val samples:   {len(data_loader.val_dataset)}")
    print(f"  Test samples:  {len(data_loader.test_dataset)}")

    # Get number of classes from dataloader
    num_classes = 219

    # Initialize E3NN model (no wrapper needed!)
    model = E3NNPointCloudModel(
        num_classes=num_classes,
        max_points=MAX_POINTS,
        learning_rate=1e-3,
        weight_decay=1e-4,
        save_every_epoch=10
    )

    # Show model summary
    model.summary()

    # Build pipeline
    pipeline = Pipeline(data_loader, model)

    # Train the model
    print("\n" + "=" * 70)
    print("Training E3NN model...")
    print("=" * 70)

    pipeline.fit(
        epochs=NUM_EPOCHS,
        verbose=True,
        early_stopping_patience=15,
        checkpoint_path=os.path.join(project_root, 'data/checkpoints/best_e3nn_model.pth')
    )

    # Save final model
    final_model_path = os.path.join(project_root, 'data/checkpoints/final_e3nn_model.pth')
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)

    metrics = pipeline.evaluate()
    print(f"\nTest Results:")
    print(f"  Loss:     {metrics['test_loss']:.4f}")
    print(f"  Accuracy: {metrics['test_accuracy']:.2f}%")

    # Optionally plot training history
    try:
        print("\nPlotting training history...")
        model.plot_history()
    except:
        print("Could not plot training history (matplotlib might not be available)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)