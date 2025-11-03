from utils.sampling_strategies import ProbabilisticSelectionTransform
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Tuple, Dict, List
from sklearn.model_selection import train_test_split
import json

class NPZDataset(Dataset):
    """
    Dataset for loading NPZ files with labels from folder structure.
    
    Expected directory structure:
        root_dir/
            class_1/
                sample1.npz
                sample2.npz
            class_2/
                sample3.npz
                sample4.npz
            ...
    """
    
    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        label_to_name: Dict[int, str],
        preprocess_fn: Optional[Callable] = None,
        npz_key: Optional[str] = None,
        transform: Optional[Callable] = None,
        cache_data: bool = False
    ):
        """
        Args:
            file_paths: List of paths to NPZ files
            labels: List of integer labels corresponding to file_paths
            label_to_name: Dictionary mapping label index to class name
            preprocess_fn: Optional preprocessing function (array -> processed_array)
            npz_key: Key to extract from NPZ file. If None, uses first key found
            transform: Optional additional transforms (for augmentation)
            cache_data: If True, cache all data in memory (faster but uses more RAM)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.label_to_name = label_to_name
        self.preprocess_fn = preprocess_fn
        self.npz_key = npz_key
        self.transform = transform
        self.cache_data = cache_data
        
        # Cache for data if enabled
        self._cache = {} if cache_data else None
        
        # Preload all data if caching enabled
        if self.cache_data:
            print(f"Caching {len(self.file_paths)} files in memory...")
            for idx in range(len(self.file_paths)):
                self._cache[idx] = self._load_file(idx)
            print("Caching complete!")
    
    def _load_file(self, idx: int) -> np.ndarray:
        """Load and preprocess a single NPZ file."""
        file_path = self.file_paths[idx]
        
        # Load NPZ file
        data = np.load(file_path)
        
        # Extract array using key or first key
        if self.npz_key is not None:
            array = data[self.npz_key]
        else:
            # Use first key found
            keys = list(data.keys())
            if len(keys) == 0:
                raise ValueError(f"NPZ file {file_path} is empty")
            array = data[keys[0]]
        
        # Apply preprocessing
        if self.preprocess_fn is not None:
            array = self.preprocess_fn(array)
        
        return array
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            data: Preprocessed data as tensor
            label: Integer label
        """
        # Load from cache or disk
        if self.cache_data:
            array = self._cache[idx]
        else:
            array = self._load_file(idx)
        
        # Apply additional transforms (e.g., augmentation)
        if self.transform is not None:
            array = self.transform(array)
        
        # Convert to tensor
        if not isinstance(array, torch.Tensor):
            data = torch.from_numpy(array).float()
        else:
            data = array.float()
        
        label = self.labels[idx]
        
        return data, label
    
    def get_label_name(self, label: int) -> str:
        """Get class name from label index."""
        return self.label_to_name[label]


class NPZDataLoader:
    """
    Manager for loading NPZ datasets with train/val/test splits.
    """
    
    def __init__(
        self,
        root_dir: str,
        sampler = None,
        preprocess_fn: Optional[Callable] = None,
        npz_key: Optional[str] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 16,
        num_workers: int = 4,
        cache_data: bool = False,
        file_pattern: str = "*.npz",
        logging: bool = False
    ):
        """
        Args:
            root_dir: Root directory containing class folders
            sampler = Sampler for data loading
            preprocess_fn: Preprocessing function to apply to each array
            npz_key: Key to extract from NPZ files (None = use first key)
            train_split: Proportion of data for training (0-1)
            val_split: Proportion for validation (0-1)
            test_split: Proportion for testing (0-1)
            random_seed: Random seed for reproducibility
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes for loading
            cache_data: Whether to cache all data in memory
            file_pattern: Glob pattern for finding files (default: "*.npz")
            logging: Whether to print detailed logs
        """
        self.root_dir = Path(root_dir)
        self.sampler = sampler
        self.preprocess_fn = preprocess_fn
        self.npz_key = npz_key
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_data = cache_data
        self.file_pattern = file_pattern
        self.logging = logging
        
        # Validate splits
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Train, val, and test splits must sum to 1.0"
        
        # Scan directory and create datasets
        self._scan_directory()
        self._create_splits()
        
        # Initialize sampler with data source
        if self.sampler is not None:
            self.sampler.set_data_source(self.train_dataset)
        
    def _scan_directory(self):
        """Scan directory structure and collect files with labels."""
        if self.logging:
            print(f"Scanning directory: {self.root_dir}")
        
        if not self.root_dir.exists():
            raise ValueError(f"Directory not found: {self.root_dir}")
        
        # Get all class directories
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {self.root_dir}")
        
        # Sort for consistent ordering
        class_dirs = sorted(class_dirs, key=lambda x: x.name)
        
        # Create label mappings
        self.name_to_label = {d.name: idx for idx, d in enumerate(class_dirs)}
        self.label_to_name = {idx: d.name for idx, d in enumerate(class_dirs)}
        self.num_classes = len(class_dirs)
        
        print(f"Found {self.num_classes} classes:")
        if self.logging:
            for label, name in self.label_to_name.items():
                print(f"  {label}: {name}")
        
        # Collect all files
        self.all_file_paths = []
        self.all_labels = []
        
        for class_dir in class_dirs:
            label = self.name_to_label[class_dir.name]
            
            # Find all NPZ files in this class
            files = list(class_dir.glob(self.file_pattern))
            
            if len(files) == 0:
                print(f"Warning: No files found in {class_dir}")
                continue
            
            self.all_file_paths.extend(files)
            self.all_labels.extend([label] * len(files))
            
            if self.logging:
                print(f"  Class '{class_dir.name}': {len(files)} files")
        
        if self.logging:
            print(f"\nTotal files: {len(self.all_file_paths)}")
        
        # Convert to arrays
        self.all_file_paths = np.array(self.all_file_paths)
        self.all_labels = np.array(self.all_labels)
    
    def _create_splits(self):
        """Create train/val/test splits."""
        if self.logging:
            print(f"\nCreating splits (train={self.train_split}, val={self.val_split}, test={self.test_split})...")
            
        small_classes = []
        for label in range(self.num_classes):
            count = np.sum(self.all_labels == label)
            if count < 3:
                small_classes.append((self.label_to_name[label], count))
        
        # Remove small classes from dataset and add them to train set only
        if len(small_classes) > 0:
            print("\nWarning: The following classes have less than 3 samples and will be included only in the training set:")
            for class_name, count in small_classes:
                print(f"  Class '{class_name}': {count} samples")
            mask = np.isin(self.all_labels, [self.name_to_label[name] for name, _ in small_classes], invert=True)
            small_class_files = self.all_file_paths[~mask]
            small_class_labels = self.all_labels[~mask]
            self.all_file_paths = self.all_file_paths[mask]
            self.all_labels = self.all_labels[mask]
        
        # First split: test vs (train+val)
        train_val_files, test_files, train_val_labels, test_labels = train_test_split(
            self.all_file_paths,
            self.all_labels,
            test_size=self.test_split,
            random_state=self.random_seed,
            stratify=self.all_labels
        )
        
        # Second split: train vs val
        relative_val_size = self.val_split / (self.train_split + self.val_split)
        train_files, val_files, train_labels, val_labels = train_test_split(
            train_val_files,
            train_val_labels,
            test_size=relative_val_size,
            random_state=self.random_seed,
            stratify=train_val_labels
        )
        
        # Add small classes to training set
        if len(small_classes) > 0:
            train_files = np.concatenate([train_files, small_class_files])
            train_labels = np.concatenate([train_labels, small_class_labels])
        
        # Create datasets
        self.train_dataset = NPZDataset(
            list(train_files), list(train_labels), self.label_to_name,
            self.preprocess_fn, self.npz_key, cache_data=self.cache_data
        )
        
        self.val_dataset = NPZDataset(
            list(val_files), list(val_labels), self.label_to_name,
            self.preprocess_fn, self.npz_key, cache_data=self.cache_data
        )
        
        self.test_dataset = NPZDataset(
            list(test_files), list(test_labels), self.label_to_name,
            self.preprocess_fn, self.npz_key, cache_data=self.cache_data
        )
        
        if self.logging:
            print(f"Train set: {len(self.train_dataset)} samples")
            print(f"Val set: {len(self.val_dataset)} samples")
            print(f"Test set: {len(self.test_dataset)} samples")
        
        # Print class distribution
        if self.logging:
            self._print_class_distribution()
    
    def _print_class_distribution(self):
        """Print distribution of classes in each split."""
        print("\nClass distribution:")
        
        for split_name, labels in [
            ("Train", self.train_dataset.labels),
            ("Val", self.val_dataset.labels),
            ("Test", self.test_dataset.labels)
        ]:
            print(f"\n{split_name}:")
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                class_name = self.label_to_name[label]
                percentage = 100 * count / len(labels)
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def get_train_loader(self, shuffle: bool = True) -> DataLoader:
        """Get training DataLoader."""
        if self.sampler is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=self.sampler
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=True
            )
    def get_val_loader(self, shuffle: bool = False) -> DataLoader:
        """Get validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_test_loader(self, shuffle: bool = False) -> DataLoader:
        """Get test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all three DataLoaders at once."""
        return (
            self.get_train_loader(),
            self.get_val_loader(),
            self.get_test_loader()
        )
    
    def save_split_info(self, output_path: str):
        """Save split information to JSON file for reproducibility."""
        split_info = {
            'num_classes': self.num_classes,
            'label_to_name': self.label_to_name,
            'name_to_label': self.name_to_label,
            'train_files': [str(p) for p in self.train_dataset.file_paths],
            'val_files': [str(p) for p in self.val_dataset.file_paths],
            'test_files': [str(p) for p in self.test_dataset.file_paths],
            'train_labels': self.train_dataset.labels,
            'val_labels': self.val_dataset.labels,
            'test_labels': self.test_dataset.labels,
            'splits': {
                'train': self.train_split,
                'val': self.val_split,
                'test': self.test_split
            },
            'random_seed': self.random_seed
        }
        
        with open(output_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"\nSplit information saved to: {output_path}")
    
    def summary(self):
        """Print summary of the dataset."""
        print("\n" + "=" * 80)
        print("Dataset Summary")
        print("=" * 80)
        print(f"Root directory: {self.root_dir}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total samples: {len(self.all_file_paths)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Train/Val/Test split: {self.train_split}/{self.val_split}/{self.test_split}")
        print(f"Preprocessing: {'Yes' if self.preprocess_fn else 'No'}")
        print(f"Caching: {'Yes' if self.cache_data else 'No'}")
        print("=" * 80)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("NPZ DataLoader Example")
    print("=" * 80)
    
    # Example 1: Point cloud preprocessing
    print("\n" + "=" * 80)
    print("Example 1: Point Cloud DataLoader")
    print("=" * 80)
    
    transformer = ProbabilisticSelectionTransform(4096)
    def preprocess(blob: np.ndarray) -> np.ndarray:
        """ Create point cloud from voxel grid blob. """
        blob = transformer.preprocess(blob)    
        # blob consists of 0s and 1s, get 1s coordinates
        points = np.argwhere(blob > 0)
        # pad points to 4096
        points = np.pad(points, ((0, 4096 - points.shape[0]), (0, 0)), mode='constant', constant_values=0)
        return points.astype(np.float32)
    
    # Create dataloader with point cloud preprocessing
    data_loader = NPZDataLoader(
        root_dir="../../data/cryoem_blobs/grouped_blobs",  # Change this to your data directory
        sampler=None,
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
    
    # Get summary
    data_loader.summary()
    
    # Get loaders
    train_loader, val_loader, test_loader = data_loader.get_all_loaders()
    
    # Test loading a batch
    print("\nTesting data loading...")
    try:
        for batch_data, batch_labels in train_loader:
            print(f"Batch data shape: {batch_data.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            print(f"Data type: {batch_data.dtype}")
            print(f"Label range: [{batch_labels.min()}, {batch_labels.max()}]")
            break
    except Exception as e:
        print(f"Note: Actual data loading will work when you provide a valid data directory")
        print(f"Error: {e}")
    
    # Save split information
    print("\n" + "=" * 80)
    print("Saving split information...")
    print("=" * 80)
    
    try:
        data_loader.save_split_info("split_info.json")
    except:
        print("Split info will be saved when you provide a valid data directory")