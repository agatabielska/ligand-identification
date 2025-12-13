from src.models.clifford.model import CliffordSteerableNetwork
from src.models.e3nn.model import E3NNPointCloudModel
from src.pipeline.dataset import BlobDataset
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from src.utils.set_seed import set_seed
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import hydra
import torch
import torch.nn.functional as F
import json
import pandas as pd
from sklearn.metrics import confusion_matrix, top_k_accuracy_score


def load_class_mapping(mapping_file: Path):
    if not mapping_file.exists():
        return None
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()}


def transform_clifford(npz):
    indices = npz["indices"]
    points = np.pad(indices, ((0, 2000 - indices.shape[0]), (0, 0)), mode="constant", constant_values=0)
    points = points.reshape(5, 20, 20, 3).transpose(3, 0, 1, 2).astype(np.float32)
    return torch.from_numpy(points)


def transform_e3nn(npz):
    indices = npz["indices"]
    values = npz["values"]
    coords = indices.astype(np.float32)
    values = values.astype(np.float32)
    points = np.column_stack([coords, values])
    current_points = points.shape[0]
    if current_points < 2000:
        padding = np.zeros((2000 - current_points, 4), dtype=np.float32)
        points = np.vstack([points, padding])
    return torch.from_numpy(points)


def get_dataset(path: str, cfg, transform):
    return BlobDataset(
        path=path,
        transform=transform,
        normalize=cfg.train.normalize_data,
        cache=cfg.machine.cache_dataset,
        num_workers=cfg.machine.num_workers,
    )


def get_dataloader(dataset, cfg, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=cfg.machine.batch_size,
        shuffle=shuffle,
        num_workers=cfg.machine.num_workers if not cfg.machine.cache_dataset else 1,
        pin_memory=cfg.machine.pin_memory,
        persistent_workers=True if cfg.machine.num_workers > 0 else False,
    )


def collect_predictions(model, dataloader, device):
    model.eval()
    model.to(device)
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(cm, class_names, output_path, normalize=False):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
        fmt, title = '.2f', 'Normalized Confusion Matrix'
    else:
        fmt, title = 'd', 'Confusion Matrix'
    
    n_classes = len(class_names)
    fig_size = max(12, n_classes * 0.3)
    plt.figure(figsize=(fig_size, fig_size))
    mask = cm == 0
    
    sns.heatmap(
        cm, annot=False if n_classes > 50 else True, fmt=fmt, cmap='YlOrRd',
        xticklabels=class_names if n_classes <= 30 else False,
        yticklabels=class_names if n_classes <= 30 else False,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        mask=mask, linewidths=0.01 if n_classes <= 50 else 0,
    )
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    if n_classes <= 30:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top10_confusion_matrix(y_true, y_pred, class_names, output_path):
    unique, counts = np.unique(y_true, return_counts=True)
    top10_indices = unique[np.argsort(counts)[-10:]][::-1]
    
    mask = np.isin(y_true, top10_indices)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    idx_to_pos = {idx: i for i, idx in enumerate(top10_indices)}
    y_true_remapped = np.array([idx_to_pos[val] for val in y_true_filtered])
    
    y_pred_remapped, valid_mask = [], []
    for pred_val in y_pred_filtered:
        if pred_val in idx_to_pos:
            y_pred_remapped.append(idx_to_pos[pred_val])
            valid_mask.append(True)
        else:
            y_pred_remapped.append(0)
            valid_mask.append(False)
    
    y_pred_remapped = np.array(y_pred_remapped)
    valid_mask = np.array(valid_mask)
    y_true_final = y_true_remapped[valid_mask]
    y_pred_final = y_pred_remapped[valid_mask]
    
    top10_names = [class_names[idx] for idx in top10_indices]
    cm = confusion_matrix(y_true_final, y_pred_final, labels=range(len(top10_indices)))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=top10_names, yticklabels=top10_names,
        cbar_kws={'label': 'Count'}, linewidths=1, linecolor='gray',
        ax=ax, annot_kws={"size": 10}
    )
    
    plt.title('Confusion Matrix - Top 10 Most Frequent Classes', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def find_most_confused_pairs(cm, class_names, top_n=20):
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = cm[i, j]
                if count > 0:
                    confused_pairs.append({
                        'true_class': class_names[i],
                        'true_idx': i,
                        'pred_class': class_names[j],
                        'pred_idx': j,
                        'count': count,
                        'true_total': cm[i, :].sum(),
                        'percentage': 100 * count / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                    })
    confused_pairs.sort(key=lambda x: x['count'], reverse=True)
    return confused_pairs[:top_n]


def compute_per_class_metrics(y_true, y_pred, y_probs, class_names, actual_classes):
    metrics = []
    for i, class_idx in enumerate(actual_classes):
        true_binary = (y_true == i).astype(int)
        pred_binary = (y_pred == i).astype(int)
        
        tp = np.sum((true_binary == 1) & (pred_binary == 1))
        fp = np.sum((true_binary == 0) & (pred_binary == 1))
        fn = np.sum((true_binary == 1) & (pred_binary == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(true_binary)
        pred_confidence = np.mean(y_probs[pred_binary == 1, i]) if np.sum(pred_binary) > 0 else 0
        
        ranks = []
        for j in np.where(true_binary == 1)[0]:
            sorted_probs = np.argsort(y_probs[j])[::-1]
            rank = np.where(sorted_probs == i)[0][0] + 1
            ranks.append(rank)
        avg_rank = np.mean(ranks) if ranks else len(actual_classes)
        
        metrics.append({
            'class': class_names[i],
            'class_idx': class_idx,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'avg_confidence': pred_confidence,
            'avg_rank': avg_rank,
        })
    return pd.DataFrame(metrics)


def analyze_error_patterns(y_true, y_pred, y_probs):
    incorrect_mask = y_true != y_pred
    error_confidences = np.max(y_probs[incorrect_mask], axis=1) if np.sum(incorrect_mask) > 0 else []
    correct_confidences = np.max(y_probs[~incorrect_mask], axis=1) if np.sum(~incorrect_mask) > 0 else []
    avg_error_confidence = np.mean(error_confidences) if len(error_confidences) > 0 else 0
    avg_correct_confidence = np.mean(correct_confidences) if len(correct_confidences) > 0 else 0
    
    error_counts = defaultdict(int)
    for true_label in y_true[incorrect_mask]:
        error_counts[true_label] += 1
    
    return {
        'total_errors': np.sum(incorrect_mask),
        'avg_error_confidence': avg_error_confidence,
        'avg_correct_confidence': avg_correct_confidence,
        'error_counts': error_counts,
    }


def get_test_classes_from_dataset(dataset):
    all_labels = []
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        all_labels.append(label.item() if torch.is_tensor(label) else label)
    return sorted(set(all_labels))


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if not cfg.test.checkpoint_path:
        raise ValueError("Checkpoint path not set in config")

    set_seed(cfg.random_seed)
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    print("MODEL PERFORMANCE ANALYSIS")
    print("=" * 80)

    full_class_mapping = None
    if cfg.test.class_mapping_path:
        full_class_mapping = load_class_mapping(Path(cfg.test.class_mapping_path))
    else:
        full_class_mapping = load_class_mapping(Path(cfg.paths.test_data).parent / "class_mapping.json")

    if cfg.model.type == "clifford":
        transform = transform_clifford
    elif cfg.model.type == "e3nn":
        transform = transform_e3nn
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    test_dataset = get_dataset(cfg.paths.test_data, cfg, transform)
    test_dataloader = get_dataloader(test_dataset, cfg, shuffle=False)

    test_classes = get_test_classes_from_dataset(test_dataset)
    test_class_names = []
    for class_idx in test_classes:
        if full_class_mapping and class_idx in full_class_mapping:
            test_class_names.append(full_class_mapping[class_idx])
        else:
            test_class_names.append(f"Class_{class_idx}")

    checkpoint_path = Path(cfg.test.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if cfg.model.type == "clifford":
        model = CliffordSteerableNetwork.load_from_checkpoint(
            str(checkpoint_path), p=cfg.model.p, q=cfg.model.q,
            in_channels=cfg.model.in_channels, hidden_channels=cfg.model.hidden_channels,
            out_channels=cfg.train.out_channels, n_shells=cfg.model.n_shells,
            kernel_size=cfg.model.kernel_size, learning_rate=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )
    elif cfg.model.type == "e3nn":
        model = E3NNPointCloudModel.load_from_checkpoint(
            str(checkpoint_path), num_classes=cfg.train.out_channels,
            learning_rate=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred_raw, y_true_raw, y_probs_raw = collect_predictions(model, test_dataloader, device)
    
    y_probs_filtered = y_probs_raw[:, test_classes]
    class_idx_to_position = {cls: i for i, cls in enumerate(test_classes)}
    y_true = np.array([class_idx_to_position[label] for label in y_true_raw])
    y_pred = np.argmax(y_probs_filtered, axis=1)
    
    accuracy = np.mean(y_pred == y_true)
    
    results = {
        'accuracy': accuracy,
        'correct': np.sum(y_pred == y_true),
        'total': len(y_true),
        'n_classes': len(test_classes)
    }
    
    for k in [1, 5, 10, 20]:
        if k <= len(test_classes):
            results[f'top_{k}_accuracy'] = top_k_accuracy_score(y_true, y_probs_filtered, k=k)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plot_confusion_matrix(cm, test_class_names, output_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(cm, test_class_names, output_dir / "confusion_matrix_normalized.png", normalize=True)
    plot_top10_confusion_matrix(y_true, y_pred, test_class_names, output_dir / "confusion_matrix_top10.png")
    
    confused_pairs = find_most_confused_pairs(cm, test_class_names, top_n=min(20, len(test_classes) * 2))
    if confused_pairs:
        pd.DataFrame(confused_pairs).to_csv(output_dir / "most_confused_pairs.csv", index=False)
    
    df_metrics = compute_per_class_metrics(y_true, y_pred, y_probs_filtered, test_class_names, test_classes)
    df_metrics.to_csv(output_dir / "per_class_metrics.csv", index=False)
    
    macro_recall = df_metrics['recall'].mean()
    
    metrics_summary = {
        'macro_recall': macro_recall,
        'precision_mean': df_metrics['precision'].mean(),
        'precision_median': df_metrics['precision'].median(),
        'precision_std': df_metrics['precision'].std(),
        'recall_mean': df_metrics['recall'].mean(),
        'recall_median': df_metrics['recall'].median(),
        'recall_std': df_metrics['recall'].std(),
        'f1_mean': df_metrics['f1_score'].mean(),
        'f1_median': df_metrics['f1_score'].median(),
        'f1_std': df_metrics['f1_score'].std(),
    }
    
    error_analysis = analyze_error_patterns(y_true, y_pred, y_probs_filtered)
    results.update({
        'total_errors': error_analysis['total_errors'],
        'error_rate': error_analysis['total_errors'] / len(y_true),
        'avg_error_confidence': error_analysis['avg_error_confidence'],
        'avg_correct_confidence': error_analysis['avg_correct_confidence'],
        'confidence_gap': error_analysis['avg_correct_confidence'] - error_analysis['avg_error_confidence']
    })
    
    pd.DataFrame([results]).to_csv(output_dir / "overall_metrics.csv", index=False)
    pd.DataFrame([metrics_summary]).to_csv(output_dir / "metrics_summary.csv", index=False)
    
    print(f"\nAccuracy: {accuracy:.4f} ({results['correct']}/{results['total']})")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    for k in [1, 5, 10, 20]:
        if f'top_{k}_accuracy' in results:
            print(f"Top-{k} Accuracy: {results[f'top_{k}_accuracy']:.4f}")
    
    print(f"\nResults saved to: {output_dir.absolute()}")
    print("Generated files:")
    print("  - confusion_matrix.png")
    print("  - confusion_matrix_normalized.png")
    print("  - confusion_matrix_top10.png")
    print("  - per_class_metrics.csv")
    print("  - most_confused_pairs.csv")
    print("  - overall_metrics.csv")
    print("  - metrics_summary.csv")


if __name__ == "__main__":
    main()