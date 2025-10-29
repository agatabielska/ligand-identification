import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import glob
from scipy.ndimage import rotate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import json
from collections import Counter

from pointcloud_detector import PointCloudE3NNModel, load_and_process


def get_label_from_filename(filepath):
    """Extract ligand name from filename."""
    name = Path(filepath).stem
    parts = name.split("_")
    ligand = parts[-1].upper() if len(parts) > 3 else "UNKNOWN"
    return ligand


def evaluate_model(model, files, encoder, device, sampling_methods, desc="Evaluation"):
    """Comprehensive evaluation on a dataset."""
    model.eval()

    results_per_method = {method: {
        'predictions': [],
        'labels': [],
        'confidences': []
    } for method in sampling_methods}

    with torch.no_grad():
        for filepath in files:
            density = load_and_process(filepath)
            label_name = get_label_from_filename(filepath)
            label = encoder.transform([label_name])[0]

            x = torch.tensor(np.ascontiguousarray(density), dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0).to(device)

            # Test each sampling method
            for method in sampling_methods:
                logits, _ = model(x, sampling_method=method)
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1).item()
                conf = probs[0, pred].item()

                results_per_method[method]['predictions'].append(pred)
                results_per_method[method]['labels'].append(label)
                results_per_method[method]['confidences'].append(conf)

    # Compute metrics for each method
    metrics = {}
    for method in sampling_methods:
        preds = np.array(results_per_method[method]['predictions'])
        labels = np.array(results_per_method[method]['labels'])
        confs = np.array(results_per_method[method]['confidences'])

        acc = accuracy_score(labels, preds) * 100
        f1 = f1_score(labels, preds, average='weighted') * 100
        avg_conf = np.mean(confs) * 100

        metrics[method] = {
            'accuracy': acc,
            'f1_score': f1,
            'avg_confidence': avg_conf,
            'predictions': preds,
            'labels': labels,
            'confidences': confs
        }
    return metrics

def print_evaluation_report(metrics, encoder, sampling_methods, title="Evaluation Report"):
    """Print evaluation report."""
    print(f"{title:^70}")

    # Per-method results
    print("\nPer-Method Performance:")
    print(f"{'Method':<15} {'Accuracy':<12} {'F1-Score':<12} {'Avg Confidence':<15}")

    for method in sampling_methods:
        acc = metrics[method]['accuracy']
        f1 = metrics[method]['f1_score']
        conf = metrics[method]['avg_confidence']
        print(f"{method:<15} {acc:>10.2f}%  {f1:>10.2f}%  {conf:>13.2f}%")




def train_model(model, train_files, val_files, encoder, device, epochs=30, lr=0.001):
    """Training loop for point cloud ligand identification."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0

    # Cycle through sampling methods during training for robustness
    sampling_methods = ['probabilistic', 'topk', 'fps', 'uniform']

    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_loss': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        np.random.shuffle(train_files)
        # Rotate sampling method each epoch
        sampling_method = sampling_methods[epoch % len(sampling_methods)]

        for filepath in train_files:
            density = load_and_process(filepath)
            # augmentation can be added here
            label_name = get_label_from_filename(filepath)
            label = encoder.transform([label_name])[0]

            x = torch.tensor(np.ascontiguousarray(density), dtype=torch.float32)
            x = x.unsqueeze(0).unsqueeze(0).to(device)
            y = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits, _ = model(x, sampling_method=sampling_method)
            loss = criterion(logits, y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += 1

        train_acc = train_correct / train_total * 100
        train_loss /= train_total

        # Validation - test on multiple sampling methods
        model.eval()
        val_results = {method: {'correct': 0, 'total': 0, 'loss': 0}
                       for method in sampling_methods}

        with torch.no_grad():
            for filepath in val_files:
                density = load_and_process(filepath)
                label_name = get_label_from_filename(filepath)
                label = encoder.transform([label_name])[0]
                x = torch.tensor(np.ascontiguousarray(density), dtype=torch.float32)
                x = x.unsqueeze(0).unsqueeze(0).to(device)
                y = torch.tensor([label], dtype=torch.long).to(device)

                # Test each sampling method
                for method in sampling_methods:
                    logits, _ = model(x, sampling_method=method)
                    loss = criterion(logits, y)
                    val_results[method]['loss'] += loss.item()
                    pred = logits.argmax(dim=1)
                    val_results[method]['correct'] += (pred == y).sum().item()
                    val_results[method]['total'] += 1

        # Average validation accuracy across all methods
        val_accs = []
        val_losses = []
        print(f"Epoch {epoch + 1}/{epochs} [method: {sampling_method}]")
        print(f"  Train: loss={train_loss:.3f}, acc={train_acc:.1f}%")

        for method in sampling_methods:
            acc = val_results[method]['correct'] / val_results[method]['total'] * 100
            loss = val_results[method]['loss'] / val_results[method]['total']
            val_accs.append(acc)
            val_losses.append(loss)
            print(f"  Val ({method:>12}): loss={loss:.3f}, acc={acc:.1f}%")

        val_acc = np.mean(val_accs)
        val_loss = np.mean(val_losses)
        print(f"  Val (avg): {val_acc:.1f}%")

        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_pointcloud_model.pt")
            print("New best model saved!\n")
        else:
            print()

    return best_val_acc, history


def main():
    print("Training Point Cloud E3NN Ligand Identifier")

    all_files = sorted(glob.glob("cryoem_blobs/*.npz"))
    print(f"Found {len(all_files)} total files")

    ligand_names = [get_label_from_filename(f) for f in all_files]

    # encoder for all data
    encoder = LabelEncoder()
    encoder.fit(ligand_names)
    num_classes = len(encoder.classes_)
    joblib.dump(encoder, "ligand_encoder.pkl")
    print(f"Encoder with {num_classes} classes: {encoder.classes_}\n")

    counts = Counter(ligand_names)
    print("Ligand counts:", counts)
    frequent_files = [f for f in all_files if counts[get_label_from_filename(f)] >= 2]
    rare_files = [f for f in all_files if counts[get_label_from_filename(f)] < 2]

    train_val_files, test_files = train_test_split(
        frequent_files, test_size=0.2, random_state=42,
        stratify=[encoder.transform([get_label_from_filename(f)])[0] for f in frequent_files]
    )

    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, random_state=42,
        stratify=[encoder.transform([get_label_from_filename(f)])[0] for f in train_val_files]
    )

    train_files.extend(rare_files)

    print(f"\nDataset Split:")
    print(f"  Train: {len(train_files)} samples")
    print(f"  Val:   {len(val_files)} samples")
    print(f"  Test:  {len(test_files)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    model = PointCloudE3NNModel(num_classes, max_points=512).to(device)
    print("Starting Training")

    # Limit training samples for speed if needed
    train_subset = train_files[:min(800, len(train_files))]
    val_subset = val_files[:min(200, len(val_files))]

    train_model(
        model,
        train_subset,
        val_subset,
        encoder,
        device,
        epochs=20,
        lr=0.001,
    )

    print("Loading Best Model for Final Evaluation")
    model.load_state_dict(torch.load("best_pointcloud_model.pt", map_location=device))

    sampling_methods = ['probabilistic', 'topk', 'fps', 'uniform']

    # Evaluate on validation set
    val_metrics = evaluate_model(model, val_files, encoder, device, sampling_methods, "Validation")
    print_evaluation_report(val_metrics, encoder, sampling_methods, "VALIDATION SET RESULTS")

    # Evaluate on test set (held-out data)
    test_metrics = evaluate_model(model, test_files, encoder, device, sampling_methods, "Test")
    print_evaluation_report(test_metrics, encoder, sampling_methods, "TEST SET RESULTS (HELD-OUT)")

    print(f"Model saved to: best_pointcloud_model.pt")



if __name__ == "__main__":
    main()