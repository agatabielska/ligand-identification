import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import torch


@torch.no_grad()
def compute_metrics(y_hat, y, n_channels):
    loss = F.cross_entropy(y_hat, y).cpu().item()
    prob = F.softmax(y_hat, dim=1).cpu().numpy()
    y = y.squeeze().cpu().numpy()
    y_hat = y_hat.cpu().numpy()

    pred = y_hat.argmax(axis=1)
    labels = np.arange(n_channels)

    ranks = []
    for true_label, scores in zip(y, y_hat):
        sorted_idx = np.argsort(-scores)
        rank = np.where(sorted_idx == true_label)[0][0] + 1
        ranks.append(rank)

    return {
        "loss": loss,
        "acc": metrics.accuracy_score(y, pred),
        "top_10_acc": metrics.top_k_accuracy_score(y, y_hat, k=10, labels=labels),
        "brier_score": metrics.brier_score_loss(y, prob, labels=labels),
        "macro_recall": metrics.recall_score(
            y, pred, average="macro", zero_division=0, labels=labels
        ),
        "mean_rank": np.mean(ranks),
    }
