import gc
import random

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import logger_wandb as log
import models
import os
from cfg import read_config
from data_new import BlobDataset, collation_fn_sparse
from tqdm import tqdm

os.makedirs("checkpoints", exist_ok=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, save_path="checkpoints/best_model_uniform.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False  # Initialize the file

    def __call__(self, val_loss, model):  # Pass the model here
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save the best weights
            torch.save(model.state_dict(), self.save_path)
            print(f"Validation loss decreased. Saving model to {self.save_path}")
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


if __name__ == "__main__":
    torch.manual_seed(23)
    torch.use_deterministic_algorithms(True)

    cfg = read_config("../cfg/train.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] != "cpu" else "cpu")
    cpu = torch.device("cpu")

    run = log.get_run()

    run.config.update({
        "dataset/name": cfg["dataset_dir"].split("/")[-1],
        "batch_accum":  cfg["accum_iter"],
        "batch_size":   cfg["batch_size"],
    })

    # Create a unified label_to_idx mapping from all three splits to ensure consistent label encoding
    from pathlib import Path
    all_classes = set()
    for split in ["train", "val", "test"]:
        split_path = Path(cfg["dataset_dir"] + "/" + split)
        class_dirs = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        all_classes.update(class_dirs)
    
    all_classes = sorted(all_classes)
    label_to_idx = {cls_name: idx for idx, cls_name in enumerate(all_classes)}

    train = BlobDataset(cfg["dataset_dir"] + "/train", label_to_idx=label_to_idx)
    val = BlobDataset(cfg["dataset_dir"] + "/val", label_to_idx=label_to_idx)
    test = BlobDataset(cfg["dataset_dir"] + "/test", label_to_idx=label_to_idx)

    g_train, g_test = torch.Generator(), torch.Generator()
    g_train.manual_seed(42)
    g_test.manual_seed(42)
    train_dataloader = DataLoader(dataset=train, batch_size=cfg["batch_size"], collate_fn=collation_fn_sparse,
                                  num_workers=cfg["no_workers"], worker_init_fn=seed_worker, generator=g_train,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val, batch_size=cfg["batch_size"], collate_fn=collation_fn_sparse,
                                 num_workers=cfg["no_workers"], worker_init_fn=seed_worker, generator=g_test,
                                 shuffle=False)
    test_dataloader = DataLoader(dataset=test, batch_size=cfg["batch_size"], collate_fn=collation_fn_sparse,
                                  num_workers=cfg["no_workers"], worker_init_fn=seed_worker, generator=g_test,
                                  shuffle=False)

    model = models.create(cfg["model"])
    print(model)
    model.to(device)
    save_path = "checkpoints/original_sched_best_model_{dataset}.pth".format(dataset=cfg["dataset_dir"].split("/")[-1])

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    
    # important: scheduler changed
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.1)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    # mindelta changed, not checked yet
    early_stopper = EarlyStopping(patience=5, min_delta=0.1, save_path=save_path)

    log.log_config(run=run, model=model, criterion=criterion, optimizer=optimizer, dataset=train)

    accum_iter = cfg["accum_iter"]
    for e in tqdm(range(cfg["epochs"]), desc="epoch"):
        # train_dataloader.dataset.sample(e)
        model.train()
        for idx, (coords, feats, labels) in tqdm(enumerate(train_dataloader), desc="batch", total=len(train_dataloader)):
            labels = labels.long().to(device=device)
            batch = ME.SparseTensor(feats, coords, device=device)
            try:
                labels_hat = model(batch)

                loss = criterion(labels_hat, labels) / accum_iter
                loss.backward()
                del labels_hat

                if (idx + 1) % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    log.log_train_batch(run, loss=loss.item() * accum_iter, epoch=e, batch_idx=idx)
            except Exception as ex:
                print(ex)
                pass

            if device == torch.device("cuda"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del (batch, labels)
            gc.collect()
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            groundtruth, predictions = None, None
            for idx, (coords, feats, labels) in enumerate(val_dataloader):
                # Removed mid-loop empty_cache for performance

                batch = ME.SparseTensor(feats, coords, device=device)
                preds = model(batch)

                # --- NEW: Compute validation loss ---
                labels_device = labels.long().to(device)
                val_loss = criterion(preds, labels_device)
                val_loss_accum += val_loss.item()
                val_batches += 1
                # ------------------------------------

                labels = labels.long().to(cpu)
                preds = preds.to(cpu)

                if groundtruth is None:
                    groundtruth = labels
                    predictions = preds
                else:
                    try:
                        groundtruth = torch.cat([groundtruth, labels], 0)
                        predictions = torch.cat([predictions, preds], 0)
                    except Exception as ex:
                        print(f"Concatenation error: {ex}")
                        pass

        # Now these variables are correctly populated
        avg_val_loss = val_loss_accum / max(val_batches, 1)
        scheduler.step()
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered. Ending training.")
            break

        log.log_epoch_and_model(run=run, model=model, epoch=e, preds=predictions,
                                target=groundtruth, model_name=cfg["model"])

    # evaluate on test set
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        groundtruth, predictions = None, None
        for idx, (coords, feats, labels) in enumerate(test_dataloader):
            torch.cuda.empty_cache()

            batch = ME.SparseTensor(feats, coords, device=device)
            preds = model(batch)

            labels = labels.long().to(cpu)
            preds = preds.to(cpu)

            if groundtruth is None:
                groundtruth = labels
                predictions = preds
            else:
                try:
                    groundtruth = torch.cat([groundtruth, labels], 0)
                    predictions = torch.cat([predictions, preds], 0)
                except Exception as ex:
                    print(ex)
                    pass
    log.log_epoch(run=run, preds=predictions, target=groundtruth, epoch=-1, prefix="test")
    
    run.finish()
