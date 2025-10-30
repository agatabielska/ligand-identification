from pipeline.data_loader import NPZDataLoader
import torch.nn as nn
from typing import Optional


class Pipeline():
    def __init__(self, data_loader: NPZDataLoader, model) -> None:
        self.data_loader = data_loader
        train_loader, val_loader, test_loader = data_loader.get_all_loaders()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.model = model
        
    def fit(self,
        epochs: int = 10,
        optimizer=None,
        criterion=None,
        scheduler=None,
        verbose: bool = True,
        early_stopping_patience: Optional[int] = None,
        checkpoint_path: Optional[str] = None
    ) -> None:
        self.model.fit(
            self.train_loader,
            val_loader=self.val_loader,
            epochs=epochs,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            verbose=verbose,
            early_stopping_patience=early_stopping_patience,
            checkpoint_path=checkpoint_path
        )
        
    def predict(self, data_loader, return_probabilities: bool = False):
        return self.model.predict(data_loader, return_probabilities=return_probabilities)
    
    def evaluate(self, criterion=None):
        return self.model.evaluate(self.test_loader, criterion=criterion)