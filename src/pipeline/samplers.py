# /pipeline/samplers.py
import torch
from torch.utils.data import Sampler

class StochasticSampler(Sampler):
    def __init__(self, num_samples=None, random_seed=42, replacement=True):
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.replacement = replacement
        
    def set_data_source(self, data_source):
        self.data_source = data_source
        self.num_samples = self.num_samples or len(data_source)

    def __iter__(self):
        if not hasattr(self, 'data_source'):
            raise ValueError("Data source not set. Please call set_data_source() before using the sampler.")
        
        g = torch.Generator()
        g.manual_seed(self.random_seed)
        if self.replacement:
            indices = torch.randint(0, len(self.data_source), (self.num_samples,), generator=g)
        else:
            indices = torch.randperm(len(self.data_source), generator=g)[:self.num_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
