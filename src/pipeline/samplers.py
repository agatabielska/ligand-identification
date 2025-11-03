# /pipeline/samplers.py
import torch
from torch.utils.data import Sampler

class StochasticSampler(Sampler):
    def __init__(self, data_source, num_samples=None, random_seed=42, replacement=True):
        self.data_source = data_source
        self.num_samples = num_samples or len(data_source)
        self.random_seed = random_seed
        self.replacement = replacement

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.random_seed + torch.randint(0, 100000, (1,)).item())
        if self.replacement:
            indices = torch.randint(0, len(self.data_source), (self.num_samples,), generator=g)
        else:
            indices = torch.randperm(len(self.data_source), generator=g)[:self.num_samples]
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples
