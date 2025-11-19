import torch.nn as nn

from models.create import minkloc3dv2

MODELS = {
    "minkloc3d": minkloc3dv2,
}


def create(model_type: str) -> nn.Module:
    return MODELS[model_type.lower()]()
