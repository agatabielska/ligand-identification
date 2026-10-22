
from models.sparse import MinkLoc3Dv2
from models.sparse.MinkLoc3Dv2.misc.utils import ModelParams
from models.sparse.MinkLoc3Dv2.models.model_factory import model_factory



def minkloc3dv2(overrides=None) -> MinkLoc3Dv2:
    minkloc_params = ModelParams('../cfg/models/minkloc3dv2.yaml')
    for name, value in (overrides or {}).items():
        if name in {"planes", "layers"}:
            value = tuple(int(item) for item in value)
        setattr(minkloc_params, name, value)
    model = model_factory(minkloc_params)
    return model
