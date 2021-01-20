import asteroid
from .sepformer_tasnet import SepFormerTasNet, SepFormer2TasNet
asteroid.models.register_model(SepFormerTasNet)
asteroid.models.register_model(SepFormer2TasNet)

__all__ = [
    "SepFormerTasNet",
    "SepFormer2TasNet",
]
