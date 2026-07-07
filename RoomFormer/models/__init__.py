# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .roomformer import build as build_v0
from .roomformer_v1 import build as build_v1


def build_model_v0(args, train=True):
    return build_v0(args, train)


def build_model_v1(args, train=True):
    return build_v1(args, train)
