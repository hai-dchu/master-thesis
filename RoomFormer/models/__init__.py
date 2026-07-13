# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .roomformer import build as build_v0
from .roomformer_v1 import build as build_v1
from .roomformer_v2 import build as build_v2


def build_model_v0(args, train=True):
    return build_v0(args, train)


def build_model_v1(args, train=True):
    return build_v1(args, train)


def build_model_v2(args, train=True):
    return build_v2(args, train)