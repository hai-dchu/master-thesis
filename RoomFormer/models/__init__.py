# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# ------------------------------------------------------------------------------------

from .roomformer_v1 import build


def build_model(args, train=True):
    return build(args, train)

