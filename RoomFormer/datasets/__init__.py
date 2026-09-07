from .poly_data import build as build_poly
from .dino_cube_data import build as build_cube_poly
from .litept_normal_data import build as build_litept_normal


def build_poly_dataset(image_set, args):
    if args.semantic_classes > 0:
        assert args.dataset_name == "stru3d", (
            "Semantically-rich floorplans only support Structured3D"
        )
    if args.dataset_name == "stru3d" or args.dataset_name == "scenecad":
        return build_poly(image_set, args)
    raise ValueError(f"dataset {args.dataset_name} not supported")


def build_mixed_dataset(image_set, args):
    return build_cube_poly(image_set, args)


def build_normal_dataset(image_set, args, point_transforms):
    return build_litept_normal(image_set, args, point_transforms)
