from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import Tensor, nn
from torchvision.transforms import v2
from tqdm import tqdm


def make_transform(resize_size: int = 256):
    # to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([resize, to_float, normalize])


def load_DINO(
    repo: str | Path,
    checkpoint: str | Path,
    *,
    model_name: str = "dinov3_vits16",
    device: str = "cuda",
) -> nn.Module:
    model = torch.hub.load(
        str(repo),
        model_name,
        source="local",
        weights=str(checkpoint),
    )
    model.eval().requires_grad_(False).to(device)
    return model


@torch.inference_mode()
def extract_patch_grid(model: nn.Module, batch: Tensor):
    """
    Args:
    batch: normalized RGB [B,3,H,W], H and W divisible by 16.
    Returns:
    normalized patch features [B,C,H/16,W/16].
    """
    if batch.ndim != 4 or batch.shape[1] != 3:
        raise ValueError(f"Expected [B,3,H,W], got {tuple(batch.shape)}")

    patch = int(model.patch_size)
    if batch.shape[-2] % patch or batch.shape[-1] % patch:
        raise ValueError("DINO input dimensions must be divisible by patch size")

    # n=1 requests the last block. reshape=True removes class/storage tokens
    # and returns a spatial [B,C,H_patch,W_patch] tensor.
    with torch.autocast(
        device_type=batch.device.type, dtype=torch.bfloat16, enabled=batch.is_cuda
    ):
        (features,) = model.get_intermediate_layers(
            batch,
            n=1,
            reshape=True,
            norm=True,
            return_class_token=False,
            return_extra_tokens=False,
        )

    features = F.normalize(features.float(), dim=1)
    if not torch.isfinite(features).all():
        raise RuntimeError("DINO produced non-finite patch features")
    return features


FACES = sorted(["U", "F", "R", "L", "B", "D"])


def _points_to_bev(
    scene_pc: torch.Tensor,  # [N, 3] (x, y, z)
    point_features: torch.Tensor,  # [N, C] DINO features per point
    point_mask: torch.Tensor,  # [N] boolean mask (True if valid DINO signal)
    point_agreement: torch.Tensor,  # [N] multi-view agreement score (in [0, 1])
    width: int = 256,
    height: int = 256,
) -> dict[str, Tensor]:
    """
    Implements Point-to-BEV pooling exactly aligned with RoomFormer's generate_density.
    """
    device = scene_pc.device
    _, embed_dim = point_features.shape

    xy = scene_pc[:, :2]  # [N, 2] -> (x, y)

    min_coords = xy.min(dim=0).values
    max_coords = xy.max(dim=0).values
    range_coords = max_coords - min_coords
    min_coords = min_coords - 0.1 * range_coords
    max_coords = max_coords + 0.1 * range_coords

    image_res = torch.tensor([width, height], device=device, dtype=torch.float32)

    denom = (max_coords - min_coords).clamp_min(1e-6)

    coords = torch.round((xy - min_coords) / denom * image_res)

    coords[:, 0] = coords[:, 0].clamp(0, width - 1)  # x -> column index (gx)
    coords[:, 1] = coords[:, 1].clamp(0, height - 1)  # y -> row index (gy)

    gx = coords[:, 0].to(torch.long)
    gy = coords[:, 1].to(torch.long)

    # Keep points with positive no. of appearance
    valid_keep = point_mask & torch.isfinite(point_features).all(dim=1)

    # Keep valid coordinates, features, agreed 3D points
    gx_valid = gx[valid_keep]
    gy_valid = gy[valid_keep]
    feats_valid = F.normalize(point_features[valid_keep].float(), dim=-1)
    agreed_valid = point_agreement[valid_keep].float().clamp(0.0, 1.0)

    # Map (gy, gx) to flat cell index to match density[y, x] indexing
    flat_indices = gy_valid * width + gx_valid
    num_cells = height * width

    # Flatten the DINO-BEV output, DINO mask, DINO log_count to a [256*256, embed_dim]
    sum_f = torch.zeros((num_cells, embed_dim), dtype=torch.float32, device=device)
    sum_a = torch.zeros((num_cells,), dtype=torch.float32, device=device)
    count = torch.zeros((num_cells,), dtype=torch.float32, device=device)

    # Accumulate features, agreement, and point counts into BEV cells
    sum_f.index_add_(0, flat_indices, feats_valid)
    sum_a.index_add_(0, flat_indices, agreed_valid)
    count.index_add_(0, flat_indices, torch.ones_like(agreed_valid))

    occupied = count > 0

    # Mean-pooling and L2 normalization
    # TODO: ablation, test other methods
    mean_f = torch.zeros_like(sum_f)
    mean_f[occupied] = sum_f[occupied] / count[occupied, None]
    mean_f[occupied] = F.normalize(mean_f[occupied], dim=-1)

    mean_a = torch.zeros_like(sum_a)
    mean_a[occupied] = sum_a[occupied] / count[occupied]

    # Reshape to spatial rasters [C, H, W]
    dino_bev_feature = mean_f.T.reshape(embed_dim, height, width)
    dino_mask = occupied.reshape(1, height, width).float()
    dino_log_count = torch.log1p(count).reshape(1, height, width)
    dino_agreement = mean_a.reshape(1, height, width)

    return {
        "dino_bev": dino_bev_feature,  # [C_R, 256, 256]
        "dino_mask": dino_mask,  # [1, 256, 256]
        "dino_count": dino_log_count,  # [1, 256, 256]
        "dino_agreement": dino_agreement,  # [1, 256, 256]
    }


class DINO_BEV(nn.Module):
    def __init__(self, DINOv3_base: nn.Module, pca: nn.Module):
        super().__init__()
        self.DINO = DINOv3_base
        self.pca = pca
        self._transform = make_transform()
        self.DINO.requires_grad_(False)

    def forward(
        self,
        rooms: list[Tensor],
        masks: list[Tensor],
        point_cloud: list[Tensor],
        project_points: list[list[list[Tensor]]],
    ):
        """
        params
        rooms: {room, (batchx6x3x256x256)}
        masks: {room, (batchx6xnum_pointsx1)}
        point_cloud: (batchxnum_pointsx3) point cloud for each scene in batch
        project_points: [num_visible_points] visible points projected on face
            for each face for each room for each scene

        note:
        - len(mask[scene] == True) == len(project_points[scene])
        """
        # Get DINO patches for each scene in batch for each room in scene
        batch_pc_out = []
        batch_valid_mask = []
        batch_scene_bev = []
        batch_scene_mask = []
        batch_scene_cnt = []
        batch_scene_agree = []
        for scene_idx in range(len(rooms)):
            scene = rooms[scene_idx]
            B, _, _, _ = scene.shape
            num_room = B // len(FACES)  # Should be divisible
            # (num_room*6)x(3x256x256) -> (num_room*6)x(dino_embed_dimx16x16)

            model_out = extract_patch_grid(self.DINO, self._transform(scene))

            # PCA reduction on DINOv3 output
            # TODO: other dimension reducer methods
            model_out = self.pca(model_out)

            # (num_room*6)x(64x16x16)
            _, embed_dim, patch_size, _ = model_out.shape
            model_out = model_out.reshape(
                num_room, len(FACES), embed_dim, patch_size, patch_size
            )

            scene_pc = point_cloud[scene_idx]
            num_3D_points, _ = scene_pc.shape
            device = scene_pc.device
            feature_aggregation = torch.zeros([num_3D_points, embed_dim], device=device)

            # mask_norm stores the number of scenes in which a point appear
            mask_norm = torch.ones([num_3D_points], device=device)

            room_prj_points = project_points[scene_idx]
            for room_idx, room in enumerate(room_prj_points):
                for face_idx in range(len(FACES)):
                    u_idx = room[face_idx][:, 0].to(torch.int64)
                    v_idx = room[face_idx][:, 1].to(torch.int64)

                    u_idx = torch.clamp(u_idx, 0, 255)
                    v_idx = torch.clamp(v_idx, 0, 255)

                    feature_map = (
                        model_out[room_idx, face_idx, :, :]
                        .repeat(1, 1, 16, 16)
                        .squeeze()
                    )
                    feature_map = F.normalize(feature_map)
                    points_3D_features = feature_map[:, v_idx, u_idx].T  # permute(1, 0)
                    mask = masks[scene_idx][room_idx][face_idx]

                    # Aggregation method: average over num_mask for each point
                    # TODO: ablation
                    # - other methods for aggregation

                    feature_aggregation[mask] += points_3D_features
                    mask_norm += mask

            # Multi-view fusion
            # TODO: ablation
            # - add weighted mean (currently just mean)
            feature_aggregation /= mask_norm[:, None]
            feature_aggregation = F.normalize(feature_aggregation)

            batch_valid_mask.append(mask_norm)
            batch_pc_out.append(feature_aggregation)

            # BEV cell feature pooling
            # - weighted mean of the features
            # - L2-normalize
            pb_out = _points_to_bev(
                scene_pc, feature_aggregation, mask_norm > 0, mask_norm
            )
            scene_bev = pb_out["dino_bev"]
            scene_mask = pb_out["dino_mask"]
            scene_cnt = pb_out["dino_count"]
            scene_agree = pb_out["dino_agreement"]
            batch_scene_bev.append(scene_bev)
            batch_scene_mask.append(scene_mask)
            batch_scene_cnt.append(scene_cnt)
            batch_scene_agree.append(scene_agree)

        return (
            torch.stack(batch_scene_bev),
            torch.stack(batch_scene_mask),
            torch.stack(batch_scene_cnt),
            torch.stack(batch_scene_agree),
        )


class DINORoomFormer(nn.Module):
    def __init__(self, roomformer: nn.Module, dino_bev: nn.Module):
        super().__init__()
        self.roomformer = roomformer
        self.dino_bev = dino_bev

    def forward(self, x):
        (
            _,
            samples,
            _,
            _,
            batched_room_faces,
            _,
            batched_point_clouds,
            batched_prj_points,
            batched_masks,
        ) = x

        scene_bev, _, _, _ = self.dino_bev(
            batched_room_faces, batched_masks, batched_point_clouds, batched_prj_points
        )

        enhanced_samples = []
        for img, feat in zip(samples, scene_bev[:]):
            enhanced_samples.append(torch.cat([img, feat], dim=0))

        return self.roomformer(enhanced_samples)


class PCAWrapper(nn.Module):
    def __init__(
        self,
        out_channels=64,
        max_samples=10000,
        verbose=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.max_samples = max_samples
        self.is_fitted = False

    @torch.no_grad()
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
    ):
        total_samples = 0
        collected_features = []
        for _, batch in tqdm(enumerate(train_loader)):
            _, C, _, _ = batch.shape
            x_flat = batch.permute(0, 2, 3, 1).reshape(-1, C)  # .to(args.device)
            collected_features.append(x_flat)
            total_samples += x_flat.shape[0]
            if total_samples >= self.max_samples:
                break
        collected_features = (
            torch.cat(collected_features, dim=0)[: self.max_samples].cpu().numpy()
        )
        pca = PCA(n_components=self.out_channels).fit(collected_features)
        self.is_fitted = True

        self.register_buffer("components", torch.from_numpy(pca.components_).float())
        self.register_buffer("mean", torch.from_numpy(pca.mean_).float())

    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError(
                "PCAReducer must be fitted using `.fit(train_loader)` before calling forward!"
            )

        B, C, H, W = x.shape
        # device = x.device
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # .cpu().numpy()
        x_centered = x_flat - self.mean
        x_out = torch.matmul(x_centered, self.components.T)
        x_out = x_out.view(B, H, W, self.out_channels).permute(0, 3, 1, 2).contiguous()
        return x_out


def build(args, train=True):
    # dino_embed_dims = {
    #     "dinov3_vits16": 384,
    #     "dinov3_vitb16": 768,
    #     "dinov3_vitl16": 1024,
    # }

    DINOv3 = load_DINO(
        repo=args.dinov3_repo,
        checkpoint=args.dinov3_checkpoint,
        model_name=args.dinov3_model,
        device=args.device,
    )
    pca = PCAWrapper(out_channels=args.pca_outdim)
    model = DINO_BEV(DINOv3, pca)

    return model, pca
