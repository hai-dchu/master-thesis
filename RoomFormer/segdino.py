"""
Copied & modified from https://github.com/script-Yang/segdino

Many changes are to make the code more compact and readable :)))
Hai Chu
"""

import argparse
import datetime
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from models.dino_bev import load_DINO, make_transform
from PIL import Image
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)
from tqdm import tqdm


class DPTHead(nn.Module):
    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024],
    ):
        super().__init__()
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        scratch = [
            nn.Conv2d(
                out_channel,
                features,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                groups=1,
            )
            for out_channel in out_channels
        ]
        self.scratch = nn.ModuleList(scratch)

        self.scratch.stem_transpose = None
        self.output_conv = nn.Conv2d(
            features * 4, nclass, kernel_size=1, stride=1, padding=0
        )
        self.proj = nn.ConvTranspose2d(
            features, features, 4, stride=4, padding=0, bias=False
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            out.append(x)

        layers_rn = []
        for i, layer in enumerate(out):
            layers_rn.append(self.scratch[i](layer))
        layers_rn[0] = self.proj(layers_rn[0])
        target_hw = layers_rn[0].shape[-2:]
        for i in range(1, len(layers_rn)):
            layers_rn[i] = F.interpolate(
                layers_rn[i], size=target_hw, mode="bilinear", align_corners=True
            )

        fused = torch.cat(layers_rn, dim=1)
        out = self.output_conv(fused)
        return out


class DPT(nn.Module):
    def __init__(
        self,
        encoder_size="base",
        nclass=2,
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        backbone=None,
    ):
        super().__init__()

        self.intermediate_layer_idx = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
            "large": [4, 11, 17, 23],
        }

        self.encoder_size = encoder_size
        self.backbone = backbone
        self.head = DPTHead(
            nclass, self.backbone.embed_dim, features, use_bn, out_channels=out_channels
        )

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )
        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(
            out, (patch_h * 16, patch_w * 16), mode="bilinear", align_corners=True
        )
        return out


FACES = ["U", "F", "R", "L", "B", "D"]


# Technically this segmentation only cares about the image, not the scene
# so one can actually ignore the scene entirely
class CubeMapDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode="train",
    ):
        assert os.path.exists(os.path.abspath(data_dir)), "data folder does not exist"
        assert mode in ["train", "test", "val"], (
            "mode should be one of (train, test, val), default=train"
        )
        super().__init__()

        self.data_dir = Path(os.path.abspath(data_dir))
        self.data_root = self.data_dir / mode

        self.aug_rotate = mode == "train"
        self.aug_flip = mode == "train"

        self._transform = make_transform()

        scene_ids = os.listdir(self.data_root)
        self.img_path = []
        self.sem_path = []

        for scene in scene_ids:
            if not (self.data_root / scene).is_dir():
                continue
            scene_dir = self.data_root / scene
            room_ids = os.listdir(scene_dir)
            for room in room_ids:
                if not (scene_dir / room).is_dir():
                    continue
                room_dir = scene_dir / room
                for face in FACES:
                    img_path = room_dir / f"img_{face}.png"
                    sem_path = room_dir / f"sem_{face}.png"
                    if not (img_path.exists() and sem_path.exists()):
                        continue
                    self.img_path.append(img_path)
                    self.sem_path.append(sem_path)

        print(f"Finished loading {mode} dataset")
        print(f"\tFrom: {self.data_root}")
        print(f"\tNum samples: {len(self.img_path)}")

    def __len__(self):
        return len(self.img_path)

    def _augmentation(self, img, hor=True, ver=True, rotate: int | float = 90):
        out = img
        if hor:
            out = TF.hflip(out)
        if ver:
            out = TF.vflip(out)
        out = TF.rotate(out, angle=rotate, interpolation=Image.BILINEAR)
        return out

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        sem_path = self.sem_path[idx]

        img = np.array(Image.open(img_path))
        sem = np.array(Image.open(sem_path))

        img = torch.from_numpy(img).float()  # H, W, 3
        sem = torch.from_numpy(sem).long()  # H, W

        img = img.moveaxis(-1, 0)  # channel-first
        sem = sem[None, :, :]  # expand to channel-first

        # augmentation so that input and target still match
        _hor = np.random.randn() > 0.5
        _ver = np.random.randn() > 0.5
        _rot = np.random.choice([0, 90, 180, 270]).item()

        img = self._augmentation(img, _hor, _ver, _rot)
        sem = self._augmentation(sem, _hor, _ver, _rot).squeeze()

        transformed_img = self._transform(img)
        onehot_sem = F.one_hot(sem, num_classes=41).moveaxis(-1, 0).float()

        meta = {"img_path": str(img_path), "sem_path": str(sem_path)}
        return transformed_img, onehot_sem, meta


def config():
    ap = argparse.ArgumentParser(description="Running segmentation on cubemap faces")

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument(
        "-n",
        "--num_classes",
        default=41,
        type=int,
        help="number of classes supported by the model",
    )

    ap.add_argument(
        "--dataset_root",
        help="Data directory",
    )
    ap.add_argument(
        "-o",
        "--output_dir",
        default="output_segdino",
        help="Output directory. Default is output_segdino",
    )
    ap.add_argument(
        "-m", "--mode", default="train", help="mode to build and run the model"
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)

    ap.add_argument("--dinov3_repo", help="root directory of dinov3")
    ap.add_argument("--dinov3_checkpoint", help="checkpoint directory of dinov3")
    ap.add_argument(
        "--dinov3_model", default="dinov3_vits16", help="dinov3 exact model name"
    )

    ap.add_argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Print output if included",
    )  # Actually not quiet
    ap.add_argument(
        "-d", "--dry_run", default=False, action="store_true", help="For testing"
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return args


def build_model(args) -> nn.Module:
    DINO = load_DINO(
        repo=args.dinov3_repo,
        checkpoint=args.dinov3_checkpoint,
        model_name=args.dinov3_model,
        device=args.device,
    )

    model = DPT(nclass=args.num_classes, backbone=DINO)

    # Frozen backbone, could change to finetune backbone (not recommended)
    model.lock_backbone()

    return model


def build_dataset(args, mode="train") -> Dataset:
    dataset = CubeMapDataset(data_dir=args.dataset_root, mode=mode)
    return dataset


def train_one_epoch(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss | nn.BCEWithLogitsLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    epoch: int,
    max_norm: float = 0,
):
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"[Train epoch {epoch}]")

    model.train()
    for step, (inputs, targets, _meta) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)

        # print(inputs.shape)
        # print(targets.shape)
        # print(logits.shape)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(data_loader))
    print(f"[Epoch {epoch}] loss={avg_loss:.4f}")
    return avg_loss


def evaluate(
    model: nn.Module,
    criterion: nn.CrossEntropyLoss | nn.BCEWithLogitsLoss,
    data_loader: Iterable,
    device: str | torch.device,
    epoch: int,
    max_norm: float = 0,
):
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"[Train epoch {epoch}]")

    for inputs, targets, _ in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(data_loader))
    print(f"[Epoch {epoch}] loss={avg_loss:.4f}")
    return avg_loss


# def batch_collator(batch):


def main(args):
    device = torch.device(args.device)

    now = datetime.datetime.now()  # noqa: DTZ005
    out_dir = (
        Path(os.path.abspath(args.output_dir))
        / f"{now.strftime('%Y-%m-%d-%H-%M-%S')}_segmentation"
    )
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"out_dir: {out_dir}")

    model = build_model(args).to(device)
    dataset_train = build_dataset(args, mode="train")
    dataset_val = build_dataset(args, mode="val")
    # dataset_test = build_dataset(args, mode="test")

    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    # sampler_test = SequentialSampler(dataset_test)

    batch_sampler_train = BatchSampler(
        sampler_train, batch_size=args.batch_size, drop_last=True
    )

    train_loader = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )
    # test_loader = DataLoader(
    #     dataset_test,
    #     args.batch_size,
    #     sampler=sampler_test,
    #     drop_last=False,
    #     num_workers=2,
    #     pin_memory=True,
    # )

    for n, p in model.named_parameters():
        param_state = "[Active]" if p.requires_grad else ""
        print(f"{param_state} {n}")

    # finetuning DINO
    param_dicts = [{"params": [p for _, p in model.named_parameters()]}]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model, criterion, train_loader, optimizer, device=device, epoch=epoch
        )
        train_losses.append(train_loss)

        val_loss = evaluate(model, criterion, val_loader, device=device, epoch=epoch)
        val_losses.append(val_loss)

        if not args.dry_run and ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), out_dir / f"checkpoint_{epoch}.pth")

    if not args.dry_run:
        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

        torch.save(model.state_dict(), out_dir / "checkpoint.pth")

        np.save(out_dir / "train_losses", train_losses)
        np.save(out_dir / "val_losses", val_losses)


if __name__ == "__main__":
    main(config())
