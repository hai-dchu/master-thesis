"""
Copied & modified from https://github.com/script-Yang/segdino_v2

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

import matplotlib.pyplot as plt

import util.misc as utils
import wandb


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels, use_group_norm=True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        return x + self.gamma * residual


class TPAResampleProject(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)


class TPASADDecoder(nn.Module):
    def __init__(
        self, in_dims, decoder_channels=128, num_classes=2, use_group_norm=True
    ):
        super().__init__()
        assert len(in_dims) == 4

        # TPA: project backbone tokens into decoder channels and align them to
        # the four spatial branches used by the decoder.
        self.token_projections = nn.ModuleList(
            [
                nn.Conv2d(channels, decoder_channels, 1, bias=False)
                for channels in in_dims
            ]
        )
        self.tpa_branch_1 = TPAResampleProject(decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(decoder_channels, scale_factor=1)

        # SAD: refine each branch independently, then fuse them from coarse to fine.
        self.sad_intra_1 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_intra_2 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_intra_3 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_intra_4 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )

        self.sad_inter_4 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_inter_3 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_inter_2 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )
        self.sad_inter_1 = ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm
        )

        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    def _tokens_to_feature_map(self, x, patch_h, patch_w):
        if isinstance(x, (list, tuple)):
            x = x[0]

        num_patches = patch_h * patch_w
        if x.ndim != 3:
            raise ValueError(
                f"Expected token tensor with 3 dims, got shape {tuple(x.shape)}"
            )
        if x.shape[1] < num_patches:
            raise ValueError(
                f"Token count {x.shape[1]} is smaller than expected patch grid {num_patches}"
            )
        if x.shape[1] != num_patches:
            x = x[:, -num_patches:, :]

        return x.transpose(1, 2).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)

    def forward(self, features, patch_h, patch_w):
        # TPA starts here: token sequences become a four-branch feature pyramid.
        branches = []
        for index, tokens in enumerate(features):
            feature_map = self._tokens_to_feature_map(tokens, patch_h, patch_w)
            feature_map = self.token_projections[index](feature_map)
            branches.append(feature_map)

        branch_1 = self.tpa_branch_1(branches[0])
        branch_2 = self.tpa_branch_2(branches[1])
        branch_3 = self.tpa_branch_3(branches[2])
        branch_4 = self.tpa_branch_4(branches[3])

        # SAD starts here: each branch is refined, then merged top-down.
        level_1 = self.sad_intra_1(branch_1)
        level_2 = self.sad_intra_2(branch_2)
        level_3 = self.sad_intra_3(branch_3)
        level_4 = self.sad_intra_4(branch_4)

        x4 = self.sad_inter_4(level_4)
        x3_up = F.interpolate(
            x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False
        )
        x3 = self.sad_inter_3(x3_up + level_3)

        x2_up = F.interpolate(
            x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False
        )
        x2 = self.sad_inter_2(x2_up + level_2)

        x1_up = F.interpolate(
            x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False
        )
        x1 = self.sad_inter_1(x1_up + level_1)

        return self.out_conv(x1)


class DPT(nn.Module):
    def __init__(
        self,
        encoder_size="base",
        nclass=2,
        decoder_channels=128,
        patch_size=16,
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
        self.patch_size = patch_size
        self.backbone = backbone
        self.nclass = nclass
        self.in_dims = [self.backbone.embed_dim] * 4
        self.decoder = TPASADDecoder(
            self.in_dims,
            decoder_channels=decoder_channels,
            num_classes=self.nclass,
            use_group_norm=not use_bn,
        )

        self._transform = make_transform()

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x, return_feats=False):
        x = self._transform(x)
        patch_h, patch_w = (
            x.shape[-2] // self.patch_size,
            x.shape[-1] // self.patch_size,
        )
        feats = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size]
        )

        out = self.decoder(feats, patch_h, patch_w)
        out = F.interpolate(
            out, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        if return_feats:
            return out, feats[-1]
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

        img = torch.from_numpy(img)  # H, W, 3
        sem = torch.from_numpy(sem).long()  # H, W

        img = img.moveaxis(-1, 0)  # channel-first
        sem = sem[None, :, :]  # expand to channel-first

        # augmentation so that input and target still match
        _hor = np.random.randn() > 0.5
        _ver = np.random.randn() > 0.5
        _rot = np.random.choice([0, 90, 180, 270]).item()

        img = self._augmentation(img, _hor, _ver, _rot)
        sem = self._augmentation(sem, _hor, _ver, _rot).squeeze()

        # transformed_img = self._transform(img)
        onehot_sem = F.one_hot(sem, num_classes=41).moveaxis(-1, 0).float()

        meta = {"img_path": str(img_path), "sem_path": str(sem_path)}
        return img, onehot_sem, meta


def config():
    ap = argparse.ArgumentParser(description="Running segmentation on cubemap faces")
    ap.add_argument("--job_name", default="train_SegDINO")

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
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--dinov3_repo", help="root directory of dinov3")
    ap.add_argument("--dinov3_checkpoint", help="checkpoint directory of dinov3")
    ap.add_argument(
        "--dinov3_model", default="dinov3_vits16", help="dinov3 exact model name"
    )

    ap.add_argument(
        "--checkpoint", help="path to checkpoint (for testing and resume training)"
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
    ap.add_argument("--wandb", default=False, action="store_true")
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

        optimizer.zero_grad()

        logits = model(inputs)

        # print(inputs.shape)
        # print(targets.shape)
        # print(logits.shape)

        #

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
    epoch: int | None,
    max_norm: float = 0,
):
    total_loss = 0
    pbar = tqdm(data_loader, desc=f"[Eval epoch {epoch}]")

    for inputs, targets, _ in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(1, len(data_loader))
    if epoch:
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}")

    return avg_loss


# def batch_collator(batch):


def train(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = args.output_dir

    if args.wandb:
        utils.setup_wandb()
        wandb.init(project="SegDINO")
        wandb.run.name = args.run_name

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

        if args.wandb:
            wandb.log({"train_loss": train_loss})
            wandb.log({"val_loss": val_loss})

        if not args.dry_run and ((epoch + 1) % 10 == 0):
            torch.save(model.state_dict(), out_dir / f"checkpoint_{epoch}.pth")

    if not args.dry_run:
        train_losses = np.array(train_losses)
        val_losses = np.array(val_losses)

        torch.save(model.state_dict(), out_dir / "checkpoint.pth")

        np.save(out_dir / "train_losses", train_losses)
        np.save(out_dir / "val_losses", val_losses)


def test(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args).to(device)
    checkpoint = Path(os.path.abspath(args.checkpoint))

    assert checkpoint.exists(), f"{checkpoint} not found"

    model.load_state_dict(torch.load(checkpoint))
    out_dir = os.path.dirname(checkpoint)

    dataset_test = build_dataset(args, mode="test")
    sampler_test = SequentialSampler(dataset_test)
    test_loader = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
    )

    model.eval()

    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(test_loader, desc="Running tests")
    avg_loss = 0
    all_targets = []
    all_preds = []
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)
        preds = torch.argmax(
            logits, dim=1
        )  # (batch_size, num_classes=41, 256, 256) [0, 40]

        avg_loss += loss.item()

        all_targets.append(torch.argmax(targets, dim=1).cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        if (step + 1) % 10 == 0:
            sample = np.random.choice([_ for _ in range(len(inputs))])
            inputs_np = np.moveaxis(inputs[sample].cpu().numpy(), 0, -1)
            targets_np = np.moveaxis(all_targets[-1][sample], 0, -1)
            preds_np = np.moveaxis(all_preds[-1][sample], 0, -1)

            plt.figure(figsize=(15, 6))
            plt.suptitle(f"samples {step + 1}")
            plt.axis("off")

            plt.subplot(1, 3, 1)
            plt.title("input image")
            plt.imshow(inputs_np)

            plt.subplot(1, 3, 2)
            plt.title("target mask")
            plt.imshow(targets_np)

            plt.subplot(1, 3, 3)
            plt.title("predicted mask")
            plt.imshow(preds_np)

            plt.tight_layout()
            if not args.dry_run:
                plt.savefig(out_dir / f"samples_{step + 1}.png", dpi=300)

            plt.close()
            # plt.show()

        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)

        #


if __name__ == "__main__":
    args = config()
    now = datetime.datetime.now()  # noqa: DTZ005
    args.run_name = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}_segmentation"
    out_dir = Path(os.path.abspath(args.output_dir)) / args.run_name
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"out_dir: {out_dir}")

    args.output_dir = out_dir
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
