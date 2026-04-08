import os
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm  # 导入 tqdm
from dataset import FullDataset
from SDDNet import SDDNet
import torch.nn as nn

parser = argparse.ArgumentParser("SDDNet")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=50,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--val_split", type=float, default=0.2,
                    help="proportion of training data to use as validation")
# parser.add_argument("--train_depth_path", type=str, required=True,
#                     help="path to the depth images used to train the model")

args = parser.parse_args()

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def compute_mae_prnet(model, dataloader, device):
    model.eval()
    mae_sum = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            pred = model(x)[0]  # 假设第一个输出是主要预测
            pred = torch.sigmoid(pred)
            mae_sum += torch.sum(torch.abs(pred - target)).item() / target.numel()
    return mae_sum / len(dataloader)  # 对每个 batch 的 MAE 取均值

def main(args):
    print("========== Initializing Dataset ==========")
    # full_dataset = FullDataset(args.train_image_path, args.train_mask_path, 352, mode='train')
    full_dataset = FullDataset(
        image_root=args.train_image_path,
        # depth_root=args.train_depth_path,
        gt_root=args.train_mask_path,
        size=352,
        mode='train'
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    print(f"Total samples: {len(full_dataset)}, Training: {train_size}, Validation: {val_size}")

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("========== Initializing Model ==========")
    model = SDDNet(args.hiera_path)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=5.0e-8)
    os.makedirs(args.save_path, exist_ok=True)

    best_mae = float('inf')  # 初始化最优 MAE
    best_epoch = -1  # 记录最佳 MAE 所在的 epoch

    print("========== Training Start ==========")
    for epoch in range(args.epoch):
        print(f"\n[Epoch {epoch + 1}/{args.epoch}] Training...")
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", leave=False)

        for i, batch in enumerate(progress_bar):
            x = batch['image'].to(device)
            target = batch['label'].to(device)
            optimizer.zero_grad()
            pred0, pred1, pred2 = model(x)
            loss0 = structure_loss(pred0, target)
            loss1 = structure_loss(pred1, target)
            loss2 = structure_loss(pred2, target)
            loss = loss0 + loss1 + loss2
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Average Training Loss: {avg_epoch_loss:.4f}")

        # 进行验证
        print(f"[Epoch {epoch + 1}] Running Validation...")
        val_mae = compute_mae_prnet(model, val_loader, device)
        print(f"[Epoch {epoch + 1}] Validation MAE: {val_mae:.4f}")

        # 检查是否是最优 MAE，如果是则保存模型
        if val_mae < best_mae:
            best_mae = val_mae
            best_epoch = epoch + 1
            best_model_path = os.path.join(args.save_path, 'SDDNet-best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"[Saving Best Model] Epoch {epoch + 1}, MAE: {best_mae:.4f}")

        # 每 5 轮保存一次模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_path, f'SDDNet-epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Checkpoint Saved] Epoch {epoch + 1} model saved at {checkpoint_path}")

        scheduler.step()

    print("\n========== Training Completed ==========")
    print(f"Best Validation MAE: {best_mae:.4f} at Epoch {best_epoch}")
    print(f"Best model saved at: {os.path.join(args.save_path, 'SDDNet-best.pth')}")

if __name__ == "__main__":
    # 可选：启用种子设置以保证可重复性
    # seed_torch(1024)
    main(args)
