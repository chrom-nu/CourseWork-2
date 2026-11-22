import os
import sys
import json
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from dataset import get_dataloader, CLASS_TO_IDX

# Set mpl / cache dir if needed
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(".tmp", "matplotlib")))

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(arch: str = "resnet34", num_classes: int = 2):
    if arch == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)
    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def calculate_class_weights(dataset, device):
    """
    Computes balanced class weights: total / (num_classes * class_count)
    """
    targets = [sample[1] for sample in dataset.samples]
    total_samples = len(targets)
    class_counts = np.bincount(targets, minlength=2)
    weights = total_samples / (2.0 * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class counts: healthy={class_counts[0]}, sick={class_counts[1]}")
    print(f"Calculated class weights: healthy={weights[0]:.4f}, sick={weights[1]:.4f}")
    return weights_tensor

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    sick_recall = recall_score(all_targets, all_preds, pos_label=CLASS_TO_IDX["sick"], zero_division=0)
    sick_precision = precision_score(all_targets, all_preds, pos_label=CLASS_TO_IDX["sick"], zero_division=0)
    healthy_recall = recall_score(all_targets, all_preds, pos_label=CLASS_TO_IDX["healthy"], zero_division=0)

    metrics = {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "sick_recall": sick_recall,
        "sick_precision": sick_precision,
        "healthy_recall": healthy_recall
    }
    return metrics, all_preds, all_targets

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_name = args.run_name
    if not run_name:
        weight_tag = "weighted" if args.use_class_weights else "unweighted"
        run_name = f"{args.arch}_{weight_tag}_seed{args.seed}"

    run_dir = os.path.join("ml", "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    print(f"\n--- Starting Training Run: {run_name} ---")
    print(f"Config: arch={args.arch}, epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}, img_size={args.img_size}")

    # Data loaders
    data_dir = os.path.join("ml", "data")
    train_loader = get_dataloader(data_dir, split="train", batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)
    val_loader = get_dataloader(data_dir, split="val", batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size)

    # Model
    model = get_model(arch=args.arch, num_classes=2).to(device)

    # Loss & Weights
    class_weights = None
    if args.use_class_weights:
        class_weights = calculate_class_weights(train_loader.dataset, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    # Save run config
    config = {
        "arch": args.arch,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "img_size": args.img_size,
        "use_class_weights": args.use_class_weights,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "device": str(device)
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for images, targets in train_pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader.dataset)

        # Validation
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_macro_f1 = val_metrics["macro_f1"]
        val_loss = val_metrics["loss"]
        val_sick_recall = val_metrics["sick_recall"]
        val_sick_precision = val_metrics["sick_precision"]

        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_macro_f1:.4f} | "
            f"Val Sick Recall: {val_sick_recall:.4f} | "
            f"Val Sick Prec: {val_sick_precision:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Log TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/val_macro", val_macro_f1, epoch)
        writer.add_scalar("Recall/val_sick", val_sick_recall, epoch)
        writer.add_scalar("Precision/val_sick", val_sick_precision, epoch)
        writer.add_scalar("Accuracy/val", val_metrics["acc"], epoch)
        writer.add_scalar("LR", current_lr, epoch)

        # Save history row
        row = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "val_sick_recall": val_sick_recall,
            "val_sick_precision": val_sick_precision,
            "val_healthy_recall": val_metrics["healthy_recall"],
            "val_acc": val_metrics["acc"],
            "lr": current_lr
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

        # Save checkpoints
        is_best = val_macro_f1 > best_val_macro_f1
        checkpoint = {
            "epoch": epoch,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_macro_f1": val_macro_f1,
            "val_metrics": val_metrics,
            "config": config
        }

        # Always save last checkpoint
        torch.save(checkpoint, os.path.join(run_dir, "last_model.pt"))

        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(checkpoint, os.path.join(run_dir, "best_model.pt"))
            print(f"⭐ New best checkpoint saved at epoch {epoch} (Val Macro F1: {val_macro_f1:.4f})")

    writer.close()
    print(f"\nTraining completed for {run_name}!")
    print(f"Best Epoch: {best_epoch} with Val Macro F1: {best_val_macro_f1:.4f}")
    print(f"Artifacts saved in: {run_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline Plant Disease Classifier Training")
    parser.add_argument("--arch", type=str, default="resnet34", choices=["resnet34", "resnet50"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
