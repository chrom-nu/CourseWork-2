import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from dataset import get_dataloader, CLASS_TO_IDX, IDX_TO_CLASS
from train import get_model

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join(".tmp", "matplotlib")))
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def evaluate_checkpoint(checkpoint_path: str, split: str = "val", batch_size: int = 32, data_dir: str = os.path.join("ml", "data"), output_dir: str = None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", {})
    arch = checkpoint.get("arch", config.get("arch", "resnet34"))
    img_size = config.get("img_size", 224)

    # Initialize model
    model = get_model(arch=arch, num_classes=2).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Dataloader
    loader = get_dataloader(root_dir=data_dir, split=split, batch_size=batch_size, num_workers=2, img_size=img_size, shuffle=False)

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"Running evaluation on '{split}' split ({len(loader.dataset)} samples)...")
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_targets,
        all_preds,
        target_names=["healthy (0)", "sick (1)"],
        digits=4,
        output_dict=True
    )
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 55)
    print(f"EVALUATION RESULTS: Split = {split.upper()} | Model = {arch}")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("-" * 55)
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 55)
    for cls_name in ["healthy (0)", "sick (1)"]:
        r = report[cls_name]
        print(f"{cls_name:<15} | {r['precision']:<10.4f} | {r['recall']:<10.4f} | {r['f1-score']:<10.4f} | {int(r['support']):<8}")
    print("-" * 55)
    print(f"Confusion Matrix (counts):\n{cm}")
    print("=" * 55)

    # Determine output directory
    if not output_dir:
        output_dir = os.path.dirname(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save Confusion Matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Sick"], yticklabels=["Healthy", "Sick"], ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix ({split.upper()}) - Macro F1: {macro_f1:.4f}")
    plt.tight_layout()

    cm_img_path = os.path.join(output_dir, f"confusion_matrix_{split}.png")
    plt.savefig(cm_img_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Confusion Matrix visualization to: {cm_img_path}")

    # Save metrics JSON
    results = {
        "split": split,
        "checkpoint": checkpoint_path,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }
    json_path = os.path.join(output_dir, f"evaluation_{split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved evaluation metrics to: {json_path}")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Classifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=os.path.join("ml", "data"))
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        split=args.split,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
