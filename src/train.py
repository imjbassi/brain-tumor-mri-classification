# train.py
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix

from data_loader import get_dataloaders
from model import get_model
from utils import (plot_confusion_matrix, plot_training_curves, save_checkpoint,
                    save_history, set_seed)


def run_epoch(model, device, loader, criterion, optimizer=None):
    """Shared loop for train/validation. Pass optimizer=None to run in eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train a brain tumor classifier.")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing Training/ and Testing/ subfolders.")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "efficientnet_b0"])
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Only train the new classification head.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--augment", action="store_true",
                        help="Apply random flip/rotation/color-jitter augmentation to training images.")
    parser.add_argument("--balance_classes", action="store_true",
                        help="Use a class-weighted sampler to counter class imbalance.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Stop early if validation loss doesn't improve for this many epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save the trained model checkpoint in.")
    parser.add_argument("--fig_dir", type=str, default="paper/figures",
                        help="Directory to save training_curves.png / confusion_matrix_val.png / history.json in.")
    parser.add_argument("--group_map", type=str, default=None,
                        help="JSON mapping image path -> group id (patient or near-duplicate "
                             "cluster). When given, the train/validation split is grouped so no "
                             "patient straddles it. Strongly recommended: without it, model "
                             "selection is done against slices of patients the model trained on.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    output_model = os.path.join(args.checkpoint_dir, "tumor_model.pth")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.group_map:
        print(f"Grouped train/validation split using {args.group_map}")
    else:
        print("WARNING: ungrouped train/validation split; model selection may be "
              "contaminated by patients shared between train and validation.")

    train_loader, val_loader, _, class_names = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, val_split=args.val_split,
        augment=args.augment, balance_classes=args.balance_classes, seed=args.seed,
        group_map=args.group_map,
    )
    print(f"Classes: {class_names}")

    model = get_model(num_classes=len(class_names), backbone=args.backbone,
                       freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                       factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_preds, best_val_labels = None, None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, device, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}/{args.epochs}: "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_preds, best_val_labels = val_preds, val_labels
            epochs_without_improvement = 0
            save_checkpoint(output_model, model, class_names, args.backbone)
            print(f"  New best model saved to {output_model}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break

    save_history(history, os.path.join(args.fig_dir, "history.json"))
    plot_training_curves(history, os.path.join(args.fig_dir, "training_curves.png"))

    cm = confusion_matrix(best_val_labels, best_val_preds, labels=range(len(class_names)))
    plot_confusion_matrix(cm, class_names, os.path.join(args.fig_dir, "confusion_matrix_val.png"))

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
