"""
src/models/train.py
Training loop with MLflow experiment tracking.
Run:  python -m src.models.train --model_type cnn --epochs 10
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import mlflow
import mlflow.pytorch

from src.data.preprocess import get_dataloaders
from src.models.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CLASSES = ["Cat", "Dog"]


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def plot_loss_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # Detect device: CUDA (NVIDIA), ROCm (AMD), or CPU
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using device: {device} (NVIDIA GPU)")
        logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
    else:
        device = "cpu"
        logger.info(f"Using device: {device}")
        logger.info("No GPU detected. Training will be slow.")

    # Paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.workers
    )

    # Model
    model = build_model(args.model_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("cats_dogs_classification")

    with mlflow.start_run(run_name=f"{args.model_type}_ep{args.epochs}"):
        # Log hyperparameters
        mlflow.log_params({
            "model_type": args.model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "device": device,
        })

        train_losses, val_losses = [], []
        best_val_acc = 0.0
        best_model_path = out_dir / "best_model.pt"

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            elapsed = time.time() - t0

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            logger.info(
                f"Epoch {epoch}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | {elapsed:.1f}s"
            )

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": val_acc,
                        "model_type": args.model_type,
                    },
                    best_model_path,
                )

        # ── Evaluation on test set ──────────────────────────────────────────
        _, test_acc, test_preds, test_labels = evaluate(
            model, test_loader, criterion, device
        )
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        mlflow.log_metric("test_accuracy", test_acc)

        # Save metrics for DVC
        metrics = {
            "best_val_accuracy": float(best_val_acc),
            "test_accuracy": float(test_acc),
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
        }
        metrics_path = out_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Classification report
        report = classification_report(test_labels, test_preds, target_names=CLASSES)
        logger.info("\n" + report)
        report_path = out_dir / "classification_report.txt"
        report_path.write_text(report)

        # Plots
        lc_path = out_dir / "loss_curves.png"
        cm_path = out_dir / "confusion_matrix.png"
        plot_loss_curves(train_losses, val_losses, lc_path)
        plot_confusion_matrix(test_labels, test_preds, cm_path)

        # Log artifacts
        mlflow.log_artifact(str(best_model_path))
        mlflow.log_artifact(str(lc_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metrics_path))

        # Log model with pip requirements (handle ROCm version)
        pip_requirements = [
            f"torch=={torch.__version__.split('+')[0]}",  # Remove +rocm suffix
            f"torchvision",
            "pillow",
            "numpy",
        ]
        mlflow.pytorch.log_model(model, "model", pip_requirements=pip_requirements)

        logger.info(f"Best val accuracy: {best_val_acc:.4f}")
        logger.info(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument("--data_dir",    default="data/processed", help="Processed data root")
    parser.add_argument("--output_dir",  default="models/artifacts", help="Output dir for model & plots")
    parser.add_argument("--model_type",  default="cnn", choices=["cnn", "mobilenet"])
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--workers",     type=int, default=4)
    parser.add_argument("--mlflow_uri",  default="http://localhost:5000")
    main(parser.parse_args())
