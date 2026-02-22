"""
scripts/evaluate.py
Standalone evaluation on the test set. Called by DVC 'evaluate' stage.
"""
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import get_dataloaders
from src.models.model import load_model

CLASSES = ["Cat", "Dog"]
device = "cuda" if torch.cuda.is_available() else "cpu"

_, _, test_loader = get_dataloaders("data/processed", batch_size=32)
model = load_model("models/artifacts/best_model.pt", device=device)

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
acc = (all_preds == all_labels).mean()

# Save metrics
metrics = {"test_accuracy": float(acc)}
json.dump(metrics, open("models/artifacts/metrics.json", "w"), indent=2)

# Save report
report = classification_report(all_labels, all_preds, target_names=CLASSES)
open("models/artifacts/classification_report.txt", "w").write(report)

# Save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASSES, yticklabels=CLASSES, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Test Acc: {acc:.4f})")
plt.tight_layout()
plt.savefig("models/artifacts/confusion_matrix.png")
plt.close()

print(f"Test Accuracy: {acc:.4f}")
print(report)
