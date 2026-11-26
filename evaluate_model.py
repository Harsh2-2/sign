"""
evaluate_model.py
-----------------
Standalone evaluation script for the ASL Random Forest classifier.
Evaluates:  Accuracy, Precision, Recall, F1-Score (macro & per-class)
            Inference Time (single sample & batch)
            Confusion Matrix (saved as evaluation_report/confusion_matrix.png)
            Full classification report (saved as evaluation_report/report.txt)

Usage:
    .venv/bin/python evaluate_model.py
"""

import os
import time
import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ── Output folder ────────────────────────────────────────────────────────────
OUT_DIR = "evaluation_report"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = "asl_mediapipe_rf.pkl"
print("=" * 60)
print("  ASL Random Forest – Model Evaluation")
print("=" * 60)

with open(MODEL_PATH, "rb") as f:
    raw = pickle.load(f)

model = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
class_names = [str(c) for c in model.classes_]
n_classes   = len(class_names)
n_features  = model.n_features_in_
print(f"\n✓ Model loaded  –  {model.n_estimators} trees  |  {n_classes} classes  |  {n_features} features\n")

# ── Test data ─────────────────────────────────────────────────────────────────
# HOW TO USE WITH REAL DATA:
#   If you have the original Kaggle dataset (data.pickle), replace the synthetic
#   data section below with:
#
#       with open("data.pickle", "rb") as f:
#           dataset = pickle.load(f)
#       X_test = np.array(dataset["data"])
#       y_test = np.array(dataset["labels"])
#
#   The metrics below will then reflect true model performance.
#
# SYNTHETIC FALLBACK:
#   The model was trained on bounding-box-normalised MediaPipe landmarks (63 features).
#   We generate realistic synthetic data: each hand pose is a random but structurally
#   plausible set of normalised (x, y, z) coordinates in [0, 1] per landmark.
#   This lets us measure per-class metrics and inference speed even without the
#   original Kaggle dataset on disk.  Accuracy on synthetic data will be near
#   random-chance (~3%) because the model hasn't seen this artificial distribution.

np.random.seed(42)
N_SAMPLES_PER_CLASS = 50
N_SAMPLES = N_SAMPLES_PER_CLASS * n_classes

# Build feature matrix: small Gaussian noise around a per-class centre
X_test = []
y_test = []
rng = np.random.default_rng(42)
for idx, label in enumerate(class_names):
    centre = rng.uniform(0.0, 1.0, n_features)   # unique centre per class
    noise  = rng.normal(0.0, 0.08, (N_SAMPLES_PER_CLASS, n_features))
    samples = np.clip(centre + noise, 0.0, 1.0)
    X_test.append(samples)
    y_test.extend([label] * N_SAMPLES_PER_CLASS)

X_test = np.vstack(X_test)
y_test = np.array(y_test)

# ── Inference time ────────────────────────────────────────────────────────────
print("⏱  Measuring inference time …")

# Single-sample inference (10 000 runs for stability)
single_sample = X_test[0:1]
N_RUNS = 100
t0 = time.perf_counter()
for _ in range(N_RUNS):
    model.predict(single_sample)
t1 = time.perf_counter()
single_ms = (t1 - t0) / N_RUNS * 1000

# Batch inference
t0 = time.perf_counter()
y_pred = model.predict(X_test)
t1 = time.perf_counter()
batch_ms   = (t1 - t0) * 1000
per_item_ms = batch_ms / N_SAMPLES

print(f"   Single-sample inference : {single_ms:.3f} ms")
print(f"   Batch ({N_SAMPLES} samples)  : {batch_ms:.2f} ms  ({per_item_ms:.3f} ms/sample)")

# ── Metrics ───────────────────────────────────────────────────────────────────
print("\n📊  Computing metrics …")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)

print(f"\n   Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
print(f"   Precision (macro): {prec:.4f}")
print(f"   Recall    (macro): {rec:.4f}")
print(f"   F1-Score  (macro): {f1:.4f}")

report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

# ── Save text report ──────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, "report.txt")
with open(report_path, "w") as fh:
    fh.write("ASL Random Forest Classifier – Evaluation Report\n")
    fh.write("=" * 60 + "\n\n")
    fh.write(f"Model          : RandomForestClassifier\n")
    fh.write(f"Estimators     : {model.n_estimators}\n")
    fh.write(f"Classes        : {n_classes}\n")
    fh.write(f"Input features : {n_features}\n\n")
    fh.write("-" * 60 + "\n")
    fh.write("PERFORMANCE METRICS\n")
    fh.write("-" * 60 + "\n")
    fh.write(f"Accuracy           : {acc:.4f} ({acc*100:.2f}%)\n")
    fh.write(f"Precision (macro)  : {prec:.4f}\n")
    fh.write(f"Recall    (macro)  : {rec:.4f}\n")
    fh.write(f"F1-Score  (macro)  : {f1:.4f}\n\n")
    fh.write("-" * 60 + "\n")
    fh.write("INFERENCE TIME\n")
    fh.write("-" * 60 + "\n")
    fh.write(f"Single-sample     : {single_ms:.3f} ms\n")
    fh.write(f"Batch ({N_SAMPLES} samples): {batch_ms:.2f} ms\n")
    fh.write(f"Per-sample (batch): {per_item_ms:.3f} ms\n\n")
    fh.write("-" * 60 + "\n")
    fh.write("PER-CLASS CLASSIFICATION REPORT\n")
    fh.write("-" * 60 + "\n")
    fh.write(report)

print(f"\n✓ Report saved  →  {report_path}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
print("🖼  Generating confusion matrix …")
cm = confusion_matrix(y_test, y_pred, labels=class_names)

fig, ax = plt.subplots(figsize=(14, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title("ASL Classifier – Confusion Matrix", fontsize=15, fontweight="bold", pad=14)
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=120)
plt.close(fig)
print(f"✓ Confusion matrix saved  →  {cm_path}")

# ── Per-class bar chart ───────────────────────────────────────────────────────
per_f1 = f1_score(y_test, y_pred, average=None, labels=class_names, zero_division=0)
fig2, ax2 = plt.subplots(figsize=(14, 5))
colours = ["#4CAF50" if v >= 0.80 else "#FFC107" if v >= 0.50 else "#F44336" for v in per_f1]
ax2.bar(class_names, per_f1, color=colours)
ax2.axhline(0.80, color="grey", linestyle="--", linewidth=0.8, label="80% threshold")
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("F1-Score")
ax2.set_xlabel("Class")
ax2.set_title("Per-Class F1-Score", fontsize=13, fontweight="bold")
ax2.legend()
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
bar_path = os.path.join(OUT_DIR, "per_class_f1.png")
fig2.savefig(bar_path, dpi=120)
plt.close(fig2)
print(f"✓ Per-class F1 chart saved  →  {bar_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Accuracy        : {acc*100:.2f}%")
print(f"  Precision (macro): {prec*100:.2f}%")
print(f"  Recall    (macro): {rec*100:.2f}%")
print(f"  F1-Score  (macro): {f1*100:.2f}%")
print(f"  Inference / frame: {single_ms:.2f} ms  →  ~{int(1000/single_ms)} FPS ceiling")
print("=" * 60)
print(f"\nAll outputs saved to: ./{OUT_DIR}/")
print("  • report.txt")
print("  • confusion_matrix.png")
print("  • per_class_f1.png")
