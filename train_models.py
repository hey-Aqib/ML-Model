# =============================================================================
# Project B6: Comparative Study of Evaluation Metrics
# Dataset: UCI Adult Income Dataset
# Models: Logistic Regression, Decision Tree, Random Forest, SVM, KNN
# Metrics: Accuracy, F1-Score, ROC-AUC, PR-AUC
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Spyder/Flask
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print("=" * 60)
print("  PROJECT B6: Comparative Study of Evaluation Metrics")
print("=" * 60)
print("\n[1/5] Loading dataset...")

df = pd.read_csv('data/adult.data')
print(f"  Shape: {df.shape}")
print(f"  Income distribution:\n{df['income'].value_counts()}")

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING & CLEANING
# ─────────────────────────────────────────────
print("\n[2/5] Preprocessing & Cleaning...")

# Drop fnlwgt (sampling weight - not predictive)
df.drop(columns=['fnlwgt'], inplace=True)

# Replace '?' with NaN and drop missing rows
df.replace(' ?', np.nan, inplace=True)
df.replace('?', np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"  Dropped {before - len(df)} rows with missing values.")
print(f"  Clean dataset shape: {df.shape}")

# Encode target: <=50K -> 0, >50K -> 1
df['income'] = df['income'].str.strip().map({'<=50K': 0, '>50K': 1})
df.dropna(subset=['income'], inplace=True)
df['income'] = df['income'].astype(int)

# Label encode all categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
print(f"  Label-encoded columns: {categorical_cols}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING & SCALING
# ─────────────────────────────────────────────
print("\n[3/5] Feature Scaling & Splitting...")

X = df.drop(columns=['income'])
y = df['income']

feature_names = X.columns.tolist()

# Train-Test Split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standard Scaling (required for LR, SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")
print(f"  Class balance (test) - 0:{(y_test==0).sum()}, 1:{(y_test==1).sum()}")

# ─────────────────────────────────────────────
# 4. DEFINE & TRAIN CLASSIFIERS
# ─────────────────────────────────────────────
print("\n[4/5] Training Classifiers...")

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree'      : DecisionTreeClassifier(max_depth=8, random_state=42),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'SVM'                : SVC(kernel='rbf', probability=True, random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=7),
}

# Models needing scaled data
needs_scaling = {'Logistic Regression', 'SVM', 'KNN'}

results = {}
trained_models = {}

for name, clf in classifiers.items():
    print(f"  Training {name}...", end='', flush=True)
    Xtr = X_train_scaled if name in needs_scaling else X_train.values
    Xte = X_test_scaled  if name in needs_scaling else X_test.values

    clf.fit(Xtr, y_train)
    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='binary')
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc   = average_precision_score(y_test, y_proba)

    results[name] = {
        'accuracy'   : round(acc, 4),
        'f1_score'   : round(f1, 4),
        'roc_auc'    : round(roc_auc, 4),
        'pr_auc'     : round(pr_auc, 4),
        'fpr'        : fpr,
        'tpr'        : tpr,
        'precision'  : precision,
        'recall'     : recall,
        'y_pred'     : y_pred,
        'y_proba'    : y_proba,
        'report'     : classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])
    }
    trained_models[name] = clf
    print(f" Acc={acc:.3f} | F1={f1:.3f} | ROC-AUC={roc_auc:.3f} | PR-AUC={pr_auc:.3f}")

# ─────────────────────────────────────────────
# 5. PLOTS
# ─────────────────────────────────────────────
print("\n[5/5] Generating Plots...")
os.makedirs('plots', exist_ok=True)

PALETTE = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261']
FONT = {'family': 'DejaVu Sans', 'size': 11}
plt.rc('font', **FONT)

model_names = list(results.keys())

# ── Plot 1: Metrics Bar Chart ────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(model_names))
width = 0.2
metrics = ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']
metric_labels = ['Accuracy', 'F1-Score', 'ROC-AUC', 'PR-AUC']
colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a']

for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    vals = [results[m][metric] for m in model_names]
    bars = ax.bar(x + i*width, vals, width, label=label, color=color, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(model_names, rotation=15, ha='right')
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Score')
ax.set_title('Classifier Comparison: All Evaluation Metrics', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('plots/metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/metrics_comparison.png")

# ── Plot 2: ROC Curves ───────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random Classifier')
for (name, res), color in zip(results.items(), PALETTE):
    ax.plot(res['fpr'], res['tpr'], color=color, lw=2,
            label=f"{name} (AUC={res['roc_auc']:.3f})")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves – All Classifiers', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('plots/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/roc_curves.png")

# ── Plot 3: PR Curves ────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
baseline = y_test.sum() / len(y_test)
ax.axhline(baseline, color='k', linestyle='--', alpha=0.4, label=f'Baseline (P={baseline:.2f})')
for (name, res), color in zip(results.items(), PALETTE):
    ax.plot(res['recall'], res['precision'], color=color, lw=2,
            label=f"{name} (AP={res['pr_auc']:.3f})")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves – All Classifiers', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('plots/pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/pr_curves.png")

# ── Plot 4: Confusion Matrices (grid) ────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K'])
    disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
    axes[i].set_title(name, fontweight='bold', fontsize=11)
axes[-1].set_visible(False)
fig.suptitle('Confusion Matrices – All Classifiers', fontsize=14, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig('plots/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/confusion_matrices.png")

# ── Plot 5: Feature Importance (RF) ─────────
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1][:10]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh([feature_names[i] for i in idx[::-1]], importances[idx[::-1]], color='#457b9d', edgecolor='white')
ax.set_xlabel('Feature Importance')
ax.set_title('Top 10 Feature Importances (Random Forest)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig('plots/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/feature_importance.png")

# ── Plot 6: Heatmap of metrics ───────────────
metrics_df = pd.DataFrame({
    'Accuracy': [results[m]['accuracy'] for m in model_names],
    'F1-Score': [results[m]['f1_score'] for m in model_names],
    'ROC-AUC' : [results[m]['roc_auc']  for m in model_names],
    'PR-AUC'  : [results[m]['pr_auc']   for m in model_names],
}, index=model_names)
fig, ax = plt.subplots(figsize=(9, 5))
sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0.5, vmax=1.0, annot_kws={'size': 12, 'weight': 'bold'})
ax.set_title('Metrics Heatmap', fontsize=14, fontweight='bold', pad=12)
fig.tight_layout()
fig.savefig('plots/metrics_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plots/metrics_heatmap.png")

# ─────────────────────────────────────────────
# SAVE MODELS & ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
for name, clf in trained_models.items():
    safe_name = name.lower().replace(' ', '_')
    joblib.dump(clf, f'models/{safe_name}.pkl')

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoders, 'models/label_encoders.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')

# Save summary CSV
metrics_df.to_csv('plots/metrics_summary.csv')

print("\n" + "=" * 60)
print("  TRAINING COMPLETE — Summary")
print("=" * 60)
print(metrics_df.to_string())
print("\nAll models saved to models/")
print("All plots saved to plots/")
print("Run app.py to start the Flask web interface.")
print("=" * 60)
