# ============================================================
# 🩺 Diabetes Prediction — End-to-End ML Script
# Dataset: Pima Indians Diabetes Dataset
# Model: Random Forest Classifier
# ============================================================

# ── 1. Import Libraries ──────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer

# Plot style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 120

print('✅ Libraries imported successfully!')


# ── 2. Load & Inspect Data ───────────────────────────────────
URL = 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv'
df = pd.read_csv(URL)

print(f'\nDataset shape: {df.shape}')
print(df.head())

print('\n=== Data Types & Non-null Counts ===')
df.info()

print('\n=== Descriptive Statistics ===')
print(df.describe().T)

print('\n=== Target Variable Distribution ===')
print(df['Outcome'].value_counts())
print(f"\nDiabetes prevalence: {df['Outcome'].mean()*100:.1f}%")


# ── 3. Exploratory Data Analysis (EDA) ──────────────────────

# Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['Outcome'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'],
                                   edgecolor='white', width=0.5)
axes[0].set_title('Target Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'], rotation=0)
axes[0].set_ylabel('Count')
df['Outcome'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                   colors=['steelblue', 'tomato'], startangle=90,
                                   labels=['No Diabetes', 'Diabetes'])
axes[1].set_title('Target Distribution (%)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig('plot_target_distribution.png')
plt.show()
print('📊 Saved: plot_target_distribution.png')

# Feature distributions by Outcome
features = [c for c in df.columns if c != 'Outcome']
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    df[df['Outcome'] == 0][feat].hist(ax=axes[i], alpha=0.6, color='steelblue', label='No Diabetes', bins=20)
    df[df['Outcome'] == 1][feat].hist(ax=axes[i], alpha=0.6, color='tomato', label='Diabetes', bins=20)
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].legend(fontsize=8)
plt.suptitle('Feature Distributions by Diabetes Outcome', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('plot_feature_distributions.png')
plt.show()
print('📊 Saved: plot_feature_distributions.png')

# Correlation heatmap
plt.figure(figsize=(10, 7))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_correlation_heatmap.png')
plt.show()
print('📊 Saved: plot_correlation_heatmap.png')

# Boxplots
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
for i, feat in enumerate(features):
    df.boxplot(column=feat, by='Outcome', ax=axes[i],
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='tomato', linewidth=2))
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Outcome (0=No, 1=Yes)')
plt.suptitle('Feature Boxplots by Diabetes Outcome', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot_boxplots.png')
plt.show()
print('📊 Saved: plot_boxplots.png')


# ── 4. Data Preprocessing ────────────────────────────────────
# Replace biologically invalid 0s with NaN
zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_clean = df.copy()
df_clean[zero_not_allowed] = df_clean[zero_not_allowed].replace(0, np.nan)

print('\nMissing values after replacing 0s:')
missing = df_clean.isnull().sum()
missing_pct = (df_clean.isnull().sum() / len(df_clean)) * 100
print(pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct.round(1)}).query('`Missing Count` > 0'))

# Impute with median
imputer = SimpleImputer(strategy='median')
df_clean[zero_not_allowed] = imputer.fit_transform(df_clean[zero_not_allowed])
print(f'\nRemaining missing values: {df_clean.isnull().sum().sum()}')
print('✅ Imputation complete!')


# ── 5. Feature Engineering ───────────────────────────────────
df_clean['BMI_Age']         = df_clean['BMI'] * df_clean['Age']
df_clean['Glucose_Insulin'] = df_clean['Glucose'] / (df_clean['Insulin'] + 1)
df_clean['BP_Category']     = pd.cut(df_clean['BloodPressure'],
                                      bins=[0, 80, 90, 200],
                                      labels=[0, 1, 2]).astype(float)

print(f'\nFeatures after engineering: {df_clean.shape[1] - 1}')

# Split features and target
X = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'Train set: {X_train.shape[0]} samples')
print(f'Test  set: {X_test.shape[0]} samples')

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print('✅ Scaling complete!')


# ── 6. Baseline Random Forest ────────────────────────────────
rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train_scaled, y_train)

y_pred_base = rf_base.predict(X_test_scaled)
print(f'\nBaseline Accuracy: {accuracy_score(y_test, y_pred_base):.4f}')
print(f'Baseline ROC-AUC:  {roc_auc_score(y_test, rf_base.predict_proba(X_test_scaled)[:,1]):.4f}')

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_base, X_train_scaled, y_train, cv=cv, scoring='roc_auc')
print(f'\n5-Fold CV ROC-AUC Scores: {cv_scores.round(4)}')
print(f'Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')


# ── 7. Hyperparameter Tuning ─────────────────────────────────
param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features'     : ['sqrt', 'log2'],
    'class_weight'     : [None, 'balanced']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)
rf_grid.fit(X_train_scaled, y_train)

print(f'\n✅ Best Parameters: {rf_grid.best_params_}')
print(f'Best CV ROC-AUC:   {rf_grid.best_score_:.4f}')

best_rf = rf_grid.best_estimator_
y_pred      = best_rf.predict(X_test_scaled)
y_pred_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

print(f'\nTuned Model Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Tuned Model ROC-AUC:  {roc_auc_score(y_test, y_pred_prob):.4f}')


# ── 8. Model Evaluation ──────────────────────────────────────
print('\n=== Classification Report ===')
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix + ROC Curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Diabetes', 'Diabetes'])
disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
auc = roc_auc_score(y_test, y_pred_prob)
axes[1].plot(fpr, tpr, color='tomato', lw=2, label=f'Random Forest (AUC = {auc:.3f})')
axes[1].plot([0,1],[0,1],'k--', lw=1, label='Random Classifier')
axes[1].fill_between(fpr, tpr, alpha=0.1, color='tomato')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve', fontsize=13, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('plot_confusion_roc.png')
plt.show()
print('📊 Saved: plot_confusion_roc.png')


# ── 9. Feature Importance ────────────────────────────────────
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(10, 6))
colors = ['tomato' if v > importances_sorted.median() else 'steelblue' for v in importances_sorted]
bars = plt.barh(importances_sorted.index, importances_sorted.values, color=colors, edgecolor='white')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest — Feature Importance', fontsize=14, fontweight='bold')
for bar, val in zip(bars, importances_sorted.values):
    plt.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('plot_feature_importance.png')
plt.show()
print('📊 Saved: plot_feature_importance.png')

print('\nTop 5 Most Important Features:')
print(importances.sort_values(ascending=False).head())


# ── 10. Save Model & Scaler ──────────────────────────────────
joblib.dump(best_rf, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ Model saved as diabetes_model.pkl")
print("✅ Scaler saved as scaler.pkl")

print("\n🎉 All done! Script completed successfully.")
