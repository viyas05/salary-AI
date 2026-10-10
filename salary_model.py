"""
Salary Prediction ML Model
Dataset: Salary Data.csv (375 rows, features: Age, Gender, Education Level, Job Title, Years of Experience)
Target: Salary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/Salary Data.csv")
df.columns = df.columns.str.strip()

# Drop rows where target is missing
df = df.dropna(subset=["Salary"])

print(f"Dataset shape after dropping missing target: {df.shape}")
print(f"Salary range: ${df['Salary'].min():,.0f} – ${df['Salary'].max():,.0f}")
print(f"Mean salary: ${df['Salary'].mean():,.0f}\n")

# ─── 2. Feature / Target Split ────────────────────────────────────────────────

FEATURES = ["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]
TARGET   = "Salary"

X = df[FEATURES]
y = df[TARGET]

# ─── 3. Preprocessing ─────────────────────────────────────────────────────────

numeric_features     = ["Age", "Years of Experience"]
categorical_features = ["Gender", "Education Level", "Job Title"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ─── 4. Train / Test Split ────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

# ─── 5. Models ────────────────────────────────────────────────────────────────

models = {
    "Ridge Regression":        Ridge(alpha=10),
    "Random Forest":           RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
}

results = {}

print(f"{'Model':<25} {'MAE':>10} {'RMSE':>10} {'R²':>8}  {'CV R² (5-fold)':>15}")
print("-" * 72)

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    cv_mean   = cv_scores.mean()

    results[name] = {
        "pipeline": pipe,
        "y_pred":   y_pred,
        "MAE":      mae,
        "RMSE":     rmse,
        "R2":       r2,
        "CV_R2":    cv_mean,
    }

    print(f"{name:<25} ${mae:>9,.0f} ${rmse:>9,.0f} {r2:>7.4f}  {cv_mean:>8.4f} ± {cv_scores.std():.4f}")

# ─── 6. Best Model ────────────────────────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["R2"])
best      = results[best_name]
print(f"\n[BEST] Best model: {best_name}  (Test R2 = {best['R2']:.4f})")

# ─── 7. Feature Importances (Random Forest) ───────────────────────────────────

rf_pipe     = results["Random Forest"]["pipeline"]
rf_model    = rf_pipe.named_steps["model"]
ohe_cols    = rf_pipe.named_steps["preprocessor"] \
                .named_transformers_["cat"] \
                .named_steps["encoder"] \
                .get_feature_names_out(categorical_features)
all_features = numeric_features + list(ohe_cols)
importances  = rf_model.feature_importances_

# Collapse OHE columns back to original feature groups for readability
feature_group_importance = {}
for feat in FEATURES:
    if feat in numeric_features:
        idx = all_features.index(feat)
        feature_group_importance[feat] = importances[idx]
    else:
        prefix = feat + "_"
        idxs   = [i for i, f in enumerate(all_features) if f.startswith(prefix)]
        feature_group_importance[feat] = importances[idxs].sum()

print("\nFeature group importances (Random Forest):")
for feat, imp in sorted(feature_group_importance.items(), key=lambda x: -x[1]):
    print(f"  {feat:<25} {imp:.4f}")

# ─── 8. Plots ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Salary Prediction – Model Evaluation", fontsize=14, fontweight="bold")

# 8a. Actual vs Predicted (best model)
ax = axes[0]
ax.scatter(y_test, best["y_pred"], alpha=0.6, edgecolors="k", linewidths=0.4)
lim = [y_test.min() * 0.9, y_test.max() * 1.05]
ax.plot(lim, lim, "r--", linewidth=1.5)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual Salary ($)"); ax.set_ylabel("Predicted Salary ($)")
ax.set_title(f"Actual vs Predicted\n{best_name} (R²={best['R2']:.4f})")

# 8b. Residuals distribution (best model)
residuals = y_test.values - best["y_pred"]
ax = axes[1]
ax.hist(residuals, bins=25, color="steelblue", edgecolor="white")
ax.axvline(0, color="red", linestyle="--")
ax.set_xlabel("Residual ($)"); ax.set_ylabel("Count")
ax.set_title("Residuals Distribution")

# 8c. Feature group importance (Random Forest)
ax = axes[2]
labels, values = zip(*sorted(feature_group_importance.items(), key=lambda x: x[1]))
ax.barh(labels, values, color="teal")
ax.set_xlabel("Importance"); ax.set_title("Feature Importances\n(Random Forest)")

plt.tight_layout()
plt.savefig("salary_model_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> salary_model_results.png")

# ─── 9. Sample Predictions ────────────────────────────────────────────────────

best_pipe = best["pipeline"]

sample = pd.DataFrame([
    {"Age": 30, "Gender": "Male",   "Education Level": "Bachelor's", "Job Title": "Software Engineer",  "Years of Experience": 4},
    {"Age": 35, "Gender": "Female", "Education Level": "Master's",   "Job Title": "Data Analyst",        "Years of Experience": 8},
    {"Age": 50, "Gender": "Male",   "Education Level": "PhD",        "Job Title": "Senior Manager",      "Years of Experience": 20},
    {"Age": 27, "Gender": "Female", "Education Level": "Bachelor's", "Job Title": "Marketing Coordinator","Years of Experience": 2},
])

preds = best_pipe.predict(sample)
print(f"\nSample predictions ({best_name}):")
for i, row in sample.iterrows():
    print(f"  {row['Job Title']:<28} {row['Education Level']:<12} {int(row['Years of Experience'])} yrs exp  ->  ${preds[i]:>10,.0f}")
