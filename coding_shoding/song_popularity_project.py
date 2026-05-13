

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Global plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        11,
})

COLORS = ["#6C5CE7","#00B894","#E17055","#0984E3","#FDCB6E","#D63031","#A29BFE"]

print("=" * 60)
print("  SONG POPULARITY SCORE — PROJECT")
print("=" * 60)


# TASK 1 — Load & Clean Data

print("\n[TASK 1] Loading and cleaning data...")

df_raw = pd.read_csv("spotify.csv")

vars_needed = ["popularity", "danceability", "energy",
               "loudness", "valence", "tempo", "duration_ms"]
df = df_raw[vars_needed].copy()

# Drop missing values and zero-popularity rows
df.dropna(inplace=True)
df = df[df["popularity"] > 0].copy()

# Convert duration from ms to minutes
df["duration_min"] = df["duration_ms"] / 60000
df.drop(columns=["duration_ms"], inplace=True)

# Reproducible sample of 500
df = df.sample(n=500, random_state=42).reset_index(drop=True)

print(f"  Dataset shape : {df.shape}")
print(f"  Variables     : {list(df.columns)}")

# Export to Excel for submission
df.to_excel("spotify_data.xlsx", index=False)
print("  Saved: spotify_data.xlsx")


# TASK 2 — Summary Statistics

print("\n" + "=" * 60)
print("  TASK 2 — SUMMARY STATISTICS")
print("=" * 60)

# --- Basic describe ---
desc = df.describe().T
desc.columns = ["Count","Mean","Std","Min","Q1","Median","Q3","Max"]
print("\n--- describe() ---")
print(desc.round(3).to_string())

# --- Mode ---
print("\n--- Mode ---")
for col in df.columns:
    mode_val = df[col].mode()[0]
    print(f"  {col:<15}: {mode_val:.4f}")

# --- Skewness & Kurtosis ---
print("\n--- Skewness & Kurtosis ---")
for col in df.columns:
    sk = df[col].skew()
    ku = df[col].kurtosis()
    print(f"  {col:<15}: Skewness = {sk:+.3f}  |  Kurtosis = {ku:+.3f}")

# --- IQR & Variance ---
print("\n--- IQR & Variance ---")
for col in df.columns:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    print(f"  {col:<15}: IQR = {q3-q1:.4f}  |  Variance = {df[col].var():.4f}")

# Full summary table saved to CSV
summary_full = pd.DataFrame({
    "Mean":     df.mean(),
    "Median":   df.median(),
    "Mode":     df.apply(lambda x: x.mode()[0]),
    "Std":      df.std(),
    "Variance": df.var(),
    "Min":      df.min(),
    "Max":      df.max(),
    "Q1":       df.quantile(0.25),
    "Q2":       df.quantile(0.50),
    "Q3":       df.quantile(0.75),
    "Skewness": df.skew(),
    "Kurtosis": df.kurtosis(),
}).round(4)
summary_full.to_csv("summary_statistics.csv")
print("\n  Saved: summary_statistics.csv")
print(summary_full.to_string())


# TASK 3 — Box and Whisker Plots

print("\n" + "=" * 60)
print("  TASK 3 — BOX AND WHISKER PLOTS")
print("=" * 60)

# Normalize to [0,1] so all variables fit one axis
df_norm = (df - df.min()) / (df.max() - df.min())
df_norm.columns = df.columns

fig, ax = plt.subplots(figsize=(14, 7))

bp = ax.boxplot(
    [df_norm[col].values for col in df_norm.columns],
    patch_artist=True,
    notch=False,
    vert=True,
    widths=0.55,
    flierprops=dict(marker="*", color="red", markersize=6,
                    markerfacecolor="red", linestyle="none"),
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.4),
    capprops=dict(linewidth=1.4),
)

for patch, color in zip(bp["boxes"], COLORS):
    patch.set_facecolor(color)
    patch.set_alpha(0.80)

ax.set_xticks(range(1, len(df.columns) + 1))
ax.set_xticklabels(df.columns, fontsize=12, rotation=20, ha="right")
ax.set_ylabel("Normalized Value [0–1]", fontsize=12)
ax.set_title("Box and Whisker Plots — All Variables (Normalized)",
             fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Variable", fontsize=12)

subtitle = "Red asterisks (*) indicate outliers  |  Variables scaled to [0, 1] for comparability"
fig.text(0.5, 0.01, subtitle, ha="center", fontsize=10, color="gray")

# Custom legend
from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=COLORS[i], alpha=0.8, label=col)
                  for i, col in enumerate(df.columns)]
ax.legend(handles=legend_handles, loc="upper right", fontsize=10,
          framealpha=0.85, ncol=2)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("boxplot_all_variables.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: boxplot_all_variables.png")

# --- Outlier detection (IQR rule) ---
print("\n--- Outlier Detection (IQR Rule: 1.5×IQR) ---")
for col in df.columns:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lb) | (df[col] > ub)).sum()
    print(f"  {col:<15}: Outliers = {n_out:>3}  |"
          f"  Lower fence = {lb:>8.3f}  |  Upper fence = {ub:>8.3f}")


# TASK 4 — Scatter Plot Grid

print("\n" + "=" * 60)
print("  TASK 4 — SCATTER PLOT GRID")
print("=" * 60)

cols  = df.columns.tolist()
n     = len(cols)
fig, axes = plt.subplots(n, n, figsize=(18, 16))
fig.suptitle("Scatter Plot Grid — All Variables\n"
             "Diagonal: distribution  |  Lower: scatter  |  Upper: correlation",
             fontsize=14, fontweight="bold", y=1.01)

for i, col_y in enumerate(cols):
    for j, col_x in enumerate(cols):
        ax = axes[i, j]
        ax.tick_params(labelsize=7)

        if i == j:
            # Diagonal — KDE + histogram
            ax.hist(df[col_x], bins=20, color=COLORS[i],
                    alpha=0.55, edgecolor="white", density=True)
            kde_x = np.linspace(df[col_x].min(), df[col_x].max(), 200)
            kde   = stats.gaussian_kde(df[col_x])
            ax.plot(kde_x, kde(kde_x), color=COLORS[i], linewidth=2)
            ax.set_facecolor("#f9f9f9")

        elif i > j:
            # Lower triangle — scatter plot
            ax.scatter(df[col_x], df[col_y],
                       alpha=0.35, s=12, color="#6C5CE7", edgecolors="none")
            # Regression line
            m, b, _, _, _ = stats.linregress(df[col_x], df[col_y])
            x_line = np.linspace(df[col_x].min(), df[col_x].max(), 100)
            ax.plot(x_line, m * x_line + b, color="#E17055",
                    linewidth=1.4, alpha=0.85)

        else:
            # Upper triangle — correlation coefficient
            r, p = stats.pearsonr(df[col_x], df[col_y])
            color_r = "#D63031" if abs(r) > 0.5 else \
                      "#0984E3" if abs(r) > 0.3 else "#636e72"
            ax.text(0.5, 0.5, f"r = {r:.3f}",
                    ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color_r,
                    transform=ax.transAxes)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else \
                  "*"   if p < 0.05  else "ns"
            ax.text(0.5, 0.28, sig,
                    ha="center", va="center",
                    fontsize=10, color="gray",
                    transform=ax.transAxes)
            ax.set_facecolor("#fafafa")
            ax.set_xticks([])
            ax.set_yticks([])

        # Row/column labels on edges
        if i == n - 1:
            ax.set_xlabel(col_x, fontsize=8, labelpad=4)
        if j == 0:
            ax.set_ylabel(col_y, fontsize=8, labelpad=4)

plt.tight_layout()
plt.savefig("scatterplot_grid.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: scatterplot_grid.png")

# Correlation matrix printout
print("\n--- Pearson Correlation Matrix ---")
print(df.corr().round(3).to_string())

# TASK 5 — Models

print("\n" + "=" * 60)
print("  TASK 5 — MODEL ESTIMATION")
print("=" * 60)

# --- Train / Test Split (80 / 20) ---
X = df.drop(columns=["popularity"])
y = df["popularity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\n  Training set : {X_train.shape[0]} observations")
print(f"  Test set     : {X_test.shape[0]} observations")


# MODEL 1 — Multiple Linear Regression (statsmodels for full output)

print("\n--- Model 1: Multiple Linear Regression ---")

X_train_sm = sm.add_constant(X_train)
X_test_sm  = sm.add_constant(X_test)

mlr_model  = sm.OLS(y_train, X_train_sm).fit()
print(mlr_model.summary())

# Predictions
pred_mlr = mlr_model.predict(X_test_sm)

mse_mlr  = mean_squared_error(y_test, pred_mlr)
rmse_mlr = np.sqrt(mse_mlr)
mae_mlr  = mean_absolute_error(y_test, pred_mlr)
r2_mlr   = r2_score(y_test, pred_mlr)

print(f"\n  MLR Test Performance:")
print(f"    MSE  = {mse_mlr:.4f}")
print(f"    RMSE = {rmse_mlr:.4f}")
print(f"    MAE  = {mae_mlr:.4f}")
print(f"    R²   = {r2_mlr:.4f}")

# Print equation
coefs = mlr_model.params
eq = f"Popularity = {coefs['const']:.3f}"
for col in X.columns:
    sign = "+" if coefs[col] >= 0 else "-"
    eq  += f" {sign} {abs(coefs[col]):.4f}×{col}"
print(f"\n  MLR Equation:\n  {eq}")


# MODEL 2 — Random Forest

print("\n--- Model 2: Random Forest ---")

rf_model = RandomForestRegressor(
    n_estimators = 500,
    max_features = 2,
    random_state = 42,
    n_jobs       = -1
)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

mse_rf   = mean_squared_error(y_test, pred_rf)
rmse_rf  = np.sqrt(mse_rf)
mae_rf   = mean_absolute_error(y_test, pred_rf)
r2_rf    = r2_score(y_test, pred_rf)

print(f"\n  Random Forest Test Performance:")
print(f"    MSE  = {mse_rf:.4f}")
print(f"    RMSE = {rmse_rf:.4f}")
print(f"    MAE  = {mae_rf:.4f}")
print(f"    R²   = {r2_rf:.4f}")

# Variable Importance
imp_df = pd.DataFrame({
    "Variable":  X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)
print(f"\n  Variable Importance (RF):\n{imp_df.to_string(index=False)}")


# COMPARISON PLOTS

print("\n--- Generating Comparison Plots ---")

results = pd.DataFrame({
    "Actual":   y_test.values,
    "MLR":      pred_mlr.values,
    "RF":       pred_rf,
    "Index":    range(len(y_test))
})

fig = plt.figure(figsize=(16, 13))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

# --- Plot 1: Actual vs Predicted — MLR ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(results["Actual"], results["MLR"],
            color="#6C5CE7", alpha=0.6, s=30, edgecolors="none",
            label="Observations")
lim = [min(results["Actual"].min(), results["MLR"].min()) - 2,
       max(results["Actual"].max(), results["MLR"].max()) + 2]
ax1.plot(lim, lim, "r--", linewidth=1.6, label="Perfect prediction")
ax1.set_xlim(lim); ax1.set_ylim(lim)
ax1.set_xlabel("Actual Popularity");  ax1.set_ylabel("Predicted Popularity")
ax1.set_title("MLR: Actual vs Predicted", fontweight="bold")
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f"MSE = {mse_mlr:.2f}\nR² = {r2_mlr:.3f}",
         transform=ax1.transAxes, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.7))

# --- Plot 2: Actual vs Predicted — RF ---
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(results["Actual"], results["RF"],
            color="#00B894", alpha=0.6, s=30, edgecolors="none",
            label="Observations")
ax2.plot(lim, lim, "r--", linewidth=1.6, label="Perfect prediction")
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel("Actual Popularity"); ax2.set_ylabel("Predicted Popularity")
ax2.set_title("Random Forest: Actual vs Predicted", fontweight="bold")
ax2.legend(fontsize=9)
ax2.text(0.05, 0.92, f"MSE = {mse_rf:.2f}\nR² = {r2_rf:.3f}",
         transform=ax2.transAxes, fontsize=9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", alpha=0.7))

# --- Plot 3: Line chart — Actual vs both predictions ---
ax3 = fig.add_subplot(gs[1, 0])
idx = results["Index"]
ax3.plot(idx, results["Actual"], color="black",     lw=1.2,
         alpha=0.85, label="Actual")
ax3.plot(idx, results["MLR"],    color="#6C5CE7",   lw=1.2,
         alpha=0.80, label="MLR Predicted", linestyle="--")
ax3.plot(idx, results["RF"],     color="#00B894",   lw=1.2,
         alpha=0.80, label="RF Predicted",  linestyle="-.")
ax3.set_xlabel("Observation Index"); ax3.set_ylabel("Popularity Score")
ax3.set_title("Actual vs Predicted — Both Models", fontweight="bold")
ax3.legend(fontsize=9)

# --- Plot 4: MSE Bar Chart ---
ax4 = fig.add_subplot(gs[1, 1])
models = ["Multiple Linear\nRegression", "Random Forest"]
mse_vals = [mse_mlr, mse_rf]
bars = ax4.bar(models, mse_vals, color=["#6C5CE7", "#00B894"],
               width=0.45, alpha=0.88, edgecolor="white")
for bar, val in zip(bars, mse_vals):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.3,
             f"{val:.2f}", ha="center", va="bottom",
             fontweight="bold", fontsize=12)
ax4.set_ylabel("Mean Squared Error (MSE)")
ax4.set_title("Model Comparison — MSE\n(lower is better)", fontweight="bold")
ax4.set_ylim(0, max(mse_vals) * 1.25)

fig.suptitle("Song Popularity Score — Model Comparison Dashboard",
             fontsize=15, fontweight="bold", y=1.01)
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: model_comparison.png")

# --- Variable Importance Plot ---
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(imp_df["Variable"], imp_df["Importance"],
               color="#00B894", alpha=0.85, edgecolor="white")
for bar, val in zip(bars, imp_df["Importance"]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=10)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
ax.set_title("Random Forest — Variable Importance",
             fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("variable_importance.png", dpi=300, bbox_inches="tight")
plt.show()
print("  Saved: variable_importance.png")


# FINAL SUMMARY TABLE
print("\n" + "=" * 62)
print("  FINAL MODEL COMPARISON SUMMARY")
print("=" * 62)
print(f"  {'Metric':<10} {'MLR':>14} {'Random Forest':>16} {'Winner':>10}")
print("  " + "-" * 52)
metrics = {
    "MSE":  (mse_mlr,  mse_rf),
    "RMSE": (rmse_mlr, rmse_rf),
    "MAE":  (mae_mlr,  mae_rf),
    "R²":   (r2_mlr,   r2_rf),
}
for metric, (v1, v2) in metrics.items():
    if metric == "R²":
        winner = "MLR" if v1 > v2 else "RF"
    else:
        winner = "MLR" if v1 < v2 else "RF"
    print(f"  {metric:<10} {v1:>14.4f} {v2:>16.4f} {winner:>10}")

print("\n  Files saved:")
for f in ["spotify_data.xlsx", "summary_statistics.csv",
          "boxplot_all_variables.png", "scatterplot_grid.png",
          "model_comparison.png", "variable_importance.png"]:
    print(f"    ✓ {f}")
print("\n  Done dana don done ! \n")




# SAVE RESULTS FOR DYNAMIC REPORT GENERATION

import json

def get_mode(series):
    return round(float(series.mode()[0]), 4)

# Summary statistics per variable
summary_data = {}
for col in df.columns:
    q1  = float(df[col].quantile(0.25))
    q3  = float(df[col].quantile(0.75))
    summary_data[col] = {
        "mean":     round(float(df[col].mean()), 4),
        "median":   round(float(df[col].median()), 4),
        "mode":     get_mode(df[col]),
        "std":      round(float(df[col].std()), 4),
        "q1":       round(q1, 4),
        "q3":       round(q3, 4),
        "min":      round(float(df[col].min()), 4),
        "max":      round(float(df[col].max()), 4),
        "skewness": round(float(df[col].skew()), 4),
        "kurtosis": round(float(df[col].kurtosis()), 4),
        "variance": round(float(df[col].var()), 4),
    }

# MLR coefficients
mlr_results = {}
for var in X_train_sm.columns:
    mlr_results[var] = {
        "coef":    round(float(mlr_model.params[var]), 4),
        "std_err": round(float(mlr_model.bse[var]), 4),
        "t_stat":  round(float(mlr_model.tvalues[var]), 4),
        "p_value": round(float(mlr_model.pvalues[var]), 4),
    }

# Outlier counts
outliers = {}
for col in df.columns:
    q1  = df[col].quantile(0.25)
    q3  = df[col].quantile(0.75)
    iqr = q3 - q1
    outliers[col] = int(((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum())

# RF variable importance
importance_dict = {
    col: round(float(val), 6)
    for col, val in zip(X.columns, rf_model.feature_importances_)
}
top_rf_var    = max(importance_dict, key=importance_dict.get)
second_rf_var = sorted(importance_dict, key=importance_dict.get, reverse=True)[1]

# MLR significant variables (p < 0.05), exclude intercept
sig_vars = [
    k for k, v in mlr_results.items()
    if v["p_value"] < 0.05 and k != "const"
]

results_json = {
    "n_total":      len(df),
    "n_train":      len(X_train),
    "n_test":       len(X_test),
    "summary":      summary_data,
    "mlr":          mlr_results,
    "mlr_r2":       round(float(mlr_model.rsquared), 4),
    "mlr_adj_r2":   round(float(mlr_model.rsquared_adj), 4),
    "mlr_f_stat":   round(float(mlr_model.fvalue), 4),
    "mlr_f_pval":   round(float(mlr_model.f_pvalue), 6),
    "mlr_aic":      round(float(mlr_model.aic), 4),
    "mlr_bic":      round(float(mlr_model.bic), 4),
    "metrics": {
        "mse_mlr":   round(mse_mlr,  4),
        "rmse_mlr":  round(rmse_mlr, 4),
        "mae_mlr":   round(mae_mlr,  4),
        "r2_mlr":    round(r2_mlr,   4),
        "mse_rf":    round(mse_rf,   4),
        "rmse_rf":   round(rmse_rf,  4),
        "mae_rf":    round(mae_rf,   4),
        "r2_rf":     round(r2_rf,    4),
    },
    "importance":    importance_dict,
    "top_rf_var":    top_rf_var,
    "second_rf_var": second_rf_var,
    "sig_vars":      sig_vars,
    "outliers":      outliers,
    "winner_mse":    "Random Forest" if mse_rf < mse_mlr else "MLR",
}

with open("results.json", "w") as f:
    json.dump(results_json, f, indent=2)

print("\n  Saved: results.json  (used by generate_report.py)")