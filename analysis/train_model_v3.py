"""
train_model_v3.py — TTC Bus Delay Regression (LightGBM)
Predicts minutes behind schedule given route, time, weather, incident type.
Target: Min_Delay (log-transformed during training, reported in minutes)
Run with: python train_model_v3.py
"""
import os
import sys
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "..", "data", "processed", "master_ttc_model_ready.csv")
LOG_PATH    = os.path.join(SCRIPT_DIR, "..", "logs",    "train_model_v3_log.txt")
BUNDLE_PATH = os.path.join(SCRIPT_DIR, "..", "models",  "lgbm_ttc_regressor_bundle.pkl")
FIG_DIR     = os.path.join(SCRIPT_DIR, "..", "outputs")

for path in [LOG_PATH, BUNDLE_PATH, os.path.join(FIG_DIR, "_")]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Column config ─────────────────────────────────────────────────────────────
TARGET   = "Min_Delay"
CAT_COLS = ["Day", "Season", "Incident_Category"]
NUM_COLS = [
    "Hour", "Month", "Is_Rush_Hour", "Is_Weekend",
    "Temp_C", "Visibility_km", "Wind_Spd_kmh", "Rel_Humidity_pct",
]
EXTRA_NUM = ["Route_Delay_Avg", "Rush_x_Route_Avg", "Hour_x_Month"]


# ── Feature engineering ───────────────────────────────────────────────────────
def build_route_encoding(train_df: pd.DataFrame) -> tuple[dict, float]:
    """
    Per-route average delay computed on training data only.
    Applied to val/test to prevent leakage.
    """
    rate      = train_df.groupby("Route_Number")[TARGET].mean().to_dict()
    global_avg = train_df[TARGET].mean()
    log.info(f"Route encoding: {len(rate)} unique routes | global avg: {global_avg:.1f} min")
    return rate, global_avg


def engineer_features(df: pd.DataFrame, route_encoding: dict, global_avg: float) -> pd.DataFrame:
    df = df.copy()
    df["Route_Delay_Avg"]   = df["Route_Number"].map(route_encoding).fillna(global_avg)
    df["Rush_x_Route_Avg"]  = df["Is_Rush_Hour"] * df["Route_Delay_Avg"]
    df["Hour_x_Month"]      = df["Hour"] * df["Month"]
    return df


def encode_and_align(train, val, test):
    feature_cols = NUM_COLS + EXTRA_NUM + CAT_COLS
    combined     = pd.concat(
        [train[feature_cols], val[feature_cols], test[feature_cols]],
        keys=["train", "val", "test"]
    )
    encoded    = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)
    encoded.fillna(0, inplace=True)

    X_train    = encoded.xs("train").values
    X_val      = encoded.xs("val").values
    X_test     = encoded.xs("test").values
    feat_names = encoded.columns.tolist()

    log.info(f"Feature matrix shape (train): {X_train.shape}")
    log.info(f"Features ({len(feat_names)}): {feat_names}")
    return X_train, X_val, X_test, feat_names


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X_train, y_train_log) -> lgb.LGBMRegressor:
    log.info("Training LightGBM regressor on log1p(Min_Delay) ...")
    clf = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train_log)
    log.info("Training complete.")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(clf, X, y_true_raw, split_name: str) -> dict:
    """
    Predict in log space, convert back to minutes, report metrics in minutes.
    """
    y_pred_log = clf.predict(X)
    y_pred     = np.expm1(y_pred_log)          # back to minutes
    y_pred     = np.clip(y_pred, 1, 300)        # keep in valid range

    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred))
    mae  = mean_absolute_error(y_true_raw, y_pred)
    r2   = r2_score(y_true_raw, y_pred)

    # Practical accuracy: % predictions within ±5 and ±10 minutes
    within_5  = np.mean(np.abs(y_pred - y_true_raw) <= 5)  * 100
    within_10 = np.mean(np.abs(y_pred - y_true_raw) <= 10) * 100

    log.info(f"\n{'='*60}")
    log.info(f"  {split_name.upper()}")
    log.info(f"{'='*60}")
    log.info(f"  RMSE:          {rmse:.2f} min")
    log.info(f"  MAE:           {mae:.2f} min")
    log.info(f"  R²:            {r2:.4f}")
    log.info(f"  Within ±5 min: {within_5:.1f}%")
    log.info(f"  Within ±10min: {within_10:.1f}%")

    # Per-incident-category breakdown
    log.info(f"\n  Per-category MAE (predicted vs actual minutes):")
    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "within_5": within_5, "within_10": within_10,
        "y_pred": y_pred,
    }


def evaluate_by_category(y_true, y_pred, categories, split_name):
    """MAE broken down by incident category — reveals where model struggles."""
    results = {}
    for cat in sorted(set(categories)):
        mask     = categories == cat
        cat_mae  = mean_absolute_error(y_true[mask], y_pred[mask])
        cat_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        results[cat] = {"mae": cat_mae, "rmse": cat_rmse, "n": mask.sum()}
        log.info(f"    {cat:<25}  MAE: {cat_mae:>5.1f} min  RMSE: {cat_rmse:>6.1f} min  n={mask.sum():,}")
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_feature_importance(clf, feat_names):
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(20, len(feat_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices[:top_n]][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in indices[:top_n]][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (LightGBM)")
    ax.set_title("Top 20 Feature Importances — LightGBM Regressor (TTC Bus Delays)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "v3_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance chart saved: {path}")

    log.info(f"\nTop 20 features:")
    for rank, idx in enumerate(indices[:top_n], 1):
        log.info(f"  {rank:>2}. {feat_names[idx]:<40} {importances[idx]:.4f}")


def plot_predicted_vs_actual(y_true, y_pred, split_name):
    """Scatter of predicted vs actual — good diagnostic for regression."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.05, s=2, color="steelblue")
    lim = max(y_true.max(), y_pred.max())
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual delay (min)")
    ax.set_ylabel("Predicted delay (min)")
    ax.set_title(f"Predicted vs Actual — {split_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"v3_pred_vs_actual_{split_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Predicted vs actual chart saved: {path}")


def plot_residuals(y_true, y_pred, split_name):
    """Residual distribution — should be centred near zero."""
    residuals = y_pred - y_true
    fig, ax   = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Residual (predicted − actual, minutes)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {split_name}")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"v3_residuals_{split_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Residual distribution chart saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== TTC Bus Delay — LightGBM Regression (v3) ===")
    log.info(f"Loading data from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Time-based split — no shuffling
    train_df = df[df["Year"] <= 2023].copy()
    val_df   = df[df["Year"] == 2024].copy()
    test_df  = df[df["Year"] == 2025].copy()
    log.info(f"Split — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        log.info(f"{name} delay — mean: {split[TARGET].mean():.1f} min  median: {split[TARGET].median():.1f} min")

    # Feature engineering (route encoding built on train only)
    route_encoding, global_avg = build_route_encoding(train_df)

    train_df = engineer_features(train_df, route_encoding, global_avg)
    val_df   = engineer_features(val_df,   route_encoding, global_avg)
    test_df  = engineer_features(test_df,  route_encoding, global_avg)

    X_train, X_val, X_test, feat_names = encode_and_align(train_df, val_df, test_df)

    # Log-transform target — train in log space, evaluate in minute space
    y_train_raw = train_df[TARGET].values
    y_val_raw   = val_df[TARGET].values
    y_test_raw  = test_df[TARGET].values

    y_train_log = np.log1p(y_train_raw)

    clf = train_model(X_train, y_train_log)

    # Evaluate
    val_results  = evaluate(clf, X_val,  y_val_raw,  "Validate 2024")
    test_results = evaluate(clf, X_test, y_test_raw, "Test 2025")

    # Per-category breakdown on test set
    log.info(f"\n── Per-category breakdown (Test 2025) ──")
    evaluate_by_category(
        y_test_raw,
        test_results["y_pred"],
        test_df["Incident_Category"].values,
        "Test 2025"
    )

    # Plots
    plot_feature_importance(clf, feat_names)
    plot_predicted_vs_actual(y_val_raw,  val_results["y_pred"],  "Validate 2024")
    plot_predicted_vs_actual(y_test_raw, test_results["y_pred"], "Test 2025")
    plot_residuals(y_val_raw,  val_results["y_pred"],  "Validate 2024")
    plot_residuals(y_test_raw, test_results["y_pred"], "Test 2025")

    # Save bundle — everything needed for prediction in app_v2.py
    bundle = {
        "model":          clf,
        "route_encoding": route_encoding,
        "global_avg":     global_avg,
        "feature_columns": feat_names,
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"\nModel bundle saved: {BUNDLE_PATH}")
    log.info("=== Script complete. ===")


if __name__ == "__main__":
    main()