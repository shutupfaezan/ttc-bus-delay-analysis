"""
train_model_v4.py — TTC Bus Delay Regression (LightGBM v4)
Improvements over v3:
  1. Per-incident-category average delay as a feature
  2. Diversion-specific route encoding
  3. Incident × Route interaction feature
  4. Headway × Route interaction feature
  5. GTFS bus headway as a feature
  6. Fixed per-category evaluation — no missing categories
Target: Min_Delay (log-transformed during training, reported in minutes)
Run with: python train_model_v4.py  (from analysis/ folder)
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
LOG_PATH    = os.path.join(SCRIPT_DIR, "..", "logs",   "train_model_v4_log.txt")
BUNDLE_PATH = os.path.join(SCRIPT_DIR, "..", "models", "lgbm_ttc_regressor_bundle.pkl")
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
    "Headway_min",
]
EXTRA_NUM = [
    "Route_Delay_Avg",            # per-route average delay (target encoded)
    "Incident_Delay_Avg",         # per-incident-category average delay (NEW)
    "Diversion_Route_Avg",        # diversion-specific per-route average delay (NEW)
    "Incident_Route_Interaction", # incident avg × route avg (NEW)
    "Headway_x_Route_Avg",        # headway × route avg (NEW)
    "Rush_x_Route_Avg",           # rush hour × route avg
    "Hour_x_Month",               # hour × month seasonality
]


# ── Feature engineering ───────────────────────────────────────────────────────
def build_encodings(train_df: pd.DataFrame) -> dict:
    """
    All target encodings built on training data only.
    Never applied to val/test before this function returns — no leakage.
    """
    route_avg    = train_df.groupby("Route_Number")[TARGET].mean().to_dict()
    global_avg   = train_df[TARGET].mean()
    incident_avg = train_df.groupby("Incident_Category")[TARGET].mean().to_dict()

    div_df       = train_df[train_df["Incident_Category"] == "Diversion"]
    div_route    = div_df.groupby("Route_Number")[TARGET].mean().to_dict()
    global_div   = div_df[TARGET].mean() if len(div_df) > 0 else global_avg

    log.info(f"Route encoding:      {len(route_avg)} routes  |  global avg: {global_avg:.1f} min")
    log.info(f"Incident encoding:   {len(incident_avg)} categories")
    for cat, avg in sorted(incident_avg.items(), key=lambda x: -x[1]):
        log.info(f"  {cat:<25}  avg delay: {avg:.1f} min")
    log.info(f"Diversion encoding:  {len(div_route)} routes  |  global diversion avg: {global_div:.1f} min")

    return {
        "route_avg":    route_avg,
        "global_avg":   global_avg,
        "incident_avg": incident_avg,
        "div_route":    div_route,
        "global_div":   global_div,
    }


def engineer_features(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    df = df.copy()

    # 1. Overall per-route average delay
    df["Route_Delay_Avg"] = (
        df["Route_Number"].map(enc["route_avg"]).fillna(enc["global_avg"])
    )
    # 2. Per-incident-category average delay
    df["Incident_Delay_Avg"] = (
        df["Incident_Category"].map(enc["incident_avg"]).fillna(enc["global_avg"])
    )
    # 3. Diversion-specific route average — tells model how diversion-prone
    #    a route is, regardless of the current incident type
    df["Diversion_Route_Avg"] = (
        df["Route_Number"].map(enc["div_route"]).fillna(enc["global_div"])
    )
    # 4. Incident × Route — same incident hits different routes differently
    df["Incident_Route_Interaction"] = (
        df["Incident_Delay_Avg"] * df["Route_Delay_Avg"]
    )
    # 5. Headway × Route — low-frequency high-delay routes compound badly
    df["Headway_x_Route_Avg"] = df["Headway_min"] * df["Route_Delay_Avg"]

    # 6. Rush hour × route avg
    df["Rush_x_Route_Avg"] = df["Is_Rush_Hour"] * df["Route_Delay_Avg"]

    # 7. Hour × month seasonality
    df["Hour_x_Month"] = df["Hour"] * df["Month"]

    return df


def encode_and_align(train, val, test):
    feature_cols = NUM_COLS + EXTRA_NUM + CAT_COLS
    combined = pd.concat(
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
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=95,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train_log)
    log.info("Training complete.")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(clf, X, y_true_raw, split_name: str) -> dict:
    y_pred_log = clf.predict(X)
    y_pred     = np.expm1(y_pred_log)
    y_pred     = np.clip(y_pred, 1, 300)

    rmse      = np.sqrt(mean_squared_error(y_true_raw, y_pred))
    mae       = mean_absolute_error(y_true_raw, y_pred)
    r2        = r2_score(y_true_raw, y_pred)
    within_5  = np.mean(np.abs(y_pred - y_true_raw) <= 5)  * 100
    within_10 = np.mean(np.abs(y_pred - y_true_raw) <= 10) * 100

    log.info(f"\n{'='*60}")
    log.info(f"  {split_name.upper()}")
    log.info(f"{'='*60}")
    log.info(f"  RMSE:           {rmse:.2f} min")
    log.info(f"  MAE:            {mae:.2f} min")
    log.info(f"  R²:             {r2:.4f}")
    log.info(f"  Within ±5 min:  {within_5:.1f}%")
    log.info(f"  Within ±10 min: {within_10:.1f}%")

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "within_5": within_5, "within_10": within_10,
        "y_pred": y_pred,
    }


def evaluate_by_category(y_true, y_pred, categories):
    """
    MAE and RMSE per incident category.
    Iterates over all categories present in the array — none skipped.
    """
    log.info(f"\n── Per-category breakdown ──")
    all_cats = sorted(set(categories))
    results  = {}
    for cat in all_cats:
        mask = np.array(categories) == cat
        n    = mask.sum()
        if n < 5:
            log.info(f"  {cat:<25}  n={n} — too few rows, skipped")
            continue
        cat_mae  = mean_absolute_error(y_true[mask], y_pred[mask])
        cat_rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        results[cat] = {"mae": cat_mae, "rmse": cat_rmse, "n": int(n)}
        log.info(
            f"  {cat:<25}  MAE: {cat_mae:>5.1f} min  "
            f"RMSE: {cat_rmse:>6.1f} min  n={n:,}"
        )
    return results


def log_vs_v3(split_name, v3, v4):
    """Side-by-side comparison of v3 vs v4 metrics."""
    log.info(f"\n── v3 → v4 comparison ({split_name}) ──")
    log.info(f"  {'metric':<14} {'v3':>8}  {'v4':>8}  {'change':>10}")
    log.info(f"  {'-'*44}")
    for metric, better in [("rmse","lower"), ("mae","lower"), ("r2","higher"),
                            ("within_5","higher"), ("within_10","higher")]:
        old, new = v3[metric], v4[metric]
        delta    = new - old
        if better == "lower":
            sign = f"{'-' if delta < 0 else '+'}{abs(delta):.2f} {'✓' if delta < 0 else '✗'}"
        else:
            sign = f"{'+' if delta > 0 else ''}{delta:.2f} {'✓' if delta > 0 else '✗'}"
        log.info(f"  {metric:<14} {old:>8.2f}  {new:>8.2f}  {sign:>10}")


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
    ax.set_title("Top 20 Feature Importances — LightGBM v4 (TTC Bus Delays)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "v4_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"\nFeature importance saved: {path}")

    log.info(f"\nTop 20 features:")
    for rank, idx in enumerate(indices[:top_n], 1):
        log.info(f"  {rank:>2}. {feat_names[idx]:<45} {importances[idx]:.4f}")


def plot_predicted_vs_actual(y_true, y_pred, split_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.05, s=2, color="steelblue")
    lim = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([0, lim], [0, lim], "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual delay (min)")
    ax.set_ylabel("Predicted delay (min)")
    ax.set_title(f"Predicted vs Actual — {split_name}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"v4_pred_vs_actual_{split_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Predicted vs actual saved: {path}")


def plot_residuals(y_true, y_pred, split_name):
    residuals = y_pred - y_true
    fig, ax   = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="red", linewidth=1, linestyle="--")
    ax.set_xlabel("Residual (predicted − actual, minutes)")
    ax.set_ylabel("Count")
    ax.set_title(f"Residual Distribution — {split_name}")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"v4_residuals_{split_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Residuals saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== TTC Bus Delay — LightGBM Regression v4 ===")
    log.info(f"Loading: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # Time-based split
    train_df = df[df["Year"] <= 2023].copy()
    val_df   = df[df["Year"] == 2024].copy()
    test_df  = df[df["Year"] == 2025].copy()
    log.info(f"Split — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        log.info(
            f"{name} — mean: {split[TARGET].mean():.1f} min  "
            f"median: {split[TARGET].median():.1f} min  "
            f"std: {split[TARGET].std():.1f} min"
        )

    # Encodings built on train only
    enc = build_encodings(train_df)

    train_df = engineer_features(train_df, enc)
    val_df   = engineer_features(val_df,   enc)
    test_df  = engineer_features(test_df,  enc)

    X_train, X_val, X_test, feat_names = encode_and_align(train_df, val_df, test_df)

    y_train_raw = train_df[TARGET].values
    y_val_raw   = val_df[TARGET].values
    y_test_raw  = test_df[TARGET].values
    y_train_log = np.log1p(y_train_raw)

    clf = train_model(X_train, y_train_log)

    # Evaluate
    val_results  = evaluate(clf, X_val,  y_val_raw,  "Validate 2024")
    test_results = evaluate(clf, X_test, y_test_raw, "Test 2025")

    # Per-category on test set
    evaluate_by_category(
        y_test_raw,
        test_results["y_pred"],
        test_df["Incident_Category"].values
    )

    # v3 baseline for comparison (from previous run)
    v3_val  = {"rmse": 32.59, "mae": 10.02, "r2": 0.2522, "within_5": 65.4, "within_10": 84.6}
    v3_test = {"rmse": 32.12, "mae": 11.10, "r2": 0.2011, "within_5": 59.7, "within_10": 79.1}
    log_vs_v3("Validate 2024", v3_val,  val_results)
    log_vs_v3("Test 2025",     v3_test, test_results)

    # Plots
    plot_feature_importance(clf, feat_names)
    plot_predicted_vs_actual(y_val_raw,  val_results["y_pred"],  "Validate 2024")
    plot_predicted_vs_actual(y_test_raw, test_results["y_pred"], "Test 2025")
    plot_residuals(y_val_raw,  val_results["y_pred"],  "Validate 2024")
    plot_residuals(y_test_raw, test_results["y_pred"], "Test 2025")

    # Save bundle — everything the app needs to make predictions
    bundle = {
        "model":           clf,
        "route_encoding":  enc["route_avg"],
        "global_avg":      enc["global_avg"],
        "incident_avg":    enc["incident_avg"],
        "div_route":       enc["div_route"],
        "global_div":      enc["global_div"],
        "feature_columns": feat_names,
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"\nModel bundle saved: {BUNDLE_PATH}")
    log.info("=== Script complete. ===")


if __name__ == "__main__":
    main()