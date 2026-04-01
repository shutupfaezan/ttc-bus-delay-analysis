import os
import sys
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = r"C:\Users\faeza\OneDrive\Desktop\Open_data_TTC_files\data\processed\master_ttc_model_ready.csv"
LOG_PATH    = os.path.join(SCRIPT_DIR, "..", "logs",    "train_model_v2_log.txt")
BUNDLE_PATH = os.path.join(SCRIPT_DIR, "..", "models",  "lgbm_ttc_bundle.pkl")
FIG_DIR     = os.path.join(SCRIPT_DIR, "..", "outputs")

# ── Column config ─────────────────────────────────────────────────────────────
TARGET    = "Is_Severe"
REFERENCE = ["Min_Delay", "Year", "Service_Date"]
CAT_COLS  = ["Day", "Season", "Incident_Category"]
NUM_COLS  = [
    "Hour", "Month", "Is_Rush_Hour", "Is_Weekend",
    "Temp_C", "Visibility_km", "Wind_Spd_kmh", "Rel_Humidity_pct",
]

# ── Logging ───────────────────────────────────────────────────────────────────
for path in [LOG_PATH, BUNDLE_PATH, os.path.join(FIG_DIR, "_")]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ── Feature engineering ───────────────────────────────────────────────────────
def build_route_encoding(train_df: pd.DataFrame) -> tuple[dict, float]:
    """Compute per-route severe delay rate from training data only."""
    rate      = train_df.groupby("Route_Number")[TARGET].mean().to_dict()
    global_rt = train_df[TARGET].mean()
    log.info(f"Route encoding built for {len(rate)} unique routes. Global rate: {global_rt:.3f}")
    return rate, global_rt


def engineer_features(df: pd.DataFrame, route_encoding: dict, global_rate: float) -> pd.DataFrame:
    df = df.copy()

    # Target-encode Route_Number → historical severe delay rate per route
    df["Route_Severe_Rate"] = df["Route_Number"].map(route_encoding).fillna(global_rate)

    # Interaction features
    df["Rush_x_Route_Rate"] = df["Is_Rush_Hour"] * df["Route_Severe_Rate"]
    df["Hour_x_Month"]      = df["Hour"] * df["Month"]

    return df


def encode_and_align(train, val, test, extra_num_cols):
    feature_cols = NUM_COLS + extra_num_cols + CAT_COLS
    combined     = pd.concat(
        [train[feature_cols], val[feature_cols], test[feature_cols]],
        keys=["train", "val", "test"]
    )
    encoded = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)
    encoded.fillna(0, inplace=True)

    X_train    = encoded.xs("train").values
    X_val      = encoded.xs("val").values
    X_test     = encoded.xs("test").values
    feat_names = encoded.columns.tolist()

    log.info(f"Feature matrix shape (train): {X_train.shape}")
    return X_train, X_val, X_test, feat_names


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(X_train, y_train) -> lgb.LGBMClassifier:
    log.info("Training LightGBM classifier ...")
    clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train)
    log.info("Training complete.")
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(clf, X, y_true, split_name: str) -> np.ndarray:
    y_pred = clf.predict(X)
    report = classification_report(y_true, y_pred, target_names=["Not Severe", "Severe"])
    cm     = confusion_matrix(y_true, y_pred)

    log.info(f"\n{'='*60}\n  {split_name.upper()} SET RESULTS\n{'='*60}")
    log.info(f"\nClassification Report:\n{report}")
    log.info(f"Confusion Matrix:\n{cm}")

    print(f"\n{'='*60}\n  {split_name.upper()} SET RESULTS\n{'='*60}")
    print(report)
    print(f"Confusion Matrix:\n{cm}")
    return cm


def save_confusion_matrix(cm, split_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Not Severe", "Severe"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"v2_confusion_matrix_{split_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved: {path}")


def save_feature_importance(clf, feat_names: list):
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(20, len(feat_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices[:top_n]][::-1], color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in indices[:top_n]][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (LightGBM)")
    ax.set_title("Top 20 Feature Importances — LightGBM v2 (TTC Bus Delays)")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "v2_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance chart saved: {path}")

    log.info("\nTop 20 features by importance:")
    for rank, idx in enumerate(indices[:top_n], 1):
        log.info(f"  {rank:>2}. {feat_names[idx]:<40} {importances[idx]:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== TTC Bus Delay — LightGBM v2 Training Script ===")

    log.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    train_df = df[df["Year"] <= 2023]
    val_df   = df[df["Year"] == 2024]
    test_df  = df[df["Year"] == 2025]
    log.info(f"Split — train: {len(train_df):,}  val: {len(val_df):,}  test: {len(test_df):,}")

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vc = split[TARGET].value_counts(normalize=True)
        log.info(f"{name} class balance — Not Severe: {vc.get(0,0):.1%}  Severe: {vc.get(1,0):.1%}")

    # Feature engineering (encoding built on training data only)
    route_encoding, global_rate = build_route_encoding(train_df)

    train_df = engineer_features(train_df, route_encoding, global_rate)
    val_df   = engineer_features(val_df,   route_encoding, global_rate)
    test_df  = engineer_features(test_df,  route_encoding, global_rate)

    extra_num = ["Route_Severe_Rate", "Rush_x_Route_Rate", "Hour_x_Month"]
    X_train, X_val, X_test, feat_names = encode_and_align(train_df, val_df, test_df, extra_num)

    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    clf = train_model(X_train, y_train)

    cm_val  = evaluate(clf, X_val,  y_val,  "Validate 2024")
    cm_test = evaluate(clf, X_test, y_test, "Test 2025")

    save_confusion_matrix(cm_val,  "Validate 2024")
    save_confusion_matrix(cm_test, "Test 2025")
    save_feature_importance(clf, feat_names)

    # Save everything needed for prediction in one bundle
    bundle = {
        "model":          clf,
        "route_encoding": route_encoding,
        "global_rate":    global_rate,
        "feature_columns": feat_names,
    }
    with open(BUNDLE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Model bundle saved: {BUNDLE_PATH}")
    log.info("=== Script complete. ===")


if __name__ == "__main__":
    main()