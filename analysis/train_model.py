import os
import sys
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = r"C:\Users\faeza\OneDrive\Desktop\Open_data_TTC_files\data\processed\master_ttc_model_ready.csv"
LOG_PATH     = os.path.join(SCRIPT_DIR, "..", "logs", "train_model_log.txt")
MODEL_PATH   = os.path.join(SCRIPT_DIR, "..", "models", "random_forest_ttc.pkl")
FIG_PATH     = os.path.join(SCRIPT_DIR, "..", "outputs", "feature_importance.png")

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(LOG_PATH),   exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(FIG_PATH),   exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Column config ─────────────────────────────────────────────────────────────
TARGET       = "Is_Severe"
REFERENCE    = ["Min_Delay", "Year", "Service_Date"]
CAT_COLS     = ["Day", "Season", "Incident_Category"]
NUM_COLS     = [
    "Route_Number", "Hour", "Month",
    "Is_Rush_Hour", "Is_Weekend",
    "Temp_C", "Visibility_km", "Wind_Spd_kmh", "Rel_Humidity_pct",
]


def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def split_by_year(df: pd.DataFrame):
    train = df[df["Year"] <= 2023]
    val   = df[df["Year"] == 2024]
    test  = df[df["Year"] == 2025]
    log.info(f"Split sizes — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    return train, val, test


def build_features(train, val, test):
    feature_cols = NUM_COLS + CAT_COLS

    # One-hot encode using training categories only
    combined = pd.concat([train[feature_cols], val[feature_cols], test[feature_cols]], keys=["train", "val", "test"])
    encoded  = pd.get_dummies(combined, columns=CAT_COLS, drop_first=False)
    encoded.fillna(0, inplace=True)

    X_train = encoded.xs("train").values
    X_val   = encoded.xs("val").values
    X_test  = encoded.xs("test").values
    feat_names = encoded.columns.tolist()

    log.info(f"Feature matrix shape (train): {X_train.shape}")
    log.info(f"Encoded feature names ({len(feat_names)}): {feat_names}")

    return X_train, X_val, X_test, feat_names


def train_model(X_train, y_train) -> RandomForestClassifier:
    log.info("Training Random Forest (n_estimators=300, class_weight='balanced') ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    log.info("Training complete.")
    return clf


def evaluate(clf, X, y_true, split_name: str):
    y_pred = clf.predict(X)

    report = classification_report(y_true, y_pred, target_names=["Not Severe", "Severe"])
    cm     = confusion_matrix(y_true, y_pred)

    log.info(f"\n{'='*60}")
    log.info(f"  {split_name.upper()} SET RESULTS")
    log.info(f"{'='*60}")
    log.info(f"\nClassification Report:\n{report}")
    log.info(f"Confusion Matrix:\n{cm}")

    # Print to console in a readable block
    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    return cm


def plot_confusion_matrix(cm, split_name: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Severe", "Severe"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    cm_path = FIG_PATH.replace("feature_importance", f"confusion_matrix_{split_name.lower()}")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Confusion matrix saved: {cm_path}")


def save_feature_importance(clf, feat_names):
    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1]
    top_n       = min(20, len(feat_names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        range(top_n),
        importances[indices[:top_n]][::-1],
        color="steelblue",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feat_names[i] for i in indices[:top_n]][::-1], fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Top 20 Feature Importances — Random Forest (TTC Bus Delays)")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Feature importance chart saved: {FIG_PATH}")

    log.info("\nTop 20 features by importance:")
    for rank, idx in enumerate(indices[:top_n], 1):
        log.info(f"  {rank:>2}. {feat_names[idx]:<40} {importances[idx]:.4f}")


def save_model(clf):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    log.info(f"Model saved: {MODEL_PATH}")


def main():
    log.info("=== TTC Bus Delay — Random Forest Training Script ===")

    df = load_data(DATA_PATH)

    # Validate expected columns
    required = NUM_COLS + CAT_COLS + [TARGET, "Year"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing columns in CSV: {missing}")
        sys.exit(1)

    train_df, val_df, test_df = split_by_year(df)

    # Log class balance per split
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        vc = split[TARGET].value_counts(normalize=True)
        log.info(f"{name} class balance — Not Severe: {vc.get(0,0):.1%}  Severe: {vc.get(1,0):.1%}")

    X_train, X_val, X_test, feat_names = build_features(train_df, val_df, test_df)
    y_train = train_df[TARGET].values
    y_val   = val_df[TARGET].values
    y_test  = test_df[TARGET].values

    clf = train_model(X_train, y_train)

    cm_val  = evaluate(clf, X_val,  y_val,  "Validate (2024)")
    cm_test = evaluate(clf, X_test, y_test, "Test (2025)")

    plot_confusion_matrix(cm_val,  "Validate 2024")
    plot_confusion_matrix(cm_test, "Test 2025")

    save_feature_importance(clf, feat_names)
    save_model(clf)

    log.info("=== Script complete. ===")


if __name__ == "__main__":
    main()