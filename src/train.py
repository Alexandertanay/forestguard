
import joblib
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_sample_weight


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "dataset" / "deforestation_risk.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "best_model.joblib"


# ==========================================
# FEATURES AND TARGET
# ==========================================

FEATURES = [
    "dist_to_road_km",
    "dist_to_settlement_km",
    "dist_to_forest_edge_km",
    "slope_deg",
    "elevation_m",
    "pop_density",
    "agri_price_index",
    "rainfall_mm",
    "fire_count_last_year",
    "protected_area",
    "tree_cover_pct",
    "prior_loss_pct"
]

TARGET = "risk_level"


# ==========================================
# MODEL CREATION
# ==========================================

def create_models():

    models = {}

    # Logistic Regression
    models["Logistic Regression"] = (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]),
        False
    )

    # Decision Tree
    models["Decision Tree"] = (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=6,
                min_samples_leaf=3,
                random_state=42
            ))
        ]),
        False
    )

    # Random Forest
    models["Random Forest"] = (
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ]),
        False
    )

    # XGBoost
    try:

        from xgboost import XGBClassifier

        models["XGBoost"] = (
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    num_class=3,
                    eval_metric="mlogloss",
                    random_state=42,
                    n_jobs=-1
                ))
            ]),
            True
        )

    except ImportError:

        print("WARNING: XGBoost is not installed.")
        print("Run: python -m pip install xgboost")

    return models


# ==========================================
# TRAIN MODEL WITH CLASS WEIGHTS
# ==========================================

def fit_model(model, X, y, use_sample_weights):

    if use_sample_weights:

        weights = compute_sample_weight(
            class_weight="balanced",
            y=y
        )

        model.fit(
            X,
            y,
            classifier__sample_weight=weights
        )

    else:

        model.fit(X, y)

    return model


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================

def train():

    print("\n======================================")
    print("       FORESTGUARD MODEL TRAINING")
    print("======================================")

    # --------------------------------------
    # LOAD DATASET
    # --------------------------------------

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\n1. DATASET INFORMATION")

    print("Dataset shape:", df.shape)

    print("Missing values:", df.isnull().sum().sum())

    print("Duplicate records:", df.duplicated().sum())

    # --------------------------------------
    # VALIDATE COLUMNS
    # --------------------------------------

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------
    # PREPARE TARGET
    # --------------------------------------

    df[TARGET] = df[TARGET].astype("string").str.strip().str.title()

    df = df.dropna(subset=[TARGET]).copy()

    # --------------------------------------
    # PREPARE INPUT FEATURES
    # --------------------------------------

    X = df[FEATURES].copy()

    for column in FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    y_text = df[TARGET]

    if X.isnull().all().any():

        raise ValueError(
            "At least one input feature contains no valid numeric values."
        )

    # --------------------------------------
    # ENCODE TARGET
    # --------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(y_text)

    print("\n2. CLASS DISTRIBUTION")

    print(y_text.value_counts())

    print("\nEncoded classes:", encoder.classes_)

    if len(encoder.classes_) != 3:

        raise ValueError(
            "Expected exactly three classes: High, Low and Medium."
        )

    # --------------------------------------
    # EXPLAIN LEAKAGE PREVENTION
    # --------------------------------------

    print("\n3. DATA LEAKAGE PREVENTION")

    print("Excluded: plot_id")
    print("Excluded: risk_probability")
    print("Excluded: deforested_next_year")

    print("\nSelected input features:")

    for feature in FEATURES:
        print("-", feature)

    print("\nTarget:", TARGET)

    # --------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("\n4. TRAIN TEST SPLIT")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print(
        "Training class counts:",
        pd.Series(
            encoder.inverse_transform(y_train)
        ).value_counts().to_dict()
    )

    print(
        "Testing class counts:",
        pd.Series(
            encoder.inverse_transform(y_test)
        ).value_counts().to_dict()
    )

    # --------------------------------------
    # STRATIFIED CROSS VALIDATION
    # --------------------------------------

    print("\n5. STRATIFIED CROSS VALIDATION")

    models = create_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    cv_results = {}

    for model_name, (model, use_weights) in models.items():

        print(f"\nEvaluating {model_name}...")

        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_train, y_train), start=1
        ):

            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            y_fold_train = y_train[train_idx]
            y_fold_val = y_train[val_idx]

            try:

                fit_model(
                    model,
                    X_fold_train,
                    y_fold_train,
                    use_weights
                )

                predictions = model.predict(X_fold_val)

                score = f1_score(
                    y_fold_val,
                    predictions,
                    average="macro",
                    zero_division=0
                )

                fold_scores.append(score)

                print(f"Fold {fold}: Macro F1 = {score:.4f}")

            except Exception as error:

                print(
                    f"ERROR in {model_name}, fold {fold}: {error}"
                )

                fold_scores = []
                break

        if fold_scores:

            cv_results[model_name] = {
                "Mean CV Macro F1": np.mean(fold_scores),
                "CV Standard Deviation": np.std(fold_scores)
            }

            print(
                "Mean CV Macro F1:",
                round(np.mean(fold_scores), 4)
            )

    if not cv_results:

        raise RuntimeError("All model training attempts failed.")

    # --------------------------------------
    # MODEL COMPARISON
    # --------------------------------------

    print("\n6. MODEL COMPARISON")

    comparison_df = pd.DataFrame(cv_results).T

    comparison_df = comparison_df.sort_values(
        "Mean CV Macro F1",
        ascending=False
    )

    print(comparison_df.round(4))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(
        MODEL_DIR / "model_comparison.csv"
    )

    # --------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------

    best_model_name = comparison_df.index[0]

    best_model, use_weights = models[best_model_name]

    print("\nBEST MODEL:", best_model_name)

    # --------------------------------------
    # FINAL TRAINING
    # --------------------------------------

    print("\n7. TRAINING FINAL MODEL")

    fit_model(
        best_model,
        X_train,
        y_train,
        use_weights
    )

    # --------------------------------------
    # TEST EVALUATION
    # --------------------------------------

    print("\n8. FINAL TEST EVALUATION")

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_pred
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

    print("\nCLASSIFICATION REPORT")

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0
        )
    )

    pd.DataFrame(report).T.to_csv(
        MODEL_DIR / "classification_report.csv"
    )

    # --------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(len(encoder.classes_))
    )

    cm_df = pd.DataFrame(
        cm,
        index=encoder.classes_,
        columns=encoder.classes_
    )

    print("\nCONFUSION MATRIX")
    print(cm_df)

    cm_df.to_csv(
        MODEL_DIR / "confusion_matrix.csv"
    )

    # --------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------

    classifier = best_model.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):

        importance = classifier.feature_importances_

    elif hasattr(classifier, "coef_"):

        importance = np.abs(classifier.coef_).mean(axis=0)

    else:

        importance = np.zeros(len(FEATURES))

    if np.sum(importance) > 0:

        importance = importance / np.sum(importance)

    # Preserve compatibility with existing dashboard
    best_model.feature_importances_ = importance

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": importance
    }).sort_values(
        "Importance",
        ascending=False
    )

    print("\nFEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        MODEL_DIR / "feature_importance.csv",
        index=False
    )

    # --------------------------------------
    # SAVE MODEL
    # --------------------------------------

    artifact = {
        "model": best_model,
        "label_encoder": encoder,
        "features": FEATURES,
        "high_threshold": 0.50,
        "best_model_name": best_model_name,
        "cv_results": comparison_df.to_dict()
    }

    joblib.dump(
        artifact,
        MODEL_PATH
    )

    # --------------------------------------
    # FINAL SUMMARY
    # --------------------------------------

    print("\n======================================")
    print("       TRAINING COMPLETED")
    print("======================================")

    print("Selected model:", best_model_name)

    print("Test Accuracy:", round(accuracy, 4))
    print("Test Macro F1:", round(macro_f1, 4))

    print("\nModel saved at:")
    print(MODEL_PATH)

    print("\nGenerated files:")
    print("- models/best_model.joblib")
    print("- models/model_comparison.csv")
    print("- models/classification_report.csv")
    print("- models/confusion_matrix.csv")
    print("- models/feature_importance.csv")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    train()

