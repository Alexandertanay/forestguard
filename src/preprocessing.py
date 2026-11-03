from pathlib import Path
import pandas as pd

# Identifiers and fields that would leak the answer or represent a separate outcome
DROP_COLUMNS = ["plot_id", "risk_probability", "deforested_next_year"]
TARGET = "risk_level"


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the project CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}. Put deforestation_risk.csv in the dataset folder."
        )

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"The dataset must contain a '{TARGET}' column.")

    df[TARGET] = df[TARGET].astype(str).str.strip().str.title()
    valid_labels = {"Low", "Medium", "High"}
    unexpected = set(df[TARGET].dropna().unique()) - valid_labels
    if unexpected:
        raise ValueError(f"Unexpected risk labels: {sorted(unexpected)}")
    if df[TARGET].isna().any():
        raise ValueError("The target column contains missing values. Please inspect the CSV.")
    return df


def make_features_and_target(df: pd.DataFrame):
    """Return model features and target, excluding IDs and leakage-prone columns."""
    excluded = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=excluded + [TARGET], errors="ignore")
    y = df[TARGET]

    # Keep numeric predictors only; the supplied dataset uses numeric predictors.
    X = X.select_dtypes(include="number")
    if X.empty:
        raise ValueError("No numeric predictor columns were found.")
    return X, y