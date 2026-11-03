
import joblib
import pandas as pd
from pathlib import Path

# Locate project directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"

# Load trained model
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run python src/train.py first."
        )

    artifact = joblib.load(MODEL_PATH)

    return (
        artifact["model"],
        artifact["label_encoder"],
        artifact["features"],
        artifact.get("high_threshold", 0.50)
    )


def predict_risk(input_data):

    model, encoder, features, threshold = load_model()

    # Convert input into DataFrame
    input_df = pd.DataFrame(
        [input_data],
        columns=features
    )

    # Predict class probabilities
    probabilities = model.predict_proba(input_df)[0]

    classes = list(encoder.classes_)

    # Normal prediction
    predicted_index = probabilities.argmax()

    # Apply high-risk threshold
    high_index = classes.index("High")

    if probabilities[high_index] >= threshold:
        predicted_index = high_index

    predicted_label = classes[predicted_index]

    probability_results = {
        label: round(float(probabilities[i]) * 100, 2)
        for i, label in enumerate(classes)
    }

    return {
        "risk_level": predicted_label,
        "probabilities": probability_results,
        "high_risk_probability": round(
            float(probabilities[high_index]) * 100, 2
        ),
        "threshold": threshold
    }


if __name__ == "__main__":

    sample_region = {
        "dist_to_road_km": 2.5,
        "dist_to_settlement_km": 4.0,
        "dist_to_forest_edge_km": 1.5,
        "slope_deg": 12.0,
        "elevation_m": 500.0,
        "pop_density": 80.0,
        "agri_price_index": 110.0,
        "rainfall_mm": 1200.0,
        "fire_count_last_year": 3,
        "protected_area": 0,
        "tree_cover_pct": 65.0,
        "prior_loss_pct": 15.0
    }

    result = predict_risk(sample_region)

    print("\nDEFORESTATION RISK PREDICTION")
    print("--------------------------------")

    print("Predicted risk:", result["risk_level"])

    print("\nClass probabilities:")

    for category, probability in result["probabilities"].items():
        print(f"{category}: {probability}%")

    print(
        "\nHigh-risk threshold:",
        result["threshold"]
    )
