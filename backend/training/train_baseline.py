"""
Training script for the NASA NEO Hazard Classifier.
Trains a Random Forest model and saves it with evaluation metrics.
"""
import os
import sys
import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.tabular_predictor import NEOHazardPredictor


def main():
    csv_path = os.path.join(project_root, "dataset", "neo_hazard_dataset.csv")
    model_path = os.path.join(project_root, "models", "neo_rf_model.pkl")
    results_path = os.path.join(project_root, "experiments", "results.csv")

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}")
        print("Run: python scraping/scraping_neo.py")
        return

    # Load and inspect dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset: {len(df)} records")
    print(f"Hazardous: {df['is_potentially_hazardous'].sum()} ({df['is_potentially_hazardous'].mean():.1%})")
    print(f"Not Hazardous: {(~df['is_potentially_hazardous']).sum()}")
    print(f"\nFeature statistics:")
    print(df[NEOHazardPredictor.FEATURES].describe())

    # Train
    predictor = NEOHazardPredictor(n_estimators=200)
    X_train, X_test, y_train, y_test = predictor.prepare_data(csv_path)

    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train hazardous ratio: {y_train.mean():.1%}")
    print(f"Test hazardous ratio: {y_test.mean():.1%}")

    predictor.train(X_train, y_train)
    metrics = predictor.evaluate(X_test, y_test)

    # Save model
    predictor.save_model(model_path)

    # Save metrics
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    pd.DataFrame([{
        'model': 'RandomForest_v2',
        'n_estimators': 200,
        'accuracy': metrics['accuracy'],
        'dataset': 'neo_hazard_dataset.csv',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
    }]).to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    # Feature importances
    print("\nFeature Importances:")
    for feat, imp in sorted(predictor.feature_importances.items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.4f}")

    # Test prediction with example
    print("\n--- Example Prediction ---")
    example = {
        'absolute_magnitude_h': 20.5,
        'estimated_diameter_min_km': 0.2,
        'estimated_diameter_max_km': 0.5,
        'relative_velocity_kmh': 75000,
        'miss_distance_km': 3000000,
    }
    label, conf, explanation, _ = predictor.predict(example)
    # Use encode/decode to handle emoji on Windows terminals
    result = f"\n{label} (confidence: {conf:.1%})\n{explanation}"
    print(result.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))


if __name__ == "__main__":
    main()
