import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from models.tabular_predictor import NEOHazardPredictor


def run_error_analysis():
    print("--- Starting Error Analysis for NEO Hazard Predictor ---")
    csv_path = os.path.join(project_root, "dataset", "neo_hazard_dataset.csv")
    model_path = os.path.join(project_root, "models", "neo_rf_model.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found. Please train the model first.")
        return

    predictor = NEOHazardPredictor()
    predictor.load_model(model_path)
    
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=predictor.FEATURES + ['is_potentially_hazardous'])
    
    X = df[predictor.FEATURES]
    y_true = df['is_potentially_hazardous'].astype(int)
    
    # Get predictions and probabilities
    y_pred = predictor.model.predict(X)
    y_probs = predictor.model.predict_proba(X)[:, 1]
    
    df['prediction'] = y_pred
    df['probability'] = y_probs
    
    # Identify error categories
    false_positives = df[(y_true == 0) & (y_pred == 1)]
    false_negatives = df[(y_true == 1) & (y_pred == 0)]
    
    print(f"\nTotal Samples Evaluated: {len(df)}")
    print(f"True Negatives: {len(df[(y_true == 0) & (y_pred == 0)])}")
    print(f"True Positives: {len(df[(y_true == 1) & (y_pred == 1)])}")
    print(f"False Positives (Safe but flagged Hazard): {len(false_positives)}")
    print(f"False Negatives (Hazard but flagged Safe): {len(false_negatives)}  <-- CRITICAL ERRORS")
    
    if len(false_negatives) > 0:
        print("\n--- Analyzing False Negatives ---")
        print(false_negatives[predictor.FEATURES + ['probability']].describe())
        print("\nExample False Negative Explainability:")
        sample_fn = false_negatives.iloc[0].to_dict()
        _, _, explanation, _ = predictor.predict(sample_fn)
        # Handle Windows terminal char encoding for emojis
        print(explanation.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
        
    if len(false_positives) > 0:
        print("\n--- Analyzing False Positives ---")
        print(false_positives[predictor.FEATURES + ['probability']].describe())


if __name__ == "__main__":
    run_error_analysis()
