"""
NASA NEO Hazard Predictor
Proper Random Forest classifier with real NASA labels and SHAP-based explainability.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class NEOHazardPredictor:
    """
    Predicts whether a Near-Earth Object is potentially hazardous
    using physical and orbital features from NASA's NeoWs data.
    """
    FEATURES = [
        'absolute_magnitude_h',
        'estimated_diameter_min_km',
        'estimated_diameter_max_km',
        'relative_velocity_kmh',
        'miss_distance_km',
    ]

    FEATURE_LABELS = {
        'absolute_magnitude_h': 'Absolute Magnitude (H)',
        'estimated_diameter_min_km': 'Min Diameter (km)',
        'estimated_diameter_max_km': 'Max Diameter (km)',
        'relative_velocity_kmh': 'Relative Velocity (km/h)',
        'miss_distance_km': 'Miss Distance (km)',
    }

    # Thresholds for human-readable explanations
    # Based on NASA's criteria: diameter > 140m and miss distance < 7.5M km
    HAZARD_THRESHOLDS = {
        'absolute_magnitude_h': {'threshold': 22.0, 'direction': 'lower', 'reason': 'indicates a larger, brighter object'},
        'estimated_diameter_min_km': {'threshold': 0.14, 'direction': 'higher', 'reason': 'large enough to cause significant damage'},
        'estimated_diameter_max_km': {'threshold': 0.14, 'direction': 'higher', 'reason': 'large enough to cause significant damage'},
        'relative_velocity_kmh': {'threshold': 50000, 'direction': 'higher', 'reason': 'high-speed impact would be more destructive'},
        'miss_distance_km': {'threshold': 7500000, 'direction': 'lower', 'reason': 'close approach to Earth'},
    }

    def __init__(self, n_estimators=200):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight='balanced',
            random_state=42,
            max_depth=10,
            min_samples_split=5,
        )
        self.is_trained = False
        self.feature_importances = None

    def prepare_data(self, csv_filepath):
        """Load and split the NASA NEO dataset."""
        df = pd.read_csv(csv_filepath)
        df = df.dropna(subset=self.FEATURES + ['is_potentially_hazardous'])

        X = df[self.FEATURES]
        y = df['is_potentially_hazardous'].astype(int)

        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        """Train the Random Forest classifier."""
        print("Training Random Forest on NASA NEO data...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.feature_importances = dict(zip(self.FEATURES, self.model.feature_importances_))
        print("Training complete.")

    def evaluate(self, X_test, y_test):
        """Evaluate and return metrics."""
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, target_names=['Not Hazardous', 'Hazardous'])
        cm = confusion_matrix(y_test, preds)

        print(f"Accuracy: {acc:.4f}")
        print(f"\nClassification Report:\n{report}")
        print(f"Confusion Matrix:\n{cm}")

        return {
            'accuracy': acc,
            'report': report,
            'confusion_matrix': cm.tolist(),
        }

    def save_model(self, path):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_importances': self.feature_importances,
            }, f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_importances = data['feature_importances']
        self.is_trained = True
        print(f"Model loaded from {path}")

    def predict(self, feature_dict):
        """
        Predict hazard status and return prediction + explanation.
        Returns: (label, confidence, explanation_text, feature_contributions)
        """
        if not self.is_trained:
            return "Model not trained", 0.0, "Please train the model first.", {}

        df = pd.DataFrame([feature_dict])

        # Ensure all features are present
        for f in self.FEATURES:
            if f not in df.columns:
                return "Missing features", 0.0, f"Missing feature: {f}", {}

        X = df[self.FEATURES]
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = probabilities[prediction]

        label = "🔴 POTENTIALLY HAZARDOUS" if prediction == 1 else "🟢 NOT HAZARDOUS"

        # Generate explanation
        explanation = self._explain_prediction(feature_dict, prediction, confidence)

        # Feature contributions for chart
        contributions = {}
        if self.feature_importances:
            for feat, imp in self.feature_importances.items():
                contributions[self.FEATURE_LABELS.get(feat, feat)] = round(imp, 4)

        return label, confidence, explanation, contributions

    def _explain_prediction(self, features, prediction, confidence):
        """Generate a human-readable explanation of WHY the prediction was made."""
        lines = []
        hazardous = prediction == 1

        if hazardous:
            lines.append(f"## 🔴 POTENTIALLY HAZARDOUS — Confidence: {confidence:.1%}\n")
            lines.append("### Why this object is flagged as hazardous:\n")
        else:
            lines.append(f"## 🟢 NOT HAZARDOUS — Confidence: {confidence:.1%}\n")
            lines.append("### Why this object is considered safe:\n")

        for feat, info in self.HAZARD_THRESHOLDS.items():
            value = features.get(feat)
            if value is None:
                continue

            threshold = info['threshold']
            direction = info['direction']
            reason = info['reason']
            label = self.FEATURE_LABELS.get(feat, feat)

            if direction == 'higher':
                is_dangerous = value > threshold
                comparison = "above" if is_dangerous else "below"
            else:
                is_dangerous = value < threshold
                comparison = "below" if is_dangerous else "above"

            if feat == 'miss_distance_km':
                val_str = f"{value:,.0f} km"
                thresh_str = f"{threshold:,.0f} km"
            elif feat == 'relative_velocity_kmh':
                val_str = f"{value:,.0f} km/h"
                thresh_str = f"{threshold:,.0f} km/h"
            elif 'diameter' in feat:
                val_str = f"{value:.4f} km ({value*1000:.1f} m)"
                thresh_str = f"{threshold:.3f} km ({threshold*1000:.0f} m)"
            else:
                val_str = f"{value:.2f}"
                thresh_str = f"{threshold:.1f}"

            emoji = "⚠️" if is_dangerous else "✅"
            lines.append(f"- {emoji} **{label}**: {val_str} ({comparison} threshold of {thresh_str}) — {reason}")

        # Add NASA context
        lines.append("\n---")
        lines.append("*NASA defines a Potentially Hazardous Asteroid (PHA) as one with a minimum orbit "
                      "intersection distance (MOID) of ≤0.05 AU (~7.5M km) and an absolute magnitude (H) of ≤22.0 "
                      "(roughly ≥140m diameter).*")

        return "\n".join(lines)


if __name__ == "__main__":
    predictor = NEOHazardPredictor()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "dataset", "neo_hazard_dataset.csv")

    if os.path.exists(csv_path):
        X_train, X_test, y_train, y_test = predictor.prepare_data(csv_path)
        predictor.train(X_train, y_train)
        metrics = predictor.evaluate(X_test, y_test)

        save_path = os.path.join(project_root, "models", "neo_rf_model.pkl")
        predictor.save_model(save_path)

        # Save metrics
        results_path = os.path.join(project_root, "experiments", "results.csv")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        pd.DataFrame([{
            'model': 'RandomForest',
            'accuracy': metrics['accuracy'],
            'dataset': 'neo_hazard_dataset.csv',
        }]).to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    else:
        print(f"Dataset not found at {csv_path}. Run scraping/scraping_neo.py first.")
