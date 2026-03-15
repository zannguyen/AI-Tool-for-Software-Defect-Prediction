"""
Machine Learning Models for Software Defect Prediction
Implements Logistic Regression, Random Forest, and Neural Network models
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow/Keras for Neural Network
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    Sequential = None  # type: ignore
    print("TensorFlow not available. Neural Network will use sklearn MLPClassifier as fallback.")
    from sklearn.neural_network import MLPClassifier


class DefectPredictionModels:
    """Container class for all ML models"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.is_trained = False

    def _get_neural_network(self):
        """Create Neural Network model using TensorFlow/Keras"""
        if not TF_AVAILABLE:
            return None

        model = Sequential([
            Dense(64, activation='relu', input_shape=(None,)),
            BatchNormalization(),
            Dropout(0.3),

            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),

            Dense(16, activation='relu'),
            BatchNormalization(),

            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def _get_mlp_classifier(self) -> 'MLPClassifier':
        """Get MLPClassifier as fallback for Neural Network"""
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )

    def initialize_models(self) -> None:
        """Initialize all three models"""
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )

        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced'
        )

        # Neural Network
        if TF_AVAILABLE:
            self.models['neural_network'] = self._get_neural_network()
        else:
            self.models['neural_network'] = self._get_mlp_classifier()

        print("All models initialized successfully.")

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None,
                     y_val: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """
        Train all models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Dictionary with training history for each model
        """
        from sklearn.preprocessing import StandardScaler

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        training_history = {}

        # Train Logistic Regression
        print("Training Logistic Regression...")
        self.models['logistic_regression'].fit(X_train_scaled, y_train)
        training_history['logistic_regression'] = {
            'status': 'completed'
        }

        # Train Random Forest
        print("Training Random Forest...")
        self.models['random_forest'].fit(X_train_scaled, y_train)
        training_history['random_forest'] = {
            'status': 'completed'
        }

        # Train Neural Network
        print("Training Neural Network...")
        if TF_AVAILABLE and isinstance(self.models['neural_network'], Sequential):
            callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

            history = self.models['neural_network'].fit(
                X_train_scaled, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val) if X_val is not None else None,
                callbacks=callbacks,
                verbose=0
            )
            training_history['neural_network'] = {
                'history': history.history,
                'status': 'completed'
            }
        else:
            self.models['neural_network'].fit(X_train_scaled, y_train)
            training_history['neural_network'] = {
                'status': 'completed'
            }

        self.is_trained = True
        print("All models trained successfully!")

        return training_history

    def predict(self, X: np.ndarray, model_name: str = 'all') -> Dict[str, np.ndarray]:
        """
        Make predictions using trained models

        Args:
            X: Features to predict
            model_name: 'all' or specific model name

        Returns:
            Dictionary with predictions from each model
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models first.")

        X_scaled = self.scaler.transform(X)
        predictions = {}

        if model_name == 'all':
            for name, model in self.models.items():
                if TF_AVAILABLE and name == 'neural_network' and isinstance(model, Sequential):
                    pred = model.predict(X_scaled, verbose=0)
                    predictions[name] = (pred > 0.5).astype(int).flatten()
                    predictions[f'{name}_proba'] = pred.flatten()
                else:
                    predictions[name] = model.predict(X_scaled)
                    if hasattr(model, 'predict_proba'):
                        predictions[f'{name}_proba'] = model.predict_proba(X_scaled)[:, 1]
        else:
            model = self.models[model_name]
            if TF_AVAILABLE and model_name == 'neural_network' and isinstance(model, Sequential):
                pred = model.predict(X_scaled, verbose=0)
                predictions[model_name] = (pred > 0.5).astype(int).flatten()
                predictions[f'{model_name}_proba'] = pred.flatten()
            else:
                predictions[model_name] = model.predict(X_scaled)
                if hasattr(model, 'predict_proba'):
                    predictions[f'{model_name}_proba'] = model.predict_proba(X_scaled)[:, 1]

        return predictions

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get probability predictions for all models

        Args:
            X: Features to predict

        Returns:
            Dictionary with probability predictions
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models first.")

        X_scaled = self.scaler.transform(X)
        probabilities = {}

        for name, model in self.models.items():
            if TF_AVAILABLE and name == 'neural_network' and isinstance(model, Sequential):
                proba = model.predict(X_scaled, verbose=0).flatten()
            else:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[:, 1]
                else:
                    proba = model.predict(X_scaled)

            probabilities[name] = proba

        return probabilities

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Evaluate all models on test data

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics for each model
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train_models first.")

        X_scaled = self.scaler.transform(X_test)
        results = {}

        for name, model in self.models.items():
            if TF_AVAILABLE and name == 'neural_network' and isinstance(model, Sequential):
                y_pred = (model.predict(X_scaled, verbose=0) > 0.5).astype(int).flatten()
                y_proba = model.predict(X_scaled, verbose=0).flatten()
            else:
                y_pred = model.predict(X_scaled)
                y_proba = model.predict_proba(X_scaled)[:, 1]

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }

        return results

    def get_feature_importance(self, feature_names: list) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from Random Forest

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if 'random_forest' not in self.models:
            return {}

        rf_model = self.models['random_forest']
        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return {'random_forest': importance_df}

    def save_models(self, path: str) -> None:
        """Save all trained models to disk"""
        import os
        os.makedirs(path, exist_ok=True)

        # Save sklearn models
        joblib.dump(self.models['logistic_regression'],
                    f"{path}/logistic_model.pkl")
        joblib.dump(self.models['random_forest'],
                    f"{path}/rf_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")

        # Save Neural Network
        if TF_AVAILABLE and isinstance(self.models['neural_network'], Sequential):
            self.models['neural_network'].save(f"{path}/neural_network.h5")

        print(f"Models saved to {path}")

    def load_models(self, path: str) -> None:
        """Load trained models from disk"""
        self.models['logistic_regression'] = joblib.load(f"{path}/logistic_model.pkl")
        self.models['random_forest'] = joblib.load(f"{path}/rf_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")

        if TF_AVAILABLE:
            try:
                self.models['neural_network'] = tf.keras.models.load_model(
                    f"{path}/neural_network.h5"
                )
            except:
                self.models['neural_network'] = self._get_mlp_classifier()
                self.models['neural_network'].fit = lambda X, y: None  # Placeholder

        self.is_trained = True
        print(f"Models loaded from {path}")

    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Dict]:
        """
        Perform cross-validation for all models

        Args:
            X: Features
            y: Labels
            cv: Number of folds

        Returns:
            Dictionary with cross-validation results
        """
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        cv_results = {}

        # Logistic Regression
        lr_scores = cross_val_score(
            self.models['logistic_regression'], X_scaled, y, cv=cv, scoring='f1'
        )
        cv_results['logistic_regression'] = {
            'mean_f1': lr_scores.mean(),
            'std_f1': lr_scores.std()
        }

        # Random Forest
        rf_scores = cross_val_score(
            self.models['random_forest'], X_scaled, y, cv=cv, scoring='f1'
        )
        cv_results['random_forest'] = {
            'mean_f1': rf_scores.mean(),
            'std_f1': rf_scores.std()
        }

        return cv_results


def create_sample_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Create sample software metrics data for demonstration

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed

    Returns:
        DataFrame with sample data
    """
    np.random.seed(random_state)

    # Generate features
    loc = np.random.exponential(500, n_samples)  # Lines of Code
    complexity = np.random.exponential(10, n_samples)  # Cyclomatic complexity
    coupling = np.random.exponential(5, n_samples)  # Coupling
    code_churn = np.random.exponential(100, n_samples)  # Code churn
    decision_count = np.random.exponential(15, n_samples)
    essential_complexity = np.random.exponential(5, n_samples)

    # Generate target (defect) based on feature relationships
    defect_prob = (
        0.3 * (loc / 1000) +
        0.3 * (complexity / 50) +
        0.2 * (coupling / 20) +
        0.2 * (code_churn / 200)
    )
    defect_prob = np.clip(defect_prob, 0, 1)
    defects = (np.random.random(n_samples) < defect_prob).astype(int)

    df = pd.DataFrame({
        'LOC': loc,
        'CYCLOMATIC_COMPLEXITY': complexity,
        'COUPLING': coupling,
        'CODE_CHURN': code_churn,
        'DECISION_COUNT': decision_count,
        'ESSENTIAL_COMPLEXITY': essential_complexity,
        'LABEL': defects
    })

    return df


if __name__ == "__main__":
    # Test with sample data
    print("Creating sample data...")
    df = create_sample_data(500)

    print("\nDataset shape:", df.shape)
    print("\nClass distribution:")
    print(df['LABEL'].value_counts())

    # Split data
    X = df.drop('LABEL', axis=1)
    y = df['LABEL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train models
    model_container = DefectPredictionModels()
    model_container.initialize_models()
    model_container.train_models(X_train.values, y_train.values)

    # Evaluate
    results = model_container.evaluate_models(X_test.values, y_test.values)

    print("\n" + "="*50)
    print("Model Evaluation Results")
    print("="*50)

    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
