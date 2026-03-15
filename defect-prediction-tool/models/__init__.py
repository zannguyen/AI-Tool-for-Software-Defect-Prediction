"""
Models Directory
Store trained models
"""

import os
import joblib
import pandas as pd
from datetime import datetime

class ModelManager:
    """Quan ly model - luu va tai model"""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(self, model_container, model_name='default'):
        """Luu model vao file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.models_dir, f'{model_name}_{timestamp}')

        os.makedirs(model_path, exist_ok=True)

        # Save models
        joblib.dump(model_container.models['logistic_regression'],
                   f"{model_path}/logistic_regression.pkl")
        joblib.dump(model_container.models['random_forest'],
                   f"{model_path}/random_forest.pkl")
        joblib.dump(model_container.scaler, f"{model_path}/scaler.pkl")

        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'feature_names': model_container.feature_names if hasattr(model_container, 'feature_names') else []
        }
        pd.DataFrame([metadata]).to_csv(f"{model_path}/metadata.csv", index=False)

        return model_path

    def load_model(self, model_path):
        """Tai model tu file"""
        model_container = joblib.load(f"{model_path}/random_forest.pkl")
        return model_container

    def list_models(self):
        """Lay danh sach model da luu"""
        models = []
        for item in os.listdir(self.models_dir):
            item_path = os.path.join(self.models_dir, item)
            if os.path.isdir(item_path):
                metadata_path = f"{item_path}/metadata.csv"
                if os.path.exists(metadata_path):
                    metadata = pd.read_csv(metadata_path)
                    models.append({
                        'name': item,
                        'path': item_path,
                        'timestamp': metadata.iloc[0]['timestamp']
                    })
        return models
