"""
Evaluation Module for Software Defect Prediction Models
Provides comprehensive evaluation metrics and visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """Comprehensive evaluation for defect prediction models"""

    def __init__(self):
        self.results = {}
        self.best_model = None

    def evaluate_single_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: np.ndarray, model_name: str) -> Dict:
        """
        Evaluate a single model

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        cm = confusion_matrix(y_true, y_pred)

        result = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba),
            'confusion_matrix': cm,
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1]),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }

        # Calculate additional metrics
        result['specificity'] = result['tn'] / (result['tn'] + result['fp']) if (result['tn'] + result['fp']) > 0 else 0
        result['npv'] = result['tn'] / (result['tn'] + result['fn']) if (result['tn'] + result['fn']) > 0 else 0

        self.results[model_name] = result
        return result

    def compare_models(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            results: Dictionary with results for each model

        Returns:
            DataFrame with comparison
        """
        comparison = []
        for model_name, metrics in results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc'],
                'Specificity': metrics['specificity']
            })

        return pd.DataFrame(comparison)

    def get_best_model(self, metric: str = 'f1_score') -> str:
        """
        Get the best model based on a specific metric

        Args:
            metric: Metric to use for comparison

        Returns:
            Name of the best model
        """
        if not self.results:
            return None

        best_name = max(self.results.keys(),
                        key=lambda x: self.results[x].get(metric, 0))
        self.best_model = best_name
        return best_name

    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> go.Figure:
        """
        Plot confusion matrix using Plotly

        Args:
            cm: Confusion matrix
            model_name: Name of the model

        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Defect', 'Predicted Defect'],
            y=['Actual No Defect', 'Actual Defect'],
            colorscale='Blues',
            text=cm,
            texttemplate='%d',
            textfont={"size": 20}
        ))

        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=500,
            height=400
        )

        return fig

    def plot_roc_curves(self, results: Dict[str, Dict]) -> go.Figure:
        """
        Plot ROC curves for all models

        Args:
            results: Dictionary with results for each model

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for model_name, metrics in results.items():
            # Calculate ROC curve (would need y_true and y_proba stored)
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='gray')
            ))

        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500
        )

        return fig

    def plot_metrics_comparison(self, results: Dict[str, Dict]) -> go.Figure:
        """
        Plot bar chart comparing metrics across models

        Args:
            results: Dictionary with results for each model

        Returns:
            Plotly figure
        """
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        fig = go.Figure()

        for metric in metrics:
            values = [results[m][metric] for m in models]
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            width=800,
            height=500
        )

        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        """
        Plot feature importance

        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to display

        Returns:
            Plotly figure
        """
        top_features = importance_df.head(top_n)

        fig = go.Figure(go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='steelblue'
        ))

        fig.update_layout(
            title='Top Feature Importance (Random Forest)',
            xaxis_title='Importance',
            yaxis_title='Feature',
            height=600,
            width=800
        )

        return fig

    def generate_risk_heatmap(self, probabilities: np.ndarray,
                            module_names: List[str],
                            threshold: float = 0.5) -> go.Figure:
        """
        Generate risk heatmap for modules

        Args:
            probabilities: Defect probabilities for each module
            module_names: Names of the modules
            threshold: Risk threshold

        Returns:
            Plotly figure
        """
        # Categorize risk levels
        risk_levels = []
        for prob in probabilities:
            if prob >= 0.7:
                risk_levels.append('High Risk')
            elif prob >= 0.4:
                risk_levels.append('Medium Risk')
            else:
                risk_levels.append('Low Risk')

        # Create DataFrame for visualization
        df = pd.DataFrame({
            'Module': module_names,
            'Defect Probability': probabilities,
            'Risk Level': risk_levels
        }).sort_values('Defect Probability', ascending=False)

        # Color mapping
        color_map = {
            'High Risk': 'red',
            'Medium Risk': 'orange',
            'Low Risk': 'green'
        }

        fig = go.Figure(data=go.Table(
            header=dict(
                values=['Module', 'Defect Probability', 'Risk Level'],
                fill_color='lightblue',
                align='left'
            ),
            cells=dict(
                values=[df['Module'], df['Defect Probability'].round(3), df['Risk Level']],
                fill_color=[[color_map[r] for r in df['Risk Level']]],
                align='left'
            )
        ))

        fig.update_layout(
            title='Module Risk Assessment',
            width=600,
            height=400 + len(df) * 30
        )

        return fig

    def plot_prediction_distribution(self, probabilities: Dict[str, np.ndarray]) -> go.Figure:
        """
        Plot distribution of prediction probabilities

        Args:
            probabilities: Dictionary with probabilities for each model

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for model_name, probas in probabilities.items():
            fig.add_trace(go.Histogram(
                x=probas,
                name=model_name,
                opacity=0.75,
                nbinsx=20
            ))

        fig.update_layout(
            title='Distribution of Defect Probabilities',
            xaxis_title='Defect Probability',
            yaxis_title='Count',
            barmode='overlay',
            width=800,
            height=500
        )

        return fig

    def generate_evaluation_report(self, results: Dict[str, Dict],
                                   dataset_info: Dict) -> str:
        """
        Generate text evaluation report

        Args:
            results: Evaluation results
            dataset_info: Dataset information

        Returns:
            Report as string
        """
        report = []
        report.append("="*60)
        report.append("SOFTWARE DEFECT PREDICTION - EVALUATION REPORT")
        report.append("="*60)

        report.append(f"\nDataset Information:")
        report.append(f"  - Total samples: {dataset_info.get('total_samples', 'N/A')}")
        report.append(f"  - Features: {dataset_info.get('num_features', 'N/A')}")
        report.append(f"  - Defect rate: {dataset_info.get('defect_rate', 'N/A')}")

        report.append("\n" + "-"*60)
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-"*60)

        comparison = self.compare_models(results)
        report.append(f"\n{comparison.to_string(index=False)}")

        report.append("\n" + "-"*60)
        report.append("BEST MODEL RECOMMENDATION")
        report.append("-"*60)

        best_model = self.get_best_model()
        if best_model:
            best_metrics = results[best_model]
            report.append(f"\nBest Model: {best_model}")
            report.append(f"  - F1-Score: {best_metrics['f1_score']:.4f}")
            report.append(f"  - ROC-AUC: {best_metrics['roc_auc']:.4f}")
            report.append(f"  - Recall: {best_metrics['recall']:.4f}")

        report.append("\n" + "="*60)

        return "\n".join(report)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate various performance metrics"""
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': cm.tolist()
    }
