"""
Feature Engineering Module for Software Defect Prediction
Handles feature extraction and transformation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def calculate_cyclomatic_complexity(df: pd.DataFrame) -> pd.Series:
    """
    Calculate cyclomatic complexity from decision count

    Args:
        df: DataFrame with decision count

    Returns:
        Series with cyclomatic complexity
    """
    if 'DECISION_COUNT' in df.columns:
        return df['DECISION_COUNT'] + 1
    return pd.Series([0] * len(df))


def calculate_code_density(df: pd.DataFrame) -> pd.Series:
    """
    Calculate code density (executable LOC / total LOC)

    Args:
        df: DataFrame with LOC metrics

    Returns:
        Series with code density
    """
    if 'LOC_EXECUTABLE' in df.columns and 'LOC_TOTAL' in df.columns:
        return df['LOC_EXECUTABLE'] / (df['LOC_TOTAL'] + 1)
    return pd.Series([0] * len(df))


def calculate_comment_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculate comment ratio (comments / total LOC)

    Args:
        df: DataFrame with comment metrics

    Returns:
        Series with comment ratio
    """
    if 'LOC_COMMENTS' in df.columns and 'LOC_TOTAL' in df.columns:
        return df['LOC_COMMENTS'] / (df['LOC_TOTAL'] + 1)
    return pd.Series([0] * len(df))


def calculate_design_complexity_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculate design complexity ratio

    Args:
        df: DataFrame with complexity metrics

    Returns:
        Series with design complexity ratio
    """
    if 'DESIGN_COMPLEXITY' in df.columns and 'CYCLOMATIC_COMPLEXITY' in df.columns:
        return df['DESIGN_COMPLEXITY'] / (df['CYCLOMATIC_COMPLEXITY'] + 1)
    return pd.Series([0] * len(df))


def calculate_essential_complexity_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Calculate essential complexity ratio

    Args:
        df: DataFrame with complexity metrics

    Returns:
        Series with essential complexity ratio
    """
    if 'ESSENTIAL_COMPLEXITY' in df.columns and 'CYCLOMATIC_COMPLEXITY' in df.columns:
        return df['ESSENTIAL_COMPLEXITY'] / (df['CYCLOMATIC_COMPLEXITY'] + 1)
    return pd.Series([0] * len(df))


def create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features from raw metrics

    Args:
        df: DataFrame with raw metrics

    Returns:
        DataFrame with engineered features
    """
    df_features = df.copy()

    # Add calculated features
    df_features['CYCLOMATIC_COMPLEXITY'] = calculate_cyclomatic_complexity(df)
    df_features['CODE_DENSITY'] = calculate_code_density(df)
    df_features['COMMENT_RATIO'] = calculate_comment_ratio(df)
    df_features['DESIGN_COMPLEXITY_RATIO'] = calculate_design_complexity_ratio(df)
    df_features['ESSENTIAL_COMPLEXITY_RATIO'] = calculate_essential_complexity_ratio(df)

    return df_features


def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int = 10) -> List[str]:
    """
    Select top features based on correlation with target

    Args:
        X: Features DataFrame
        y: Target Series
        n_features: Number of top features to select

    Returns:
        List of top feature names
    """
    correlations = {}

    for col in X.columns:
        corr = X[col].corr(y)
        if not np.isnan(corr):
            correlations[col] = abs(corr)

    # Sort by correlation
    sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    return [f[0] for f in sorted_features[:n_features]]


def normalize_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features to 0-1 range

    Args:
        X: Features DataFrame

    Returns:
        Normalized DataFrame
    """
    X_norm = X.copy()

    for col in X.columns:
        min_val = X[col].min()
        max_val = X[col].max()

        if max_val > min_val:
            X_norm[col] = (X[col] - min_val) / (max_val - min_val)
        else:
            X_norm[col] = 0

    return X_norm


def get_feature_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistical summary of features

    Args:
        df: DataFrame with features

    Returns:
        Dictionary with feature statistics
    """
    stats = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'descriptive_stats': df.describe().to_dict()
    }

    return stats
