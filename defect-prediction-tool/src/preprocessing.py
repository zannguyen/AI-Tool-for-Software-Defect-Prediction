"""
Data Preprocessing Module for Software Defect Prediction
Handles loading, cleaning, and transforming code metrics data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the code metrics dataset from CSV file

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame containing the dataset
    """
    df = pd.read_csv(filepath)
    return df


def load_nasa_kc1() -> pd.DataFrame:
    """
    Load NASA KC1 dataset
    KC1 is a CM1 (NASA spacecraft instrument) dataset

    Returns:
        DataFrame with KC1 data
    """
    # NASA KC1 dataset columns (common software metrics)
    columns = [
        'UNIQUE_ID', 'KM1', 'LOC_BLANK', 'BRANCH_COUNT', 'LOC_CODE_AND_COMMENT',
        'LOC_COMMENTS', 'CYCLOMATIC_COMPLEXITY', 'CYCLOMATIC_DENSITY',
        'DECISION_COUNT', 'DECISION_DENSITY', 'DESIGN_COMPLEXITY',
        'DESIGN_DENSITY', 'EDGE_COUNT', 'ESSENTIAL_COMPLEXITY', 'ESSENTIAL_DENSITY',
        'LABEL', 'LOC_EXECUTABLE', 'LOC_TOTAL', 'NODE_COUNT', 'NUM_OPERANDS',
        'NUM_OPERATORS', 'NUM_UNIQUE_OPERANDS', 'NUM_UNIQUE_OPERATORS',
        'PATHOLOGICAL_COMPLEXITY', 'PERCENT_COMMENTS', 'LOC_BLANK_NORMALIZED'
    ]

    # For demonstration, we'll create sample data structure
    # In production, load actual NASA dataset
    return pd.DataFrame(columns=columns)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers

    Args:
        df: Raw DataFrame

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Remove rows with missing values
    df_clean = df_clean.dropna()

    # Remove duplicate rows
    df_clean = df_clean.drop_duplicates()

    return df_clean


def extract_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract features and target variable from the dataset

    Args:
        df: Cleaned DataFrame

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Define target column (usually 'defects' or 'label')
    target_col = 'LABEL' if 'LABEL' in df.columns else 'defects'

    # Get feature columns (exclude ID, target, and non-numeric columns)
    exclude_cols = ['UNIQUE_ID', 'LABEL', 'defects', 'module_name', 'file_name', 'file_path']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Only keep numeric columns
    X = df[feature_cols].select_dtypes(include=['number'])
    y = df[target_col]

    return X, y


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
    """
    Handle missing values in the dataset

    Args:
        df: DataFrame with potential missing values
        strategy: 'mean', 'median', 'drop'

    Returns:
        DataFrame with missing values handled
    """
    df_filled = df.copy()

    if strategy == 'mean':
        df_filled = df_filled.fillna(df_filled.mean())
    elif strategy == 'median':
        df_filled = df_filled.fillna(df_filled.median())
    elif strategy == 'drop':
        df_filled = df_filled.dropna()

    return df_filled


def remove_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers using z-score method

    Args:
        df: DataFrame
        columns: Columns to check for outliers
        threshold: Z-score threshold

    Returns:
        DataFrame without outliers
    """
    df_no_outliers = df.copy()

    for col in columns:
        if col in df.columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_no_outliers = df_no_outliers[z_scores < threshold]

    return df_no_outliers


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                random_state: int = 42) -> Tuple:
    """
    Split data into training and testing sets

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scale features using StandardScaler

    Args:
        X_train: Training features
        X_test: Test features

    Returns:
        Tuple of scaled (X_train, X_test)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def get_feature_importance(df: pd.DataFrame, feature_names: List[str]) -> Dict[str, float]:
    """
    Calculate feature importance based on correlation with target

    Args:
        df: DataFrame with features and target
        feature_names: List of feature column names

    Returns:
        Dictionary of feature importance scores
    """
    target_col = 'LABEL' if 'LABEL' in df.columns else 'defects'

    importance = {}
    for col in feature_names:
        if col in df.columns:
            correlation = df[col].corr(df[target_col])
            importance[col] = abs(correlation) if not np.isnan(correlation) else 0

    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Get summary statistics of the dataset

    Args:
        df: DataFrame

    Returns:
        Dictionary with summary statistics
    """
    return {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': df.describe().to_dict()
    }
