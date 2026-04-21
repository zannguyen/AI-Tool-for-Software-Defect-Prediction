"""
Preprocessing Pipeline for Software Defect Prediction (SDP)
============================================================
Targets: NASA PROMISE datasets — KC1, KC2, kc1_kc2.csv (ARFF + CSV)

Pipeline stages
───────────────
  1. Data loading     – CSV and ARFF formats, auto-detect label column
  2. Data cleaning    – duplicates, integrity checks (LOC > 0),
                        missing values, outlier clipping (IQR-based)
  3. Feature engineering
                      – log1p transforms for heavy-tailed metrics
                      – interaction / ratio features
                      – feature selection via mutual-information + RF importance
                        (inspired by filter-wrapper hybrid / DAOAFS principle)
  4. Imbalance handling
                      – SMOTE or ADASYN (oversampling, preferred for recall↑)
                      – class_weight fallback when imbalanced-learn unavailable
  5. Train / test split with StratifiedKFold support

Design decisions
────────────────
• Outlier REMOVAL is replaced by IQR-based CLIPPING for SDP.
  Reason: extreme metric values (e.g. huge Halstead Volume) ARE informative
  in defect prediction — removing rows loses minority-class samples.
• Log1p transform applied BEFORE scaling because most SDP metrics follow
  a power-law distribution (validated in Menzies et al. 2004).
• SMOTE preferred over random-oversampling: interpolates minority samples →
  improves model generalisation vs. pure duplication.
• Feature selection uses a two-stage filter-wrapper:
    stage-1  mutual-information (univariate filter, fast)
    stage-2  Random-Forest importance (wrapper, captures interactions)
  Keeps the union of top-k from each stage.
• StandardScaler is fitted ONLY on X_train and applied to X_test to prevent
  data leakage.
"""

from __future__ import annotations

import io
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── optional: imbalanced-learn ────────────────────────────────────────────────
try:
    from imblearn.over_sampling import ADASYN, SMOTE

    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# ── constants ─────────────────────────────────────────────────────────────────
# Columns that must be excluded from feature matrix (identifiers / labels)
_EXCLUDE_COLS = {
    "unique_id", "id", "module", "module_name", "file_name", "file_path",
    "label", "defects",
}

# Columns that represent line-counts and must be > 0 (integrity check)
_LOC_COLS = {"loc", "locode", "lOCode", "loc_total", "loc_executable"}

# Metrics known to require log1p (heavy right-skewed in KC1/KC2)
_LOG_COLS = {
    "loc", "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank",
    "total_Op", "total_Opnd", "uniq_Op", "uniq_Opnd",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_arff(filepath: str) -> pd.DataFrame:
    """
    Parse an ARFF file into a DataFrame.

    Why custom parser?
    scipy.io.arff.loadarff encodes string attributes as bytes and handles
    the {true, false} label attribute poorly. This lightweight parser is
    sufficient for the numeric-only NASA KC1/KC2 files.
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8", errors="replace")

    # ── extract attribute names ───────────────────────────────────────────
    attribute_re = re.compile(r"@attribute\s+['\"]?(\S+?)['\"]?\s+(.+)", re.IGNORECASE)
    attr_names: List[str] = []
    attr_types: List[str] = []
    for m in attribute_re.finditer(text):
        attr_names.append(m.group(1).strip())
        attr_types.append(m.group(2).strip().lower())

    # ── find @data section ────────────────────────────────────────────────
    data_start = re.search(r"^@data", text, re.IGNORECASE | re.MULTILINE)
    if data_start is None:
        raise ValueError(f"No @data section found in {filepath}")
    data_text = text[data_start.end():]

    # ── parse rows ────────────────────────────────────────────────────────
    rows = []
    for line in data_text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        rows.append(line.split(","))

    df = pd.DataFrame(rows, columns=attr_names)

    # ── type coercion ─────────────────────────────────────────────────────
    for col, typ in zip(attr_names, attr_types):
        if "{" in typ:
            # Categorical / nominal — convert {true/false} → 1/0
            df[col] = (
                df[col]
                .str.strip()
                .str.lower()
                .map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})
            )
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file. Handles both standard CSV and the kc1_kc2.csv format."""
    df = pd.read_csv(filepath)
    return df


def load_data(filepath: str) -> pd.DataFrame:
    """
    Auto-detect format (ARFF / CSV) and load dataset.

    Post-load normalisation
    ───────────────────────
    • Rename label column → 'LABEL' (int 0/1)
    • Lowercase all column names for uniform access
    """
    fp = filepath.strip().lower()
    if fp.endswith(".arff"):
        df = load_arff(filepath)
    else:
        df = load_csv(filepath)

    df = _normalise_label(df)
    return df


def _normalise_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single 'LABEL' column with int values 0/1."""
    df = df.copy()
    candidates = [c for c in df.columns if c.lower() in {"label", "defects"}]
    if not candidates:
        raise ValueError(
            "Dataset must contain a 'LABEL' or 'defects' column. "
            f"Columns found: {list(df.columns)}"
        )
    lbl_col = candidates[0]
    if lbl_col != "LABEL":
        df = df.rename(columns={lbl_col: "LABEL"})

    # Map text values if needed
    df["LABEL"] = (
        df["LABEL"]
        .map(lambda x: 1 if str(x).strip().lower() in {"1", "true", "yes"} else 0)
        .astype(int)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Stateless collection of cleaning steps.

    Steps applied (in order)
    ──────────────────────────
    1. Drop exact duplicate rows
    2. Integrity check: LOC-related metrics must be > 0
    3. Fill missing values with column median (robust to skew)
    4. IQR-based CLIPPING (not removal) of outliers
    5. Drop zero-variance columns (carry no information)

    Why IQR clipping instead of z-score removal?
    ─────────────────────────────────────────────
    NASA SDP metrics are extremely skewed (Halstead volume can be millions).
    Z-score removal would eliminate valid defective modules with huge metrics,
    hurting recall. IQR clipping preserves the relative ordering while
    bounding extreme values.

    Why median imputation?
    ──────────────────────
    Mean is sensitive to the heavy tails of SDP metrics. Median is a more
    robust central estimate for right-skewed distributions.
    """

    def __init__(self, iqr_multiplier: float = 3.0):
        self.iqr_multiplier = iqr_multiplier
        self._clip_bounds: Dict[str, Tuple[float, float]] = {}
        self._median_values: Dict[str, float] = {}
        self._zero_var_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit cleaner on df and return cleaned copy."""
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._integrity_checks(df)
        df = self._fill_missing(df, fit=True)
        df = self._clip_outliers(df, fit=True)
        df = self._drop_zero_variance(df, fit=True)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted parameters (for test split)."""
        df = df.copy()
        df = self._fill_missing(df, fit=False)
        df = self._clip_outliers(df, fit=False)
        df = df.drop(columns=[c for c in self._zero_var_cols if c in df.columns])
        return df

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates()
        removed = n_before - len(df)
        if removed:
            print(f"  [clean] Removed {removed} duplicate rows.")
        return df

    @staticmethod
    def _integrity_checks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows where any LOC-related metric is ≤ 0.
        Reason: a module with 0 lines of code is unanalysable and likely
        a data entry error in the NASA MDP dataset.
        """
        loc_present = [c for c in _LOC_COLS if c in df.columns]
        mask = pd.Series(True, index=df.index)
        for c in loc_present:
            mask &= df[c] > 0
        removed = (~mask).sum()
        if removed:
            print(f"  [clean] Removed {removed} rows failing LOC > 0 check.")
        return df[mask].reset_index(drop=True)

    def _fill_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num_cols = df.select_dtypes(include="number").columns
        if fit:
            self._median_values = df[num_cols].median().to_dict()
        for col in num_cols:
            if col in self._median_values:
                df[col] = df[col].fillna(self._median_values[col])
        return df

    def _clip_outliers(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c != "LABEL"
        ]
        if fit:
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lo = q1 - self.iqr_multiplier * iqr
                hi = q3 + self.iqr_multiplier * iqr
                self._clip_bounds[col] = (lo, hi)

        for col in num_cols:
            if col in self._clip_bounds:
                lo, hi = self._clip_bounds[col]
                df[col] = df[col].clip(lower=lo, upper=hi)
        return df

    def _drop_zero_variance(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num_cols = [c for c in df.select_dtypes(include="number").columns if c != "LABEL"]
        if fit:
            self._zero_var_cols = [c for c in num_cols if df[c].nunique() <= 1]
            if self._zero_var_cols:
                print(f"  [clean] Dropping zero-variance columns: {self._zero_var_cols}")
        df = df.drop(columns=[c for c in self._zero_var_cols if c in df.columns])
        return df


def clean_data(df: pd.DataFrame, iqr_multiplier: float = 3.0) -> pd.DataFrame:
    """Convenience wrapper — returns cleaned DataFrame using DataCleaner."""
    cleaner = DataCleaner(iqr_multiplier=iqr_multiplier)
    return cleaner.fit_transform(df)


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Adds log-transformed and interaction features, then applies
    a two-stage filter-wrapper feature selection.

    Stage 1 — Mutual Information (filter):
        Fast, model-agnostic. Captures non-linear dependencies that
        Pearson correlation misses (important for skewed SDP data).

    Stage 2 — Random Forest Importance (wrapper):
        Captures feature interactions. RF is the go-to for SDP because
        it handles imbalance well with class_weight='balanced'.

    Final set = union(top-k_mi, top-k_rf) — union keeps complementary
    information from both perspectives (DAOAFS principle).
    """

    # Pairs of existing metrics to multiply → interaction features
    _INTERACTION_PAIRS = [
        ("v(g)", "loc"),       # complexity × size
        ("ev(g)", "v(g)"),     # essential × cyclomatic
        ("iv(g)", "v(g)"),     # design × cyclomatic
        ("d", "v"),            # halstead difficulty × volume
        ("branchCount", "loc"),
    ]

    # Ratios that encode normalised complexity
    _RATIO_DEFS = [
        ("v(g)_per_loc",     "v(g)",     "loc"),
        ("ev(g)_per_vg",     "ev(g)",    "v(g)"),
        ("iv(g)_per_vg",     "iv(g)",    "v(g)"),
        ("comment_density",  "lOComment", "loc"),
        ("branch_density",   "branchCount", "loc"),
    ]

    def __init__(self, n_select: int = 12):
        """
        Args:
            n_select: Number of top features to keep from each selection
                      stage (MI and RF). Final feature count ≤ 2 × n_select.
        """
        self.n_select = n_select
        self.selected_features_: Optional[List[str]] = None
        self._log_cols_present_: List[str] = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()
        X = self._add_log_features(X, fit=True)
        X = self._add_interactions(X)
        X = self._add_ratios(X)
        X, self.selected_features_ = self._select_features(X, y)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X = self._add_log_features(X, fit=False)
        X = self._add_interactions(X)
        X = self._add_ratios(X)
        # Keep only the columns selected during fit
        keep = [c for c in self.selected_features_ if c in X.columns]
        return X[keep]

    # ── private helpers ───────────────────────────────────────────────────

    def _add_log_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Apply log1p to heavy-tailed metrics.
        log1p(x) = log(x+1) is numerically stable at x=0.
        Fitted to avoid transform mismatch between train/test.
        """
        if fit:
            self._log_cols_present_ = [
                c for c in X.columns
                if c.lower().replace("(", "").replace(")", "") in {
                    s.lower() for s in _LOG_COLS
                }
            ]
        for col in self._log_cols_present_:
            if col in X.columns:
                new_col = f"log_{col}"
                X[new_col] = np.log1p(X[col].clip(lower=0))
        return X

    def _add_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        for a, b in self._INTERACTION_PAIRS:
            if a in X.columns and b in X.columns:
                X[f"{a}x{b}"] = X[a] * X[b]
        return X

    def _add_ratios(self, X: pd.DataFrame) -> pd.DataFrame:
        for name, num, den in self._RATIO_DEFS:
            if num in X.columns and den in X.columns:
                X[name] = X[num] / (X[den].replace(0, np.nan)).fillna(0)
        return X

    def _select_features(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, List[str]]:
        cols = [c for c in X.columns if X[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
        X_num = X[cols].fillna(0)

        k = min(self.n_select, len(cols))

        # Stage 1: Mutual Information
        mi_scores = mutual_info_classif(X_num, y, random_state=42)
        mi_top = set(
            pd.Series(mi_scores, index=cols)
            .nlargest(k).index.tolist()
        )

        # Stage 2: Random Forest Importance
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_num, y)
        rf_top = set(
            pd.Series(rf.feature_importances_, index=cols)
            .nlargest(k).index.tolist()
        )

        # Union — keeps complementary information
        selected = sorted(mi_top | rf_top)
        print(
            f"  [feat] MI top-{k}: {len(mi_top)} feats | "
            f"RF top-{k}: {len(rf_top)} feats | "
            f"Union: {len(selected)} feats selected."
        )
        return X[selected], selected


def engineer_features(X: pd.DataFrame, y: pd.Series, n_select: int = 12) -> Tuple[pd.DataFrame, "FeatureEngineer"]:
    """Convenience wrapper. Returns (X_engineered, fitted_engineer)."""
    eng = FeatureEngineer(n_select=n_select)
    X_new = eng.fit_transform(X, y)
    return X_new, eng


# ─────────────────────────────────────────────────────────────────────────────
# 4. IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def handle_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample the training set to address class imbalance.

    strategy options
    ────────────────
    'smote'        SMOTE oversampling (default).
                   Preferred: interpolates minority samples → better
                   generalisation than random duplication. Maximises recall
                   for defective modules (the primary SDP goal).

    'adasyn'       ADASYN oversampling.
                   Generates more samples near the decision boundary.
                   More aggressive than SMOTE; use when SMOTE under-samples.

    'none'         No resampling. Use class_weight='balanced' in the models
                   instead. Useful when dataset is small or SMOTE distorts.

    Why oversampling > undersampling for SDP?
    ─────────────────────────────────────────
    KC1+KC2 has ~16.5% defective. Undersampling to 50/50 discards ~83% of
    clean modules, wasting information. Oversampling keeps all data.

    Why apply ONLY to training data?
    ─────────────────────────────────
    Applying to the full dataset before split would cause data leakage:
    synthetic minority samples would appear in both train and test,
    inflating performance metrics artificially.
    """
    if strategy == "none":
        return X, y

    if not IMBLEARN_AVAILABLE:
        print(
            "  [imbalance] imbalanced-learn not installed. "
            "Falling back to 'none' (use class_weight='balanced' in models)."
        )
        return X, y

    counts = np.bincount(y.astype(int))
    ratio = counts.min() / counts.max()
    print(f"  [imbalance] Before SMOTE: {counts} — minority ratio {ratio:.2%}")

    try:
        if strategy == "smote":
            sampler = SMOTE(random_state=random_state, k_neighbors=5)
        elif strategy == "adasyn":
            sampler = ADASYN(random_state=random_state, n_neighbors=5)
        else:
            raise ValueError(f"Unknown imbalance strategy: {strategy!r}")

        X_res, y_res = sampler.fit_resample(X, y)
        counts_after = np.bincount(y_res.astype(int))
        print(f"  [imbalance] After {strategy.upper()}: {counts_after}")
        return X_res, y_res

    except Exception as exc:
        print(f"  [imbalance] {strategy.upper()} failed ({exc}). Using original data.")
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on X_train, apply to both.

    Returns scaled arrays AND the fitted scaler (needed to transform
    new single-module inputs at prediction time).

    Why StandardScaler (not MinMaxScaler)?
    ───────────────────────────────────────
    SDP metrics have outliers even after IQR clipping. StandardScaler is
    more robust because it scales by mean/std, not min/max. MinMaxScaler
    would compress most values towards 0 whenever an extreme value exists.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified train/test split.

    Why stratified?
    ───────────────
    With ~16.5% minority class, a random split could place most defective
    modules in train or test by chance. Stratification ensures both splits
    have the same defect rate.
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def get_stratified_kfold(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """
    Return a StratifiedKFold object for cross-validation.

    Usage in model training:
        skf = get_stratified_kfold(n_splits=5)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
            ...
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER PIPELINE — used by backend/api.py and Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

class SDPPreprocessor:
    """
    End-to-end preprocessing pipeline for Software Defect Prediction.

    Typical usage (training)
    ────────────────────────
        pp = SDPPreprocessor()
        X_tr, X_te, y_tr, y_te = pp.fit_transform(df)

    Typical usage (inference on new data)
    ──────────────────────────────────────
        X_new_sc = pp.transform_new(df_new)   # returns scaled np.ndarray

    Attributes (available after fit_transform)
    ───────────────────────────────────────────
        pp.cleaner_          DataCleaner (fitted)
        pp.engineer_         FeatureEngineer (fitted)
        pp.scaler_           StandardScaler (fitted)
        pp.selected_features_ list of selected feature names
        pp.class_weight_dict_  {0: w0, 1: w1} for use in sklearn models
        pp.report_           dict with dataset statistics
    """

    def __init__(
        self,
        iqr_multiplier: float = 3.0,
        n_select: int = 12,
        test_size: float = 0.20,
        imbalance_strategy: str = "smote",
        random_state: int = 42,
    ):
        self.iqr_multiplier = iqr_multiplier
        self.n_select = n_select
        self.test_size = test_size
        self.imbalance_strategy = imbalance_strategy
        self.random_state = random_state

        # Set after fit_transform
        self.cleaner_: Optional[DataCleaner] = None
        self.engineer_: Optional[FeatureEngineer] = None
        self.scaler_: Optional[StandardScaler] = None
        self.selected_features_: Optional[List[str]] = None
        self.class_weight_dict_: Optional[Dict[int, float]] = None
        self.report_: Dict = {}

    # ── public API ────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline: clean → split → engineer → scale → resample (train only).

        Returns
        ───────
        X_train_res, X_test_sc, y_train_res, y_test
        (all numpy arrays, ready for sklearn / TensorFlow)
        """
        print("\n[SDPPreprocessor] Starting pipeline...")
        print(f"  Raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── step 1: clean ─────────────────────────────────────────────────
        self.cleaner_ = DataCleaner(iqr_multiplier=self.iqr_multiplier)
        df_clean = self.cleaner_.fit_transform(df)
        print(f"  After cleaning: {df_clean.shape[0]} rows")

        # ── step 2: separate X / y ─────────────────────────────────────────
        y = df_clean["LABEL"].astype(int)
        X = df_clean.drop(
            columns=[c for c in df_clean.columns if c.lower() in _EXCLUDE_COLS]
        ).select_dtypes(include="number")

        # ── step 3: stratified split (BEFORE feature engineering!) ────────
        # Reason: feature selection must be fitted on train only to prevent
        # information leakage from test set into feature selection.
        X_tr, X_te, y_tr, y_te = split_stratified(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # ── step 4: feature engineering on train ──────────────────────────
        self.engineer_ = FeatureEngineer(n_select=self.n_select)
        X_tr_eng = self.engineer_.fit_transform(X_tr, y_tr)
        X_te_eng = self.engineer_.transform(X_te)
        self.selected_features_ = self.engineer_.selected_features_

        # ── step 5: scale (fit on train) ──────────────────────────────────
        X_tr_sc, X_te_sc, self.scaler_ = scale_features(
            X_tr_eng.values, X_te_eng.values
        )

        # ── step 6: imbalance (train only) ────────────────────────────────
        X_tr_res, y_tr_res = handle_imbalance(
            X_tr_sc, y_tr.values,
            strategy=self.imbalance_strategy,
            random_state=self.random_state,
        )

        # ── compute class weights for models that accept it ───────────────
        n_total = len(y_tr_res)
        n_pos = y_tr_res.sum()
        n_neg = n_total - n_pos
        self.class_weight_dict_ = {
            0: n_total / (2 * n_neg) if n_neg > 0 else 1.0,
            1: n_total / (2 * n_pos) if n_pos > 0 else 1.0,
        }

        # ── build report ──────────────────────────────────────────────────
        self._build_report(df, df_clean, y_tr_res, y_te)

        print(
            f"  [done] Train: {X_tr_res.shape} | Test: {X_te_sc.shape} | "
            f"Features: {len(self.selected_features_)}"
        )
        return X_tr_res, X_te_sc, y_tr_res, y_te.values

    def transform_new(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Transform a new (unseen) feature matrix using fitted pipeline.
        Used for inference on single modules or new files.
        """
        assert self.cleaner_ and self.engineer_ and self.scaler_, \
            "Call fit_transform() before transform_new()."
        X_clean = self.cleaner_.transform(X_new)
        X_eng = self.engineer_.transform(X_clean)
        return self.scaler_.transform(X_eng.values)

    # ── helpers ───────────────────────────────────────────────────────────

    def _build_report(
        self,
        df_raw: pd.DataFrame,
        df_clean: pd.DataFrame,
        y_train: np.ndarray,
        y_test: pd.Series,
    ) -> None:
        raw_defect_rate = df_raw.get("LABEL", df_raw.get("defects", pd.Series(dtype=int))).mean()
        train_defect_rate = y_train.mean()
        test_defect_rate = y_test.mean()
        self.report_ = {
            "raw_shape": df_raw.shape,
            "clean_shape": df_clean.shape,
            "raw_defect_rate": float(raw_defect_rate) if hasattr(raw_defect_rate, '__float__') else 0.0,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_defect_rate": float(train_defect_rate),
            "test_defect_rate": float(test_defect_rate),
            "n_features": len(self.selected_features_) if self.selected_features_ else 0,
            "selected_features": self.selected_features_ or [],
            "class_weights": self.class_weight_dict_,
            "imbalance_strategy": self.imbalance_strategy,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 8. LEGACY COMPATIBILITY WRAPPERS (keep old call signatures working)
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath: str) -> pd.DataFrame:
    """Legacy wrapper → load_data()."""
    return load_data(filepath)


def extract_features(df: pd.DataFrame):
    """Legacy wrapper. Returns (X, y) with minimal processing."""
    y = df["LABEL"].astype(int)
    exclude = {c.lower() for c in _EXCLUDE_COLS}
    X = df[[c for c in df.columns if c.lower() not in exclude]].select_dtypes("number")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """Legacy wrapper → split_stratified()."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify if stratify is not None else y,
    )


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Legacy compatibility — returns basic dataset summary."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "defect_rate": float(df["LABEL"].mean()) if "LABEL" in df.columns else 0.0,
        "statistics": df.describe().to_dict(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE-TEST (python backend/preprocessing.py)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    DATA_DIR = Path(__file__).parent / "data"
    csv_path = DATA_DIR / "kc1_kc2.csv"
    arff_path = DATA_DIR / "kc1.arff"

    # Test CSV
    if csv_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading CSV: {csv_path}")
        df_csv = load_data(str(csv_path))
        print(f"Shape: {df_csv.shape} | Defect rate: {df_csv['LABEL'].mean():.2%}")
        pp = SDPPreprocessor(imbalance_strategy="smote")
        X_tr, X_te, y_tr, y_te = pp.fit_transform(df_csv)
        print("\nPipeline report:")
        for k, v in pp.report_.items():
            print(f"  {k}: {v}")
    else:
        print(f"[skip] {csv_path} not found")

    # Test ARFF
    if arff_path.exists():
        print(f"\n{'='*60}")
        print(f"Loading ARFF: {arff_path}")
        df_arff = load_data(str(arff_path))
        print(f"Shape: {df_arff.shape} | Defect rate: {df_arff['LABEL'].mean():.2%}")
        pp2 = SDPPreprocessor(imbalance_strategy="none")
        X_tr2, X_te2, y_tr2, y_te2 = pp2.fit_transform(df_arff)
        print(f"\nTrain: {X_tr2.shape} | Test: {X_te2.shape}")
    else:
        print(f"[skip] {arff_path} not found")
