"""
ML Training Pipeline for Software Defect Prediction (SDP) — 2025 Best Practices
=================================================================================

Model catalogue
───────────────
  Base models (always available via sklearn):
    • Logistic Regression      – baseline linear model, interpretable
    • Random Forest            – ensemble, handles imbalance, feature importance
    • MLP (Neural Network)     – sklearn MLPClassifier (no TF dependency here)
    • Gradient Boosting        – sklearn fallback when XGBoost is absent

  Boosting models (optional, graceful fallback):
    • XGBoost                  – best single model in most SDP benchmarks 2020-25
    • LightGBM                 – faster than XGBoost, better on sparse features

  Ensemble layer:
    • VotingClassifier (soft)  – probability averaging across base models
    • StackingClassifier       – LR meta-learner on top of base predictions

  Hyperparameter tuning:
    • Optuna (preferred)       – Bayesian optimisation, 50 trials, maximise ROC-AUC
    • GridSearchCV (fallback)  – exhaustive search on coarser grid

Design notes
────────────
  1. Primary metric = ROC-AUC (not accuracy) because of class imbalance.
     Recall is secondary — missing a defective module is costlier than a
     false alarm in practice (Menzies 2004, Hall et al. 2012).
  2. All models receive class_weight or scale_pos_weight equivalent to
     handle the ~16.5% minority class even without SMOTE.
  3. StratifiedKFold CV is run on the TRAINING set only.
     The held-out test set is touched exactly ONCE at the end.
  4. Optuna suppresses its verbose output; set LOG_OPTUNA=True to re-enable.
  5. Models are saved as joblib bundles in backend/models/ directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)  # suppress sklearn deprecation noise

# ── optional heavy libraries ──────────────────────────────────────────────────
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# ── default paths ─────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).parent
MODELS_DIR = _BACKEND_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

LOG_OPTUNA = False  # set True locally for verbose Optuna output


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    """Holds all evaluation artefacts for a single trained model."""

    name: str
    model: Any
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    avg_precision: float = 0.0
    cv_roc_auc_mean: float = 0.0
    cv_roc_auc_std: float = 0.0
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_proba: np.ndarray = field(default_factory=lambda: np.array([]))
    train_time_s: float = 0.0
    best_params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialisable summary (exclude large arrays and model object)."""
        return {
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "avg_precision": round(self.avg_precision, 4),
            "cv_roc_auc_mean": round(self.cv_roc_auc_mean, 4),
            "cv_roc_auc_std": round(self.cv_roc_auc_std, 4),
            "train_time_s": round(self.train_time_s, 2),
            "best_params": self.best_params,
        }


# ─────────────────────────────────────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
    train_time: float,
    best_params: Dict,
    cv_folds: int = 5,
) -> ModelResult:
    """Compute all metrics for a fitted model."""
    # CV on training data
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
    )

    # Test predictions
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    return ModelResult(
        name=name,
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, zero_division=0),
        recall=recall_score(y_test, y_pred, zero_division=0),
        f1=f1_score(y_test, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_test, y_proba),
        avg_precision=average_precision_score(y_test, y_proba),
        cv_roc_auc_mean=cv_scores.mean(),
        cv_roc_auc_std=cv_scores.std(),
        y_pred=y_pred,
        y_proba=y_proba,
        train_time_s=train_time,
        best_params=best_params,
    )


def _get_pos_weight(y: np.ndarray) -> float:
    """XGBoost scale_pos_weight = n_negative / n_positive."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def _tune_with_optuna(
    factory,
    param_space_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 50,
    cv_folds: int = 5,
) -> Dict:
    """
    Run Optuna Bayesian optimisation to maximise ROC-AUC (StratifiedKFold).

    Args:
        factory      : callable(params) → sklearn estimator
        param_space_fn: callable(trial) → params dict
        n_trials     : Optuna trial budget (default 50, ~30s on KC1+KC2)
        cv_folds     : number of CV folds inside tuning loop
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial):
        params = param_space_fn(trial)
        clf = factory(params)
        scores = cross_val_score(
            clf, X_train, y_train, cv=skf, scoring="roc_auc", n_jobs=-1
        )
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=LOG_OPTUNA,
        n_jobs=1,
    )
    return study.best_params


def _tune_with_grid(estimator, param_grid, X_train, y_train, cv_folds=5) -> Dict:
    """GridSearchCV fallback when Optuna is unavailable."""
    from sklearn.model_selection import GridSearchCV

    gs = GridSearchCV(
        estimator,
        param_grid,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs.best_params_


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL MODEL TRAINERS
# ─────────────────────────────────────────────────────────────────────────────

def _train_logistic_regression(
    X_train, y_train, X_test, y_test, tune: bool, class_weight_dict, cv_folds
) -> ModelResult:
    """
    Logistic Regression — linear baseline.
    Strength: fast, interpretable, good AUC on linearly separable data.
    SDP context: often competitive on Halstead-heavy feature sets.
    """
    if tune and OPTUNA_AVAILABLE:
        def space(trial):
            return {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
                "max_iter": trial.suggest_int("max_iter", 500, 2000),
            }
        best = _tune_with_optuna(
            lambda p: LogisticRegression(
                **p, class_weight=class_weight_dict, random_state=42
            ),
            space, X_train, y_train, n_trials=40, cv_folds=cv_folds,
        )
    elif tune:
        best = _tune_with_grid(
            LogisticRegression(class_weight=class_weight_dict, max_iter=1000, random_state=42),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["lbfgs", "liblinear"]},
            X_train, y_train, cv_folds,
        )
    else:
        best = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}

    # Ensure max_iter is in best dict before unpacking to avoid duplicate kwarg
    best.setdefault("max_iter", 1000)
    t0 = time.time()
    clf = LogisticRegression(**best, class_weight=class_weight_dict, random_state=42)
    clf.fit(X_train, y_train)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "Logistic Regression", time.time() - t0, best, cv_folds)


def _train_random_forest(
    X_train, y_train, X_test, y_test, tune: bool, class_weight_dict, cv_folds
) -> ModelResult:
    """
    Random Forest — tree ensemble, robust feature importance.
    Strength: captures nonlinear interactions, built-in OOB estimate.
    SDP context: consistently top-3 in NASA MDP benchmarks.
    """
    if tune and OPTUNA_AVAILABLE:
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 4, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        best = _tune_with_optuna(
            lambda p: RandomForestClassifier(
                **p, class_weight=class_weight_dict, random_state=42, n_jobs=-1
            ),
            space, X_train, y_train, n_trials=50, cv_folds=cv_folds,
        )
    elif tune:
        best = _tune_with_grid(
            RandomForestClassifier(class_weight=class_weight_dict, random_state=42),
            {"n_estimators": [100, 300], "max_depth": [5, 10, None],
             "min_samples_split": [2, 5]},
            X_train, y_train, cv_folds,
        )
    else:
        best = {"n_estimators": 200, "max_depth": 10, "min_samples_split": 5,
                "min_samples_leaf": 2, "max_features": "sqrt"}

    t0 = time.time()
    clf = RandomForestClassifier(
        **best, class_weight=class_weight_dict, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "Random Forest", time.time() - t0, best, cv_folds)


def _train_mlp(
    X_train, y_train, X_test, y_test, tune: bool, class_weight_dict, cv_folds
) -> ModelResult:
    """
    MLP (Neural Network via sklearn) — no TF dependency.
    Strength: captures complex nonlinear patterns.
    SDP context: sensitive to scaling → preprocessor must StandardScale first.
    Note: class_weight not natively supported by MLPClassifier;
          use sample_weight via fit() if needed.
    """
    if tune and OPTUNA_AVAILABLE:
        def space(trial):
            layer_size = trial.suggest_categorical("layer_size", [64, 128, 256])
            n_layers = trial.suggest_int("n_layers", 1, 3)
            return {
                "hidden_layer_sizes": tuple([layer_size] * n_layers),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        best = _tune_with_optuna(
            lambda p: MLPClassifier(
                **p, max_iter=500, early_stopping=True, random_state=42
            ),
            space, X_train, y_train, n_trials=40, cv_folds=cv_folds,
        )
    else:
        best = {"hidden_layer_sizes": (128, 64), "alpha": 1e-4, "learning_rate_init": 1e-3}

    # Build sample_weight to compensate for imbalance
    n_total = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_total - n_pos
    sw = np.where(y_train == 1, n_total / (2 * n_pos), n_total / (2 * n_neg))

    t0 = time.time()
    clf = MLPClassifier(
        **best, max_iter=500, early_stopping=True, random_state=42
    )
    clf.fit(X_train, y_train, sample_weight=sw)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "Neural Network (MLP)", time.time() - t0, best, cv_folds)


def _train_xgboost(
    X_train, y_train, X_test, y_test, tune: bool, cv_folds
) -> Optional[ModelResult]:
    """
    XGBoost — gradient boosted trees, state-of-the-art on tabular SDP.
    Handles imbalance via scale_pos_weight = n_neg / n_pos.
    Strength: usually #1 single model on NASA MDP datasets in recent papers.
    """
    if not XGB_AVAILABLE:
        return None

    spw = _get_pos_weight(y_train)

    if tune and OPTUNA_AVAILABLE:
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            }
        best = _tune_with_optuna(
            lambda p: xgb.XGBClassifier(
                **p, scale_pos_weight=spw, eval_metric="auc",
                random_state=42, verbosity=0,
                use_label_encoder=False,
            ),
            space, X_train, y_train, n_trials=50, cv_folds=cv_folds,
        )
    elif tune:
        best = _tune_with_grid(
            xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="auc",
                               random_state=42, verbosity=0),
            {"n_estimators": [100, 300], "max_depth": [4, 7],
             "learning_rate": [0.05, 0.1]},
            X_train, y_train, cv_folds,
        )
    else:
        best = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8}

    t0 = time.time()
    clf = xgb.XGBClassifier(
        **best, scale_pos_weight=spw, eval_metric="auc",
        random_state=42, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "XGBoost", time.time() - t0, best, cv_folds)


def _train_lightgbm(
    X_train, y_train, X_test, y_test, tune: bool, cv_folds
) -> Optional[ModelResult]:
    """
    LightGBM — fast GBDT, excellent on high-dimensional SDP data.
    is_unbalance=True internally adjusts for class imbalance.
    Strength: 3–10× faster than XGBoost on large datasets; similar AUC.
    """
    if not LGB_AVAILABLE:
        return None

    if tune and OPTUNA_AVAILABLE:
        def space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1.0, log=True),
            }
        best = _tune_with_optuna(
            lambda p: lgb.LGBMClassifier(
                **p, is_unbalance=True, random_state=42, verbosity=-1,
            ),
            space, X_train, y_train, n_trials=50, cv_folds=cv_folds,
        )
    elif tune:
        best = _tune_with_grid(
            lgb.LGBMClassifier(is_unbalance=True, random_state=42, verbosity=-1),
            {"n_estimators": [100, 300], "num_leaves": [31, 63],
             "learning_rate": [0.05, 0.1]},
            X_train, y_train, cv_folds,
        )
    else:
        best = {"n_estimators": 300, "num_leaves": 63, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8}

    t0 = time.time()
    clf = lgb.LGBMClassifier(
        **best, is_unbalance=True, random_state=42, verbosity=-1,
    )
    clf.fit(X_train, y_train)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "LightGBM", time.time() - t0, best, cv_folds)


def _train_gradient_boosting(
    X_train, y_train, X_test, y_test, tune: bool, class_weight_dict, cv_folds
) -> ModelResult:
    """
    sklearn GradientBoosting — fallback when both XGB and LGB absent.
    Slower than XGB/LGB but no extra dependency.
    Uses sample_weight to handle imbalance.
    """
    spw = np.where(
        y_train == 1,
        class_weight_dict.get(1, 1.0),
        class_weight_dict.get(0, 1.0),
    )
    best = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8}

    t0 = time.time()
    clf = GradientBoostingClassifier(**best, random_state=42)
    clf.fit(X_train, y_train, sample_weight=spw)
    return _evaluate(clf, X_train, y_train, X_test, y_test,
                     "Gradient Boosting", time.time() - t0, best, cv_folds)


# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE TRAINERS
# ─────────────────────────────────────────────────────────────────────────────

def _train_voting_ensemble(
    base_models: List[Tuple[str, Any]],
    X_train, y_train, X_test, y_test, cv_folds
) -> ModelResult:
    """
    Soft VotingClassifier — averages predict_proba across base models.

    Why soft voting?
    ────────────────
    Hard voting uses majority vote on binary predictions, discarding
    probability information. Soft voting averages probabilities → lower
    variance → typically higher AUC, especially with calibrated models.

    In SDP, combining LR (linear boundary) + RF (nonlinear) + XGB (boosted)
    often outperforms any single model by 1-3% AUC.
    """
    t0 = time.time()
    vc = VotingClassifier(estimators=base_models, voting="soft", n_jobs=-1)
    vc.fit(X_train, y_train)
    return _evaluate(vc, X_train, y_train, X_test, y_test,
                     "Voting Ensemble (Soft)", time.time() - t0, {}, cv_folds)


def _train_stacking_ensemble(
    base_models: List[Tuple[str, Any]],
    X_train, y_train, X_test, y_test, cv_folds
) -> ModelResult:
    """
    StackingClassifier — LR meta-learner on out-of-fold predictions.

    Why stacking?
    ─────────────
    Base classifiers' predictions become features for a meta-model.
    Stacking learns WHICH base model to trust for which region of input
    space. Typically outperforms voting by 0.5–2% AUC.

    Meta-learner = Logistic Regression:
    • Fast, interpretable weights
    • Less prone to overfitting on the meta-features
    • passthrough=True adds original features → meta-learner has full context
    """
    meta = LogisticRegression(C=1.0, class_weight="balanced",
                              max_iter=1000, random_state=42)
    skf_inner = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    t0 = time.time()
    sc = StackingClassifier(
        estimators=base_models,
        final_estimator=meta,
        cv=skf_inner,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )
    sc.fit(X_train, y_train)
    return _evaluate(sc, X_train, y_train, X_test, y_test,
                     "Stacking Ensemble", time.time() - t0, {}, cv_folds)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    High-level interface for the SDP model training pipeline.

    Typical usage
    ─────────────
        trainer = ModelTrainer()
        results = trainer.train_all_models(X_tr, X_te, y_tr, y_te,
                                           tune=True, cv_folds=5)
        summary_df = trainer.get_leaderboard()
        trainer.save_all_models("backend/models")
        loaded = ModelTrainer.load_model("backend/models/XGBoost.joblib")

    Parameters
    ──────────
        class_weight_dict : {0: w0, 1: w1} from SDPPreprocessor.class_weight_dict_
                            Used by sklearn models that accept class_weight=.
        include_ensembles : build VotingClassifier + StackingClassifier after
                            individual models are trained (default True).
    """

    def __init__(
        self,
        class_weight_dict: Optional[Dict[int, float]] = None,
        include_ensembles: bool = True,
    ):
        self.class_weight_dict = class_weight_dict or {0: 1.0, 1: 1.0}
        self.include_ensembles = include_ensembles
        self.results_: Dict[str, ModelResult] = {}
        self._log: List[str] = []

    # ── public API ────────────────────────────────────────────────────────

    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        tune: bool = True,
        cv_folds: int = 5,
    ) -> Dict[str, ModelResult]:
        """
        Train all models sequentially, evaluate on held-out test set.

        Args
        ────
            X_train, y_train : pre-processed training data (from SDPPreprocessor)
            X_test,  y_test  : held-out test data — touched ONCE at end
            tune             : run Optuna / GridSearchCV (adds ~2-5 min)
            cv_folds         : StratifiedKFold splits for CV and tuning

        Returns
        ───────
            Dict[model_name → ModelResult]
        """
        self._log = []
        self._info(f"Training pipeline started | tune={tune} | cv_folds={cv_folds}")
        self._info(f"Train: {X_train.shape} | Test: {X_test.shape}")
        cw = self.class_weight_dict

        # ── base models ───────────────────────────────────────────────────
        self._run("Logistic Regression",
                  _train_logistic_regression,
                  X_train, y_train, X_test, y_test, tune, cw, cv_folds)

        self._run("Random Forest",
                  _train_random_forest,
                  X_train, y_train, X_test, y_test, tune, cw, cv_folds)

        self._run("Neural Network (MLP)",
                  _train_mlp,
                  X_train, y_train, X_test, y_test, tune, cw, cv_folds)

        # ── boosted trees ─────────────────────────────────────────────────
        if XGB_AVAILABLE:
            self._run("XGBoost",
                      _train_xgboost,
                      X_train, y_train, X_test, y_test, tune, cv_folds)
        else:
            self._info("XGBoost not installed — training sklearn GradientBoosting instead.")
            self._run("Gradient Boosting",
                      _train_gradient_boosting,
                      X_train, y_train, X_test, y_test, tune, cw, cv_folds)

        if LGB_AVAILABLE:
            self._run("LightGBM",
                      _train_lightgbm,
                      X_train, y_train, X_test, y_test, tune, cv_folds)

        # ── ensemble layer ────────────────────────────────────────────────
        if self.include_ensembles and len(self.results_) >= 2:
            self._info("Building ensemble models...")
            base = [
                (name, res.model)
                for name, res in self.results_.items()
                if hasattr(res.model, "predict_proba")
            ]
            if len(base) >= 2:
                self._run("Voting Ensemble (Soft)",
                          _train_voting_ensemble,
                          base, X_train, y_train, X_test, y_test, cv_folds)

                self._run("Stacking Ensemble",
                          _train_stacking_ensemble,
                          base, X_train, y_train, X_test, y_test, cv_folds)

        self._info(
            f"Pipeline complete. "
            f"Best AUC: {self._best_by('roc_auc')!r} | "
            f"Best Recall: {self._best_by('recall')!r}"
        )
        return self.results_

    def get_leaderboard(self, sort_by: str = "roc_auc") -> "pd.DataFrame":
        """Return a sorted DataFrame of all model metrics."""
        import pandas as pd

        rows = [r.to_dict() for r in self.results_.values()]
        df = pd.DataFrame(rows).sort_values(sort_by, ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    def get_logs(self) -> List[str]:
        return self._log.copy()

    def best_model(self, metric: str = "roc_auc") -> Optional[ModelResult]:
        if not self.results_:
            return None
        return max(self.results_.values(), key=lambda r: getattr(r, metric, 0.0))

    # ── persistence ───────────────────────────────────────────────────────

    def save_all_models(self, directory: str = str(MODELS_DIR)) -> List[str]:
        """
        Save each trained model as a .joblib file.

        Why joblib over pickle?
        ───────────────────────
        joblib uses memory-mapped arrays for large numpy arrays inside
        sklearn estimators → 3–6× faster serialisation for RF/XGB.
        """
        saved = []
        Path(directory).mkdir(parents=True, exist_ok=True)
        for name, res in self.results_.items():
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            path = Path(directory) / f"{safe_name}.joblib"
            bundle = {
                "model": res.model,
                "metrics": res.to_dict(),
                "name": res.name,
            }
            joblib.dump(bundle, path, compress=3)
            self._info(f"Saved: {path}")
            saved.append(str(path))
        return saved

    def save_model(self, name: str, directory: str = str(MODELS_DIR)) -> str:
        """Save a single model by name."""
        if name not in self.results_:
            raise KeyError(f"Model {name!r} not found. Available: {list(self.results_)}")
        res = self.results_[name]
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = Path(directory) / f"{safe_name}.joblib"
        joblib.dump({"model": res.model, "metrics": res.to_dict(), "name": res.name},
                    path, compress=3)
        return str(path)

    @staticmethod
    def load_model(path: str) -> Dict:
        """
        Load a saved model bundle.

        Returns
        ───────
        dict with keys: 'model', 'metrics', 'name'
        """
        return joblib.load(path)

    @staticmethod
    def list_saved_models(directory: str = str(MODELS_DIR)) -> List[Dict]:
        """List all saved .joblib model files with their metrics."""
        results = []
        for f in sorted(Path(directory).glob("*.joblib")):
            try:
                bundle = joblib.load(f)
                results.append({
                    "path": str(f),
                    "name": bundle.get("name", f.stem),
                    "metrics": bundle.get("metrics", {}),
                })
            except Exception:
                pass
        return results

    # ── private helpers ───────────────────────────────────────────────────

    def _run(self, label: str, fn, *args, **kwargs) -> None:
        self._info(f"[TRAIN] {label}...")
        try:
            result = fn(*args, **kwargs)
            if result is None:
                self._info(f"[SKIP]  {label} — library unavailable.")
                return
            self.results_[result.name] = result
            self._info(
                f"[OK]    {result.name} | "
                f"AUC={result.roc_auc:.4f} | "
                f"Recall={result.recall:.4f} | "
                f"F1={result.f1:.4f} | "
                f"CV-AUC={result.cv_roc_auc_mean:.4f}±{result.cv_roc_auc_std:.4f} | "
                f"Time={result.train_time_s:.1f}s"
            )
        except Exception as exc:
            self._info(f"[ERR]   {label} failed: {exc}")

    def _info(self, msg: str) -> None:
        self._log.append(msg)
        print(f"  {msg}")

    def _best_by(self, metric: str) -> str:
        if not self.results_:
            return "N/A"
        best = max(self.results_.values(), key=lambda r: getattr(r, metric, 0.0))
        return f"{best.name} ({getattr(best, metric):.4f})"


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION ONLY (no test leakage)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate_model(
    model, X: np.ndarray, y: np.ndarray, cv_folds: int = 10
) -> Dict:
    """
    10-fold stratified CV — use when no fixed test split is available.
    Returns mean ± std for AUC, F1, Recall.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1", n_jobs=-1)
    rec = cross_val_score(model, X, y, cv=skf, scoring="recall", n_jobs=-1)
    return {
        "roc_auc_mean": auc.mean(), "roc_auc_std": auc.std(),
        "f1_mean": f1.mean(),       "f1_std": f1.std(),
        "recall_mean": rec.mean(),  "recall_std": rec.std(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD-COMPAT: old DefectPredictionModels class (kept for API stability)
# ─────────────────────────────────────────────────────────────────────────────

class DefectPredictionModels:
    """
    Legacy wrapper — keeps old call signatures working.
    New code should use ModelTrainer directly.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trainer = ModelTrainer()
        self.is_trained = False
        self.scaler = None
        self.feature_names = None

    def initialize_models(self) -> None:
        print("DefectPredictionModels: use ModelTrainer.train_all_models() instead.")

    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        print("Legacy train_models — use ModelTrainer.train_all_models() for full pipeline.")
        self.is_trained = True
        return {}

    def evaluate_models(self, X_test, y_test) -> Dict:
        if not self.trainer.results_:
            return {}
        out = {}
        for name, res in self.trainer.results_.items():
            out[name] = res.to_dict()
        return out

    def save_models(self, path: str) -> None:
        self.trainer.save_all_models(path)

    def load_models(self, path: str) -> None:
        for f in Path(path).glob("*.joblib"):
            bundle = ModelTrainer.load_model(str(f))
            name = bundle["name"]
            # Reconstruct minimal ModelResult
            res = ModelResult(name=name, model=bundle["model"])
            for k, v in bundle.get("metrics", {}).items():
                if hasattr(res, k):
                    setattr(res, k, v)
            self.trainer.results_[name] = res
        self.is_trained = True


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import SDPPreprocessor, load_data

    DATA = Path(__file__).parent / "data" / "kc1_kc2.csv"
    if not DATA.exists():
        print(f"[skip] {DATA} not found")
        sys.exit(0)

    print("=" * 60)
    print("Loading + preprocessing KC1+KC2...")
    df = load_data(str(DATA))
    pp = SDPPreprocessor(imbalance_strategy="none")
    X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
    print(f"Train: {X_tr.shape} | Test: {X_te.shape}")

    print("\n" + "=" * 60)
    print("Training all models (tune=False for speed)...")
    trainer = ModelTrainer(
        class_weight_dict=pp.class_weight_dict_,
        include_ensembles=True,
    )
    results = trainer.train_all_models(X_tr, X_te, y_tr, y_te, tune=False, cv_folds=5)

    print("\n" + "=" * 60)
    print("LEADERBOARD")
    print(trainer.get_leaderboard().to_string())

    saved = trainer.save_all_models()
    print(f"\nSaved {len(saved)} model files.")
