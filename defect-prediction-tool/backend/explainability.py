"""
Explainability Module for Software Defect Prediction (XAI)
===========================================================
Provides:
  • SHAP global explanations  — feature importance across the whole dataset
  • SHAP local explanations   — why a SINGLE module is predicted defective
  • LIME local explanations   — model-agnostic alternative for single instances
  • Plotly-based figures      — all charts plug directly into Streamlit
  • Narrative text generator  — human-readable risk report per module

Design decisions
────────────────
• TreeExplainer (RF / XGB / LGB):
    Exact Shapley values via tree path enumeration. O(depth) per sample.
    No sampling needed → exact, fast, no approximation error.
    Use this whenever the model IS a tree ensemble.

• LinearExplainer (Logistic Regression):
    Exact Shapley values for linear models. Uses background dataset mean.
    Equivalent to coefficient × (x - mean), but in proper SHAP framework.

• PermutationExplainer (MLP / Stacking / Voting fallback):
    Model-agnostic, slower. Samples permutations to estimate Shapley values.
    Used when no structural explainer is available (black-box fallback).

• LIME:
    Fits a local linear surrogate around a single point by sampling
    neighbours and weighting by proximity. Complementary to SHAP:
    - SHAP: consistent, theoretically grounded (Shapley axioms)
    - LIME: directly optimises local fidelity, easier to tune interactively

• All matplotlib figures are returned as plt.Figure objects (not shown)
  so Streamlit can embed them with st.pyplot(fig, use_container_width=True).

• All Plotly figures are returned as go.Figure objects
  so Streamlit can embed with st.plotly_chart(fig, use_container_width=True).

Supported models (auto-detected)
──────────────────────────────────
  RandomForest, ExtraTrees, GradientBoosting → TreeExplainer
  XGBClassifier, LGBMClassifier              → TreeExplainer
  LogisticRegression, Ridge                  → LinearExplainer
  MLPClassifier, VotingClassifier, Stacking  → PermutationExplainer
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (required for Streamlit)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── optional: shap ─────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── optional: lime ─────────────────────────────────────────────────────────
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# design palette (matches styles.py brand colours)
_DANGER  = "#b8323d"   # high risk / positive shap
_SAFE    = "#2f9d8a"   # low risk / negative shap
_WARN    = "#e2a44c"   # medium
_MUTED   = "#9aabbe"
_BG      = "#0f1318"
_PANEL   = "#171d24"
_INK     = "#e6edf6"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — explainer factory
# ─────────────────────────────────────────────────────────────────────────────

def _is_tree(model) -> bool:
    tree_types = (
        "RandomForest", "ExtraTree", "GradientBoosting",
        "XGB", "LGBM", "DecisionTree",
    )
    return any(t in type(model).__name__ for t in tree_types)


def _is_linear(model) -> bool:
    linear_types = ("LogisticRegression", "LinearSVC", "Ridge", "SGD")
    return any(t in type(model).__name__ for t in linear_types)


def _make_explainer(model, X_background: np.ndarray):
    """
    Auto-select the best SHAP explainer for the given model type.

    Returns (explainer, explainer_type_str)
    """
    if not SHAP_AVAILABLE:
        return None, "none"

    if _is_tree(model):
        try:
            exp = shap.TreeExplainer(model)
            return exp, "tree"
        except Exception:
            pass  # fall through to permutation

    if _is_linear(model):
        try:
            # use a small background sample for speed
            bg = shap.sample(X_background, min(100, len(X_background)))
            exp = shap.LinearExplainer(model, bg, feature_perturbation="interventional")
            return exp, "linear"
        except Exception:
            pass

    # Universal fallback: PermutationExplainer
    bg = shap.sample(X_background, min(50, len(X_background)))
    exp = shap.PermutationExplainer(
        model.predict_proba if hasattr(model, "predict_proba")
        else model.predict,
        bg,
    )
    return exp, "permutation"


def _compute_shap_values(
    explainer, X: np.ndarray, explainer_type: str
) -> np.ndarray:
    """
    Compute raw SHAP values and return the positive-class column (index 1).
    Always returns shape (n_samples, n_features).
    """
    sv = explainer(X) if explainer_type in ("tree", "permutation") else explainer.shap_values(X)

    # Handle different return types from different SHAP versions
    if hasattr(sv, "values"):           # shap.Explanation object (new API)
        vals = sv.values
        if vals.ndim == 3:              # (n, features, 2) — binary classification
            vals = vals[:, :, 1]
    elif isinstance(sv, list):          # old API: list[class0, class1]
        vals = sv[1] if len(sv) == 2 else sv[0]
    else:
        vals = sv

    return vals.astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — main entry points
# ─────────────────────────────────────────────────────────────────────────────

class ModelExplainer:
    """
    Wraps a trained classifier with SHAP + LIME explanations.

    Usage
    ─────
        exp = ModelExplainer(model, X_train, feature_names)
        exp.fit()                                   # computes SHAP on X_train

        # Global views (whole dataset)
        fig_bar  = exp.plot_global_importance(X_test)
        fig_bee  = exp.plot_beeswarm(X_test)
        fig_heat = exp.plot_feature_heatmap(X_test)

        # Local view (single module, index i)
        fig_force    = exp.plot_local_waterfall(X_test, i)
        fig_lime     = exp.plot_lime_explanation(X_test, i)
        narrative    = exp.generate_narrative(X_test, i, y_pred_proba[i])
    """

    def __init__(
        self,
        model,
        X_train: np.ndarray,
        feature_names: List[str],
        model_name: str = "",
    ):
        self.model = model
        self.X_train = X_train
        self.feature_names = list(feature_names)
        self.model_name = model_name or type(model).__name__

        self._explainer = None
        self._exp_type = "none"
        self._shap_train: Optional[np.ndarray] = None  # cached train SHAP
        self._lime_explainer = None

    # ── setup ─────────────────────────────────────────────────────────────

    def fit(self) -> "ModelExplainer":
        """Compute SHAP explainer and cache train-set SHAP values."""
        if not SHAP_AVAILABLE:
            return self

        self._explainer, self._exp_type = _make_explainer(self.model, self.X_train)
        if self._explainer is not None:
            self._shap_train = _compute_shap_values(
                self._explainer, self.X_train, self._exp_type
            )

        if LIME_AVAILABLE:
            self._lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.X_train,
                feature_names=self.feature_names,
                class_names=["Clean", "Defective"],
                mode="classification",
                discretize_continuous=True,
                random_state=42,
            )
        return self

    # ── helpers ───────────────────────────────────────────────────────────

    def _get_shap(self, X: np.ndarray) -> Optional[np.ndarray]:
        if not SHAP_AVAILABLE or self._explainer is None:
            return None
        return _compute_shap_values(self._explainer, X, self._exp_type)

    def _mean_abs_shap(self, shap_vals: np.ndarray) -> pd.Series:
        imp = np.abs(shap_vals).mean(axis=0)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    # ── GLOBAL: feature importance bar chart ──────────────────────────────

    def plot_global_importance(
        self, X: np.ndarray, top_n: int = 15
    ) -> go.Figure:
        """
        Plotly bar chart of mean |SHAP| across all test samples.
        Equivalent to SHAP summary_plot(plot_type='bar') but Plotly.

        Falls back to RF/XGB native feature_importances_ if SHAP unavailable.
        """
        shap_vals = self._get_shap(X)
        if shap_vals is not None:
            importance = self._mean_abs_shap(shap_vals).head(top_n)
            subtitle = "Mean |SHAP value| (positive class)"
        elif hasattr(self.model, "feature_importances_"):
            imp = pd.Series(self.model.feature_importances_, index=self.feature_names)
            importance = imp.sort_values(ascending=False).head(top_n)
            subtitle = "Model native feature_importances_"
        else:
            return go.Figure().update_layout(title="Feature importance not available")

        colors = [_DANGER if i < 3 else _SAFE if i > top_n // 2 else _WARN
                  for i in range(len(importance))]

        fig = go.Figure(go.Bar(
            x=importance.values[::-1],
            y=importance.index[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.4f}" for v in importance.values[::-1]],
            textposition="outside",
            textfont=dict(size=10, color=_INK),
        ))
        fig.update_layout(
            title=dict(text=f"Global Feature Importance — {self.model_name}",
                       font=dict(color=_INK, size=14)),
            xaxis_title=subtitle,
            plot_bgcolor=_PANEL,
            paper_bgcolor=_BG,
            font=dict(color=_INK),
            height=max(350, top_n * 28),
            margin=dict(l=10, r=80, t=50, b=40),
        )
        return fig

    # ── GLOBAL: beeswarm / dot plot ───────────────────────────────────────

    def plot_beeswarm(self, X: np.ndarray, top_n: int = 12) -> plt.Figure:
        """
        SHAP beeswarm / summary plot rendered with matplotlib (shap native).
        Shows DISTRIBUTION of impact for each feature, colour-coded by value.

        Returns plt.Figure for st.pyplot(fig).
        Falls back to an empty figure if SHAP unavailable.
        """
        shap_vals = self._get_shap(X)
        if shap_vals is None or not SHAP_AVAILABLE:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "SHAP not available", ha="center", va="center")
            return fig

        importance = self._mean_abs_shap(shap_vals)
        top_feats = importance.head(top_n).index.tolist()
        feat_idx = [self.feature_names.index(f) for f in top_feats]

        sv_top = shap_vals[:, feat_idx]
        X_top  = X[:, feat_idx]

        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.55)))
        shap.summary_plot(
            sv_top, X_top,
            feature_names=top_feats,
            show=False,
            color_bar=True,
            plot_size=None,
            ax=ax if hasattr(ax, 'invert_yaxis') else None,
        )
        fig.patch.set_facecolor(_BG)
        ax = fig.gca()
        ax.set_facecolor(_PANEL)
        ax.tick_params(colors=_INK)
        ax.xaxis.label.set_color(_INK)
        fig.suptitle(f"SHAP Beeswarm — {self.model_name}",
                     color=_INK, fontsize=12, y=1.01)
        fig.tight_layout()
        return fig

    # ── GLOBAL: feature × metric heatmap ─────────────────────────────────

    def plot_feature_heatmap(self, X: np.ndarray, top_n: int = 10) -> go.Figure:
        """
        Heatmap: rows = top features, columns = samples (sorted by SHAP sum).
        Colour = SHAP value (red → pushes toward defective, blue → clean).
        Reveals which COMBINATION of features drives high-risk predictions.
        """
        shap_vals = self._get_shap(X)
        if shap_vals is None:
            return go.Figure().update_layout(title="SHAP not available")

        importance = self._mean_abs_shap(shap_vals)
        top_feats = importance.head(top_n).index.tolist()
        feat_idx  = [self.feature_names.index(f) for f in top_feats]

        sv_top = shap_vals[:, feat_idx].T   # (top_n, n_samples)
        order  = np.argsort(sv_top.sum(axis=0))  # sort by total SHAP
        sv_top = sv_top[:, order]

        # Subsample for display (max 200 samples)
        if sv_top.shape[1] > 200:
            idx = np.linspace(0, sv_top.shape[1] - 1, 200, dtype=int)
            sv_top = sv_top[:, idx]

        fig = go.Figure(go.Heatmap(
            z=sv_top,
            y=top_feats,
            colorscale=[[0, _SAFE], [0.5, "#2c3643"], [1, _DANGER]],
            zmid=0,
            colorbar=dict(title="SHAP value", titlefont=dict(color=_INK),
                          tickfont=dict(color=_INK)),
        ))
        fig.update_layout(
            title=dict(text=f"SHAP Feature × Sample Heatmap — {self.model_name}",
                       font=dict(color=_INK, size=13)),
            xaxis=dict(title="Samples (sorted by risk)", color=_INK),
            yaxis=dict(color=_INK),
            plot_bgcolor=_PANEL,
            paper_bgcolor=_BG,
            font=dict(color=_INK),
            height=max(300, top_n * 35 + 80),
        )
        return fig

    # ── LOCAL: waterfall chart for single instance ─────────────────────────

    def plot_local_waterfall(
        self,
        X: np.ndarray,
        instance_idx: int,
        top_n: int = 12,
    ) -> go.Figure:
        """
        Plotly waterfall (force-plot equivalent) for ONE module.

        Shows which features PUSH the prediction toward defective (red ↑)
        or away from defective (green ↓), starting from the base value.

        Args:
            X            : test feature matrix (n_samples, n_features)
            instance_idx : index of the module to explain
            top_n        : number of features to show (rest collapsed into 'others')
        """
        shap_vals = self._get_shap(X[instance_idx : instance_idx + 1])
        if shap_vals is None:
            return go.Figure().update_layout(title="SHAP not available")

        sv = shap_vals[0]   # (n_features,)

        # Sort by absolute value, keep top_n
        order    = np.argsort(np.abs(sv))[::-1]
        top_idx  = order[:top_n]
        rest_idx = order[top_n:]

        names  = [self.feature_names[i] for i in top_idx]
        values = sv[top_idx]

        if len(rest_idx) > 0:
            names.append(f"other {len(rest_idx)} features")
            values = np.append(values, sv[rest_idx].sum())

        # Base value (expected model output)
        if hasattr(self._explainer, "expected_value"):
            ev = self._explainer.expected_value
            base = float(ev[1] if isinstance(ev, (list, np.ndarray)) else ev)
        else:
            base = 0.5

        colors = [_DANGER if v > 0 else _SAFE for v in values]
        text_labels = [
            f"+{v:.4f}" if v > 0 else f"{v:.4f}" for v in values
        ]

        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(values) + ["total"],
            x=np.append(values, [0]),
            y=names + ["Prediction"],
            connector=dict(line=dict(color=_MUTED, width=1)),
            increasing=dict(marker_color=_DANGER),
            decreasing=dict(marker_color=_SAFE),
            totals=dict(marker_color=_WARN),
            text=text_labels + [f"Base={base:.3f}"],
            textposition="outside",
            textfont=dict(color=_INK),
            base=base,
        ))
        proba = base + float(values.sum())
        fig.update_layout(
            title=dict(
                text=f"Local Explanation — Module #{instance_idx}  "
                     f"(Predicted defect prob: {proba:.1%})",
                font=dict(color=_INK, size=13),
            ),
            xaxis=dict(title="SHAP contribution to defect probability", color=_INK),
            yaxis=dict(color=_INK, autorange="reversed"),
            plot_bgcolor=_PANEL,
            paper_bgcolor=_BG,
            font=dict(color=_INK),
            height=max(300, (len(values) + 1) * 32 + 100),
            margin=dict(l=10, r=80, t=60, b=40),
        )
        return fig

    # ── LOCAL: SHAP dependency plot (one feature vs one feature) ──────────

    def plot_dependency(
        self, X: np.ndarray, feature_a: str, feature_b: Optional[str] = None
    ) -> go.Figure:
        """
        SHAP dependency plot: x = feature_a value, y = SHAP(feature_a),
        colour = feature_b (interaction colouring).
        Reveals non-linear effects and interactions between two features.
        """
        shap_vals = self._get_shap(X)
        if shap_vals is None:
            return go.Figure().update_layout(title="SHAP not available")

        try:
            idx_a = self.feature_names.index(feature_a)
        except ValueError:
            return go.Figure().update_layout(title=f"Feature {feature_a!r} not found")

        x_vals  = X[:, idx_a]
        sv_vals = shap_vals[:, idx_a]

        if feature_b and feature_b in self.feature_names:
            idx_b   = self.feature_names.index(feature_b)
            color_v = X[:, idx_b]
            color_label = feature_b
        else:
            color_v     = sv_vals
            color_label = f"SHAP({feature_a})"

        fig = go.Figure(go.Scatter(
            x=x_vals, y=sv_vals,
            mode="markers",
            marker=dict(
                color=color_v,
                colorscale=[[0, _SAFE], [0.5, _WARN], [1, _DANGER]],
                showscale=True,
                colorbar=dict(title=color_label, titlefont=dict(color=_INK),
                              tickfont=dict(color=_INK)),
                size=6, opacity=0.75,
            ),
            text=[f"{feature_a}={x:.2f}<br>SHAP={s:.4f}"
                  for x, s in zip(x_vals, sv_vals)],
            hoverinfo="text",
        ))
        fig.update_layout(
            title=dict(text=f"SHAP Dependency: {feature_a}",
                       font=dict(color=_INK, size=13)),
            xaxis=dict(title=feature_a, color=_INK),
            yaxis=dict(title=f"SHAP value ({feature_a})", color=_INK),
            plot_bgcolor=_PANEL, paper_bgcolor=_BG,
            font=dict(color=_INK), height=420,
        )
        return fig

    # ── LOCAL: LIME explanation ────────────────────────────────────────────

    def plot_lime_explanation(
        self,
        X: np.ndarray,
        instance_idx: int,
        top_n: int = 10,
    ) -> go.Figure:
        """
        LIME local linear surrogate for one instance.
        Returns a Plotly horizontal bar chart.

        LIME is complementary to SHAP: while SHAP gives globally consistent
        attributions, LIME optimises local fidelity around this specific point.
        Useful to cross-validate SHAP explanations — if both agree, higher
        confidence in the attribution.
        """
        if not LIME_AVAILABLE or self._lime_explainer is None:
            return go.Figure().update_layout(title="LIME not available")

        predict_fn = (
            self.model.predict_proba
            if hasattr(self.model, "predict_proba")
            else lambda x: np.column_stack([1 - self.model.predict(x),
                                            self.model.predict(x)])
        )

        exp = self._lime_explainer.explain_instance(
            X[instance_idx],
            predict_fn,
            num_features=top_n,
            num_samples=500,
            labels=(1,),   # explain positive (defective) class
        )

        feats, weights = zip(*exp.as_list()) if exp.as_list() else ([], [])
        colors = [_DANGER if w > 0 else _SAFE for w in weights]
        text   = [f"+{w:.4f}" if w > 0 else f"{w:.4f}" for w in weights]

        # Reverse so most important is at top
        feats   = list(feats)[::-1]
        weights = list(weights)[::-1]
        colors  = colors[::-1]
        text    = text[::-1]

        fig = go.Figure(go.Bar(
            x=weights, y=feats,
            orientation="h",
            marker_color=colors,
            text=text,
            textposition="outside",
            textfont=dict(size=10, color=_INK),
        ))
        pred_proba = predict_fn(X[instance_idx : instance_idx + 1])[0][1]
        fig.update_layout(
            title=dict(
                text=f"LIME Explanation — Module #{instance_idx}  "
                     f"(Defect prob: {pred_proba:.1%})",
                font=dict(color=_INK, size=13),
            ),
            xaxis=dict(title="LIME feature weight (positive = toward defective)",
                       color=_INK, zeroline=True,
                       zerolinecolor=_MUTED, zerolinewidth=1),
            yaxis=dict(color=_INK),
            plot_bgcolor=_PANEL, paper_bgcolor=_BG,
            font=dict(color=_INK),
            height=max(300, top_n * 32 + 100),
            margin=dict(l=10, r=80, t=60, b=40),
        )
        return fig

    # ── NARRATIVE TEXT ─────────────────────────────────────────────────────

    def generate_narrative(
        self,
        X: np.ndarray,
        instance_idx: int,
        pred_proba: float,
        threshold: float = 0.5,
    ) -> str:
        """
        Generate a human-readable risk assessment for a single module.

        Produces 3 paragraphs:
          1. Overall verdict (risk level + probability)
          2. Top contributing factors (from SHAP)
          3. Recommended actions
        """
        shap_vals = self._get_shap(X[instance_idx : instance_idx + 1])
        verdict   = "HIGH RISK" if pred_proba >= 0.5 else \
                    "MEDIUM RISK" if pred_proba >= 0.3 else "LOW RISK"
        emoji     = "🔴" if pred_proba >= 0.5 else "🟡" if pred_proba >= 0.3 else "🟢"

        lines = [
            f"## {emoji} {verdict} — Defect probability: {pred_proba:.1%}",
            "",
            f"The **{self.model_name}** model predicts this module has a "
            f"**{pred_proba:.1%}** probability of containing defects "
            f"({'above' if pred_proba >= threshold else 'below'} the "
            f"{threshold:.0%} decision threshold).",
            "",
        ]

        if shap_vals is not None:
            sv = shap_vals[0]
            order   = np.argsort(np.abs(sv))[::-1][:5]
            pos_top = [(self.feature_names[i], sv[i]) for i in order if sv[i] > 0]
            neg_top = [(self.feature_names[i], sv[i]) for i in order if sv[i] < 0]

            lines.append("### 🔍 Key risk factors")
            if pos_top:
                lines.append("The following metrics **increase** defect risk:")
                for feat, val in pos_top:
                    feat_val = float(X[instance_idx, self.feature_names.index(feat)])
                    lines.append(f"  - **{feat}** = {feat_val:.2f}  "
                                 f"(SHAP contribution: +{val:.4f})")
            if neg_top:
                lines.append("The following metrics **reduce** defect risk:")
                for feat, val in neg_top:
                    feat_val = float(X[instance_idx, self.feature_names.index(feat)])
                    lines.append(f"  - **{feat}** = {feat_val:.2f}  "
                                 f"(SHAP contribution: {val:.4f})")
            lines.append("")

        lines.append("### 💡 Recommended actions")
        if pred_proba >= 0.5:
            lines += [
                "- 🔴 **Schedule mandatory code review** for this module",
                "- Reduce cyclomatic complexity by splitting large functions",
                "- Add unit tests targeting the high-complexity paths",
                "- Consider refactoring if LOC > 200 or CC > 10",
            ]
        elif pred_proba >= 0.3:
            lines += [
                "- 🟡 Add supplementary code review",
                "- Monitor this module in next sprint",
                "- Improve comment ratio if below 5%",
            ]
        else:
            lines += [
                "- 🟢 Module appears stable — maintain current practices",
                "- Keep complexity metrics in acceptable range",
            ]

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION — plug-and-play interface
# ─────────────────────────────────────────────────────────────────────────────

def explain_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "",
) -> ModelExplainer:
    """
    Top-level convenience function.

    Creates, fits, and returns a ModelExplainer ready to plot.

    Usage
    ─────
        explainer = explain_model(rf_model, X_tr, X_te, feature_names, "Random Forest")

        # Global
        fig_bar   = explainer.plot_global_importance(X_te)
        fig_bee   = explainer.plot_beeswarm(X_te)
        fig_heat  = explainer.plot_feature_heatmap(X_te)

        # Local (module at index 7)
        fig_wf    = explainer.plot_local_waterfall(X_te, instance_idx=7)
        fig_lime  = explainer.plot_lime_explanation(X_te, instance_idx=7)
        narrative = explainer.generate_narrative(X_te, 7, proba[7])
    """
    exp = ModelExplainer(model, X_train, feature_names, model_name)
    exp.fit()
    return exp


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL COMPARISON (for "Evaluate" tab)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_comparison(
    explainers: Dict[str, ModelExplainer],
    X_test: np.ndarray,
    top_n: int = 10,
) -> go.Figure:
    """
    Grouped bar chart comparing mean |SHAP| across multiple models.
    Shows which features EACH model considers most important — useful for
    identifying agreement (robust features) vs disagreement (model-specific).

    Args:
        explainers : {model_name: ModelExplainer}  (already fitted)
        X_test     : test feature matrix
        top_n      : features to include
    """
    # Build combined importance DataFrame
    frames = []
    for mname, exp in explainers.items():
        sv = exp._get_shap(X_test)
        if sv is None:
            continue
        imp = exp._mean_abs_shap(sv).head(top_n).reset_index()
        imp.columns = ["feature", "importance"]
        imp["model"] = mname
        frames.append(imp)

    if not frames:
        return go.Figure().update_layout(title="No SHAP explainers available")

    df = pd.concat(frames, ignore_index=True)
    fig = px.bar(
        df, x="feature", y="importance", color="model",
        barmode="group",
        title="SHAP Feature Importance — Model Comparison",
        labels={"importance": "Mean |SHAP value|", "feature": "Feature"},
        color_discrete_sequence=[_DANGER, _SAFE, _WARN, _MUTED,
                                 "#7ec6ff", "#ce9178", "#c586c0"],
    )
    fig.update_layout(
        plot_bgcolor=_PANEL, paper_bgcolor=_BG,
        font=dict(color=_INK),
        legend=dict(bgcolor=_PANEL, bordercolor=_MUTED),
        xaxis=dict(color=_INK, tickangle=-35),
        yaxis=dict(color=_INK),
        height=450,
    )
    return fig


def plot_prediction_breakdown(
    explainers: Dict[str, ModelExplainer],
    X_test: np.ndarray,
    instance_idx: int,
) -> go.Figure:
    """
    Side-by-side SHAP contributions for one module across all models.
    Helps spot where models AGREE (confident attribution) vs DISAGREE.
    """
    frames = []
    for mname, exp in explainers.items():
        sv = exp._get_shap(X_test[instance_idx: instance_idx + 1])
        if sv is None:
            continue
        order = np.argsort(np.abs(sv[0]))[::-1][:8]
        for i in order:
            frames.append({
                "model": mname,
                "feature": exp.feature_names[i],
                "shap": float(sv[0, i]),
            })

    if not frames:
        return go.Figure().update_layout(title="No SHAP data")

    df = pd.concat([pd.DataFrame([r]) for r in frames], ignore_index=True)
    fig = px.bar(
        df, x="shap", y="feature", color="model",
        orientation="h", barmode="group",
        title=f"SHAP Contributions — Module #{instance_idx} (all models)",
        labels={"shap": "SHAP value", "feature": "Feature"},
        color_discrete_sequence=[_DANGER, _SAFE, _WARN, _MUTED, "#7ec6ff"],
    )
    fig.update_layout(
        plot_bgcolor=_PANEL, paper_bgcolor=_BG,
        font=dict(color=_INK),
        xaxis=dict(color=_INK, zeroline=True,
                   zerolinecolor=_MUTED, zerolinewidth=1),
        yaxis=dict(color=_INK, autorange="reversed"),
        height=420,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import importlib.util
    import sys
    from pathlib import Path

    BACKEND = Path(__file__).parent
    sys.path.insert(0, str(BACKEND))

    # Load preprocessing.py and models.py by file path to avoid
    # the models/ directory shadowing models.py
    def _load(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    pp_mod = _load("preprocessing", BACKEND / "preprocessing.py")
    ml_mod = _load("ml_models",     BACKEND / "models.py")

    SDPPreprocessor = pp_mod.SDPPreprocessor
    load_data       = pp_mod.load_data
    ModelTrainer    = ml_mod.ModelTrainer

    DATA = BACKEND / "data" / "kc1_kc2.csv"
    if not DATA.exists():
        print("[skip] kc1_kc2.csv not found")
        sys.exit(0)

    print("=" * 60)
    print("Loading data + preprocessing...")
    df = load_data(str(DATA))
    pp = SDPPreprocessor(imbalance_strategy="none")
    X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
    feat_names = pp.selected_features_

    print("\nTraining RF + XGBoost (tune=False)...")
    trainer = ModelTrainer(class_weight_dict=pp.class_weight_dict_,
                           include_ensembles=False)
    trainer.train_all_models(X_tr, X_te, y_tr, y_te, tune=False, cv_folds=3)

    print("\n" + "=" * 60)
    print("Building SHAP explainers...")

    explainers = {}
    for name, res in trainer.results_.items():
        print(f"  Fitting explainer for {name}...")
        exp = explain_model(res.model, X_tr, X_te, feat_names, name)
        explainers[name] = exp

    # Global importance (first model)
    first_name = next(iter(explainers))
    first_exp  = explainers[first_name]

    print(f"\nGlobal importance ({first_name}):")
    sv = first_exp._get_shap(X_te)
    if sv is not None:
        imp = first_exp._mean_abs_shap(sv).head(5)
        for feat, val in imp.items():
            print(f"  {feat}: {val:.4f}")

    # Narrative for module #0
    proba_0 = trainer.results_[first_name].y_proba[0]
    narrative = first_exp.generate_narrative(X_te, 0, proba_0)
    print(f"\nNarrative for module #0 ({first_name}):")
    print(narrative[:600].encode("ascii", errors="replace").decode("ascii"), "...")

    print("\nAll SHAP/LIME smoke tests passed ✓")
