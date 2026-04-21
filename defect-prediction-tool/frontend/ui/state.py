"""Session state helpers for DefectSight UI."""

import streamlit as st

DEFAULTS = {
    # Workspace
    "files":            [],
    "analysis":         None,
    "analyzed":         False,
    "logs":             [],
    "active_file":      None,
    "tree_query":       "",
    "tree_risk_filter": "All",
    "tree_expanded":    {},
    "model_choice":     "All Models",
    # Training
    "trainer":          None,
    "pp":               None,
    "X_train":          None,
    "X_test":           None,
    "y_train":          None,
    "y_test":           None,
    "feature_names":    None,
    "train_logs":       [],
    "trained":          False,
}


def ensure_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis_state() -> None:
    """Reset only workspace analysis state (keep training state)."""
    workspace_keys = {
        "files", "analysis", "analyzed", "logs", "active_file",
        "tree_query", "tree_risk_filter", "tree_expanded",
    }
    for key in workspace_keys:
        st.session_state[key] = DEFAULTS[key]


def reset_all() -> None:
    """Full reset of all state."""
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
