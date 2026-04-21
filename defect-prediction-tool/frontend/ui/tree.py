"""File-tree helpers for DefectSight explorer pane."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st


def build_tree(files_list: list[dict[str, Any]]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for item in files_list:
        normalized = item["name"].replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        current = tree
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = item
    return tree


def _node_key(parent: str, node_name: str) -> str:
    raw = f"{parent}/{node_name}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]


def _file_matches(item: dict[str, Any], query: str, risk_filter: str) -> bool:
    if query and query.lower() not in item["name"].lower():
        return False
    if risk_filter == "All":
        return True
    score = float(item.get("risk_score", 0))
    if risk_filter == "High":
        return score >= 0.5
    if risk_filter == "Medium":
        return 0.3 <= score < 0.5
    if risk_filter == "Low":
        return score < 0.3
    return True


def node_has_match(node: dict[str, Any], query: str, risk_filter: str) -> bool:
    for _, value in node.items():
        is_file = isinstance(value, dict) and "content" in value
        if is_file:
            if _file_matches(value, query, risk_filter):
                return True
        elif node_has_match(value, query, risk_filter):
            return True
    return False


def _risk_dot(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 0.5:
        return "🔴"
    if score >= 0.3:
        return "🟡"
    return "🟢"


def _render_indented_button(label: str, key: str, depth: int) -> bool:
    if depth <= 0:
        return st.button(label, key=key, use_container_width=True)

    # Indent nested nodes by adding a spacer column proportional to depth.
    spacer, content = st.columns([max(depth * 0.8, 0.8), 24], gap="small")
    with content:
        return st.button(label, key=key, use_container_width=True)


def render_tree(
    node,
    active_file,
    analyzed,
    query="",
    risk_filter="All",
    parent_path="root",
    depth=0,
):
    items = sorted(
        node.items(),
        key=lambda pair: (
            isinstance(pair[1], dict) and "content" in pair[1],
            pair[0].lower(),
        ),
    )

    for name, value in items:
        is_file = isinstance(value, dict) and "content" in value

        # ===== FILE =====
        if is_file:
            if not _file_matches(value, query, risk_filter):
                continue

            is_active = value["name"] == active_file
            score = value.get("risk_score") if analyzed else None
            risk = _risk_dot(score) if analyzed else ""

            if _render_indented_button(
                f"{value['lang'][0]} {name} {risk}",
                key=f"file_{_node_key(parent_path, value['name'])}",
                depth=depth,
            ):
                st.session_state.active_file = value["name"]
                st.rerun()
            continue

        # ===== FOLDER =====
        if not node_has_match(value, query, risk_filter):
            continue

        dir_key = _node_key(parent_path, name)

        if "tree_expanded" not in st.session_state:
            st.session_state.tree_expanded = {}

        expanded_map = st.session_state.tree_expanded
        default_open = depth < 1
        is_open = expanded_map.get(dir_key, default_open)

        icon = "▾" if is_open else "▸"

        if _render_indented_button(
            f"{icon} {name}",
            key=f"dir_{dir_key}",
            depth=depth,
        ):
            expanded_map[dir_key] = not is_open
            st.session_state.tree_expanded = expanded_map
            st.rerun()

        if is_open:
            render_tree(
                value,
                active_file,
                analyzed,
                query,
                risk_filter,
                f"{parent_path}/{name}",
                depth + 1,
            )

