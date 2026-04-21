"""Script to patch _tab_prediction and insert helper functions."""
import pathlib

src_path = pathlib.Path("frontend/app.py")
src = src_path.read_text(encoding="utf-8")

# ── define the new block ────────────────────────────────────────────────────
NEW_BLOCK = '''# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI RISK ASSESSMENT (Groq-powered)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ai_reviewer():
    """Import ai_reviewer module via importlib."""
    import importlib.util, pathlib as _pl
    spec = importlib.util.spec_from_file_location(
        "ai_reviewer", _pl.Path(ROOT) / "backend" / "ai_reviewer.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _risk_gauge_html(score: float) -> str:
    """Compact risk score gauge in HTML/CSS."""
    pct   = score * 100
    color = DANGER if score >= 0.67 else (WARN if score >= 0.33 else SAFE)
    label = "HIGH RISK" if score >= 0.67 else ("MEDIUM RISK" if score >= 0.33 else "LOW RISK")
    return (
        f\'<div style="text-align:center;background:{PANEL};border:1px solid {LINE};\'\n
        f\'border-radius:14px;padding:20px 10px;margin-bottom:8px">\'\n
        f\'<div style="font-size:2.6rem;font-weight:800;color:{color};line-height:1">{pct:.0f}%</div>\'\n
        f\'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;\'\n
        f\'color:{color};margin-top:4px;font-weight:700">{label}</div>\'\n
        f\'<div style="background:{LINE};border-radius:4px;height:6px;margin:10px 0 4px">\'\n
        f\'<div style="background:{color};height:6px;border-radius:4px;width:{pct:.0f}%"></div>\'\n
        f\'</div></div>\'
    )


def _tab_prediction() -> None:
    files    = st.session_state.files
    analyzed = st.session_state.analyzed

    if not analyzed or not files:
        st.markdown(
            f\'\'\'<div style="background:{PANEL};border:1px dashed {LINE};border-radius:14px;
                    padding:48px;text-align:center">
                <p style="font-size:2rem;margin:0">&#x1f916;</p>
                <p style="font-size:1.05rem;font-weight:700;margin:8px 0 4px">AI Risk Assessment</p>
                <p style="font-size:.86rem;color:{MUTED};margin:0">
                    Upload &amp; analyze source files in the <b>Workspace</b> tab first.</p>
            </div>\'\'\', unsafe_allow_html=True)
        return

    try:
        ai_mod = _load_ai_reviewer()
    except Exception as e:
        st.error(f"Cannot load ai_reviewer module: {e}")
        return

    left, right = st.columns([1.1, 2.9], gap="medium")

    with left:
        st.markdown(\'<p class="ds-section-header">Groq AI Configuration</p>\',
                    unsafe_allow_html=True)

        api_key = st.text_input(
            "Groq API Key", type="password", placeholder="gsk_...",
            key="groq_api_key", help="Get your free key at console.groq.com",
        )

        model_id = st.selectbox(
            "Model", options=list(ai_mod.GROQ_MODELS.keys()),
            format_func=lambda m: ai_mod.GROQ_MODELS[m], key="groq_model",
        )

        lang = st.radio(
            "Language", ["vi", "en"],
            format_func=lambda x: "Tieng Viet" if x == "vi" else "English",
            horizontal=True, key="groq_lang",
        )

        if st.button("Validate API Key", use_container_width=True):
            if not api_key:
                st.warning("Nhap API key truoc.")
            else:
                with st.spinner("Kiem tra..."):
                    ok, msg = ai_mod.validate_api_key(api_key)
                if ok:
                    st.success("API Key hop le")
                    st.session_state["groq_key_valid"] = True
                else:
                    st.error(msg)
                    st.session_state["groq_key_valid"] = False

        if st.session_state.get("groq_key_valid"):
            st.markdown(
                f\'<div style="background:{SAFE}22;border:1px solid {SAFE}55;\'\n
                f\'border-radius:6px;padding:4px 10px;font-size:.78rem;color:{SAFE};\'\n
                f\'text-align:center">API Key Valid</div>\',
                unsafe_allow_html=True,
            )

        st.divider()
        st.markdown(\'<p class="ds-section-header">Analysis Mode</p>\', unsafe_allow_html=True)
        mode = st.radio(
            "Mode", ["single_file", "workspace"],
            format_func=lambda x: (
                "Single File Deep Dive" if x == "single_file" else "Workspace Overview"
            ),
            label_visibility="collapsed", key="ai_mode",
        )

        if mode == "single_file":
            st.divider()
            st.markdown(\'<p class="ds-section-header">Select File</p>\', unsafe_allow_html=True)
            sorted_files = sorted(files, key=lambda f: f.get("risk_score", 0), reverse=True)
            file_labels  = [
                f"{f[\'lang\'][0]} {f.get(\'fname\', f[\'name\'])} -- {f[\'risk_score\']*100:.0f}%"
                for f in sorted_files
            ]
            sel_idx = st.selectbox(
                "File", range(len(sorted_files)),
                format_func=lambda i: file_labels[i], key="ai_file_select",
            )
            selected_file = sorted_files[sel_idx]
            st.markdown(_risk_gauge_html(selected_file["risk_score"]), unsafe_allow_html=True)

            m = selected_file.get("metrics", {})
            c1, c2 = st.columns(2)
            c1.metric("LOC",       m.get("loc",   "--"))
            c2.metric("CC",        m.get("cc",    "--"))
            c1.metric("Functions", m.get("funcs", "--"))
            c2.metric("Comment%",
                      f"{m.get(\'comment_ratio\', 0)*100:.1f}%"
                      if m.get("comment_ratio") else "--")

            ctx_preview = ai_mod.build_file_context(selected_file)
            n_v, n_h = len(ctx_preview.violations), len(ctx_preview.hotspots)
            if n_v or n_h:
                st.markdown(
                    f\'<div style="background:{DANGER}18;border:1px solid {DANGER}44;\'\n
                    f\'border-radius:8px;padding:8px 12px;font-size:.8rem;margin-top:4px">\'\n
                    f\'{n_v} metric violations &middot; {n_h} hotspots</div>\',
                    unsafe_allow_html=True,
                )
            run_label = "Analyze This File"
        else:
            selected_file = None
            high = sum(1 for f in files if f.get("risk_score", 0) >= 0.67)
            med  = sum(1 for f in files if 0.33 <= f.get("risk_score", 0) < 0.67)
            st.metric("Files to review", f"{high} HIGH + {med} MED")
            run_label = "Analyze Workspace"

        st.divider()
        run_btn = st.button(run_label, type="primary",
                            use_container_width=True, disabled=not api_key)
        st.caption("Powered by Groq -- ultra-fast inference")

    with right:
        st.markdown(\'<p class="ds-section-header">AI Analysis Report</p>\',
                    unsafe_allow_html=True)

        cache_key = (
            f"ai_report_{selected_file[\'name\']}" if selected_file
            else "ai_report_workspace"
        )

        if not run_btn:
            cached = st.session_state.get(cache_key)
            if cached:
                st.markdown(cached)
                if st.button("Re-analyze", key="reanalyze_cached"):
                    del st.session_state[cache_key]
                    st.rerun()
            else:
                st.markdown(
                    f\'<div style="background:{PANEL};border:1px dashed {LINE};\'\n
                    f\'border-radius:12px;padding:48px;text-align:center">\'\n
                    f\'<p style="font-size:.9rem;color:{MUTED};margin:0">\'\n
                    f\'Configure your API key and click <b>{run_label}</b>.\'\n
                    f\'</p></div>\', unsafe_allow_html=True,
                )
            return

        if not api_key:
            st.error("Enter your Groq API key first.")
            return

        try:
            reviewer = ai_mod.GroqReviewer(api_key=api_key.strip(), model=model_id)
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            return

        report_box  = st.empty()
        full_report = ""
        token_box   = st.empty()
        model_label = ai_mod.GROQ_MODELS.get(model_id, model_id)

        with st.status(f"Analyzing with {model_label}...", expanded=True) as status:
            try:
                if mode == "single_file" and selected_file:
                    ctx    = ai_mod.build_file_context(selected_file)
                    stream = reviewer.review_stream(ctx, language=lang)
                    st.write(f"Analyzing: {selected_file[\'name\']}")
                    st.write(f"Risk: {selected_file[\'risk_score\']:.1%} | "
                             f"Violations: {len(ctx.violations)} | "
                             f"Hotspots: {len(ctx.hotspots)}")
                else:
                    stream = reviewer.review_workspace(files, language=lang)
                    st.write(f"Analyzing {len(files)} files...")

                token_count = 0
                for chunk in stream:
                    full_report += chunk
                    token_count += len(chunk.split())
                    report_box.markdown(full_report + " |")
                    token_box.caption(f"{token_count} tokens streamed...")

                report_box.markdown(full_report)
                token_box.empty()
                status.update(label="Analysis complete!", state="complete")
                st.session_state[cache_key] = full_report

            except Exception as e:
                status.update(label="Error", state="error")
                err = str(e)
                if "401" in err:
                    st.error("Invalid API key — check at console.groq.com")
                elif "429" in err:
                    st.error("Rate limit hit. Wait a moment and retry.")
                else:
                    st.error(f"Groq API error: {err}")
                return

        st.divider()
        ca, cb = st.columns(2)
        with ca:
            fname = (
                selected_file["fname"].replace(".", "_") + ".md"
                if selected_file else "workspace_risk_report.md"
            )
            st.download_button(
                "Download Report (.md)", data=full_report,
                file_name=fname, mime="text/markdown", use_container_width=True,
            )
        with cb:
            if st.button("Re-analyze", use_container_width=True, key="reanalyze_after"):
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
                st.rerun()
'''

# ── find start / end markers ────────────────────────────────────────────────
START_MARKER = "# TAB 5"
END_MARKER   = "\n# ─────────────────────────────────────────────────────────────────────────────\n# ANALYSIS RUNNER"

start_idx = src.find(START_MARKER)
end_idx   = src.find(END_MARKER, start_idx)

assert start_idx != -1, "START_MARKER not found"
assert end_idx   != -1, f"END_MARKER not found (searched from {start_idx})"

# Walk back to include the comment dashes line before START_MARKER
dash_idx = src.rfind("\n# ───", 0, start_idx)

new_src = src[:dash_idx] + "\n\n" + NEW_BLOCK + "\n" + src[end_idx:]

src_path.write_text(new_src, encoding="utf-8")
print(f"Done. File length: {len(new_src)} chars")

# quick syntax check
import ast
ast.parse(new_src)
print("Syntax OK")
