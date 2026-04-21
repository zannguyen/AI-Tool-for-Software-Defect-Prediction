"""
AI-Powered Code Risk Reviewer — Groq backend
=============================================
Nhận kết quả phân tích từ DefectSight (metrics, risk score, line-level issues)
và gọi Groq LLM để sinh báo cáo đánh giá rủi ro + hướng cải thiện chi tiết.

Supported Groq models (free tier, tốc độ cao):
  • llama-3.3-70b-versatile   — BEST: reasoning sâu nhất
  • llama-3.1-8b-instant      — FASTEST: phản hồi < 1s
  • mixtral-8x7b-32768        — context window lớn (32k tokens)
  • gemma2-9b-it              — Google's Gemma 2

Prompt Strategy
───────────────
Dùng System Prompt kiểu "Senior Code Reviewer" chuyên SDP:
  1. Phân tích từng metric vi phạm ngưỡng
  2. Đánh giá line-level hotspots (eval, while True, ...)
  3. Phân loại rủi ro theo SEVERITY (Critical / High / Medium / Low)
  4. Đề xuất mitigation cụ thể, actionable, có code example
  5. Ước lượng effort (giờ/ngày) cho từng action item
  6. Tóm tắt executive summary cho PM/tech lead
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List, Optional

# ── Groq thresholds ────────────────────────────────────────────────────────
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Best quality)",
    "llama-3.1-8b-instant":    "Llama 3.1 8B (Fastest)",
    "mixtral-8x7b-32768":      "Mixtral 8x7B (Large context)",
    "gemma2-9b-it":            "Gemma 2 9B (Google)",
}

# ── Metric thresholds (NASA MDP + style guide consensus) ──────────────────
METRIC_THRESHOLDS = {
    "loc":           {"warn": 150, "critical": 300,  "unit": "lines",
                      "name": "Lines of Code (LOC)"},
    "cc":            {"warn": 7,   "critical": 15,   "unit": "",
                      "name": "Cyclomatic Complexity"},
    "funcs":         {"warn": 10,  "critical": 20,   "unit": "functions",
                      "name": "Function Count"},
    "comment_ratio": {"warn": 0.05, "critical": 0.02, "unit": "%",
                      "name": "Comment Ratio", "inverse": True},
    "classes":       {"warn": 5,   "critical": 10,   "unit": "classes",
                      "name": "Class Count"},
}

DANGEROUS_PATTERNS = {
    "eval(":        ("CRITICAL", "Arbitrary code execution — security vulnerability"),
    "exec(":        ("CRITICAL", "Arbitrary code execution — security vulnerability"),
    "while True":   ("HIGH",     "Infinite loop — potential hang/DoS"),
    "except:":      ("HIGH",     "Bare except catches everything including SystemExit"),
    "except Exception:": ("MEDIUM", "Overly broad exception handling masks bugs"),
    "os.system(":   ("HIGH",     "Shell injection risk — use subprocess instead"),
    "pickle.load":  ("HIGH",     "Deserialization of untrusted data — security risk"),
    "TODO":         ("LOW",      "Unfinished implementation"),
    "FIXME":        ("MEDIUM",   "Known bug not yet fixed"),
    "HACK":         ("MEDIUM",   "Workaround that should be properly implemented"),
    "global ":      ("LOW",      "Global variable usage increases coupling"),
    "__import__(":  ("MEDIUM",   "Dynamic import — hard to analyze statically"),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricViolation:
    metric:    str
    value:     float
    threshold: float
    severity:  str      # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    message:   str


@dataclass
class LineHotspot:
    line_no:  int
    code:     str
    pattern:  str
    severity: str
    reason:   str


@dataclass
class FileRiskContext:
    """All risk info for one file, prepared for the LLM prompt."""
    file_path:       str
    language:        str
    risk_score:      float           # 0–1
    risk_level:      str             # HIGH / MEDIUM / LOW
    metrics:         dict
    violations:      List[MetricViolation] = field(default_factory=list)
    hotspots:        List[LineHotspot]     = field(default_factory=list)
    code_snippet:    str = ""        # first 80 lines of the file


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_file_context(file_entry: dict) -> FileRiskContext:
    """
    Chuyển đổi một file entry từ session_state.files sang FileRiskContext
    đầy đủ thông tin cho LLM prompt.
    """
    metrics    = file_entry.get("metrics", {})
    risk_score = file_entry.get("risk_score", 0.0)
    risk_level = ("HIGH" if risk_score >= 0.67
                  else "MEDIUM" if risk_score >= 0.33
                  else "LOW")
    lang       = file_entry.get("lang", ("", "", "Unknown"))[2]
    content    = file_entry.get("content", "")

    # ── metric violations ─────────────────────────────────────────────────
    violations: List[MetricViolation] = []
    for key, cfg in METRIC_THRESHOLDS.items():
        val = metrics.get(key)
        if val is None:
            continue
        inverse = cfg.get("inverse", False)

        if inverse:
            # lower is worse (e.g. comment_ratio)
            if val <= cfg["critical"]:
                sev, thr = "CRITICAL", cfg["critical"]
            elif val <= cfg["warn"]:
                sev, thr = "MEDIUM", cfg["warn"]
            else:
                continue
            msg = (f"{cfg['name']} is {val:.1%} — below safe minimum "
                   f"({thr:.0%}). Insufficient documentation coverage.")
        else:
            if val >= cfg["critical"]:
                sev, thr = "CRITICAL", cfg["critical"]
            elif val >= cfg["warn"]:
                sev, thr = "HIGH", cfg["warn"]
            else:
                continue
            msg = (f"{cfg['name']} is {val} {cfg['unit']} — exceeds "
                   f"threshold ({thr}). Module is too complex/large.")

        violations.append(MetricViolation(key, val, thr, sev, msg))

    # ── line-level hotspots ───────────────────────────────────────────────
    hotspots: List[LineHotspot] = []
    lines = content.splitlines()
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        for pattern, (sev, reason) in DANGEROUS_PATTERNS.items():
            if pattern in stripped:
                hotspots.append(LineHotspot(
                    line_no=line_no,
                    code=stripped[:120],
                    pattern=pattern,
                    severity=sev,
                    reason=reason,
                ))
                break  # one pattern per line

    # ── code snippet (first 80 lines, trimmed) ────────────────────────────
    snippet = "\n".join(lines[:80])
    if len(lines) > 80:
        snippet += f"\n... ({len(lines) - 80} more lines)"

    return FileRiskContext(
        file_path=file_entry.get("name", "unknown"),
        language=lang,
        risk_score=risk_score,
        risk_level=risk_level,
        metrics=metrics,
        violations=violations,
        hotspots=hotspots,
        code_snippet=snippet,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Senior Software Engineer and Code Quality Expert specializing in Software Defect Prediction (SDP) and static code analysis.

Your role is to:
1. Analyze code metrics and identify defect-prone patterns
2. Provide ACTIONABLE, SPECIFIC improvement recommendations
3. Prioritize issues by severity and business impact
4. Estimate remediation effort in hours
5. Write concise, technical reports suitable for code review

Always structure your response in the exact Markdown format requested.
Be concrete: reference specific line numbers, metric values, and code patterns.
Avoid generic advice — every recommendation must be traceable to a specific finding."""


def build_analysis_prompt(ctx: FileRiskContext, language: str = "vi") -> str:
    """
    Xây dựng user prompt chi tiết cho LLM.

    language: "vi" = Tiếng Việt (default), "en" = English
    """
    # ── violations section ────────────────────────────────────────────────
    viol_lines = []
    if ctx.violations:
        for v in sorted(ctx.violations, key=lambda x: x.severity):
            viol_lines.append(
                f"  - [{v.severity}] {v.message}"
            )
    else:
        viol_lines.append("  - No metric threshold violations detected.")

    # ── hotspots section ──────────────────────────────────────────────────
    spot_lines = []
    if ctx.hotspots:
        for h in ctx.hotspots[:15]:  # cap at 15 to keep token count clean
            spot_lines.append(
                f"  - Line {h.line_no:4d} [{h.severity}]: `{h.code[:80]}` "
                f"→ {h.reason}"
            )
    else:
        spot_lines.append("  - No dangerous code patterns detected.")

    # ── metrics table ─────────────────────────────────────────────────────
    m = ctx.metrics
    metrics_block = f"""| Metric               | Value        |
|----------------------|--------------|
| Lines of Code (LOC)  | {m.get('loc', 'N/A')}  |
| Cyclomatic Complexity| {m.get('cc', 'N/A')}   |
| Function Count       | {m.get('funcs', 'N/A')} |
| Comment Ratio        | {m.get('comment_ratio', 0)*100 if m.get('comment_ratio') else 'N/A'}%|
| Class Count          | {m.get('classes', 'N/A')} |"""

    lang_note = ("Trả lời HOÀN TOÀN bằng Tiếng Việt."
                 if language == "vi"
                 else "Answer in English.")

    prompt = f"""# Code Risk Analysis Request

{lang_note}

## File Information
- **File:** `{ctx.file_path}`
- **Language:** {ctx.language}
- **DefectSight Risk Score:** {ctx.risk_score:.1%} ({ctx.risk_level} RISK)

## Source Metrics
{metrics_block}

## Metric Threshold Violations
{chr(10).join(viol_lines)}

## Line-Level Code Hotspots
{chr(10).join(spot_lines)}

## Source Code Preview (first 80 lines)
```{ctx.language.lower()}
{ctx.code_snippet}
```

---

## Required Analysis Output

Please provide a **complete defect risk assessment report** with the following sections:

### 1. Executive Summary (2–3 sentences)
> Risk level verdict + most critical finding + overall recommendation.

### 2. Detailed Risk Findings
For each violation and hotspot found:
- **Finding:** Description of the issue
- **Evidence:** Metric value / line number / code snippet
- **Severity:** CRITICAL | HIGH | MEDIUM | LOW
- **Impact:** What bugs/failures this can cause

### 3. Prioritized Action Plan
List action items ordered by priority (P1 → P3):

| Priority | Action | Effort (hours) | Expected Improvement |
|----------|--------|---------------|---------------------|
| P1 | ... | ... | ... |

### 4. Code Improvement Examples
For the top 2-3 most critical issues, provide **before/after code snippets**.

### 5. Refactoring Roadmap
A phased approach:
- **Phase 1 (Immediate — this sprint):** ...
- **Phase 2 (Short-term — next sprint):** ...
- **Phase 3 (Long-term — next quarter):** ...

### 6. Risk Score Projection
If all P1 actions are completed, estimate the new risk score and explain why.
"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class GroqReviewer:
    """
    Gọi Groq API để phân tích rủi ro code.

    Usage:
        reviewer = GroqReviewer(api_key="gsk_...", model="llama-3.3-70b-versatile")
        ctx = build_file_context(file_entry)

        # Streaming (recommended for Streamlit)
        for chunk in reviewer.review_stream(ctx, language="vi"):
            print(chunk, end="", flush=True)

        # Non-streaming
        report = reviewer.review(ctx, language="vi")
    """

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Install groq: `pip install groq`")
        self.client = Groq(api_key=api_key)
        self.model  = model

    def review_stream(
        self,
        ctx: FileRiskContext,
        language: str = "vi",
        max_tokens: int = 4096,
    ) -> Generator[str, None, None]:
        """
        Stream LLM response token-by-token.
        Dùng trong Streamlit:
            with st.empty() as box:
                full = ""
                for chunk in reviewer.review_stream(ctx):
                    full += chunk
                    box.markdown(full)
        """
        prompt = build_analysis_prompt(ctx, language)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,      # lower = more consistent/factual
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def review(
        self,
        ctx: FileRiskContext,
        language: str = "vi",
        max_tokens: int = 4096,
    ) -> str:
        """Non-streaming version — collects full response."""
        return "".join(self.review_stream(ctx, language, max_tokens))

    def review_workspace(
        self,
        files: list[dict],
        language: str = "vi",
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        Phân tích TOÀN BỘ workspace — chỉ lấy HIGH/MEDIUM risk files.
        Tóm tắt tổng thể workspace chứ không đi sâu từng file.
        """
        high_files = [f for f in files if f.get("risk_score", 0) >= 0.67]
        med_files  = [f for f in files if 0.33 <= f.get("risk_score", 0) < 0.67]

        # Build workspace summary
        total_loc  = sum(f.get("metrics", {}).get("loc", 0) for f in files)
        avg_risk   = sum(f.get("risk_score", 0) for f in files) / max(len(files), 1)

        high_list  = "\n".join(
            f"  - `{f['name']}` (risk: {f['risk_score']:.1%}, "
            f"LOC: {f.get('metrics',{}).get('loc','?')}, "
            f"CC: {f.get('metrics',{}).get('cc','?')})"
            for f in high_files[:10]
        ) or "  (none)"

        med_list = "\n".join(
            f"  - `{f['name']}` (risk: {f['risk_score']:.1%})"
            for f in med_files[:8]
        ) or "  (none)"

        lang_note = ("Trả lời HOÀN TOÀN bằng Tiếng Việt."
                     if language == "vi" else "Answer in English.")

        workspace_prompt = f"""# Workspace-Level Code Quality Assessment

{lang_note}

## Workspace Statistics
- **Total files analyzed:** {len(files)}
- **Total Lines of Code:** {total_loc:,}
- **Average Risk Score:** {avg_risk:.1%}
- **HIGH risk files:** {len(high_files)} ({len(high_files)/max(len(files),1):.0%} of codebase)
- **MEDIUM risk files:** {len(med_files)}
- **LOW risk files:** {len(files) - len(high_files) - len(med_files)}

## HIGH Risk Files (require immediate attention)
{high_list}

## MEDIUM Risk Files (schedule for next sprint)
{med_list}

---

## Required Output

Please provide a **Workspace Risk Assessment Report**:

### 1. Overall Health Assessment
Rate the codebase health (🔴 Critical / 🟡 Needs Improvement / 🟢 Healthy)
and explain your verdict in 3–4 sentences.

### 2. Top 3 Most Critical Issues
The 3 issues that pose the greatest risk to codebase stability.

### 3. Sprint Planning Recommendation
A concrete action plan:
- **This Sprint (P1):** What to fix NOW
- **Next Sprint (P2):** What to schedule
- **Backlog (P3):** Technical debt to track

### 4. Estimated Total Remediation Effort
Provide time estimates for P1 + P2 work.

### 5. Key Metrics to Watch
Which metrics, if improved, would have the highest impact on overall quality.
"""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": workspace_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Kiểm tra API key có hợp lệ không bằng cách gọi model nhỏ."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        return True, "API key hợp lệ."
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg.lower():
            return False, "API key không hợp lệ. Kiểm tra lại tại console.groq.com"
        if "429" in msg:
            return False, "Rate limit exceeded. Thử lại sau vài giây."
        return False, f"Lỗi kết nối: {msg[:120]}"
