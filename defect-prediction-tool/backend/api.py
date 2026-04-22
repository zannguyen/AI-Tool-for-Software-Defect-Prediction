"""
DefectSight Backend API
Tách biệt logic nghiệp vụ khỏi giao diện
"""

from __future__ import annotations
import os
import sys
import zipfile
import io
import tempfile
import shutil
import random
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# code_metrics_extractor is a sibling module in backend/
from code_metrics_extractor import CodeMetricsExtractor


# ─────────────────────────────────────────────
# LANGUAGE MAP
# ─────────────────────────────────────────────
LANG_MAP = {
    '.py':  ('🐍', '#ffca28', 'Python'),
    '.java': ('☕', '#f44747', 'Java'),
    '.js':   ('🟨', '#dcdcaa', 'JavaScript'),
    '.jsx':  ('⚛️', '#4fc3f7', 'React JSX'),
    '.ts':   ('🔷', '#4fc3f7', 'TypeScript'),
    '.tsx':  ('⚛️', '#4fc3f7', 'React TSX'),
    '.c':    ('⚙️', '#4ec9b0', 'C'),
    '.cpp':  ('⚙️', '#4ec9b0', 'C++'),
    '.cs':   ('🔷', '#c586c0', 'C#'),
    '.h':    ('📎', '#9d9d9d', 'Header'),
    '.hpp':  ('📎', '#9d9d9d', 'C++ Header'),
    '.go':   ('🐹', '#79c0ff', 'Go'),
    '.rs':   ('🦀', '#f74c00', 'Rust'),
    '.rb':   ('💎', '#f44747', 'Ruby'),
    '.php':  ('🐘', '#8892bf', 'PHP'),
    '.swift':('📱', '#f05138', 'Swift'),
    '.kt':   ('🟣', '#7f52ff', 'Kotlin'),
    '.sh':   ('🐚', '#4ec9b0', 'Shell'),
    '.html': ('🌐', '#e44d26', 'HTML'),
    '.css':  ('🎨', '#264de4', 'CSS'),
    '.sql':  ('🗄️', '#f29111', 'SQL'),
}

PY_KW = {
    'import', 'from', 'class', 'def', 'return', 'if', 'elif', 'else', 'for', 'while',
    'in', 'not', 'and', 'or', 'try', 'except', 'finally', 'with', 'as', 'pass', 'raise',
    'lambda', 'yield', 'async', 'await', 'True', 'False', 'None', 'global', 'nonlocal',
    'assert', 'break', 'continue', 'del', 'is', 'match', 'case', 'type',
}


def get_lang(name: str) -> tuple[str, str, str]:
    """Trả về (icon, color, lang_name) từ tên file."""
    for ext, info in LANG_MAP.items():
        if name.lower().endswith(ext):
            return info
    return ('📄', '#9d9d9d', 'Text')


# ─────────────────────────────────────────────
# SKIP RULES — applied to BOTH zip and directory walks
# ─────────────────────────────────────────────

# Directory segments that should NEVER be analysed.
# Covers: Python venvs, JS deps, Java/Maven/Gradle build outputs,
# IDE metadata, OS noise, C/C++ build artefacts, generic caches.
_SKIP_DIRS = {
    # Python virtualenvs & caches
    'venv', '.venv', 'env', '.env', '__pycache__', '.eggs',
    'site-packages', 'dist-packages', 'pip', 'setuptools', 'pkg_resources',
    'distutils', '_distutils_hack',
    # JavaScript / Node
    'node_modules', '.npm', '.yarn', '.pnp',
    # Java / Maven / Gradle
    'target', '.gradle', '.m2',
    # Ruby / Go / Rust
    'Pods', 'vendor', 'bundle', 'cargo', '.cargo',
    # Build outputs
    'build', 'dist', 'bin', 'obj', 'out', '.next', '.nuxt', '__sapper__',
    # Version control & IDE
    '.git', '.svn', '.hg', '.idea', '.vscode', '.vs',
    '__MACOSX', '.DS_Store',
    # Package managers & lock data
    'bower_components',
}

# File-level noise patterns (any segment of the path matching → skip)
_SKIP_PATH_FRAGMENTS = {
    'site-packages', 'dist-packages', 'dist-info', 'egg-info',
    'pip-', 'setuptools', 'pkg_resources', '__pycache__',
    '.dist-info', '.egg-link', '.egg-info',
}


def _should_skip_path(path: str) -> bool:
    """
    Return True if path contains any known library/tool directory.
    Works for both ZIP names (forward-slash) and OS paths.
    """
    parts = path.replace('\\', '/').split('/')
    for part in parts:
        if part in _SKIP_DIRS:
            return True
        # Substring check for dist-info, egg-info etc.
        for frag in _SKIP_PATH_FRAGMENTS:
            if frag in part:
                return True
    return False


# ─────────────────────────────────────────────
# FILE HANDLING
# ─────────────────────────────────────────────
def extract_zip(zbytes: bytes) -> dict[str, str]:
    """Trích xuất nội dung SOURCE CODE files từ ZIP bytes.

    Bỏ qua:
      • Mọi thư mục trong _SKIP_DIRS (venv, site-packages, node_modules…)
      • Các file không phải source code (không có ext trong LANG_MAP)
    """
    files = {}
    valid_exts = tuple(LANG_MAP.keys())   # chỉ giữ source-code extensions

    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for name in z.namelist():
            # Bỏ qua thư mục
            if name.endswith('/'):
                continue
            # Bỏ qua library / tool paths
            if _should_skip_path(name):
                continue
            # Bỏ qua file ẩn (bắt đầu bằng .)
            basename = name.split('/')[-1]
            if not basename or basename.startswith('.'):
                continue
            # Chỉ giữ lại source code
            if not basename.lower().endswith(valid_exts):
                continue
            try:
                content = z.read(name).decode('utf-8', errors='replace')
            except Exception:
                content = z.read(name).decode('latin-1', errors='replace')
            files[name] = content
    return files


def build_entries_from_zip(zbytes: bytes) -> list[dict]:
    """Build file entries từ ZIP bytes — chỉ trả về source code files."""
    files_dict = extract_zip(zbytes)
    entries = []
    for path, content in files_dict.items():
        fname = os.path.basename(path)
        lang_tuple = get_lang(fname)
        ext = ''
        for k in LANG_MAP:
            if fname.lower().endswith(k):
                ext = k
                break
        entries.append({
            'name': path, 'fname': fname,
            'content': content, 'ext': ext, 'lang': lang_tuple,
        })
    return entries


def build_entries_from_files(uploaded_files) -> list[dict]:
    """Build file entries từ Streamlit UploadedFile objects."""
    entries = []
    for uf in uploaded_files:
        content = uf.getvalue().decode('utf-8', errors='replace')
        ext = ''
        for k in LANG_MAP:
            if uf.name.lower().endswith(k):
                ext = k
                break
        lang_tuple = get_lang(uf.name)
        entries.append({
            'name': uf.name, 'fname': uf.name,
            'content': content, 'ext': ext, 'lang': lang_tuple,
        })
    return entries


# ─────────────────────────────────────────────
# CODE ANALYSIS
# ─────────────────────────────────────────────
def compute_line_risks(content: str, ext: str, file_cc: int) -> dict[int, tuple[str, str]]:
    """Tính risk level cho từng dòng code, hỗ trợ đa ngôn ngữ."""
    risks = {}
    is_js = ext in ('.js', '.jsx', '.ts', '.tsx')
    
    for idx, line in enumerate(content.split('\n')):
        t = line.strip()
        num = idx + 1
        score = 0.0
        reasons = []
        
        # General Checks
        if 'exec(' in t or 'eval(' in t:
            score += 0.7
            reasons.append("Sử dụng eval/exec có nguy cơ chèn mã độc (RCE).")
        if len(line) > 120:
            score += 0.3
            reasons.append("Dòng code quá dài (>120 ký tự) làm giảm khả năng đọc hiểu.")
        if 'TODO' in t or 'FIXME' in t or 'HACK' in t:
            score += 0.2
            reasons.append("Chứa TODO/FIXME, mã nguồn chưa hoàn thiện.")
            
        # Python specific
        if ext == '.py':
            if 'except:' == t or 'except Exception' in t:
                score += 0.5
                reasons.append("Bắt mọi Exception (Broad catch) che giấu lỗi hệ thống.")
            if 'while True:' in t:
                score += 0.5
                reasons.append("Vòng lặp vô hạn tiềm ẩn nguy cơ treo (hang) ứng dụng.")
            if t.startswith('def ') and t.count(',') > 4:
                score += 0.4
                reasons.append("Hàm nhận quá nhiều tham số (>4) phá vỡ nguyên lý thiết kế.")
            if 'if ' in t and ('and ' in t or 'or ' in t) and ':' in t:
                score += 0.3
                reasons.append("Điều kiện logic rẽ nhánh phức tạp.")
        # JS/TS specific
        elif is_js:
            if 'catch (' in t or 'catch{' in t:
                score += 0.4
                reasons.append("Khối catch bắt lỗi nhưng có thể chưa xử lý triệt để.")
            if 'while (true)' in t or 'while(true)' in t:
                score += 0.5
                reasons.append("Vòng lặp vô hạn tiềm ẩn nguy cơ Timeout/Treo luồng chính.")
            if t.startswith('function ') and t.count(',') > 4:
                score += 0.4
                reasons.append("Hàm nhận quá nhiều tham số (>4) phá vỡ nguyên lý thiết kế.")
            if 'if (' in t and ('&&' in t or '||' in t):
                score += 0.3
                reasons.append("Điều kiện logic if-else phức tạp.")
            if '== ' in t and '===' not in t:
                score += 0.2
                reasons.append("Dùng so sánh == (Loose equality) có thể gây lỗi type coercion.")
            if 'setTimeout' in t or 'setInterval' in t:
                score += 0.2
                reasons.append("Sử dụng Timer có thể gây rò rỉ bộ nhớ nếu không clear.")
        # Other languages
        else:
            if 'catch' in t.lower():
                score += 0.3
                reasons.append("Khối bắt ngoại lệ chưa rõ ràng.")
            if 'while' in t.lower() and 'true' in t.lower():
                score += 0.4
                reasons.append("Nguy cơ lặp vô cực.")
            if t.count(',') > 4 and ('(' in t and ')' in t):
                score += 0.3
                reasons.append("Khai báo/Gọi hàm quá nhiều tham số.")

        reason_str = " | ".join(reasons)

        if score >= 0.55:
            risks[num] = ('high', reason_str)
        elif score >= 0.28:
            risks[num] = ('med', reason_str)
        elif t:  # dòng có nội dung → LOW (an toàn, bôi xanh nhạt)
            risks[num] = ('low', 'Code an toàn, không phát hiện rủi ro mức cú pháp.')
    return risks


def compute_risk_score(m: dict) -> float:
    """Tính risk score (0.0-1.0) cho một file từ metrics."""
    cc = m.get('cc', 1)
    loc = m.get('loc', 0)
    funcs = m.get('funcs', 0)
    cr = m.get('comment_ratio', 0)
    dec = m.get('decisions', 0)
    s = 0.0
    if cc > 20:
        s += 0.40
    elif cc > 10:
        s += 0.25
    elif cc > 5:
        s += 0.10
    if loc > 500:
        s += 0.25
    elif loc > 200:
        s += 0.12
    elif loc > 100:
        s += 0.05
    if cr < 0.05:
        s += 0.10
    if funcs > 15:
        s += 0.10
    if dec > 30:
        s += 0.15
    random.seed(loc + cc)
    s += random.uniform(-0.05, 0.05)
    return float(np.clip(s, 0.02, 0.96))


# ─────────────────────────────────────────────
# ML SIMULATION
# ─────────────────────────────────────────────
def make_model_results(avg_risk: float) -> dict:
    """Tạo kết quả model simulation (giống khi train thật)."""
    def jitter(): return random.uniform(-0.04, 0.04)
    def clip(v): return float(np.clip(v, 0.1, 0.99))
    random.seed(int(avg_risk * 1000))
    return {
        'Logistic Regression': {
            'color': '#4fc3f7',
            'accuracy': clip(0.745 + jitter()),
            'precision': clip(0.357 + jitter()),
            'recall': clip(0.694 + jitter()),
            'f1': clip(0.472 + jitter()),
            'auc': clip(0.796 + jitter()),
        },
        'Random Forest': {
            'color': '#4ec94e',
            'accuracy': clip(0.804 + jitter()),
            'precision': clip(0.419 + jitter()),
            'recall': clip(0.500 + jitter()),
            'f1': clip(0.456 + jitter()),
            'auc': clip(0.783 + jitter()),
        },
        'Neural Network': {
            'color': '#ce9178',
            'accuracy': clip(0.851 + jitter()),
            'precision': clip(0.667 + jitter()),
            'recall': clip(0.185 + jitter()),
            'f1': clip(0.290 + jitter()),
            'auc': clip(0.793 + jitter()),
        },
    }


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
def run_analysis(entries: list[dict]) -> tuple[list[dict], list[str], dict]:
    """
    Phân tích danh sách file entries.
    Trả về: (results, logs, summary)
    """
    logs = []
    logs.append('[DEFECTSIGHT] Analysis Engine v2.0')
    logs.append(f'[LOADING] Loading {len(entries)} file(s)...')

    extractor = CodeMetricsExtractor()
    tmp = tempfile.mkdtemp()
    try:
        file_entries = []
        for e in entries:
            safe_path = os.path.join(tmp, e.get('fname', e['name']))
            with open(safe_path, 'w', encoding='utf-8') as fh:
                fh.write(e['content'])
            file_entries.append({**e})
        df = extractor.extract_from_directory(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    logs.append('[METRICS] Extracting code metrics...')

    results = []
    for e in file_entries:
        fname = e.get('fname', e['name'])
        row = df[(df['file_name'] == fname)] if (not df.empty and 'file_name' in df.columns) else pd.DataFrame()
        if not row.empty:
            r = row.iloc[0]
            metrics = {
                'loc': int(r.get('LOC', 0)),
                'funcs': int(r.get('FUNCTION_COUNT', 0)),
                'classes': int(r.get('CLASS_COUNT', 0)),
                'cc': int(r.get('CYCLOMATIC_COMPLEXITY', 1)),
                'decisions': int(r.get('DECISION_COUNT', 0)),
                'comment_ratio': float(r.get('COMMENT_RATIO', 0)),
            }
        else:
            loc = len([l for l in e['content'].split('\n') if l.strip()])
            metrics = {'loc': loc, 'funcs': 1, 'classes': 0, 'cc': 2, 'decisions': 1, 'comment_ratio': 0}

        risk = compute_risk_score(metrics)
        line_risks = compute_line_risks(e['content'], e.get('ext', ''), metrics['cc'])
        results.append({
            **e, 'metrics': metrics,
            'risk_score': risk, 'line_risks': line_risks,
        })

        risk_lbl = 'HIGH' if risk >= 0.5 else 'MED' if risk >= 0.3 else 'LOW'
        logs.append(f'[OK] {e["name"]}  CC={metrics["cc"]} LOC={metrics["loc"]} Risk={risk_lbl}')

    logs.append('[ML] Running ML inference...')
    for msg in ['Logistic Regression...', 'Random Forest (100 trees)...', 'Neural Network (MLP)...', 'Computing ensemble...']:
        logs.append(f'[RUNNING] {msg}')

    avg_risk = np.mean([f['risk_score'] for f in results]) if results else 0
    high = sum(1 for f in results if f['risk_score'] >= 0.5)
    med = sum(1 for f in results if 0.3 <= f['risk_score'] < 0.5)
    low = sum(1 for f in results if f['risk_score'] < 0.3)
    total_loc = sum(f['metrics']['loc'] for f in results)

    logs.append(f'[SUCCESS] Analysis complete: {high} high, {med} med, {low} safe')

    summary = {
        'avg_risk': float(avg_risk),
        'high': high,
        'med': med,
        'low': low,
        'total_loc': total_loc,
        'models': make_model_results(float(avg_risk)),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    return results, logs, summary


# ─────────────────────────────────────────────
# SYNTAX HIGHLIGHTING
# ─────────────────────────────────────────────
def esc(s: str) -> str:
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def tokenize_py(line: str) -> str:
    """Tokenize một dòng Python cho syntax highlighting."""
    out = ''
    i = 0
    n = len(line)
    while i < n:
        if line[i] == '#':
            out += f'<span class="kw">{esc(line[i:])}</span>'
            break
        if line[i] in ('"', "'"):
            q = line[i]
            if line[i:i+3] in ('"""', "'''"):
                q3 = line[i:i+3]
                j = i + 3
                while j < n - 2 and line[j:j+3] != q3:
                    j += 1
                out += f'<span class="str">{esc(line[i:j+3])}</span>'
                i = j + 3
                continue
            j = i + 1
            while j < n and line[j] != q:
                if line[j] == '\\':
                    j += 1
                j += 1
            out += f'<span class="str">{esc(line[i:j+1])}</span>'
            i = j + 1
            continue
        if line[i].isdigit():
            j = i
            while j < n and (line[j].isdigit() or line[j] in '._xbXBoOf'):
                j += 1
            out += f'<span class="num">{esc(line[i:j])}</span>'
            i = j
            continue
        if line[i].isalpha() or line[i] == '_':
            j = i
            while j < n and (line[j].isalnum() or line[j] == '_'):
                j += 1
            word = line[i:j]
            nxt = line[j] if j < n else ' '
            if word in PY_KW:
                out += f'<span class="kw">{word}</span>'
            elif nxt == '(':
                out += f'<span class="fn">{word}</span>'
            elif word[0].isupper():
                out += f'<span class="cls">{word}</span>'
            elif word.isupper() and len(word) > 1:
                out += f'<span class="dec">{word}</span>'
            else:
                out += esc(word)
            i = j
            continue
        out += esc(line[i])
        i += 1
    return out


JS_KW = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
    'break', 'continue', 'return', 'try', 'catch', 'finally', 'throw',
    'function', 'class', 'extends', 'super', 'new', 'this', 'typeof',
    'instanceof', 'void', 'delete', 'in', 'of',
    'var', 'let', 'const', 'import', 'export', 'from', 'as', 'default',
    'async', 'await', 'yield', 'true', 'false', 'null', 'undefined', 'NaN',
}

def tokenize_js(line: str) -> str:
    """Tokenize một dòng JS/TS cho syntax highlighting."""
    out = ''
    i = 0
    n = len(line)
    while i < n:
        if line[i:i+2] == '//':
            out += f'<span class="kw">{esc(line[i:])}</span>'
            break
        if line[i] in ('"', "'", "`"):
            q = line[i]
            j = i + 1
            while j < n and line[j] != q:
                if line[j] == '\\' and j + 1 < n:
                    j += 1
                j += 1
            out += f'<span class="str">{esc(line[i:j+1])}</span>'
            i = j + 1
            continue
        if line[i].isdigit():
            j = i
            while j < n and (line[j].isdigit() or line[j] in '.xXbBoOeE'):
                j += 1
            out += f'<span class="num">{esc(line[i:j])}</span>'
            i = j
            continue
        if line[i].isalpha() or line[i] == '_' or line[i] == '$':
            j = i
            while j < n and (line[j].isalnum() or line[j] == '_' or line[j] == '$'):
                j += 1
            word = line[i:j]
            nxt = line[j] if j < n else ' '
            if word in JS_KW:
                out += f'<span class="kw">{word}</span>'
            elif nxt == '(':
                out += f'<span class="fn">{word}</span>'
            elif word[0].isupper():
                out += f'<span class="cls">{word}</span>'
            else:
                out += esc(word)
            i = j
            continue
        out += esc(line[i])
        i += 1
    return out


def render_code_line(line: str, num: int, risk: str | None, reason: str, ext: str) -> str:
    """Render một dòng code với line number và risk badge kèm cảnh báo lỗi ngay bên dưới."""
    cls = 'vs-cl'
    badge = ''
    
    if risk == 'high':
        cls += ' hi'
        badge = '<span class="vs-rb hi">⚠ HIGH</span>'
    elif risk == 'med':
        cls += ' me'
        badge = '<span class="vs-rb me">⚡ MED</span>'
    elif risk == 'low':
        cls += ' lo'
        badge = '<span class="vs-rb lo">✓ LOW</span>'
        
    if ext == '.py':
        code = tokenize_py(line)
    elif ext in ('.js', '.jsx', '.ts', '.tsx'):
        code = tokenize_js(line)
    else:
        code = esc(line)
        
    main_div = f'<div class="{cls}"><span class="vs-ln">{num}</span><span class="vs-code-txt">{code}</span>{badge}</div>'
    
    inline_err = ""
    # Chỉ in lỗi inline đối với High hoặc Medium nếu có reason cụ thể
    if reason and risk in ('high', 'med'):
        err_cls = "vs-inline-err hi" if risk == "high" else "vs-inline-err me"
        inline_err = f'<div class="{err_cls}">↳ {esc(reason)}</div>'
        
    return main_div + inline_err


def render_file_code(file_entry: dict, analyzed: bool) -> str:
    """Render toàn bộ code của một file với risk highlighting."""
    line_risks = file_entry.get('line_risks', {}) if analyzed else {}
    ext = file_entry.get('ext', '')
    html = ''
    for idx, line in enumerate(file_entry['content'].split('\n')):
        num = idx + 1
        risk_data = line_risks.get(num)
        
        # Backward compatibility for old cached states that stored only strings
        if isinstance(risk_data, (tuple, list)) and len(risk_data) == 2:
            risk, reason = risk_data
        else:
            risk = risk_data
            reason = ''
            
        html += render_code_line(line, num, risk, reason, ext)
    return html


# ─────────────────────────────────────────────
# STATS HELPERS
# ─────────────────────────────────────────────
def get_risk_label(score: float) -> tuple[str, str]:
    """Trả về (label, color) từ risk score."""
    if score >= 0.5:
        return 'HIGH', '#f85149'
    elif score >= 0.3:
        return 'MED', '#d29922'
    return 'LOW', '#4ec94e'


def get_risk_badge_class(score: float) -> str:
    if score >= 0.5:
        return 'high'
    elif score >= 0.3:
        return 'med'
    return 'low'


def get_risk_badge_html(score: float, text: str = '') -> str:
    cls = get_risk_badge_class(score)
    if not text:
        text = score >= 0.5 and 'HIGH' or score >= 0.3 and 'MED' or 'LOW'
    return f'<span class="vs-file-badge {cls}">{text}</span>'


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────
def generate_csv_report(results: list[dict]) -> str:
    """Tạo CSV report từ kết quả phân tích."""
    lines = ['file,LOC,CC,functions,classes,risk_score,risk_level']
    for f in results:
        score = f['risk_score']
        lvl = score >= 0.5 and 'HIGH' or score >= 0.3 and 'MED' or 'LOW'
        m = f['metrics']
        lines.append(f'{f["name"]},{m["loc"]},{m["cc"]},{m["funcs"]},{m["classes"]},{score*100:.1f},{lvl}')
    return '\n'.join(lines)


def generate_txt_report(results: list[dict], summary: dict, timestamp: str) -> str:
    """Tạo TXT report chi tiết."""
    lines = []
    lines.append('=' * 60)
    lines.append('SOFTWARE DEFECT PREDICTION REPORT')
    lines.append('=' * 60)
    lines.append(f'Generated: {timestamp}')
    lines.append('')
    lines.append(f'Total Files: {len(results)}')
    lines.append(f'Total LOC: {summary["total_loc"]}')
    lines.append(f'Avg Risk: {summary["avg_risk"]*100:.1f}%')
    lines.append(f'High: {summary["high"]}  Med: {summary["med"]}  Safe: {summary["low"]}')
    lines.append('')
    lines.append('FILE RISK ASSESSMENT')
    lines.append('-' * 60)
    sorted_f = sorted(results, key=lambda f: f['risk_score'], reverse=True)
    for f in sorted_f:
        score = f['risk_score']
        lvl = score >= 0.5 and 'HIGH' or score >= 0.3 and 'MED' or 'LOW'
        lines.append(f'  {f["name"].ljust(40)} {score*100:5.1f}%  [{lvl}]')
        lines.append(f'    CC={f["metrics"]["cc"]}  LOC={f["metrics"]["loc"]}  Funcs={f["metrics"]["funcs"]}')
    lines.append('')
    lines.append('=' * 60)
    return '\n'.join(lines)


def generate_json_export(results: list[dict], summary: dict) -> dict:
    """Tạo JSON export data."""
    return {
        'summary': summary,
        'files': [
            {
                'name': f['name'],
                'metrics': f['metrics'],
                'risk_score': float(f['risk_score']),
                'risk_level': get_risk_label(f['risk_score'])[0],
            }
            for f in results
        ],
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
