# DefectSight — AI Software Defect Prediction Tool
> Cập nhật lần cuối: 2026-04-17

---

## 📁 Cấu trúc dự án

```
defect-prediction-tool/
├── frontend/
│   └── app.py                  ← Toàn bộ UI (Streamlit, VS Code style)
├── backend/
│   ├── api.py                  ← Business logic chính (phân tích, render)
│   ├── models.py               ← ML models (LR, RF, NN)
│   ├── code_metrics_extractor.py ← Trích xuất metrics từ source code
│   ├── features.py             ← Feature engineering
│   ├── preprocessing.py        ← Tiền xử lý dữ liệu
│   ├── evaluation.py           ← Đánh giá model
│   ├── database.py             ← Quản lý cơ sở dữ liệu
│   ├── data/                   ← Dữ liệu huấn luyện
│   ├── models/                 ← Model files đã train (.pkl, .h5)
│   └── reports/                ← Báo cáo xuất file
├── dashboard/
│   └── Dashboard.py            ← Dashboard phân tích (Streamlit riêng)
├── requirements.txt
├── run.bat                     ← Script chạy app (Windows)
└── CLAUDE.md                   ← File này
```

---

## 🚀 Cách chạy

```bash
# Cách 1: Script Windows
run.bat

# Cách 2: Manual
streamlit run frontend/app.py

# URL: http://localhost:8501
```

**Python path cứng trong run.bat:**
```
C:\Users\Dell 7630\AppData\Local\Programs\Python\Python313\Scripts\streamlit.exe
```

---

## 🎨 Frontend: `frontend/app.py` (869 dòng)

### Kiến trúc layout — VS Code Single Viewport

```
┌──────────────────────────────────────────────────────────────┐
│  Title Bar (30px) — HTML div .ds-titlebar                    │
├──────────────┬─────────────────────────────┬─────────────────┤
│  LEFT        │  CENTER (Editor)            │  RIGHT          │
│  col 1.4     │  col 3.5                    │  col 1.4        │
│              │                             │                 │
│  Pre-analysis│  Tab bar (35px)             │  Analysis panel │
│  → Import WS │  Breadcrumb (24px)          │  .ds-right-     │
│              │  Code area                  │   analysis      │
│  Post-analy  │  calc(100vh-263px)          │  calc(100vh-    │
│  → Explorer  │                             │   237px)        │
│  → Model sel │                             │                 │
│  → File tree │                             │                 │
│  (scrollable)│                             │                 │
├──────────────┴─────────────────────────────┴─────────────────┤
│  TERMINAL (position:fixed; bottom:22px; height:150px)        │
│  .ds-terminal-fixed — full width, scrollable logs            │
├──────────────────────────────────────────────────────────────┤
│  Status Bar (position:fixed; bottom:0; height:22px)          │
│  .ds-statusbar                                               │
└──────────────────────────────────────────────────────────────┘
```

### Công thức tính chiều cao (QUAN TRỌNG)

```
Tổng: 100vh

Title bar:    30px  (flow)
3 cột:        calc(100vh - 202px) = 100vh - (30+150+22)
Terminal:     150px  fixed bottom=22px   → top = 100vh-172px
Status bar:   22px   fixed bottom=0

Column bottom = 30 + (100vh-202) = 100vh-172px = Terminal top ✓

Code area:    calc(100vh - 263px) = col_height - tabs(35) - breadcrumb(24) - 2px
Right panel:  calc(100vh - 237px) = col_height - pane_header(35)
```

### CSS design tokens

```css
--bg0: #1e1e1e    /* Editor background */
--bg1: #252526    /* Sidebar/panel background */
--bg2: #2d2d2d    /* Header/hover background */
--bg3: #3c3c3c    /* Progress bar track */
--accent: #007acc /* VS Code blue */
--green: #4ec9b0
--red: #f85149
--yellow: #e2c08d
--txt: #cccccc    /* Primary text */
--txt2: #969696   /* Secondary text */
--txt3: #6e6e6e   /* Muted text */
--border: #3c3c3c
```

### CSS Classes tổng hợp

| Class | Mô tả |
|---|---|
| `.ds-titlebar` | Title bar trên cùng 30px |
| `.ds-pane-header` | Header 35px cho mỗi cột |
| `.ds-tabs` | Tab bar file trong editor 35px |
| `.ds-breadcrumb` | Breadcrumb path 24px |
| `.ds-code-area` | Vùng hiển thị code, scrollable |
| `.ds-welcome` | Welcome screen khi chưa import |
| `.ds-right-analysis` | Vùng phân tích cột phải, scrollable |
| `.ds-stat-grid` | Grid 3 cột cho High/Med/Low counts |
| `.ds-stat` | Card thống kê risk |
| `.ds-mc` | Model card (LR/RF/NN metrics) |
| `.ds-terminal-fixed` | Terminal panel fixed bottom |
| `.ds-terminal-hdr` | Terminal header 30px |
| `.ds-terminal-scroll` | Nội dung log scrollable |
| `.ds-statusbar` | Status bar 22px fixed bottom:0 |

### Session State keys

```python
_DEFAULTS = {
    'files': [],           # List[dict] — file entries đã load
    'analysis': None,      # dict — summary phân tích
    'analyzed': False,     # bool — đã chạy phân tích chưa
    'model_choice': 'All Models',  # str — model đang chọn
    'logs': [],            # List[str] — log messages
    'active_file': None,   # str — tên file đang xem
}
# + 'dir_...' keys cho expand/collapse thư mục trong tree
```

### Layout code structure (theo dòng)

```
L1-24:    Import, page config
L25-526:  CSS injection (st.markdown với unsafe_allow_html)
  L47-75:   Kill scroll, body reset
  L77-92:   Kill Streamlit vertical gaps
  L94-116:  [data-testid="stHorizontalBlock"] — column heights
  L118-168: .ds-terminal-fixed CSS
  L170-177: .ds-statusbar CSS (position:fixed)
  L179-231: Title bar, Left panel, Panel header CSS
  L233-295: Tree buttons, Import form CSS
  L297-331: Status bar content, Center panel CSS
  L333-409: Tab bar, Breadcrumb, Code area CSS
  L412-434: Welcome screen, Right panel CSS
  L436-463: Risk cards, Model cards CSS
  L510-525: Run button override CSS

L529-548:  Session state initialization
L550-568:  Title bar render (HTML)
L570-631:  Tree helpers: build_tree(), render_tree()
L634-637:  st.columns([1.4, 3.5, 1.4]) — 3 cột

L639-723:  LEFT PANEL
  L643-676:   Import Mode (file_uploader, ZIP, Run button)
  L677-723:   Explorer Mode (pane header, New Import btn,
               spacer, Model selector, divider, File tree)

L726-768:  CENTER PANEL (tab bar, breadcrumb, code viewer)

L770-815:  RIGHT PANEL (HTML block .ds-right-analysis)
  L779-794:   Risk stats (Workspace Risk %, High/Med/Low grid)
  L796-810:   Model cards (LR, RF, NN performance bars)

L818-847:  TERMINAL PANEL (position:fixed HTML div)
L850-868:  STATUS BAR (HTML div)
```

### Điều chỉnh thủ công dễ dàng

| Muốn thay đổi | Vị trí |
|---|---|
| Khoảng cách "New Import" → Model | L692: `height:10px` |
| Chiều cao Terminal | L126: `height: 150px` |
| Chiều cao File Tree | L721: `st.container(height=500)` |
| Chiều cao Code area | L388: `calc(100vh - 263px)` |
| Màu accent (VS Code blue) | L37: `--accent: #007acc` |
| Font size code | L394: `font-size: 13px` |

---

## ⚙️ Backend: `backend/api.py` (477 dòng)

### Các hàm chính

```python
# File handling
build_entries_from_zip(zbytes: bytes) -> List[dict]
build_entries_from_files(uploaded_files) -> List[dict]

# Analysis engine
run_analysis(entries: List[dict]) -> (results, logs, summary)
  # results: List[dict] với keys: name, fname, content, ext, lang,
  #          metrics, risk_score, line_risks
  # logs: List[str]
  # summary: {avg_risk, high, med, low, total_loc, models, timestamp}

# Risk scoring
compute_risk_score(m: dict) -> float   # 0.0 - 1.0
compute_line_risks(content, ext, file_cc) -> {line_num: 'high'|'med'}

# Rendering
render_file_code(file_entry, analyzed) -> str   # HTML
render_code_line(line, num, risk, ext) -> str   # HTML
tokenize_py(line: str) -> str                   # Python syntax highlight

# Helpers
get_risk_label(score: float) -> (label, color)
get_lang(name: str) -> (icon, color, lang_name)

# Reports (chưa dùng trong UI hiện tại)
generate_csv_report(results) -> str
generate_txt_report(results, summary, timestamp) -> str
generate_json_export(results, summary) -> dict
```

### Risk scoring formula

```python
# compute_risk_score()
score = 0.0
if cc > 20:   score += 0.40
elif cc > 10: score += 0.25
elif cc > 5:  score += 0.10
if loc > 500: score += 0.25
elif loc > 200: score += 0.12
if comment_ratio < 0.05: score += 0.10
if funcs > 15: score += 0.10
if decisions > 30: score += 0.15
score = clip(score + random_jitter, 0.02, 0.96)
```

### Line-level risk triggers

```python
exec(/eval(         → +0.7
except: (bare)      → +0.5
while True:         → +0.5
line > 120 chars    → +0.3
TODO/FIXME          → +0.2
def với >4 params   → +0.4
complex if+and/or   → +0.3
file CC > 15        → ×1.4 multiplier
```

### ML simulation (`make_model_results`)

> ⚠️ Hiện tại là **simulation** (hardcoded values + noise), chưa train thật

| Model | Base Accuracy | Base AUC |
|---|---|---|
| Logistic Regression | 74.5% | 79.6% |
| Random Forest | 80.4% | 78.3% |
| Neural Network | 85.1% | 79.3% |

### Language support

| Extension | Icon | Name |
|---|---|---|
| .py | 🐍 | Python |
| .java | ☕ | Java |
| .js/.jsx | 🟨/⚛️ | JavaScript/React |
| .ts/.tsx | 🔷/⚛️ | TypeScript/React TSX |
| .c/.cpp | ⚙️ | C/C++ |
| .cs | 🔷 | C# |
| .go | 🐹 | Go |
| .rs | 🦀 | Rust |
| .rb | 💎 | Ruby |
| .php | 🐘 | PHP |
| .swift | 📱 | Swift |
| .kt | 🟣 | Kotlin |
| .html/.css | 🌐/🎨 | HTML/CSS |
| .sql | 🗄️ | SQL |

---

## 📊 Backend: `backend/models.py` (458 dòng)

Class `DefectPredictionModels` — train/evaluate thật với sklearn + TensorFlow:

```python
model = DefectPredictionModels(random_state=42)
model.initialize_models()           # Khởi tạo LR, RF, NN
model.train_models(X_train, y_train)
model.evaluate_models(X_test, y_test)
model.save_models(path)
model.load_models(path)
```

**Neural Network architecture (Keras):**
```
Dense(64, relu) → BatchNorm → Dropout(0.3)
Dense(32, relu) → BatchNorm → Dropout(0.3)
Dense(16, relu) → BatchNorm
Dense(1, sigmoid)
```

---

## 📐 Backend: `backend/code_metrics_extractor.py` (403 dòng)

Class `CodeMetricsExtractor` — trích xuất metrics thực tế từ code:

```python
extractor = CodeMetricsExtractor()
df = extractor.extract_from_directory(tmp_dir)
# df columns: file_name, LOC, LOC_BLANK, LOC_TOTAL, LOC_COMMENTS,
#             LOC_CODE, FUNCTION_COUNT, CLASS_COUNT,
#             CYCLOMATIC_COMPLEXITY, DECISION_COUNT,
#             COMMENT_RATIO, IMPORT_COUNT, RETURN_COUNT, ...
```

**Ngôn ngữ hỗ trợ:** Python, Java, JavaScript, C, C++, C# (và H/HPP)

---

## 📦 Dependencies (`requirements.txt`)

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
# TensorFlow optional (fallback to sklearn MLPClassifier)
```

---

## 🔍 Luồng hoạt động UI

```
[1] User mở http://localhost:8501
    → Left panel: "Import Workspace" (file_uploader + ZIP uploader)
    → Center: Welcome screen "DefectSight"
    → Right: "Analysis" (trống)
    → Terminal: "Terminal ready."

[2] User upload files/ZIP → Click "▶ Run Analysis"
    → build_entries_from_zip() / build_entries_from_files()
    → run_analysis(entries)
       → CodeMetricsExtractor.extract_from_directory()
       → compute_risk_score() cho từng file
       → compute_line_risks() cho line-level
       → make_model_results() (ML simulation)
    → session_state.analyzed = True → st.rerun()

[3] Post-analysis state:
    → Left panel: Explorer mode
       - "EXPLORER — N files" header
       - "⊕ New Import" button
       - "MODEL" label + [All][LR][RF][NN] buttons
       - File tree (st.container height=500, scrollable)
    → Center: Tab bar + Breadcrumb + Code viewer (highlighted)
    → Right: Workspace Risk % + High/Med/Low stats + Model cards
    → Terminal (fixed): Analysis logs với màu sắc phân loại

[4] User click file trong tree → st.session_state.active_file = … → rerun
[5] User click "⊕ New Import" → reset session_state → rerun về state [1]
```

---

## 🐛 Gotchas & Notes

1. **Streamlit CSS conflicts**: Dùng `!important` + target `[data-testid]` selectors. Không dùng class targets vì Streamlit có thể đổi tên lúc update.

2. **gap:0 kills spacing**: CSS `[data-testid="stVerticalBlock"] { gap: 0 }` xoá hết khoảng cách mặc định. Nếu cần khoảng cách, dùng `st.markdown('<div style="height:Xpx">')`.

3. **flex:1 không hoạt động** trong `st.markdown` blocks vì chúng không nằm trong flex container của Streamlit. Dùng `height: calc(...)` explicit thay thế.

4. **position:fixed** cho Terminal và Status bar để tránh khoảng hở do Streamlit tự thêm margin giữa blocks.

5. **build_tree() phân biệt** file entry (`dict` có key `'content'`) vs directory node (`dict` không có `'content'`). Quan trọng để tránh TypeError trong render_tree().

6. **Model selector duplicate**: Không dùng `st.markdown(label) + st.button(same_text)` — sẽ render 2 lần. Chỉ dùng `st.button()`.

7. **Tree container height=500**: Số cứng, phụ thuộc viewport. Nếu màn hình nhỏ hơn 900px, cần giảm.

8. **ML trong api.py là simulation**: `make_model_results()` không train thật — chỉ là hardcoded values + random jitter. Model thật nằm trong `backend/models.py` nhưng chưa được tích hợp vào UI.
