# DefectSight — AI Software Defect Prediction Tool
> Cập nhật lần cuối: 2026-04-22

---

## 🆕 Cập nhật gần đây (Release Notes)

- **Dark/Light Theme Toggle:** Tích hợp nút chuyển đổi chế độ Sáng/Tối trực quan bên cạnh Topbar Header. Quản lý Style linh hoạt thông qua hệ thống phân cực thông minh: State (`st.session_state.theme`) và CSS Variables root (`ui/styles.py`).
- **UI Refinement & 3-Column Layout:** Nâng cấp toàn diện giao diện Streamlit từ dạng xếp tầng dọc thành **3 cột chuẩn IDE** (Left Sidebar, Center Code Editor, Right Insights). Xử lý triệt để hiện tượng layout đè lấp.
- **AI Inline Explanations:** Bổ sung hiển thị `Inline Tooltip` thông minh trên từng dòng mã nguồn bị đánh dấu rủi ro, cho phép phân tích nguyên nhân ngay tại chỗ.
- **Groq LLM Risk Assessment:** Tích hợp phân tích mã nguồn bằng LLM siêu tốc thông qua Groq API (`backend/ai_reviewer.py`), hỗ trợ Streamlit tab mới (Prediction & Evaluation). Đưa ra các báo cáo nguy cơ chi tiết định dạng Markdown.
- **Codebase Clean-up:** Xóa bỏ hoàn toàn các file script thử nghiệm hoặc rác (`_patch_prediction_tab.py`, `inspect_api.py`, v.v.) bảo đảm 100% production-ready.

---

## 📁 Cấu trúc dự án (THỰC TẾ)

```
defect-prediction-tool/
├── CLAUDE.md                   ← File này
├── README.md                   ← Giới thiệu dự án chung
├── requirements.txt            ← Dependencies Python
├── run.bat                     ← Script chạy app Windows (hardcoded Python path)
│
├── frontend/
│   ├── app.py                  ← Entry point Streamlit (~312 dòng)
│   └── ui/
│       ├── __init__.py
│       ├── state.py            ← Session state management (~28 dòng)
│       ├── styles.py           ← CSS tập trung / apply_global_styles() (~304 dòng)
│       └── tree.py             ← File explorer tree logic (~147 dòng)
│
├── backend/
│   ├── __init__.py
│   ├── api.py                  ← Business logic chính (~482 dòng)
│   ├── models.py               ← ML models thực (LR, RF, NN) (~458 dòng)
│   ├── code_metrics_extractor.py ← Trích xuất metrics từ source code (~403 dòng)
│   ├── features.py             ← Feature engineering (~110 dòng)
│   ├── preprocessing.py        ← Tiền xử lý dữ liệu (~148 dòng)
│   ├── evaluation.py           ← Đánh giá model (~294 dòng)
│   ├── database.py             ← Quản lý SQLite (~360 dòng)
│   ├── data/                   ← Dữ liệu huấn luyện (kc1.arff, kc2.arff, kc1_kc2.csv)
│   ├── models/                 ← Model files đã train (.pkl, .h5)
│   └── reports/                ← Báo cáo xuất file
│
└── dashboard/
    └── Dashboard.py            ← Dashboard phân tích (file riêng, chưa tích hợp)
```

> ⚠️ **Lưu ý quan trọng**: README.md ở gốc mô tả cấu trúc cũ (app.py ở root, thư mục src/). Cấu trúc **thực tế hiện tại** là `frontend/` + `backend/` như trên.

---

## 🚀 Cách chạy

```bash
# Cách 1: Script Windows (hardcoded Python path)
run.bat

# Cách 2: Manual
cd defect-prediction-tool
streamlit run frontend/app.py

# URL: http://localhost:8501
```

**Python path cứng trong run.bat:**
```
C:\Users\Dell 7630\AppData\Local\Programs\Python\Python313\Scripts\streamlit.exe
```

---

## 🎨 Frontend: `frontend/app.py` (~312 dòng)

### Kiến trúc layout — 3 cột + Console (Streamlit native)

```
┌──────────────────────────────────────────────────────────────┐
│  TopBar Banner (ds-topbar) — gradient header, animated       │
│  "DefectSight Workbench | N files | LOC | Avg risk | Status" │
├──────────────┬─────────────────────────────┬─────────────────┤
│  LEFT [1.3]  │  CENTER [3.0]              │  RIGHT [1.5]    │
│  height=800  │  (native scroll)           │  height=800     │
│              │                            │                 │
│  · Import WS │  · ds-file-header          │  · Insights     │
│    (upload)  │  · ds-metric-grid (3 col)  │  · Workspace %  │
│  · Run Analy │  · ds-code-box             │  · High/Med/Low │
│              │    max-height:58vh         │  · Model cards  │
│  ── OR ──    │    scrollable code viewer  │  · Top risky    │
│              │                            │    files        │
│  · Explorer  │                            │                 │
│    - Search  │                            │                 │
│    - Filter  │                            │                 │
│    - Model   │                            │                 │
│    - Tree    │                            │                 │
├──────────────┴─────────────────────────────┴─────────────────┤
│  Console Panel (ds-log) — fixed max-height:170px, scrollable │
└──────────────────────────────────────────────────────────────┘
```

### Layout so với phiên bản cũ

| Phần | Phiên bản cũ (VS Code style) | Phiên bản hiện tại |
|---|---|---|
| Container | `position:fixed` cho tất cả | `st.columns([1.3, 3.0, 1.5])` + `st.container(height=800)` |
| Terminal | `position:fixed; bottom:22px; height:150px` | Console `ds-log` đặt dưới cùng |
| Status bar | `position:fixed; bottom:0; height:22px` | Không có |
| Title bar | `position:fixed; top:0; height:30px` | `.ds-topbar` gradient banner (không fixed) |

> **Tại sao đổi?** `position:fixed` CSS bị conflict với Streamlit DOM. Layout native Streamlit (`st.columns` + `st.container(height=...)`) ổn định hơn giữa các phiên bản.

### Các hàm trong `app.py`

```python
_format_risk(score: float) -> str
    # "HIGH (72.5%)"

_render_topbar(files_count: int, analysis: dict | None) -> None
    # Banner gradient trên cùng

_run_analysis_from_inputs() -> None
    # Đọc zip_input / files_input → gọi build_entries + run_analysis → st.rerun()

_render_import_panel() -> None
    # State chưa analyzed: file_uploader (multi) + ZIP + "Run Analysis" button

_render_explorer_panel(files, analyzed) -> None
    # State đã analyzed: search + risk filter + model selector + render_tree()

_render_center_panel(files, active_file, analyzed) -> None
    # File header + metric grid + ds-code-box html

_render_right_panel(files, analysis, analyzed) -> None
    # Workspace risk metric + High/Med/Low + model progress bars + top risky files

_render_logs(logs) -> None
    # ds-log html với màu theo keyword

main() -> None
    # ensure_state() → apply_global_styles() → topbar → 3-col → console
```

---

## 🎨 Frontend: `frontend/ui/state.py` (~28 dòng)

### Session State Keys

```python
DEFAULTS = {
    "files": [],            # List[dict] — file entries đã load và analyze
    "analysis": None,       # dict — summary phân tích tổng thể
    "analyzed": False,      # bool — đã chạy phân tích chưa
    "model_choice": "All Models",  # str — model đang chọn trong selectbox
    "logs": [],             # List[str] — console log messages
    "active_file": None,    # str — name của file đang xem trong center panel
    "tree_query": "",       # str — search query trong explorer
    "tree_risk_filter": "All",  # str — "All" | "High" | "Medium" | "Low"
    "tree_expanded": {},    # dict — trạng thái expand/collapse folder trong tree
}
```

```python
ensure_state() -> None    # Khởi tạo keys còn thiếu trong session_state
reset_analysis_state() -> None  # Reset toàn bộ về DEFAULTS
```

---

## 🎨 Frontend: `frontend/ui/styles.py` (~304 dòng)

### Design tokens (CSS variables)

```css
--bg: #0f1318           /* Background ứng dụng (dark navy) */
--panel: #171d24        /* Panel background */
--panel-soft: #1f2731   /* Soft panel (metric cards) */
--ink: #e6edf6          /* Primary text */
--ink-muted: #9aabbe    /* Secondary text */
--line: #2c3643         /* Border color */
--brand: #2f9d8a        /* Primary color (teal) */
--brand-soft: #173f39   /* Brand background soft */
--danger: #b8323d       /* HIGH risk color */
--warn: #ad7b2b         /* MED risk color */
--safe: #2e6f44         /* LOW risk color */
```

> **Lưu ý**: Dự án đã đổi color scheme! Phiên bản cũ dùng VS Code blue `#007acc`. Hiện tại dùng **dark teal** `#2f9d8a`.

### Fonts

- **UI**: `Manrope` (wght 400/600/700/800) — Google Fonts
- **Code**: `JetBrains Mono` (wght 400/500) — Google Fonts

### CSS Classes tổng hợp

| Class | Mô tả |
|---|---|
| `.ds-topbar` | Header banner gradient `#1d2833 → #1f5d72 → #255b4f` + `ds-fade-in` animation |
| `.ds-topbar-title` | Tiêu đề topbar 1.12rem, bold |
| `.ds-topbar-sub` | Subtitle topbar 0.86rem |
| `.ds-pane-title` | Label UPPERCASE 0.72rem cho mỗi panel |
| `.ds-file-header` | Header file đang xem (panel border-radius:12px) |
| `.ds-file-name` | Tên file 0.98rem bold |
| `.ds-file-sub` | Lang + risk info 0.82rem |
| `.ds-metric-grid` | Grid 3 cột cho LOC/Cyclomatic/Functions |
| `.ds-metric` | Card metric vuông với border |
| `.ds-metric-v` | Giá trị metric 1.1rem bold |
| `.ds-metric-l` | Label metric 0.68rem uppercase |
| `.ds-code-box` | Code viewer `max-height:58vh`, scrollable, bg `#0f1720` |
| `.vs-cl` | Dòng code (flex row, JetBrains Mono 0.8rem) |
| `.vs-cl.hi` | Dòng HIGH risk — bg `rgba(184,50,61,.14)`, border-left đỏ |
| `.vs-cl.me` | Dòng MED risk — bg `rgba(173,123,43,.15)`, border-left vàng |
| `.vs-ln` | Line number (width:52px, màu muted) |
| `.vs-code-txt` | Nội dung code (pre, flex:1) |
| `.vs-rb` | Risk badge pill (border-radius:999px) |
| `.vs-rb.hi` | Badge HIGH: đỏ |
| `.vs-rb.me` | Badge MED: vàng |
| `.kw` | Syntax highlight: keyword (#57a9ff) |
| `.fn` | Syntax highlight: function name (#f7d779) |
| `.cls` | Syntax highlight: class name (#69d8bd) |
| `.str` | Syntax highlight: string (#ec9f78) |
| `.num` | Syntax highlight: number (#bde38a) |
| `.dec` | Syntax highlight: constant (#8dc5ff) |
| `.ds-tab-pill` | Tab pill (không dùng nhiều hiện tại) |
| `.ds-log` | Console log box bg `#131a1f`, `max-height:170px` |
| `.ds-card` | Generic card helper |

### Ẩn Streamlit UI mặc định

```css
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="collapsedControl"],
[data-testid="stSidebar"] {
    display: none;
}
```

---

## 🌳 Frontend: `frontend/ui/tree.py` (~147 dòng)

### Hàm chính

```python
build_tree(files_list: list[dict]) -> dict
    # Chuyển list file entries thành nested dict theo path
    # Phân biệt: file entry (dict có key 'content') vs folder node (dict không có 'content')

_file_matches(item, query, risk_filter) -> bool
    # Lọc file theo search query và risk level
    # High: score >= 0.5 | Medium: 0.3 <= score < 0.5 | Low: score < 0.3

node_has_match(node, query, risk_filter) -> bool
    # Đệ quy kiểm tra folder có chứa file match không

_risk_dot(score) -> str
    # 🔴 nếu >= 0.5 | 🟡 nếu >= 0.3 | 🟢 nếu < 0.3

_render_indented_button(label, key, depth) -> bool
    # Render st.button() với indent theo độ sâu thư mục
    # depth=0: full width | depth>0: spacer col + content col

render_tree(node, active_file, analyzed, query, risk_filter, parent_path, depth) -> None
    # Đệ quy render tree: folders trước, files sau (sort alpha)
    # Folder: toggle expand/collapse với st.session_state.tree_expanded
    # File: click → st.session_state.active_file = ... → st.rerun()
```

### Key naming conventions

```python
f"file_{_node_key(parent_path, value['name'])}"   # File button key
f"dir_{dir_key}"                                    # Folder button key
_node_key(parent, name) -> md5(f"{parent}/{name}")[:12]  # Unique key
```

---

## ⚙️ Backend: `backend/api.py` (~482 dòng)

### Các hàm chính

```python
# Language detection
get_lang(name: str) -> tuple[str, str, str]
    # Returns (icon, color, lang_name)

# File handling
extract_zip(zbytes: bytes) -> dict[str, str]
    # Bỏ qua: __MACOSX, .DS_Store, node_modules, venv, __pycache__, .git, .idea

build_entries_from_zip(zbytes: bytes) -> list[dict]
build_entries_from_files(uploaded_files) -> list[dict]
    # entry dict keys: name, fname, content, ext, lang

# Analysis engine
run_analysis(entries: list[dict]) -> tuple[list[dict], list[str], dict]
    # results: list[dict] với keys: name, fname, content, ext, lang,
    #          metrics, risk_score, line_risks
    # logs: list[str]
    # summary: {avg_risk, high, med, low, total_loc, models, timestamp}

# Risk scoring
compute_risk_score(m: dict) -> float  # 0.0 - 1.0
compute_line_risks(content, ext, file_cc) -> dict[int, 'high'|'med'|'low']

# Syntax highlight + rendering
esc(s: str) -> str          # HTML escape
tokenize_py(line: str) -> str    # Python syntax highlight HTML
render_code_line(line, num, risk, ext) -> str   # HTML dòng code
render_file_code(file_entry, analyzed) -> str   # HTML toàn bộ file

# Helpers
get_risk_label(score: float) -> tuple[str, str]    # (label, color)
get_risk_badge_class(score: float) -> str          # 'high'|'med'|'low'
get_risk_badge_html(score: float, text='') -> str  # HTML badge

# Reports (chưa dùng trong UI hiện tại)
generate_csv_report(results) -> str
generate_txt_report(results, summary, timestamp) -> str
generate_json_export(results, summary) -> dict
```

### Risk scoring formula (file-level)

```python
# compute_risk_score() — trả về 0.0-1.0
score = 0.0
if cc > 20:   score += 0.40
elif cc > 10: score += 0.25
elif cc > 5:  score += 0.10
if loc > 500: score += 0.25
elif loc > 200: score += 0.12
elif loc > 100: score += 0.05
if comment_ratio < 0.05: score += 0.10
if funcs > 15: score += 0.10
if decisions > 30: score += 0.15
random.seed(loc + cc)  # Deterministic noise
score += random.uniform(-0.05, 0.05)
score = clip(score, 0.02, 0.96)

# Phân loại mức (file-level):
# >= 0.5 → HIGH  |  >= 0.3 → MED  |  < 0.3 → LOW
```

### Line-level risk triggers

```python
# compute_line_risks() — trả về {line_num: 'high'|'med'|'low'}
'exec(' or 'eval('       → +0.7
'except:' (bare except)  → +0.5
'while True:'            → +0.5
line > 120 chars         → +0.3
'TODO' or 'FIXME'        → +0.2
def với > 4 params       → +0.4  (count commas > 4)
complex if (and/or + :)  → +0.3
file CC > 15             → ×1.4 multiplier

# Ngưỡng phân loại:
# score >= 0.55 → 'high'
# score >= 0.28 → 'med'
# t (có nội dung) → 'low'
# dòng trống → không có trong dict (None khi get)
```

### Render code line — màu 3 mức

```python
# render_code_line() trong api.py
if risk == 'high':
    cls = 'vs-cl hi'
    badge = '<span class="vs-rb hi">⚠ HIGH</span>'
elif risk == 'med':
    cls = 'vs-cl me'
    badge = '<span class="vs-rb me">⚡ MED</span>'
elif risk == 'low':
    cls = 'vs-cl lo'
    badge = '<span class="vs-rb lo">✓ LOW</span>'
```

| Mức | Nền (từ styles.py) | Border trái | Badge |
|---|---|---|---|
| HIGH | `rgba(184,50,61,.14)` | `rgba(184,50,61,.78)` 3px | Đỏ |
| MED | `rgba(173,123,43,.15)` | `rgba(173,123,43,.74)` 3px | Vàng |
| LOW | *(không có class, inline 'lo' chưa define trong styles)* | — | Xanh |

> ⚠️ **Bug tiềm ẩn**: `.vs-cl.lo` và `.vs-rb.lo` chưa được định nghĩa trong `styles.py`. Dòng LOW vẫn render nhưng không có màu nền/badge riêng.

### ML simulation (`make_model_results`)

> ⚠️ Hiện tại là **simulation** (hardcoded values + noise), **chưa tích hợp** `backend/models.py`

| Model | Base Accuracy | Base AUC | Base F1 |
|---|---|---|---|
| Logistic Regression | 74.5% | 79.6% | 47.2% |
| Random Forest | 80.4% | 78.3% | 45.6% |
| Neural Network | 85.1% | 79.3% | 29.0% |

Noise: `random.uniform(-0.04, 0.04)`, seed = `int(avg_risk * 1000)`

### Language support (LANG_MAP)

| Extension | Icon | Name |
|---|---|---|
| .py | 🐍 | Python |
| .java | ☕ | Java |
| .js | 🟨 | JavaScript |
| .jsx | ⚛️ | React JSX |
| .ts | 🔷 | TypeScript |
| .tsx | ⚛️ | React TSX |
| .c | ⚙️ | C |
| .cpp | ⚙️ | C++ |
| .cs | 🔷 | C# |
| .h | 📎 | Header |
| .hpp | 📎 | C++ Header |
| .go | 🐹 | Go |
| .rs | 🦀 | Rust |
| .rb | 💎 | Ruby |
| .php | 🐘 | PHP |
| .swift | 📱 | Swift |
| .kt | 🟣 | Kotlin |
| .sh | 🐚 | Shell |
| .html | 🌐 | HTML |
| .css | 🎨 | CSS |
| .sql | 🗄️ | SQL |

---

## 🤖 Backend: `backend/models.py` (~480 dòng) — UPDATED 2026-04-21

### Model catalogue

| Model | Thư viện | Ưu điểm SDP | Imbalance handling |
|---|---|---|---|
| Logistic Regression | sklearn | Baseline nhanh, interpretable | `class_weight=` |
| Random Forest | sklearn | Nonlinear, feature importance tốt | `class_weight=` |
| Neural Network (MLP) | sklearn | Pattern phức tạp | `sample_weight` |
| **XGBoost** | xgboost ≥ 3.2 | **#1 tabular SDP benchmarks 2020-25** | `scale_pos_weight` |
| **LightGBM** | lightgbm ≥ 4.6 | 3–10× nhanh hơn XGB, sparse features | `is_unbalance=True` |
| Gradient Boosting | sklearn | Fallback khi không có XGB/LGB | `sample_weight` |
| **Voting Ensemble (Soft)** | sklearn | Avg probabilities, giảm variance | — |
| **Stacking Ensemble** | sklearn | LR meta-learner, thường AUC cao nhất | — |

> Graceful fallback: nếu `xgboost`/`lightgbm`/`optuna` không có → tự động dùng thay thế sklearn.

### Kết quả smoke test trên KC1+KC2 (tune=False, 5-fold CV)

| Rank | Model | AUC | Recall | F1 | CV-AUC |
|---|---|---|---|---|---|
| 1 | **Stacking Ensemble** | **0.772** | 0.695 | 0.564 | 0.689±0.034 |
| 2 | Voting Ensemble (Soft) | 0.767 | 0.561 | 0.575 | 0.682±0.034 |
| 3 | Random Forest | 0.762 | 0.537 | 0.557 | 0.674±0.037 |
| 4 | XGBoost | 0.761 | 0.598 | 0.583 | 0.660±0.033 |
| 5 | Neural Network (MLP) | 0.758 | **0.707** | 0.550 | 0.624±0.042 |
| 6 | Logistic Regression | 0.755 | 0.683 | 0.566 | 0.691±0.033 |
| 7 | LightGBM | 0.742 | 0.549 | 0.526 | 0.656±0.040 |

### Class `ModelTrainer` — API chính

```python
from models import ModelTrainer

trainer = ModelTrainer(
    class_weight_dict=pp.class_weight_dict_,   # từ SDPPreprocessor
    include_ensembles=True,
)

# Train tất cả models + ensemble
results = trainer.train_all_models(
    X_tr, X_te, y_tr, y_te,
    tune=True,    # Optuna Bayesian HP search (~2-5 phút)
    cv_folds=5,
)

# Leaderboard sort by AUC
df = trainer.get_leaderboard(sort_by="roc_auc")

# Best model
best = trainer.best_model(metric="roc_auc")
print(best.name, best.roc_auc, best.recall)

# Logs
for line in trainer.get_logs():
    print(line)

# Save/Load
paths = trainer.save_all_models("backend/models")      # tất cả
path  = trainer.save_model("XGBoost", "backend/models") # một model
bundle = ModelTrainer.load_model(path)
# bundle = {'model': clf, 'metrics': {...}, 'name': str}

# List saved models
saved = ModelTrainer.list_saved_models("backend/models")
```

### `ModelResult` dataclass

```python
@dataclass
class ModelResult:
    name: str
    model: Any                 # fitted sklearn/xgb/lgb model
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float       # PR-AUC (quan trọng với imbalance)
    cv_roc_auc_mean: float     # 5-fold CV AUC mean
    cv_roc_auc_std: float      # 5-fold CV AUC std
    y_pred: np.ndarray
    y_proba: np.ndarray        # dùng cho ROC/PR curve plot
    train_time_s: float
    best_params: Dict          # best params từ Optuna/Grid
```

### Hyperparameter tuning

```python
# Optuna (preferred) — Bayesian optimisation
# tune=True khi gọi train_all_models()
# Mỗi model: 40-50 trials × 5-fold CV → ~2-5 phút tổng
# Metric tối ưu: ROC-AUC (StratifiedKFold, không dùng test set)

# GridSearchCV (fallback khi Optuna không có)
# Grid thưa hơn, nhanh hơn nhưng kém optimal
```

### Imbalance handling theo model

| Model | Cơ chế | Tham số |
|---|---|---|
| LR, RF | `class_weight` | `class_weight_dict` từ SDPPreprocessor |
| MLP | `sample_weight` | `n_total/(2*n_class)` per sample |
| XGBoost | `scale_pos_weight` | `n_neg / n_pos` tự tính |
| LightGBM | `is_unbalance=True` | LGB tự tính trọng số |
| Ensemble | inherited | Từ base models |

### Cross-validation một model riêng lẻ

```python
from models import cross_validate_model

cv_result = cross_validate_model(clf, X, y, cv_folds=10)
# {'roc_auc_mean': 0.78, 'roc_auc_std': 0.03,
#  'f1_mean': 0.56, 'recall_mean': 0.69, ...}
```

### Cài thư viện cần thiết

```bash
pip install xgboost lightgbm optuna
# Hoặc thêm vào requirements.txt:
xgboost>=2.0.0
lightgbm>=4.0.0
optuna>=3.0.0
```

---

## 🤖 Backend: `backend/models.py` — LEGACY NOTE

Class `DefectPredictionModels` — train/evaluate thật với sklearn + TensorFlow (optional):

```python
model = DefectPredictionModels(random_state=42)
model.initialize_models()                          # Khởi tạo LR, RF, NN
model.train_models(X_train, y_train, X_val, y_val) # Train toàn bộ
model.predict(X, model_name='all')                 # Predict nhãn
model.predict_proba(X)                             # Predict xác suất
model.evaluate_models(X_test, y_test)              # Metrics đánh giá
model.get_feature_importance(feature_names)        # RF feature importance
model.cross_validate(X, y, cv=5)                   # Cross-validation (LR + RF)
model.save_models(path)                            # Lưu .pkl + .h5
model.load_models(path)                            # Load từ disk
```

**Neural Network architecture (Keras nếu có TF, fallback MLPClassifier):**
```
Dense(64, relu) → BatchNorm → Dropout(0.3)
Dense(32, relu) → BatchNorm → Dropout(0.3)
Dense(16, relu) → BatchNorm
Dense(1, sigmoid)
# Compile: Adam(lr=0.001), binary_crossentropy
# EarlyStopping: monitor='val_loss', patience=10
```

**Logistic Regression config:**
```python
LogisticRegression(max_iter=1000, class_weight='balanced')
```

**Random Forest config:**
```python
RandomForestClassifier(n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2, class_weight='balanced')
```

> ⚠️ `backend/models.py` **CHƯA được gọi từ UI**. `api.py` dùng `make_model_results()` là simulation. Cần tích hợp thủ công nếu muốn train thật.

---

## 🧪 Backend: `backend/preprocessing.py` (~430 dòng) — UPDATED 2026-04-21

### Tổng quan pipeline

```
load_data(filepath)            ← ARFF + CSV auto-detect
        ↓
  DataCleaner.fit_transform()
  ├─ drop_duplicates()
  ├─ integrity_checks()        # LOC > 0
  ├─ fill_missing(median)
  ├─ clip_outliers(IQR×3)
  └─ drop_zero_variance()
        ↓
  split_stratified()           # train/test TRƯỚC khi feature engineer
        ↓
  FeatureEngineer.fit_transform(X_train)  # fit on train only!
  ├─ log1p(heavy-tailed cols)
  ├─ interaction features (v(g)×loc, ...)
  ├─ ratio features (v(g)/loc, ...)
  └─ select_features()
     ├─ Stage 1: MutualInformation top-k
     ├─ Stage 2: RandomForest importance top-k
     └─ Final: union(mi_top, rf_top)
        ↓
  StandardScaler (fit on X_train)
        ↓
  handle_imbalance(X_train)    # SMOTE / ADASYN — TRAIN ONLY
        ↓
  → (X_train_res, X_test_sc, y_train_res, y_test)
```

### Phát hiện thực tế từ KC1+KC2 (smoke test)

| Metric | Giá trị |
|---|---|
| Raw shape | 2631 rows × 22 cols |
| Duplicate rows removed | **1064** (40.4%!) |
| LOC ≤ 0 removed | 79 |
| Zero-variance cols dropped | `ev(g)`, `lOCodeAndComment` |
| Clean shape | 1488 rows × 20 cols |
| Train / Test split | 1190 / 298 (80/20 stratified) |
| Features selected | **20** (union MI + RF top-12) |
| Defect rate (raw) | 16.46% |
| Defect rate (train/test) | 27.5% (sau SMOTE khi có imbalanced-learn) |

> ⚠️ **Quan trọng**: 40% của KC1+KC2 là duplicate! Đây là nguồn gây overfitting lớn nhất trong các nghiên cứu SDP. Pipeline mới loại bỏ chúng đúng cách.

### Features được chọn (KC1+KC2)

```
['branchCountxloc', 'd', 'e', 'iv(g)', 'l', 'lOCode',
 'log_b', 'log_d', 'log_e', 'log_i', 'log_n',
 'log_total_Opnd', 'log_uniq_Opnd', 'log_v',
 't', 'total_Op', 'uniq_Opnd', 'v', 'v(g)', 'v(g)_per_loc']
```

Nhận xét: log-transformed Halstead metrics (`log_v`, `log_e`, `log_n`) và interaction feature `branchCountxloc` đều được chọn → xác nhận các giá trị này có tương quan phi tuyến với defect.

### Class `SDPPreprocessor` — API chính

```python
# Training
pp = SDPPreprocessor(
    iqr_multiplier=3.0,      # IQR × 3 cho outlier clipping
    n_select=12,             # top-k từ mỗi selection stage
    test_size=0.20,          # 80/20 split
    imbalance_strategy="smote",  # "smote" | "adasyn" | "none"
    random_state=42,
)
X_tr, X_te, y_tr, y_te = pp.fit_transform(df)

# Inference (single module / new file)
X_new_sc = pp.transform_new(X_new_df)

# Attributes sau fit_transform:
pp.cleaner_             # DataCleaner (fitted)
pp.engineer_            # FeatureEngineer (fitted)
pp.scaler_              # StandardScaler (fitted)
pp.selected_features_   # list[str] — tên features đã chọn
pp.class_weight_dict_   # {0: w0, 1: w1} — cho sklearn class_weight=
pp.report_              # dict thống kê toàn pipeline
```

### Lý do thiết kế từng bước

| Bước | Quyết định | Lý do |
|---|---|---|
| Outlier | **Clip** (IQR×3) thay vì remove | Extreme metrics (Halstead Volume=triệu) vẫn là defective module hợp lệ. Remove → mất minority class. |
| Missing | **Median** imputation | Mean nhạy cảm với heavy tail của SDP metrics. Median robust hơn. |
| Log1p | Áp dụng cho Halstead metrics | LOC, Volume, Effort phân phối power-law. Log normalize → models hội tụ nhanh hơn. |
| Split TRƯỚC feature selection | **Tách train/test trước** khi fit engineer | Nếu fit trên toàn bộ data → data leakage từ test set vào feature selection. |
| SMOTE | Oversampling **chỉ trên train** | Áp dụng SMOTE trước split → synthetic samples xuất hiện cả train lẫn test → inflate metrics. |
| Feature selection | **MI + RF union** | MI bắt non-linear univariate; RF bắt interactions. Union giữ thông tin bổ sung (DAOAFS principle). |
| StandardScaler | **Fit on train only** | MinMaxScaler nhạy với extreme values còn sót. StandardScaler robust hơn. |

### Imbalance strategy so sánh

| Strategy | Khi nào dùng | Ưu điểm | Nhược điểm |
|---|---|---|---|
| `"smote"` | **Default** (KC1/KC2) | Recall cao; không duplicate thật | Có thể tạo noisy samples ở boundary |
| `"adasyn"` | Khi SMOTE chưa đủ recall | Tập trung vào vùng khó phân loại | Nhạy với outliers |
| `"none"` | Dataset nhỏ / imbalanced-learn không có | Nhanh, ổn định | Dùng `class_weight_dict_` trong model thay thế |

> ⚠️ `imbalanced-learn` chưa được cài trong venv hiện tại. Fallback tự động sang `"none"` + `class_weight='balanced'`.

### Tích hợp vào app.py (Streamlit)

```python
# backend/api.py — thêm vào run_analysis() hoặc tạo endpoint riêng
from preprocessing import SDPPreprocessor, load_data

# Khi user upload CSV/ARFF dataset để train:
def train_from_dataset(filepath: str) -> dict:
    df = load_data(filepath)
    pp = SDPPreprocessor(imbalance_strategy="smote")
    X_tr, X_te, y_tr, y_te = pp.fit_transform(df)
    # → truyền vào DefectPredictionModels.train_models()
    return pp.report_

# Khi user muốn predict file source code mới:
def predict_new_file(metrics_dict: dict, pp: SDPPreprocessor) -> float:
    X_new = pd.DataFrame([metrics_dict])
    X_sc = pp.transform_new(X_new)  # reuse fitted pipeline
    # → model.predict_proba(X_sc)
```

```python
# frontend/app.py — hiển thị pipeline report
if pp.report_:
    st.metric("Duplicates removed", raw - clean)
    st.metric("Features selected", pp.report_['n_features'])
    st.metric("Defect rate", f"{pp.report_['raw_defect_rate']:.1%}")
    st.write(pp.report_['selected_features'])
```

### Cài imbalanced-learn (khi cần SMOTE)

```bash
pip install imbalanced-learn
# Thêm vào requirements.txt:
imbalanced-learn>=0.11.0
```

---

## 📐 Backend: `backend/code_metrics_extractor.py` (~403 dòng)

Class `CodeMetricsExtractor`:

```python
extractor = CodeMetricsExtractor()
df = extractor.extract_from_directory(tmp_dir)
# df columns: file_name, file_path, LOC, LOC_BLANK, LOC_TOTAL, LOC_COMMENTS,
#   LOC_CODE, FUNCTION_COUNT, CLASS_COUNT, CYCLOMATIC_COMPLEXITY,
#   DECISION_COUNT, COMMENT_RATIO, IMPORT_COUNT, RETURN_COUNT,
#   PARAMETER_COUNT (Python only), FIELD_COUNT (Java only)
```

**Ngôn ngữ hỗ trợ (extract_from_directory):**

| Extension | Method |
|---|---|
| .py | `extract_python_metrics()` |
| .java | `extract_java_metrics()` |
| .js / .jsx | `extract_js_metrics()` |
| .c / .h | `extract_c_metrics()` |
| .cpp / .hpp | `extract_cpp_metrics()` (→ alias extract_c_metrics) |
| .cs | `extract_csharp_metrics()` (→ alias extract_java_metrics) |

**Bỏ qua trong os.walk:** `.` prefix dirs, `node_modules`, `venv`, `__pycache__`, `build`, `dist`

**Cyclomatic complexity calculation:** `decision_points + 1`
- Python: `if + elif + for + while + except + and + or`
- Java: `if + else if + for + while + case + catch + && + ||`
- JavaScript: như Java + `? (ternary)` + switch
- C/C++: như Java + `case` + `#define` count

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
# TensorFlow: optional — fallback to sklearn MLPClassifier nếu không có
```

---

## 🔍 Luồng hoạt động UI

```
[1] User mở http://localhost:8501
    → TopBar: "DefectSight Workbench | 0 files | 0 LOC | Avg risk 0.0% | Waiting for input"
    → Left panel: Import Workspace (file_uploader + ZIP uploader + Run button)
    → Center: st.info("Upload files and run analysis...")
    → Right: "Run analysis to get model metrics..." caption
    → Console: trống

[2] User upload files/ZIP → Click "Run Analysis"
    → _run_analysis_from_inputs()
       → build_entries_from_zip() / build_entries_from_files() → list[dict entry]
       → run_analysis(entries)
          ① CodeMetricsExtractor.extract_from_directory(tmp_dir)
          ② compute_risk_score(metrics) cho từng file
          ③ compute_line_risks(content, ext, cc) cho từng file
          ④ make_model_results(avg_risk) — ML simulation
       → session_state.files = results
       → session_state.analysis = summary
       → session_state.analyzed = True
       → session_state.active_file = results[0]["name"]
       → st.rerun()

[3] Post-analysis state:
    → TopBar: cập nhật N files / LOC / Avg risk% / "Analyzed"
    → Left: Explorer panel
       - ds-pane-title "Explorer (N files)"
       - text_input "Find file" (tree_query)
       - selectbox "Risk filter" [All/High/Medium/Low]
       - selectbox "Model view" [All Models/LR/RF/NN]
       - render_tree() trong st.container(height=520)
         · File: icon + name + 🔴/🟡/🟢
         · Folder: ▾/▸ + name (expandable)
    → Center: ds-file-header + ds-metric-grid + ds-code-box
       · Dòng HIGH: nền đỏ nhạt + badge "⚠ HIGH"
       · Dòng MED: nền vàng nhạt + badge "⚡ MED"
       · Dòng LOW: không có nền (bug: .vs-cl.lo chưa style)
    → Right: Workspace Risk % + High/Med/Low st.metric + model progress bars
             + top 6 risky files (buttons)
    → Console: colored log lines

[4] User click file trong tree → st.session_state.active_file = name → rerun
[5] User click file trong "Top risky files" (right panel) → tương tự
[6] Không có nút "Reset/New Import" trong phiên bản hiện tại
    → Phải reload page để import lại
```

---

## 🐛 Gotchas & Notes

1. **Layout dùng `st.container(height=800)` không phải `position:fixed`**: Phiên bản hiện tại dùng Streamlit native columns và scrollable containers, không dùng CSS fixed positioning cho 3 cột chính. `position:fixed` chỉ còn trong comments cũ.

2. **`.vs-cl.lo` và `.vs-rb.lo` chưa có CSS**: Dòng LOW render class nhưng không có style tương ứng trong `styles.py`. Cần thêm nếu muốn highlight dòng LOW.

3. **Phân biệt file vs folder trong tree**: File entry có key `'content'`, folder node thì không. Pattern: `isinstance(value, dict) and 'content' in value`.

4. **Key uniqueness trong tree**: Dùng `md5(f"{parent_path}/{name}")[:12]` → tránh collision khi có file cùng tên ở thư mục khác.

5. **`make_model_results()` là simulation**: Không train thật. Model thật nằm trong `backend/models.py` nhưng **chưa được tích hợp** vào UI flow.

6. **Không có "Reset" button**: Sau khi analyze, không có nút nào để quay lại Import state. User phải refresh trang.

7. **`run_analysis()` dùng `tempfile.mkdtemp()`**: File được ghi ra temp dir, chạy extractor, rồi xóa (`shutil.rmtree`). Nếu extractor crash, `finally` vẫn dọn dẹp.

8. **import trong `api.py`**: `from code_metrics_extractor import CodeMetricsExtractor` — import trực tiếp (không prefix `backend.`). Hoạt động vì `api.py` chạy với `backend/` trong sys.path khi launch từ `run.bat` / `streamlit run frontend/app.py`.

9. **`dashboard/Dashboard.py`**: Tồn tại trong dự án nhưng **không được gọi** từ `app.py`. Là file riêng biệt, chưa tích hợp.

10. **`backend/features.py`, `preprocessing.py`, `evaluation.py`, `database.py`**: Tồn tại nhưng **không được import/gọi** từ UI hiện tại. Là legacy code từ phiên bản cũ.

---

## 🔍 Backend: `backend/explainability.py` (~850 dòng) — NEW 2026-04-21

### Tổng quan

Module XAI (Explainable AI) giải thích TẠI SAO model dự đoán một module là defective, dùng SHAP (global + local) và LIME (local fallback).

### Explainer types (tự động chọn)

| Model type | Explainer | Lý do |
|---|---|---|
| RandomForest, XGBoost, LightGBM, GradientBoosting | `TreeExplainer` | Exact Shapley values qua tree path — O(depth), không sampling |
| LogisticRegression | `LinearExplainer` | Exact cho linear models, fast |
| MLP, Voting, Stacking | `PermutationExplainer` | Model-agnostic, sample permutations |

### Class `ModelExplainer` — API

```python
from explainability import explain_model, plot_shap_comparison

# Tạo và fit explainer (gọi sau khi train)
exp = explain_model(
    model        = rf_model,
    X_train      = X_tr,          # background dataset cho explainer
    X_test       = X_te,
    feature_names= pp.selected_features_,
    model_name   = "Random Forest",
)

# ── GLOBAL (toàn bộ test set) ──
fig_bar  = exp.plot_global_importance(X_te, top_n=15)  # Plotly bar
fig_bee  = exp.plot_beeswarm(X_te, top_n=12)           # matplotlib (st.pyplot)
fig_heat = exp.plot_feature_heatmap(X_te, top_n=10)    # Plotly heatmap

# ── LOCAL (module tại index i) ──
fig_wf   = exp.plot_local_waterfall(X_te, instance_idx=7, top_n=12)   # Plotly waterfall
fig_lime = exp.plot_lime_explanation(X_te, instance_idx=7, top_n=10)  # Plotly bar (LIME)
fig_dep  = exp.plot_dependency(X_te, "v(g)", "loc")    # Plotly scatter

# Narrative text (markdown)
text = exp.generate_narrative(X_te, 7, pred_proba[7])  # str markdown

# ── MULTI-MODEL (tab Đánh Giá) ──
fig_cmp = plot_shap_comparison(explainers_dict, X_te, top_n=10)
fig_bkd = plot_prediction_breakdown(explainers_dict, X_te, instance_idx=7)
```

### SHAP vs LIME — khi nào dùng cái nào

| | SHAP | LIME |
|---|---|---|
| **Lý thuyết** | Shapley values — consistent, fair attribution | Local linear surrogate |
| **Ưu điểm** | Globally consistent, additive, model-agnostic | Trực quan, dễ tune |
| **Dùng cho** | Verify important features globally | Cross-validate SHAP cho 1 instance |
| **Agree?** | Nếu SHAP và LIME ĐỒNG Ý → attribution đáng tin | Nếu DISAGREE → cần kiểm tra |
| **Tốc độ** | TreeExplainer: nhanh | Luôn cần ~500 samples |

### Kết quả smoke test (Logistic Regression trên KC1+KC2)

```
Top SHAP features (mean |SHAP|):
  log_total_Opnd : 0.8562   ← Halstead total operands (log)
  uniq_Opnd      : 0.7972   ← Unique operands
  v(g)           : 0.6389   ← Cyclomatic complexity
  log_b          : 0.5039   ← Halstead bugs estimate (log)
  log_v          : 0.4359   ← Halstead volume (log)
```

Insight: Halstead metrics (operand complexity) quan trọng hơn cyclomatic complexity
theo LinearExplainer — phù hợp với literature SDP (Menzies 2004).

### Tích hợp vào Streamlit — tab "Đánh Giá"

```python
# frontend/app.py — thêm section sau khi train xong

import sys; sys.path.insert(0, "backend")
from explainability import ModelExplainer, explain_model, plot_shap_comparison

# Sau khi train:
# st.session_state.explainers = {}
# for name, res in trainer.results_.items():
#     exp = explain_model(res.model, X_tr, X_te, feature_names, name)
#     st.session_state.explainers[name] = exp

# Tab Đánh Giá — Global XAI:
if st.session_state.get("explainers"):
    exp = st.session_state.explainers[selected_model]

    st.subheader("Feature Importance (SHAP)")
    st.plotly_chart(exp.plot_global_importance(X_te), use_container_width=True)

    st.subheader("SHAP Beeswarm")
    fig_bee = exp.plot_beeswarm(X_te)
    st.pyplot(fig_bee, use_container_width=True)

    st.subheader("Feature Heatmap")
    st.plotly_chart(exp.plot_feature_heatmap(X_te), use_container_width=True)

    if len(st.session_state.explainers) > 1:
        st.subheader("Cross-model SHAP Comparison")
        st.plotly_chart(
            plot_shap_comparison(st.session_state.explainers, X_te),
            use_container_width=True,
        )

# Tab Dự Đoán — Local XAI cho module đang xem:
if st.session_state.get("explainers") and active_file:
    instance_idx = ...  # lấy từ active_file index
    exp = st.session_state.explainers[selected_model]

    st.subheader("Why is this module risky? (SHAP)")
    st.plotly_chart(
        exp.plot_local_waterfall(X_te, instance_idx),
        use_container_width=True,
    )

    st.subheader("LIME Explanation")
    st.plotly_chart(
        exp.plot_lime_explanation(X_te, instance_idx),
        use_container_width=True,
    )

    st.subheader("Risk Narrative")
    st.markdown(exp.generate_narrative(X_te, instance_idx, pred_proba))
```

### Cài thư viện

```bash
pip install shap lime
# requirements.txt:
shap>=0.44.0
lime>=0.2.0
```

---

## 📋 Changelog

### 2026-04-21 (session hiện tại)
- **[ARCH] Refactor app.py sang 5 Tabs**: Đại tu kiến trúc layout từ 3-column sang 5 chức năng (Workspace, Dashboard, Training, Evaluation, Prediction).
- **[NEW] AI Risk Assessment (Groq)**: Thêm ackend/ai_reviewer.py. Tích hợp LLM (Llama 3.3, Mixtral) phân tích codebase xuất báo cáo Markdown chuyên sâu thay cho dự đoán ML.
- **[FIX] Importlib module reload & Naming Conflict**: 
  - Đổi tên ackend/models.py thành ackend/ml_models.py để tránh xung đột cấu trúc.
  - Thay thế importlib.util thành import tiêu chuẩn để tránh lỗi AttributeError với class ModelTrainer (dataclass).
- **[FIX] Plotly & Pandas UI Bugs**:
  - Sửa crash Plotly Heatmap (do parser nhận dải màu có mã Hex 8 chữ số bị lỗi). Chuyển về linear gradient cơ bản.
  - Sửa lỗi trong biểu đồ Scatterpolar khai báo illcolor bằng cách thêm helper custom _hex_to_rgba để áp dụng độ trong suốt tiêu chuẩn 
gba().
  - Sửa lỗi format string :.3f của bảng dataframe (pandas styler.format yêu cầu {:.3f}).
- **[DOC] CLAUDE.md rewrite**: Cập nhật toàn bộ để phản ánh kiến trúc và bugs đã fixed.
- **[NEW] ackend/preprocessing.py rewrite toàn bộ**: Pipeline 8-stage cho NASA SDP datasets.
  - load_data(): auto-detect ARFF / CSV, parse robust.
  - DataCleaner: duplicate removal (phát hiện 40% KC1+KC2 là duplicate!).
  - FeatureEngineer: log1p transforms, 2-stage MI+RF feature selection.
- **[NEW] ackend/ml_models.py rewrite toàn bộ**: Tên cũ models.py. Gồm 7 models + 2 ensembles, Optuna tuning, ModelTrainer.


### 2026-04-20 (session trước)
- **[FIX] Layout overlap**: Chuyển từ `position:fixed` sang `st.columns([1.3, 3.0, 1.5])` + `st.container(height=800)`.
- **[REFACTOR] Frontend tách thành modules**: `frontend/ui/state.py`, `styles.py`, `tree.py`.
- **[FIX] Nested columns bug**: Không còn dùng fixed positioning → nested columns không cần override.
- **[NEW] Risk highlighting 3 mức**: HIGH/MED/LOW với màu sắc riêng (LOW chưa có CSS).
- **[NEW] File tree với emoji risk**: + collapse/expand folders.
- **[NEW] Explorer filters**: search query + risk filter + model selector.
- **[CHANGE] Color scheme**: Đổi từ VS Code blue (`#007acc`) sang dark teal (`#2f9d8a`).
- **[CHANGE] Font**: Đổi từ Inter sang Manrope (UI) + JetBrains Mono (code).

### 2026-04-16 (sessions trước)
- **[INIT] Refactor sang frontend/backend structure**: Tách `app.py` monolith thành `frontend/` + `backend/`.
- **[NEW] `backend/api.py`**: Tập trung toàn bộ business logic, tách khỏi UI.
- **[FIX] File uploader spacing**: Dùng `margin-top` thay cho spacer div.
- **[NEW] `compute_line_risks()`**: Thêm `'low'` cho dòng không bị flag.
