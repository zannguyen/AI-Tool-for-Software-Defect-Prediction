"""
AI-Assisted Software Defect Prediction Tool
Công cụ dự đoán lỗi phần mềm sử dụng Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models import DefectPredictionModels, create_sample_data
from src.evaluation import ModelEvaluator
from src.preprocessing import (
    load_dataset, clean_data, extract_features,
    split_data, scale_features, get_data_summary
)
from src.code_metrics_extractor import CodeMetricsExtractor
from src.database import HistoryDatabase

# Import modules from new directories
from models import ModelManager
from reports.ReportGenerator import ReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Dự Đoán Lỗi Phần Mềm",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        padding: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .risk-high {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #ffa500;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #4caf50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Khởi tạo các biến session state"""
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'train_results' not in st.session_state:
        st.session_state.train_results = None
    if 'feature_names' not in st.session_state:
        st.session_state.feature_names = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'db' not in st.session_state:
        st.session_state.db = HistoryDatabase("database/history.db")
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'selected_session' not in st.session_state:
        st.session_state.selected_session = None
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager("models")
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator("reports")


def load_sample_data():
    """Tải hoặc tạo dữ liệu mẫu"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'kc1_kc2.csv')
        if os.path.exists(data_path):
            df = load_dataset(data_path)
            return df
    except:
        pass
    return create_sample_data(1000)


def show_header():
    """Hiển thị header chính"""
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #1f77b4; margin-bottom: 5px;">🛡️ Công Cụ AI Dự Đoán Lỗi Phần Mềm</h1>
        <p style="color: #666; font-size: 1.1rem;">AI-Assisted Software Defect Prediction Tool</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")


def show_sidebar():
    """Hiển thị sidebar"""
    st.sidebar.title("📋 Điều Hướng")
    st.sidebar.markdown("---")

    # Thông tin dự án
    st.sidebar.markdown("### Thông Tin Dự Án")
    st.sidebar.info("""
    **Dự án:** P7 - AI Tool for Software Defect Prediction

    **Môn học:** Software Measurement & Analysis

    **Nhóm:** 5 sinh viên
    """)

    # Menu chính
    menu = st.sidebar.radio(
        "📌 Menu",
        [
            "🏠 Trang Chủ",
            "📊 Dashboard",
            "📂 Tải Dữ Liệu",
            "🤖 Huấn Luyện Model",
            "🔮 Dự Đoán",
            "📊 Đánh Giá",
            "💾 Xuất Báo Cáo",
            "📜 Lịch Sử",
            "📁 Models",
            "ℹ️ Giới Thiệu"
        ]
    )

    st.sidebar.markdown("---")

    # Trạng thái
    st.sidebar.markdown("### 📈 Trạng Thái")

    if st.session_state.data_loaded:
        st.sidebar.success("✅ Đã tải dữ liệu")
    else:
        st.sidebar.warning("⚠️ Chưa tải dữ liệu")

    if st.session_state.predictions_made:
        st.sidebar.success("✅ Đã huấn luyện model")
    else:
        st.sidebar.warning("⚠️ Chưa huấn luyện")

    # Lịch sử phân tích
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Lịch Sử")

    # Lấy danh sách sessions
    try:
        sessions = st.session_state.db.get_all_sessions()

        if sessions:
            # Hiển thị danh sách sessions
            for session in sessions[:10]:  # Chỉ hiển thị 10 cái mới nhất
                session_time = session['timestamp']
                files_count = session['files_count']
                source_type = session['source_type']

                # Tạo nút click được
                if st.sidebar.button(
                    f"📄 {session_time[:16]} ({files_count} files)",
                    key=f"session_{session['session_id']}"
                ):
                    # Khi click vào session, hiển thị chi tiết
                    st.session_state.selected_session = session['session_id']
                    st.rerun()

            # Nút xem tất cả
            if len(sessions) > 10:
                st.sidebar.markdown(f"📊 Còn {len(sessions) - 10} phiên khác...")
        else:
            st.sidebar.info("Chưa có lịch sử")
    except:
        st.sidebar.info("Chưa có lịch sử")

    return menu


def show_home_page():
    """Trang chủ"""
    show_header()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        ## Chào Mừng! 👋

        Công cụ này sử dụng **Machine Learning** để dự đoán các lỗi phần mềm dựa trên các metrics của code.

        ---

        ### 🎯 Các Tính Năng Chính:

        1. **📂 Tải Dữ Liệu** - Upload file CSV hoặc sử dụng dữ liệu mẫu

        2. **🤖 Huấn Luyện Model** - Train 3 model ML:
           - Logistic Regression
           - Random Forest
           - Neural Network (MLP)

        3. **🔮 Dự Đoán** - Dự đoán lỗi cho module mới

        4. **📊 Trực Quan** - Heatmap đánh giá rủi ro

        5. **📈 Đánh Giá** - So sánh các chỉ số hiệu suất

        6. **💾 Xuất Báo Cáo** - Xuất kết quả ra file

        ---

        ### 📊 Các Metrics Được Hỗ Trợ:
        - **LOC** - Số dòng code
        - **Cyclomatic Complexity** - Độ phức tạp
        - **Coupling** - Liên kết
        - **Code Churn** - Số dòng thay đổi
        - Và nhiều metrics khác...

        """)

    # Nút bắt đầu
    st.markdown("---")

    if not st.session_state.data_loaded:
        st.info("👋 Chào mừng! Vui lòng tải dữ liệu thực tế của bạn.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📄 Cách 1: Từ CSV
            Upload file CSV chứa metrics của code
            """)

        with col2:
            st.markdown("""
            ### 📁 Cách 2: Từ Source Code
            Upload trực tiếp các file code (.py, .java, .js, ...)
            """)

        st.info("💡 Vào menu **'📂 Tải Dữ Liệu'** ở thanh bên trái để tải dữ liệu thực tế!")
    else:
        st.success("✅ Dữ liệu đã sẵn sàng! Vui lòng chọn menu bên trái.")


def show_data_upload_page():
    """Trang tải dữ liệu"""
    st.markdown('<p class="sub-header">📂 Tải Dữ Liệu</p>', unsafe_allow_html=True)

    # Tab options
    tab1, tab2 = st.tabs(["📄 Từ CSV", "📁 Từ Thư Mục Code"])

    with tab1:
        st.markdown("""
        ### Yêu Cầu Dữ Liệu CSV

        Upload file CSV chứa các metrics của code. File cần có:
        - Các cột feature (metrics như LOC, complexity, coupling,...)
        - Cột target tên là `LABEL` hoặc `defects` (nhị phân: 0 hoặc 1)
        """)

        uploaded_file = st.file_uploader("📁 Chọn file CSV", type="csv", key="csv_uploader")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.session_state.data_loaded = True

                X, y = extract_features(df)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.feature_names = list(X.columns)

                st.success(f"✅ Đã tải thành công {len(df)} mẫu!")

                # Hiển thị data preview
                st.markdown("### 👀 Xem Trước Dữ Liệu")
                st.dataframe(df.head(10), use_container_width='stretch')

                # Thống kê
                st.markdown("### 📊 Thống Kê Dữ Liệu")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Tổng Mẫu", len(df))
                with col2:
                    st.metric("Số Features", len(X.columns))
                with col3:
                    st.metric("Module Lỗi", y.sum())
                with col4:
                    st.metric("Tỷ Lệ Lỗi", f"{(y.sum()/len(y))*100:.1f}%")

                # Nút tiếp tục
                st.markdown("---")
                if st.button("🚀 Tiếp Tục Phân Tích", type="primary", use_container_width='stretch', key="btn_continue_csv"):
                    st.success("✅ Đã sẵn sàng! Vào menu 'Huấn Luyện Model' để tiếp tục.")
                    st.balloons()

            except Exception as e:
                st.error(f"❌ Lỗi khi tải file: {str(e)}")

    with tab2:
        st.markdown("""
        ### 📁 Trích Xuất Metrics Từ Source Code

        Tải lên thư mục chứa source code. Hệ thống sẽ tự động trích xuất các metrics:
        - **LOC**: Số dòng code
        - **Cyclomatic Complexity**: Độ phức tạp
        - **Function Count**: Số hàm
        - **Class Count**: Số lớp
        - **Comment Ratio**: Tỷ lệ comment
        - Và nhiều metrics khác...

        **Hỗ trợ**: Python, Java, JavaScript, C/C++, C#
        """)

        # Upload folder (as multiple files)
        uploaded_files = st.file_uploader(
            "📂 Chọn các file code",
            type=['py', 'java', 'js', 'c', 'cpp', 'cs', 'h', 'hpp'],
            accept_multiple_files=True,
            key="code_uploader"
        )

        # Nút phân tích
        if uploaded_files:
            st.success(f"✅ Đã chọn {len(uploaded_files)} file!")
            if st.button("🔍 Phân Tích Metrics", type="primary", use_container_width='stretch'):
                try:
                    with st.spinner("Đang trích xuất metrics..."):
                        # Save files to temp directory
                        import tempfile
                        import shutil

                        temp_dir = tempfile.mkdtemp()

                        for uploaded_file in uploaded_files:
                            with open(os.path.join(temp_dir, uploaded_file.name), 'wb') as f:
                                f.write(uploaded_file.getvalue())

                        # Extract metrics
                        extractor = CodeMetricsExtractor()
                        df = extractor.extract_from_directory(temp_dir)

                        # Clean up
                        shutil.rmtree(temp_dir)

                    if not df.empty:
                        # Prepare for prediction (add LABEL column)
                        df['LABEL'] = 0  # Default: no defect

                        st.session_state.df = df
                        st.session_state.data_loaded = True

                        # Get feature columns
                        feature_cols = [col for col in df.columns
                                       if col not in ['file_path', 'file_name', 'LABEL']]
                        X = df[feature_cols]
                        y = df['LABEL']

                        st.session_state.X = X
                        st.session_state.y = y
                        st.session_state.feature_names = feature_cols

                        # Lưu vào database
                        file_names = [f.name for f in uploaded_files]
                        session_id = st.session_state.db.save_session(
                            source_type="code_files",
                            files=file_names,
                            df=df
                        )
                        st.session_state.current_session_id = session_id

                        st.success(f"✅ Đã trích xuất metrics từ {len(df)} file!")

                        # Show metrics
                        st.markdown("### 📊 Metrics Đã Trích Xuất")
                        st.dataframe(df, use_container_width='stretch')

                        # Download CSV option
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Tải CSV",
                            data=csv,
                            file_name="extracted_metrics.csv",
                            mime="text/csv"
                        )

                        st.warning("⚠️ Lưu ý: Label = 0 (giả định không có lỗi). Bạn cần đánh dấu thủ công các module có lỗi trong CSV nếu có.")

                        # Nút tiếp tục
                        st.markdown("---")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("🚀 Tiếp Tục Phân Tích", type="primary", use_container_width='stretch'):
                                st.success("✅ Đã sẵn sàng! Vào menu 'Huấn Luyện Model' để tiếp tục.")
                                st.balloons()

                except Exception as e:
                    st.error(f"❌ Lỗi khi trích xuất: {str(e)}")


def show_model_training_page():
    """Trang huấn luyện model"""
    st.markdown('<p class="sub-header">🤖 Huấn Luyện Model</p>', unsafe_allow_html=True)

    if not st.session_state.data_loaded:
        st.warning("⚠️ Vui lòng tải dữ liệu trước!")
        return

    # Kiểm tra số lượng mẫu
    X = st.session_state.get('X')
    if X is None or len(X) < 10:
        st.error(f"❌ Dữ liệu không đủ! Cần ít nhất 10 mẫu để huấn luyện. Hiện tại: {len(X) if X is not None else 0} mẫu")

        # Tạo dữ liệu mẫu để test
        st.markdown("### 📊 Tạo Dữ Liệu Mẫu Để Test")

        if st.button("🎲 Tạo Dữ Liệu Mẫu (200 mẫu)", type="secondary"):
            import numpy as np

            np.random.seed(42)
            n_samples = 200

            LOC = np.random.randint(50, 500, n_samples)
            CYCLOMATIC_COMPLEXITY = np.random.randint(1, 30, n_samples)
            FUNCTION_COUNT = np.random.randint(1, 20, n_samples)
            CLASS_COUNT = np.random.randint(0, 10, n_samples)
            COMMENT_RATIO = np.random.uniform(0, 0.3, n_samples)
            DECISION_COUNT = np.random.randint(0, 15, n_samples)
            CODE_CHURN = np.random.randint(0, 200, n_samples)
            COUPLING = np.random.randint(1, 20, n_samples)

            defect_prob = (
                0.3 * (CYCLOMATIC_COMPLEXITY > 15).astype(int) +
                0.3 * (COMMENT_RATIO < 0.1).astype(int) +
                0.2 * (CODE_CHURN > 100).astype(int) +
                0.2 * (LOC > 300).astype(int)
            )
            LABEL = (np.random.random(n_samples) < defect_prob * 0.5).astype(int)

            df = pd.DataFrame({
                'LOC': LOC,
                'CYCLOMATIC_COMPLEXITY': CYCLOMATIC_COMPLEXITY,
                'FUNCTION_COUNT': FUNCTION_COUNT,
                'CLASS_COUNT': CLASS_COUNT,
                'COMMENT_RATIO': COMMENT_RATIO,
                'DECISION_COUNT': DECISION_COUNT,
                'CODE_CHURN': CODE_CHURN,
                'COUPLING': COUPLING,
                'LABEL': LABEL
            })

            st.session_state.df = df
            feature_cols = [col for col in df.columns if col != 'LABEL']
            X = df[feature_cols]
            y = df['LABEL']

            st.session_state.X = X
            st.session_state.y = y
            st.session_state.feature_names = feature_cols
            st.session_state.data_loaded = True

            st.success(f"Da tao {len(df)} mau mau! Bay gio co the bat dau huan luyen.")
            st.rerun()

        return

    # Cấu hình huấn luyện
    st.markdown("### ⚙️ Cấu Hình Huấn Luyện")

    col1, col2, col3 = st.columns(3)

    with col1:
        test_size = st.slider("📊 Tỷ Lệ Test", 0.1, 0.4, 0.2, help="Tỷ lệ dữ liệu test so với tổng dữ liệu")

    with col2:
        random_state = st.number_input("🔢 Random State", value=42, help="Seed cho việc random")

    with col3:
        stratify = st.checkbox("🔀 Stratified Split", value=True, help="Giữ nguyên tỷ lệ nhãn trong train/test")

    st.markdown("---")

    # Chọn models
    st.markdown("### 🤖 Chọn Models")

    col1, col2, col3 = st.columns(3)

    with col1:
        use_lr = st.checkbox("✅ Logistic Regression", value=True)
    with col2:
        use_rf = st.checkbox("✅ Random Forest", value=True)
    with col3:
        use_nn = st.checkbox("✅ Neural Network (MLP)", value=True)

    # Nút train
    st.markdown("---")

    if st.button("🚀 Huấn Luyện Tất Cả Models", type="primary", use_container_width='stretch'):
        with st.spinner("Đang huấn luyện models... Vui lòng đợi!"):
            try:
                X = st.session_state.X
                y = st.session_state.y

                # Chia dữ liệu
                if stratify:
                    X_train, X_test, y_train, y_test = split_data(
                        X.values, y.values, test_size=test_size,
                        random_state=random_state
                    )
                else:
                    X_train, X_test, y_train, y_test = split_data(
                        X.values, y.values, test_size=test_size,
                        random_state=random_state, stratify=None
                    )

                st.session_state.X_test = X_test
                st.session_state.y_test = y_test

                # Khởi tạo và train models
                models = DefectPredictionModels(random_state=random_state)
                models.initialize_models()
                models.train_models(X_train, y_train)

                # Đánh giá
                results = models.evaluate_models(X_test, y_test)
                probabilities = models.predict_proba(X_test)

                # Lưu kết quả
                st.session_state.models = models
                st.session_state.train_results = results
                st.session_state.probabilities = probabilities
                st.session_state.predictions_made = True

                st.success("✅ Huấn luyện hoàn tất!")

                # Hiển thị kết quả
                st.markdown("### 📊 Kết Quả Huấn Luyện")

                for model_name, metrics in results.items():
                    with st.expander(f"📌 {model_name.replace('_', ' ').title()}"):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        col2.metric("Precision", f"{metrics['precision']:.4f}")
                        col3.metric("Recall", f"{metrics['recall']:.4f}")
                        col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")

                        col1, col2 = st.columns(2)
                        col1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                        col2.metric("Specificity", f"{metrics.get('specificity', 0):.4f}")

                        # Confusion matrix
                        st.markdown("**Confusion Matrix:**")
                        cm = metrics['confusion_matrix']
                        cm_df = pd.DataFrame(
                            cm,
                            index=['Thực tế: Không Lỗi', 'Thực tế: Có Lỗi'],
                            columns=['Dự đoán: Không Lỗi', 'Dự đoán: Có Lỗi']
                        )
                        st.dataframe(cm_df)

            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình huấn luyện: {str(e)}")

    # Hiển trạng thái
    if st.session_state.predictions_made:
        st.success("✅ Models đã được huấn luyện! Vào mục 'Dự Đoán' để xem kết quả.")


def show_predictions_page():
    """Trang dự đoán"""
    st.markdown('<p class="sub-header">🔮 Dự Đoán Lỗi</p>', unsafe_allow_html=True)

    if not st.session_state.predictions_made:
        st.warning("⚠️ Vui lòng huấn luyện model trước!")
        return

    st.markdown("### 🎯 Lựa Chọn Đầu Vào")

    prediction_option = st.radio(
        "Chọn loại dữ liệu dự đoán:",
        ["Sử Dụng Dữ Liệu Test", "Nhập Metrics Thủ Công"]
    )

    if prediction_option == "Sử Dụng Dữ Liệu Test":
        if st.session_state.X_test is not None:
            models = st.session_state.models
            probabilities = st.session_state.probabilities

            # Lấy 10 mẫu đầu
            n_samples = min(10, len(st.session_state.X_test))

            # Dự đoán từng model
            predictions = models.predict(st.session_state.X_test[:n_samples])

            # Ensemble prediction
            ensemble_proba = (
                probabilities['logistic_regression'][:n_samples] +
                probabilities['random_forest'][:n_samples] +
                probabilities['neural_network'][:n_samples]
            ) / 3

            # Tạo dataframe kết quả
            results_df = pd.DataFrame({
                'Module': [f'Module_{i+1}' for i in range(n_samples)],
                'LR_Probability': probabilities['logistic_regression'][:n_samples],
                'RF_Probability': probabilities['random_forest'][:n_samples],
                'NN_Probability': probabilities['neural_network'][:n_samples],
                'Ensemble_Probability': ensemble_proba,
                'Dự Đoán': ['Có Lỗi' if p >= 0.5 else 'Không Lỗi' for p in ensemble_proba]
            })

            if st.session_state.y_test is not None:
                results_df['Thực Tế'] = ['Có Lỗi' if y == 1 else 'Không Lỗi' for y in st.session_state.y_test[:n_samples]]
                results_df['Đúng'] = results_df['Dự Đoán'].str.contains('Lỗi') == results_df['Thực Tế'].str.contains('Lỗi')

            st.markdown("### 📋 Kết Quả Dự Đoán")
            st.dataframe(results_df, use_container_width='stretch')

            # Risk assessment
            st.markdown("### 🔴 Đánh Giá Rủi Ro")

            risk_data = pd.DataFrame({
                'Module': [f'Module_{i+1}' for i in range(n_samples)],
                'Xác Suất Lỗi': ensemble_proba,
                'Mức Độ Rủi Ro': ['CAO' if p >= 0.5 else 'TRUNG BÌNH' if p >= 0.3 else 'THẤP'
                                  for p in ensemble_proba]
            }).sort_values('Xác Suất Lỗi', ascending=False)

            # Bar chart
            fig = px.bar(
                risk_data,
                x='Module',
                y='Xác Suất Lỗi',
                color='Mức Độ Rủi Ro',
                color_discrete_map={
                    'CAO': '#e74c3c',
                    'TRUNG BÌNH': '#f39c12',
                    'THẤP': '#27ae60'
                },
                title='🔴 Xác Suất Lỗi Theo Module',
                range_y=[0, 1]
            )

            # Thêm ngưỡng
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Ngưỡng cao")
            fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Ngưỡng thấp")

            st.plotly_chart(fig, use_container_width='stretch')

            # Table chi tiết
            st.markdown("### 📊 Bảng Chi Tiết Rủi Ro")

            for idx, row in risk_data.iterrows():
                risk_color = "🔴" if row['Mức Độ Rủi Ro'] == 'CAO' else "🟡" if row['Mức Độ Rủi Ro'] == 'TRUNG BÌNH' else "🟢"
                st.write(f"{risk_color} **{row['Module']}**: Xác suất lỗi = {row['Xác Suất Lỗi']:.4f} ({row['Mức Độ Rủi Ro']})")

    else:  # Nhập thủ công
        st.markdown("### ✏️ Nhập Metrics")

        feature_names = st.session_state.get('feature_names', [])

        if feature_names:
            st.info("Nhập giá trị cho các metrics:")

            input_values = {}
            cols = st.columns(3)

            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    input_values[feature] = st.number_input(
                        f"📊 {feature}",
                        value=0.0,
                        step=1.0,
                        help=f"Nhập giá trị cho {feature}"
                    )

            if st.button("🔮 Dự Đoán", type="primary", use_container_width='stretch'):
                X_input = np.array([[input_values[f] for f in feature_names]])

                models = st.session_state.models
                probabilities = models.predict_proba(X_input)

                st.markdown("### 🎯 Kết Quả Dự Đoán")

                for model_name, proba in probabilities.items():
                    prob = proba[0]
                    risk = "CAO" if prob >= 0.5 else "TRUNG BÌNH" if prob >= 0.3 else "THẤP"
                    risk_icon = "🔴" if risk == "CAO" else "🟡" if risk == "TRUNG BÌNH" else "🟢"

                    col1, col2 = st.columns(2)
                    col1.write(f"**{model_name.replace('_', ' ').title()}:**")
                    col2.write(f"{prob:.4f} {risk_icon} ({risk})")

                # Ensemble
                ensemble_proba = np.mean(list(probabilities.values()))
                ensemble_risk = "CAO" if ensemble_proba >= 0.5 else "TRUNG BÌNH" if ensemble_proba >= 0.3 else "THẤP"
                ensemble_icon = "🔴" if ensemble_risk == "CAO" else "🟡" if ensemble_risk == "TRUNG BÌNH" else "🟢"

                st.markdown("---")
                st.markdown(f"### 🎯 Dự Đoán Tổng Hợp: {ensemble_proba:.4f} {ensemble_icon}")
                st.markdown(f"**Mức Độ Rủi Ro: {ensemble_risk}**")


def show_evaluation_page():
    """Trang đánh giá"""
    st.markdown('<p class="sub-header">📊 Đánh Giá Model</p>', unsafe_allow_html=True)

    if st.session_state.train_results is None:
        st.warning("⚠️ Vui lòng huấn luyện model trước!")
        return

    results = st.session_state.train_results

    # So sánh metrics
    st.markdown("### 📈 So Sánh Hiệu Suất Models")

    comparison_data = []
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'ROC-AUC': metrics['roc_auc'],
            'Specificity': metrics.get('specificity', 0)
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width='stretch')

    # Biểu đồ so sánh
    st.markdown("### 📊 Biểu Đồ So Sánh")

    fig = go.Figure()

    models_list = list(results.keys())
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']

    for metric, color in zip(metrics_list, colors):
        values = [results[m][metric] for m in models_list]
        fig.add_trace(go.Bar(
            name=metric.replace('_', ' ').title(),
            x=[m.replace('_', ' ').title() for m in models_list],
            y=values,
            marker_color=color
        ))

    fig.update_layout(
        barmode='group',
        title='So Sánh Hiệu Suất Models',
        xaxis_title='Model',
        yaxis_title='Điểm Số',
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width='stretch')

    # Confusion matrices
    st.markdown("### 🔢 Ma Trận Nhầm Lẫn (Confusion Matrix)")

    for model_name, metrics in results.items():
        cm = metrics['confusion_matrix']

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Không Lỗi (0)', 'Có Lỗi (1)'],
            y=['Không Lỗi (0)', 'Có Lỗi (1)'],
            colorscale='Blues',
            text=cm,
            texttemplate='%d',
            textfont={"size": 20},
            showscale=False
        ))

        fig.update_layout(
            title=f'Ma Trận Nhầm Lẫn - {model_name.replace("_", " ").title()}',
            xaxis_title='Dự Đoán',
            yaxis_title='Thực Tế',
            width=500,
            height=400
        )

        st.plotly_chart(fig, use_container_width='stretch')

    # Feature importance
    st.markdown("### 📊 Độ Quan Trọng Của Features (Random Forest)")

    if st.session_state.models is not None and 'random_forest' in st.session_state.models.models:
        models_container = st.session_state.models
        feature_names = st.session_state.get('feature_names', [])

        if feature_names:
            importance = models_container.get_feature_importance(feature_names)

            if 'random_forest' in importance:
                imp_df = importance['random_forest']

                # Top features
                fig = px.bar(
                    imp_df.head(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Features Quan Trọng',
                    color='importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width='stretch')

    # ROC Curve
    st.markdown("### 📈 Đường Cong ROC")

    if st.session_state.y_test is not None and st.session_state.probabilities is not None:
        y_test = st.session_state.y_test

        fig = go.Figure()

        # Random line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Ngẫu nhiên',
            line=dict(dash='dash', color='gray')
        ))

        for model_name, probas in st.session_state.probabilities.items():
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(y_test, probas)
            auc_score = results[model_name]['roc_auc']

            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name.replace("_", " ").title()} (AUC={auc_score:.3f})'
            ))

        fig.update_layout(
            title='Đường Cong ROC',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=700,
            height=500
        )

        st.plotly_chart(fig, use_container_width='stretch')

    # Precision-Recall Curve
    st.markdown("### 📈 Đường Cong Precision-Recall")

    if st.session_state.y_test is not None and st.session_state.probabilities is not None:
        y_test = st.session_state.y_test

        fig = go.Figure()

        for model_name, probas in st.session_state.probabilities.items():
            from sklearn.metrics import precision_recall_curve, average_precision_score

            precision, recall, _ = precision_recall_curve(y_test, probas)
            ap_score = average_precision_score(y_test, probas)

            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode='lines',
                name=f'{model_name.replace("_", " ").title()} (AP={ap_score:.3f})'
            ))

        fig.update_layout(
            title='Đường Cong Precision-Recall',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=700,
            height=500
        )

        st.plotly_chart(fig, use_container_width='stretch')


def show_export_page():
    """Trang xuất báo cáo"""
    st.markdown('<p class="sub-header">💾 Xuất Báo Cáo</p>', unsafe_allow_html=True)

    if st.session_state.train_results is None:
        st.warning("⚠️ Vui lòng huấn luyện model trước!")
        return

    results = st.session_state.train_results

    # Tạo báo cáo
    st.markdown("### 📋 Tạo Báo Cáo")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Xuất Kết Quả Đánh Giá")

        # Tạo dataframe so sánh
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })

        comparison_df = pd.DataFrame(comparison_data)

        # CSV
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Tải CSV",
            data=csv,
            file_name="model_evaluation_results.csv",
            mime="text/csv"
        )

        # Excel (simulated with CSV)
        st.info("Tải file CSV để mở trong Excel")

    with col2:
        st.markdown("#### 🔮 Xuất Kết Quả Dự Đoán")

        if st.session_state.X_test is not None and st.session_state.probabilities is not None:
            n_samples = len(st.session_state.X_test)

            pred_df = pd.DataFrame({
                'Module': [f'Module_{i+1}' for i in range(n_samples)],
                'LR_Probability': st.session_state.probabilities['logistic_regression'],
                'RF_Probability': st.session_state.probabilities['random_forest'],
                'NN_Probability': st.session_state.probabilities['neural_network'],
                'Ensemble_Probability': (
                    st.session_state.probabilities['logistic_regression'] +
                    st.session_state.probabilities['random_forest'] +
                    st.session_state.probabilities['neural_network']
                ) / 3
            })

            if st.session_state.y_test is not None:
                pred_df['Actual'] = st.session_state.y_test

            csv_pred = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Tải Dự Đoán",
                data=csv_pred,
                file_name="prediction_results.csv",
                mime="text/csv"
            )

    # Tạo text report
    st.markdown("---")
    st.markdown("### 📄 Tạo Báo Cáo Text")

    if st.button("📋 Tạo Báo Cáo Chi Tiết", use_container_width='stretch'):
        report = []
        report.append("="*60)
        report.append("BÁO CÁO ĐÁNH GIÁ DỰ ĐOÁN LỖI PHẦN MỀM")
        report.append("="*60)
        report.append(f"\nNgày: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        report.append("\n" + "-"*60)
        report.append("1. THÔNG TIN DỮ LIỆU")
        report.append("-"*60)

        if st.session_state.df is not None:
            report.append(f"- Tổng số mẫu: {len(st.session_state.df)}")
            report.append(f"- Số features: {len(st.session_state.feature_names)}")
            report.append(f"- Số mẫu test: {len(st.session_state.X_test)}")

        report.append("\n" + "-"*60)
        report.append("2. KẾT QUẢ ĐÁNH GIÁ MODEL")
        report.append("-"*60)

        for model_name, metrics in results.items():
            report.append(f"\n>>> {model_name.replace('_', ' ').title()}")
            report.append(f"- Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"- Precision: {metrics['precision']:.4f}")
            report.append(f"- Recall: {metrics['recall']:.4f}")
            report.append(f"- F1-Score: {metrics['f1_score']:.4f}")
            report.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}")

        report.append("\n" + "="*60)
        report.append("KẾT THÚC BÁO CÁO")
        report.append("="*60)

        report_text = "\n".join(report)

        st.download_button(
            label="📥 Tải Báo Cáo Text",
            data=report_text,
            file_name="evaluation_report.txt",
            mime="text/plain"
        )

        st.text_area("Xem trước báo cáo:", report_text, height=300)


def show_dashboard_page():
    """Trang Dashboard"""
    from dashboard.Dashboard import main as dashboard_main
    dashboard_main()


def show_models_page():
    """Trang quản lý Models"""
    st.markdown('<p class="sub-header">📁 Quản Lý Models</p>', unsafe_allow_html=True)

    model_manager = st.session_state.model_manager

    # Save current model
    st.markdown("### 💾 Lưu Model Hiện Tại")

    if st.session_state.get('models') is not None:
        model_name = st.text_input("Tên model:", value="defect_prediction")

        if st.button("💾 Lưu Model", type="primary"):
            model_path = model_manager.save_model(
                st.session_state.models,
                model_name
            )
            st.success(f"✅ Model đã lưu tại: {model_path}")
    else:
        st.warning("⚠️ Chưa có model để lưu!")

    st.markdown("---")

    # List saved models
    st.markdown("### 📂 Models Đã Lưu")

    saved_models = model_manager.list_models()

    if saved_models:
        for model in saved_models:
            with st.expander(f"📦 {model['name']}"):
                st.write(f"**Thời gian:** {model['timestamp']}")
                st.write(f"**Đường dẫn:** {model['path']}")
    else:
        st.info("Chưa có model nào được lưu!")


def show_history_page():
    """Trang xem lịch sử"""
    st.markdown('<p class="sub-header">📜 Lịch Sử Phân Tích</p>', unsafe_allow_html=True)

    # Lấy danh sách sessions
    try:
        sessions = st.session_state.db.get_all_sessions()

        if not sessions:
            st.info("Chưa có lịch sử phân tích nào!")
            return

        # Hiển thị danh sách dạng cards
        st.markdown("### 📋 Danh Sách Phiên Phân Tích")

        # Cards layout
        for session in sessions:
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 1])

                with col1:
                    st.markdown(f"**🕐 {session['timestamp']}**")

                with col2:
                    source_icon = "📄" if session['source_type'] == 'csv' else "📁"
                    st.markdown(f"{source_icon} **{session['source_type']}** - {session['files_count']} files")

                with col3:
                    if st.button("Xem chi tiết", key=f"view_{session['session_id']}"):
                        st.session_state.selected_session = session['session_id']
                        st.rerun()

                st.markdown("---")

        # Nếu có session được chọn
        if 'selected_session' in st.session_state and st.session_state.selected_session:
            session_id = st.session_state.selected_session
            details = st.session_state.db.get_session_details(session_id)

            st.markdown(f"### 📊 Chi Tiết Phiên: {session_id}")

            # Session info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Số files", details['session'].get('files_count', 0))
            with col2:
                st.metric("Loại nguồn", details['session'].get('source_type', 'N/A'))
            with col3:
                st.metric("Thời gian", details['session'].get('timestamp', 'N/A')[:19])

            # Metrics
            if details['metrics']:
                st.markdown("#### 📈 Metrics Của Các File")

                metrics_df = pd.DataFrame(details['metrics'])
                st.dataframe(metrics_df, use_container_width='stretch')

                # Charts
                if len(metrics_df) > 0:
                    # LOC chart
                    fig = px.bar(
                        metrics_df,
                        x='file_name',
                        y='LOC',
                        title="Số Dòng Code (LOC) Theo File",
                        color='LOC',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width='stretch')

                    # Complexity chart
                    fig2 = px.bar(
                        metrics_df,
                        x='file_name',
                        y='CYCLOMATIC_COMPLEXITY',
                        title="Độ Phức Tạp Theo File",
                        color='CYCLOMATIC_COMPLEXITY',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig2, use_container_width='stretch')

    except Exception as e:
        st.error(f"Lỗi khi tải lịch sử: {str(e)}")


def show_about_page():
    """Trang giới thiệu"""
    st.markdown('<p class="sub-header">ℹ️ Giới Thiệu</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🛡️ Công Cụ AI Dự Đoán Lỗi Phần Mềm

    **Dự án:** P7 - AI Tool for Software Defect Prediction

    **Môn học:** Software Measurement & Analysis

    ---

    ### 🎯 Mục Tiêu

    Dự đoán các module phần mềm có khả năng chứa lỗi sử dụng Machine Learning.

    ---

    ### 📊 Các Metrics Đầu Vào

    | Metric | Mô tả |
    |--------|-------|
    | LOC | Số dòng code |
    | Cyclomatic Complexity | Độ phức tạp vòng lặp |
    | Coupling | Mức độ liên kết |
    | Code Churn | Số dòng thay đổi |
    | Decision Count | Số điểm quyết định |
    | Essential Complexity | Độ phức tạp cốt lõi |

    ---

    ### 🤖 Các Model Machine Learning

    1. **Logistic Regression**
       - Model tuyến tính cơ bản
       - Dễ diễn giải

    2. **Random Forest**
       - Phương pháp ensemble
       - Xử lý tốt các mối quan hệ phi tuyến tính

    3. **Neural Network (MLP)**
       - Mạng nơ-ron nhiều lớp
       - Học các pattern phức tạp

    ---

    ### 🛠️ Công Nghệ Sử Dụng

    - **Python** - Ngôn ngữ lập trình
    - **Streamlit** - Framework web
    - **Scikit-learn** - Thư viện ML
    - **Plotly** - Trực quan hóa

    ---

    ### 📚 Tham Khảo

    - NASA Metrics Data (KC1/KC2)
    - Tài liệu về dự đoán lỗi phần mềm

    ---

    *Developed for Software Measurement & Analysis Course Project*
    """)


def main():
    """Hàm chính"""
    initialize_session_state()
    menu = show_sidebar()

    if menu == "🏠 Trang Chủ":
        show_home_page()
    elif menu == "📊 Dashboard":
        show_dashboard_page()
    elif menu == "📂 Tải Dữ Liệu":
        show_data_upload_page()
    elif menu == "🤖 Huấn Luyện Model":
        show_model_training_page()
    elif menu == "🔮 Dự Đoán":
        show_predictions_page()
    elif menu == "📊 Đánh Giá":
        show_evaluation_page()
    elif menu == "💾 Xuất Báo Cáo":
        show_export_page()
    elif menu == "📜 Lịch Sử":
        show_history_page()
    elif menu == "📁 Models":
        show_models_page()
    elif menu == "ℹ️ Giới Thiệu":
        show_about_page()


if __name__ == "__main__":
    main()
