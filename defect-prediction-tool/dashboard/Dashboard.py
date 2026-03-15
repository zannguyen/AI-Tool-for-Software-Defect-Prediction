"""
Dashboard Page
Trang dashboard tong quan
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

def main():
    st.markdown("""
    <style>
        .dashboard-header {
            font-size: 2rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            padding: 20px;
        }
        .metric-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="dashboard-header">📊 Dashboard Tong Quan</p>', unsafe_allow_html=True)

    # Overview metrics
    st.markdown("### 📈 Tong Quan")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.get('data_loaded'):
            df = st.session_state.get('df')
            st.metric("Tong Mau", len(df) if df is not None else 0)
        else:
            st.metric("Tong Mau", "Chua tai")

    with col2:
        if st.session_state.get('data_loaded'):
            X = st.session_state.get('X')
            st.metric("So Features", len(X.columns) if X is not None else 0)
        else:
            st.metric("So Features", 0)

    with col3:
        if st.session_state.get('predictions_made'):
            st.metric("Trang Thai", "Da Train")
        else:
            st.metric("Trang Thai", "Chua Train")

    with col4:
        if st.session_state.get('train_results'):
            best_model = max(st.session_state.train_results.items(),
                          key=lambda x: x[1]['roc_auc'])
            st.metric("Model Tot Nhat", best_model[0].replace('_', ' ').title()[:15])
        else:
            st.metric("Model Tot Nhat", "N/A")

    st.markdown("---")

    # Data distribution
    if st.session_state.get('data_loaded'):
        df = st.session_state.get('df')
        if df is not None and 'LABEL' in df.columns:
            st.markdown("### 📊 Phan Bo Du Lieu")

            col1, col2 = st.columns(2)

            with col1:
                # Label distribution
                label_counts = df['LABEL'].value_counts()
                fig = px.pie(
                    values=label_counts.values,
                    names=['Khong Loi', 'Co Loi'],
                    title='Phan Bo Nhan (Label)',
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig, use_container_width='stretch')

            with col2:
                # Feature distribution
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    selected_col = st.selectbox("Chon Feature", numeric_cols[:-1])
                    fig = px.histogram(
                        df, x=selected_col,
                        title=f'Phan Bo {selected_col}',
                        color='LABEL',
                        color_discrete_sequence=['#3498db', '#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width='stretch')

    # Model performance
    if st.session_state.get('train_results'):
        st.markdown("### 🤖 Hieu Suat Model")

        results = st.session_state.train_results

        # Comparison chart
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            })

        df_comp = pd.DataFrame(comparison_data)

        # Radar chart
        fig = go.Figure()

        for i, row in df_comp.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'],
                   row['F1-Score'], row['ROC-AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                fill='toself',
                name=row['Model']
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='So Sanh Hieu Suat Model (Radar Chart)'
        )

        st.plotly_chart(fig, use_container_width='stretch')

    # Recent activity
    st.markdown("### 🕐 Hoat Dong Gan Nhat")

    if 'db' in st.session_state:
        try:
            sessions = st.session_state.db.get_all_sessions()
            if sessions:
                recent_sessions = sessions[:5]
                for session in recent_sessions:
                    st.write(f"🕐 {session['timestamp'][:19]} - {session['source_type']} - {session['files_count']} files")
            else:
                st.info("Chua co hoat dong nao")
        except:
            st.info("Chua co hoat dong nao")

if __name__ == "__main__":
    main()
