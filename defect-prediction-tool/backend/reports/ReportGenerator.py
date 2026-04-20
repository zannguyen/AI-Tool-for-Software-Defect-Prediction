"""
Reports Module
Chuc nang tao va xuat bao cao
"""

import os
import pandas as pd
from datetime import datetime

class ReportGenerator:
    """Tao va xuat bao cao"""

    def __init__(self, reports_dir='reports'):
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)

    def generate_evaluation_report(self, results, feature_names=None):
        """Tao bao cao danh gia model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.reports_dir, f'evaluation_report_{timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BAO CAO DANH GIA DU DOAN LOI PHAN MEM\n")
            f.write("=" * 60 + "\n")
            f.write(f"Ngay: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "-" * 60 + "\n")

            for model_name, metrics in results.items():
                f.write(f"\n>>> {model_name.upper()}\n")
                f.write(f"- Accuracy:   {metrics['accuracy']:.4f}\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall:    {metrics['recall']:.4f}\n")
                f.write(f"- F1-Score:  {metrics['f1_score']:.4f}\n")
                f.write(f"- ROC-AUC:   {metrics['roc_auc']:.4f}\n")

                # Confusion matrix
                cm = metrics.get('confusion_matrix', [])
                if cm:
                    f.write(f"\nConfusion Matrix:\n")
                    f.write(f"  True Neg:  {cm[0][0]}\n")
                    f.write(f"  False Pos: {cm[0][1]}\n")
                    f.write(f"  False Neg: {cm[1][0]}\n")
                    f.write(f"  True Pos:  {cm[1][1]}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("KET THUC BAO CAO\n")
            f.write("=" * 60 + "\n")

        return report_path

    def generate_prediction_report(self, predictions_df, output_format='csv'):
        """Tao bao cao du doan"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_format == 'csv':
            report_path = os.path.join(self.reports_dir, f'prediction_report_{timestamp}.csv')
            predictions_df.to_csv(report_path, index=False)
        else:
            report_path = os.path.join(self.reports_dir, f'prediction_report_{timestamp}.txt')
            predictions_df.to_csv(report_path, index=False, sep='\t')

        return report_path

    def generate_summary_report(self, df, results, feature_names=None):
        """Tao bao cao tong hop"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.reports_dir, f'summary_report_{timestamp}.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BAO CAO TONG HOP - DU DOAN LOI PHAN MEM\n")
            f.write("=" * 70 + "\n")
            f.write(f"Ngay: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data summary
            f.write("-" * 70 + "\n")
            f.write("1. THONG TIN DU LIEU\n")
            f.write("-" * 70 + "\n")
            f.write(f"- Tong so mau: {len(df)}\n")
            if 'LABEL' in df.columns:
                f.write(f"- So mau loi: {df['LABEL'].sum()}\n")
                f.write(f"- Ty le loi: {100*df['LABEL'].mean():.2f}%\n")
            f.write(f"- So features: {len(df.columns) - 1}\n\n")

            # Best model
            if results:
                f.write("-" * 70 + "\n")
                f.write("2. KET QUA DANH GIA MODEL\n")
                f.write("-" * 70 + "\n")

                best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
                f.write(f"\nModel tot nhat: {best_model[0]}\n")
                f.write(f"- ROC-AUC: {best_model[1]['roc_auc']:.4f}\n")
                f.write(f"- F1-Score: {best_model[1]['f1_score']:.4f}\n")
                f.write(f"- Accuracy: {best_model[1]['accuracy']:.4f}\n\n")

            f.write("=" * 70 + "\n")
            f.write("KET THUC BAO CAO\n")
            f.write("=" * 70 + "\n")

        return report_path

    def list_reports(self):
        """Lay danh sach bao cao"""
        reports = []
        for item in os.listdir(self.reports_dir):
            item_path = os.path.join(self.reports_dir, item)
            if os.path.isfile(item_path):
                reports.append({
                    'name': item,
                    'path': item_path,
                    'size': os.path.getsize(item_path),
                    'created': datetime.fromtimestamp(os.path.getctime(item_path))
                })
        return sorted(reports, key=lambda x: x['created'], reverse=True)
