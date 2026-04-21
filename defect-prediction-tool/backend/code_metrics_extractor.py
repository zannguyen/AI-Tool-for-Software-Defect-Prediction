"""
Code Metrics Extractor - Trích xuất metrics từ source code
Hỗ trợ: Python, Java, JavaScript, C/C++, C#
"""

import os
import re
import pandas as pd
from typing import Dict, List, Tuple
from pathlib import Path


class CodeMetricsExtractor:
    """Trích xuất metrics từ source code files"""

    def __init__(self):
        self.metrics = {}

    def extract_from_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Trích xuất metrics từ tất cả file trong thư mục

        Args:
            directory_path: Đường dẫn thư mục chứa source code

        Returns:
            DataFrame chứa metrics của từng file
        """
        results = []

        # Các đuôi file được hỗ trợ
        extensions = {
            '.py': self.extract_python_metrics,
            '.java': self.extract_java_metrics,
            '.js': self.extract_js_metrics,
            '.cpp': self.extract_cpp_metrics,
            '.c': self.extract_c_metrics,
            '.cs': self.extract_csharp_metrics,
            '.h': self.extract_cpp_metrics,
            '.hpp': self.extract_cpp_metrics,
        }

        # Duyệt qua tất cả file
        for root, dirs, files in os.walk(directory_path):
            # Bỏ qua thư mục ẩn và mọi thư mục thư viện / build output.
            # Danh sách này đồng bộ với _SKIP_DIRS trong api.py.
            _SKIP = {
                # Python
                'venv', '.venv', 'env', '.env', '__pycache__', '.eggs',
                'site-packages', 'dist-packages', 'pip', 'setuptools',
                'pkg_resources', 'distutils', '_distutils_hack',
                # JavaScript / Node
                'node_modules', '.npm', '.yarn',
                # Java / Maven / Gradle / Android
                'target', '.gradle', '.m2',
                # Ruby / Go / Rust / iOS
                'Pods', 'vendor', 'bundle', 'cargo', '.cargo',
                # Build / dist output
                'build', 'dist', 'bin', 'obj', 'out', '.next', '.nuxt',
                # IDE / VCS
                '.git', '.svn', '.hg', '.idea', '.vscode', '.vs',
                '__MACOSX', 'bower_components',
            }
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.')
                and d not in _SKIP
                and 'dist-info' not in d
                and 'egg-info' not in d
            ]

            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1]  # Lấy phần mở rộng

                if ext in extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        metrics = extensions[ext](content, file)
                        metrics['file_path'] = file_path
                        metrics['file_name'] = file
                        results.append(metrics)
                    except Exception as e:
                        print(f"Lỗi đọc file {file_path}: {e}")

        return pd.DataFrame(results)

    def extract_from_files(self, file_paths: List[str]) -> pd.DataFrame:
        """
        Trích xuất metrics từ danh sách file

        Args:
            file_paths: Danh sách đường dẫn file

        Returns:
            DataFrame chứa metrics
        """
        results = []

        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                ext = os.path.splitext(file_path)[1]
                file_name = os.path.basename(file_path)

                if ext == '.py':
                    metrics = self.extract_python_metrics(content, file_name)
                elif ext == '.java':
                    metrics = self.extract_java_metrics(content, file_name)
                elif ext in ['.js', '.jsx']:
                    metrics = self.extract_js_metrics(content, file_name)
                elif ext in ['.c', '.h']:
                    metrics = self.extract_c_metrics(content, file_name)
                elif ext in ['.cpp', '.hpp']:
                    metrics = self.extract_cpp_metrics(content, file_name)
                elif ext == '.cs':
                    metrics = self.extract_csharp_metrics(content, file_name)
                else:
                    continue

                metrics['file_path'] = file_path
                metrics['file_name'] = file_name
                results.append(metrics)
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")

        return pd.DataFrame(results)

    def extract_python_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ Python code"""
        lines = content.split('\n')

        # Basic counts
        loc = len([l for l in lines if l.strip()])
        blank_lines = len([l for l in lines if not l.strip()])
        total_lines = len(lines)

        # Comments
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        docstring_count = len(re.findall(r'""".*?"""', content, re.DOTALL)) + len(re.findall(r"'''.*?'''", content, re.DOTALL))

        # Functions and classes
        function_count = len(re.findall(r'^def\s+\w+', content, re.MULTILINE))
        class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))

        # Calculate cyclomatic complexity (simplified)
        # Count: if, elif, else, for, while, except, and, or, case, ?
        decision_points = (
            len(re.findall(r'\bif\b', content)) +
            len(re.findall(r'\belif\b', content)) +
            len(re.findall(r'\bfor\b', content)) +
            len(re.findall(r'\bwhile\b', content)) +
            len(re.findall(r'\bexcept\b', content)) +
            len(re.findall(r'\band\b', content)) +
            len(re.findall(r'\bor\b', content))
        )
        cyclomatic_complexity = decision_points + 1

        # Import statements
        import_count = len(re.findall(r'^import\s+|^from\s+\w+\s+import', content, re.MULTILINE))

        # Return statements
        return_count = len(re.findall(r'\breturn\b', content))

        # Parameters
        param_count = len(re.findall(r'def\s+\w+\(([^)]*)\)', content))
        total_params = sum(len(p.split(',')) for p in re.findall(r'def\s+\w+\(([^)]*)\)', content))

        return {
            'file_name': filename,
            'LOC': loc,
            'LOC_BLANK': blank_lines,
            'LOC_TOTAL': total_lines,
            'LOC_COMMENTS': comment_lines + docstring_count,
            'LOC_CODE': loc - comment_lines - docstring_count,
            'FUNCTION_COUNT': function_count,
            'CLASS_COUNT': class_count,
            'CYCLOMATIC_COMPLEXITY': cyclomatic_complexity,
            'DECISION_COUNT': decision_points,
            'IMPORT_COUNT': import_count,
            'RETURN_COUNT': return_count,
            'PARAMETER_COUNT': total_params,
            'COMMENT_RATIO': (comment_lines + docstring_count) / total_lines if total_lines > 0 else 0,
        }

    def extract_java_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ Java code"""
        lines = content.split('\n')

        loc = len([l for l in lines if l.strip()])
        blank_lines = len([l for l in lines if not l.strip()])
        total_lines = len(lines)

        # Comments
        single_line_comments = len([l for l in lines if l.strip().startswith('//')])
        multi_line_comments = len(re.findall(r'/\*.*?\*/', content, re.DOTALL))
        comment_lines = single_line_comments + multi_line_comments

        # Methods and classes
        method_count = len(re.findall(r'(public|private|protected)\s+(static\s+)?(\w+)\s+\w+\s*\([^)]*\)', content))
        class_count = len(re.findall(r'^class\s+\w+', content, re.MULTILINE))

        # Cyclomatic complexity
        decision_points = (
            len(re.findall(r'\bif\b', content)) +
            len(re.findall(r'\belse\s+if\b', content)) +
            len(re.findall(r'\bfor\b', content)) +
            len(re.findall(r'\bwhile\b', content)) +
            len(re.findall(r'\bcase\b', content)) +
            len(re.findall(r'\bcatch\b', content)) +
            len(re.findall(r'&&', content)) +
            len(re.findall(r'\|\|', content))
        )
        cyclomatic_complexity = decision_points + 1

        # Fields
        field_count = len(re.findall(r'(public|private|protected)\s+\w+\s+\w+\s*;', content))

        # Imports
        import_count = len(re.findall(r'^import\s+', content, re.MULTILINE))

        return {
            'file_name': filename,
            'LOC': loc,
            'LOC_BLANK': blank_lines,
            'LOC_TOTAL': total_lines,
            'LOC_COMMENTS': comment_lines,
            'LOC_CODE': loc - comment_lines,
            'FUNCTION_COUNT': method_count,
            'CLASS_COUNT': class_count,
            'CYCLOMATIC_COMPLEXITY': cyclomatic_complexity,
            'DECISION_COUNT': decision_points,
            'FIELD_COUNT': field_count,
            'IMPORT_COUNT': import_count,
            'COMMENT_RATIO': comment_lines / total_lines if total_lines > 0 else 0,
        }

    def extract_js_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ JavaScript code"""
        lines = content.split('\n')

        loc = len([l for l in lines if l.strip()])
        blank_lines = len([l for l in lines if not l.strip()])
        total_lines = len(lines)

        # Comments
        comment_lines = len([l for l in lines if l.strip().startswith('//')])
        comment_lines += len(re.findall(r'/\*.*?\*/', content, re.DOTALL))

        # Functions
        function_count = len(re.findall(r'function\s+\w+', content))
        arrow_func_count = len(re.findall(r'=>', content))
        function_count += arrow_func_count

        # Classes
        class_count = len(re.findall(r'class\s+\w+', content))

        # Cyclomatic complexity
        decision_points = (
            len(re.findall(r'\bif\b', content)) +
            len(re.findall(r'\belse\s+if\b', content)) +
            len(re.findall(r'\bfor\b', content)) +
            len(re.findall(r'\bwhile\b', content)) +
            len(re.findall(r'\bswitch\b', content)) +
            len(re.findall(r'\bcatch\b', content)) +
            len(re.findall(r'&&', content)) +
            len(re.findall(r'\|\|', content)) +
            len(re.findall(r'\?', content))
        )
        cyclomatic_complexity = decision_points + 1

        return {
            'file_name': filename,
            'LOC': loc,
            'LOC_BLANK': blank_lines,
            'LOC_TOTAL': total_lines,
            'LOC_COMMENTS': comment_lines,
            'LOC_CODE': loc - comment_lines,
            'FUNCTION_COUNT': function_count,
            'CLASS_COUNT': class_count,
            'CYCLOMATIC_COMPLEXITY': cyclomatic_complexity,
            'DECISION_COUNT': decision_points,
            'COMMENT_RATIO': comment_lines / total_lines if total_lines > 0 else 0,
        }

    def extract_c_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ C code"""
        lines = content.split('\n')

        loc = len([l for l in lines if l.strip()])
        blank_lines = len([l for l in lines if not l.strip()])
        total_lines = len(lines)

        # Comments
        comment_lines = len([l for l in lines if l.strip().startswith('//')])
        comment_lines += len(re.findall(r'/\*.*?\*/', content, re.DOTALL))

        # Functions
        function_count = len(re.findall(r'\w+\s+\w+\s*\([^)]*\)\s*\{', content))

        # Cyclomatic complexity
        decision_points = (
            len(re.findall(r'\bif\b', content)) +
            len(re.findall(r'\belse\s+if\b', content)) +
            len(re.findall(r'\bfor\b', content)) +
            len(re.findall(r'\bwhile\b', content)) +
            len(re.findall(r'\bswitch\b', content)) +
            len(re.findall(r'\bcase\b', content)) +
            len(re.findall(r'&&', content)) +
            len(re.findall(r'\|\|', content))
        )
        cyclomatic_complexity = decision_points + 1

        # Includes
        include_count = len(re.findall(r'^#include', content, re.MULTILINE))

        # Defines
        define_count = len(re.findall(r'^#define', content, re.MULTILINE))

        return {
            'file_name': filename,
            'LOC': loc,
            'LOC_BLANK': blank_lines,
            'LOC_TOTAL': total_lines,
            'LOC_COMMENTS': comment_lines,
            'LOC_CODE': loc - comment_lines,
            'FUNCTION_COUNT': function_count,
            'CLASS_COUNT': 0,
            'CYCLOMATIC_COMPLEXITY': cyclomatic_complexity,
            'DECISION_COUNT': decision_points,
            'IMPORT_COUNT': include_count + define_count,
            'COMMENT_RATIO': comment_lines / total_lines if total_lines > 0 else 0,
        }

    def extract_cpp_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ C++ code (tương tự C)"""
        return self.extract_c_metrics(content, filename)

    def extract_csharp_metrics(self, content: str, filename: str) -> Dict:
        """Trích xuất metrics từ C# code (tương tự Java)"""
        return self.extract_java_metrics(content, filename)

    def prepare_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu metrics cho việc dự đoán

        Args:
            df: DataFrame chứa metrics

        Returns:
            DataFrame đã chuẩn hóa
        """
        # Các cột cần thiết cho model
        required_columns = [
            'LOC', 'LOC_BLANK', 'LOC_TOTAL', 'LOC_COMMENTS',
            'CYCLOMATIC_COMPLEXITY', 'DECISION_COUNT',
            'FUNCTION_COUNT', 'CLASS_COUNT', 'COMMENT_RATIO'
        ]

        # Tạo DataFrame mới với các cột cần thiết
        result = pd.DataFrame()

        for col in required_columns:
            if col in df.columns:
                result[col] = df[col]
            else:
                result[col] = 0

        return result


def extract_code_metrics_from_folder(folder_path: str) -> pd.DataFrame:
    """
    Hàm tiện ích để trích xuất metrics từ thư mục

    Args:
        folder_path: Đường dẫn thư mục chứa source code

    Returns:
        DataFrame chứa metrics
    """
    extractor = CodeMetricsExtractor()
    df = extractor.extract_from_directory(folder_path)

    if not df.empty:
        # Thêm cột giả định cho label (0 = không lỗi)
        # Trong thực tế, cần đánh dấu thủ công hoặc lấy từ lịch sử
        df['LABEL'] = 0

    return df


if __name__ == "__main__":
    # Test
    print("Testing Code Metrics Extractor...")

    # Tạo file test
    test_code = '''
def calculate_factorial(n):
    """Calculate factorial of n"""
    if n <= 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        if a > b:
            return a - b
        else:
            return b - a
'''

    extractor = CodeMetricsExtractor()
    metrics = extractor.extract_python_metrics(test_code, "test.py")

    print("\nMetrics extracted:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
