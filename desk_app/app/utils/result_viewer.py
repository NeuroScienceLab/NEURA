# -*- coding: utf-8 -*-
"""
Result Viewer - 结果查看器组件

支持查看：
- 代码文件 (Python)
- 图像文件 (PNG/JPG)
- JSON数据
"""

import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QScrollArea, QStackedWidget, QSizePolicy, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont, QColor, QTextCharFormat, QSyntaxHighlighter


class PythonHighlighter(QSyntaxHighlighter):
    """简单的Python语法高亮器"""

    def __init__(self, document):
        super().__init__(document)
        self.highlighting_rules = []

        # 关键字
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#FF7B72"))  # 红色
        keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'False', 'finally', 'for',
            'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'None',
            'not', 'or', 'pass', 'raise', 'return', 'True', 'try', 'while',
            'with', 'yield'
        ]
        for word in keywords:
            pattern = f'\\b{word}\\b'
            self.highlighting_rules.append((pattern, keyword_format))

        # 字符串
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#A5D6FF"))  # 蓝色
        self.highlighting_rules.append((r'"[^"\\]*(\\.[^"\\]*)*"', string_format))
        self.highlighting_rules.append((r"'[^'\\]*(\\.[^'\\]*)*'", string_format))

        # 注释
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#8B949E"))  # 灰色
        self.highlighting_rules.append((r'#[^\n]*', comment_format))

        # 数字
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#79C0FF"))  # 浅蓝色
        self.highlighting_rules.append((r'\b\d+\.?\d*\b', number_format))

        # 函数定义
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#D2A8FF"))  # 紫色
        self.highlighting_rules.append((r'\bdef\s+(\w+)', function_format))
        self.highlighting_rules.append((r'\bclass\s+(\w+)', function_format))

    def highlightBlock(self, text):
        import re
        for pattern, format in self.highlighting_rules:
            for match in re.finditer(pattern, text):
                start = match.start()
                length = match.end() - match.start()
                self.setFormat(start, length, format)


class ResultViewer(QWidget):
    """结果查看器 - 支持代码/图像/JSON"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_file = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 文件信息标签
        self.file_label = QLabel("")
        self.file_label.setStyleSheet("""
            QLabel {
                background-color: #2D2D2D;
                color: #CCCCCC;
                padding: 8px 12px;
                font-size: 11px;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        layout.addWidget(self.file_label)

        # 内容区
        self.content_stack = QStackedWidget()

        # 代码查看器
        self.code_viewer = QTextEdit()
        self.code_viewer.setReadOnly(True)
        self.code_viewer.setFont(QFont("Consolas", 10))
        self.code_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
                padding: 10px;
            }
        """)
        self.highlighter = PythonHighlighter(self.code_viewer.document())
        self.content_stack.addWidget(self.code_viewer)

        # 图像查看器（带滚动）
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1E1E1E;
                border: none;
            }
        """)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1E1E1E;")
        self.image_scroll.setWidget(self.image_label)
        self.content_stack.addWidget(self.image_scroll)

        # JSON查看器
        self.json_viewer = QTextEdit()
        self.json_viewer.setReadOnly(True)
        self.json_viewer.setFont(QFont("Consolas", 10))
        self.json_viewer.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: none;
                padding: 10px;
            }
        """)
        self.content_stack.addWidget(self.json_viewer)

        # 空状态
        self.empty_label = QLabel("选择一个文件查看内容")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #666666;
                font-size: 14px;
            }
        """)
        self.content_stack.addWidget(self.empty_label)
        self.content_stack.setCurrentWidget(self.empty_label)

        layout.addWidget(self.content_stack, 1)

    def show_code(self, file_path: str):
        """显示代码文件"""
        path = Path(file_path)
        if not path.exists():
            self._show_error(f"文件不存在: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            self.code_viewer.setPlainText(code)
            self.file_label.setText(f"Python代码: {path.name}")
            self.content_stack.setCurrentWidget(self.code_viewer)
            self._current_file = file_path
        except Exception as e:
            self._show_error(f"读取失败: {e}")

    def show_image(self, file_path: str):
        """显示图像文件"""
        path = Path(file_path)
        if not path.exists():
            self._show_error(f"图像不存在: {file_path}")
            return

        try:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                self._show_error(f"无法加载图像: {file_path}")
                return

            # 自适应缩放
            max_width = 800
            max_height = 600
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(
                    max_width, max_height,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )

            self.image_label.setPixmap(pixmap)
            self.file_label.setText(f"图像: {path.name} ({pixmap.width()}x{pixmap.height()})")
            self.content_stack.setCurrentWidget(self.image_scroll)
            self._current_file = file_path
        except Exception as e:
            self._show_error(f"加载图像失败: {e}")

    def show_json(self, file_path: str):
        """显示JSON文件"""
        path = Path(file_path)
        if not path.exists():
            self._show_error(f"文件不存在: {file_path}")
            return

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            formatted = json.dumps(data, indent=2, ensure_ascii=False)
            self.json_viewer.setPlainText(formatted)
            self.file_label.setText(f"JSON数据: {path.name}")
            self.content_stack.setCurrentWidget(self.json_viewer)
            self._current_file = file_path
        except Exception as e:
            self._show_error(f"读取JSON失败: {e}")

    def show_file(self, file_path: str):
        """根据文件类型自动选择查看方式"""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.py':
            self.show_code(file_path)
        elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
            self.show_image(file_path)
        elif suffix == '.json':
            self.show_json(file_path)
        else:
            # 尝试作为文本显示
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.code_viewer.setPlainText(content)
                self.file_label.setText(f"文件: {path.name}")
                self.content_stack.setCurrentWidget(self.code_viewer)
                self._current_file = file_path
            except Exception as e:
                self._show_error(f"无法读取文件: {e}")

    def _show_error(self, message: str):
        """显示错误信息"""
        self.empty_label.setText(message)
        self.empty_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #FF6B6B;
                font-size: 12px;
            }
        """)
        self.content_stack.setCurrentWidget(self.empty_label)

    def clear(self):
        """清空显示"""
        self.file_label.setText("")
        self.empty_label.setText("选择一个文件查看内容")
        self.empty_label.setStyleSheet("""
            QLabel {
                background-color: #1E1E1E;
                color: #666666;
                font-size: 14px;
            }
        """)
        self.content_stack.setCurrentWidget(self.empty_label)
        self._current_file = None

    def set_theme(self, theme: str):
        """设置主题"""
        if theme == 'light':
            bg_color = "#F5F5F5"
            text_color = "#333333"
            border_color = "#CCCCCC"
        else:
            bg_color = "#1E1E1E"
            text_color = "#D4D4D4"
            border_color = "#3C3C3C"

        self.code_viewer.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 10px;
            }}
        """)
        self.json_viewer.setStyleSheet(f"""
            QTextEdit {{
                background-color: {bg_color};
                color: {text_color};
                border: none;
                padding: 10px;
            }}
        """)
        self.image_scroll.setStyleSheet(f"""
            QScrollArea {{
                background-color: {bg_color};
                border: none;
            }}
        """)
        self.image_label.setStyleSheet(f"background-color: {bg_color};")
        self.empty_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: #666666;
                font-size: 14px;
            }}
        """)
        self.file_label.setStyleSheet(f"""
            QLabel {{
                background-color: {'#EEEEEE' if theme == 'light' else '#2D2D2D'};
                color: {text_color};
                padding: 8px 12px;
                font-size: 11px;
                border-bottom: 1px solid {border_color};
            }}
        """)
