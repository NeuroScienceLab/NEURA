import re
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget, QLabel, QScrollArea


class MermaidRenderer(QWidget):
    """
    工作流显示器 - 纯文本版本
    完全移除QWebEngineView，避免闪退问题
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        # 纯文本显示（滚动区域）
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #2D2D2D;
            }
        """)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.label.setStyleSheet("""
            QLabel {
                font-family: "JetBrains Mono", "Consolas", monospace;
                font-size: 11px;
                color: #CCCCCC;
                padding: 15px;
                background-color: #2D2D2D;
            }
        """)
        self.scroll.setWidget(self.label)
        self._layout.addWidget(self.scroll)

    def render_mermaid(self, mermaid_code):
        """
        渲染Mermaid代码为易读的文本格式

        Args:
            mermaid_code: Mermaid格式的图表代码
        Returns:
            True 总是返回True
        """
        formatted = self._format_workflow(mermaid_code)
        self.label.setText(formatted)
        return True

    def _format_workflow(self, mermaid_code):
        """将Mermaid代码格式化为易读的流程文本"""
        lines = mermaid_code.strip().split('\n')
        steps = []

        for line in lines:
            line = line.strip()
            # 跳过非节点定义行
            if not line or line.startswith('graph') or line.startswith('style'):
                continue

            # 解析箭头连接 A[text] --> B[text]
            if '-->' in line:
                parts = line.split('-->')
                if len(parts) >= 2:
                    # 提取第一个节点的文本
                    left = parts[0].strip()
                    match = re.search(r'\[(.+?)\]', left)
                    if match:
                        step_text = match.group(1)
                        if step_text not in steps:
                            steps.append(step_text)

                    # 提取最后一个节点的文本（用于最后一步）
                    right = parts[-1].strip()
                    match = re.search(r'\[(.+?)\]', right)
                    if match:
                        step_text = match.group(1)
                        if step_text not in steps:
                            steps.append(step_text)

        if steps:
            # 格式化为流程列表
            result = "工作流程:\n"
            result += "─" * 30 + "\n\n"
            for i, step in enumerate(steps):
                result += f"  {i+1:02d}. {step}\n"
                if i < len(steps) - 1:
                    result += "       ↓\n"
            return result
        else:
            # 无法解析时显示原始代码
            return f"工作流代码:\n─" + "─" * 29 + f"\n\n{mermaid_code}"

    def clear(self):
        """清空显示"""
        self.label.setText("")

    def set_theme(self, theme):
        """
        设置主题

        Args:
            theme: 主题名称（'light'或'dark'）
        """
        if theme == 'dark':
            self.label.setStyleSheet("""
                QLabel {
                    font-family: "JetBrains Mono", "Consolas", monospace;
                    font-size: 11px;
                    color: #CCCCCC;
                    padding: 15px;
                    background-color: #2D2D2D;
                }
            """)
            self.scroll.setStyleSheet("QScrollArea { border: none; background-color: #2D2D2D; }")
        else:
            self.label.setStyleSheet("""
                QLabel {
                    font-family: "JetBrains Mono", "Consolas", monospace;
                    font-size: 11px;
                    color: #333333;
                    padding: 15px;
                    background-color: #F5F5F5;
                }
            """)
            self.scroll.setStyleSheet("QScrollArea { border: none; background-color: #F5F5F5; }")
