# -*- coding: utf-8 -*-
"""
Step Result Dialog - 步骤结果对话框

用于展示步骤的详细结果，包括：
- 输出文件列表
- 代码查看
- 图像查看
- JSON数据查看
"""

from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTreeWidget, QTreeWidgetItem, QLabel, QPushButton,
    QWidget, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from desk_app.app.utils.result_viewer import ResultViewer
from desk_app.app.utils.step_result_loader import StepResultLoader


class StepResultDialog(QDialog):
    """步骤结果对话框"""

    # 步骤名称映射
    STEP_NAMES = {
        "01_parse_question": "解析研究问题",
        "02_search_knowledge": "检索知识库",
        "03_generate_plan": "生成研究计划",
        "04_map_data_fields": "数据字段映射",
        "05_build_cohort": "构建研究队列",
        "06_materialize_data": "物化数据集",
        "07_select_tools": "选择分析工具",
        "08_execute_analysis": "执行分析",
        "09_validate_results": "验证结果",
        "10_generate_report": "生成报告"
    }

    # 文件类型图标
    FILE_ICONS = {
        'py': '📄',
        'json': '📋',
        'png': '🖼️',
        'jpg': '🖼️',
        'jpeg': '🖼️',
        'csv': '📊',
        'nii': '🧠',      # NIfTI神经影像
        'mat': '📐',      # MATLAB数据
        'txt': '📝',      # 文本文件
        'log': '📝',      # 日志文件
    }

    def __init__(self, step_id: str, run_dir: Path, parent=None):
        super().__init__(parent)
        self.step_id = step_id
        self.run_dir = Path(run_dir)
        self.loader = StepResultLoader(run_dir)

        self._setup_ui()
        self._load_results()

    def _setup_ui(self):
        """设置UI"""
        step_name = self.STEP_NAMES.get(self.step_id, self.step_id)
        self.setWindowTitle(f"步骤结果 - {step_name}")
        self.resize(900, 650)
        self.setStyleSheet("""
            QDialog {
                background-color: #1E1E1E;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 标题栏
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel(f"📁 {step_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        layout.addWidget(header)

        # 主内容区 - 分隔器
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #3C3C3C;
                width: 1px;
            }
        """)

        # 左侧：文件列表
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #252526;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # 文件列表标题
        list_header = QLabel("文件列表")
        list_header.setStyleSheet("""
            QLabel {
                background-color: #2D2D2D;
                color: #CCCCCC;
                padding: 8px 12px;
                font-size: 11px;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        left_layout.addWidget(list_header)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #252526;
                border: none;
                outline: none;
            }
            QTreeWidget::item {
                color: #CCCCCC;
                padding: 4px 8px;
            }
            QTreeWidget::item:hover {
                background-color: #2A2D2E;
            }
            QTreeWidget::item:selected {
                background-color: #094771;
                color: #FFFFFF;
            }
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {
                border-image: none;
                image: url(none);
            }
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {
                border-image: none;
                image: url(none);
            }
        """)
        self.file_tree.itemClicked.connect(self._on_tree_item_clicked)
        left_layout.addWidget(self.file_tree)

        splitter.addWidget(left_panel)

        # 右侧：内容查看器
        self.result_viewer = ResultViewer()
        splitter.addWidget(self.result_viewer)

        splitter.setSizes([220, 680])
        layout.addWidget(splitter, 1)

        # 底部状态栏
        status_bar = QWidget()
        status_bar.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-top: 1px solid #3C3C3C;
            }
        """)
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(12, 6, 12, 6)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #8A8A8A; font-size: 11px;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        layout.addWidget(status_bar)

    def _load_results(self):
        """加载步骤产出的所有文件"""
        step_dir = self.run_dir / "steps" / self.step_id
        file_count = 0

        # 1. 加载步骤目录下的文件
        if step_dir.exists():
            files = [f for f in sorted(step_dir.glob("*")) if f.is_file()]
            if files:
                parent = self._add_category_item("📁 步骤输出", len(files))
                for f in files:
                    self._add_file_to_parent(parent, f)
                    file_count += 1

        # 2. 加载生成的代码（仅对执行分析节点）
        if self.step_id == "08_execute_analysis":
            code_dir = self.run_dir / "generated_code"
            if code_dir.exists():
                py_files = list(sorted(code_dir.glob("*.py")))
                if py_files:
                    parent = self._add_category_item("📄 生成代码", len(py_files))
                    for f in py_files:
                        self._add_file_to_parent(parent, f)
                        file_count += 1

            # 加载分析结果
            results_dir = self.run_dir / "analysis_results"
            if results_dir.exists():
                result_files = [f for f in sorted(results_dir.glob("*")) if f.is_file()]
                if result_files:
                    parent = self._add_category_item("📊 分析结果", len(result_files))
                    for f in result_files:
                        self._add_file_to_parent(parent, f)
                        file_count += 1

            # 加载工具输出（按task分组）
            file_count += self._load_tools_grouped()

        # 3. 加载计划（仅对生成计划节点）
        if self.step_id == "03_generate_plan":
            plan_file = step_dir / "plan.json"
            if plan_file.exists():
                parent = self._add_category_item("📋 研究计划", 1)
                self._add_file_to_parent(parent, plan_file)
                file_count += 1

        # 更新状态
        self.status_label.setText(f"共 {file_count} 个文件")

        # 展开第一个节点
        if self.file_tree.topLevelItemCount() > 0:
            first_item = self.file_tree.topLevelItem(0)
            first_item.setExpanded(True)
            # 选择第一个子文件
            if first_item.childCount() > 0:
                self.file_tree.setCurrentItem(first_item.child(0))

    def _add_category_item(self, title: str, count: int) -> QTreeWidgetItem:
        """添加分类节点（可折叠的父节点）"""
        item = QTreeWidgetItem()
        item.setText(0, f"{title} ({count})")
        item.setData(0, Qt.UserRole, None)  # 分类节点不可查看
        self.file_tree.addTopLevelItem(item)
        return item

    def _add_file_to_parent(self, parent: QTreeWidgetItem, file_path: Path):
        """添加文件到父节点下"""
        suffix = file_path.suffix.lower()[1:] if file_path.suffix else ""
        icon = self.FILE_ICONS.get(suffix, '📄')

        child = QTreeWidgetItem(parent)
        child.setText(0, f"{icon} {file_path.name}")
        child.setData(0, Qt.UserRole, str(file_path))

    def _load_tools_grouped(self) -> int:
        """按task分组加载tools目录，返回文件数量"""
        tools_dir = self.run_dir / "tools"
        if not tools_dir.exists():
            return 0

        file_count = 0
        task_dirs = [d for d in sorted(tools_dir.iterdir()) if d.is_dir()]

        if not task_dirs:
            return 0

        # 创建"工具输出"主分类
        tools_parent = QTreeWidgetItem()
        tools_parent.setText(0, f"🔧 工具输出 ({len(task_dirs)}个任务)")
        tools_parent.setData(0, Qt.UserRole, None)
        self.file_tree.addTopLevelItem(tools_parent)

        # 遍历每个task目录
        for task_dir in task_dirs:
            # 解析task名称，提取简短显示名
            task_name = task_dir.name  # e.g., "task_01_dicom_to_nifti_xxx"
            parts = task_name.split('_')
            if len(parts) >= 4:
                # 提取 task_01_toolname 形式
                display_name = f"{parts[0]}_{parts[1]}_{parts[2]}"
            else:
                display_name = task_name

            # 获取该目录下所有文件
            files = [f for f in task_dir.glob("*") if f.is_file()]
            if not files:
                continue

            # 创建task分组节点
            task_item = QTreeWidgetItem(tools_parent)
            task_item.setText(0, f"📂 {display_name} ({len(files)}个文件)")
            task_item.setData(0, Qt.UserRole, None)

            # 添加子文件（最多显示20个）
            sorted_files = sorted(files, key=lambda x: x.name)
            for f in sorted_files[:20]:
                suffix = f.suffix.lower()[1:] if f.suffix else ""
                icon = self.FILE_ICONS.get(suffix, '📄')
                child = QTreeWidgetItem(task_item)
                child.setText(0, f"{icon} {f.name}")
                child.setData(0, Qt.UserRole, str(f))

            # 如果超过20个，添加"更多"提示
            if len(files) > 20:
                more_item = QTreeWidgetItem(task_item)
                more_item.setText(0, f"... 还有 {len(files) - 20} 个文件")
                more_item.setData(0, Qt.UserRole, None)

            file_count += len(files)

        return file_count

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, column: int):
        """树节点点击事件"""
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            self.result_viewer.show_file(file_path)
        else:
            # 如果是分类节点，切换展开/折叠状态
            item.setExpanded(not item.isExpanded())

    def set_theme(self, theme: str):
        """设置主题"""
        if theme == 'light':
            self.setStyleSheet("QDialog { background-color: #FFFFFF; }")
            # 更新其他组件样式...
        else:
            self.setStyleSheet("QDialog { background-color: #1E1E1E; }")

        self.result_viewer.set_theme(theme)


class TaskResultDialog(QDialog):
    """任务结果对话框 - 用于显示单个任务的结果"""

    def __init__(self, task: dict, run_dir: Path, parent=None):
        super().__init__(parent)
        self.task = task
        self.run_dir = Path(run_dir)

        self._setup_ui()
        self._load_task_result()

    def _setup_ui(self):
        """设置UI"""
        task_name = self.task.get("description", "任务")
        self.setWindowTitle(f"任务结果 - {task_name[:30]}")
        self.resize(700, 500)
        self.setStyleSheet("QDialog { background-color: #1E1E1E; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏
        header = QWidget()
        header.setStyleSheet("""
            QWidget {
                background-color: #2D2D2D;
                border-bottom: 1px solid #3C3C3C;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        title = QLabel(f"🔧 {task_name}")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #FFFFFF;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # 状态标签
        status = self.task.get("status", "pending")
        status_colors = {
            "completed": "#34C759",
            "in_progress": "#FF9500",
            "pending": "#8A8A8A",
            "failed": "#FF3B30"
        }
        status_text = {
            "completed": "已完成",
            "in_progress": "执行中",
            "pending": "等待中",
            "failed": "失败"
        }
        status_label = QLabel(status_text.get(status, status))
        status_label.setStyleSheet(f"""
            background-color: {status_colors.get(status, '#8A8A8A')};
            color: #FFFFFF;
            padding: 4px 12px;
            border-radius: 10px;
            font-size: 11px;
        """)
        header_layout.addWidget(status_label)

        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: none;
                padding: 6px 16px;
                border-radius: 4px;
                margin-left: 8px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)

        layout.addWidget(header)

        # 内容区
        self.result_viewer = ResultViewer()
        layout.addWidget(self.result_viewer, 1)

    def _load_task_result(self):
        """加载任务结果"""
        result = self.task.get("result", {})

        if result:
            # 格式化结果为JSON显示
            import json
            formatted = json.dumps(result, indent=2, ensure_ascii=False)
            self.result_viewer.json_viewer.setPlainText(formatted)
            self.result_viewer.file_label.setText("任务结果")
            self.result_viewer.content_stack.setCurrentWidget(self.result_viewer.json_viewer)
        else:
            self.result_viewer.clear()
