# -*- coding: utf-8 -*-
"""
NEURA - Main Window
VSCode Layout + Apple Design Language
"""

import sys
import os
import json
import threading
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QTextEdit, QLabel, QFileDialog, QSplitter, QTabWidget,
    QListWidget, QListWidgetItem, QCheckBox, QMessageBox, QProgressBar,
    QStatusBar, QMenuBar, QMenu, QApplication, QInputDialog, QDialog,
    QFrame, QComboBox, QGroupBox, QToolButton, QScrollArea, QButtonGroup,
    QStackedWidget, QGraphicsDropShadowEffect, QTreeWidget, QTreeWidgetItem
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QTime, QSize
from PySide6.QtGui import QAction, QFont, QIcon, QTextCursor, QColor

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from desk_app.app.utils.config_manager import config_manager
from desk_app.app.utils.mermaid_renderer import MermaidRenderer
from desk_app.app.utils.log_extractor import NodeLogManager
from desk_app.app.utils.step_result_loader import StepResultLoader
from desk_app.app.dialogs.result_dialog import StepResultDialog
from desk_app.app.styles import DARK_THEME, LIGHT_THEME, ThemeManager


# ============================================================
#                    WORKER THREADS
# ============================================================

class LogStream:
    """Custom stream for redirecting print output to UI"""
    def __init__(self, log_signal):
        self.log_signal = log_signal
        self.buffer = ''

    def write(self, text):
        self.buffer += text
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                if line.strip():
                    self.log_signal.emit(line)
            self.buffer = lines[-1]

    def flush(self):
        if self.buffer.strip():
            self.log_signal.emit(self.buffer)
            self.buffer = ''


class AgentWorker(QThread):
    """Worker thread for running the Neuroimaging Agent"""
    update_signal = Signal(dict)
    finished_signal = Signal(dict)
    error_signal = Signal(str)
    task_update_signal = Signal(dict)
    log_signal = Signal(str)
    run_dir_signal = Signal(str)  # 传递运行目录路径
    iteration_signal = Signal(dict)  # 迭代状态更新信号

    _modules_imported = False
    _create_agent = None
    _TaskManager = None
    _TaskStatus = None
    _Path = None
    _get_run_tracker = None

    def __init__(self, question, thread_id=None):
        super().__init__()
        self.question = question
        self.thread_id = thread_id
        self._is_paused = False
        self._pause_condition = threading.Condition()

    def pause(self):
        with self._pause_condition:
            self._is_paused = True

    def resume(self):
        with self._pause_condition:
            self._is_paused = False
            self._pause_condition.notify()

    def _import_modules(self):
        if not AgentWorker._modules_imported:
            try:
                from src.agent.research_graph import create_agent
                from src.agent.task_manager import TaskManager, TaskStatus
                from pathlib import Path
                from src.agent.run_tracker import get_run_tracker

                AgentWorker._create_agent = create_agent
                AgentWorker._TaskManager = TaskManager
                AgentWorker._TaskStatus = TaskStatus
                AgentWorker._Path = Path
                AgentWorker._get_run_tracker = get_run_tracker
                AgentWorker._modules_imported = True
            except Exception as e:
                self.error_signal.emit(f"Module import failed: {e}")
                raise

    def run(self):
        import sys
        try:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            log_stream = LogStream(self.log_signal)
            sys.stdout = log_stream
            sys.stderr = log_stream

            try:
                self._import_modules()
                agent = AgentWorker._create_agent()

                # 用于控制监控循环
                self._monitoring = True
                self._run_dir_sent = False

                def monitor_tasks_once():
                    """执行一次监控检查"""
                    tracker = AgentWorker._get_run_tracker()
                    if tracker and tracker.run_dir:
                        # 发送运行目录
                        if not self._run_dir_sent:
                            self.run_dir_signal.emit(str(tracker.run_dir))
                            self._run_dir_sent = True
                        try:
                            task_manager = AgentWorker._TaskManager(AgentWorker._Path(tracker.run_dir))
                            if task_manager.tasks:
                                for task in task_manager.tasks:
                                    self.task_update_signal.emit({
                                        "task_id": task.task_id,
                                        "description": task.description,
                                        "status": task.status.value,
                                        "tool_name": task.tool_name
                                    })
                        except Exception:
                            pass  # 忽略任务管理器的错误

                        # 检查并发送迭代状态更新
                        try:
                            import json
                            run_path = AgentWorker._Path(tracker.run_dir)
                            # 检查迭代评估文件
                            eval_file = run_path / "steps" / "11_evaluate_iteration" / "iteration_evaluation.json"
                            if eval_file.exists():
                                with open(eval_file, 'r', encoding='utf-8') as f:
                                    eval_data = json.load(f)
                                    self.iteration_signal.emit({"type": "evaluation", "data": eval_data})
                            # 检查agent_state.json
                            state_file = run_path / "agent_state.json"
                            if state_file.exists():
                                with open(state_file, 'r', encoding='utf-8') as f:
                                    state = json.load(f)
                                    if "iteration_count" in state:
                                        self.iteration_signal.emit({
                                            "type": "state",
                                            "iteration_count": state.get("iteration_count", 0),
                                            "max_iterations": state.get("max_iterations", 5),
                                            "quality_score": state.get("scientific_quality_score", 0),
                                            "needs_deeper": state.get("needs_deeper_analysis", False)
                                        })
                        except Exception:
                            pass  # 忽略迭代检查错误

                def monitor_loop():
                    """持续监控循环，每秒检查一次"""
                    import time
                    while self._monitoring:
                        monitor_tasks_once()
                        time.sleep(1)  # 每秒检查一次

                from threading import Thread
                monitor_thread = Thread(target=monitor_loop, daemon=True)
                monitor_thread.start()

                result = agent.run(self.question, self.thread_id)

                # 停止监控循环
                self._monitoring = False

                # 最终检查：确保 run_dir 被发送
                tracker = AgentWorker._get_run_tracker()
                if tracker and tracker.run_dir:
                    self.run_dir_signal.emit(str(tracker.run_dir))

                # 最后一次更新任务状态
                monitor_tasks_once()
                self.finished_signal.emit(result)
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
        except Exception as e:
            self.error_signal.emit(str(e))


class GraphWorker(QThread):
    """Worker thread for getting the state graph"""
    finished_signal = Signal(str)
    error_signal = Signal(str)

    # 备用的静态工作流图 - 与10步标准步骤对齐
    FALLBACK_MERMAID = """
graph TD
    A[01 解析研究问题] --> B[02 检索知识库]
    B --> C[03 生成研究计划]
    C --> D[04 数据字段映射]
    D --> E[05 构建研究队列]
    E --> F[06 物化数据集]
    F --> G[07 选择分析工具]
    G --> H[08 执行分析]
    H --> I[09 验证结果]
    I --> J[10 生成报告]

    style A fill:#4CAF50,stroke:#388E3C,color:#fff
    style B fill:#2D2D2D,stroke:#555,color:#ccc
    style C fill:#2D2D2D,stroke:#555,color:#ccc
    style D fill:#2D2D2D,stroke:#555,color:#ccc
    style E fill:#2D2D2D,stroke:#555,color:#ccc
    style F fill:#2D2D2D,stroke:#555,color:#ccc
    style G fill:#2D2D2D,stroke:#555,color:#ccc
    style H fill:#2D2D2D,stroke:#555,color:#ccc
    style I fill:#2D2D2D,stroke:#555,color:#ccc
    style J fill:#2D2D2D,stroke:#555,color:#ccc
"""

    def run(self):
        try:
            from src.agent.research_graph import create_agent
            agent = create_agent()
            mermaid = agent.get_graph_image()

            # 检查返回的mermaid是否有效
            if not mermaid or "无法生成" in mermaid:
                # 使用备用图，不发送错误信号
                self.finished_signal.emit(self.FALLBACK_MERMAID)
            else:
                self.finished_signal.emit(mermaid)
        except Exception as e:
            # 只发送错误信号，不同时发送finished信号（避免冲突）
            self.error_signal.emit(str(e))


# ============================================================
#                    DIALOGS
# ============================================================

class InteractiveDialog(QDialog):
    """Interactive chat dialog"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interactive Session")
        self.setMinimumSize(600, 500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("logDisplay")
        layout.addWidget(self.chat_display)

        # Input area
        input_widget = QWidget()
        input_layout = QHBoxLayout(input_widget)
        input_layout.setContentsMargins(12, 8, 12, 8)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        send_btn = QPushButton("Send")
        send_btn.setProperty("primary", True)
        send_btn.clicked.connect(self.send_message)
        input_layout.addWidget(send_btn)

        layout.addWidget(input_widget)

    def send_message(self):
        message = self.input_field.text().strip()
        if message:
            self.chat_display.append(f"<b>You:</b> {message}")
            self.input_field.clear()
            # TODO: Send to agent and display response


# ============================================================
#                    MAIN WINDOW
# ============================================================

class MainWindow(QMainWindow):
    """Main application window - VSCode style layout"""

    def __init__(self):
        super().__init__()

        # Apply theme
        self.setStyleSheet(ThemeManager.get_stylesheet())

        # Initialize state
        self.agent_worker = None
        self.graph_worker = None
        self.tasks = {}
        self.is_executing = False
        self.is_paused = False

        # Setup UI
        self.init_ui()
        self.setup_menu()
        self.setup_shortcuts()
        self.load_config()

    def init_ui(self):
        """Initialize the main UI layout"""
        self.setWindowTitle("AI Scientist for Neuroimaging")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1000, 700)

        # 定义10步标准步骤（供侧边栏使用）
        self.standard_steps = [
            ("01_parse_question", "解析研究问题"),
            ("02_search_knowledge", "检索知识库"),
            ("03_generate_plan", "生成研究计划"),
            ("04_map_data_fields", "数据字段映射"),
            ("05_build_cohort", "构建研究队列"),
            ("06_materialize_data", "物化数据集"),
            ("07_select_tools", "选择分析工具"),
            ("08_execute_analysis", "执行分析"),
            ("09_validate_results", "验证结果"),
            ("10_generate_report", "生成报告")
        ]

        # 节点映射表（LangGraph节点 -> UI索引）
        self.node_mapping = {
            "init": -1,
            "parse_question": 0,
            "search_knowledge": 1,
            "generate_plan": 2,
            "map_data_fields": 3,
            "build_cohort": 4,
            "materialize_data": 5,
            "select_tools": 6,
            "execute_tool": 7,
            "execute_next_task": 7,
            "check_tasks_complete": 7,
            "generate_algorithm_code": 7,
            "execute_analysis": 7,
            "validate_results": 8,
            "generate_report": 9,
            "evaluate_iteration": -1,
            "reflect_and_fix": -1,
            "end": -1,
        }

        # 节点之间的数据流信息
        self.data_flows = {
            0: "问题文本",      # 解析研究问题 → 检索知识库
            1: "知识上下文",    # 检索知识库 → 生成研究计划
            2: "研究计划",      # 生成研究计划 → 数据字段映射
            3: "字段映射",      # 数据字段映射 → 构建研究队列
            4: "队列定义",      # 构建研究队列 → 物化数据集
            5: "数据集",        # 物化数据集 → 选择分析工具
            6: "工具列表",      # 选择分析工具 → 执行分析
            7: "分析结果",      # 执行分析 → 验证结果
            8: "验证报告",      # 验证结果 → 生成报告
        }

        # 节点日志管理器（用于提取关键日志）
        self.node_log_manager = NodeLogManager(num_nodes=10)

        # 节点任务列表（每个节点的任务）
        self._node_tasks = {i: [] for i in range(10)}

        # 当前运行目录和结果加载器
        self._current_run_dir = None
        self._result_loader = StepResultLoader()

        # 当前选中的节点索引（用于查看结果）
        self._selected_node_index = None

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Activity Bar (left icon sidebar)
        self.activity_bar = self.create_activity_bar()
        main_layout.addWidget(self.activity_bar)

        # 2. Horizontal Splitter for Sidebar + Editor Area (可拖拽调整宽度)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(4)  # 拖拽手柄宽度
        self.main_splitter.setStyleSheet("""
            QSplitter::handle:horizontal {
                background-color: #3C3C3C;
                width: 4px;
            }
            QSplitter::handle:horizontal:hover {
                background-color: #0A84FF;
            }
        """)

        # 2a. Sidebar Panel (可伸缩)
        self.sidebar_panel = self.create_sidebar_panel()
        self.main_splitter.addWidget(self.sidebar_panel)

        # 2b. Main Editor Area (with vertical splitter for bottom panel)
        editor_splitter = QSplitter(Qt.Vertical)
        editor_splitter.setHandleWidth(1)

        # Editor content
        self.editor_area = self.create_editor_area()
        editor_splitter.addWidget(self.editor_area)

        # Bottom panel (logs, results, terminal)
        self.bottom_panel = self.create_bottom_panel()
        editor_splitter.addWidget(self.bottom_panel)

        # 使用弹性比例代替固定像素，防止布局压缩
        editor_splitter.setStretchFactor(0, 7)  # 编辑区70%
        editor_splitter.setStretchFactor(1, 3)  # 底部面板30%

        # 防止完全压缩
        editor_splitter.setCollapsible(0, False)  # 编辑区不可压缩
        editor_splitter.setCollapsible(1, False)  # 底部面板不可压缩

        self.main_splitter.addWidget(editor_splitter)

        # 设置初始宽度和拉伸因子
        self.main_splitter.setSizes([280, 800])  # 初始宽度
        self.main_splitter.setStretchFactor(0, 0)  # 侧边栏不自动伸展
        self.main_splitter.setStretchFactor(1, 1)  # 编辑区自动伸展

        main_layout.addWidget(self.main_splitter, 1)

        # Status bar
        self.setup_status_bar()

    def create_activity_bar(self):
        """Create the left activity bar (VSCode style)"""
        activity_bar = QWidget()
        activity_bar.setObjectName("activityBar")
        activity_bar.setFixedWidth(48)

        layout = QVBoxLayout(activity_bar)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)

        # Activity buttons
        self.activity_buttons = QButtonGroup(self)
        self.activity_buttons.setExclusive(True)

        buttons_config = [
            ("explorer", "Explorer", self.show_explorer_panel),
            ("search", "Search", self.show_search_panel),
            ("workflow", "Workflow", self.show_workflow_panel),
            ("history", "History", self.show_history_panel),
            ("settings", "Settings", self.show_settings_panel),
        ]

        # 中文标签
        button_labels = {
            "explorer": "资源",
            "search": "搜索",
            "workflow": "流程",
            "history": "历史",
            "settings": "设置",
        }

        for name, tooltip, callback in buttons_config:
            btn = QPushButton(button_labels.get(name, name))
            btn.setObjectName(f"activity_{name}")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(callback)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 11px;
                    padding: 8px 4px;
                    min-height: 40px;
                }
            """)
            self.activity_buttons.addButton(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Theme toggle at bottom
        self.theme_btn = QPushButton("主题")
        self.theme_btn.setToolTip("切换主题")
        self.theme_btn.clicked.connect(self.toggle_theme)
        layout.addWidget(self.theme_btn)

        # Set first button as active
        first_btn = self.activity_buttons.buttons()[0]
        first_btn.setChecked(True)

        return activity_bar

    def create_sidebar_panel(self):
        """Create the sidebar panel (可拖拽调整宽度)"""
        sidebar = QWidget()
        sidebar.setObjectName("sidebarPanel")
        sidebar.setMinimumWidth(200)  # 最小宽度
        sidebar.setMaximumWidth(500)  # 最大宽度

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("sidebarHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)

        self.sidebar_title = QLabel("EXPLORER")
        self.sidebar_title.setObjectName("sidebarTitle")
        header_layout.addWidget(self.sidebar_title)
        header_layout.addStretch()

        layout.addWidget(header)

        # Stacked widget for different panels
        self.sidebar_stack = QStackedWidget()
        layout.addWidget(self.sidebar_stack, 1)

        # Add different panels
        self.sidebar_stack.addWidget(self.create_explorer_content())
        self.sidebar_stack.addWidget(self.create_search_content())
        self.sidebar_stack.addWidget(self.create_workflow_content())
        self.sidebar_stack.addWidget(self.create_history_content())
        self.sidebar_stack.addWidget(self.create_settings_content())

        return sidebar

    def create_explorer_content(self):
        """Create explorer panel - displays literature and data files (VSCode style)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ===== Literature Section =====
        lit_header = self._create_section_header("文献列表", "literature")
        layout.addWidget(lit_header)

        # Literature tree
        self.literature_tree = QTreeWidget()
        self.literature_tree.setHeaderHidden(True)
        self.literature_tree.setIndentation(16)
        self.literature_tree.setObjectName("explorerTree")
        self.literature_tree.itemClicked.connect(self._on_literature_clicked)
        self.literature_tree.itemDoubleClicked.connect(self._on_literature_double_clicked)
        layout.addWidget(self.literature_tree)

        # ===== Data Section =====
        data_header = self._create_section_header("数据文件", "data")
        layout.addWidget(data_header)

        # Data tree
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderHidden(True)
        self.data_tree.setIndentation(16)
        self.data_tree.setObjectName("explorerTree")
        self.data_tree.itemClicked.connect(self._on_data_clicked)
        self.data_tree.itemDoubleClicked.connect(self._on_data_double_clicked)
        layout.addWidget(self.data_tree, 1)

        layout.addStretch()

        # Initial load
        self._load_literature_list()
        self._load_data_folder()

        return widget

    def _create_section_header(self, title, section_id):
        """Create a collapsible section header"""
        header = QWidget()
        header.setObjectName("sectionHeader")
        header.setFixedHeight(28)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(6)

        # Collapse arrow
        arrow = QLabel("▼")
        arrow.setObjectName(f"{section_id}_arrow")
        arrow.setFixedWidth(12)
        arrow.setStyleSheet("color: #888; font-size: 10px;")
        header_layout.addWidget(arrow)

        # Title
        label = QLabel(title.upper())
        label.setObjectName("sidebarTitle")
        header_layout.addWidget(label, 1)

        # Refresh button
        refresh_btn = QPushButton("⟳")
        refresh_btn.setFixedSize(18, 18)
        refresh_btn.setToolTip("刷新")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #888;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #FFF;
            }
        """)
        refresh_btn.clicked.connect(lambda: self._refresh_section(section_id))
        header_layout.addWidget(refresh_btn)

        header.setStyleSheet("""
            QWidget#sectionHeader {
                background-color: transparent;
                border-bottom: 1px solid #404040;
            }
        """)

        return header

    def _load_literature_list(self):
        """Load literature list from paper folder"""
        literature_path = Path("I:/AGENT-think/paper")
        self.literature_tree.clear()

        if literature_path.exists():
            pdf_files = list(literature_path.glob("*.pdf"))
            if pdf_files:
                for file in sorted(pdf_files):
                    item = QTreeWidgetItem([f"📄 {file.name}"])
                    item.setData(0, Qt.UserRole, str(file))
                    item.setToolTip(0, str(file))
                    self.literature_tree.addTopLevelItem(item)
            else:
                empty_item = QTreeWidgetItem(["(无PDF文件)"])
                empty_item.setDisabled(True)
                self.literature_tree.addTopLevelItem(empty_item)
        else:
            missing_item = QTreeWidgetItem([f"(路径不存在: {literature_path})"])
            missing_item.setDisabled(True)
            self.literature_tree.addTopLevelItem(missing_item)

    def _load_data_folder(self):
        """Load data folder structure"""
        data_path = Path("I:/AGENT-think/data")
        self.data_tree.clear()

        if data_path.exists():
            self._add_folder_to_tree(data_path, self.data_tree.invisibleRootItem(), max_depth=3)
        else:
            missing_item = QTreeWidgetItem([f"(路径不存在: {data_path})"])
            missing_item.setDisabled(True)
            self.data_tree.addTopLevelItem(missing_item)

    def _add_folder_to_tree(self, folder_path, parent_item, current_depth=0, max_depth=3):
        """Recursively add folder contents to tree"""
        if current_depth >= max_depth:
            return

        try:
            items = sorted(folder_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            for item_path in items:
                if item_path.name.startswith('.'):
                    continue  # Skip hidden files

                if item_path.is_dir():
                    folder_item = QTreeWidgetItem([f"📁 {item_path.name}"])
                    folder_item.setData(0, Qt.UserRole, str(item_path))
                    folder_item.setToolTip(0, str(item_path))
                    parent_item.addChild(folder_item)
                    self._add_folder_to_tree(item_path, folder_item, current_depth + 1, max_depth)
                else:
                    # File icon based on extension
                    ext = item_path.suffix.lower()
                    icon = self._get_file_icon(ext)
                    file_item = QTreeWidgetItem([f"{icon} {item_path.name}"])
                    file_item.setData(0, Qt.UserRole, str(item_path))
                    file_item.setToolTip(0, str(item_path))
                    parent_item.addChild(file_item)
        except PermissionError:
            pass

    def _get_file_icon(self, ext):
        """Get icon for file extension"""
        icon_map = {
            '.csv': '📊',
            '.xlsx': '📊',
            '.xls': '📊',
            '.json': '📋',
            '.txt': '📄',
            '.pdf': '📕',
            '.py': '🐍',
            '.md': '📝',
            '.png': '🖼️',
            '.jpg': '🖼️',
            '.jpeg': '🖼️',
            '.nii': '🧠',
            '.gz': '📦',
        }
        return icon_map.get(ext, '📄')

    def _refresh_section(self, section_id):
        """Refresh a section's content"""
        if section_id == "literature":
            self._load_literature_list()
        elif section_id == "data":
            self._load_data_folder()

    def _on_literature_clicked(self, item, column):
        """Handle literature item click"""
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            self.append_output(f"📄 选中文献: {Path(file_path).name}", "info")

    def _on_literature_double_clicked(self, item, column):
        """Handle literature item double-click - open file"""
        file_path = item.data(0, Qt.UserRole)
        if file_path and Path(file_path).exists():
            os.startfile(file_path)

    def _on_data_clicked(self, item, column):
        """Handle data item click"""
        file_path = item.data(0, Qt.UserRole)
        if file_path:
            path = Path(file_path)
            if path.is_file():
                self.append_output(f"📄 选中文件: {path.name}", "info")
            else:
                self.append_output(f"📁 选中文件夹: {path.name}", "info")

    def _on_data_double_clicked(self, item, column):
        """Handle data item double-click - open file/folder"""
        file_path = item.data(0, Qt.UserRole)
        if file_path and Path(file_path).exists():
            os.startfile(file_path)

    def create_search_content(self):
        """Create search panel content"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Panel title
        title = QLabel("知识库搜索")
        title.setObjectName("panelTitle")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 8px 0;
            border-bottom: 1px solid #404040;
            margin-bottom: 8px;
        """)
        layout.addWidget(title)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词搜索...")
        layout.addWidget(self.search_input)

        # Results list
        self.search_results = QListWidget()
        layout.addWidget(self.search_results, 1)

        return widget

    def create_workflow_content(self):
        """Create workflow panel with visual graph (可视化流程图)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Panel title
        title = QLabel("工作流可视化")
        title.setObjectName("panelTitle")
        title.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 4px 0;
            border-bottom: 1px solid #404040;
            margin-bottom: 4px;
        """)
        layout.addWidget(title)

        # ===== 当前阶段标签 =====
        self.stage_label = QLabel("当前阶段：等待开始")
        self.stage_label.setStyleSheet("font-size: 12px; color: #FF9500; font-weight: bold;")
        layout.addWidget(self.stage_label)

        # ===== 可视化流程图（替换原来的节点列表和Mermaid文本）=====
        from .utils.workflow_graph import WorkflowGraphView
        self.workflow_graph = WorkflowGraphView(self.standard_steps, self.data_flows)
        self.workflow_graph.setMinimumHeight(450)
        self.workflow_graph.node_clicked.connect(self._on_workflow_node_clicked)
        layout.addWidget(self.workflow_graph, 1)

        # ===== 总体进度条 =====
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setTextVisible(True)
        self.overall_progress_bar.setFormat("%p%")
        self.overall_progress_bar.setFixedHeight(20)
        self.overall_progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #404040;
                border-radius: 4px;
                background-color: #2D2D2D;
                text-align: center;
                font-size: 11px;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background-color: #34C759;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.overall_progress_bar)

        # ===== 节点详情面板（点击节点时显示）=====
        self.node_details_panel = QWidget()
        self.node_details_panel.setObjectName("nodeDetailsPanel")
        self.node_details_panel.setVisible(False)
        self.node_details_panel.setStyleSheet("""
            QWidget#nodeDetailsPanel {
                background-color: rgba(0, 0, 0, 0.2);
                border-radius: 6px;
                padding: 8px;
            }
        """)
        details_layout = QVBoxLayout(self.node_details_panel)
        details_layout.setContentsMargins(10, 8, 10, 8)
        details_layout.setSpacing(4)

        self.node_details_title = QLabel("节点详情")
        self.node_details_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #FFFFFF;")
        details_layout.addWidget(self.node_details_title)

        self.node_details_content = QLabel("等待执行...")
        self.node_details_content.setWordWrap(True)
        self.node_details_content.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        details_layout.addWidget(self.node_details_content)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        # 查看结果按钮
        self.view_result_btn = QPushButton("查看结果")
        self.view_result_btn.setStyleSheet("""
            QPushButton {
                background-color: #0A84FF;
                color: #FFFFFF;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0070E0;
            }
            QPushButton:disabled {
                background-color: #4C4C4C;
                color: #8A8A8A;
            }
        """)
        self.view_result_btn.clicked.connect(self._open_result_dialog)
        self.view_result_btn.setEnabled(False)  # 默认禁用，有数据时启用
        btn_row.addWidget(self.view_result_btn)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: none;
                padding: 4px 12px;
                border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        close_btn.clicked.connect(lambda: self.node_details_panel.setVisible(False))
        btn_row.addWidget(close_btn)

        btn_row.addStretch()
        details_layout.addLayout(btn_row)

        layout.addWidget(self.node_details_panel)

        # ===== 迭代历史面板 =====
        self.iteration_panel = self._create_iteration_panel()
        layout.addWidget(self.iteration_panel)

        # 保留node_widgets为空列表以兼容旧代码
        self.node_widgets = []
        self.mermaid_available = False  # 不再使用MermaidRenderer

        return widget

    def _create_iteration_panel(self):
        """创建迭代历史面板"""
        panel = QWidget()
        panel.setObjectName("iterationPanel")
        panel.setStyleSheet("""
            QWidget#iterationPanel {
                background-color: rgba(0, 0, 0, 0.15);
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # 标题栏（可折叠）
        header = QHBoxLayout()
        header.setSpacing(8)

        self.iteration_title = QLabel("⟳ 迭代历史")
        self.iteration_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #FFFFFF;")
        header.addWidget(self.iteration_title)

        header.addStretch()

        # 折叠/展开按钮
        self.iteration_toggle_btn = QPushButton("−")
        self.iteration_toggle_btn.setFixedSize(20, 20)
        self.iteration_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C3C3C;
                color: #CCCCCC;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4C4C4C;
            }
        """)
        self.iteration_toggle_btn.clicked.connect(self._toggle_iteration_panel)
        header.addWidget(self.iteration_toggle_btn)

        layout.addLayout(header)

        # 迭代内容区（可折叠）
        self.iteration_content = QWidget()
        content_layout = QVBoxLayout(self.iteration_content)
        content_layout.setContentsMargins(0, 4, 0, 0)
        content_layout.setSpacing(4)

        # 当前状态行
        status_row = QHBoxLayout()
        status_row.setSpacing(16)

        self.iteration_status_label = QLabel("迭代: 0 / 5")
        self.iteration_status_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        status_row.addWidget(self.iteration_status_label)

        self.quality_score_label = QLabel("质量评分: --")
        self.quality_score_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        status_row.addWidget(self.quality_score_label)

        status_row.addStretch()
        content_layout.addLayout(status_row)

        # 迭代历史列表
        self.iteration_list = QListWidget()
        self.iteration_list.setMaximumHeight(120)
        self.iteration_list.setStyleSheet("""
            QListWidget {
                background-color: #1E1E1E;
                border: 1px solid #3C3C3C;
                border-radius: 4px;
                font-size: 10px;
            }
            QListWidget::item {
                color: #CCCCCC;
                padding: 4px 8px;
                border-bottom: 1px solid #2D2D2D;
            }
            QListWidget::item:selected {
                background-color: #094771;
                color: #FFFFFF;
            }
        """)
        content_layout.addWidget(self.iteration_list)

        layout.addWidget(self.iteration_content)

        # 初始化状态
        self._iteration_panel_expanded = True

        return panel

    def _toggle_iteration_panel(self):
        """切换迭代面板的展开/折叠状态"""
        self._iteration_panel_expanded = not self._iteration_panel_expanded
        self.iteration_content.setVisible(self._iteration_panel_expanded)
        self.iteration_toggle_btn.setText("−" if self._iteration_panel_expanded else "+")

    def _update_iteration_display(self):
        """更新迭代显示"""
        if not self._current_run_dir:
            return

        # 获取迭代数据
        current, max_iter = self._result_loader.get_current_iteration()
        score = self._result_loader.get_quality_score()
        history = self._result_loader.get_iteration_history()

        # 更新状态标签
        self.iteration_status_label.setText(f"迭代: {current} / {max_iter}")

        if score > 0:
            # 根据分数显示不同颜色
            if score >= 8:
                color = "#34C759"  # 绿色
            elif score >= 6:
                color = "#FF9500"  # 橙色
            else:
                color = "#FF3B30"  # 红色
            self.quality_score_label.setText(f"质量评分: {score:.1f}")
            self.quality_score_label.setStyleSheet(f"font-size: 11px; color: {color};")
        else:
            self.quality_score_label.setText("质量评分: --")
            self.quality_score_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")

        # 更新历史列表
        self.iteration_list.clear()
        for record in history:
            iteration = record.get("iteration", 0)
            q_score = record.get("quality_score", 0)
            needs_deeper = record.get("needs_deeper", False)
            feedback = record.get("feedback", "")[:40]
            if len(record.get("feedback", "")) > 40:
                feedback += "..."

            # 状态图标
            status_icon = "●" if needs_deeper else "✓"

            # 创建列表项
            from PySide6.QtWidgets import QListWidgetItem
            item_text = f"{status_icon} 第{iteration}次迭代 | 评分: {q_score:.1f}/10"
            if feedback:
                item_text += f"\n   {feedback}"

            item = QListWidgetItem(item_text)
            self.iteration_list.addItem(item)

    def _on_workflow_node_clicked(self, index, details):
        """处理流程图节点点击事件 - 显示详细执行信息"""
        if index < 0 or index >= len(self.standard_steps):
            return

        step_id, step_name = self.standard_steps[index]
        self.node_details_title.setText(f"{step_name}")
        self._selected_node_index = index

        # 获取状态文本和图标
        state_info = {
            'pending': ('等待中', '○'),
            'running': ('执行中', '●'),
            'completed': ('已完成', '✓'),
            'error': ('出错', '✗')
        }

        # 从 _node_details 获取最新信息
        if hasattr(self, '_node_details') and index in self._node_details:
            details = self._node_details[index]

        # 构建详情文本
        lines = []

        if details:
            state = details.get('state', 'pending')
            state_text, state_icon = state_info.get(state, ('等待中', '○'))

            # 状态信息
            lines.append(f"{state_icon} 状态: {state_text}")

            # 时间信息
            if 'start_time' in details:
                lines.append(f"   开始: {details['start_time']}")
            if 'end_time' in details:
                lines.append(f"   结束: {details['end_time']}")
            if 'duration' in details:
                lines.append(f"   耗时: {details['duration']:.1f}秒")

            # 当前动作
            if 'current_action' in details:
                lines.append(f"\n当前: {details['current_action']}")
        else:
            state = 'pending'
            lines.append("○ 状态: 等待中")

        # ===== 节点6: 选择分析工具 - 显示选中的工具列表 =====
        if index == 6 and self._current_run_dir:
            tools = self._result_loader.get_pipeline_tools()
            if tools:
                lines.append(f"\n选中的工具 ({len(tools)}个):")
                for i, tool in enumerate(tools[:6], 1):
                    tool_name = tool.get('tool', '未知')
                    tool_desc = tool.get('description', '')[:25]
                    if tool_desc and len(tool.get('description', '')) > 25:
                        tool_desc += '...'
                    lines.append(f"  {i}. {tool_name}")
                    if tool_desc:
                        lines.append(f"      {tool_desc}")
                if len(tools) > 6:
                    lines.append(f"  ... 还有 {len(tools) - 6} 个工具")

        # ===== 节点7: 执行分析 - 显示任务列表 =====
        if index == 7 and self._current_run_dir:
            tasks = self._result_loader.get_tasks()
            if tasks:
                completed = sum(1 for t in tasks if t.get('status') == 'completed')
                running = sum(1 for t in tasks if t.get('status') in ['in_progress', 'running'])
                lines.append(f"\n任务列表 ({len(tasks)}个, 完成{completed}个):")
                task_icons = {
                    'completed': '✓', 'in_progress': '●', 'running': '●',
                    'pending': '○', 'failed': '✗', 'blocked': '⊘'
                }
                for task in tasks[:8]:
                    task_name = task.get('description', '未知任务')[:30]
                    if len(task.get('description', '')) > 30:
                        task_name += '...'
                    task_status = task.get('status', 'pending')
                    icon = task_icons.get(task_status, '○')
                    tool_name = task.get('tool_name', '')
                    lines.append(f"  {icon} {task_name}")
                    if tool_name:
                        lines.append(f"      工具: {tool_name}")
                    # 显示结果摘要
                    result_summary = self._result_loader.get_task_result_summary(task)
                    if result_summary:
                        lines.append(f"      {result_summary}")
                if len(tasks) > 8:
                    lines.append(f"  ... 还有 {len(tasks) - 8} 个任务")
            else:
                # 从_node_tasks获取任务
                if hasattr(self, '_node_tasks') and index in self._node_tasks and self._node_tasks[index]:
                    node_tasks = self._node_tasks[index]
                    lines.append(f"\n任务列表 ({len(node_tasks)}个):")
                    task_icons = {'completed': '✓', 'running': '●', 'pending': '○', 'error': '✗'}
                    for task in node_tasks[:8]:
                        if isinstance(task, dict):
                            task_name = task.get('name', str(task))[:30]
                            task_status = task.get('status', 'pending')
                            icon = task_icons.get(task_status, '○')
                            lines.append(f"  {icon} {task_name}")
                        else:
                            lines.append(f"  ○ {task}")

        # 其他节点的任务列表（从_node_tasks获取）
        if index not in [6, 7]:
            if details and 'tasks' in details and details['tasks']:
                lines.append("\n任务列表:")
                task_icons = {'completed': '✓', 'running': '●', 'pending': '○', 'error': '✗'}
                for task in details['tasks'][:8]:
                    if isinstance(task, dict):
                        task_name = task.get('name', str(task))
                        task_status = task.get('status', 'pending')
                        icon = task_icons.get(task_status, '○')
                        lines.append(f"  {icon} {task_name}")
                    else:
                        lines.append(f"  ○ {task}")

        # 关键日志
        if details and 'key_logs' in details and details['key_logs']:
            lines.append("\n关键日志:")
            for log in details['key_logs'][-5:]:
                log_text = log[:45] + "..." if len(log) > 45 else log
                lines.append(f"  • {log_text}")

        # 使用的工具
        if details and 'tools_used' in details and details['tools_used']:
            lines.append("\n使用工具:")
            for tool in details['tools_used'][:4]:
                lines.append(f"  • {tool}")

        self.node_details_content.setText("\n".join(lines))

        # 启用/禁用查看结果按钮
        can_view = (state in ['completed', 'running'] and self._current_run_dir is not None)
        self.view_result_btn.setEnabled(can_view)

        self.node_details_panel.setVisible(True)

    def _open_result_dialog(self):
        """打开结果对话框"""
        if self._selected_node_index is None or self._current_run_dir is None:
            return

        step_id = self.standard_steps[self._selected_node_index][0]
        dialog = StepResultDialog(step_id, self._current_run_dir, self)
        dialog.exec()

    def _create_node_row(self, index, step_id, step_name):
        """创建可点击的节点行（带展开详情功能）"""
        # 容器 widget
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # 节点行（可点击）
        row = QPushButton()
        row.setObjectName("nodeRow")
        row.setFixedHeight(30)
        row.setCursor(Qt.PointingHandCursor)
        row.setStyleSheet("""
            QPushButton#nodeRow {
                background-color: transparent;
                border: none;
                border-radius: 4px;
                text-align: left;
                padding: 2px 4px;
            }
            QPushButton#nodeRow:hover {
                background-color: rgba(255, 255, 255, 0.05);
            }
        """)

        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(6)

        # 展开/折叠箭头
        arrow = QLabel("▶")
        arrow.setFixedWidth(12)
        arrow.setStyleSheet("color: #666; font-size: 9px;")
        row_layout.addWidget(arrow)

        # 节点圆圈
        circle = QLabel(str(index + 1))
        circle.setFixedSize(20, 20)
        circle.setAlignment(Qt.AlignCenter)
        circle.setStyleSheet("""
            QLabel {
                border: 2px solid #555;
                border-radius: 10px;
                background-color: #3D3D3D;
                font-size: 9px;
                color: #888;
            }
        """)
        row_layout.addWidget(circle)

        # 步骤名称
        name_label = QLabel(step_name)
        name_label.setStyleSheet("font-size: 11px; color: #CCCCCC;")
        row_layout.addWidget(name_label, 1)

        # 状态指示
        status_label = QLabel("○")
        status_label.setFixedWidth(14)
        status_label.setStyleSheet("color: #555; font-size: 11px;")
        row_layout.addWidget(status_label)

        container_layout.addWidget(row)

        # 详情面板（初始隐藏）
        details_panel = QWidget()
        details_panel.setObjectName("nodeDetails")
        details_panel.setVisible(False)
        details_panel.setStyleSheet("""
            QWidget#nodeDetails {
                background-color: rgba(0, 0, 0, 0.15);
                border-radius: 4px;
                margin-left: 18px;
                margin-right: 4px;
            }
        """)
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(10, 6, 6, 6)
        details_layout.setSpacing(2)

        details_label = QLabel("等待执行...")
        details_label.setWordWrap(True)
        details_label.setStyleSheet("font-size: 10px; color: #888;")
        details_layout.addWidget(details_label)

        container_layout.addWidget(details_panel)

        # 存储数据
        container.step_id = step_id
        container.step_name = step_name
        container.step_index = index
        container.arrow = arrow
        container.circle = circle
        container.name_label = name_label
        container.status_label = status_label
        container.details_panel = details_panel
        container.details_label = details_label
        container.state = 'pending'
        container.details = {}
        container.expanded = False

        # 连接点击事件
        row.clicked.connect(lambda: self._toggle_node_details(container))

        return container

    def _toggle_node_details(self, node):
        """切换节点详情展开/折叠"""
        node.expanded = not node.expanded

        if node.expanded:
            node.arrow.setText("▼")
            node.details_panel.setVisible(True)
            self._update_node_details_content(node)
        else:
            node.arrow.setText("▶")
            node.details_panel.setVisible(False)

    def _update_node_details_content(self, node):
        """更新节点详情内容"""
        details = node.details
        state = node.state

        text = f"状态: {self._get_node_state_text(state)}\n"

        if not details:
            text += "等待执行..."
            node.details_label.setText(text)
            return

        if 'start_time' in details:
            text += f"开始: {details['start_time']}\n"
        if 'end_time' in details:
            text += f"结束: {details['end_time']}\n"
        if 'duration' in details:
            text += f"耗时: {details['duration']:.2f}秒\n"
        if 'tasks' in details and details['tasks']:
            text += "\n任务:\n"
            for task in details['tasks'][:5]:
                text += f"  · {task[:40]}...\n" if len(task) > 40 else f"  · {task}\n"
        if 'tools' in details and details['tools']:
            text += "\n工具:\n"
            for tool in details['tools'][:3]:
                text += f"  · {tool[:30]}...\n" if len(tool) > 30 else f"  · {tool}\n"

        node.details_label.setText(text)

    def _get_node_state_text(self, state):
        """获取状态文本"""
        state_map = {
            'pending': '等待中',
            'running': '执行中',
            'completed': '已完成',
            'error': '出错'
        }
        return state_map.get(state, state)

    def create_history_content(self):
        """Create history panel content (懒加载)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Panel title
        title = QLabel("历史记录")
        title.setObjectName("panelTitle")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 8px 0;
            border-bottom: 1px solid #404040;
            margin-bottom: 8px;
        """)
        layout.addWidget(title)

        # Loading indicator
        self.history_loading_label = QLabel("点击刷新加载历史记录...")
        self.history_loading_label.setObjectName("historyLoadingLabel")
        self.history_loading_label.setAlignment(Qt.AlignCenter)
        self.history_loading_label.setStyleSheet("color: #888888; font-size: 12px; padding: 20px;")
        layout.addWidget(self.history_loading_label)

        # History list
        self.history_list = QListWidget()
        self.history_list.setVisible(False)
        layout.addWidget(self.history_list, 1)

        # Refresh button
        refresh_btn = QPushButton("刷新历史记录")
        refresh_btn.setObjectName("refreshHistoryBtn")
        refresh_btn.clicked.connect(self.load_historical_runs)
        layout.addWidget(refresh_btn)

        # Flag for lazy loading
        self.history_loaded = False

        return widget

    def create_settings_content(self):
        """Create settings panel content with path management"""
        widget = QWidget()
        widget.setObjectName("settingsPanel")

        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        # Panel title
        title = QLabel("系统设置")
        title.setObjectName("panelTitle")
        title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #FFFFFF;
            padding: 8px 0;
            border-bottom: 1px solid #404040;
            margin-bottom: 8px;
        """)
        layout.addWidget(title)

        # Paths Section
        paths_section = QWidget()
        paths_section.setObjectName("settingsSection")
        paths_layout = QVBoxLayout(paths_section)
        paths_layout.setContentsMargins(16, 16, 16, 16)
        paths_layout.setSpacing(12)

        paths_title = QLabel("路径设置")
        paths_title.setObjectName("settingsSectionTitle")
        paths_layout.addWidget(paths_title)

        # Data path row
        data_row = self.create_path_row(
            "数据路径:",
            config_manager.get_data_path(),
            self.set_data_path
        )
        self.data_path_input = data_row.findChild(QLineEdit)
        paths_layout.addWidget(data_row)

        # Paper path row
        paper_row = self.create_path_row(
            "文献路径:",
            config_manager.get_paper_path(),
            self.set_paper_path
        )
        self.paper_path_input = paper_row.findChild(QLineEdit)
        paths_layout.addWidget(paper_row)

        # Memory path row
        memory_row = self.create_path_row(
            "记忆库路径:",
            config_manager.get_memory_path(),
            self.set_memory_path
        )
        self.memory_path_input = memory_row.findChild(QLineEdit)
        paths_layout.addWidget(memory_row)

        layout.addWidget(paths_section)

        # Theme Section
        theme_section = QWidget()
        theme_section.setObjectName("settingsSection")
        theme_layout = QVBoxLayout(theme_section)
        theme_layout.setContentsMargins(16, 16, 16, 16)
        theme_layout.setSpacing(12)

        theme_title = QLabel("外观设置")
        theme_title.setObjectName("settingsSectionTitle")
        theme_layout.addWidget(theme_title)

        # Theme toggle
        theme_row = QWidget()
        theme_row_layout = QHBoxLayout(theme_row)
        theme_row_layout.setContentsMargins(0, 0, 0, 0)
        theme_row_layout.setSpacing(12)

        theme_label = QLabel("主题:")
        theme_label.setObjectName("pathLabel")
        theme_row_layout.addWidget(theme_label)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["深色模式", "浅色模式"])
        self.theme_combo.setCurrentIndex(0 if ThemeManager.get_current_theme() == "dark" else 1)
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        theme_row_layout.addWidget(self.theme_combo)
        theme_row_layout.addStretch()

        theme_layout.addWidget(theme_row)
        layout.addWidget(theme_section)

        # Buttons
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(0, 8, 0, 0)
        btn_layout.setSpacing(12)

        reset_btn = QPushButton("重置默认")
        reset_btn.setObjectName("resetSettingsBtn")
        reset_btn.clicked.connect(self.reset_settings)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()

        save_btn = QPushButton("保存设置")
        save_btn.setObjectName("saveSettingsBtn")
        save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(save_btn)

        layout.addWidget(btn_row)
        layout.addStretch()

        return widget

    def create_path_row(self, label_text, path_value, browse_callback):
        """Create a path configuration row with label, input, and browse button"""
        row = QWidget()
        row.setObjectName("pathRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        label = QLabel(label_text)
        label.setObjectName("pathLabel")
        label.setFixedWidth(80)
        row_layout.addWidget(label)

        path_input = QLineEdit(path_value)
        path_input.setObjectName("pathInput")
        path_input.setMinimumWidth(150)
        path_input.setPlaceholderText("请选择路径...")
        row_layout.addWidget(path_input, 1)

        browse_btn = QPushButton("浏览")
        browse_btn.setObjectName("browseBtn")
        browse_btn.setFixedWidth(60)
        browse_btn.clicked.connect(browse_callback)
        row_layout.addWidget(browse_btn)

        return row

    def on_theme_changed(self, index):
        """Handle theme change from combo box"""
        theme = "dark" if index == 0 else "light"
        ThemeManager.set_theme(theme)
        self.setStyleSheet(ThemeManager.get_stylesheet())
        icon = "☀" if index == 0 else "🌙"
        self.theme_btn.setText(icon)

        # Sync child component themes
        if hasattr(self, 'workflow_graph'):
            self.workflow_graph.set_theme(theme)
        if hasattr(self, 'mermaid_renderer'):
            self.mermaid_renderer.set_theme(theme)
        self._update_status_bar_for_theme(theme)
        self._update_node_styles_for_theme(theme)

    def _update_status_bar_for_theme(self, theme):
        """Update status bar style based on theme"""
        if theme == 'light':
            self.statusBar().setStyleSheet("""
                QStatusBar { background: #FFFFFF; border-top: 1px solid #E5E5E5; }
            """)
            if hasattr(self, 'status_label'):
                self.status_label.setStyleSheet("color: #6E6E73;")
        else:
            self.statusBar().setStyleSheet("""
                QStatusBar { background: #252526; border-top: 1px solid #404040; }
            """)
            if hasattr(self, 'status_label'):
                self.status_label.setStyleSheet("color: #CCCCCC;")

    def _update_node_styles_for_theme(self, theme):
        """Update node status styles based on theme"""
        if not hasattr(self, 'node_widgets'):
            return

        for node in self.node_widgets:
            state = getattr(node, 'state', 'pending')
            # Re-apply current state with new theme colors
            self._apply_node_state_style(node, state, theme)

    def _apply_node_state_style(self, node, state, theme):
        """Apply state-specific style to a node"""
        is_dark = theme == 'dark'

        if state == 'pending':
            bg_color = '#3D3D3D' if is_dark else '#E8E8E8'
            border_color = '#555' if is_dark else '#CCC'
            text_color = '#888' if is_dark else '#999'
            node.circle.setStyleSheet(f"""
                QLabel {{
                    border: 2px solid {border_color};
                    border-radius: 11px;
                    background-color: {bg_color};
                    font-size: 11px;
                    color: {text_color};
                }}
            """)
            node.status_label.setStyleSheet(f"color: {border_color};")
        elif state == 'running':
            node.circle.setStyleSheet("""
                QLabel {
                    border: 2px solid #FF9500;
                    border-radius: 11px;
                    background-color: #FF9500;
                    font-size: 11px;
                    color: #FFF;
                }
            """)
            node.status_label.setStyleSheet("color: #FF9500;")
        elif state == 'completed':
            node.circle.setStyleSheet("""
                QLabel {
                    border: 2px solid #34C759;
                    border-radius: 11px;
                    background-color: #34C759;
                    font-size: 11px;
                    color: #FFF;
                }
            """)
            node.status_label.setStyleSheet("color: #34C759;")
        elif state == 'error':
            node.circle.setStyleSheet("""
                QLabel {
                    border: 2px solid #FF3B30;
                    border-radius: 11px;
                    background-color: #FF3B30;
                    font-size: 11px;
                    color: #FFF;
                }
            """)
            node.status_label.setStyleSheet("color: #FF3B30;")

    def save_settings(self):
        """Save all settings"""
        # Save paths from inputs
        if hasattr(self, 'data_path_input'):
            config_manager.set_data_path(self.data_path_input.text())
        if hasattr(self, 'paper_path_input'):
            config_manager.set_paper_path(self.paper_path_input.text())
        if hasattr(self, 'memory_path_input'):
            config_manager.set_memory_path(self.memory_path_input.text())

        self.log("设置已保存")
        QMessageBox.information(self, "成功", "设置已保存")

    def reset_settings(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(
            self, "确认", "确定要重置所有设置吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Reset to defaults
            default_data = r"I:\AGENT-think\data"
            default_paper = r"I:\AGENT-think\paper"
            default_memory = r"I:\AGENT-think\memory"

            config_manager.set_data_path(default_data)
            config_manager.set_paper_path(default_paper)
            config_manager.set_memory_path(default_memory)

            # Update UI
            if hasattr(self, 'data_path_input'):
                self.data_path_input.setText(default_data)
            if hasattr(self, 'paper_path_input'):
                self.paper_path_input.setText(default_paper)
            if hasattr(self, 'memory_path_input'):
                self.memory_path_input.setText(default_memory)

            self.log("设置已重置为默认值")

    def set_memory_path(self):
        """Set memory storage path"""
        current = config_manager.get_memory_path() or ""
        path = QFileDialog.getExistingDirectory(self, "选择记忆库路径", current)
        if path:
            config_manager.set_memory_path(path)
            if hasattr(self, 'memory_path_input'):
                self.memory_path_input.setText(path)
            self.log(f"记忆库路径设置为: {path}")

    def create_editor_area(self):
        """Create the main editor area"""
        editor = QWidget()
        editor.setObjectName("editorArea")

        layout = QVBoxLayout(editor)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Welcome header
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 20, 0, 20)
        header_layout.setAlignment(Qt.AlignCenter)

        title = QLabel("AI Scientist for Neuroimaging")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("神经影像智能分析 Agent")
        subtitle.setObjectName("welcomeSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)

        layout.addWidget(header_widget)

        # Input card - 全宽度布局
        self.input_card = self.create_input_card()
        layout.addWidget(self.input_card)

        # 节点流程图已移至侧边栏"流程"板块
        # 这里只保留最小化进度指示器

        # Progress indicator
        self.progress_widget = self.create_progress_widget()
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)

        layout.addStretch()

        return editor

    def create_input_card(self):
        """Create the input card with full-width layout and arrow button"""
        card = QWidget()
        card.setObjectName("inputCard")
        # 去掉宽度限制，改为全宽度

        # Add shadow effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 10)
        card.setGraphicsEffect(shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        # Text input - 增加高度
        self.txt_question = QTextEdit()
        self.txt_question.setObjectName("inputField")
        self.txt_question.setPlaceholderText("输入您的研究问题...\n\n示例: 分析SCA3患者与健康对照组的灰质体积差异")
        self.txt_question.setMinimumHeight(150)
        self.txt_question.setMaximumHeight(300)
        self.txt_question.textChanged.connect(self.on_input_changed)
        card_layout.addWidget(self.txt_question)

        # Toolbar
        toolbar = QWidget()
        toolbar.setObjectName("inputToolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(16, 10, 16, 10)
        toolbar_layout.setSpacing(12)

        # Template button
        template_btn = QPushButton("模板")
        template_btn.setFixedHeight(32)
        template_btn.clicked.connect(self.show_question_templates)
        toolbar_layout.addWidget(template_btn)

        # Thread ID input
        thread_label = QLabel("会话ID:")
        thread_label.setStyleSheet("color: #8A8A8A; font-size: 12px;")
        toolbar_layout.addWidget(thread_label)
        self.txt_thread_id = QLineEdit()
        self.txt_thread_id.setPlaceholderText("可选")
        self.txt_thread_id.setFixedWidth(120)
        self.txt_thread_id.setFixedHeight(32)
        toolbar_layout.addWidget(self.txt_thread_id)

        toolbar_layout.addStretch()

        # Character count
        self.char_count = QLabel("0 字符")
        self.char_count.setStyleSheet("color: #6A6A6A; font-size: 11px;")
        toolbar_layout.addWidget(self.char_count)

        # Execute button with arrow icon ↑
        self.btn_execute = QPushButton("↑")
        self.btn_execute.setObjectName("btnExecute")
        self.btn_execute.setToolTip("开始分析 (Ctrl+Enter)")
        self.btn_execute.setEnabled(False)
        self.btn_execute.clicked.connect(self.toggle_execution)
        toolbar_layout.addWidget(self.btn_execute)

        card_layout.addWidget(toolbar)

        return card

    def create_progress_widget(self):
        """Create minimal progress indicator"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        layout.addWidget(self.progress_bar)

        # Status label
        self.progress_label = QLabel("初始化中...")
        self.progress_label.setStyleSheet("color: #8A8A8A; font-size: 12px;")
        self.progress_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_label)

        return widget

    def update_node_status(self, node_id, status):
        """Update the status of a specific node (流程图版本)

        Args:
            node_id: The node identifier (e.g., 'parse_question') or index (0-9)
            status: One of 'pending', 'running', 'completed', 'error'
        """
        # Get node index
        if isinstance(node_id, str):
            idx = self.node_mapping.get(node_id, -1)
            if idx == -1:
                # Try to match partial name
                for key, val in self.node_mapping.items():
                    if key in node_id.lower() or node_id.lower() in key:
                        idx = val
                        break
        else:
            idx = node_id

        if idx < 0 or idx >= len(self.standard_steps):
            return

        step_name = self.standard_steps[idx][1]

        # 记录状态变化时间
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")

        # 维护节点状态和详情（用于进度统计）
        if not hasattr(self, '_node_states'):
            self._node_states = {}
        if not hasattr(self, '_node_details'):
            self._node_details = {}

        self._node_states[idx] = status

        if idx not in self._node_details:
            self._node_details[idx] = {}

        if status == 'running':
            self._node_details[idx]['start_time'] = now
            self._node_details[idx]['state'] = 'running'
            # 初始化任务列表和关键日志（如果不存在）
            if 'tasks' not in self._node_details[idx]:
                self._node_details[idx]['tasks'] = []
            if 'key_logs' not in self._node_details[idx]:
                self._node_details[idx]['key_logs'] = []
            if 'current_action' not in self._node_details[idx]:
                self._node_details[idx]['current_action'] = f"正在执行: {step_name}..."
        elif status == 'completed':
            self._node_details[idx]['end_time'] = now
            self._node_details[idx]['state'] = 'completed'
            self._node_details[idx]['current_action'] = f"已完成: {step_name}"
            # 计算耗时
            if 'start_time' in self._node_details[idx]:
                try:
                    from datetime import datetime
                    start = datetime.strptime(self._node_details[idx]['start_time'], "%H:%M:%S")
                    end = datetime.strptime(now, "%H:%M:%S")
                    duration = (end - start).total_seconds()
                    self._node_details[idx]['duration'] = duration
                except:
                    pass
        elif status == 'error':
            self._node_details[idx]['state'] = 'error'
            self._node_details[idx]['current_action'] = f"出错: {step_name}"
        else:
            self._node_details[idx]['state'] = 'pending'
            self._node_details[idx]['current_action'] = "等待开始..."

        # 从节点日志管理器获取关键日志
        if hasattr(self, 'node_log_manager'):
            node_logs = self.node_log_manager.get_node_display_logs(idx)
            if node_logs:
                self._node_details[idx]['key_logs'] = node_logs

        # 从节点任务列表获取任务
        if hasattr(self, '_node_tasks') and idx in self._node_tasks:
            self._node_details[idx]['tasks'] = self._node_tasks[idx]

        # 更新流程图节点状态
        if hasattr(self, 'workflow_graph'):
            self.workflow_graph.update_node_state(idx, status, self._node_details.get(idx, {}))

        # 更新阶段标签
        if status == 'running' and hasattr(self, 'stage_label'):
            self.stage_label.setText(f"当前阶段：{step_name}")
            self.stage_label.setStyleSheet("font-size: 12px; color: #FF9500; font-weight: bold;")

        # 更新总体进度条
        completed_count = sum(1 for s in self._node_states.values() if s == 'completed')
        running_count = sum(1 for s in self._node_states.values() if s == 'running')
        error_count = sum(1 for s in self._node_states.values() if s == 'error')

        progress = completed_count * 10 + running_count * 5

        if hasattr(self, 'overall_progress_bar'):
            self.overall_progress_bar.setValue(progress)

            if error_count > 0:
                self.overall_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #FF3B30;
                        border-radius: 4px;
                        background-color: #2D2D2D;
                        text-align: center;
                        font-size: 11px;
                        color: #FFFFFF;
                    }
                    QProgressBar::chunk {
                        background-color: #FF3B30;
                        border-radius: 3px;
                    }
                """)
            elif running_count > 0:
                self.overall_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #FF9500;
                        border-radius: 4px;
                        background-color: #2D2D2D;
                        text-align: center;
                        font-size: 11px;
                        color: #FFFFFF;
                    }
                    QProgressBar::chunk {
                        background-color: #FF9500;
                        border-radius: 3px;
                    }
                """)
            elif completed_count == 10:
                self.overall_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #34C759;
                        border-radius: 4px;
                        background-color: #2D2D2D;
                        text-align: center;
                        font-size: 11px;
                        color: #FFFFFF;
                    }
                    QProgressBar::chunk {
                        background-color: #34C759;
                        border-radius: 3px;
                    }
                """)
                if hasattr(self, 'stage_label'):
                    self.stage_label.setText("当前阶段：已完成所有步骤 ✓")
                    self.stage_label.setStyleSheet("font-size: 12px; color: #34C759; font-weight: bold;")
            else:
                self.overall_progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #404040;
                        border-radius: 4px;
                        background-color: #2D2D2D;
                        text-align: center;
                        font-size: 11px;
                        color: #FFFFFF;
                    }
                    QProgressBar::chunk {
                        background-color: #34C759;
                        border-radius: 3px;
                    }
                """)

    def reset_node_status(self):
        """Reset all nodes to pending state (流程图版本)"""
        # 重置状态追踪
        self._node_states = {}
        self._node_details = {}

        # 重置节点日志管理器
        if hasattr(self, 'node_log_manager'):
            self.node_log_manager.clear_all()

        # 重置节点任务列表
        if hasattr(self, '_node_tasks'):
            self._node_tasks = {i: [] for i in range(10)}

        # 重置流程图中的所有节点
        if hasattr(self, 'workflow_graph'):
            self.workflow_graph.reset_all_states()

        # Reset stage label
        if hasattr(self, 'stage_label'):
            self.stage_label.setText("当前阶段：等待开始")
            self.stage_label.setStyleSheet("font-size: 12px; color: #FF9500; font-weight: bold;")

        # Reset overall progress bar
        if hasattr(self, 'overall_progress_bar'):
            self.overall_progress_bar.setValue(0)
            self.overall_progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #404040;
                    border-radius: 4px;
                    background-color: #2D2D2D;
                    text-align: center;
                    font-size: 11px;
                    color: #FFFFFF;
                }
                QProgressBar::chunk {
                    background-color: #34C759;
                    border-radius: 3px;
                }
            """)

    def create_bottom_panel(self):
        """Create the bottom panel (logs, results, terminal)"""
        panel = QWidget()
        panel.setObjectName("bottomPanel")
        panel.setMinimumHeight(150)  # 设置最小高度，防止被压缩

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Panel header with tabs
        header = QWidget()
        header.setObjectName("panelHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 0, 16, 0)
        header_layout.setSpacing(0)

        self.panel_tabs = QButtonGroup(self)
        self.panel_tabs.setExclusive(True)

        tabs = ["输出", "结果", "错误"]
        for i, name in enumerate(tabs):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setChecked(i == 0)
            btn.clicked.connect(lambda checked, idx=i: self.panel_stack.setCurrentIndex(idx))
            self.panel_tabs.addButton(btn)
            header_layout.addWidget(btn)

        header_layout.addStretch()

        # Clear button
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(self.clear_output)
        header_layout.addWidget(clear_btn)

        layout.addWidget(header)

        # Stacked widget for panel content
        self.panel_stack = QStackedWidget()

        # Output tab
        self.log_display = QTextEdit()
        self.log_display.setObjectName("logDisplay")
        self.log_display.setReadOnly(True)
        self.panel_stack.addWidget(self.log_display)

        # Results tab
        self.results_display = QTextEdit()
        self.results_display.setObjectName("logDisplay")
        self.results_display.setReadOnly(True)
        self.panel_stack.addWidget(self.results_display)

        # Errors tab
        self.error_display = QTextEdit()
        self.error_display.setObjectName("logDisplay")
        self.error_display.setReadOnly(True)
        self.panel_stack.addWidget(self.error_display)

        layout.addWidget(self.panel_stack, 1)

        return panel

    def setup_status_bar(self):
        """Setup the status bar"""
        status_widget = QWidget()
        status_widget.setObjectName("statusBar")

        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(8, 0, 8, 0)
        status_layout.setSpacing(16)

        # Status labels
        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        # Version
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("statusLabel")
        status_layout.addWidget(version_label)

        # Create actual status bar and add widget
        # Note: Status bar style is now controlled by global theme stylesheet
        self.statusBar().addPermanentWidget(status_widget, 1)

    def setup_menu(self):
        """Setup the menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction("New Analysis", self.clear_input, "Ctrl+N")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Ctrl+Q")

        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction("Clear Output", self.clear_output)

        # View menu
        view_menu = menubar.addMenu("View")
        view_menu.addAction("Toggle Sidebar", self.toggle_sidebar, "Ctrl+B")
        view_menu.addAction("Toggle Bottom Panel", self.toggle_bottom_panel, "Ctrl+J")
        view_menu.addSeparator()
        view_menu.addAction("Toggle Theme", self.toggle_theme, "Ctrl+T")

        # Help menu
        help_menu = menubar.addMenu("Help")
        help_menu.addAction("About", self.show_about)
        help_menu.addAction("FAQ", self.show_faq)

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Ctrl+Enter to execute
        execute_action = QAction(self)
        execute_action.setShortcut("Ctrl+Return")
        execute_action.triggered.connect(self.execute_analysis)
        self.addAction(execute_action)

    # ============================================================
    #                    PANEL SWITCHING
    # ============================================================

    def show_explorer_panel(self):
        self.sidebar_title.setText("EXPLORER")
        self.sidebar_stack.setCurrentIndex(0)

    def show_search_panel(self):
        self.sidebar_title.setText("SEARCH")
        self.sidebar_stack.setCurrentIndex(1)

    def show_workflow_panel(self):
        self.sidebar_title.setText("WORKFLOW")
        self.sidebar_stack.setCurrentIndex(2)

    def show_history_panel(self):
        self.sidebar_title.setText("HISTORY")
        self.sidebar_stack.setCurrentIndex(3)
        # 懒加载：只在首次访问时加载
        if not getattr(self, 'history_loaded', False):
            self.load_historical_runs()

    def show_settings_panel(self):
        self.sidebar_title.setText("SETTINGS")
        self.sidebar_stack.setCurrentIndex(4)

    def toggle_sidebar(self):
        self.sidebar_panel.setVisible(not self.sidebar_panel.isVisible())

    def toggle_bottom_panel(self):
        self.bottom_panel.setVisible(not self.bottom_panel.isVisible())

    def toggle_theme(self):
        stylesheet = ThemeManager.toggle_theme()
        self.setStyleSheet(stylesheet)
        icon = "" if ThemeManager.get_current_theme() == "dark" else ""
        self.theme_btn.setText(icon)

    # ============================================================
    #                    INPUT HANDLING
    # ============================================================

    def on_input_changed(self):
        text = self.txt_question.toPlainText()
        self.char_count.setText(f"{len(text)} 字符")
        self.btn_execute.setEnabled(bool(text.strip()))

    def clear_input(self):
        self.txt_question.clear()
        self.txt_thread_id.clear()
        self.log("输入已清空")

    def clear_output(self):
        self.log_display.clear()
        self.results_display.clear()
        self.error_display.clear()

    def show_question_templates(self):
        """Show question templates menu"""
        menu = QMenu(self)
        templates = [
            "Analyze gray matter volume differences between SCA3 patients and healthy controls",
            "Resting-state functional connectivity analysis",
            "Structural MRI data analysis",
            "Multi-modal neuroimaging data fusion analysis",
            "Brain age prediction study",
            "Alzheimer's disease early diagnosis research",
            "Depression patient brain network characteristics analysis",
        ]

        for template in templates:
            action = menu.addAction(template)
            action.triggered.connect(lambda checked, t=template: self.txt_question.setPlainText(t))

        # Show menu at button position
        btn = self.sender()
        if btn:
            menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    # ============================================================
    #                    EXECUTION
    # ============================================================

    def toggle_execution(self):
        if not self.is_executing:
            self.execute_analysis()
        elif self.is_paused:
            self.resume_analysis()
        else:
            self.pause_analysis()

    def execute_analysis(self):
        """Start the analysis"""
        data_path = config_manager.get_data_path()
        paper_path = config_manager.get_paper_path()
        question = self.txt_question.toPlainText().strip()
        thread_id = self.txt_thread_id.text().strip() or None

        # Validation
        if not question:
            QMessageBox.warning(self, "警告", "请输入研究问题")
            return

        if not os.path.exists(data_path):
            try:
                os.makedirs(data_path)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法创建数据路径: {e}")
                return

        if not os.path.exists(paper_path):
            try:
                os.makedirs(paper_path)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"无法创建文献路径: {e}")
                return

        # Log
        self.log("开始分析...")
        self.log(f"问题: {question}")
        if thread_id:
            self.log(f"会话ID: {thread_id}")

        # Update state
        self.is_executing = True
        self.is_paused = False
        self.current_node_index = 0

        # Update UI - 节点流程图已在侧边栏显示，这里只更新按钮和进度
        self.btn_execute.setText("⏸")
        self.btn_execute.setToolTip("暂停分析")
        # node_status_widget 已移至侧边栏
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在初始化智能体...")
        self.status_label.setText("运行中...")

        # Reset and start first node
        self.reset_node_status()
        self.update_node_status(0, 'running')

        # Start worker
        self.agent_worker = AgentWorker(question, thread_id)
        self.agent_worker.update_signal.connect(self.on_task_update)
        self.agent_worker.task_update_signal.connect(self.on_task_update)
        self.agent_worker.finished_signal.connect(self.on_analysis_finished)
        self.agent_worker.error_signal.connect(self.on_analysis_error)
        self.agent_worker.log_signal.connect(self.on_agent_log)
        self.agent_worker.run_dir_signal.connect(self.on_run_dir_received)
        self.agent_worker.iteration_signal.connect(self.on_iteration_update)
        self.agent_worker.start()

    def pause_analysis(self):
        """Pause the analysis"""
        if self.agent_worker and self.agent_worker.isRunning():
            self.agent_worker.pause()
            self.is_paused = True
            self.btn_execute.setText("▶")
            self.btn_execute.setToolTip("继续分析")
            self.progress_label.setText("已暂停")
            self.status_label.setText("已暂停")
            self.log("分析已暂停")

    def resume_analysis(self):
        """Resume the analysis"""
        if self.agent_worker and self.is_paused:
            self.agent_worker.resume()
            self.is_paused = False
            self.btn_execute.setText("⏸")
            self.btn_execute.setToolTip("暂停分析")
            self.progress_label.setText("运行中...")
            self.status_label.setText("运行中...")
            self.log("分析已继续")

    def on_run_dir_received(self, run_dir_path):
        """接收运行目录路径"""
        if run_dir_path and run_dir_path != str(self._current_run_dir):
            self._current_run_dir = Path(run_dir_path)
            self._result_loader.set_run_dir(self._current_run_dir)
            self.log(f"运行目录: {self._current_run_dir.name}")
            # 刷新当前选中节点的详情（如果有）
            if self._selected_node_index is not None:
                self._refresh_selected_node_details()

    def _refresh_selected_node_details(self):
        """刷新当前选中节点的详情显示"""
        if self._selected_node_index is None:
            return
        # 从 _node_details 获取当前节点的详情
        details = {}
        if hasattr(self, '_node_details') and self._selected_node_index in self._node_details:
            details = self._node_details[self._selected_node_index]
        # 重新调用节点点击处理来刷新显示
        self._on_workflow_node_clicked(self._selected_node_index, details)

    def on_iteration_update(self, data):
        """处理迭代状态更新"""
        data_type = data.get("type", "")

        if data_type == "state":
            # 更新迭代状态标签
            current = data.get("iteration_count", 0)
            max_iter = data.get("max_iterations", 5)
            score = data.get("quality_score", 0)

            self.iteration_status_label.setText(f"迭代: {current} / {max_iter}")

            if score > 0:
                if score >= 8:
                    color = "#34C759"
                elif score >= 6:
                    color = "#FF9500"
                else:
                    color = "#FF3B30"
                self.quality_score_label.setText(f"质量评分: {score:.1f}")
                self.quality_score_label.setStyleSheet(f"font-size: 11px; color: {color};")

        elif data_type == "evaluation":
            # 更新迭代历史列表
            eval_data = data.get("data", {})
            if eval_data:
                self._update_iteration_display()

    def on_task_update(self, data):
        """Handle task status updates and update node status"""
        description = data.get("description", "")
        status = data.get("status", "")
        tool_name = data.get("tool_name", "")
        task_id = data.get("task_id", "")

        # Update progress label
        self.progress_label.setText(f"{description}...")
        self.log(f"任务: {description} - {status}")

        # Try to match to a node and update its status
        node_id = self.match_task_to_node(description, tool_name)
        if node_id is not None:
            # 将任务添加到节点任务列表
            if hasattr(self, '_node_tasks'):
                # 查找是否已存在该任务
                existing_task = None
                for task in self._node_tasks.get(node_id, []):
                    if isinstance(task, dict) and task.get('task_id') == task_id:
                        existing_task = task
                        break

                if existing_task:
                    # 更新现有任务状态
                    existing_task['status'] = status
                else:
                    # 添加新任务
                    if node_id not in self._node_tasks:
                        self._node_tasks[node_id] = []
                    self._node_tasks[node_id].append({
                        'task_id': task_id,
                        'name': description[:40] + ('...' if len(description) > 40 else ''),
                        'status': status,
                        'tool_name': tool_name
                    })
                    # 限制任务数量
                    if len(self._node_tasks[node_id]) > 10:
                        self._node_tasks[node_id].pop(0)

            # 更新节点详情中的工具列表
            if hasattr(self, '_node_details'):
                if node_id not in self._node_details:
                    self._node_details[node_id] = {}
                if 'tools_used' not in self._node_details[node_id]:
                    self._node_details[node_id]['tools_used'] = []
                if tool_name and tool_name not in self._node_details[node_id]['tools_used']:
                    self._node_details[node_id]['tools_used'].append(tool_name)
                # 更新当前动作
                self._node_details[node_id]['current_action'] = description

            if status == "completed":
                self.update_node_status(node_id, 'completed')
                # Start next node
                if node_id + 1 < len(self.standard_steps):
                    self.update_node_status(node_id + 1, 'running')
            elif status == "running" or status == "pending":
                self.update_node_status(node_id, 'running')

    def match_task_to_node(self, description, tool_name):
        """Match a task description to a node index"""
        desc_lower = description.lower()

        # Mapping keywords to node indices
        keyword_map = {
            0: ["parse", "解析", "问题", "question"],
            1: ["search", "检索", "知识", "knowledge"],
            2: ["plan", "计划", "generate", "生成"],
            3: ["map", "映射", "field", "字段"],
            4: ["cohort", "队列", "build"],
            5: ["materialize", "物化", "data"],
            6: ["tool", "工具", "select"],
            7: ["execute", "执行", "analysis"],
            8: ["validate", "验证", "result"],
            9: ["report", "报告", "generate"],
        }

        for idx, keywords in keyword_map.items():
            for kw in keywords:
                if kw in desc_lower:
                    return idx

        return None

    def on_agent_log(self, message):
        """Handle log messages from agent and update node status"""
        self.log(message)

        # 获取当前活跃节点索引
        current_node_idx = self.node_log_manager.current_node_idx

        # 使用日志提取器处理消息
        log_info = self.node_log_manager.process_log(message, current_node_idx)

        # 如果提取到关键信息，更新节点详情
        if log_info and hasattr(self, '_node_details'):
            if current_node_idx not in self._node_details:
                self._node_details[current_node_idx] = {}
            if 'key_logs' not in self._node_details[current_node_idx]:
                self._node_details[current_node_idx]['key_logs'] = []

            # 添加关键日志
            display_text = log_info.get('display', '')
            if display_text and display_text not in self._node_details[current_node_idx]['key_logs']:
                self._node_details[current_node_idx]['key_logs'].append(display_text)
                # 保留最新10条
                if len(self._node_details[current_node_idx]['key_logs']) > 10:
                    self._node_details[current_node_idx]['key_logs'].pop(0)

            # 更新当前动作
            if log_info.get('type') in ('start', 'tool', 'analysis', 'task'):
                self._node_details[current_node_idx]['current_action'] = display_text

            # 同步到流程图
            if hasattr(self, 'workflow_graph'):
                self.workflow_graph.update_node_state(
                    current_node_idx,
                    self._node_states.get(current_node_idx, 'pending'),
                    self._node_details.get(current_node_idx, {})
                )

        # 方法1：解析 [STREAM] 格式（最可靠 - 直接从LangGraph获取节点名）
        if "[STREAM]" in message and "节点" in message:
            import re
            match = re.search(r'\[STREAM\] 节点 \d+: (\w+)', message)
            if match:
                node_name = match.group(1)
                self._update_node_by_name(node_name)
                return

        # 方法2：备用 - 文本匹配（向后兼容）
        msg_lower = message.lower()
        for idx, (step_id, step_name) in enumerate(self.standard_steps):
            if step_id in msg_lower or step_name in msg_lower:
                if "完成" in message or "finished" in msg_lower or "completed" in msg_lower:
                    self.update_node_status(idx, 'completed')
                    if idx + 1 < len(self.standard_steps):
                        self.update_node_status(idx + 1, 'running')
                elif "开始" in message or "starting" in msg_lower or "running" in msg_lower:
                    self.update_node_status(idx, 'running')

    def _update_node_by_name(self, node_name):
        """根据LangGraph节点名实时更新UI状态"""
        if node_name in self.node_mapping:
            idx = self.node_mapping[node_name]
            if idx >= 0:
                # 更新节点日志管理器的当前节点
                self.node_log_manager.set_current_node(idx)

                # 将之前running的节点标记为completed
                if hasattr(self, '_node_states'):
                    for i in range(len(self.standard_steps)):
                        if self._node_states.get(i) == 'running' and i != idx:
                            self.update_node_status(i, 'completed')

                # 设置当前节点为running
                self.update_node_status(idx, 'running')

                # 更新阶段标签
                if idx < len(self.standard_steps):
                    step_name = self.standard_steps[idx][1]
                    self.stage_label.setText(f"当前阶段：{step_name}")

    def on_analysis_finished(self, result):
        """Handle analysis completion"""
        self.is_executing = False
        self.is_paused = False

        # Mark all nodes as completed
        for i in range(len(self.node_widgets)):
            self.update_node_status(i, 'completed')

        # Update UI
        self.btn_execute.setText("↑")
        self.btn_execute.setToolTip("开始分析 (Ctrl+Enter)")
        self.progress_bar.setValue(100)
        self.progress_label.setText("分析完成")
        self.status_label.setText("就绪")

        # Show results
        self.log("分析成功完成")
        results_text = f"阶段: {result.get('phase', 'unknown')}\n"
        if result.get("report_path"):
            results_text += f"报告路径: {result['report_path']}\n"
        self.results_display.setPlainText(results_text)

        # Switch to results tab
        self.panel_tabs.buttons()[1].click()

        QMessageBox.information(self, "完成", "分析成功完成!")

    def on_analysis_error(self, error):
        """Handle analysis error"""
        self.is_executing = False
        self.is_paused = False

        # Mark current node as error
        if hasattr(self, 'current_node_index'):
            for i, node in enumerate(self.node_widgets):
                if node['state'] == 'running':
                    self.update_node_status(i, 'error')
                    break

        # Update UI
        self.btn_execute.setText("↑")
        self.btn_execute.setToolTip("开始分析 (Ctrl+Enter)")
        self.progress_widget.setVisible(False)
        self.status_label.setText("错误")

        # Show error
        self.log(f"错误: {error}")
        self.error_display.append(f"[错误] {error}")

        # Switch to errors tab
        self.panel_tabs.buttons()[2].click()

        QMessageBox.critical(self, "错误", f"分析失败: {error}")

    # ============================================================
    #                    UTILITIES
    # ============================================================

    def log(self, message):
        """Add message to log display"""
        timestamp = QTime.currentTime().toString("HH:mm:ss")
        self.log_display.append(f"[{timestamp}] {message}")
        self.log_display.moveCursor(QTextCursor.End)

    def load_config(self):
        """Load configuration"""
        pass

    def load_historical_runs(self):
        """Load historical run data (懒加载)"""
        # Show loading state
        if hasattr(self, 'history_loading_label'):
            self.history_loading_label.setText("正在加载历史记录...")
            self.history_loading_label.setStyleSheet("color: #FF9500; font-size: 12px; padding: 20px;")

        self.history_list.clear()
        count = 0

        try:
            project_root = Path(os.path.dirname(__file__)).parent.parent
            runs_dir = project_root / "outputs" / "runs"

            if runs_dir.exists():
                for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                    if run_dir.is_dir():
                        meta_path = run_dir / "meta.json"
                        if meta_path.exists():
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                            run_id = meta.get("run_id", "unknown")
                            question = meta.get("question", "No question")[:50]
                            item = QListWidgetItem(f"{run_id}\n{question}...")
                            item.setData(Qt.UserRole, run_id)
                            self.history_list.addItem(item)
                            count += 1

            # Show list and hide loading label
            if hasattr(self, 'history_loading_label'):
                if count > 0:
                    self.history_loading_label.setVisible(False)
                    self.history_list.setVisible(True)
                else:
                    self.history_loading_label.setText("暂无历史记录")
                    self.history_loading_label.setStyleSheet("color: #888888; font-size: 12px; padding: 20px;")

            self.history_loaded = True

        except Exception as e:
            self.log(f"Failed to load history: {e}")
            if hasattr(self, 'history_loading_label'):
                self.history_loading_label.setText(f"加载失败: {e}")
                self.history_loading_label.setStyleSheet("color: #FF3B30; font-size: 12px; padding: 20px;")

    def show_graph(self):
        """Show the workflow graph (带错误处理)"""
        # 如果mermaid不可用，显示文本版工作流
        if not getattr(self, 'mermaid_available', False):
            self.log("使用文本版工作流图")
            if hasattr(self, 'text_workflow'):
                self.text_workflow.setVisible(True)
            if hasattr(self, 'workflow_status_label'):
                self.workflow_status_label.setText("Mermaid组件不可用，显示文本版工作流")
                self.workflow_status_label.setStyleSheet("color: #888888; font-size: 11px; padding: 5px;")
            return

        # Show loading state
        if hasattr(self, 'workflow_status_label'):
            self.workflow_status_label.setText("正在加载工作流图...")
            self.workflow_status_label.setStyleSheet("color: #FF9500; font-size: 12px; padding: 20px;")
            self.workflow_status_label.setVisible(True)

        self.log("Loading workflow graph...")
        try:
            self.graph_worker = GraphWorker()
            self.graph_worker.finished_signal.connect(self.on_graph_loaded)
            self.graph_worker.error_signal.connect(self.on_graph_error)
            self.graph_worker.start()
        except Exception as e:
            self.on_graph_error(str(e))

    def on_graph_loaded(self, mermaid_code):
        """Handle graph loaded - 简化逻辑，参考原版实现"""
        try:
            if hasattr(self, 'mermaid_renderer') and self.mermaid_renderer:
                # 直接渲染，不检查返回值（render_mermaid始终返回True）
                self.mermaid_renderer.render_mermaid(mermaid_code)
                self.mermaid_renderer.setVisible(True)

                if hasattr(self, 'workflow_status_label'):
                    self.workflow_status_label.setVisible(False)
                self.log("Workflow graph loaded")
        except Exception as e:
            self.on_graph_error(str(e))

    def on_graph_error(self, error):
        """Handle graph loading error - 显示文本版备用工作流"""
        self.log(f"Graph error: {error}")
        if hasattr(self, 'workflow_status_label'):
            # 显示文本版工作流作为备用
            self.workflow_status_label.setText(self._get_text_workflow())
            self.workflow_status_label.setStyleSheet("""
                font-family: "JetBrains Mono", "Consolas", monospace;
                font-size: 10px;
                color: #CCCCCC;
                padding: 10px;
                background-color: #2D2D2D;
                border-radius: 8px;
            """)
            self.workflow_status_label.setVisible(True)

    def _get_text_workflow(self):
        """获取文本版工作流图（当Mermaid不可用时的备用）"""
        return """
研究工作流程
═══════════════════════════

  ┌─────────────────────┐
  │  01 解析研究问题    │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  02 检索知识库      │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  03 搜索文献        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  04 分析数据        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  05 生成假设        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  06 验证假设        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  07 综合结论        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  08 撰写报告        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  09 质量检查        │
  └──────────┬──────────┘
             ↓
  ┌─────────────────────┐
  │  10 输出结果        │
  └─────────────────────┘
"""

    def set_data_path(self):
        """Set research data path"""
        current = config_manager.get_data_path() or ""
        path = QFileDialog.getExistingDirectory(self, "选择数据路径", current)
        if path:
            config_manager.set_data_path(path)
            if hasattr(self, 'data_path_input'):
                self.data_path_input.setText(path)
            self.log(f"数据路径设置为: {path}")

    def set_paper_path(self):
        """Set research paper path"""
        current = config_manager.get_paper_path() or ""
        path = QFileDialog.getExistingDirectory(self, "选择文献路径", current)
        if path:
            config_manager.set_paper_path(path)
            if hasattr(self, 'paper_path_input'):
                self.paper_path_input.setText(path)
            self.log(f"文献路径设置为: {path}")

    def open_data_directory(self):
        """Open data directory in file explorer"""
        data_path = config_manager.get_data_path()
        if os.path.exists(data_path):
            import subprocess
            if sys.platform == "win32":
                subprocess.Popen(f'explorer "{data_path}"')
            elif sys.platform == "darwin":
                subprocess.Popen(["open", data_path])
            else:
                subprocess.Popen(["xdg-open", data_path])
        else:
            QMessageBox.warning(self, "警告", f"路径不存在: {data_path}")

    def preview_report(self):
        """Preview the latest report"""
        try:
            project_root = Path(os.path.dirname(__file__)).parent.parent
            runs_dir = project_root / "outputs" / "runs"

            if runs_dir.exists():
                for run_dir in sorted(runs_dir.iterdir(), reverse=True):
                    report_path = run_dir / "report.md"
                    if report_path.exists():
                        import subprocess
                        if sys.platform == "win32":
                            subprocess.Popen(f'explorer "{report_path}"')
                        elif sys.platform == "darwin":
                            subprocess.Popen(["open", str(report_path)])
                        else:
                            subprocess.Popen(["xdg-open", str(report_path)])
                        return

            QMessageBox.information(self, "提示", "未找到报告")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"打开报告失败: {e}")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "关于",
            "神经影像智能体 v1.0.0\n\n"
            "基于LangGraph构建的智能神经影像数据分析平台\n\n"
            "仅供科研和教育目的使用")

    def show_faq(self):
        """Show FAQ dialog"""
        faq = """
常见问题解答:

1. 数据路径错误
   检查数据路径是否存在且可访问

2. 网络连接失败
   检查网络连接和代理设置

3. 内存不足
   关闭其他程序或减小数据集大小

4. 模型加载失败
   检查模型文件是否存在或重新下载
        """
        QMessageBox.information(self, "常见问题", faq.strip())

    def open_interactive_mode(self):
        """Open interactive dialog"""
        dialog = InteractiveDialog(self)
        dialog.exec()
