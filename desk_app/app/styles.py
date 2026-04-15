# -*- coding: utf-8 -*-
"""
Modern UI Styles - VSCode Layout + Apple Design Language
Inspired by VSCode's layout and Apple's design aesthetics
"""

# ============================================================
#                    COLOR PALETTE
# ============================================================

class Colors:
    """Color palette for both themes"""

    # Brand Colors (Apple Blue)
    PRIMARY = "#0A84FF"
    PRIMARY_HOVER = "#0077ED"
    PRIMARY_PRESSED = "#006ADB"

    # Semantic Colors
    SUCCESS = "#30D158"
    WARNING = "#FF9F0A"
    ERROR = "#FF453A"
    INFO = "#64D2FF"

    # Dark Theme
    class Dark:
        # Backgrounds
        BG_PRIMARY = "#1E1E1E"        # Main background (VSCode style)
        BG_SECONDARY = "#252526"       # Sidebar background
        BG_TERTIARY = "#2D2D2D"        # Cards, panels
        BG_ELEVATED = "#323233"        # Elevated surfaces
        BG_INPUT = "#3C3C3C"           # Input fields

        # Borders
        BORDER = "#404040"
        BORDER_LIGHT = "#4A4A4A"
        BORDER_FOCUS = "#0A84FF"

        # Text
        TEXT_PRIMARY = "#FFFFFF"
        TEXT_SECONDARY = "#CCCCCC"
        TEXT_TERTIARY = "#8A8A8A"
        TEXT_DISABLED = "#5A5A5A"

        # Activity Bar (VSCode left sidebar)
        ACTIVITY_BG = "#333333"
        ACTIVITY_ACTIVE = "#FFFFFF"
        ACTIVITY_INACTIVE = "#858585"
        ACTIVITY_INDICATOR = "#0A84FF"

        # Status Bar
        STATUS_BG = "#007ACC"
        STATUS_TEXT = "#FFFFFF"

    # Light Theme
    class Light:
        # Backgrounds
        BG_PRIMARY = "#FFFFFF"
        BG_SECONDARY = "#F3F3F3"
        BG_TERTIARY = "#FAFAFA"
        BG_ELEVATED = "#FFFFFF"
        BG_INPUT = "#FFFFFF"

        # Borders
        BORDER = "#E5E5E5"
        BORDER_LIGHT = "#EBEBEB"
        BORDER_FOCUS = "#0A84FF"

        # Text
        TEXT_PRIMARY = "#1D1D1F"
        TEXT_SECONDARY = "#6E6E73"
        TEXT_TERTIARY = "#8E8E93"
        TEXT_DISABLED = "#C7C7CC"

        # Activity Bar
        ACTIVITY_BG = "#2C2C2C"
        ACTIVITY_ACTIVE = "#FFFFFF"
        ACTIVITY_INACTIVE = "#858585"
        ACTIVITY_INDICATOR = "#0A84FF"

        # Status Bar
        STATUS_BG = "#007ACC"
        STATUS_TEXT = "#FFFFFF"


# ============================================================
#                    DARK THEME STYLESHEET
# ============================================================

DARK_THEME = """
/* =========================================================
   Global Styles
   ========================================================= */
QWidget {
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px;
    color: #D1D1D6;
    background-color: transparent;
    letter-spacing: 0.2px;
}

QMainWindow {
    background-color: #161618;
}

/* =========================================================
   Activity Bar (Left Icon Sidebar - VSCode Style)
   ========================================================= */
QWidget#activityBar {
    background-color: #1A1A1D;
    border: none;
    border-right: 1px solid #2A2A2D;
    min-width: 48px;
    max-width: 48px;
}

QWidget#activityBar QPushButton {
    background-color: transparent;
    border: none;
    border-radius: 0;
    padding: 12px;
    margin: 0;
    min-width: 48px;
    max-width: 48px;
    min-height: 48px;
    max-height: 48px;
    color: #6E6E73;
    font-size: 20px;
}

QWidget#activityBar QPushButton:hover {
    color: #FFFFFF;
    background-color: rgba(255, 255, 255, 0.08);
}

QWidget#activityBar QPushButton:checked {
    color: #0A84FF;
    border-left: 3px solid #0A84FF;
    background-color: rgba(10, 132, 255, 0.1);
}

/* =========================================================
   Sidebar Panel
   ========================================================= */
QWidget#sidebarPanel {
    background-color: #1E1E21;
    border-right: 1px solid #2A2A2D;
}

QWidget#sidebarHeader {
    background-color: #1E1E21;
    border-bottom: 1px solid #2A2A2D;
    padding: 14px 18px;
}

QLabel#sidebarTitle {
    color: #8E8E93;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* =========================================================
   Editor Area (Main Content)
   ========================================================= */
QWidget#editorArea {
    background-color: #161618;
}

QWidget#editorHeader {
    background-color: #1A1A1D;
    border-bottom: 1px solid #2A2A2D;
}

/* Tab Bar */
QTabWidget::pane {
    border: none;
    background-color: #161618;
}

QTabBar {
    background-color: #1A1A1D;
}

QTabBar::tab {
    background-color: #222225;
    color: #8E8E93;
    border: none;
    border-right: 1px solid #1A1A1D;
    padding: 10px 18px;
    min-width: 120px;
    font-size: 12px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #161618;
    color: #FFFFFF;
    border-bottom: 2px solid #0A84FF;
}

QTabBar::tab:hover:!selected {
    background-color: #28282B;
    color: #D1D1D6;
}

/* =========================================================
   Input Card (Hero Section)
   ========================================================= */
QWidget#inputCard {
    background-color: #252528;
    border: 1px solid #3A3A3D;
    border-radius: 16px;
}

QWidget#inputCard:hover {
    border-color: #4A4A4D;
    background-color: #282830;
}

QWidget#inputCard:focus-within {
    border-color: #0A84FF;
}

QTextEdit#inputField {
    background-color: transparent;
    border: none;
    color: #FFFFFF;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 15px;
    line-height: 1.6;
    padding: 20px 24px;
    selection-background-color: rgba(10, 132, 255, 0.3);
}

QTextEdit#inputField:focus {
    border: none;
}

/* Input Toolbar */
QWidget#inputToolbar {
    background-color: #2A2A2D;
    border-top: 1px solid #3A3A3D;
    border-radius: 0 0 16px 16px;
    padding: 12px 20px;
}

QWidget#inputToolbar QPushButton {
    background-color: transparent;
    border: none;
    color: #8A8A8A;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
}

QWidget#inputToolbar QPushButton:hover {
    background-color: #3C3C3C;
    color: #FFFFFF;
}

/* Execute Button */
QPushButton#btnExecute {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2997FF, stop:1 #0A84FF);
    color: white;
    border: none;
    border-radius: 18px;
    min-width: 36px;
    max-width: 36px;
    min-height: 36px;
    max-height: 36px;
    font-size: 18px;
    font-weight: bold;
}

QPushButton#btnExecute:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3DA5FF, stop:1 #0077ED);
}

QPushButton#btnExecute:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0077ED, stop:1 #006ADB);
}

QPushButton#btnExecute:disabled {
    background: #3A3A3D;
    color: #5A5A5D;
}

/* =========================================================
   Bottom Panel (Logs, Results, Terminal)
   ========================================================= */
QWidget#bottomPanel {
    background-color: #161618;
    border-top: 1px solid #2A2A2D;
}

QWidget#panelHeader {
    background-color: #1A1A1D;
    border-bottom: 1px solid #2A2A2D;
}

QWidget#panelHeader QPushButton {
    background-color: transparent;
    border: none;
    color: #6E6E73;
    padding: 8px 14px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

QWidget#panelHeader QPushButton:checked {
    color: #FFFFFF;
    border-bottom: 2px solid #0A84FF;
}

QWidget#panelHeader QPushButton:hover:!checked {
    color: #D1D1D6;
}

/* Log Display */
QTextEdit#logDisplay {
    background-color: #161618;
    border: none;
    color: #D1D1D6;
    font-family: "SF Mono", "JetBrains Mono", "Fira Code", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    line-height: 1.5;
    padding: 14px;
    selection-background-color: rgba(10, 132, 255, 0.3);
    letter-spacing: 0.3px;
}

/* =========================================================
   Status Bar - Dark Theme (matches theme color)
   ========================================================= */
QWidget#statusBar {
    background-color: #1A1A1D;
    border-top: 1px solid #2A2A2D;
    min-height: 24px;
    max-height: 24px;
}

QLabel#statusLabel {
    color: #8E8E93;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 11px;
    padding: 0 10px;
}

/* =========================================================
   Buttons
   ========================================================= */
QPushButton {
    background-color: #2A2A2D;
    border: 1px solid #3A3A3D;
    border-radius: 8px;
    color: #D1D1D6;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3A3A3D;
    border-color: #4A4A4D;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: #4A4A4D;
}

QPushButton:disabled {
    background-color: #1E1E21;
    border-color: #2A2A2D;
    color: #4A4A4D;
}

/* Primary Button */
QPushButton[primary="true"] {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2997FF, stop:1 #0A84FF);
    border: none;
    color: #FFFFFF;
    font-weight: 600;
}

QPushButton[primary="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3DA5FF, stop:1 #0077ED);
}

QPushButton[primary="true"]:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0077ED, stop:1 #006ADB);
}

/* =========================================================
   Input Fields
   ========================================================= */
QLineEdit {
    background-color: #252528;
    border: 1px solid #3A3A3D;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 8px 14px;
    selection-background-color: rgba(10, 132, 255, 0.4);
}

QLineEdit:focus {
    border-color: #0A84FF;
    background-color: #28282B;
}

QLineEdit::placeholder {
    color: #5A5A5D;
}

QTextEdit, QPlainTextEdit {
    background-color: #1E1E21;
    border: 1px solid #2A2A2D;
    border-radius: 10px;
    color: #FFFFFF;
    padding: 10px 14px;
    selection-background-color: rgba(10, 132, 255, 0.3);
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0A84FF;
}

/* =========================================================
   Scrollbars (Modern Refined Style - Dark)
   ========================================================= */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px 0;
}

QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(255, 255, 255, 0.25);
}

QScrollBar::handle:vertical:pressed {
    background: #0A84FF;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
    border: none;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0 2px;
}

QScrollBar::handle:horizontal {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(255, 255, 255, 0.25);
}

QScrollBar::handle:horizontal:pressed {
    background: #0A84FF;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0;
    border: none;
}

/* =========================================================
   Progress Bar
   ========================================================= */
QProgressBar {
    background-color: #404040;
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #0A84FF;
    border-radius: 4px;
}

/* =========================================================
   ComboBox
   ========================================================= */
QComboBox {
    background-color: #3C3C3C;
    border: 1px solid #505050;
    border-radius: 6px;
    color: #FFFFFF;
    padding: 6px 12px;
    min-width: 100px;
}

QComboBox:hover {
    border-color: #606060;
}

QComboBox:focus {
    border-color: #0A84FF;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #3C3C3C;
    border: 1px solid #505050;
    border-radius: 6px;
    color: #FFFFFF;
    selection-background-color: #0A84FF;
    padding: 4px;
}

/* =========================================================
   Menu
   ========================================================= */
QMenuBar {
    background-color: #3C3C3C;
    border-bottom: 1px solid #404040;
    padding: 0;
}

QMenuBar::item {
    background-color: transparent;
    color: #CCCCCC;
    padding: 6px 12px;
}

QMenuBar::item:selected {
    background-color: #505050;
    color: #FFFFFF;
}

QMenu {
    background-color: #3C3C3C;
    border: 1px solid #505050;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    background-color: transparent;
    color: #CCCCCC;
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0A84FF;
    color: #FFFFFF;
}

QMenu::separator {
    height: 1px;
    background-color: #505050;
    margin: 4px 8px;
}

/* =========================================================
   Group Box
   ========================================================= */
QGroupBox {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 16px;
    font-weight: 500;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: #8A8A8A;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* =========================================================
   List Widget
   ========================================================= */
QListWidget {
    background-color: #252526;
    border: none;
    color: #CCCCCC;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    margin: 2px 4px;
}

QListWidget::item:hover {
    background-color: #2A2A2A;
}

QListWidget::item:selected {
    background-color: #094771;
    color: #FFFFFF;
}

/* =========================================================
   Tree Widget (Explorer Style - Dark)
   ========================================================= */
QTreeWidget#explorerTree {
    background-color: transparent;
    border: none;
    color: #CCCCCC;
    outline: none;
    font-size: 12px;
}

QTreeWidget#explorerTree::item {
    padding: 4px 8px;
    border-radius: 3px;
    margin: 1px 4px;
    min-height: 20px;
}

QTreeWidget#explorerTree::item:hover {
    background-color: rgba(255, 255, 255, 0.05);
}

QTreeWidget#explorerTree::item:selected {
    background-color: rgba(10, 132, 255, 0.3);
    color: #FFFFFF;
}

QTreeWidget#explorerTree::branch {
    background: transparent;
}

QTreeWidget#explorerTree::branch:has-children:!has-siblings:closed,
QTreeWidget#explorerTree::branch:closed:has-children:has-siblings {
    image: none;
    border-image: none;
}

QTreeWidget#explorerTree::branch:open:has-children:!has-siblings,
QTreeWidget#explorerTree::branch:open:has-children:has-siblings {
    image: none;
    border-image: none;
}

/* =========================================================
   Splitter
   ========================================================= */
QSplitter::handle {
    background-color: #404040;
}

QSplitter::handle:horizontal {
    width: 1px;
}

QSplitter::handle:vertical {
    height: 1px;
}

QSplitter::handle:hover {
    background-color: #0A84FF;
}

/* =========================================================
   Tooltip
   ========================================================= */
QToolTip {
    background-color: #3C3C3C;
    border: 1px solid #505050;
    border-radius: 6px;
    color: #FFFFFF;
    padding: 6px 10px;
    font-size: 12px;
}

/* =========================================================
   Message Box
   ========================================================= */
QMessageBox {
    background-color: #2D2D2D;
}

QMessageBox QLabel {
    color: #FFFFFF;
}

/* =========================================================
   CheckBox
   ========================================================= */
QCheckBox {
    color: #CCCCCC;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #606060;
    background-color: #3C3C3C;
}

QCheckBox::indicator:hover {
    border-color: #0A84FF;
}

QCheckBox::indicator:checked {
    background-color: #0A84FF;
    border-color: #0A84FF;
}

/* =========================================================
   Run History Card
   ========================================================= */
QWidget#runCard {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 12px;
}

QWidget#runCard:hover {
    border-color: #0A84FF;
    background-color: #323233;
}

/* =========================================================
   Status Indicators
   ========================================================= */
QLabel#statusSuccess {
    color: #30D158;
}

QLabel#statusWarning {
    color: #FF9F0A;
}

QLabel#statusError {
    color: #FF453A;
}

QLabel#statusRunning {
    color: #0A84FF;
}

/* =========================================================
   Welcome Screen
   ========================================================= */
QWidget#welcomeScreen {
    background-color: #1E1E1E;
}

QLabel#welcomeTitle {
    color: #FFFFFF;
    font-family: "SF Pro Display", "PingFang SC", "Microsoft YaHei UI Light", -apple-system, sans-serif;
    font-size: 36px;
    font-weight: 600;
    letter-spacing: -1px;
    padding: 8px 0;
}

QLabel#welcomeSubtitle {
    color: #0A84FF;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 2px;
}

QPushButton#quickAction {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    border-radius: 10px;
    color: #CCCCCC;
    padding: 16px 24px;
    text-align: left;
    min-height: 60px;
}

QPushButton#quickAction:hover {
    background-color: #3C3C3C;
    border-color: #0A84FF;
    color: #FFFFFF;
}

/* =========================================================
   Node Status (Workflow Visualization)
   ========================================================= */
QWidget#nodeContainer {
    background-color: #2D2D2D;
    border: 1px solid #404040;
    border-radius: 8px;
    padding: 8px;
}

QLabel#nodePending {
    color: #6A6A6A;
}

QLabel#nodeRunning {
    color: #0A84FF;
}

QLabel#nodeComplete {
    color: #30D158;
}

QLabel#nodeError {
    color: #FF453A;
}

/* =========================================================
   10-Node Status Graph (Workflow Progress)
   ========================================================= */
QWidget#nodeStatusGraph {
    background-color: #1E1E21;
    border: 1px solid #2A2A2D;
    border-radius: 14px;
    padding: 18px;
}

QWidget#nodeStatusGraph QLabel#graphTitle {
    color: #FFFFFF;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-bottom: 14px;
}

/* Node Item Container */
QWidget#nodeItem {
    background-color: #252528;
    border: 1px solid #3A3A3D;
    border-radius: 10px;
    padding: 10px 14px;
    min-height: 42px;
}

QWidget#nodeItem:hover {
    background-color: #2A2A2D;
    border-color: #4A4A4D;
}

/* Node Status Indicator Circle */
QWidget#nodeIndicator {
    min-width: 12px;
    max-width: 12px;
    min-height: 12px;
    max-height: 12px;
    border-radius: 6px;
}

QWidget#nodeIndicator[status="pending"] {
    background-color: #6A6A6A;
    border: 2px solid #505050;
}

QWidget#nodeIndicator[status="running"] {
    background-color: #FF9500;
    border: 2px solid #FF9500;
}

QWidget#nodeIndicator[status="completed"] {
    background-color: #30D158;
    border: 2px solid #30D158;
}

QWidget#nodeIndicator[status="error"] {
    background-color: #FF453A;
    border: 2px solid #FF453A;
}

/* Node Labels */
QLabel#nodeNumber {
    color: #8A8A8A;
    font-size: 11px;
    font-weight: 600;
    min-width: 24px;
}

QLabel#nodeName {
    color: #CCCCCC;
    font-size: 13px;
}

QLabel#nodeName[status="running"] {
    color: #FF9500;
    font-weight: 500;
}

QLabel#nodeName[status="completed"] {
    color: #30D158;
}

QLabel#nodeName[status="error"] {
    color: #FF453A;
}

/* Connector Lines */
QWidget#nodeConnector {
    background-color: #404040;
    min-width: 2px;
    max-width: 2px;
    min-height: 12px;
}

QWidget#nodeConnector[completed="true"] {
    background-color: #30D158;
}

/* Progress Summary */
QLabel#progressSummary {
    color: #8A8A8A;
    font-size: 12px;
    padding: 8px 0;
}

QProgressBar#nodeProgress {
    background-color: #404040;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}

QProgressBar#nodeProgress::chunk {
    background-color: #30D158;
    border-radius: 3px;
}

/* Settings Panel Styles */
QWidget#settingsPanel {
    background-color: #161618;
    padding: 24px;
}

QWidget#settingsSection {
    background-color: #1E1E21;
    border: 1px solid #2A2A2D;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
}

QLabel#settingsSectionTitle {
    color: #FFFFFF;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 0.2px;
    margin-bottom: 14px;
}

QWidget#pathRow {
    background-color: transparent;
}

QLabel#pathLabel {
    color: #D1D1D6;
    font-size: 13px;
    min-width: 80px;
}

QLineEdit#pathInput {
    background-color: #252528;
    border: 1px solid #3A3A3D;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 10px 14px;
    font-size: 13px;
}

QLineEdit#pathInput:focus {
    border-color: #0A84FF;
}

QPushButton#browseBtn {
    background-color: #2A2A2D;
    border: 1px solid #3A3A3D;
    border-radius: 8px;
    color: #D1D1D6;
    padding: 10px 18px;
    font-size: 12px;
    font-weight: 500;
    min-width: 60px;
}

QPushButton#browseBtn:hover {
    background-color: #3A3A3D;
    color: #FFFFFF;
}

QPushButton#saveSettingsBtn {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2997FF, stop:1 #0A84FF);
    border: none;
    border-radius: 10px;
    color: #FFFFFF;
    padding: 12px 28px;
    font-size: 14px;
    font-weight: 600;
}

QPushButton#saveSettingsBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3DA5FF, stop:1 #0077ED);
}

QPushButton#resetSettingsBtn {
    background-color: transparent;
    border: 1px solid #3A3A3D;
    border-radius: 10px;
    color: #D1D1D6;
    padding: 12px 28px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton#resetSettingsBtn:hover {
    background-color: #2A2A2D;
    color: #FFFFFF;
}
"""


# ============================================================
#                    LIGHT THEME STYLESHEET
# ============================================================

LIGHT_THEME = """
/* =========================================================
   Global Styles
   ========================================================= */
QWidget {
    font-family: "PingFang SC", "Microsoft YaHei UI", "SF Pro Text", "Segoe UI", sans-serif;
    font-size: 13px;
    color: #1D1D1F;
    background-color: transparent;
    letter-spacing: 0.3px;
}

QMainWindow {
    background-color: #FFFFFF;
}

/* =========================================================
   Activity Bar (Left Icon Sidebar) - Light Theme
   ========================================================= */
QWidget#activityBar {
    background-color: #E8E8E8;
    border: none;
    border-right: 1px solid #D0D0D0;
    min-width: 48px;
    max-width: 48px;
}

QWidget#activityBar QPushButton {
    background-color: transparent;
    border: none;
    border-radius: 0;
    padding: 12px;
    margin: 0;
    min-width: 48px;
    max-width: 48px;
    min-height: 48px;
    max-height: 48px;
    color: #6E6E73;
    font-size: 20px;
}

QWidget#activityBar QPushButton:hover {
    color: #1D1D1F;
    background-color: rgba(0, 0, 0, 0.05);
}

QWidget#activityBar QPushButton:checked {
    color: #007AFF;
    border-left: 2px solid #007AFF;
    background-color: rgba(0, 122, 255, 0.1);
}

/* =========================================================
   Sidebar Panel
   ========================================================= */
QWidget#sidebarPanel {
    background-color: #F3F3F3;
    border-right: 1px solid #E5E5E5;
}

QWidget#sidebarHeader {
    background-color: #F3F3F3;
    border-bottom: 1px solid #E5E5E5;
    padding: 12px 16px;
}

QLabel#sidebarTitle {
    color: #6E6E73;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* =========================================================
   Editor Area
   ========================================================= */
QWidget#editorArea {
    background-color: #FFFFFF;
}

/* Tab Bar */
QTabWidget::pane {
    border: none;
    background-color: #FFFFFF;
}

QTabBar {
    background-color: #F3F3F3;
}

QTabBar::tab {
    background-color: #ECECEC;
    color: #6E6E73;
    border: none;
    border-right: 1px solid #E5E5E5;
    padding: 8px 16px;
    min-width: 120px;
    font-size: 13px;
}

QTabBar::tab:selected {
    background-color: #FFFFFF;
    color: #1D1D1F;
    border-bottom: 2px solid #0A84FF;
}

QTabBar::tab:hover:!selected {
    background-color: #E8E8E8;
}

/* =========================================================
   Input Card
   ========================================================= */
QWidget#inputCard {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
}

QWidget#inputCard:hover {
    border-color: #D0D0D0;
}

QTextEdit#inputField {
    background-color: transparent;
    border: none;
    color: #1D1D1F;
    font-size: 15px;
    padding: 16px 20px;
    selection-background-color: rgba(10, 132, 255, 0.2);
}

/* Input Toolbar */
QWidget#inputToolbar {
    background-color: #FAFAFA;
    border-top: 1px solid #E5E5E5;
    border-radius: 0 0 12px 12px;
    padding: 8px 16px;
}

QWidget#inputToolbar QPushButton {
    background-color: transparent;
    border: none;
    color: #6E6E73;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
}

QWidget#inputToolbar QPushButton:hover {
    background-color: #EBEBEB;
    color: #1D1D1F;
}

/* Execute Button */
QPushButton#btnExecute {
    background-color: #0A84FF;
    color: white;
    border: none;
    border-radius: 16px;
    min-width: 32px;
    max-width: 32px;
    min-height: 32px;
    max-height: 32px;
    font-size: 16px;
    font-weight: bold;
}

QPushButton#btnExecute:hover {
    background-color: #0077ED;
}

QPushButton#btnExecute:disabled {
    background-color: #D0D0D0;
    color: #999999;
}

/* =========================================================
   Bottom Panel
   ========================================================= */
QWidget#bottomPanel {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E5E5;
}

QWidget#panelHeader {
    background-color: #F3F3F3;
    border-bottom: 1px solid #E5E5E5;
}

QWidget#panelHeader QPushButton {
    background-color: transparent;
    border: none;
    color: #6E6E73;
    padding: 6px 12px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

QWidget#panelHeader QPushButton:checked {
    color: #1D1D1F;
    border-bottom: 2px solid #0A84FF;
}

/* Log Display */
QTextEdit#logDisplay {
    background-color: #FFFFFF;
    border: none;
    color: #1D1D1F;
    font-family: "JetBrains Mono", "Fira Code", "SF Mono", "Cascadia Code", "Consolas", monospace;
    font-size: 12px;
    padding: 12px;
    letter-spacing: 0.5px;
}

/* =========================================================
   Status Bar - Light Theme (White)
   ========================================================= */
QWidget#statusBar {
    background-color: #FFFFFF;
    border-top: 1px solid #E5E5E5;
    min-height: 22px;
    max-height: 22px;
}

QLabel#statusLabel {
    color: #6E6E73;
    font-size: 12px;
    padding: 0 8px;
}

/* =========================================================
   Buttons
   ========================================================= */
QPushButton {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 6px 14px;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #F5F5F5;
    border-color: #C0C0C0;
}

QPushButton:pressed {
    background-color: #EBEBEB;
}

QPushButton:disabled {
    background-color: #F5F5F5;
    border-color: #E5E5E5;
    color: #C7C7CC;
}

/* Primary Button */
QPushButton[primary="true"] {
    background-color: #0A84FF;
    border: none;
    color: #FFFFFF;
    font-weight: 500;
}

QPushButton[primary="true"]:hover {
    background-color: #0077ED;
}

/* =========================================================
   Input Fields
   ========================================================= */
QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 6px 12px;
}

QLineEdit:focus {
    border-color: #0A84FF;
}

QLineEdit::placeholder {
    color: #8E8E93;
}

QTextEdit, QPlainTextEdit {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    color: #1D1D1F;
    padding: 8px 12px;
}

QTextEdit:focus, QPlainTextEdit:focus {
    border-color: #0A84FF;
}

/* =========================================================
   Scrollbars (Modern Refined Style - Light)
   ========================================================= */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px 0;
}

QScrollBar::handle:vertical {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(0, 0, 0, 0.25);
}

QScrollBar::handle:vertical:pressed {
    background: #007AFF;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: transparent;
    height: 0;
    border: none;
}

QScrollBar:horizontal {
    background: transparent;
    height: 8px;
    margin: 0 2px;
}

QScrollBar::handle:horizontal {
    background: rgba(0, 0, 0, 0.15);
    border-radius: 4px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(0, 0, 0, 0.25);
}

QScrollBar::handle:horizontal:pressed {
    background: #007AFF;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {
    background: transparent;
    width: 0;
    border: none;
}

/* =========================================================
   Progress Bar
   ========================================================= */
QProgressBar {
    background-color: #E5E5E5;
    border: none;
    border-radius: 4px;
    height: 6px;
}

QProgressBar::chunk {
    background-color: #0A84FF;
    border-radius: 4px;
}

/* =========================================================
   ComboBox
   ========================================================= */
QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 6px 12px;
}

QComboBox:focus {
    border-color: #0A84FF;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    selection-background-color: #0A84FF;
    selection-color: #FFFFFF;
}

/* =========================================================
   Menu
   ========================================================= */
QMenuBar {
    background-color: #F3F3F3;
    border-bottom: 1px solid #E5E5E5;
}

QMenuBar::item {
    background-color: transparent;
    color: #1D1D1F;
    padding: 6px 12px;
}

QMenuBar::item:selected {
    background-color: #E5E5E5;
}

QMenu {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 4px;
}

QMenu::item {
    background-color: transparent;
    color: #1D1D1F;
    padding: 8px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #0A84FF;
    color: #FFFFFF;
}

/* =========================================================
   Group Box
   ========================================================= */
QGroupBox {
    background-color: #FAFAFA;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    margin-top: 16px;
    padding-top: 16px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 8px;
    color: #6E6E73;
    font-size: 11px;
    text-transform: uppercase;
}

/* =========================================================
   List Widget
   ========================================================= */
QListWidget {
    background-color: #F3F3F3;
    border: none;
    color: #1D1D1F;
    outline: none;
}

QListWidget::item {
    padding: 8px 12px;
    border-radius: 4px;
    margin: 2px 4px;
}

QListWidget::item:hover {
    background-color: #EBEBEB;
}

QListWidget::item:selected {
    background-color: #DCE8F7;
    color: #0A5DC2;
}

/* =========================================================
   Tree Widget (Explorer Style - Light)
   ========================================================= */
QTreeWidget#explorerTree {
    background-color: transparent;
    border: none;
    color: #1D1D1F;
    outline: none;
    font-size: 12px;
}

QTreeWidget#explorerTree::item {
    padding: 4px 8px;
    border-radius: 3px;
    margin: 1px 4px;
    min-height: 20px;
}

QTreeWidget#explorerTree::item:hover {
    background-color: rgba(0, 0, 0, 0.05);
}

QTreeWidget#explorerTree::item:selected {
    background-color: rgba(0, 122, 255, 0.2);
    color: #0A5DC2;
}

QTreeWidget#explorerTree::branch {
    background: transparent;
}

QTreeWidget#explorerTree::branch:has-children:!has-siblings:closed,
QTreeWidget#explorerTree::branch:closed:has-children:has-siblings {
    image: none;
    border-image: none;
}

QTreeWidget#explorerTree::branch:open:has-children:!has-siblings,
QTreeWidget#explorerTree::branch:open:has-children:has-siblings {
    image: none;
    border-image: none;
}

/* =========================================================
   Splitter
   ========================================================= */
QSplitter::handle {
    background-color: #E5E5E5;
}

QSplitter::handle:horizontal {
    width: 1px;
}

QSplitter::handle:vertical {
    height: 1px;
}

QSplitter::handle:hover {
    background-color: #0A84FF;
}

/* =========================================================
   Tooltip
   ========================================================= */
QToolTip {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 6px 10px;
    font-size: 12px;
}

/* =========================================================
   CheckBox
   ========================================================= */
QCheckBox {
    color: #1D1D1F;
    spacing: 8px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #C7C7CC;
    background-color: #FFFFFF;
}

QCheckBox::indicator:hover {
    border-color: #0A84FF;
}

QCheckBox::indicator:checked {
    background-color: #0A84FF;
    border-color: #0A84FF;
}

/* =========================================================
   Welcome Screen
   ========================================================= */
QWidget#welcomeScreen {
    background-color: #FFFFFF;
}

QLabel#welcomeTitle {
    color: #1D1D1F;
    font-family: "SF Pro Display", "PingFang SC", "Microsoft YaHei UI Light", -apple-system, sans-serif;
    font-size: 36px;
    font-weight: 600;
    letter-spacing: -1px;
    padding: 8px 0;
}

QLabel#welcomeSubtitle {
    color: #007AFF;
    font-family: "SF Pro Text", "PingFang SC", "Microsoft YaHei UI", sans-serif;
    font-size: 16px;
    font-weight: 500;
    letter-spacing: 2px;
}

QPushButton#quickAction {
    background-color: #FAFAFA;
    border: 1px solid #E5E5E5;
    border-radius: 10px;
    color: #1D1D1F;
    padding: 16px 24px;
    text-align: left;
    min-height: 60px;
}

QPushButton#quickAction:hover {
    background-color: #F0F0F0;
    border-color: #0A84FF;
}

/* =========================================================
   10-Node Status Graph (Light Theme)
   ========================================================= */
QWidget#nodeStatusGraph {
    background-color: #F3F3F3;
    border: 1px solid #E5E5E5;
    border-radius: 12px;
    padding: 16px;
}

QWidget#nodeStatusGraph QLabel#graphTitle {
    color: #1D1D1F;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Node Item Container */
QWidget#nodeItem {
    background-color: #FFFFFF;
    border: 1px solid #E5E5E5;
    border-radius: 8px;
    padding: 8px 12px;
    min-height: 40px;
}

QWidget#nodeItem:hover {
    background-color: #FAFAFA;
    border-color: #D0D0D0;
}

/* Node Labels */
QLabel#nodeNumber {
    color: #6E6E73;
    font-size: 11px;
    font-weight: 600;
    min-width: 24px;
}

QLabel#nodeName {
    color: #1D1D1F;
    font-size: 13px;
}

QLabel#nodeName[status="running"] {
    color: #FF9500;
    font-weight: 500;
}

QLabel#nodeName[status="completed"] {
    color: #34C759;
}

QLabel#nodeName[status="error"] {
    color: #FF3B30;
}

/* Connector Lines */
QWidget#nodeConnector {
    background-color: #E5E5E5;
    min-width: 2px;
    max-width: 2px;
    min-height: 12px;
}

QWidget#nodeConnector[completed="true"] {
    background-color: #34C759;
}

/* Progress Summary */
QLabel#progressSummary {
    color: #6E6E73;
    font-size: 12px;
    padding: 8px 0;
}

QProgressBar#nodeProgress {
    background-color: #E5E5E5;
    border: none;
    border-radius: 3px;
    height: 6px;
}

QProgressBar#nodeProgress::chunk {
    background-color: #34C759;
    border-radius: 3px;
}

/* Settings Panel Styles */
QWidget#settingsPanel {
    background-color: #FFFFFF;
    padding: 20px;
}

QWidget#settingsSection {
    background-color: #FAFAFA;
    border: 1px solid #E5E5E5;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}

QLabel#settingsSectionTitle {
    color: #1D1D1F;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

QLabel#pathLabel {
    color: #1D1D1F;
    font-size: 13px;
    min-width: 80px;
}

QLineEdit#pathInput {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 8px 12px;
    font-size: 13px;
}

QLineEdit#pathInput:focus {
    border-color: #0A84FF;
}

QPushButton#browseBtn {
    background-color: #FFFFFF;
    border: 1px solid #D0D0D0;
    border-radius: 6px;
    color: #1D1D1F;
    padding: 8px 16px;
    font-size: 12px;
    min-width: 60px;
}

QPushButton#browseBtn:hover {
    background-color: #F5F5F5;
}

QPushButton#saveSettingsBtn {
    background-color: #0A84FF;
    border: none;
    border-radius: 8px;
    color: #FFFFFF;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 500;
}

QPushButton#saveSettingsBtn:hover {
    background-color: #0077ED;
}

QPushButton#resetSettingsBtn {
    background-color: transparent;
    border: 1px solid #D0D0D0;
    border-radius: 8px;
    color: #1D1D1F;
    padding: 10px 24px;
    font-size: 14px;
}

QPushButton#resetSettingsBtn:hover {
    background-color: #F5F5F5;
}
"""


# ============================================================
#               BACKWARD COMPATIBILITY
# ============================================================

# Keep the old name for backward compatibility
APPLE_STYLE = DARK_THEME


# ============================================================
#               THEME MANAGER
# ============================================================

class ThemeManager:
    """Theme manager for switching between light and dark themes"""

    DARK = "dark"
    LIGHT = "light"

    _current_theme = DARK

    @classmethod
    def get_current_theme(cls):
        return cls._current_theme

    @classmethod
    def set_theme(cls, theme: str):
        cls._current_theme = theme

    @classmethod
    def get_stylesheet(cls, theme: str = None):
        if theme is None:
            theme = cls._current_theme
        return DARK_THEME if theme == cls.DARK else LIGHT_THEME

    @classmethod
    def toggle_theme(cls):
        if cls._current_theme == cls.DARK:
            cls._current_theme = cls.LIGHT
        else:
            cls._current_theme = cls.DARK
        return cls.get_stylesheet()
