#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURA桌面应用入口
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QCoreApplication
from desk_app.app.main_window import MainWindow


if __name__ == "__main__":
    # 设置Qt插件路径，解决平台插件问题
    import PySide6
    import os
    plugin_path = os.path.join(os.path.dirname(PySide6.__file__), 'plugins')
    if os.path.exists(plugin_path):
        QApplication.setLibraryPaths([plugin_path])
    
    # 设置应用信息
    QCoreApplication.setApplicationName("Neuroimaging Agent")
    QCoreApplication.setApplicationVersion("1.0.0")
    QCoreApplication.setOrganizationName("Encore-Agent")
    QCoreApplication.setOrganizationDomain("encore-agent.com")
    
    # 创建应用实例
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用
    sys.exit(app.exec())
