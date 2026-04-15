# -*- coding: utf-8 -*-
"""
Log Extractor - 从终端日志中提取关键信息

用于过滤噪音日志，提取对用户有意义的关键信息
"""

import re
from typing import Optional, Dict, List


class LogExtractor:
    """从终端日志中提取关键信息"""

    # 关键词模式 (pattern, log_type, display_format)
    KEY_PATTERNS = [
        # 工具调用
        (r'调用工具[：:]\s*(\S+)', 'tool', '调用工具: {}'),
        (r'Tool called[：:]\s*(\S+)', 'tool', 'Tool: {}'),
        (r'使用工具[：:]\s*(\S+)', 'tool', '使用工具: {}'),

        # 步骤开始/完成
        (r'开始执行[：:]\s*(.+)', 'start', '开始: {}'),
        (r'完成[：:]\s*(.+)', 'complete', '完成: {}'),
        (r'Starting[：:]\s*(.+)', 'start', 'Starting: {}'),
        (r'Completed[：:]\s*(.+)', 'complete', 'Completed: {}'),

        # 数据处理
        (r'处理\s*(\d+)\s*条', 'count', '处理 {} 条记录'),
        (r'加载数据[：:]\s*(.+)', 'data', '加载数据: {}'),
        (r'Loading data[：:]\s*(.+)', 'data', 'Loading: {}'),
        (r'读取文件[：:]\s*(.+)', 'data', '读取文件: {}'),

        # 结果
        (r'结果[：:]\s*(.+)', 'result', '结果: {}'),
        (r'Result[：:]\s*(.+)', 'result', 'Result: {}'),
        (r'输出[：:]\s*(.+)', 'result', '输出: {}'),

        # 错误
        (r'错误[：:]\s*(.+)', 'error', '错误: {}'),
        (r'Error[：:]\s*(.+)', 'error', 'Error: {}'),
        (r'失败[：:]\s*(.+)', 'error', '失败: {}'),

        # 节点流
        (r'\[STREAM\]\s*节点\s*(\d+)[：:]\s*(\w+)', 'node', '节点 {}: {}'),

        # 任务
        (r'任务[：:]\s*(.+)', 'task', '任务: {}'),
        (r'Task[：:]\s*(.+)', 'task', 'Task: {}'),

        # 进度
        (r'进度[：:]\s*(\d+)%', 'progress', '进度: {}%'),
        (r'Progress[：:]\s*(\d+)%', 'progress', 'Progress: {}%'),

        # 分析相关
        (r'分析方法[：:]\s*(.+)', 'analysis', '分析方法: {}'),
        (r'选择工具[：:]\s*(.+)', 'analysis', '选择工具: {}'),
        (r'执行分析[：:]\s*(.+)', 'analysis', '执行分析: {}'),

        # 模型/SQL
        (r'执行SQL[：:]\s*(.+)', 'sql', 'SQL: {}'),
        (r'SQL结果[：:]\s*(.+)', 'sql', 'SQL结果: {}'),
        (r'模型[：:]\s*(.+)', 'model', '模型: {}'),
    ]

    # 忽略模式（噪音过滤）
    IGNORE_PATTERNS = [
        r'^DEBUG',
        r'^INFO\s+httpx',
        r'^INFO\s+urllib',
        r'^INFO\s+openai',
        r'^INFO\s+anthropic',
        r'^\s*$',
        r'^---+$',
        r'^===+$',
        r'^\[\d{4}-\d{2}-\d{2}',  # 时间戳开头的调试日志
        r'^HTTP Request',
        r'^Retrying',
        r'^Connection',
        r'^Rate limit',
    ]

    def __init__(self, max_logs: int = 15):
        """
        初始化日志提取器

        Args:
            max_logs: 保存的最大关键日志数量
        """
        self.key_logs: List[Dict] = []
        self.max_logs = max_logs
        self._current_node_idx: Optional[int] = None

    def process(self, log_line: str) -> Optional[Dict]:
        """
        处理日志行，返回提取的关键信息

        Args:
            log_line: 原始日志行

        Returns:
            提取的信息字典或None
        """
        if not log_line or not isinstance(log_line, str):
            return None

        # 检查是否应忽略
        for pattern in self.IGNORE_PATTERNS:
            if re.search(pattern, log_line, re.IGNORECASE):
                return None

        # 提取关键信息
        for pattern, log_type, display_format in self.KEY_PATTERNS:
            match = re.search(pattern, log_line)
            if match:
                groups = match.groups()
                if len(groups) == 1:
                    content = groups[0]
                    display = display_format.format(content)
                else:
                    content = groups
                    display = display_format.format(*groups)

                return {
                    'type': log_type,
                    'content': content,
                    'display': display,
                    'raw': log_line.strip()[:200]  # 限制长度
                }

        return None

    def add_log(self, log_info: Optional[Dict]) -> None:
        """
        添加到关键日志列表

        Args:
            log_info: 日志信息字典
        """
        if log_info:
            # 避免重复
            if self.key_logs and self.key_logs[-1].get('display') == log_info.get('display'):
                return

            self.key_logs.append(log_info)
            if len(self.key_logs) > self.max_logs:
                self.key_logs.pop(0)

    def process_and_add(self, log_line: str) -> Optional[Dict]:
        """
        处理日志行并添加到列表

        Args:
            log_line: 原始日志行

        Returns:
            提取的信息字典或None
        """
        log_info = self.process(log_line)
        self.add_log(log_info)
        return log_info

    def get_key_logs(self) -> List[Dict]:
        """获取关键日志列表"""
        return self.key_logs.copy()

    def get_display_logs(self) -> List[str]:
        """获取用于显示的日志文本列表"""
        return [log.get('display', '') for log in self.key_logs]

    def get_logs_by_type(self, log_type: str) -> List[Dict]:
        """获取指定类型的日志"""
        return [log for log in self.key_logs if log.get('type') == log_type]

    def get_current_action(self) -> str:
        """
        获取当前动作描述

        Returns:
            当前动作的描述文本
        """
        if not self.key_logs:
            return "等待开始..."

        # 查找最近的开始/工具/分析日志
        for log in reversed(self.key_logs):
            log_type = log.get('type', '')
            if log_type in ('start', 'tool', 'analysis', 'task'):
                return log.get('display', '处理中...')

        # 默认返回最后一条日志
        return self.key_logs[-1].get('display', '处理中...')

    def clear(self) -> None:
        """清空日志"""
        self.key_logs = []
        self._current_node_idx = None

    def set_current_node(self, node_idx: int) -> None:
        """设置当前节点索引"""
        self._current_node_idx = node_idx

    def get_current_node(self) -> Optional[int]:
        """获取当前节点索引"""
        return self._current_node_idx


class NodeLogManager:
    """
    节点日志管理器 - 为每个节点维护独立的日志提取器
    """

    def __init__(self, num_nodes: int = 10):
        """
        初始化节点日志管理器

        Args:
            num_nodes: 节点数量
        """
        self.num_nodes = num_nodes
        self.node_extractors: Dict[int, LogExtractor] = {}
        self.current_node_idx: int = 0

        # 为每个节点创建独立的提取器
        for i in range(num_nodes):
            self.node_extractors[i] = LogExtractor(max_logs=10)

    def set_current_node(self, node_idx: int) -> None:
        """设置当前活跃节点"""
        if 0 <= node_idx < self.num_nodes:
            self.current_node_idx = node_idx

    def process_log(self, log_line: str, node_idx: Optional[int] = None) -> Optional[Dict]:
        """
        处理日志并添加到对应节点

        Args:
            log_line: 日志行
            node_idx: 节点索引（None则使用当前节点）

        Returns:
            提取的日志信息
        """
        idx = node_idx if node_idx is not None else self.current_node_idx
        if 0 <= idx < self.num_nodes:
            return self.node_extractors[idx].process_and_add(log_line)
        return None

    def get_node_logs(self, node_idx: int) -> List[Dict]:
        """获取指定节点的日志"""
        if 0 <= node_idx < self.num_nodes:
            return self.node_extractors[node_idx].get_key_logs()
        return []

    def get_node_display_logs(self, node_idx: int) -> List[str]:
        """获取指定节点的显示日志"""
        if 0 <= node_idx < self.num_nodes:
            return self.node_extractors[node_idx].get_display_logs()
        return []

    def get_node_current_action(self, node_idx: int) -> str:
        """获取指定节点的当前动作"""
        if 0 <= node_idx < self.num_nodes:
            return self.node_extractors[node_idx].get_current_action()
        return "等待开始..."

    def clear_node(self, node_idx: int) -> None:
        """清空指定节点的日志"""
        if 0 <= node_idx < self.num_nodes:
            self.node_extractors[node_idx].clear()

    def clear_all(self) -> None:
        """清空所有节点的日志"""
        for extractor in self.node_extractors.values():
            extractor.clear()
        self.current_node_idx = 0
