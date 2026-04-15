"""
Intelligent error handling module
Supports strategies such as automatic retry, model upgrade, and human intervention
"""
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"  # 轻微错误，自动重试
    MEDIUM = "medium"  # 中等错误，模型升级
    HIGH = "high"  # 严重错误，需要人工介入
    CRITICAL = "critical"  # 致命错误，停止执行


class ErrorCategory(Enum):
    """错误类别"""
    PARAMETER_MISSING = "parameter_missing"  # 参数缺失
    FILE_NOT_FOUND = "file_not_found"  # 文件未找到
    TOOL_EXECUTION_FAILED = "tool_execution_failed"  # 工具执行失败
    DATA_FORMAT_ERROR = "data_format_error"  # 数据格式错误
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # 资源不可用
    PERMISSION_DENIED = "permission_denied"  # 权限拒绝
    TIMEOUT = "timeout"  # 超时
    UNKNOWN = "unknown"  # 未知错误


class RecoveryStrategy(Enum):
    """恢复策略"""
    RETRY = "retry"  # 重试
    RETRY_WITH_UPGRADE = "retry_with_upgrade"  # 使用更高级模型重试
    SKIP_TASK = "skip_task"  # 跳过任务
    USE_ALTERNATIVE = "use_alternative"  # 使用备选方案
    HUMAN_INTERVENTION = "human_intervention"  # 人工介入
    ABORT = "abort"  # 中止执行


@dataclass
class ErrorAnalysis:
    """错误分析结果"""
    category: ErrorCategory
    severity: ErrorSeverity
    suggested_strategy: RecoveryStrategy
    error_message: str
    context: Dict[str, Any]
    retry_count: int
    can_auto_fix: bool
    fix_suggestions: List[str]


class ErrorHandler:
    """智能错误处理器"""

    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.max_retries = 3
        self.model_upgrade_threshold = 2  # 重试2次后升级模型

    def analyze_error(self, error: str, context: Dict[str, Any]) -> ErrorAnalysis:
        """
        分析错误并确定处理策略

        Args:
            error: 错误信息
            context: 错误上下文（包含task_id, tool_name等）
        """
        error_lower = (error or "").lower()

        # 分类错误
        category = self._classify_error(error_lower)
        severity = self._assess_severity(category, error_lower)
        retry_count = context.get("retry_count", 0)

        # 确定恢复策略
        strategy = self._determine_strategy(category, severity, retry_count, error_lower)

        # 生成修复建议
        suggestions = self._generate_fix_suggestions(category, error, context)

        # 判断是否可以自动修复
        can_auto_fix = strategy in [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.RETRY_WITH_UPGRADE,
            RecoveryStrategy.USE_ALTERNATIVE
        ]

        return ErrorAnalysis(
            category=category,
            severity=severity,
            suggested_strategy=strategy,
            error_message=error,
            context=context,
            retry_count=retry_count,
            can_auto_fix=can_auto_fix,
            fix_suggestions=suggestions
        )

    def _classify_error(self, error: str) -> ErrorCategory:
        """错误分类"""
        if "parameter" in error or "required" in error or "missing" in error:
            return ErrorCategory.PARAMETER_MISSING
        elif "file not found" in error or "no such file" in error:
            return ErrorCategory.FILE_NOT_FOUND
        elif "未能转换任何" in error or "no files converted" in error or "未找到" in error:
            # 特殊情况: 数据已经是目标格式，无需转换
            return ErrorCategory.RESOURCE_UNAVAILABLE
        elif "failed" in error or "error" in error:
            return ErrorCategory.TOOL_EXECUTION_FAILED
        elif "format" in error or "invalid" in error:
            return ErrorCategory.DATA_FORMAT_ERROR
        elif "timeout" in error:
            return ErrorCategory.TIMEOUT
        elif "permission" in error or "denied" in error:
            return ErrorCategory.PERMISSION_DENIED
        else:
            return ErrorCategory.UNKNOWN

    def _assess_severity(self, category: ErrorCategory, error: str) -> ErrorSeverity:
        """评估错误严重程度"""
        # 参数缺失通常是中等错误
        if category == ErrorCategory.PARAMETER_MISSING:
            return ErrorSeverity.MEDIUM

        # 文件未找到是中等错误
        if category == ErrorCategory.FILE_NOT_FOUND:
            return ErrorSeverity.MEDIUM

        # 权限错误是严重错误
        if category == ErrorCategory.PERMISSION_DENIED:
            return ErrorSeverity.HIGH

        # 超时是中等错误
        if category == ErrorCategory.TIMEOUT:
            return ErrorSeverity.MEDIUM

        # 工具执行失败根据具体情况判断
        if category == ErrorCategory.TOOL_EXECUTION_FAILED:
            if "critical" in error or "fatal" in error:
                return ErrorSeverity.HIGH
            return ErrorSeverity.MEDIUM

        # 未知错误默认中等
        return ErrorSeverity.MEDIUM

    def _determine_strategy(self, category: ErrorCategory, severity: ErrorSeverity,
                           retry_count: int, error_message: str = "") -> RecoveryStrategy:
        """确定恢复策略"""

        # 超过最大重试次数
        if retry_count >= self.max_retries:
            if severity == ErrorSeverity.CRITICAL:
                return RecoveryStrategy.ABORT
            elif severity == ErrorSeverity.HIGH:
                return RecoveryStrategy.HUMAN_INTERVENTION
            else:
                return RecoveryStrategy.SKIP_TASK

        # 资源不可用 (如数据已经是目标格式) - 直接跳过
        if category == ErrorCategory.RESOURCE_UNAVAILABLE:
            return RecoveryStrategy.SKIP_TASK

        # 参数缺失 - 尝试自动修复或人工介入
        if category == ErrorCategory.PARAMETER_MISSING:
            if retry_count < 2:
                return RecoveryStrategy.RETRY_WITH_UPGRADE
            return RecoveryStrategy.HUMAN_INTERVENTION

        # 文件未找到 - 检查路径或跳过
        if category == ErrorCategory.FILE_NOT_FOUND:
            # 检查是否是TBSS pipeline的前置依赖问题
            # TBSS任务依赖是强依赖，跳过会导致整个pipeline失败
            if "tbss" in error_message:
                return RecoveryStrategy.HUMAN_INTERVENTION
            if retry_count == 0:
                return RecoveryStrategy.RETRY
            return RecoveryStrategy.SKIP_TASK

        # 工具执行失败 - 重试或升级模型
        if category == ErrorCategory.TOOL_EXECUTION_FAILED:
            if retry_count >= self.model_upgrade_threshold:
                return RecoveryStrategy.RETRY_WITH_UPGRADE
            return RecoveryStrategy.RETRY

        # 超时 - 重试
        if category == ErrorCategory.TIMEOUT:
            return RecoveryStrategy.RETRY

        # 权限错误 - 人工介入
        if category == ErrorCategory.PERMISSION_DENIED:
            return RecoveryStrategy.HUMAN_INTERVENTION

        # 默认重试
        if retry_count < 2:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.HUMAN_INTERVENTION

    def _generate_fix_suggestions(self, category: ErrorCategory, error: str,
                                  context: Dict[str, Any]) -> List[str]:
        """生成修复建议"""
        suggestions = []

        if category == ErrorCategory.PARAMETER_MISSING:
            suggestions.append("检查任务输入参数是否完整")
            suggestions.append("从state中动态填充缺失的参数")
            suggestions.append("查看前置任务是否成功完成")

        elif category == ErrorCategory.FILE_NOT_FOUND:
            suggestions.append("验证文件路径是否正确")
            suggestions.append("检查cohort数据是否已构建")
            suggestions.append("确认前置任务是否生成了预期文件")

        elif category == ErrorCategory.RESOURCE_UNAVAILABLE:
            suggestions.append("数据可能已经是目标格式，此步骤可跳过")
            suggestions.append("检查是否存在预处理的数据文件")
            suggestions.append("直接使用现有数据进行下一步分析")

        elif category == ErrorCategory.TOOL_EXECUTION_FAILED:
            tool_name = context.get("tool_name", "")
            suggestions.append(f"检查{tool_name}工具的配置和依赖")
            suggestions.append("查看工具执行日志获取详细错误")
            suggestions.append("尝试使用备选工具或方法")

        elif category == ErrorCategory.TIMEOUT:
            suggestions.append("增加超时时间限制")
            suggestions.append("减少处理数据量")
            suggestions.append("检查系统资源是否充足")

        elif category == ErrorCategory.PERMISSION_DENIED:
            suggestions.append("检查文件和目录权限")
            suggestions.append("确认运行用户有足够权限")

        else:
            suggestions.append("查看完整错误日志")
            suggestions.append("尝试手动执行该步骤")

        return suggestions

    def log_error(self, analysis: ErrorAnalysis, task_id: str):
        """记录错误"""
        self.error_history.append({
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "category": analysis.category.value,
            "severity": analysis.severity.value,
            "strategy": analysis.suggested_strategy.value,
            "error": analysis.error_message,
            "retry_count": analysis.retry_count
        })

    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误汇总"""
        if not self.error_history:
            return {"total_errors": 0}

        categories = {}
        severities = {}
        for error in self.error_history:
            cat = error["category"]
            sev = error["severity"]
            categories[cat] = categories.get(cat, 0) + 1
            severities[sev] = severities.get(sev, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": categories,
            "by_severity": severities,
            "recent_errors": self.error_history[-5:]  # 最近5个错误
        }


# 全局错误处理器实例
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
