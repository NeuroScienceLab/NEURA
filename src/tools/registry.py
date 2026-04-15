"""
Tool Registry - Unified management of all analysis tools
"""
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


class Modality(Enum):
    """影像模态"""
    ANAT = "anat"      # 解剖学T1加权像
    DWI = "dwi"        # 扩散加权成像
    FUNC = "func"      # 功能磁共振成像
    ALL = "all"        # 所有模态


class ExecutorType(Enum):
    """执行器类型"""
    PYTHON = "python"      # Python函数
    MATLAB = "matlab"      # MATLAB脚本
    CLI = "cli"            # 命令行工具
    NIPYPE = "nipype"      # Nipype工作流


@dataclass
class ToolDefinition:
    """工具定义"""
    name: str
    description: str
    category: str                           # preprocessing/analysis/statistics/visualization
    supported_modalities: List[Modality]
    executor_type: ExecutorType
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    version: str = "1.0.0"
    docker_image: Optional[str] = None
    matlab_path: Optional[str] = None       # MATLAB工具路径
    cli_command: Optional[str] = None       # CLI命令模板
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ToolCallRequest:
    """工具调用请求"""
    tool_name: str
    call_id: str
    inputs: Dict[str, Any]
    params: Dict[str, Any]
    output_dir: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # 上下文信息（cohort, subject_groups等）


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    tool_name: str
    status: str                  # pending/running/succeeded/failed
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0
    outputs: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Dict] = field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None


class BaseTool(ABC):
    """工具基类"""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """返回工具定义"""
        pass

    @abstractmethod
    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行工具"""
        pass

    def validate_inputs(self, inputs: Dict[str, Any], params: Dict[str, Any] = None) -> bool:
        """验证输入参数（同时检查inputs和params）"""
        schema = self.definition.input_schema
        required = schema.get("required", [])

        # 合并inputs和params进行检查
        all_params = {**inputs}
        if params:
            all_params.update(params)

        for field in required:
            if field not in all_params:
                raise ValueError(f"缺少必需参数: {field}")
        return True


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._definitions: Dict[str, ToolDefinition] = {}

    def register(self, tool: BaseTool):
        """注册工具（同时自动注册到知识图谱）"""
        definition = tool.definition
        self._tools[definition.name] = tool
        self._definitions[definition.name] = definition
        # 自动注册到知识图谱，让规划和技能模块能发现新工具
        try:
            from src.knowledge.tool_knowledge_graph import register_tool_to_kg
            register_tool_to_kg(definition)
        except Exception:
            pass  # 知识图谱注册失败不影响工具本身
        # 自动生成 bootstrap skills，让技能匹配模块有初始数据
        try:
            from src.agent.skill_learning.tool_calling_skill import tool_calling_skill
            tool_calling_skill.onboard_new_tool(definition)
        except Exception:
            pass  # skill onboarding 失败不影响工具注册

    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)

    def get_definition(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self._definitions.get(name)

    def list_tools(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())

    def list_by_modality(self, modality: Modality) -> List[str]:
        """按模态列出工具"""
        result = []
        for name, definition in self._definitions.items():
            if Modality.ALL in definition.supported_modalities or \
               modality in definition.supported_modalities:
                result.append(name)
        return result

    def list_by_category(self, category: str) -> List[str]:
        """按类别列出工具"""
        return [
            name for name, d in self._definitions.items()
            if d.category == category
        ]

    def get_tool_descriptions(self) -> List[Dict]:
        """获取所有工具的描述（用于LLM）"""
        descriptions = []
        for name, d in self._definitions.items():
            descriptions.append({
                "name": name,
                "description": d.description,
                "category": d.category,
                "modalities": [m.value for m in d.supported_modalities],
                "executor": d.executor_type.value,
                "inputs": d.input_schema,
                "outputs": d.output_schema
            })
        return descriptions

    def get_function_definitions(self) -> List[Dict]:
        """获取工具的函数定义（用于LLM function calling）"""
        functions = []
        for name, d in self._definitions.items():
            functions.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": d.description,
                    "parameters": d.input_schema
                }
            })
        return functions

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行工具调用"""
        tool = self.get(request.tool_name)
        if not tool:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="failed",
                error=f"工具未找到: {request.tool_name}"
            )

        try:
            tool.validate_inputs(request.inputs, request.params)
            return tool.execute(request)
        except Exception as e:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=request.tool_name,
                status="failed",
                error=str(e)
            )


# 全局注册表实例
_registry = None

def get_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
