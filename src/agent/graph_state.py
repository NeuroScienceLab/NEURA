"""
NEURA State Definition
"""
from typing import Dict, List, Any, Optional, Annotated, TypedDict
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import operator


class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NEEDS_RETRY = "needs_retry"
    NEEDS_HUMAN = "needs_human"


class ResearchPhase(Enum):
    """研究阶段"""
    INIT = "init"
    PLANNING = "planning"
    DATA_PREPARATION = "data_preparation"
    ANALYSIS = "analysis"
    REPORTING = "reporting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentState(TypedDict, total=False):
    """
  
    使用 TypedDict 定义状态结构，支持 LangGraph 的状态管理
    """
    # 基本信息
    run_id: str
    question: str
    phase: str  # ResearchPhase value

    # 解析结果
    parsed_intent: Dict[str, Any]

    # 知识库检索
    evidence: str
    citations: List[Dict]
    local_evidence: List[Dict]  # 本地PDF文献证据
    methodology_notes: List[str]

    # 脑区智能选择（基于疾病-脑区映射）
    brain_region_suggestions: Dict[str, Any]  # 包含primary_rois, secondary_rois, literature_support等

    # 研究计划
    plan: Dict[str, Any]
    plan_approved: bool

    # 数据相关
    field_mapping: Dict[str, Any]
    cohort: Dict[str, Any]
    data_manifest: Dict[str, Any]

    # 工具相关
    tool_selection: Dict[str, Any]
    tool_chain: List[Dict]
    current_tool_index: int

    # 任务列表执行模式 (新增)
    tasks_complete: bool  # 所有任务是否完成
    has_task_failures: bool  # 是否有任务失败

    # 执行结果
    tool_results: Annotated[List[Dict], operator.add]  # 使用 add 操作累积结果
    tool_results_iteration_offset: int  # 当前迭代的 tool_results 起始偏移
    analysis_results: Dict[str, Any]

    # Vibe Coding - 生成的算法代码
    generated_codes: Annotated[List[Dict], operator.add]  # 生成的代码记录列表

    # 验证
    validation: Dict[str, Any]
    validation_passed: bool

    # 报告
    report: str
    report_path: str

    # 迭代深化机制
    iteration_count: int
    max_iterations: int
    iteration_history: Annotated[List[Dict], operator.add]
    scientific_quality_score: float
    needs_deeper_analysis: bool
    iteration_feedback: str
    iteration_suggestions: List[str]

    # 人工介入
    needs_human_intervention: bool
    intervention_task: str
    intervention_error: str
    intervention_suggestions: List[str]

    # 迭代继承
    previous_successful_results: Dict[str, Any]

    # 任务列表模式标志
    has_task_list: bool  # 是否已创建任务列表（用于路由判断）
    task_retry_counts: Dict[str, int]  # 任务重试计数 {task_id: count}

    # 数据质控
    qc_results: Dict[str, Any]
    qc_passed: bool

    # MoER 审查结果
    plan_review: Dict[str, Any]      # PlanReviewer 审查结果
    stat_review: Dict[str, Any]      # StatReviewer 审查结果
    moer_reviews: Annotated[List[Dict], operator.add]  # 所有审查记录（累积）

    # 错误处理
    error_count: int
    max_retries: int
    last_error: str
    error_history: Annotated[List[str], operator.add]

    # 消息历史
    messages: Annotated[List[Dict], operator.add]

    # 元信息
    created_at: str
    updated_at: str
    node_history: Annotated[List[str], operator.add]  # 记录节点执行历史


def create_initial_state(run_id: str, question: str) -> AgentState:
    """创建初始状态"""
    now = datetime.now().isoformat()
    return AgentState(
        run_id=run_id,
        question=question,
        phase=ResearchPhase.INIT.value,

        parsed_intent={},
        evidence="",
        citations=[],
        local_evidence=[],
        methodology_notes=[],
        brain_region_suggestions={},

        plan={},
        plan_approved=False,

        field_mapping={},
        cohort={},
        data_manifest={},

        tool_selection={},
        tool_chain=[],
        current_tool_index=0,

        tasks_complete=False,
        has_task_failures=False,

        tool_results=[],
        tool_results_iteration_offset=0,
        analysis_results={},

        generated_codes=[],

        validation={},
        validation_passed=False,

        report="",
        report_path="",

        iteration_count=0,
        max_iterations=5,
        iteration_history=[],
        scientific_quality_score=0.0,
        needs_deeper_analysis=False,
        iteration_feedback="",
        iteration_suggestions=[],

        needs_human_intervention=False,
        intervention_task="",
        intervention_error="",
        intervention_suggestions=[],

        previous_successful_results={},

        qc_results={},
        qc_passed=False,

        has_task_list=False,
        task_retry_counts={},

        plan_review={},
        stat_review={},
        moer_reviews=[],

        error_count=0,
        max_retries=3,
        last_error="",
        error_history=[],

        messages=[],
        created_at=now,
        updated_at=now,
        node_history=[]
    )


def update_state_timestamp(state: AgentState) -> Dict[str, Any]:
    """
    更新状态时间戳 - 返回更新字典
    """
    return {"updated_at": datetime.now().isoformat()}


def merge_state_updates(*updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个状态更新字典

    用于节点函数中组合多个更新，然后一次性返回。
   

    Args:
        *updates: 多个更新字典

    Returns:
        合并后的更新字典
    """
    result = {}
    for update in updates:
        if update:
            result.update(update)
    # 自动添加时间戳
    result["updated_at"] = datetime.now().isoformat()
    return result
