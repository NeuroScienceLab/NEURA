"""
NEURA 
"""
import uuid
from typing import Literal, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from src.agent.graph_state import AgentState, ResearchPhase, create_initial_state
from src.agent.graph_nodes import (
    node_init,
    node_parse_question,
    node_search_knowledge,
    node_generate_plan,
    node_plan_review_gate,
    node_map_data_fields,
    node_build_cohort,
    node_materialize_data,
    node_quality_control,
    node_select_tools,
    node_execute_tool,
    node_validate_results,
    node_generate_report,
    node_evaluate_iteration,
    node_reflect_and_fix,
    node_execute_next_task,
    node_check_tasks_complete,
    node_generate_algorithm_code,
    node_end,
    node_human_review
)


# ============== 条件路由函数 ==============

def route_after_parse(state: AgentState) -> Literal["search_knowledge", "reflect_and_fix"]:
    """解析后的路由"""
    if state.get("last_error") or not state.get("parsed_intent"):
        return "reflect_and_fix"
    return "search_knowledge"


def route_after_plan(state: AgentState) -> Literal["plan_review_gate", "reflect_and_fix"]:
    """计划生成后的路由 — 成功则进入审查门控，失败则反思修复"""
    if state.get("last_error") or not state.get("plan"):
        return "reflect_and_fix"
    return "plan_review_gate"


def route_after_plan_review(state: AgentState) -> Literal["map_data_fields", "reflect_and_fix"]:
    """计划审查门控后的路由 — 用户选择继续则进入数据准备，否则反思修复"""
    if state.get("last_error") or not state.get("plan"):
        return "reflect_and_fix"
    return "map_data_fields"


def route_after_tool_selection(state: AgentState) -> Literal["execute_tool", "validate_results"]:
    """工具选择后的路由"""
    tool_chain = state.get("tool_chain", [])
    if tool_chain:
        return "execute_tool"
    return "validate_results"


def route_after_tool_execution(state: AgentState) -> Literal["execute_tool", "validate_results"]:
    """工具执行后的路由 - 实现循环"""
    tool_chain = state.get("tool_chain", [])
    current_index = state.get("current_tool_index", 0)

    # 如果还有工具需要执行，继续循环
    if current_index < len(tool_chain):
        return "execute_tool"

    # 所有工具执行完毕
    return "validate_results"


def route_after_validation(state: AgentState) -> Literal["generate_report", "reflect_and_fix"]:
    """验证后的路由"""
    validation = state.get("validation", {})
    if validation.get("overall_decision") == "rejected":
        return "reflect_and_fix"

    # stat_review 直接拒绝也触发修复
    stat_review = state.get("stat_review", {})
    if stat_review.get("status") == "rejected":
        return "reflect_and_fix"

    return "generate_report"


def route_after_reflect(state: AgentState) -> Literal["parse_question", "generate_plan", "select_tools", "execute_next_task", "quality_control", "end"]:
    """反思后的路由"""
    error_count = state.get("error_count", 0)
    max_retries = state.get("max_retries", 3)

    if error_count >= max_retries:
        return "end"

    # 根据当前阶段决定重试哪个节点
    phase = state.get("phase", "")

    if phase == ResearchPhase.PLANNING.value:
        # 检查是在哪一步失败的
        if not state.get("parsed_intent"):
            return "parse_question"
        return "generate_plan"

    if phase == ResearchPhase.ANALYSIS.value:
        # task_list 模式路由到 execute_next_task，tool_chain 模式路由到 select_tools
        if state.get("has_task_list", False):
            return "execute_next_task"
        return "select_tools"

    if phase == ResearchPhase.DATA_PREPARATION.value:
        # QC 失败时重新回到 quality_control（数据可能已修复）
        if state.get("qc_results") and not state.get("qc_passed", False):
            return "quality_control"
        return "generate_plan"  # 重新规划数据准备策略

    return "end"


def route_after_check_tasks(state: AgentState) -> Literal["execute_next_task", "generate_algorithm_code"]:
    """检查任务完成后的路由"""
    tasks_complete = state.get("tasks_complete", False)

    if tasks_complete:
        print(f"\n[ROUTING] tasks_complete={tasks_complete} → 路由到: generate_algorithm_code")
        return "generate_algorithm_code"
    else:
        print(f"\n[ROUTING] tasks_complete={tasks_complete} → 路由到: execute_next_task")
        return "execute_next_task"


def route_after_iteration(state: AgentState) -> Literal["generate_plan", "end"]:
    """
    迭代评估后的路由
    - 如果需要更深入分析，返回 generate_plan 重新规划研究方案
    - 否则结束
    """
    needs_deeper = state.get("needs_deeper_analysis", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    # 如果需要深化且未超过最大迭代次数
    if needs_deeper and iteration_count < max_iterations:
        return "generate_plan"

    return "end"


def route_after_materialize(state: AgentState) -> Literal["quality_control"]:
    """数据物化后的路由 - 固定进入质控"""
    return "quality_control"


def route_after_qc(state: AgentState) -> Literal["select_tools", "execute_next_task", "reflect_and_fix"]:
    """QC 后的路由"""
    if not state.get("qc_passed", False):
        return "reflect_and_fix"
    # 检查是否使用任务列表模式（通过 state 标志位判断，避免文件系统 I/O）
    if state.get("has_task_list", False):
        return "execute_next_task"
    return "select_tools"


# ============== 构建图 ==============

def build_research_graph() -> StateGraph:
    """
    构建研究Agent的LangGraph图

    图结构:

    START
      │
      ▼
    ┌─────────────┐
    │    init     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐      失败     ┌─────────────┐
    │parse_question│─────────────->│reflect_and_fix│
    └──────┬──────┘               └──────┬──────┘
           │成功                         │
           ▼                             │重试
    ┌─────────────┐                      │
    │search_knowledge│←──────────────────┘
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐      失败     ┌─────────────┐
    │generate_plan│──────────────->│reflect_and_fix│
    └──────┬──────┘               └─────────────┘
           │成功
           ▼
    ┌─────────────┐
    │map_data_fields│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ build_cohort │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │materialize_data│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ select_tools │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │execute_tool  │←────────┐
    └──────┬──────┘          │
           │                 │还有工具
           │完成             │
           ▼                 │
    ┌─────────────┐──────────┘
    │(检查是否完成)│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │validate_results│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │generate_report│
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │evaluate_iteration│ 迭代评估
    └──────┬──────┘
           │
           ├─────────────┐
           │             │
    需要深化│             │质量满意
           │             │
           ▼             ▼
    ┌─────────────┐  ┌─────────────┐
    │generate_plan │  │     end     │
    └──────┬──────┘  └──────┬──────┘
           │                │
           │ (重新规划，     ▼
           │  迭代循环)     END
           └───────────────>
           (返回到计划阶段，
            根据反馈调整方案)
    """

    # 创建图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("init", node_init)
    workflow.add_node("parse_question", node_parse_question)
    workflow.add_node("search_knowledge", node_search_knowledge)
    workflow.add_node("generate_plan", node_generate_plan)
    workflow.add_node("plan_review_gate", node_plan_review_gate)
    workflow.add_node("map_data_fields", node_map_data_fields)
    workflow.add_node("build_cohort", node_build_cohort)
    workflow.add_node("materialize_data", node_materialize_data)
    workflow.add_node("quality_control", node_quality_control)
    workflow.add_node("select_tools", node_select_tools)
    workflow.add_node("execute_tool", node_execute_tool)
    workflow.add_node("execute_next_task", node_execute_next_task)  # 新增：任务列表执行节点
    workflow.add_node("check_tasks_complete", node_check_tasks_complete)  # 新增：任务完成检查节点
    workflow.add_node("generate_algorithm_code", node_generate_algorithm_code)  # 新增：Vibe Coding代码生成节点
    workflow.add_node("validate_results", node_validate_results)
    workflow.add_node("generate_report", node_generate_report)
    workflow.add_node("evaluate_iteration", node_evaluate_iteration)
    workflow.add_node("human_review", node_human_review)
    workflow.add_node("reflect_and_fix", node_reflect_and_fix)
    workflow.add_node("end", node_end)

    # 设置入口
    workflow.set_entry_point("init")

    # 添加边

    # init -> parse_question
    workflow.add_edge("init", "parse_question")

    # parse_question -> search_knowledge 或 reflect_and_fix
    workflow.add_conditional_edges(
        "parse_question",
        route_after_parse,
        {
            "search_knowledge": "search_knowledge",
            "reflect_and_fix": "reflect_and_fix"
        }
    )

    # search_knowledge -> generate_plan
    workflow.add_edge("search_knowledge", "generate_plan")

    # generate_plan -> plan_review_gate 或 reflect_and_fix
    workflow.add_conditional_edges(
        "generate_plan",
        route_after_plan,
        {
            "plan_review_gate": "plan_review_gate",
            "reflect_and_fix": "reflect_and_fix"
        }
    )

    # plan_review_gate -> map_data_fields 或 reflect_and_fix
    workflow.add_conditional_edges(
        "plan_review_gate",
        route_after_plan_review,
        {
            "map_data_fields": "map_data_fields",
            "reflect_and_fix": "reflect_and_fix"
        }
    )

    # 数据准备流程
    workflow.add_edge("map_data_fields", "build_cohort")
    workflow.add_edge("build_cohort", "materialize_data")

    # materialize_data -> quality_control (固定边)
    workflow.add_edge("materialize_data", "quality_control")

    # quality_control -> select_tools / execute_next_task / reflect_and_fix
    workflow.add_conditional_edges(
        "quality_control",
        route_after_qc,
        {
            "select_tools": "select_tools",
            "execute_next_task": "execute_next_task",
            "reflect_and_fix": "reflect_and_fix"
        }
    )

    # select_tools -> execute_tool 或 validate_results
    workflow.add_conditional_edges(
        "select_tools",
        route_after_tool_selection,
        {
            "execute_tool": "execute_tool",
            "validate_results": "validate_results"
        }
    )

    # execute_tool -> execute_tool (循环) 或 validate_results
    workflow.add_conditional_edges(
        "execute_tool",
        route_after_tool_execution,
        {
            "execute_tool": "execute_tool",
            "validate_results": "validate_results"
        }
    )

    # 新增：任务列表执行流程
    # execute_next_task -> check_tasks_complete
    workflow.add_edge("execute_next_task", "check_tasks_complete")

    # check_tasks_complete -> execute_next_task (循环) 或 generate_algorithm_code
    workflow.add_conditional_edges(
        "check_tasks_complete",
        route_after_check_tasks,
        {
            "execute_next_task": "execute_next_task",
            "generate_algorithm_code": "generate_algorithm_code"
        }
    )

    # generate_algorithm_code -> validate_results (Vibe Coding生成代码后进入验证)
    workflow.add_edge("generate_algorithm_code", "validate_results")

    # validate_results -> generate_report 或 reflect_and_fix
    workflow.add_conditional_edges(
        "validate_results",
        route_after_validation,
        {
            "generate_report": "generate_report",
            "reflect_and_fix": "reflect_and_fix"
        }
    )

    # reflect_and_fix -> 重试节点 或 end
    workflow.add_conditional_edges(
        "reflect_and_fix",
        route_after_reflect,
        {
            "parse_question": "parse_question",
            "generate_plan": "generate_plan",
            "select_tools": "select_tools",
            "execute_next_task": "execute_next_task",
            "quality_control": "quality_control",
            "end": "end"
        }
    )

    # generate_report -> evaluate_iteration
    workflow.add_edge("generate_report", "evaluate_iteration")

    # evaluate_iteration -> human_review (暂停等待用户决策)
    workflow.add_edge("evaluate_iteration", "human_review")

    # human_review -> generate_plan (重新规划) 或 end
    workflow.add_conditional_edges(
        "human_review",
        route_after_iteration,
        {
            "generate_plan": "generate_plan",
            "end": "end"
        }
    )

    # end -> END
    workflow.add_edge("end", END)

    return workflow


def create_research_agent(checkpointer=None):
    """
    创建研究Agent

    Args:
        checkpointer: 可选的检查点保存器，用于状态持久化

    Returns:
        编译后的图
    """
    workflow = build_research_graph()

    if checkpointer is None:
        # 使用内存检查点
        checkpointer = MemorySaver()

    # 编译图，配置递归限制
    app = workflow.compile(
        checkpointer=checkpointer,
        debug=False
    )

    return app


class ResearchAgentLangGraph:
    """LangGraph 研究Agent封装类"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.app = create_research_agent(self.checkpointer)

    def run(self, question: str, thread_id: str = None, auto_continue: bool = False) -> dict:
        """
        运行Agent

        Args:
            question: 研究问题
            thread_id: 线程ID，用于状态持久化
            auto_continue: 自动模式，跳过所有人工审查暂停

        Returns:
            最终状态
        """
        run_id = str(uuid.uuid4())[:8]
        thread_id = thread_id or run_id

        # 创建初始状态
        initial_state = create_initial_state(run_id, question)

        print(f"\n{'='*60}")
        print(f"LangGraph 研究Agent - 运行 {run_id}")
        print(f"{'='*60}")
        print(f"研究问题: {question}\n")

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 150
        }

        # 流式执行，处理中断
        interrupted = self._stream_until_interrupt(initial_state, config)
        while interrupted is not None:
            self._display_interrupt(interrupted)
            if auto_continue:
                user_resp = {"action": "continue"}
            else:
                user_resp = self._get_user_response(interrupted)
            if user_resp.get("action") == "abort":
                print("\n[中止] 用户中止了执行")
                break
            from langgraph.types import Command
            interrupted = self._stream_until_interrupt(Command(resume=user_resp), config)

        result = self._get_final_state(config)

        print(f"\n{'='*60}")
        print(f"运行完成")
        print(f"{'='*60}")
        print(f"状态: {result.get('phase', 'unknown')}")
        print(f"节点历史: {len(result.get('node_history', []))} 个节点")

        if result.get("report_path"):
            print(f"报告: {result['report_path']}")

        return result

    def _stream_until_interrupt(self, input_, config) -> Optional[dict]:
        """流式执行直到中断或结束，返回中断数据或 None"""
        interrupt_data = None
        node_count = 0
        try:
            for chunk in self.app.stream(input_, config, stream_mode="updates"):
                node_count += 1
                if "__interrupt__" in chunk:
                    interrupts = chunk["__interrupt__"]
                    if interrupts:
                        interrupt_data = interrupts[0].value if hasattr(interrupts[0], 'value') else interrupts[0]
                elif chunk:
                    node_name = list(chunk.keys())[0]
                    print(f"[STREAM] 节点 {node_count}: {node_name}")
        except Exception as e:
            print(f"[STREAM ERROR] 流式执行异常: {e}")
            if interrupt_data is None:
                import traceback
                traceback.print_exc()
        return interrupt_data

    def _display_interrupt(self, data: dict):
        """展示中断信息"""
        interrupt_type = data.get("type", "iteration")

        if interrupt_type == "plan_review":
            print(f"\n{'='*60}")
            print(f"[MoER 计划审查] 评分: {data.get('score', 0)}/100")
            issues = data.get("issues", [])
            if issues:
                print(f"发现 {len(issues)} 个问题:")
                for i, issue in enumerate(issues, 1):
                    severity = issue.get("severity", "info")
                    print(f"  [{severity}] {issue.get('message', '')}")
            suggestions = data.get("suggestions", [])
            if suggestions:
                print("改进建议:")
                for s in suggestions:
                    print(f"  - {s}")
            print(f"{'='*60}")
            return

        print(f"\n{'='*60}")
        print(f"[暂停] 第 {data.get('iteration', 0)} 轮迭代完成")
        print(f"质量分: {data.get('quality_score', 0):.1f}/10")
        if data.get('feedback'):
            print(f"评估: {data['feedback']}")
        if data.get('suggestions'):
            print("建议:")
            for s in data['suggestions']:
                print(f"  - {s}")
        if data.get('needs_human_intervention'):
            print(f"\n[需要人工介入] 任务: {data.get('intervention_task')}")
            print(f"错误: {data.get('intervention_error')}")
            if data.get('intervention_suggestions'):
                print("处理建议:")
                for s in data['intervention_suggestions']:
                    print(f"  - {s}")
        print(f"{'='*60}")

    def _get_user_response(self, data: dict) -> dict:
        """获取用户响应"""
        interrupt_type = data.get("type", "iteration")

        if interrupt_type == "plan_review":
            print(f"\n选项: [c]忽略问题继续执行  [f]修复计划  [a]中止")
            print(f"(默认: 继续，直接回车确认)")
            try:
                choice = input("你的选择: ").strip().lower() or "c"
            except (EOFError, KeyboardInterrupt):
                return {"action": "continue"}
            if choice == 'a':
                return {"action": "abort"}
            if choice == 'f':
                return {"action": "fix"}
            return {"action": "continue"}

        needs_deeper = data.get("needs_deeper_analysis", False)
        default = "c" if needs_deeper else "s"
        print(f"\n选项: [c]继续深化分析  [s]停止并生成报告  [f]提供反馈后继续  [a]中止")
        print(f"(默认: {'继续' if needs_deeper else '停止'}，直接回车确认)")
        try:
            choice = input("你的选择: ").strip().lower() or default
        except (EOFError, KeyboardInterrupt):
            return {"action": "stop"}
        if choice == 'a':
            return {"action": "abort"}
        if choice == 's':
            return {"action": "stop"}
        if choice == 'f':
            try:
                feedback = input("请输入反馈意见: ").strip()
            except (EOFError, KeyboardInterrupt):
                feedback = ""
            return {"action": "continue", "feedback": feedback}
        return {"action": "continue"}

    def _get_final_state(self, config: dict) -> dict:
        """获取最终状态"""
        state = self.app.get_state(config)
        return state.values if state.values else {}

    def resume(self, thread_id: str) -> dict:
        """
        恢复之前的运行

        Args:
            thread_id: 线程ID

        Returns:
            恢复后的状态
        """
        config = {"configurable": {"thread_id": thread_id}}

        # 获取当前状态
        state = self.app.get_state(config)

        if state.values:
            print(f"恢复运行: {thread_id}")
            print(f"当前阶段: {state.values.get('phase')}")

            # 继续执行
            for s in self.app.stream(None, config):
                state = s

        return state

    def get_graph_image(self) -> str:
        """获取图的可视化（需要graphviz）"""
        try:
            return self.app.get_graph().draw_mermaid()
        except:
            return "无法生成图可视化"


def create_agent() -> ResearchAgentLangGraph:
    """创建Agent实例"""
    return ResearchAgentLangGraph()
