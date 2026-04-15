"""
Task Manager - Manages the list of tasks in the analysis process
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"       # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败
    SKIPPED = "skipped"       # 跳过
    BLOCKED = "blocked"       # 因依赖失败而阻塞


@dataclass
class Task:
    """单个分析任务"""
    task_id: str
    tool_name: str
    description: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING

    # 模态信息（用于多模态数据选择）
    modality: Optional[str] = None  # "anat", "dwi", "func", None(自动推断)

    # 执行结果
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # 依赖关系
    depends_on: List[str] = field(default_factory=list)  # 依赖的任务ID列表

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "description": self.description,
            "inputs": self.inputs,
            "params": self.params,
            "modality": self.modality,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建任务"""
        task = cls(
            task_id=data["task_id"],
            tool_name=data["tool_name"],
            description=data["description"],
            inputs=data.get("inputs", {}),
            params=data.get("params", {}),
            status=TaskStatus(data.get("status", "pending")),
            modality=data.get("modality"),  # 模态信息
            result=data.get("result"),
            error=data.get("error"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            depends_on=data.get("depends_on", [])
        )
        return task


class TaskManager:
    """任务列表管理器"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.tasks_file = self.run_dir / "task_list.json"
        self.tasks: List[Task] = []
        self.metadata: Dict[str, Any] = {}

        # 加载已有任务
        if self.tasks_file.exists():
            self.load()

    def create_from_plan(self, plan: Dict[str, Any], iteration_count: int = 0,
                         previous_task_results: Dict[str, Dict[str, Any]] = None) -> None:
        """
        Create a task list from the plan, supporting the inheritance of successful results from the previous iteration

        Args:
            plan: 研究计划
            iteration_count: 当前迭代次数（0表示首次执行）
            previous_task_results: 前一迭代成功完成的任务结果 {description: result_dict}
        """
        self.tasks = []

        # 提取元数据
        self.metadata = {
            "research_question": plan.get("research_question", ""),
            "analysis_strategy": plan.get("analysis_strategy", ""),
            "iteration_count": iteration_count,
            "created_at": datetime.now().isoformat()
        }

        # 从pipeline创建任务
        steps = plan.get("pipeline", [])
        task_counter = 0

        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")

            # 跳过可视化工具 - 可视化应该在Vibe Coding中完成
            if tool_name == "data_visualization":
                print(f"  [SKIP] 跳过可视化任务 - 将在Vibe Coding模块中完成")
                continue

            task_counter += 1
            # 如果是迭代（iteration_count > 0），任务ID包含迭代次数
            if iteration_count > 0:
                task_id = f"{iteration_count}_task_{task_counter:02d}"
            else:
                task_id = f"task_{task_counter:02d}"

            # 解析依赖关系（如果某个step的inputs依赖前一个step的outputs）
            depends_on = []
            if task_counter > 1:
                # 简单依赖：每个任务依赖前一个任务
                # 依赖ID格式需要与任务ID格式一致（包含迭代次数前缀）
                if iteration_count > 0:
                    depends_on = [f"{iteration_count}_task_{task_counter-1:02d}"]
                else:
                    depends_on = [f"task_{task_counter-1:02d}"]

            task = Task(
                task_id=task_id,
                tool_name=tool_name,
                description=step.get("step", step.get("description", "")),  # 使用"step"字段作为描述
                inputs=step.get("inputs", {}),
                params=step.get("parameters", step.get("params", {})),  # 使用"parameters"或"params"
                modality=step.get("modality"),  # 从计划中读取模态（anat/dwi/func）
                depends_on=depends_on
            )
            self.tasks.append(task)

        # === 继承前一迭代的成功结果 ===
        if previous_task_results and iteration_count > 0:
            inherited_count = 0
            for task in self.tasks:
                # 尝试匹配任务描述
                for prev_desc, prev_result in previous_task_results.items():
                    if self._match_task_description(task.description, prev_desc):
                        # 检查前一迭代的任务是否成功
                        if prev_result.get("status") == "completed" or prev_result.get("success"):
                            task.status = TaskStatus.SKIPPED
                            task.result = prev_result
                            task.completed_at = datetime.now().isoformat()
                            inherited_count += 1
                            print(f"  [继承] {task.task_id}: 复用前一迭代成功结果 (匹配: {prev_desc[:50]}...)")
                            break
            if inherited_count > 0:
                print(f"  [继承统计] 共 {inherited_count} 个任务从前一迭代继承成功结果")

        self.save()

    def get_next_task(self) -> Optional[Task]:
        """获取下一个待执行的任务（所有依赖都已完成）

        注意：SKIPPED任务只有在有可复用输出时才视为依赖满足。
        这防止了前置任务失败被SKIP后，后续任务错误地认为依赖已满足。
        """
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue

            # 检查依赖是否都已满足
            all_deps_satisfied = True
            for dep_id in task.depends_on:
                dep_task = self.get_task(dep_id)
                if not dep_task:
                    all_deps_satisfied = False
                    break

                if dep_task.status == TaskStatus.COMPLETED:
                    continue  # 完成的任务满足依赖
                elif dep_task.status == TaskStatus.SKIPPED:
                    # SKIPPED任务只有在有可复用输出时才视为依赖满足
                    dep_result = dep_task.result or {}
                    if dep_result.get("reused") and dep_result.get("output_files"):
                        continue  # 有复用数据，依赖满足
                    # 无可复用数据的SKIPPED任务不满足依赖
                    all_deps_satisfied = False
                    break
                else:
                    # 其他状态（PENDING, IN_PROGRESS, FAILED, BLOCKED）不满足依赖
                    all_deps_satisfied = False
                    break

            if all_deps_satisfied:
                return task

        return None

    def get_task(self, task_id: str) -> Optional[Task]:
        """根据ID获取任务"""
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def update_task_status(self, task_id: str, status: TaskStatus,
                          result: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None) -> None:
        """更新任务状态"""
        task = self.get_task(task_id)
        if not task:
            return

        task.status = status

        if status == TaskStatus.IN_PROGRESS:
            task.started_at = datetime.now().isoformat()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED, TaskStatus.BLOCKED]:
            task.completed_at = datetime.now().isoformat()
            task.result = result
            task.error = error

        self.save()

    def get_progress(self) -> Dict[str, Any]:
        """获取进度统计"""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        in_progress = sum(1 for t in self.tasks if t.status == TaskStatus.IN_PROGRESS)
        pending = sum(1 for t in self.tasks if t.status == TaskStatus.PENDING)
        blocked = sum(1 for t in self.tasks if t.status == TaskStatus.BLOCKED)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "blocked": blocked,
            "progress_pct": (completed / total * 100) if total > 0 else 0
        }

    def is_complete(self) -> bool:
        """检查所有任务是否完成（包括因依赖失败而阻塞的任务）"""
        return all(t.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED, TaskStatus.FAILED, TaskStatus.BLOCKED]
                  for t in self.tasks)

    def has_failures(self) -> bool:
        """检查是否有失败的任务"""
        return any(t.status == TaskStatus.FAILED for t in self.tasks)

    def mark_blocked_tasks(self) -> int:
        """
        标记因依赖失败而无法执行的任务为BLOCKED状态

        Returns:
            被标记为BLOCKED的任务数量
        """
        blocked_count = 0

        # 获取所有失败任务的ID集合
        failed_task_ids = {t.task_id for t in self.tasks if t.status == TaskStatus.FAILED}

        # 递归查找受影响的任务（包括间接依赖）
        blocked_task_ids = set()

        def find_blocked_tasks(failed_ids: set) -> set:
            """递归查找所有被阻塞的任务"""
            new_blocked = set()
            for task in self.tasks:
                if task.status == TaskStatus.PENDING:
                    # 检查是否有任何依赖任务失败或被阻塞
                    for dep_id in task.depends_on:
                        if dep_id in failed_ids or dep_id in blocked_task_ids:
                            new_blocked.add(task.task_id)
                            break
            return new_blocked

        # 迭代查找直到没有新的阻塞任务
        while True:
            new_blocked = find_blocked_tasks(failed_task_ids)
            if not new_blocked - blocked_task_ids:
                break
            blocked_task_ids.update(new_blocked)

        # 标记所有被阻塞的任务
        for task in self.tasks:
            if task.task_id in blocked_task_ids and task.status == TaskStatus.PENDING:
                # 找出具体是哪个依赖失败了
                failed_deps = [dep_id for dep_id in task.depends_on
                              if dep_id in failed_task_ids or dep_id in blocked_task_ids]
                error_msg = f"依赖任务失败或被阻塞: {', '.join(failed_deps)}"

                self.update_task_status(
                    task.task_id,
                    TaskStatus.BLOCKED,
                    error=error_msg
                )
                blocked_count += 1
                print(f"  [BLOCKED] {task.task_id}: {error_msg}")

        return blocked_count

    def _match_task_description(self, desc1: str, desc2: str) -> bool:
        """
        模糊匹配任务描述，用于继承前一迭代的任务结果

        Args:
            desc1: 当前任务描述
            desc2: 前一迭代任务描述

        Returns:
            是否匹配
        """
        # 标准化：转小写，移除多余空格
        d1 = desc1.lower().strip()
        d2 = desc2.lower().strip()

        # 完全匹配
        if d1 == d2:
            return True

        # 关键词匹配：提取关键词并检查重叠
        # 移除常见停用词
        stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'with', 'in', 'on', 'by', 'from'}
        words1 = set(d1.split()) - stop_words
        words2 = set(d2.split()) - stop_words

        if not words1 or not words2:
            return False

        # 计算重叠比例
        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))

        # 至少50%的关键词匹配，且至少3个词匹配
        return overlap >= 3 and overlap >= min_words * 0.5

    def save(self) -> None:
        """保存任务列表到文件"""
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "tasks": [t.to_dict() for t in self.tasks]
        }

        with open(self.tasks_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """从文件加载任务列表"""
        with open(self.tasks_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        self.tasks = [Task.from_dict(t) for t in data.get("tasks", [])]

    def get_summary(self) -> str:
        """获取任务列表摘要（用于显示）"""
        progress = self.get_progress()
        lines = [
            f"任务进度: {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)",
            ""
        ]

        for task in self.tasks:
            status_icon = {
                TaskStatus.COMPLETED: "[OK]",
                TaskStatus.FAILED: "[X]",
                TaskStatus.IN_PROGRESS: "[>]",
                TaskStatus.PENDING: "[ ]",
                TaskStatus.SKIPPED: "[-]",
                TaskStatus.BLOCKED: "[!]"
            }.get(task.status, "[?]")

            lines.append(f"{status_icon} [{task.task_id}] {task.description} ({task.tool_name})")
            if task.error:
                lines.append(f"    错误: {task.error}")

        return "\n".join(lines)
