"""
Run tracking system - runs/steps/artifacts management
"""
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.config import OUTPUT_DIR


class RunStatus(Enum):
    """运行状态"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Artifact:
    """工件定义"""
    name: str
    path: str
    artifact_type: str      # file/directory/figure/table/report
    size_bytes: int = 0
    checksum: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class StepRecord:
    """步骤记录"""
    step_id: str
    step_number: int
    step_name: str
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0
    inputs: Dict = field(default_factory=dict)
    outputs: Dict = field(default_factory=dict)
    artifacts: List[Artifact] = field(default_factory=list)
    logs: str = ""
    error: Optional[str] = None
    llm_calls: int = 0
    tokens_used: Dict = field(default_factory=dict)


@dataclass
class ToolExecutionRecord:
    """工具执行记录"""
    execution_id: str
    tool_name: str
    step_id: str
    status: str
    request: Dict
    response: Dict
    started_at: str
    finished_at: str
    duration_seconds: float
    logs: str = ""


@dataclass
class RunRecord:
    """运行记录"""
    run_id: str
    question: str
    status: RunStatus = RunStatus.QUEUED
    plan: Dict = field(default_factory=dict)
    config: Dict = field(default_factory=dict)
    steps: List[StepRecord] = field(default_factory=list)
    tool_executions: List[ToolExecutionRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    total_duration_seconds: float = 0
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


class RunTracker:
    """运行追踪器"""

    # 10个标准步骤
    STANDARD_STEPS = [
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

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else OUTPUT_DIR / "runs"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_run: Optional[RunRecord] = None
        self.run_dir: Optional[Path] = None

    def create_run(self, run_id: str, question: str, config: Dict = None) -> RunRecord:
        """创建新运行"""
        self.current_run = RunRecord(
            run_id=run_id,
            question=question,
            config=config or {}
        )

        # 创建运行目录结构
        self.run_dir = self.base_dir / run_id
        self._create_run_directories()

        # 初始化步骤
        for i, (step_id, step_name) in enumerate(self.STANDARD_STEPS, 1):
            step = StepRecord(
                step_id=step_id,
                step_number=i,
                step_name=step_name
            )
            self.current_run.steps.append(step)

        # 保存初始状态
        self._save_run_meta()

        return self.current_run

    def _create_run_directories(self):
        """创建运行目录结构"""
        dirs = [
            "steps",
            "tools",
            "data",
            "results/tables",
            "results/figures",
            "reports"
        ]
        for d in dirs:
            (self.run_dir / d).mkdir(parents=True, exist_ok=True)

        # 为每个步骤创建目录
        for step_id, _ in self.STANDARD_STEPS:
            (self.run_dir / "steps" / step_id).mkdir(exist_ok=True)

    def start_run(self):
        """开始运行"""
        if not self.current_run:
            raise ValueError("没有活动的运行")

        self.current_run.status = RunStatus.RUNNING
        self.current_run.started_at = datetime.now().isoformat()
        self._save_run_meta()

    def finish_run(self, success: bool, error: str = None):
        """完成运行"""
        if not self.current_run:
            return

        self.current_run.status = RunStatus.SUCCEEDED if success else RunStatus.FAILED
        self.current_run.finished_at = datetime.now().isoformat()
        self.current_run.error = error

        if self.current_run.started_at:
            start = datetime.fromisoformat(self.current_run.started_at)
            end = datetime.fromisoformat(self.current_run.finished_at)
            self.current_run.total_duration_seconds = (end - start).total_seconds()

        self._save_run_meta()

    def start_step(self, step_id: str, inputs: Dict = None):
        """开始步骤"""
        step = self._get_step(step_id)
        if not step:
            return

        step.status = StepStatus.RUNNING
        step.started_at = datetime.now().isoformat()
        step.inputs = inputs or {}

        # 保存输入
        self._save_step_file(step_id, "input.json", inputs or {})
        self._save_run_meta()

    def finish_step(self, step_id: str, outputs: Dict = None, success: bool = True,
                   error: str = None, logs: str = ""):
        """完成步骤"""
        step = self._get_step(step_id)
        if not step:
            return

        step.status = StepStatus.SUCCEEDED if success else StepStatus.FAILED
        step.finished_at = datetime.now().isoformat()
        step.outputs = outputs or {}
        step.error = error
        step.logs = logs

        if step.started_at:
            start = datetime.fromisoformat(step.started_at)
            end = datetime.fromisoformat(step.finished_at)
            step.duration_seconds = (end - start).total_seconds()

        # 保存输出和日志
        self._save_step_file(step_id, "output.json", outputs or {})
        if logs:
            self._save_step_file(step_id, "log.txt", logs, is_json=False)

        self._save_run_meta()

    def add_step_artifact(self, step_id: str, name: str, content: Any,
                         artifact_type: str = "file", metadata: Dict = None) -> str:
        """添加步骤工件"""
        step = self._get_step(step_id)
        if not step:
            return ""

        # 确定文件路径
        step_dir = self.run_dir / "steps" / step_id
        file_path = step_dir / name

        # 保存内容
        if artifact_type == "figure":
            # 假设content是图像数据或路径
            if isinstance(content, (str, Path)) and Path(content).exists():
                shutil.copy(content, file_path)
            else:
                # 需要matplotlib等保存
                pass
        elif artifact_type == "table":
            if isinstance(content, dict):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                # DataFrame等
                content.to_csv(file_path, index=False)
        else:
            if isinstance(content, (dict, list)):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(content))

        # 计算校验和
        checksum = self._compute_checksum(file_path) if file_path.exists() else ""
        size = file_path.stat().st_size if file_path.exists() else 0

        # 创建工件记录
        artifact = Artifact(
            name=name,
            path=str(file_path),
            artifact_type=artifact_type,
            size_bytes=size,
            checksum=checksum,
            metadata=metadata or {}
        )
        step.artifacts.append(artifact)

        self._save_run_meta()

        return str(file_path)

    def record_tool_execution(self, tool_name: str, step_id: str,
                             request: Dict, response: Dict,
                             started_at: str, finished_at: str,
                             logs: str = "") -> str:
        """记录工具执行"""
        import uuid
        execution_id = str(uuid.uuid4())[:8]

        # 计算耗时
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(finished_at)
        duration = (end - start).total_seconds()

        record = ToolExecutionRecord(
            execution_id=execution_id,
            tool_name=tool_name,
            step_id=step_id,
            status=response.get("status", "unknown"),
            request=request,
            response=response,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration,
            logs=logs
        )

        if self.current_run:
            self.current_run.tool_executions.append(record)

        # 保存到工具目录
        tool_dir = self.run_dir / "tools" / f"{tool_name}_{execution_id}"
        tool_dir.mkdir(parents=True, exist_ok=True)

        with open(tool_dir / "request.json", "w", encoding="utf-8") as f:
            json.dump(request, f, ensure_ascii=False, indent=2)

        with open(tool_dir / "response.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2, default=str)

        if logs:
            with open(tool_dir / "stdout.log", "w", encoding="utf-8") as f:
                f.write(logs)

        self._save_run_meta()

        return execution_id

    def save_plan(self, plan: Dict):
        """保存研究计划"""
        if self.current_run:
            self.current_run.plan = plan

        if self.run_dir:
            with open(self.run_dir / "plan.json", "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)

        self._save_run_meta()

    def save_data_file(self, name: str, content: Any) -> str:
        """保存数据文件"""
        if not self.run_dir:
            return ""

        file_path = self.run_dir / "data" / name

        if isinstance(content, (dict, list)):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        elif hasattr(content, "to_parquet"):
            content.to_parquet(file_path)
        elif hasattr(content, "to_csv"):
            content.to_csv(file_path, index=False)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))

        return str(file_path)

    def save_result(self, name: str, content: Any, result_type: str = "table") -> str:
        """保存结果"""
        if not self.run_dir:
            return ""

        if result_type == "figure":
            file_path = self.run_dir / "results" / "figures" / name
        else:
            file_path = self.run_dir / "results" / "tables" / name

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(content, (dict, list)):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))

        return str(file_path)

    def save_report(self, name: str, content: str) -> str:
        """保存报告"""
        if not self.run_dir:
            return ""

        file_path = self.run_dir / "reports" / name
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        return str(file_path)

    def get_run_summary(self) -> Dict:
        """获取运行摘要"""
        if not self.current_run:
            return {}

        return {
            "run_id": self.current_run.run_id,
            "question": self.current_run.question,
            "status": self.current_run.status.value,
            "total_duration": self.current_run.total_duration_seconds,
            "steps_completed": sum(
                1 for s in self.current_run.steps
                if s.status == StepStatus.SUCCEEDED
            ),
            "steps_total": len(self.current_run.steps),
            "tool_executions": len(self.current_run.tool_executions),
            "error": self.current_run.error
        }

    def _get_step(self, step_id: str) -> Optional[StepRecord]:
        """获取步骤"""
        if not self.current_run:
            return None
        for step in self.current_run.steps:
            if step.step_id == step_id:
                return step
        return None

    def _save_step_file(self, step_id: str, filename: str, content: Any, is_json: bool = True):
        """保存步骤文件"""
        if not self.run_dir:
            return

        file_path = self.run_dir / "steps" / step_id / filename

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if is_json:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=2, default=str)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))

    def _save_run_meta(self):
        """保存运行元信息"""
        if not self.run_dir or not self.current_run:
            return

        meta = {
            "run_id": self.current_run.run_id,
            "question": self.current_run.question,
            "status": self.current_run.status.value,
            "created_at": self.current_run.created_at,
            "started_at": self.current_run.started_at,
            "finished_at": self.current_run.finished_at,
            "total_duration_seconds": self.current_run.total_duration_seconds,
            "config": self.current_run.config,
            "error": self.current_run.error,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "status": s.status.value,
                    "duration_seconds": s.duration_seconds,
                    "artifacts_count": len(s.artifacts)
                }
                for s in self.current_run.steps
            ],
            "tool_executions_count": len(self.current_run.tool_executions)
        }

        with open(self.run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _compute_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_run(self, run_id: str) -> Optional[RunRecord]:
        """加载已有运行"""
        run_dir = self.base_dir / run_id
        meta_path = run_dir / "meta.json"

        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.run_dir = run_dir
        self.current_run = RunRecord(
            run_id=meta["run_id"],
            question=meta["question"],
            status=RunStatus(meta["status"]),
            config=meta.get("config", {}),
            created_at=meta["created_at"],
            started_at=meta.get("started_at"),
            finished_at=meta.get("finished_at"),
            total_duration_seconds=meta.get("total_duration_seconds", 0),
            error=meta.get("error")
        )

        # 加载计划
        plan_path = run_dir / "plan.json"
        if plan_path.exists():
            with open(plan_path, "r", encoding="utf-8") as f:
                self.current_run.plan = json.load(f)

        return self.current_run


# 全局实例
_run_tracker = None

def get_run_tracker() -> RunTracker:
    """获取全局运行追踪器"""
    global _run_tracker
    if _run_tracker is None:
        _run_tracker = RunTracker()
    return _run_tracker


def set_run_tracker(tracker: RunTracker):
    """设置全局运行追踪器（用于测试或恢复运行）"""
    global _run_tracker
    _run_tracker = tracker
