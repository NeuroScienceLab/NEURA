# -*- coding: utf-8 -*-
"""
Step Result Loader - 从outputs目录加载步骤结果

用于读取运行时产生的各类数据：
- 步骤输出 (output.json)
- 研究计划 (plan.json)
- 任务列表 (task_list.json)
- 工件文件 (图像、代码、表格等)
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class StepResultLoader:
    """从outputs目录加载步骤结果"""

    # 10个标准步骤ID
    STEP_IDS = [
        "01_parse_question",
        "02_search_knowledge",
        "03_generate_plan",
        "04_map_data_fields",
        "05_build_cohort",
        "06_materialize_data",
        "07_select_tools",
        "08_execute_analysis",
        "09_validate_results",
        "10_generate_report"
    ]

    def __init__(self, run_dir: Path = None):
        """
        初始化加载器

        Args:
            run_dir: 运行目录路径 (outputs/runs/{run_id}/)
        """
        self.run_dir = Path(run_dir) if run_dir else None

    def set_run_dir(self, run_dir: Path):
        """设置运行目录"""
        self.run_dir = Path(run_dir)

    def get_step_output(self, step_id: str) -> Dict[str, Any]:
        """
        获取步骤输出

        Args:
            step_id: 步骤ID (如 "03_generate_plan")

        Returns:
            输出数据字典
        """
        if not self.run_dir:
            return {}

        output_file = self.run_dir / "steps" / step_id / "output.json"
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_plan(self) -> Dict[str, Any]:
        """
        获取研究计划

        Returns:
            计划数据字典，包含pipeline数组
        """
        if not self.run_dir:
            return {}

        plan_file = self.run_dir / "steps" / "03_generate_plan" / "plan.json"
        if plan_file.exists():
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_tasks(self) -> List[Dict[str, Any]]:
        """
        获取任务列表

        Returns:
            任务列表，每个任务包含 task_id, tool_name, description, status, result 等
        """
        if not self.run_dir:
            return []

        task_file = self.run_dir / "task_list.json"
        if task_file.exists():
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get("tasks", [])
            except Exception:
                pass
        return []

    def get_pipeline_tools(self) -> List[Dict[str, str]]:
        """
        获取计划中的工具列表

        Returns:
            工具列表，每个元素包含 tool 和 description
        """
        plan = self.get_plan()
        pipeline = plan.get("pipeline", [])
        tools = []
        for step in pipeline:
            tools.append({
                "tool": step.get("tool", ""),
                "description": step.get("step", step.get("description", ""))
            })
        return tools

    def get_step_artifacts(self, step_id: str) -> List[Dict[str, str]]:
        """
        获取步骤产出的工件（图像、代码等）

        Args:
            step_id: 步骤ID

        Returns:
            工件列表，每个元素包含 path, name, type
        """
        if not self.run_dir:
            return []

        step_dir = self.run_dir / "steps" / step_id
        artifacts = []

        if step_dir.exists():
            for f in step_dir.glob("*"):
                if f.is_file() and f.suffix in ['.png', '.jpg', '.jpeg', '.py', '.json', '.csv']:
                    artifacts.append({
                        'path': str(f),
                        'name': f.name,
                        'type': f.suffix[1:]  # 去掉点号
                    })

        return artifacts

    def get_all_artifacts(self) -> Dict[str, List[Dict[str, str]]]:
        """
        获取所有类型的工件

        Returns:
            按类型分组的工件字典
        """
        if not self.run_dir:
            return {}

        artifacts = {
            'code': [],
            'images': [],
            'data': []
        }

        # 生成的代码
        code_dir = self.run_dir / "generated_code"
        if code_dir.exists():
            for f in code_dir.glob("*.py"):
                artifacts['code'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'py'
                })

        # 分析结果图像
        results_dir = self.run_dir / "analysis_results"
        if results_dir.exists():
            for f in results_dir.glob("*.png"):
                artifacts['images'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'png'
                })
            for f in results_dir.glob("*.jpg"):
                artifacts['images'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'jpg'
                })

        # 工具输出的图像
        tools_dir = self.run_dir / "tools"
        if tools_dir.exists():
            for f in tools_dir.glob("**/*.png"):
                artifacts['images'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'png'
                })

        # 数据文件
        if results_dir.exists():
            for f in results_dir.glob("*.json"):
                artifacts['data'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'json'
                })
            for f in results_dir.glob("*.csv"):
                artifacts['data'].append({
                    'path': str(f),
                    'name': f.name,
                    'type': 'csv'
                })

        return artifacts

    def get_task_result_summary(self, task: Dict[str, Any]) -> str:
        """
        获取任务结果摘要

        Args:
            task: 任务字典

        Returns:
            结果摘要文本
        """
        result = task.get("result", {})
        if not result:
            return ""

        summaries = []

        # 文件数量
        if "output_files" in result:
            count = len(result["output_files"])
            summaries.append(f"生成 {count} 个文件")

        if "conversion_count" in result:
            summaries.append(f"转换 {result['conversion_count']} 个")

        if "subjects_processed" in result:
            summaries.append(f"处理 {result['subjects_processed']} 个被试")

        if "success_count" in result:
            summaries.append(f"成功 {result['success_count']} 个")

        if "error" in result and result["error"]:
            summaries.append(f"错误: {result['error'][:50]}")

        return " | ".join(summaries) if summaries else "已完成"

    def get_step_id_by_index(self, index: int) -> Optional[str]:
        """
        根据索引获取步骤ID

        Args:
            index: 步骤索引 (0-9)

        Returns:
            步骤ID或None
        """
        if 0 <= index < len(self.STEP_IDS):
            return self.STEP_IDS[index]
        return None

    def get_run_meta(self) -> Dict[str, Any]:
        """
        获取运行元数据

        Returns:
            元数据字典
        """
        if not self.run_dir:
            return {}

        meta_file = self.run_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # ============== 迭代相关方法 ==============

    def get_iteration_history(self) -> List[Dict[str, Any]]:
        """
        获取迭代历史记录

        Returns:
            迭代历史列表，每个元素包含:
            - iteration: 迭代次数
            - quality_score: 质量评分
            - feedback: 反馈信息
            - suggestions: 改进建议列表
            - timestamp: 时间戳
        """
        if not self.run_dir:
            return []

        history = []

        # 从迭代评估步骤目录读取
        eval_dir = self.run_dir / "steps" / "11_evaluate_iteration"
        if eval_dir.exists():
            # 尝试读取单个评估文件
            eval_file = eval_dir / "iteration_evaluation.json"
            if eval_file.exists():
                try:
                    with open(eval_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 如果是列表形式（多次迭代）
                        if isinstance(data, list):
                            history.extend(data)
                        else:
                            history.append(data)
                except Exception:
                    pass

            # 也检查output.json
            output_file = eval_dir / "output.json"
            if output_file.exists() and not history:
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if "iteration_history" in data:
                            history.extend(data["iteration_history"])
                        elif "quality_score" in data:
                            history.append(data)
                except Exception:
                    pass

        # 也尝试从agent_state.json读取
        state_file = self.run_dir / "agent_state.json"
        if state_file.exists() and not history:
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    if "iteration_history" in state:
                        history = state["iteration_history"]
            except Exception:
                pass

        return history

    def get_current_iteration(self) -> Tuple[int, int]:
        """
        获取当前迭代次数和最大迭代次数

        Returns:
            (当前迭代次数, 最大迭代次数) 元组
        """
        if not self.run_dir:
            return 0, 5

        # 首先尝试从agent_state.json读取
        state_file = self.run_dir / "agent_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    current = state.get("iteration_count", 0)
                    max_iter = state.get("max_iterations", 5)
                    return current, max_iter
            except Exception:
                pass

        # 从迭代历史推断
        history = self.get_iteration_history()
        if history:
            current = len(history)
            # 检查最后一次是否需要更深分析
            last = history[-1]
            if last.get("needs_deeper", False):
                current += 1  # 表示正在进行下一次迭代
            return current, 5

        return 0, 5

    def get_iteration_evaluation(self, iteration: int = None) -> Dict[str, Any]:
        """
        获取特定迭代的评估结果

        Args:
            iteration: 迭代次数（从1开始），None表示获取最新的

        Returns:
            评估结果字典，包含:
            - iteration: 迭代次数
            - quality_score: 质量评分 (0-10)
            - needs_deeper: 是否需要更深分析
            - feedback: 反馈信息
            - suggestions: 改进建议列表
            - strengths: 优点列表
            - weaknesses: 缺点列表
        """
        history = self.get_iteration_history()

        if not history:
            return {}

        if iteration is None:
            # 返回最新的
            return history[-1] if history else {}

        # 查找指定迭代
        for record in history:
            if record.get("iteration") == iteration:
                return record

        return {}

    def get_quality_score(self) -> float:
        """
        获取当前质量评分

        Returns:
            质量评分 (0-10)，无数据返回0
        """
        eval_result = self.get_iteration_evaluation()
        return eval_result.get("quality_score", 0.0)
