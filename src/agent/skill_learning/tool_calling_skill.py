"""
Tool Calling Skill for Neuroimaging Agent

Design based on the Anthropic Skill quadruplet S = (C, π, T, R):
- C: Applicable conditions — preconditions (inferred from ToolDefinition)
- π: Execution strategy — procedure + parameters + tips (inferred from schema + execution learning)
- T: Termination conditions — postconditions (inferred from output_schema)
- R: Invocation interface — tool + context

Features:
- Knowledge graph-based tool recommendation
- Tool sequence validation and auto-correction
- Disease-specific ROI suggestions
- Tool dependency resolution
- Procedural knowledge inference from ToolDefinition
- Error pattern learning from execution failures
- Experience tips accumulation from successful executions
"""

from typing import Dict, Any, List, Optional
from src.knowledge.tool_knowledge_graph import (
    enhance_plan_with_knowledge_graph,
    get_tools_for_task,
    get_disease_rois,
    validate_tool_sequence,
    resolve_tool_order,
    get_tool_info,
    get_tool_dependencies,
    get_following_tools
)


# ========== 过程性知识推断函数 ==========

def _infer_preconditions(tool_definition) -> List[str]:
    """从 ToolDefinition 推断前置条件 (Skill 四元组的 C)"""
    conds = []
    for dep in (tool_definition.dependencies or []):
        conds.append(f"依赖 {dep} 完成")
    modalities = [m.value for m in tool_definition.supported_modalities]
    if modalities and "all" not in modalities:
        conds.append(f"需要 {'/'.join(modalities)} 模态数据")
    required = tool_definition.input_schema.get("required", [])
    if required:
        conds.append(f"必需参数: {', '.join(required)}")
    return conds


def _infer_procedure(tool_definition, params: Dict) -> List[str]:
    """从 ToolDefinition + 参数生成执行步骤 (Skill 四元组的 π)"""
    steps = []
    props = tool_definition.input_schema.get("properties", {})
    required = tool_definition.input_schema.get("required", [])
    input_fields = [k for k in required if k in props]
    if input_fields:
        steps.append(f"准备输入: {', '.join(input_fields)}")
    param_desc = [f"{k}={v}" for k, v in params.items()]
    if param_desc:
        steps.append(f"配置参数: {', '.join(param_desc)}")
    steps.append(f"执行 {tool_definition.name}")
    out_props = tool_definition.output_schema.get("properties", {})
    if out_props:
        steps.append(f"检查输出: {', '.join(out_props.keys())}")
    return steps


def _infer_postconditions(tool_definition) -> List[str]:
    """从 ToolDefinition 推断后置条件 (Skill 四元组的 T)"""
    conds = []
    out_props = tool_definition.output_schema.get("properties", {})
    if out_props:
        conds.append(f"输出包含: {', '.join(out_props.keys())}")
    conds.append("执行状态为 succeeded")
    return conds


_ERROR_RECOVERY_HINTS: Dict[str, str] = {
    "内存不足": "降低分辨率或减少并行被试数量",
    "文件缺失": "检查前置任务是否完成，确认输入路径正确",
    "权限不足": "检查文件和目录读写权限",
    "执行超时": "增加超时限制或减少处理数据量",
    "参数无效": "检查必需参数是否完整，参数类型是否正确",
    "格式错误": "确认输入文件格式符合工具要求",
    "其他错误": "查看完整错误日志，尝试手动执行该步骤",
}


def _categorize_error(error_message: str) -> str:
    """将错误信息归类为简短标签"""
    msg = error_message.lower()
    if "memory" in msg or "oom" in msg or "内存" in msg:
        return "内存不足"
    if "not found" in msg or "找不到" in msg or "no such file" in msg:
        return "文件缺失"
    if "permission" in msg or "权限" in msg:
        return "权限不足"
    if "timeout" in msg or "超时" in msg:
        return "执行超时"
    if "invalid" in msg or "无效" in msg or "参数" in msg:
        return "参数无效"
    if "format" in msg or "格式" in msg:
        return "格式错误"
    return "其他错误"


def _learn_error_pattern(library, skill_id: str, error_message: str, params: Dict):
    """从执行失败中学习错误模式"""
    skill = library.get_skill(skill_id)
    if not skill:
        return
    error_category = _categorize_error(error_message)
    # 检查是否已有相同模式
    for ep in skill.error_patterns:
        if ep.get("pattern") == error_category:
            ep["count"] = ep.get("count", 1) + 1
            ep.setdefault("trigger_params", []).append(
                {k: str(v) for k, v in (params or {}).items()}
            )
            ep["trigger_params"] = ep["trigger_params"][-5:]
            library._update_procedural_knowledge(skill)
            return
    # 新错误模式
    skill.error_patterns.append({
        "pattern": error_category,
        "raw_error": error_message[:200],
        "recovery": _ERROR_RECOVERY_HINTS.get(error_category, ""),
        "count": 1,
        "trigger_params": [{k: str(v) for k, v in (params or {}).items()}]
    })
    skill.error_patterns = skill.error_patterns[-10:]
    library._update_procedural_knowledge(skill)


def _learn_tips_from_success(library, skill_id: str, params: Dict,
                              disease: str, duration: float):
    """从成功执行中学习经验提示"""
    skill = library.get_skill(skill_id)
    if not skill:
        return
    param_summary = ", ".join(
        f"{k}={v}" for k, v in (params or {}).items()
        if k not in ("input_dir", "output_dir", "subject_list")
    )
    if not param_summary:
        return
    tip_parts = []
    if disease:
        tip_parts.append(f"{disease}研究中")
    tip_parts.append(f"参数 {param_summary} 执行成功")
    if duration > 0:
        tip_parts.append(f"(耗时{duration:.0f}s)")
    tip = " ".join(tip_parts)
    if tip not in skill.tips:
        skill.tips.append(tip)
        skill.tips = skill.tips[-15:]
        library._update_procedural_knowledge(skill)


class ToolCallingSkill:
    """
    Tool calling skill for intelligent tool selection and orchestration.

    This skill encapsulates tool knowledge graph functionality and provides
    a clean interface for agent nodes to use.
    """

    def __init__(self):
        """Initialize the tool calling skill."""
        self.last_enhancement = None
        self.last_validation = None

    def enhance_planning(
        self,
        research_question: str,
        disease_context: Optional[str] = None,
        parsed_intent: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance research planning with knowledge graph recommendations.

        Args:
            research_question: The research question or task description
            disease_context: Optional disease name for ROI recommendations
            parsed_intent: Optional parsed intent dictionary

        Returns:
            Enhancement dictionary with suggested tools, ROI suggestions, etc.
        """
        enhancement = enhance_plan_with_knowledge_graph(
            query=research_question,
            disease=disease_context,
            parsed_intent=parsed_intent
        )

        self.last_enhancement = enhancement
        return enhancement

    def validate_tools(self, tool_names: List[str]) -> Dict[str, Any]:
        """
        Validate a sequence of tools for correctness and dependencies.

        Args:
            tool_names: List of tool names in execution order

        Returns:
            Validation result with 'valid', 'issues', and 'warnings' keys
        """
        validation = validate_tool_sequence(tool_names)
        self.last_validation = validation
        return validation

    def fix_tool_order(self, tool_names: List[str]) -> List[str]:
        """
        Automatically fix tool order based on dependencies.

        Args:
            tool_names: Unordered list of tool names

        Returns:
            Tools sorted by dependency order
        """
        return resolve_tool_order(tool_names)

    def get_tool_details(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get detailed information for a list of tools.

        Args:
            tool_names: List of tool names

        Returns:
            List of tool detail dictionaries
        """
        details = []
        for tool_name in tool_names:
            info = get_tool_info(tool_name)
            if info:
                details.append({
                    "tool": tool_name,
                    "name": info.get("name", tool_name),
                    "category": info.get("category", "unknown"),
                    "modality": info.get("modality", "unknown"),
                    "function": info.get("function", ""),
                    "processing_time": info.get("processing_time", "unknown"),
                    "inputs": info.get("inputs", []),
                    "outputs": info.get("outputs", [])
                })
        return details

    def get_disease_recommendations(self, disease: str) -> Dict[str, Any]:
        """
        Get disease-specific brain region and tool recommendations.

        Args:
            disease: Disease name (supports Chinese, English, abbreviations)

        Returns:
            Dictionary with primary/secondary ROIs, recommended tools, etc.
        """
        return get_disease_rois(disease)

    def get_task_tools(self, task_description: str) -> List[str]:
        """
        Get recommended tools for a specific task description.

        Args:
            task_description: Task description text

        Returns:
            List of recommended tool names
        """
        return get_tools_for_task(task_description)

    def validate_and_fix(self, tool_names: List[str]) -> Dict[str, Any]:
        """
        Validate tools and automatically fix if issues are found.

        Args:
            tool_names: List of tool names

        Returns:
            Dictionary with 'valid', 'fixed', 'original_tools', 'fixed_tools',
            'issues', 'warnings' keys
        """
        validation = self.validate_tools(tool_names)

        result = {
            "valid": validation["valid"],
            "original_tools": tool_names,
            "issues": validation.get("issues", []),
            "warnings": validation.get("warnings", [])
        }

        if not validation["valid"]:
            fixed_tools = self.fix_tool_order(tool_names)
            result["fixed"] = True
            result["fixed_tools"] = fixed_tools
        else:
            result["fixed"] = False
            result["fixed_tools"] = tool_names

        return result

    def recommend_tool_params(self, tool: str, disease: str = None,
                              modality: str = None, task_description: str = "") -> Dict:
        """
        推荐工具参数 + 过程性知识

        返回格式:
        - 无匹配 skill 时: 纯参数字典 {}
        - 有匹配 skill 时: {"parameters": {...}, "tips": [...], "error_patterns": [...],
                            "procedure": [...], "postconditions": [...]}
        """
        try:
            from src.agent.skill_learning.skill_matcher import get_skill_matcher
            matcher = get_skill_matcher()
            params = matcher.recommend_parameters(tool, disease, modality, task_description)
            # 附加过程性知识
            best = matcher.get_best_skill(task_description or tool, disease, modality, tool)
            if best and best.skill and (best.skill.tips or best.skill.error_patterns
                                         or best.skill.procedure):
                return {
                    "parameters": params,
                    "tips": best.skill.tips,
                    "error_patterns": best.skill.error_patterns,
                    "procedure": best.skill.procedure,
                    "postconditions": best.skill.postconditions
                }
            return params
        except Exception:
            return {}

    def onboard_new_tool(self, tool_definition) -> None:
        """
        新工具接入时自动生成 bootstrap skills (含过程性知识)

        从 ToolDefinition 推断完整的 Skill 四元组:
        - C: preconditions (从 dependencies + modalities + required 推断)
        - π: procedure (从 input_schema + output_schema 生成执行步骤)
        - T: postconditions (从 output_schema 推断)
        - R: tool + parameters (从 input_schema 提取)

        Args:
            tool_definition: ToolDefinition 实例
        """
        from src.agent.skill_learning.skill_library import (
            AnalysisSkill, SkillContext, SkillQuality, get_skill_library
        )
        from datetime import datetime

        library = get_skill_library()
        now = datetime.now().isoformat()
        tool_name = tool_definition.name

        # 从 input_schema 提取默认参数
        props = tool_definition.input_schema.get("properties", {})
        default_params = {}
        for key, schema in props.items():
            if "default" in schema:
                default_params[key] = schema["default"]
            elif "enum" in schema and schema["enum"]:
                default_params[key] = schema["enum"][0]

        # 从 supported_modalities 推断
        modalities = [m.value for m in tool_definition.supported_modalities]

        # 推断过程性知识 (所有变体共享的前置/后置条件)
        preconditions = _infer_preconditions(tool_definition)
        postconditions = _infer_postconditions(tool_definition)

        # 找到主要变体 key（第一个有 enum 且选项 > 1 的参数）
        variant_key = None
        for key, schema in props.items():
            if "enum" in schema and len(schema["enum"]) > 1:
                variant_key = key
                break

        skills_created = 0
        if variant_key and "enum" in props[variant_key]:
            for variant in props[variant_key]["enum"]:
                skill_id = f"bootstrap_{tool_name}_{variant}"
                if library.get_skill(skill_id):
                    continue
                params = dict(default_params)
                params[variant_key] = variant
                # 每个变体有独立的执行步骤
                procedure = _infer_procedure(tool_definition, params)
                skill = AnalysisSkill(
                    skill_id=skill_id,
                    tool=tool_name,
                    parameters=params,
                    context=SkillContext(
                        diseases=[], modalities=modalities,
                        analysis_types=[variant]
                    ),
                    quality=SkillQuality(
                        confidence=0.3, success_rate=0.5,
                        effect_sizes=[], validation_count=0
                    ),
                    source="auto_bootstrap",
                    created_at=now, updated_at=now,
                    description=f"{tool_name} {variant}: {tool_definition.description[:100]}",
                    preconditions=preconditions,
                    procedure=procedure,
                    postconditions=postconditions,
                    error_patterns=[],
                    tips=[]
                )
                library.add_skill(skill)
                skills_created += 1
        else:
            skill_id = f"bootstrap_{tool_name}"
            if not library.get_skill(skill_id):
                procedure = _infer_procedure(tool_definition, default_params)
                skill = AnalysisSkill(
                    skill_id=skill_id,
                    tool=tool_name,
                    parameters=default_params,
                    context=SkillContext(
                        diseases=[], modalities=modalities, analysis_types=[]
                    ),
                    quality=SkillQuality(
                        confidence=0.3, success_rate=0.5,
                        effect_sizes=[], validation_count=0
                    ),
                    source="auto_bootstrap",
                    created_at=now, updated_at=now,
                    description=f"{tool_name}: {tool_definition.description[:100]}",
                    preconditions=preconditions,
                    procedure=procedure,
                    postconditions=postconditions,
                    error_patterns=[],
                    tips=[]
                )
                library.add_skill(skill)
                skills_created += 1

        if skills_created > 0:
            print(f"  [Skill Onboarding] {tool_name}: 自动生成 {skills_created} 个 bootstrap skills")

    def record_execution_feedback(self, tool: str, params: Dict, success: bool,
                                  disease: str = None, modality: str = None,
                                  task_description: str = "",
                                  duration_seconds: float = 0,
                                  output_files: list = None,
                                  error_message: str = None):
        """
        执行后反馈 — 更新已有技能质量、自动创建新技能，并累积工具执行档案

        Args:
            tool: 工具名称
            params: 使用的参数
            success: 是否成功
            disease: 疾病名称
            modality: 模态
            task_description: 任务描述
            duration_seconds: 执行时长（秒）
            output_files: 输出文件列表
            error_message: 错误信息
        """
        from src.agent.skill_learning.skill_matcher import get_skill_matcher
        from src.agent.skill_learning.skill_library import (
            AnalysisSkill, SkillContext, SkillQuality, get_skill_library
        )
        from datetime import datetime

        matcher = get_skill_matcher()
        library = get_skill_library()

        # 1. 更新已有技能或创建新技能
        matches = matcher.match_skills(task_description or tool, disease=disease,
                                        modality=modality, tool=tool, top_k=1)
        matched_skill_id = None
        if matches and matches[0].score > 0.7:
            matched_skill_id = matches[0].skill.skill_id
            matcher.update_skill_from_execution(
                skill_id=matched_skill_id,
                success=success,
                quality_score=1.0 if success else 0.0,
                task_description=task_description
            )
        elif success:
            now = datetime.now().isoformat()
            skill = AnalysisSkill(
                skill_id=f"auto_{tool}_{hash(str(params)) % 10000:04d}",
                tool=tool,
                parameters=params,
                context=SkillContext(
                    diseases=[disease] if disease else [],
                    modalities=[modality] if modality else [],
                    analysis_types=[]
                ),
                quality=SkillQuality(confidence=0.5, success_rate=1.0,
                                      effect_sizes=[], validation_count=0),
                source="auto_learned",
                created_at=now,
                updated_at=now,
                description=task_description or f"Auto-learned from {tool} execution"
            )
            library.add_skill(skill)
            matched_skill_id = skill.skill_id

        # 无高分匹配时，回退到 bootstrap skill（确保错误模式能被学习）
        if not matched_skill_id:
            fallback = library.get_skill(f"bootstrap_{tool}")
            if not fallback:
                # 尝试带变体的 bootstrap skill
                candidates = library.search_skills(tool=tool, min_confidence=0.0)
                if candidates:
                    fallback = candidates[0]
            if fallback:
                matched_skill_id = fallback.skill_id

        # 2. 更新工具执行档案（累积经验）
        try:
            library.update_execution_profile(
                tool_name=tool,
                success=success,
                duration=duration_seconds,
                params=params,
                error=error_message,
                output_files=output_files,
                task_description=task_description
            )
        except Exception:
            pass

        # 3. 成功时更新知识图谱的 best_for
        if success and task_description:
            try:
                from src.knowledge.tool_knowledge_graph import update_tool_best_for
                update_tool_best_for(tool, task_description)
            except Exception:
                pass

        # 4. 过程性知识学习：错误模式 + 经验提示
        if matched_skill_id:
            try:
                if not success and error_message:
                    _learn_error_pattern(library, matched_skill_id, error_message, params)
                if success:
                    _learn_tips_from_success(
                        library, matched_skill_id, params,
                        disease or "", duration_seconds
                    )
            except Exception:
                pass

    def get_best_match(self, task_description: str, disease: str = None,
                       modality: str = None, tool: str = None) -> Optional[Dict]:
        """获取最佳匹配技能"""
        try:
            from src.agent.skill_learning.skill_matcher import get_skill_matcher
            matcher = get_skill_matcher()
            match = matcher.get_best_skill(task_description, disease, modality, tool)
            if match:
                return {
                    "skill_id": match.skill.skill_id,
                    "score": match.score,
                    "parameters": match.adapted_parameters,
                    "description": match.skill.description
                }
        except Exception:
            pass
        return None

    def learn_pipeline_skill(self, tool_results: List[Dict],
                              disease: str = "", question: str = "") -> None:
        """
        从成功的 pipeline 执行中学习复合技能

        在 node_evaluate_iteration 评估满意时调用，
        将完整的工具链保存为可复用的 pipeline skill。

        Args:
            tool_results: 工具结果列表
            disease: 相关疾病
            question: 研究问题
        """
        succeeded_tools = [r for r in tool_results
                          if r.get("status") == "succeeded" and r.get("tool")]
        if len(succeeded_tools) < 2:
            return  # 单工具不算 pipeline

        tool_chain = [{"tool": r["tool"],
                       "params": r.get("params", r.get("parameters", {}))}
                      for r in succeeded_tools]

        try:
            from src.agent.skill_learning.skill_library import get_skill_library
            library = get_skill_library()
            ok = library.save_pipeline_skill(
                tool_chain=tool_chain,
                disease=disease,
                question=question
            )
            if ok:
                chain_str = " → ".join(t["tool"] for t in tool_chain)
                print(f"  [Pipeline学习] 已保存: {chain_str}")
        except Exception as e:
            print(f"  [Pipeline学习] 保存失败: {e}")

    def recommend_pipeline(self, disease: str = None, question: str = "",
                           modality: str = None) -> Optional[Dict]:
        """
        推荐匹配的 pipeline 复合技能

        在 node_generate_plan 生成计划时调用，
        如果有高置信度的匹配 pipeline，可以直接作为计划模板。

        Args:
            disease: 疾病名称
            question: 研究问题
            modality: 模态

        Returns:
            最佳匹配的 pipeline，或 None
        """
        try:
            from src.agent.skill_learning.skill_library import get_skill_library
            library = get_skill_library()

            # 从问题中推断 research_type
            research_type = None
            q = (question or "").lower()
            if "vbm" in q or "灰质" in q or "体素形态" in q:
                research_type = "VBM"
            elif "dti" in q or "白质" in q or "纤维" in q:
                research_type = "DTI"
            elif "功能连接" in q or "fmri" in q or "静息态" in q:
                research_type = "功能连接"

            matches = library.search_pipeline_skills(
                disease=disease,
                modality=modality,
                research_type=research_type,
                min_confidence=0.5
            )
            if matches:
                best = matches[0]
                return {
                    "pipeline_id": best["pipeline_id"],
                    "tool_chain": best["tool_chain"],
                    "confidence": best["confidence"],
                    "success_count": best["success_count"],
                    "research_type": best["research_type"],
                    "description": best["description"]
                }
        except Exception:
            pass
        return None

    def print_enhancement(self, enhancement: Dict[str, Any], prefix: str = "  ") -> None:
        """
        Print enhancement results in a formatted way.

        Args:
            enhancement: Enhancement dictionary from enhance_planning()
            prefix: Prefix for each line (for indentation)
        """
        if enhancement.get("suggested_tools"):
            tools = enhancement["suggested_tools"]
            print(f"{prefix}[知识图谱] 推荐工具: {', '.join(tools[:3])}")
            if len(tools) > 3:
                print(f"{prefix}            (共 {len(tools)} 个)")

        roi_suggestions = enhancement.get("roi_suggestions", {})
        if roi_suggestions.get("primary"):
            primary = roi_suggestions["primary"]
            print(f"{prefix}[知识图谱] 推荐ROI: {', '.join(primary[:3])}")
            if len(primary) > 3:
                print(f"{prefix}            (共 {len(primary)} 个)")

        confidence = enhancement.get("kg_confidence", 0.0)
        print(f"{prefix}[知识图谱] 置信度: {confidence:.2f}")

    def print_validation(self, validation: Dict[str, Any], prefix: str = "  ") -> None:
        """
        Print validation results in a formatted way.

        Args:
            validation: Validation dictionary from validate_tools()
            prefix: Prefix for each line (for indentation)
        """
        if not validation.get("valid"):
            print(f"{prefix}[知识图谱] 工具序列验证发现问题:")
            for issue in validation.get("issues", []):
                print(f"{prefix}  - 错误: {issue}")

        if validation.get("warnings"):
            print(f"{prefix}[知识图谱] 工具序列警告:")
            for warning in validation["warnings"]:
                print(f"{prefix}  - 警告: {warning}")


# Create a global instance for easy access
tool_calling_skill = ToolCallingSkill()
