"""
Neuroimaging Tool Knowledge Graph (NeuroImaging Tool Knowledge Graph)

Refer to the design of SciToolKG in the SciToolEval paper

Knowledge sources referenced:
- NIH/PubMed neuroimaging literature (2024-2025)
- Official documentation of FreeSurfer, FSL, SPM, DPABI
- ENIGMA Consortium standardized processes
- Braak staging and neurodegenerative disease staging research

"""
from typing import List, Dict, Any, Optional, Set
import re


# =============================================================================
# 动态工具存储（补充硬编码的 TOOL_KNOWLEDGE_GRAPH）
# =============================================================================

_dynamic_tools: Dict[str, Dict[str, Any]] = {}


def register_tool_to_kg(tool_definition) -> None:
    """
    从 ToolDefinition 自动生成知识图谱条目

    当 ToolRegistry.register() 调用时自动触发，
    让新接入的工具也能被知识图谱的查询函数发现。

    Args:
        tool_definition: ToolDefinition 实例
    """
    name = tool_definition.name

    # 如果静态知识图谱已有该工具，跳过
    if name in TOOL_KNOWLEDGE_GRAPH.get("tools", {}):
        return

    # 从 supported_modalities 推断 modality 字符串
    modality = _map_modality_from_definition(tool_definition.supported_modalities)

    # 从 input_schema 提取输入名称
    inputs = _extract_input_names_from_schema(tool_definition.input_schema)

    # 从 output_schema 提取输出名称
    outputs = _extract_output_names_from_schema(tool_definition.output_schema)

    # 从 input_schema 提取默认参数
    typical_params = _extract_defaults_from_schema(tool_definition.input_schema)

    entry = {
        "name": name,
        "category": tool_definition.category or "unknown",
        "modality": modality,
        "function": tool_definition.description or "",
        "inputs": inputs,
        "outputs": outputs,
        "depends_on": list(tool_definition.dependencies) if tool_definition.dependencies else [],
        "followed_by": [],
        "best_for": [],       # 初始为空，通过执行学习逐步填充
        "not_for": [],
        "typical_params": typical_params,
        "confidence": 0.5,    # 新工具初始置信度低于手动标注的工具
        "auto_registered": True
    }
    _dynamic_tools[name] = entry


def update_tool_best_for(tool_name: str, task_description: str) -> None:
    """
    从成功的执行中更新工具的 best_for 列表

    Args:
        tool_name: 工具名称
        task_description: 成功完成的任务描述
    """
    # 在动态工具中查找
    entry = _dynamic_tools.get(tool_name)
    if entry is None:
        # 静态工具也允许追加 best_for（写入动态副本）
        static = TOOL_KNOWLEDGE_GRAPH.get("tools", {}).get(tool_name)
        if static:
            entry = dict(static)  # 浅拷贝到动态存储
            _dynamic_tools[tool_name] = entry
        else:
            return

    best_for = entry.setdefault("best_for", [])
    # 截取前80字符作为标签，避免过长
    label = task_description.strip()[:80]
    if label and label not in best_for:
        best_for.append(label)
        # 保持列表不超过20个
        if len(best_for) > 20:
            entry["best_for"] = best_for[-20:]


def _map_modality_from_definition(supported_modalities) -> str:
    """从 Modality 枚举列表推断主要 modality 字符串"""
    if not supported_modalities:
        return "all"
    values = [m.value for m in supported_modalities]
    if "all" in values or len(values) > 2:
        return "all"
    return values[0]


def _extract_input_names_from_schema(input_schema: Dict) -> List[str]:
    """从 JSON Schema 提取输入参数名称"""
    props = input_schema.get("properties", {})
    return [f"{k} ({v.get('type', 'any')})" for k, v in props.items()
            if k in input_schema.get("required", list(props.keys())[:5])]


def _extract_output_names_from_schema(output_schema: Dict) -> List[str]:
    """从 JSON Schema 提取输出名称"""
    props = output_schema.get("properties", {})
    return [k for k in props.keys()]


def _extract_defaults_from_schema(input_schema: Dict) -> Dict[str, Any]:
    """从 JSON Schema 提取默认参数值"""
    defaults = {}
    props = input_schema.get("properties", {})
    for key, schema in props.items():
        if "default" in schema:
            defaults[key] = schema["default"]
        elif "enum" in schema and schema["enum"]:
            defaults[key] = schema["enum"][0]
    return defaults


# =============================================================================
# 工具知识图谱定义
# =============================================================================

TOOL_KNOWLEDGE_GRAPH = {
    # =========================================================================
    # 工具节点定义：As the knowledge graph involves the intellectual property of 
    # multiple teams and several teachers in the laboratory, it has not been made
    #  public; users can build it themselves.
    # =========================================================================
   
   
}


# =============================================================================
# 知识图谱查询函数
# =============================================================================

def get_tool_info(tool_name: str) -> Optional[Dict[str, Any]]:
    """获取工具详细信息（同时查询静态知识图谱和动态注册的工具）"""
    # 优先查静态（手动标注的高质量条目）
    tool = TOOL_KNOWLEDGE_GRAPH["tools"].get(tool_name)
    if tool:
        return tool
    # 再查动态注册的工具
    return _dynamic_tools.get(tool_name)


def get_tools_for_task(task_description: str) -> List[str]:
    """
    根据任务描述匹配合适的工具

    Args:
        task_description: 任务描述文本

    Returns:
        推荐的工具列表
    """
    task_mapping = TOOL_KNOWLEDGE_GRAPH["task_tool_mapping"]
    matched_tools = set()

    task_lower = task_description.lower()

    for keyword, tools in task_mapping.items():
        if keyword.lower() in task_lower or keyword in task_description:
            matched_tools.update(tools)

    return list(matched_tools)


def get_disease_rois(disease: str) -> Dict[str, Any]:
    """
    获取疾病相关的脑区建议

    Args:
        disease: 疾病名称

    Returns:
        脑区建议字典
    """
    disease_mapping = TOOL_KNOWLEDGE_GRAPH["disease_roi_mapping"]

    # 直接匹配
    if disease in disease_mapping:
        return disease_mapping[disease]

    # 模糊匹配
    disease_lower = disease.lower()
    for key, value in disease_mapping.items():
        if key.lower() in disease_lower or disease_lower in key.lower():
            return value

    # 未找到，返回空建议
    return {
        "primary": [],
        "secondary": [],
        "evidence": "无特定疾病-脑区映射",
        "typical_findings": "",
        "recommended_tools": [],
        "recommended_metrics": []
    }


def get_tool_dependencies(tool_name: str) -> List[str]:
    """获取工具的依赖"""
    tool_info = get_tool_info(tool_name)
    if tool_info:
        return tool_info.get("depends_on", [])
    return []


def get_following_tools(tool_name: str) -> List[str]:
    """获取可以跟在该工具后面的工具"""
    tool_info = get_tool_info(tool_name)
    if tool_info:
        return tool_info.get("followed_by", [])
    return []


def resolve_tool_order(tools: List[str]) -> List[str]:
    """
    根据依赖关系解析工具顺序

    Args:
        tools: 工具列表

    Returns:
        排序后的工具列表

    Raises:
        ValueError: 当检测到循环依赖时
    """
    import warnings

    # 简单的拓扑排序
    ordered = []
    remaining = set(tools)
    max_iterations = len(tools) * len(tools)  # 防止无限循环
    iteration = 0

    while remaining:
        iteration += 1
        if iteration > max_iterations:
            # 检测到循环依赖
            cycle_tools = list(remaining)
            # 尝试找出具体的循环
            cycle_path = _detect_cycle(cycle_tools)
            if cycle_path:
                warnings.warn(
                    f"检测到工具循环依赖: {' -> '.join(cycle_path)}. "
                    f"将按原始顺序添加剩余工具: {cycle_tools}",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"检测到工具循环依赖，涉及工具: {cycle_tools}. "
                    f"将按原始顺序添加剩余工具。",
                    UserWarning
                )
            ordered.extend(remaining)
            break

        # 找到没有未满足依赖的工具
        found = False
        for tool in list(remaining):
            deps = get_tool_dependencies(tool)
            if all(d not in remaining or d in ordered for d in deps):
                ordered.append(tool)
                remaining.remove(tool)
                found = True
                break

        if not found:
            # 没有找到可以添加的工具，说明有循环依赖
            cycle_tools = list(remaining)
            cycle_path = _detect_cycle(cycle_tools)
            if cycle_path:
                warnings.warn(
                    f"检测到工具循环依赖: {' -> '.join(cycle_path)}. "
                    f"将按原始顺序添加剩余工具: {cycle_tools}",
                    UserWarning
                )
            else:
                warnings.warn(
                    f"检测到工具循环依赖，涉及工具: {cycle_tools}. "
                    f"将按原始顺序添加剩余工具。",
                    UserWarning
                )
            ordered.extend(remaining)
            break

    return ordered


def _detect_cycle(tools: List[str]) -> Optional[List[str]]:
    """
    检测工具依赖中的循环

    Args:
        tools: 工具列表

    Returns:
        循环路径列表，如果没有循环返回None
    """
    tool_set = set(tools)

    def dfs(tool: str, path: List[str], visited: Set[str]) -> Optional[List[str]]:
        if tool in path:
            # 找到循环
            cycle_start = path.index(tool)
            return path[cycle_start:] + [tool]

        if tool in visited or tool not in tool_set:
            return None

        visited.add(tool)
        path.append(tool)

        deps = get_tool_dependencies(tool)
        for dep in deps:
            if dep in tool_set:
                result = dfs(dep, path, visited)
                if result:
                    return result

        path.pop()
        return None

    visited: Set[str] = set()
    for tool in tools:
        if tool not in visited:
            result = dfs(tool, [], visited)
            if result:
                return result

    return None


def is_tool_suitable(tool_name: str, task: str) -> bool:
    """
    检查工具是否适合特定任务

    Args:
        tool_name: 工具名称
        task: 任务描述

    Returns:
        是否适合
    """
    tool_info = get_tool_info(tool_name)
    if not tool_info:
        return False

    # 检查是否在"not_for"列表中
    not_for = tool_info.get("not_for", [])
    for excluded in not_for:
        if excluded.lower() in task.lower():
            return False

    # 检查是否在"best_for"列表中
    best_for = tool_info.get("best_for", [])
    for suitable in best_for:
        if suitable.lower() in task.lower():
            return True

    return True  # 默认允许


def enhance_plan_with_knowledge_graph(
    query: str,
    disease: str = None,
    parsed_intent: dict = None
) -> Dict[str, Any]:
    """
    使用知识图谱增强规划 - 整合动态知识图谱

    Args:
        query: 研究问题
        disease: 疾病名称（可选）
        parsed_intent: 解析后的意图（可选）

    Returns:
        增强信息字典
    """
    # 1. 根据任务描述匹配工具
    suggested_tools = get_tools_for_task(query)

    # 2. 如果有疾病信息，获取脑区建议（使用动态知识图谱）
    roi_suggestions = {}
    if disease:
        try:
            # 尝试使用动态知识图谱
            from src.knowledge.dynamic_knowledge_graph import get_dynamic_kg
            dynamic_kg = get_dynamic_kg()
            roi_suggestions = dynamic_kg.get_disease_rois(disease)

            # 合并推荐工具，并考虑动态置信度
            recommended = roi_suggestions.get("recommended_tools", [])
            for tool in recommended:
                if tool not in suggested_tools:
                    # 检查工具对该疾病的有效性
                    tool_confidence = dynamic_kg.get_tool_confidence(tool, disease)
                    if tool_confidence > 0.6:  # 只推荐置信度较高的工具
                        suggested_tools.append(tool)
        except Exception as e:
            # 降级到静态知识图谱
            print(f"  [KG] 动态知识图谱不可用，使用静态版本: {e}")
            roi_suggestions = get_disease_rois(disease)
            recommended = roi_suggestions.get("recommended_tools", [])
            for tool in recommended:
                if tool not in suggested_tools:
                    suggested_tools.append(tool)

    # 3. 解析工具顺序
    ordered_tools = resolve_tool_order(suggested_tools)

    # 4. 获取每个工具的详细信息（包含动态置信度）
    tool_details = []
    for tool in ordered_tools:
        info = get_tool_info(tool)
        if info:
            tool_detail = {
                "tool": tool,
                "name": info["name"],
                "function": info["function"],
                "best_for": info["best_for"],
                "typical_params": info.get("typical_params", {})
            }

            # 添加动态置信度
            if disease:
                try:
                    from src.knowledge.dynamic_knowledge_graph import get_dynamic_kg
                    dynamic_kg = get_dynamic_kg()
                    tool_detail["dynamic_confidence"] = dynamic_kg.get_tool_confidence(tool, disease)
                except:
                    tool_detail["dynamic_confidence"] = info.get("confidence", 0.5)

            tool_details.append(tool_detail)

    # 5. 计算置信度
    confidence = len(suggested_tools) / 3.0  # 简单的置信度估计
    confidence = min(1.0, confidence)

    return {
        "suggested_tools": ordered_tools,
        "tool_details": tool_details,
        "roi_suggestions": roi_suggestions,
        "kg_confidence": confidence,
        "source": "dynamic_knowledge_graph" if disease else "static_knowledge_graph"
    }


def get_tool_equivalents(tool_name: str) -> List[str]:
    """
    获取功能等价的工具

    Args:
        tool_name: 工具名称

    Returns:
        等价工具列表
    """
    equivalences = TOOL_KNOWLEDGE_GRAPH["tool_equivalences"]

    for group_name, tools in equivalences.items():
        if tool_name in tools:
            return [t for t in tools if t != tool_name]

    return []


# =============================================================================
# 知识图谱验证函数
# =============================================================================

def validate_tool_sequence(tools: List[str]) -> Dict[str, Any]:
    """
    验证工具序列的合理性

    Args:
        tools: 工具序列

    Returns:
        验证结果
    """
    issues = []
    warnings = []

    for i, tool in enumerate(tools):
        tool_info = get_tool_info(tool)
        if not tool_info:
            warnings.append(f"未知工具: {tool}")
            continue

        # 检查依赖
        deps = tool_info.get("depends_on", [])
        for dep in deps:
            if dep not in tools[:i]:
                issues.append(f"工具 {tool} 的依赖 {dep} 未在之前出现")

        # 检查顺序
        followed_by = tool_info.get("followed_by", [])
        if i < len(tools) - 1:
            next_tool = tools[i + 1]
            if followed_by and next_tool not in followed_by:
                warnings.append(f"工具 {tool} 通常不接 {next_tool}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings
    }


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 测试知识图谱功能
    print("=" * 60)
    print("神经影像工具知识图谱测试")
    print("=" * 60)

    # 测试任务-工具匹配
    test_queries = [
        "比较阿尔茨海默病患者与健康对照的海马体积差异",
        "分析帕金森病患者的白质纤维束完整性",
        "研究抑郁症患者的静息态功能连接",
        "使用VBM分析脊髓小脑共济失调患者的灰质萎缩"
    ]

    for query in test_queries:
        print(f"\n查询: {query}")
        tools = get_tools_for_task(query)
        print(f"推荐工具: {tools}")

    # 测试疾病-脑区映射
    print("\n" + "=" * 60)
    print("疾病-脑区映射测试")
    print("=" * 60)

    test_diseases = ["阿尔茨海默病", "SCA3", "抑郁症"]
    for disease in test_diseases:
        print(f"\n疾病: {disease}")
        rois = get_disease_rois(disease)
        print(f"主要脑区: {rois.get('primary', [])}")
        print(f"次要脑区: {rois.get('secondary', [])}")

    # 测试增强规划
    print("\n" + "=" * 60)
    print("增强规划测试")
    print("=" * 60)

    enhancement = enhance_plan_with_knowledge_graph(
        query="比较SCA3患者与健康对照的小脑和脑干体积差异",
        disease="SCA3"
    )
    print(f"建议工具: {enhancement['suggested_tools']}")
    print(f"脑区建议: {enhancement['roi_suggestions'].get('primary', [])}")
    print(f"置信度: {enhancement['kg_confidence']:.2f}")
