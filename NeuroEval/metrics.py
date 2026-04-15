"""
评估指标计算

包含Tool Planning Accuracy的计算方法：
- 旧方法（Jaccard）：仅保留用于参考
- 新方法（LLM语义评估）：参考sample.pdf，使用LLM进行语义相似度评估

参考: SciToolEval论文第148-150行
"For both tool planning accuracy and final answer accuracy, we prompt GPT-4o to perform a similarity evaluation."
"""
from typing import List, Dict, Any, Set
import json

from src.utils.llm import LLMClient

# 预处理工具列表 - 这些工具在实际工作流中必需，但文献通常不提及
PREPROCESSING_TOOLS = {
    "dicom_to_nifti",  # DICOM转NIfTI
    "data_loader",     # 数据加载
    "quality_check",   # 质量检查
}


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    计算Jaccard相似度

    Args:
        set1: 集合1
        set2: 集合2

    Returns:
        Jaccard相似度 [0, 1]
    """
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union


def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """
    计算编辑距离（Levenshtein距离）

    Args:
        seq1: 序列1
        seq2: 序列2

    Returns:
        编辑距离
    """
    m, n = len(seq1), len(seq2)

    # 创建DP表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # 删除
                    dp[i][j - 1] + 1,      # 插入
                    dp[i - 1][j - 1] + 1   # 替换
                )

    return dp[m][n]


def normalized_edit_distance(seq1: List[str], seq2: List[str]) -> float:
    """
    计算归一化编辑距离

    Args:
        seq1: 序列1
        seq2: 序列2

    Returns:
        归一化编辑距离 [0, 1]，0表示完全相同
    """
    if not seq1 and not seq2:
        return 0.0

    max_len = max(len(seq1), len(seq2))
    if max_len == 0:
        return 0.0

    edit_dist = levenshtein_distance(seq1, seq2)
    return edit_dist / max_len


def calculate_tool_selection_score(predicted_tools: List[str],
                                   reference_tools: List[str]) -> float:
    """
    计算工具选择得分（Jaccard相似度）

    Args:
        predicted_tools: 预测的工具列表
        reference_tools: 参考工具列表

    Returns:
        工具选择得分 [0, 1]
    """
    pred_set = set(predicted_tools)
    ref_set = set(reference_tools)

    return jaccard_similarity(pred_set, ref_set)


def calculate_tool_order_score(predicted_tools: List[str],
                               reference_tools: List[str]) -> float:
    """
    计算工具顺序得分（1 - 归一化编辑距离）

    Args:
        predicted_tools: 预测的工具序列
        reference_tools: 参考工具序列

    Returns:
        工具顺序得分 [0, 1]
    """
    ned = normalized_edit_distance(predicted_tools, reference_tools)
    return 1.0 - ned


def _extract_tool_names(tools: List[Dict],
                        filter_preprocessing: bool = False) -> List[str]:
    """
    从工具字典列表中提取工具名称

    Args:
        tools: 工具字典列表
        filter_preprocessing: 是否过滤预处理工具

    Returns:
        工具名称列表
    """
    names = []
    for t in tools:
        # 支持多种字段名
        name = t.get("tool") or t.get("tool_name") or t.get("name") or ""
        if name:
            # 如果需要过滤预处理工具
            if filter_preprocessing and name in PREPROCESSING_TOOLS:
                continue
            names.append(name)
    return names


def calculate_planning_accuracy(predicted_tools: List[Dict],
                                reference_tools: List[Dict],
                                selection_weight: float = 0.6,
                                order_weight: float = 0.4,
                                filter_preprocessing: bool = False) -> float:
    """
    计算规划准确率

    公式: Tool Planning Accuracy = 工具选择得分 * 0.6 + 工具顺序得分 * 0.4

    Args:
        predicted_tools: 预测的工具列表 [{"tool": "name", ...}, ...]
        reference_tools: 参考工具列表 [{"tool": "name", ...}, ...]
        selection_weight: 工具选择权重
        order_weight: 工具顺序权重
        filter_preprocessing: 是否过滤预处理工具

    Returns:
        规划准确率 [0, 1]
    """
    # 提取工具名称
    pred_names = _extract_tool_names(predicted_tools, filter_preprocessing)
    ref_names = _extract_tool_names(reference_tools, filter_preprocessing)

    # 处理空列表情况
    if not pred_names and not ref_names:
        return 1.0
    if not pred_names or not ref_names:
        return 0.0

    # 计算工具选择得分
    selection_score = calculate_tool_selection_score(pred_names, ref_names)

    # 计算工具顺序得分
    order_score = calculate_tool_order_score(pred_names, ref_names)

    # 综合得分
    return selection_score * selection_weight + order_score * order_weight


def calculate_detailed_metrics(predicted_tools: List[Dict],
                               reference_tools: List[Dict],
                               filter_preprocessing: bool = False) -> Dict[str, Any]:
    """
    计算详细的评估指标

    Args:
        predicted_tools: 预测的工具列表
        reference_tools: 参考工具列表
        filter_preprocessing: 是否过滤预处理工具

    Returns:
        详细指标字典
    """
    pred_names = _extract_tool_names(predicted_tools, filter_preprocessing)
    ref_names = _extract_tool_names(reference_tools, filter_preprocessing)

    pred_set = set(pred_names)
    ref_set = set(ref_names)

    # 计算各项指标
    true_positives = len(pred_set & ref_set)
    false_positives = len(pred_set - ref_set)
    false_negatives = len(ref_set - pred_set)

    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(ref_set) if ref_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 计算各项得分
    selection_score = calculate_tool_selection_score(pred_names, ref_names)
    order_score = calculate_tool_order_score(pred_names, ref_names)
    planning_accuracy = calculate_planning_accuracy(
        predicted_tools, reference_tools,
        filter_preprocessing=filter_preprocessing
    )

    return {
        "selection_score": selection_score,
        "order_score": order_score,
        "planning_accuracy": planning_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predicted_count": len(pred_names),
        "reference_count": len(ref_names),
        "correct_tools": list(pred_set & ref_set),
        "extra_tools": list(pred_set - ref_set),
        "missing_tools": list(ref_set - pred_set),
        "predicted_sequence": pred_names,
        "reference_sequence": ref_names
    }


# ============================================================================
# LLM语义相似度评估 (参考sample.pdf方法)
# ============================================================================

# 神经影像工具功能等价映射 - 帮助LLM理解工具等价性
EQUIVALENT_TOOLS_HINT = """
神经影像工具功能等价组（仅供参考）：

1. DTI/纤维追踪工具等价:
   - fsl_bedpostx, fsl_probtrackx, fsl_dtifit
   - dsi_studio_fiber_tracking, dsi_studio_roi_analysis
   - mrtrix_tckgen, dipy_tracking

2. 皮层/结构分析等价:
   - freesurfer_recon_all, freesurfer_cortical_thickness, freesurfer_analysis
   - cat12_surface, civet_thickness

3. VBM/灰质分析等价:
   - spm_segment, spm_vbm, spm_analysis
   - fsl_fast, cat12_segment

4. fMRI/功能连接等价:
   - dpabi_preprocessing, dpabi_alff, dpabi_reho, dpabi_analysis
   - conn_fc, afni_3dtcorrelate

5. 统计分析等价:
   - python_stats, spm_stats, fsl_randomise
"""


def _format_tools_for_prompt(tools: List[Dict]) -> str:
    """格式化工具列表用于prompt"""
    if not tools:
        return "（无工具）"

    lines = []
    for i, t in enumerate(tools, 1):
        tool_name = t.get("tool") or t.get("tool_name") or t.get("name") or "unknown"
        tool_type = t.get("type") or t.get("analysis_type") or ""
        desc = t.get("description") or t.get("step") or ""

        line = f"{i}. {tool_name}"
        if tool_type:
            line += f" ({tool_type})"
        if desc:
            line += f": {desc[:100]}"
        lines.append(line)

    return "\n".join(lines)


def llm_evaluate_planning_accuracy(
    predicted_tools: List[Dict],
    reference_tools: List[Dict],
    query: str,
    filter_preprocessing: bool = True,
    evaluator_model: str = None,
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    使用LLM进行语义相似度评估

    参考sample.pdf (SciToolEval)的评估方法：
    "For both tool planning accuracy and final answer accuracy,
    we prompt GPT-4o to perform a similarity evaluation."

    Args:
        predicted_tools: 预测的工具列表
        reference_tools: 参考工具列表
        query: 原始研究问题
        filter_preprocessing: 是否过滤预处理工具
        evaluator_model: 评估器使用的模型（默认使用配置的模型）
        strict_mode: 严格模式（用于baseline评估，要求工具名称精确匹配）

    Returns:
        评估结果字典，包含:
        - tool_selection_score: 工具选择得分 (0-1)
        - tool_order_score: 工具顺序得分 (0-1)
        - planning_accuracy: 综合规划准确率 (0-1)
        - reasoning: 评估理由
    """
    # 过滤预处理工具
    if filter_preprocessing:
        predicted_tools = [t for t in predicted_tools
                         if (t.get("tool") or t.get("tool_name") or "") not in PREPROCESSING_TOOLS]
        reference_tools = [t for t in reference_tools
                         if (t.get("tool") or t.get("tool_name") or "") not in PREPROCESSING_TOOLS]

    # 处理空列表情况
    if not predicted_tools and not reference_tools:
        return {
            "tool_selection_score": 1.0,
            "tool_order_score": 1.0,
            "planning_accuracy": 1.0,
            "reasoning": "两者都没有规划工具",
            "predicted_count": 0,
            "reference_count": 0
        }

    if not predicted_tools:
        return {
            "tool_selection_score": 0.0,
            "tool_order_score": 0.0,
            "planning_accuracy": 0.0,
            "reasoning": "预测工具链为空",
            "predicted_count": 0,
            "reference_count": len(reference_tools)
        }

    if not reference_tools:
        return {
            "tool_selection_score": 0.5,  # 给部分分数，因为有尝试规划
            "tool_order_score": 0.5,
            "planning_accuracy": 0.5,
            "reasoning": "参考工具链为空，无法对比",
            "predicted_count": len(predicted_tools),
            "reference_count": 0
        }

    # 格式化工具列表
    predicted_str = _format_tools_for_prompt(predicted_tools)
    reference_str = _format_tools_for_prompt(reference_tools)

    # 根据模式选择评估标准
    if strict_mode:
        # 严格模式：用于baseline评估，要求工具名称接近
        eval_criteria = """## 评估要求（严格模式）

请严格评估预测工具链与参考工具链的匹配程度（0-100分）：

1. **tool_selection_score（工具选择）**：预测的工具是否与参考工具**名称匹配**？
   - 只有以下工具名称被认为是等价的：
     * freesurfer_analysis, freesurfer, recon-all
     * spm_analysis, spm, vbm
     * fsl_analysis, fsl, dtifit, tbss
     * dpabi_analysis, dpabi, dparsf, conn, functional_connectivity
     * python_stats, statistics, spss, scipy
     * roi_extraction, roi, atlas
   - 预测工具名称如果不在上述列表中（如'CONN Toolbox', 'fMRIPrep', 'Brain Connectivity Toolbox'），视为**不匹配**
   - 预测工具数量与参考差异过大应**大幅扣分**
   - 评分标准：匹配0个工具=0-20分，匹配1个=20-40分，匹配2个=40-60分，匹配大部分=60-80分，完全匹配=80-100分

2. **tool_order_score（工具顺序）**：匹配的工具顺序是否正确？
   - 只考虑匹配到的工具的顺序
   - 如果没有匹配的工具，顺序分=0

3. **overall_score（整体规划）**：= (tool_selection_score + tool_order_score) / 2
   - 如果预测工具与参考工具完全不同，overall应该很低（<30分）"""
    else:
        # 宽松模式：用于Agent评估（已标准化工具名称）
        eval_criteria = f"""## 评估要求

请根据以下标准评估预测工具链与参考工具链的相似度（0-100分）：

1. **tool_selection_score（工具选择）**：预测的工具是否覆盖了解决问题所需的功能？
   - 功能等价的工具应视为正确（如fsl_dtifit ≈ dsi_studio分析）
   - 核心关注：能否完成相同的分析目标

2. **tool_order_score（工具顺序）**：工具执行顺序是否合理？
   - 是否遵循正确的分析流程（如预处理→分析→统计）

3. **overall_score（整体规划）**：综合考虑，预测工具链能否解决原始问题？

{EQUIVALENT_TOOLS_HINT}"""

    # 构建评估prompt
    prompt = f"""你是一位神经影像学专家，请评估工具规划的匹配程度。

## 原始研究问题
{query}

## 参考工具链（专家标注，标准答案）
{reference_str}

## 预测工具链（待评估）
{predicted_str}

{eval_criteria}

## 输出格式（严格JSON）
{{
    "tool_selection_score": <0-100的整数>,
    "tool_order_score": <0-100的整数>,
    "overall_score": <0-100的整数>,
    "reasoning": "<简要说明评估理由，列出匹配和不匹配的工具>"
}}

请直接输出JSON，不要添加其他内容。"""

    # 调用LLM进行评估
    llm = LLMClient()
    if evaluator_model:
        llm.set_model(evaluator_model)

    try:
        messages = [
            {"role": "system", "content": "你是神经影像学专家，请根据功能等价性评估工具规划的相似度。直接输出JSON。"},
            {"role": "user", "content": prompt}
        ]

        result = llm.generate_json(messages, temperature=0.1, max_tokens=1024)

        # 解析结果并归一化为0-1范围
        tool_selection = result.get("tool_selection_score", 50) / 100.0
        tool_order = result.get("tool_order_score", 50) / 100.0
        overall = result.get("overall_score", 50) / 100.0
        reasoning = result.get("reasoning", "")

        # 确保在有效范围内
        tool_selection = max(0.0, min(1.0, tool_selection))
        tool_order = max(0.0, min(1.0, tool_order))
        overall = max(0.0, min(1.0, overall))

        # 计算综合规划准确率（使用overall_score或加权平均）
        planning_accuracy = overall if overall > 0 else (tool_selection * 0.6 + tool_order * 0.4)

        return {
            "tool_selection_score": tool_selection,
            "tool_order_score": tool_order,
            "planning_accuracy": planning_accuracy,
            "reasoning": reasoning,
            "predicted_count": len(predicted_tools),
            "reference_count": len(reference_tools),
            "predicted_sequence": [t.get("tool", "") for t in predicted_tools],
            "reference_sequence": [t.get("tool", "") for t in reference_tools]
        }

    except Exception as e:
        print(f"[LLM评估错误] {e}，回退到Jaccard方法")
        # 回退到传统方法
        return calculate_detailed_metrics(
            predicted_tools, reference_tools,
            filter_preprocessing=False  # 已经过滤过了
        )


# ============================================================================
# 多维度评估扩展 (Multi-dimensional Evaluation)
# 参考SciToolEval论文，增加脑区选择、参数合理性、工作流完整性评估
# ============================================================================

# ROI名称标准化映射
ROI_NORMALIZATION = {
    # 海马相关
    "hippocampus": "海马", "海马": "海马", "海马体": "海马",
    "hippocampal": "海马", "hippo": "海马",
    # 内嗅皮层
    "entorhinal": "内嗅皮层", "entorhinal cortex": "内嗅皮层",
    "内嗅皮层": "内嗅皮层", "内嗅皮质": "内嗅皮层",
    # 颞叶
    "temporal": "颞叶", "temporal lobe": "颞叶",
    "颞叶": "颞叶", "颞叶皮层": "颞叶",
    # 额叶
    "frontal": "额叶", "frontal lobe": "额叶", "prefrontal": "前额叶",
    "额叶": "额叶", "前额叶": "前额叶", "额叶皮层": "额叶",
    # 顶叶
    "parietal": "顶叶", "parietal lobe": "顶叶",
    "顶叶": "顶叶", "顶叶皮层": "顶叶",
    # 枕叶
    "occipital": "枕叶", "occipital lobe": "枕叶",
    "枕叶": "枕叶", "枕叶皮层": "枕叶",
    # 扣带回
    "cingulate": "扣带回", "anterior cingulate": "前扣带回",
    "posterior cingulate": "后扣带回", "扣带回": "扣带回",
    "前扣带回": "前扣带回", "后扣带回": "后扣带回",
    # 杏仁核
    "amygdala": "杏仁核", "杏仁核": "杏仁核", "杏仁体": "杏仁核",
    # 丘脑
    "thalamus": "丘脑", "丘脑": "丘脑",
    # 基底节
    "basal ganglia": "基底节", "基底节": "基底节", "基底核": "基底节",
    "putamen": "壳核", "壳核": "壳核",
    "caudate": "尾状核", "尾状核": "尾状核",
    "pallidum": "苍白球", "苍白球": "苍白球", "globus pallidus": "苍白球",
    # 小脑
    "cerebellum": "小脑", "小脑": "小脑", "cerebellar": "小脑",
    # 脑干
    "brainstem": "脑干", "脑干": "脑干", "brain stem": "脑干",
    "pons": "脑桥", "脑桥": "脑桥",
    "medulla": "延髓", "延髓": "延髓",
    # 黑质
    "substantia nigra": "黑质", "黑质": "黑质",
    # 岛叶
    "insula": "岛叶", "岛叶": "岛叶", "insular": "岛叶",
    # 胼胝体
    "corpus callosum": "胼胝体", "胼胝体": "胼胝体", "cc": "胼胝体",
    # 白质
    "white matter": "白质", "白质": "白质", "wm": "白质",
}


def normalize_roi_names(rois: List[str]) -> Set[str]:
    """
    标准化ROI名称

    Args:
        rois: ROI名称列表

    Returns:
        标准化后的ROI集合
    """
    normalized = set()
    for roi in rois:
        roi_lower = roi.lower().strip()
        # 尝试标准化
        if roi_lower in ROI_NORMALIZATION:
            normalized.add(ROI_NORMALIZATION[roi_lower])
        elif roi in ROI_NORMALIZATION:
            normalized.add(ROI_NORMALIZATION[roi])
        else:
            normalized.add(roi)  # 保持原样
    return normalized


def calculate_roi_accuracy(
    predicted_rois: List[str],
    reference_rois: List[str]
) -> Dict[str, Any]:
    """
    计算脑区选择准确率

    Args:
        predicted_rois: 预测的脑区列表
        reference_rois: 参考脑区列表

    Returns:
        准确率详情
    """
    # 标准化脑区名称
    pred_set = normalize_roi_names(predicted_rois)
    ref_set = normalize_roi_names(reference_rois)

    # 处理空集情况
    if not pred_set and not ref_set:
        return {
            "roi_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "matched_rois": [],
            "extra_rois": [],
            "missing_rois": []
        }

    if not pred_set or not ref_set:
        return {
            "roi_accuracy": 0.0,
            "precision": 0.0 if not ref_set else (1.0 if not pred_set else 0.0),
            "recall": 0.0 if not pred_set else (1.0 if not ref_set else 0.0),
            "jaccard": 0.0,
            "matched_rois": [],
            "extra_rois": list(pred_set),
            "missing_rois": list(ref_set)
        }

    # 计算指标
    intersection = pred_set & ref_set
    union = pred_set | ref_set

    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(ref_set) if ref_set else 0.0
    jaccard = len(intersection) / len(union) if union else 0.0

    # 综合准确率 = F1分数
    roi_accuracy = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "roi_accuracy": roi_accuracy,
        "precision": precision,
        "recall": recall,
        "jaccard": jaccard,
        "matched_rois": list(intersection),
        "extra_rois": list(pred_set - ref_set),
        "missing_rois": list(ref_set - pred_set)
    }


def evaluate_workflow_completeness(
    predicted_tools: List[Dict],
    query: str,
    expected_outputs: List[str] = None
) -> Dict[str, Any]:
    """
    评估工作流完整性

    检查预测的工具链是否能完成研究问题所需的分析

    Args:
        predicted_tools: 预测的工具列表
        query: 研究问题
        expected_outputs: 期望的输出类型（可选）

    Returns:
        完整性评估结果
    """
    # 工具类别映射
    TOOL_CATEGORIES = {
        "preprocessing": ["dicom_to_nifti", "data_loader", "quality_check"],
        "structural": ["freesurfer_analysis", "spm_analysis", "fsl_bet"],
        "diffusion": ["fsl_analysis", "dsi_studio_analysis", "mrtrix"],
        "functional": ["dpabi_analysis", "conn_fc", "afni_analysis"],
        "statistics": ["python_stats", "spm_stats", "fsl_randomise"],
        "visualization": ["python_plot", "fsleyes", "freeview"]
    }

    # 提取工具名称
    tool_names = set()
    for t in predicted_tools:
        name = t.get("tool") or t.get("tool_name") or ""
        if name:
            tool_names.add(name)

    # 检查各类别覆盖情况
    coverage = {}
    for category, tools in TOOL_CATEGORIES.items():
        has_category = any(t in tool_names for t in tools)
        # 也检查是否有功能等价的工具
        for t in tool_names:
            if category in t.lower() or any(cat_tool.split('_')[0] in t for cat_tool in tools):
                has_category = True
                break
        coverage[category] = has_category

    # 根据query判断必需的类别
    required_categories = ["statistics"]  # 统计通常是必需的

    query_lower = query.lower()
    if any(kw in query_lower for kw in ["皮层", "cortical", "厚度", "thickness", "体积", "volume"]):
        required_categories.append("structural")
    if any(kw in query_lower for kw in ["dti", "白质", "纤维", "fa", "md"]):
        required_categories.append("diffusion")
    if any(kw in query_lower for kw in ["功能", "fmri", "连接", "alff", "reho"]):
        required_categories.append("functional")

    # 计算完整性得分
    required_covered = sum(1 for cat in required_categories if coverage.get(cat, False))
    completeness = required_covered / len(required_categories) if required_categories else 1.0

    return {
        "completeness_score": completeness,
        "category_coverage": coverage,
        "required_categories": required_categories,
        "missing_categories": [cat for cat in required_categories if not coverage.get(cat, False)],
        "tool_count": len(tool_names)
    }


def llm_evaluate_comprehensive(
    predicted_tools: List[Dict],
    reference_tools: List[Dict],
    predicted_rois: List[str],
    reference_rois: List[str],
    query: str,
    filter_preprocessing: bool = True
) -> Dict[str, Any]:
    """
    使用LLM进行多维度综合评估

    参考SciToolEval论文，扩展评估维度：
    1. 工具规划准确率
    2. 脑区选择准确率
    3. 工作流完整性

    Args:
        predicted_tools: 预测的工具列表
        reference_tools: 参考工具列表
        predicted_rois: 预测的脑区列表
        reference_rois: 参考脑区列表
        query: 研究问题
        filter_preprocessing: 是否过滤预处理工具

    Returns:
        综合评估结果
    """
    # 1. 工具规划准确率（使用已有的LLM评估）
    tool_metrics = llm_evaluate_planning_accuracy(
        predicted_tools, reference_tools, query, filter_preprocessing
    )

    # 2. 脑区选择准确率
    roi_metrics = calculate_roi_accuracy(predicted_rois, reference_rois)

    # 3. 工作流完整性
    workflow_metrics = evaluate_workflow_completeness(predicted_tools, query)

    # 4. 计算综合得分
    # 权重分配：工具规划40%，脑区选择30%，工作流完整性30%
    overall_score = (
        tool_metrics["planning_accuracy"] * 0.4 +
        roi_metrics["roi_accuracy"] * 0.3 +
        workflow_metrics["completeness_score"] * 0.3
    )

    return {
        # 综合得分
        "overall_score": overall_score,

        # 工具规划维度
        "tool_planning_accuracy": tool_metrics["planning_accuracy"],
        "tool_selection_score": tool_metrics["tool_selection_score"],
        "tool_order_score": tool_metrics["tool_order_score"],
        "tool_reasoning": tool_metrics.get("reasoning", ""),

        # 脑区选择维度
        "roi_selection_accuracy": roi_metrics["roi_accuracy"],
        "roi_precision": roi_metrics["precision"],
        "roi_recall": roi_metrics["recall"],
        "matched_rois": roi_metrics["matched_rois"],
        "missing_rois": roi_metrics["missing_rois"],

        # 工作流完整性维度
        "workflow_completeness": workflow_metrics["completeness_score"],
        "category_coverage": workflow_metrics["category_coverage"],
        "missing_categories": workflow_metrics["missing_categories"],

        # 详细信息
        "predicted_tool_count": len(predicted_tools),
        "reference_tool_count": len(reference_tools),
        "predicted_roi_count": len(predicted_rois),
        "reference_roi_count": len(reference_rois)
    }
