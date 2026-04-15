"""
LangGraph node function - using a professional prompt word system
Based on the ReAct model: Reasoning -> Planning -> Action -> Observation -> Reflection
"""
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from langgraph.errors import GraphBubbleUp, GraphInterrupt

from src.agent.graph_state import AgentState, ResearchPhase, update_state_timestamp, merge_state_updates
from src.agent.prompts import get_node_prompt  # 导入专业提示词系统
from src.config import DATA_DIR, OUTPUT_DIR, DATA_GROUPS, MODALITIES, DATA_LAYOUT, KNOWN_MODALITY_DIRS, SCAN_SKIP_PATTERNS, DATA_GROUP_DESCRIPTIONS
from src.config_local_tools import FSL_SUPPORTED_COMMANDS, FREESURFER_SUPPORTED_COMMANDS
from src.utils.llm import get_llm_client
from src.knowledge.pubmed import search_pubmed
from src.knowledge.knowledge_base import KnowledgeBase, get_knowledge_base
from src.tools.registry import get_registry, ToolCallRequest, Modality
from src.tools.imaging_tools import register_all_imaging_tools
from src.agent.run_tracker import get_run_tracker
from src.agent.memory import get_memory_manager, StepResult, ToolCallRecord
from src.agent.report_auditor import audit_report


# ============== 全局对象初始化 ==============

_llm = None
_registry = None
_tracker = None
_memory = None
_tool_skill = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm_client()
    return _llm

def _get_registry():
    global _registry
    if _registry is None:
        _registry = get_registry()
        register_all_imaging_tools(_registry)
    return _registry

def _get_tracker():
    global _tracker
    if _tracker is None:
        _tracker = get_run_tracker()
    return _tracker

def _get_memory():
    """获取记忆管理器"""
    global _memory
    if _memory is None:
        _memory = get_memory_manager()
        _memory.set_llm_client(_get_llm())  # 设置LLM用于压缩
    return _memory

def _get_tool_skill():
    """获取工具调用技能"""
    global _tool_skill
    if _tool_skill is None:
        from src.agent.skill_learning.tool_calling_skill import ToolCallingSkill
        _tool_skill = ToolCallingSkill()
    return _tool_skill


def _sanitize_tool_result(result) -> Dict[str, Any]:
    """
    将 ToolCallResult 转换为可安全序列化的纯 dict。
    防止 circular reference 和 msgpack recursion limit。
    """
    import dataclasses

    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        try:
            d = dataclasses.asdict(result)
        except Exception:
            d = {}
            for f in dataclasses.fields(result):
                d[f.name] = getattr(result, f.name, None)
    elif hasattr(result, '__dict__'):
        d = result.__dict__.copy()
    elif isinstance(result, dict):
        d = result.copy()
    else:
        return {"raw": str(result)}

    return _deep_sanitize(d, max_depth=15)


def _deep_sanitize(obj, depth=0, max_depth=15, _seen=None):
    """递归清理非序列化对象，限制深度，检测循环引用"""
    if depth > max_depth:
        return str(obj)

    # 循环引用检测：对可变容器（dict/list）跟踪 id
    if _seen is None:
        _seen = set()

    if isinstance(obj, (dict, list)):
        obj_id = id(obj)
        if obj_id in _seen:
            return "<circular_ref>"
        _seen.add(obj_id)

    if isinstance(obj, dict):
        return {str(k): _deep_sanitize(v, depth + 1, max_depth, _seen) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_deep_sanitize(item, depth + 1, max_depth, _seen) for item in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return str(obj)


# 本地知识库全局实例
_knowledge_base = None
_kb_indexed = False  # 标记是否已索引

def _get_knowledge_base() -> KnowledgeBase:
    """
    获取本地知识库实例
    首次调用时会索引paper目录下的PDF文件
    """
    global _knowledge_base, _kb_indexed

    if _knowledge_base is None:
        _knowledge_base = get_knowledge_base()
        # 尝试加载已保存的索引
        try:
            _knowledge_base.load()
            if _knowledge_base.vector_store.chunks:
                _kb_indexed = True
                print(f"[知识库] 已加载缓存索引，共 {len(_knowledge_base.vector_store.chunks)} 个文本块")
        except Exception as e:
            print(f"[知识库] 加载缓存失败: {e}")

    # 如果还没有索引，则索引paper目录
    if not _kb_indexed:
        try:
            print("[知识库] 首次索引paper目录...")
            _knowledge_base.index_local_papers()
            _kb_indexed = True
        except Exception as e:
            print(f"[知识库] 索引失败: {e}")

    return _knowledge_base


# ============== 工具输出文件类型配置 ==============

def get_tool_required_extensions(source_tool: str) -> dict:
    """
    根据工具类型返回文件扩展名配置

    关键改进：严格区分"处理输出"和"分析结果"
    - 处理输出：NIfTI、stats等原始处理文件
    - 分析结果：CSV、JSON等统计结果（应被排除）

    Args:
        source_tool: 工具名称 (spm, freesurfer, dsi_studio, fsl, ants)

    Returns:
        dict: {
            "required": 必须存在的扩展名列表（用于验证目录有效性）,
            "all": 所有可能的输出文件扩展名（用于收集文件列表）
        }
    """
    tool_config = {
        "spm": {
            "required": ["*.nii", "*.nii.gz"],  # SPM必须有NIfTI文件
            "all": ["*.nii", "*.nii.gz", "*.mat"]  # 不包含csv/json
        },
        "freesurfer": {
            "required": ["*.stats"],  # FreeSurfer必须有stats文件
            "all": ["*.stats", "*.mgz", "*.inflated", "*.pial", "*.white", "*.annot"]
        },
        "dsi_studio": {
            "required": ["*.fib.gz", "*.fz"],  # DSI Studio必须有FIB文件（支持新旧格式）
            "all": ["*.fib.gz", "*.fz", "*.sz", "*.src.gz.sz", "*.tt.gz", "*.trk.gz", "*.trk", "*.nii.gz"]
        },
        "fsl": {
            "required": ["*.nii.gz", "*.nii"],  # FSL必须有NIfTI
            "all": ["*.nii.gz", "*.nii", "*.mat"]
        },
        "ants": {
            "required": ["*.nii.gz", "*.nii"],  # ANTs必须有NIfTI
            "all": ["*.nii.gz", "*.nii", "*.mat", "*.h5"]
        }
    }

    # 默认配置（未知工具）
    default = {
        "required": ["*.nii", "*.nii.gz"],
        "all": ["*.nii", "*.nii.gz"]
    }

    return tool_config.get(source_tool.lower(), default)


# ============== 工具模态需求配置 ==============

# 工具到所需模态的映射
TOOL_MODALITY_REQUIREMENTS = {
    "freesurfer_analysis": "anat",      # FreeSurfer需要T1结构像
    "dsi_studio_analysis": "dwi",       # DSI Studio需要DWI
    "fsl_analysis": {                   # FSL根据command不同
        "dtifit": "dwi",
        "eddy": "dwi",
        "bedpostx": "dwi",
        "probtrackx": "dwi",
        "tbss": "dwi",
        "bet": "anat",                  # 脑提取通常用T1
        "fast": "anat",
        "flirt": "anat",
        "fnirt": "anat",
        "first": "anat",
        "feat": "func",
        "melodic": "func"
    },
    "spm_analysis": {                   # SPM根据analysis_type不同
        "vbm_segment": "anat",
        "vbm_dartel": "anat",
        "normalize": "anat",
        "smooth": "anat",               # 默认结构像
        "realign": "func",
        "slice_timing": "func",
        "coregister": "func",           # fMRI配准
        "first_level": "func",
        "second_level": "anat"          # 组分析通常用结构
    },
    "dpabi_analysis": "func",           # DPABI需要fMRI
    "ants_analysis": "anat"             # ANTs配准通常用T1
}


def infer_required_modality(tool_name: str, params: dict = None) -> Optional[str]:
    """
    根据工具和参数推断所需的数据模态

    Args:
        tool_name: 工具名称
        params: 工具参数（可能包含command或analysis_type）

    Returns:
        模态字符串 ("anat", "dwi", "func") 或 None（无法推断）
    """
    if params is None:
        params = {}

    req = TOOL_MODALITY_REQUIREMENTS.get(tool_name)

    if req is None:
        return None

    if isinstance(req, str):
        # 直接映射
        return req

    if isinstance(req, dict):
        # 根据子命令/分析类型决定
        # 优先检查command（FSL），其次检查analysis_type（SPM）
        command = params.get("command", "")
        if command and command in req:
            return req[command]

        analysis_type = params.get("analysis_type", "")
        if analysis_type and analysis_type in req:
            return req[analysis_type]

        # 返回字典中的第一个值作为默认
        return next(iter(req.values()), "anat")

    return None


def get_downstream_tasks(task_id: str, all_tasks: list) -> list:
    """
    递归获取指定任务的所有下游任务（依赖该任务的任务）

    Args:
        task_id: 当前任务ID
        all_tasks: 所有任务列表

    Returns:
        下游任务列表（包括直接和间接依赖）
    """
    downstream = []
    visited = set()  # 防止循环依赖导致无限递归

    def _find_downstream(tid):
        if tid in visited:
            return
        visited.add(tid)

        for task in all_tasks:
            depends = getattr(task, 'depends_on', None) or []
            if tid in depends:
                downstream.append(task)
                _find_downstream(task.task_id)

    _find_downstream(task_id)
    return downstream


# ============== 数据复用检查机制 ==============

def _fill_tool_params(tool_name: str, task_description: str, params: Dict[str, Any], state: AgentState) -> Dict[str, Any]:
    """
    智能填充工具参数（特别是analysis_type等关键参数）

    Args:
        tool_name: 工具名称
        task_description: 任务描述
        params: 原始参数字典
        state: 当前状态

    Returns:
        填充后的参数字典
    """
    filled_params = params.copy()

    # 如果是SPM分析，根据任务描述推断analysis_type
    if tool_name == "spm_analysis" and "analysis_type" not in filled_params:
        description_lower = (task_description or "").lower()
        if "segment" in description_lower or "vbm" in description_lower:
            filled_params["analysis_type"] = "vbm_segment"
        elif "smooth" in description_lower:
            filled_params["analysis_type"] = "smooth"
        elif "normal" in description_lower:
            filled_params["analysis_type"] = "normalize"
        elif "dartel" in description_lower:
            filled_params["analysis_type"] = "vbm_dartel"

    # 如果是统计分析，根据研究问题推断analysis_type
    elif tool_name == "python_stats" and "analysis_type" not in filled_params:
        description_lower = (task_description or "").lower()
        research_question = state.get("question", "").lower()

        # 合并描述和研究问题进行分析
        combined_text = f"{description_lower} {research_question}"

        # 检测分析类型
        if "相关" in combined_text or "correlation" in combined_text:
            filled_params["analysis_type"] = "correlation"
        elif "anova" in combined_text or "方差分析" in combined_text:
            filled_params["analysis_type"] = "anova"
        elif any(kw in combined_text for kw in ["比较", "compare", "comparison", "difference", "between", "vs", "versus"]):
            # 默认使用t检验进行组间比较
            filled_params["analysis_type"] = "ttest"
        else:
            # 如果无法明确判断，但是research_question提到了多个组，默认使用t检验
            # 检查是否有组的概念
            if any(kw in combined_text for kw in ["组", "group", "患者", "patient", "健康", "healthy", "control", "病人", "hc", "sca"]):
                filled_params["analysis_type"] = "ttest"
                print(f"  [智能参数] 根据研究问题推断analysis_type=ttest（检测到组间比较）")
            else:
                # 最后的默认值
                filled_params["analysis_type"] = "ttest"
                print(f"  [智能参数] 使用默认analysis_type=ttest")

    return filled_params

def _compute_execution_hash(tool_name: str, input_files: List[str], params: Dict) -> str:
    """
    计算工具执行的唯一标识（哈希值）

    基于：工具名称 + 输入文件列表 + 参数
    用于检测是否已有相同配置的处理结果
    """
    # 排序输入文件以确保一致性
    sorted_inputs = sorted([str(Path(f).name) for f in input_files])

    # 构建标识字符串
    identifier_parts = [
        f"tool:{tool_name}",
        f"inputs:{','.join(sorted_inputs)}",
        f"params:{json.dumps(params, sort_keys=True, default=str)}"
    ]
    identifier_str = "||".join(identifier_parts)

    # 计算MD5哈希
    hash_obj = hashlib.md5(identifier_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]  # 使用前16位


def _check_cached_result(run_dir: Path, tool_name: str, input_files: List[str],
                        params: Dict) -> Optional[Tuple[str, Dict]]:
    """
    检查是否存在可复用的处理结果

    Args:
        run_dir: 运行目录
        tool_name: 工具名称
        input_files: 输入文件列表
        params: 工具参数

    Returns:
        如果找到缓存：(output_dir, cached_result)
        如果未找到：None
    """
    exec_hash = _compute_execution_hash(tool_name, input_files, params)
    tools_dir = run_dir / "tools"

    if not tools_dir.exists():
        return None

    # 查找匹配的输出目录（包含相同哈希的目录）
    for tool_dir in tools_dir.iterdir():
        if not tool_dir.is_dir():
            continue

        # 检查是否是该工具的目录
        if tool_name not in tool_dir.name:
            continue

        # 检查是否有缓存标记文件
        cache_meta_path = tool_dir / ".cache_meta.json"
        if not cache_meta_path.exists():
            continue

        try:
            # 尝试读取缓存元数据，处理可能的编码错误（如旧版本保存的emoji）
            try:
                with open(cache_meta_path, 'r', encoding='utf-8') as f:
                    cache_meta = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"  [数据复用] 警告: 缓存元数据读取失败 ({e})，跳过此缓存")
                continue

            # 检查哈希是否匹配
            if cache_meta.get("exec_hash") == exec_hash:
                # 检查输出文件是否仍然存在
                output_files = cache_meta.get("output_files", [])
                all_exist = all(Path(f).exists() for f in output_files)

                if all_exist:
                    print(f"  [缓存命中] 找到可复用结果: {tool_dir.name}")
                    print(f"    哈希: {exec_hash}")
                    print(f"    输出文件: {len(output_files)} 个")
                    return str(tool_dir), cache_meta
        except Exception as e:
            print(f"  [缓存检查] 读取缓存元数据失败: {e}")
            continue

    return None


def _save_cache_metadata(output_dir: str, tool_name: str, input_files: List[str],
                        params: Dict, result: Dict):
    """
    保存缓存元数据，用于后续复用检查

    Args:
        output_dir: 输出目录
        tool_name: 工具名称
        input_files: 输入文件列表
        params: 工具参数
        result: 执行结果
    """
    exec_hash = _compute_execution_hash(tool_name, input_files, params)

    cache_meta = {
        "exec_hash": exec_hash,
        "tool_name": tool_name,
        "input_files": input_files,
        "params": params,
        "output_files": result.get("output_files", []),
        "status": result.get("status", "unknown"),
        "cached_at": datetime.now().isoformat(),
        "duration_seconds": result.get("duration_seconds", 0)
    }

    cache_meta_path = Path(output_dir) / ".cache_meta.json"
    with open(cache_meta_path, 'w', encoding='utf-8') as f:
        json.dump(cache_meta, f, indent=2, ensure_ascii=False)


def _filter_structural_images(file_paths: List[str]) -> List[str]:
    """
    智能过滤NIfTI文件，每个被试只保留一个主要的结构像

    对于VBM分析，每个被试应该只有一个T1加权结构像。
    如果某个被试有多个文件（如fMRI时间序列），只保留最可能是T1结构像的那个。

    Args:
        file_paths: NIfTI文件路径列表

    Returns:
        过滤后的文件路径列表（每个被试一个文件）
    """
    from pathlib import Path
    import re

    # 按被试分组文件
    subject_files = {}

    for file_path in file_paths:
        path = Path(file_path)

        # 提取被试ID（从文件名）
        # **关键修复**: 始终从文件名提取被试ID，不使用父目录
        # 因为父目录可能是组名（HC1, SCA3），而不是被试ID
        filename = path.stem  # 去掉.nii或.nii.gz

        # **关键修复**: 先移除SPM处理前缀（c1, c2, c3, m, w, wc1, wc2, y_等）
        # 这些前缀是SPM分割和处理产生的：
        # - c1-c6: 组织概率图
        # - m: 偏场校正
        # - w: warped（标准化）
        # - wc1, wc2: warped and modulated
        # - y_: 变形场
        spm_prefixes = [r'^wc\d+', r'^c\d+', r'^y_', r'^w', r'^m']
        for prefix_pattern in spm_prefixes:
            filename = re.sub(prefix_pattern, '', filename)

        # **关键修复**: 只在有多个数字后缀时移除最后一个（扫描编号）
        # 例如: HC1_0001_01 → HC1_0001（移除扫描编号）
        #      HC1_0001 → HC1_0001（保留被试编号）
        # 检查是否有两个或更多的 _数字 模式
        digit_suffixes = re.findall(r'_\d+', filename)
        if len(digit_suffixes) >= 2:
            # 有多个数字后缀，移除最后一个（扫描序号）
            subject_id = re.sub(r'_\d+$', '', filename)
        else:
            # 只有一个或没有数字后缀，保留完整文件名作为被试ID
            subject_id = filename

        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(file_path)

    # 对每个被试选择主要的结构像
    filtered = []

    for subject_id, files in subject_files.items():
        if len(files) == 1:
            # 只有一个文件，直接使用
            filtered.append(files[0])
        else:
            # 有多个文件，选择最可能是T1结构像的
            # 策略：
            # 1. 优先选择文件名不包含序号的（如HC1_0001.nii而不是HC1_0001_01.nii）
            # 2. 如果都有序号或都没有，选择文件最大的（3D T1通常比单层或fMRI大）

            files_with_info = []
            for f in files:
                path = Path(f)
                filename = path.stem

                # 检查是否有序号后缀
                has_number_suffix = bool(re.search(r'_\d+$', filename))

                # 获取文件大小（如果文件存在）
                try:
                    file_size = path.stat().st_size if path.exists() else 0
                except Exception:
                    file_size = 0

                files_with_info.append({
                    'path': f,
                    'has_number_suffix': has_number_suffix,
                    'file_size': file_size
                })

            # 排序：优先选择无序号后缀的，其次选择文件大的
            files_with_info.sort(key=lambda x: (x['has_number_suffix'], -x['file_size']))

            selected = files_with_info[0]['path']
            filtered.append(selected)

            # 记录过滤信息
            print(f"      [被试 {subject_id}] {len(files)} 个文件 → 保留: {Path(selected).name}")

    return filtered


def _detect_data_layout(data_dir: Path) -> str:
    """
    自动检测数据目录的布局类型

    Returns:
        "bids" | "grouped" | "flat"
    """
    if not data_dir.exists():
        return "grouped"

    top_dirs = [d for d in data_dir.iterdir()
                if d.is_dir() and d.name not in SCAN_SKIP_PATTERNS and not d.name.endswith(".db")]

    # 检查BIDS: 顶层有 sub- 开头的目录
    if any(d.name.startswith("sub-") for d in top_dirs):
        return "bids"

    # 检查grouped: 顶层目录包含模态子目录
    for d in top_dirs:
        sub_names = {sd.name for sd in d.iterdir() if sd.is_dir()}
        if sub_names & KNOWN_MODALITY_DIRS:
            return "grouped"

    # 检查flat: 顶层直接有NIfTI文件
    nii_files = list(data_dir.glob("*.nii")) + list(data_dir.glob("*.nii.gz"))
    if nii_files:
        return "flat"

    return "grouped"


def _discover_groups(data_dir: Path, layout: str) -> Dict[str, List[str]]:
    """
    从文件系统自动发现组别和被试

    Returns:
        {group_name: [subject_id, ...], ...}
    """
    groups = {}

    if layout == "bids":
        # BIDS: 所有 sub-* 目录归为 "all" 组
        subjects = [d.name for d in data_dir.iterdir()
                    if d.is_dir() and d.name.startswith("sub-")]
        if subjects:
            groups["all"] = sorted(subjects)

    elif layout == "flat":
        # flat: 所有NIfTI文件归为 "all" 组
        nii_files = list(data_dir.glob("*.nii")) + list(data_dir.glob("*.nii.gz"))
        subjects = list({f.name.split(".")[0] for f in nii_files})
        if subjects:
            groups["all"] = sorted(subjects)

    else:
        # grouped: 扫描顶层目录，找到包含模态子目录的
        for d in data_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name in SCAN_SKIP_PATTERNS or d.name.endswith(".db"):
                continue

            sub_names = {sd.name for sd in d.iterdir() if sd.is_dir()}
            if sub_names & KNOWN_MODALITY_DIRS:
                # 收集该组下所有被试（从所有模态目录中汇总）
                subjects = set()
                for modality_name in sub_names & KNOWN_MODALITY_DIRS:
                    modality_path = d / modality_name
                    for subject_dir in modality_path.iterdir():
                        if subject_dir.is_dir():
                            subjects.add(subject_dir.name)
                if subjects:
                    groups[d.name] = sorted(subjects)

    return groups


# 模块级缓存，避免重复扫描
_data_structure_cache = None


def _get_data_structure() -> Dict:
    """
    扫描本地数据结构，自动发现组别和布局
    包括检测文件格式（NIfTI vs DICOM）和人口统计学数据

    Returns:
        包含groups, subjects, modalities, data_format, demographics等的字典
    """
    global _data_structure_cache
    if _data_structure_cache is not None:
        return _data_structure_cache

    # 检测布局
    layout = DATA_LAYOUT
    if layout == "auto":
        layout = _detect_data_layout(DATA_DIR)
    print(f"  [数据扫描] 检测到数据布局: {layout}")

    # 自动发现组别
    discovered_groups = _discover_groups(DATA_DIR, layout)
    print(f"  [数据扫描] 自动发现组别: {list(discovered_groups.keys())}")

    structure = {
        "groups": {},
        "subjects": [],
        "modalities": list(MODALITIES.keys()),
        "data_format": "unknown",
        "needs_conversion": False,
        "demographics": None,
        "demographics_file": None,
        "layout": layout,
        "subject_modalities": {},  # subject_id -> [modality, ...]
        "modality_counts": {},     # group -> {modality: count}
    }

    nifti_count = 0
    dicom_count = 0

    for group_name, subject_ids in discovered_groups.items():
        group_desc = DATA_GROUP_DESCRIPTIONS.get(group_name, group_name)
        group_nifti = 0
        group_dicom = 0
        group_modality_counts = {}

        # 记录每个被试的可用模态
        for subject_id in subject_ids:
            available_mods = []
            for modality in MODALITIES.keys():
                # 根据布局构建路径
                if layout == "grouped":
                    subject_dir = DATA_DIR / group_name / modality / subject_id
                elif layout == "bids":
                    subject_dir = DATA_DIR / subject_id / modality
                else:
                    subject_dir = DATA_DIR  # flat不按目录区分

                if layout == "flat" or (subject_dir.exists() and subject_dir.is_dir()):
                    available_mods.append(modality)
                    group_modality_counts[modality] = group_modality_counts.get(modality, 0) + 1

                    # 检测文件格式（只检查第一个被试的anat目录）
                    if modality == "anat" and group_nifti == 0 and group_dicom == 0 and layout != "flat":
                        nii_files = list(subject_dir.glob("*.nii")) + list(subject_dir.glob("*.nii.gz"))
                        if nii_files:
                            group_nifti += 1
                        else:
                            dcm_files = list(subject_dir.glob("*.dcm"))
                            if dcm_files:
                                group_dicom += 1
                            else:
                                all_files = [f for f in subject_dir.iterdir() if f.is_file()]
                                if all_files:
                                    group_dicom += 1

            structure["subject_modalities"][subject_id] = available_mods

        nifti_count += group_nifti
        dicom_count += group_dicom

        structure["groups"][group_name] = {
            "description": group_desc,
            "subjects": subject_ids,
            "count": len(subject_ids),
            "format": "nifti" if group_nifti > 0 else ("dicom" if group_dicom > 0 else "unknown")
        }
        structure["modality_counts"][group_name] = group_modality_counts

        for subject_id in subject_ids:
            if subject_id not in structure["subjects"]:
                structure["subjects"].append(subject_id)

        # 打印组别信息
        print(f"  [数据扫描] {group_name} ({group_desc}): {len(subject_ids)} 个被试")
        for mod, cnt in group_modality_counts.items():
            print(f"    - {mod}: {cnt} 个被试")

    # 确定整体数据格式
    if nifti_count > 0 and dicom_count == 0:
        structure["data_format"] = "nifti"
        structure["needs_conversion"] = False
    elif dicom_count > 0 and nifti_count == 0:
        structure["data_format"] = "dicom"
        structure["needs_conversion"] = True
    elif nifti_count > 0 and dicom_count > 0:
        structure["data_format"] = "mixed"
        structure["needs_conversion"] = True
    else:
        structure["data_format"] = "unknown"

    # 读取人口统计学和量表数据（如果存在）
    demographics_file = DATA_DIR / "data.xlsx"
    if demographics_file.exists():
        try:
            import pandas as pd
            df = pd.read_excel(demographics_file)

            # 提取列名和基本统计信息
            columns = df.columns.tolist()
            n_subjects = len(df)

            # 识别ID列（可能是ID、Subject、SubjectID等）
            id_column = None
            for col in columns:
                if col.lower() in ['id', 'subject', 'subjectid', 'subject_id']:
                    id_column = col
                    break

            # 识别分组列（可能是group、Group、Diagnosis等）
            group_column = None
            for col in columns:
                if col.lower() in ['group', 'diagnosis', 'condition']:
                    group_column = col
                    break

            # 构建摘要信息
            demographics_summary = {
                "file_path": str(demographics_file),
                "n_subjects": n_subjects,
                "columns": columns,
                "id_column": id_column,
                "group_column": group_column,
                "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
            }

            # 如果有分组列，统计各组人数
            if group_column:
                group_counts = df[group_column].value_counts().to_dict()
                demographics_summary["groups"] = group_counts

            # 计算数值型列的基本统计
            numeric_stats = {}
            for col in demographics_summary["numeric_columns"]:
                if col != id_column:  # 跳过ID列
                    numeric_stats[col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max())
                    }
            demographics_summary["statistics"] = numeric_stats

            structure["demographics"] = demographics_summary
            structure["demographics_file"] = str(demographics_file)

            print(f"  [数据扫描] 发现人口统计学数据: {demographics_file.name}")
            print(f"    - 被试数量: {n_subjects}")
            print(f"    - 变量列: {', '.join(columns)}")
            if group_column and demographics_summary.get("groups"):
                print(f"    - 分组: {demographics_summary['groups']}")

        except Exception as e:
            print(f"  [警告] 读取人口统计学数据失败: {e}")

    _data_structure_cache = structure
    return structure


# ============== 节点函数 ==============

def node_init(state: AgentState) -> Dict[str, Any]:
    """
    初始化节点
    支持恢复模式：如果tracker已经有current_run，则跳过创建

    Returns:
        包含更新字段的字典（LangGraph 会自动合并到 state）
    """
    print(f"\n[NODE: init] 初始化研究任务...")

    # 重置模块级缓存，确保每次运行使用最新数据
    global _data_structure_cache
    _data_structure_cache = None

    tracker = _get_tracker()

    # 检查是否为恢复模式（tracker已经加载了运行）
    if tracker.current_run is None:
        # 新运行：创建并启动
        tracker.create_run(state["run_id"], state["question"])
        tracker.start_run()
        print(f"  [新运行] 创建运行: {state['run_id']}")
    else:
        # 恢复模式：使用已加载的运行
        print(f"  [恢复模式] 使用已有运行: {tracker.current_run.run_id}")
        print(f"  [恢复模式] 状态: {tracker.current_run.status.value}")

    # 返回更新字典，LangGraph 会自动合并
    # 注意：node_history 和 messages 使用 operator.add，所以返回列表会被累积
    return merge_state_updates({
        "phase": ResearchPhase.PLANNING.value,
        "node_history": ["init"],
        "messages": [{"role": "system", "content": "开始研究任务"}]
    })


def node_parse_question(state: AgentState) -> Dict[str, Any]:
    """
    解析研究问题 - 使用Reasoning阶段的专业提示词
    属于ReAct模式的Reasoning阶段
    使用K2推理模型进行深度分析（识别疾病、预测脑区）
    支持恢复模式：如果已有parsed_intent，则跳过

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: parse_question] 使用K2推理模型深度解析研究问题...")

    # 检查是否已经解析过（恢复模式）
    if state.get("parsed_intent"):
        print(f"  [恢复模式] 检测到已有解析结果，跳过重新解析")
        return merge_state_updates({
            "node_history": ["parse_question:skipped"]
        })

    llm = _get_llm()
    tracker = _get_tracker()
    data_structure = _get_data_structure()

    # 打印数据格式信息
    print(f"\n  [数据扫描结果]")
    print(f"    数据格式: {data_structure.get('data_format', 'unknown')}")
    print(f"    需要DICOM转换: {data_structure.get('needs_conversion', False)}")
    for group_name, group_info in data_structure.get('groups', {}).items():
        print(f"    {group_name}: {group_info.get('count', 0)}个被试, 格式={group_info.get('format', 'unknown')}")

    tracker.start_step("01_parse_question", {"question": state["question"]})

    # 使用专业的推理提示词
    prompts = get_node_prompt(
        "parse_question",
        question=state["question"],
        context={
            "available_data": data_structure,
            "research_domain": "neuroimaging",
            "analysis_tools": ["SPM", "DPABI", "FSL", "DSI Studio", "Nipype"]
        }
    )

    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]}
    ]

    # 设置任务类型以使用K2推理模型（需要识别疾病、预测脑区）
    llm.set_task_type("parse_question")

    try:
        # LLM进行推理分析
        result = llm.generate_json(messages)

        tracker.add_step_artifact("01_parse_question", "parsed_intent.json", result)
        tracker.finish_step("01_parse_question", outputs=result)

        print(f"  研究类型: {result.get('research_type')}")
        print(f"  推理步骤: {len(result.get('reasoning_chain', []))} 步")

        # 显示疾病识别结果
        disease_info = result.get('disease_info', {})
        if disease_info and disease_info.get('disease_type'):
            print(f"\n  [疾病识别]")
            print(f"    疾病类型: {disease_info.get('disease_type')}")
            print(f"    疾病分类: {disease_info.get('disease_category', '未知')}")

        # 显示脑区预测结果
        expected_regions = result.get('expected_brain_regions', {})
        if expected_regions and expected_regions.get('primary_regions'):
            print(f"\n  [脑区预测]")
            print(f"    主要区域: {expected_regions.get('primary_regions', [])}")
            if expected_regions.get('secondary_regions'):
                print(f"    次要区域: {expected_regions.get('secondary_regions', [])}")

        if result.get('uncertainties'):
            print(f"\n  不确定性: {result.get('uncertainties')}")

        # 返回更新字典
        return merge_state_updates({
            "parsed_intent": result,
            "reasoning_trace": result.get("reasoning_chain", []),
            "node_history": ["parse_question"],
            "messages": [{"role": "assistant", "content": f"解析完成: {result.get('research_type')}"}]
        })

    except Exception as e:
        tracker.finish_step("01_parse_question", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"parse_question: {e}"]
        })


def node_search_knowledge(state: AgentState) -> Dict[str, Any]:
    """
    检索知识库 - 增强版：包含疾病-脑区映射提取
    支持恢复模式：如果已有knowledge，则跳过

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: search_knowledge] 检索文献知识...")

    # 检查是否已经检索过（恢复模式）
    # 检查citations、local_evidence或evidence字段来判断是否已经检索过
    if state.get("citations") or state.get("local_evidence") or state.get("evidence"):
        print(f"  [恢复模式] 检测到已有知识库数据，跳过重新检索")
        return merge_state_updates({
            "node_history": ["search_knowledge:skipped"]
        })

    llm = _get_llm()
    tracker = _get_tracker()

    intent = state.get("parsed_intent", {})
    tracker.start_step("02_search_knowledge", {"keywords": intent.get("keywords_en", [])})

    try:
        citations = []
        methodology_notes = []
        brain_region_suggestions = {}

        # 获取疾病信息（来自parse_question的输出）
        disease_info = intent.get("disease_info", {})
        disease_type = disease_info.get("disease_type") if disease_info else None
        expected_regions = intent.get("expected_brain_regions", {})

        # PubMed检索 - 增强版：添加疾病-脑区相关检索
        keywords = intent.get("keywords_en", [])
        if keywords:
            # 基础检索
            query = " AND ".join(keywords[:3])
            articles = search_pubmed(query, max_results=3)
            citations = [{
                "pmid": a.pmid,
                "title": a.title,
                "authors": a.authors[:3] if a.authors else [],
                "year": a.year,
                "journal": a.journal
            } for a in articles]

            # 疾病-脑区专项检索（如果有疾病类型）
            if disease_type and disease_type.lower() not in ["null", "none", ""]:
                brain_query = f"{disease_type} brain regions MRI neuroimaging"
                print(f"  [脑区检索] 搜索: {brain_query}")
                try:
                    brain_articles = search_pubmed(brain_query, max_results=3)
                    brain_citations = [{
                        "pmid": a.pmid,
                        "title": a.title,
                        "authors": a.authors[:3] if a.authors else [],
                        "year": a.year,
                        "journal": a.journal,
                        "search_type": "brain_region_specific"
                    } for a in brain_articles]
                    # 合并文献，避免重复
                    existing_pmids = {c["pmid"] for c in citations}
                    for bc in brain_citations:
                        if bc["pmid"] not in existing_pmids:
                            citations.append(bc)
                    print(f"  [脑区检索] 找到 {len(brain_citations)} 篇脑区相关文献")
                except Exception as e:
                    print(f"  [脑区检索] 检索失败: {e}")

        # ========== 本地知识库检索 ==========
        # 检索paper目录下的PDF文献
        local_evidence = []
        try:
            kb = _get_knowledge_base()
            if kb.vector_store.chunks:  # 如果有索引的文档
                # 构建本地检索查询
                local_query = state["question"]
                if disease_type:
                    local_query = f"{disease_type} {local_query}"

                print(f"  [本地知识库] 检索本地文献...")
                local_results = kb.search_local(local_query, top_k=5)

                if local_results:
                    for result in local_results:
                        local_evidence.append({
                            "source": result.source,
                            "section_type": result.section_type,
                            "content": result.content[:500],  # 截取前500字符
                            "score": round(result.score, 3),
                            "metadata": result.metadata,
                            "source_type": "local_paper"
                        })
                    print(f"  [本地知识库] 找到 {len(local_results)} 条相关证据")

                    # 提取本地文献的方法学要点
                    if local_results:
                        local_methods_content = "\n\n".join([
                            f"[{r.source}] ({r.section_type}): {r.content[:300]}"
                            for r in local_results[:3]
                        ])
                        print(f"  [本地知识库] 提取本地文献方法学要点...")
                else:
                    print(f"  [本地知识库] 未找到相关文献")
            else:
                print(f"  [本地知识库] 知识库为空，跳过本地检索")
        except Exception as e:
            print(f"  [本地知识库] 检索失败: {e}")

        # 提取方法学建议（整合PubMed和本地文献）
        if citations or local_evidence:
            # 构建本地文献摘要
            local_summary = ""
            if local_evidence:
                local_summary = "\n\n## 本地文献证据\n"
                for ev in local_evidence[:3]:
                    local_summary += f"\n### {ev['source']} ({ev['section_type']})\n{ev['content'][:300]}...\n"

            methods_prompt = f"""基于以下文献，提取神经影像分析的方法学建议：

## PubMed文献
{json.dumps(citations, ensure_ascii=False, indent=2, default=str)}
{local_summary}

研究问题：{state["question"]}

请列出3-5个关键的方法学建议（JSON数组格式）。
优先参考本地文献中的具体方法描述。"""

            try:
                notes = llm.generate_json(methods_prompt)
                if isinstance(notes, list):
                    methodology_notes = notes
            except Exception:
                pass

        # ========== 新增：提取脑区建议 ==========
        # 综合疾病先验知识和文献证据，生成脑区选择建议
        brain_region_prompt = f"""作为神经影像专家，请根据以下信息，识别该研究应该重点关注的脑区。

## 研究问题
{state["question"]}

## 疾病信息
- 疾病类型: {disease_type if disease_type else "未指定"}
- 疾病分类: {disease_info.get("disease_category", "未知") if disease_info else "未知"}
- 病理特点: {disease_info.get("pathology_features", "未知") if disease_info else "未知"}

## 问题解析阶段的脑区预测
- 主要预期区域: {expected_regions.get("primary_regions", []) if expected_regions else []}
- 次要预期区域: {expected_regions.get("secondary_regions", []) if expected_regions else []}
- 预测理由: {expected_regions.get("region_rationale", "") if expected_regions else ""}

## PubMed相关文献
{json.dumps(citations, ensure_ascii=False, indent=2, default=str)}

## 本地文献证据
{json.dumps(local_evidence[:3], ensure_ascii=False, indent=2, default=str) if local_evidence else "无本地文献"}

## 任务
综合以上信息，输出JSON格式的脑区选择建议：

```json
{{
  "primary_rois": ["最应该优先分析的脑区1", "脑区2", "脑区3"],
  "primary_rationale": "选择这些作为主要ROI的理由（基于病理学和文献）",
  "secondary_rois": ["次要关注的脑区1", "脑区2"],
  "secondary_rationale": "选择这些作为次要ROI的理由",
  "recommended_metrics": ["thickness", "volume", "area"],
  "metrics_rationale": "推荐这些指标的理由",
  "literature_support": "文献支持的简要说明",
  "analysis_priority": "hypothesis_driven 或 exploratory",
  "special_considerations": "特殊考虑事项（如某些区域分割质量问题等）"
}}
```

重要：
1. 如果研究涉及明确的疾病，使用hypothesis_driven策略，优先分析已知受累区域
2. 如果是探索性研究或无明确疾病，使用exploratory策略，全脑筛查
3. primary_rois应该是文献中最常报告的受累区域
4. 考虑FreeSurfer/SPM能提取的标准脑区名称"""

        try:
            brain_region_suggestions = llm.generate_json(brain_region_prompt)
            if brain_region_suggestions:
                print(f"  [脑区建议] Primary ROIs: {brain_region_suggestions.get('primary_rois', [])}")
                print(f"  [脑区建议] 分析策略: {brain_region_suggestions.get('analysis_priority', 'unknown')}")

                # ========== Ur: Retrieval-driven update ==========
                # 基于文献检索结果更新动态知识图谱
                if disease_type and disease_type.lower() not in ["null", "none", ""]:
                    try:
                        from src.knowledge.dynamic_knowledge_graph import get_dynamic_kg
                        dynamic_kg = get_dynamic_kg()

                        # 构建文献发现字典
                        literature_findings = {
                            "primary_rois": brain_region_suggestions.get("primary_rois", []),
                            "secondary_rois": brain_region_suggestions.get("secondary_rois", []),
                            "confidence": 0.8,  # 基于文献的置信度
                            "source": "literature_retrieval"
                        }

                        dynamic_kg.update_retrieval_driven(
                            disease=disease_type,
                            literature_findings=literature_findings
                        )
                        print(f"  [Ur更新] 基于文献更新了知识图谱")
                    except Exception as e:
                        print(f"  [Ur更新] 失败: {e}")

        except Exception as e:
            print(f"  [脑区建议] 提取失败: {e}")
            # 使用问题解析阶段的预测作为备选
            if expected_regions:
                brain_region_suggestions = {
                    "primary_rois": expected_regions.get("primary_regions", []),
                    "secondary_rois": expected_regions.get("secondary_regions", []),
                    "primary_rationale": expected_regions.get("region_rationale", "基于疾病先验知识"),
                    "analysis_priority": "hypothesis_driven" if disease_type else "exploratory",
                    "literature_support": "来自问题解析阶段的预测"
                }

        tracker.add_step_artifact("02_search_knowledge", "citations.json", citations)
        tracker.add_step_artifact("02_search_knowledge", "local_evidence.json", local_evidence)
        tracker.add_step_artifact("02_search_knowledge", "brain_region_suggestions.json", brain_region_suggestions)
        tracker.finish_step("02_search_knowledge", outputs={
            "pubmed_articles": len(citations),
            "local_papers": len(local_evidence),
            "notes": len(methodology_notes),
            "brain_regions": len(brain_region_suggestions.get("primary_rois", []))
        })

        print(f"  [PubMed] 找到 {len(citations)} 篇相关文献")
        print(f"  [本地] 找到 {len(local_evidence)} 条本地文献证据")
        print(f"  提取 {len(methodology_notes)} 条方法学建议")
        print(f"  生成脑区建议: {len(brain_region_suggestions.get('primary_rois', []))} 个主要ROI")

        # 返回更新字典
        return merge_state_updates({
            "citations": citations,
            "local_evidence": local_evidence,  # 新增：本地文献证据
            "methodology_notes": methodology_notes,
            "brain_region_suggestions": brain_region_suggestions,
            "evidence": json.dumps({
                "pubmed_articles": citations,
                "local_papers": local_evidence,
                "notes": methodology_notes,
                "brain_regions": brain_region_suggestions
            }, ensure_ascii=False, default=str),
            "node_history": ["search_knowledge"]
        })

    except Exception as e:
        tracker.finish_step("02_search_knowledge", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"search_knowledge: {e}"]
        })


def node_generate_plan(state: AgentState) -> Dict[str, Any]:
    """
    生成研究计划 - 使用Planning阶段的专业提示词
    属于ReAct模式的Planning阶段
    支持迭代模式：根据评估反馈调整研究计划
    支持恢复模式：如果已有plan且非迭代，则跳过

    Returns:
        包含更新字段的字典
    """
    iteration_count = state.get("iteration_count", 0)
    is_iteration = iteration_count > 0

    # 用于收集迭代准备阶段产生的额外更新
    iteration_updates = {}

    if is_iteration:
        print(f"\n[NODE: generate_plan] [ITERATION {iteration_count}] 根据反馈重新规划...")
        # ========== 迭代开始时压缩历史数据 ==========
        # 注意：_prepare_state_for_iteration 返回更新字典，而非修改原 state
        iteration_updates = _prepare_state_for_iteration(state)
    else:
        print(f"\n[NODE: generate_plan] 制定详细研究方案...")

        # 检查是否已经生成过计划（恢复模式，仅在非迭代时）
        if state.get("plan"):
            print(f"  [恢复模式] 检测到已有研究计划，跳过重新生成")
            return merge_state_updates({
                "node_history": ["generate_plan:skipped"]
            })

    llm = _get_llm()
    tracker = _get_tracker()
    registry = _get_registry()
    data_structure = _get_data_structure()

    intent = state.get("parsed_intent", {})
    evidence = state.get("evidence", "")

    # ========== 工具知识图谱增强 (使用动态知识图谱) ==========
    from src.knowledge.tool_knowledge_graph import enhance_plan_with_knowledge_graph
    research_question = state.get("question", "")
    # Fix: Extract disease from disease_info structure
    disease_info = intent.get("disease_info", {})
    disease_context = disease_info.get("disease_type") if disease_info else None

    kg_enhancement = enhance_plan_with_knowledge_graph(
        query=research_question,
        disease=disease_context,
        parsed_intent=intent
    )

    # 打印知识图谱推荐信息
    if kg_enhancement.get("suggested_tools"):
        print(f"  [知识图谱] 推荐工具: {', '.join(kg_enhancement['suggested_tools'][:3])}")
        if len(kg_enhancement['suggested_tools']) > 3:
            print(f"              (共 {len(kg_enhancement['suggested_tools'])} 个)")

    if kg_enhancement.get("roi_suggestions") and kg_enhancement["roi_suggestions"].get("primary"):
        primary_rois = kg_enhancement["roi_suggestions"]["primary"]
        print(f"  [知识图谱] 推荐ROI: {', '.join(primary_rois[:3])}")
        if len(primary_rois) > 3:
            print(f"              (共 {len(primary_rois)} 个)")

    kg_confidence = kg_enhancement.get("kg_confidence", 0.0)
    print(f"  [知识图谱] 置信度: {kg_confidence:.2f}")
    print(f"  [知识图谱] 来源: {kg_enhancement.get('source', 'unknown')}")

    # ========== Uq: Query-driven update ==========
    # 如果研究问题中明确提到了某些ROI，更新动态知识图谱
    if disease_context and not is_iteration:
        try:
            from src.knowledge.dynamic_knowledge_graph import get_dynamic_kg
            dynamic_kg = get_dynamic_kg()

            # 从brain_region_suggestions中提取用户关注的ROI
            brain_region_suggestions = state.get("brain_region_suggestions", {})
            selected_rois = brain_region_suggestions.get("primary_rois", [])

            if selected_rois:
                dynamic_kg.update_query_driven(
                    query=research_question,
                    disease=disease_context,
                    selected_rois=selected_rois
                )
                print(f"  [Uq更新] 基于研究问题更新了 {len(selected_rois)} 个ROI权重")
        except Exception as e:
            print(f"  [Uq更新] 失败: {e}")

    # ========== Pipeline 复合技能推荐 ==========
    pipeline_hint = ""
    if not is_iteration:
        try:
            pipeline_match = _get_tool_skill().recommend_pipeline(
                disease=disease_context,
                question=research_question
            )
            if pipeline_match and pipeline_match.get("confidence", 0) >= 0.5:
                chain_str = " → ".join(t["tool"] for t in pipeline_match["tool_chain"])
                pipeline_hint = (
                    f"\n## 历史成功Pipeline参考（置信度: {pipeline_match['confidence']:.2f}, "
                    f"成功{pipeline_match['success_count']}次）\n"
                    f"工具链: {chain_str}\n"
                    f"研究类型: {pipeline_match.get('research_type', '')}\n"
                )
                print(f"  [Pipeline推荐] {chain_str} (置信度: {pipeline_match['confidence']:.2f})")
        except Exception:
            pass

    tracker.start_step("03_generate_plan", {
        "intent_type": intent.get("research_type", ""),
        "is_iteration": is_iteration,
        "iteration_count": iteration_count
    })

    # 获取工具列表
    tools = registry.get_tool_descriptions()
    tools_simple = [
        {"name": t["name"], "description": t["description"], "modalities": t["modalities"]}
        for t in tools
    ]

    # 构建提示词 - 区分首次规划和迭代调整
    if is_iteration:
        # 迭代模式：提供之前的结果和反馈
        previous_plan = state.get("plan", {})
        iteration_feedback = state.get("iteration_feedback", "")
        iteration_suggestions = state.get("iteration_suggestions", [])

        # 使用压缩后的摘要而非原始数据（已在_prepare_state_for_iteration中压缩）
        previous_results_summary = state.get("tool_results_summary", "")
        if not previous_results_summary:
            # 如果没有摘要，使用压缩后的tool_results
            previous_results_summary = _compress_tool_results_with_llm(state.get("tool_results", []))

        # 添加已处理数据信息（避免重复处理）
        existing_data = state.get("existing_processed_data", {})
        if existing_data and existing_data.get("summary"):
            previous_results_summary = f"""## 已存在的处理结果（不需要重复处理！）
{existing_data['summary']}

## 上一轮分析摘要
{previous_results_summary}"""
            print(f"  [迭代规划] 已检测到已处理数据，将避免重复处理")

        # 使用迭代规划提示词
        prompts = get_node_prompt(
            "generate_plan_iteration",
            intent=intent,
            previous_plan=previous_plan,
            previous_results=previous_results_summary,  # 使用压缩摘要+已处理数据信息
            iteration_feedback=iteration_feedback,
            iteration_suggestions=iteration_suggestions,
            tools=tools_simple,
            data=data_structure,
            iteration_count=iteration_count
        )
    else:
        # 首次规划 - 传入脑区建议
        brain_region_suggestions = state.get("brain_region_suggestions", {})
        prompts = get_node_prompt(
            "generate_plan",
            intent=intent,
            evidence=evidence,
            tools=tools_simple,
            data=data_structure,
            brain_region_suggestions=brain_region_suggestions
        )

    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]}
    ]

    # 将 pipeline 推荐信息追加到 user prompt（如果有）
    if pipeline_hint:
        messages[-1]["content"] += pipeline_hint

    # 设置任务类型以使用K2-Thinking推理模型
    if is_iteration:
        llm.set_task_type("planning_iteration")
    else:
        llm.set_task_type("planning")

    try:
        # 使用Planner角色生成/调整计划（规划任务需要更大的token限制）
        # 增加到65536(64K)以避免复杂计划被截断
        plan = llm.generate_json(messages, max_tokens=65536)

        tracker.add_step_artifact("03_generate_plan", "plan.json", plan)
        tracker.finish_step("03_generate_plan", outputs={"title": plan.get("title", "")})

        print(f"  计划标题: {plan.get('title')}")

        # === 处理增量计划：迭代时LLM只返回new_steps ===
        if is_iteration and "new_steps" in plan:
            new_steps = plan.get("new_steps", [])
            modified_steps = plan.get("modified_steps", [])

            # 将new_steps设置为pipeline，由task_manager处理
            plan["pipeline"] = new_steps

            print(f"\n  [增量规划] 第 {iteration_count} 次迭代:")
            print(f"    - 新增步骤: {len(new_steps)} 个")
            if modified_steps:
                print(f"    - 修改建议: {len(modified_steps)} 个")
                for mod in modified_steps:
                    print(f"      · {mod.get('original_step', 'unknown')}: {mod.get('modification', '')[:50]}...")

            # 显示新步骤详情
            for i, step in enumerate(new_steps, 1):
                step_name = step.get("step", step.get("description", "unknown"))
                tool_name = step.get("tool", "unknown")
                print(f"    [{i}] {step_name} ({tool_name})")
        else:
            print(f"  分析步骤: {len(plan.get('pipeline', []))} 步")

        if plan.get('potential_issues'):
            print(f"  潜在问题: {len(plan.get('potential_issues', []))} 个")

        # 显示ROI选择策略
        roi_selection = plan.get('roi_selection', {})
        if roi_selection:
            print(f"\n  [ROI选择策略]")
            print(f"    分析策略: {roi_selection.get('strategy', 'exploratory')}")
            primary_rois = roi_selection.get('primary_rois', [])
            if primary_rois:
                print(f"    Primary ROIs: {primary_rois[:5]}{'...' if len(primary_rois) > 5 else ''}")
            secondary_rois = roi_selection.get('secondary_rois', [])
            if secondary_rois:
                print(f"    Secondary ROIs: {secondary_rois[:3]}{'...' if len(secondary_rois) > 3 else ''}")
            if roi_selection.get('expected_findings'):
                print(f"    预期发现: {roi_selection.get('expected_findings')[:100]}...")

        # 创建任务列表（避免token超限）
        from src.agent.task_manager import TaskManager
        task_manager = TaskManager(tracker.run_dir)

        # 获取前一迭代的成功任务结果（用于继承）
        previous_task_results = state.get("previous_successful_results", {})
        if previous_task_results and iteration_count > 0:
            print(f"\n  [任务继承] 发现 {len(previous_task_results)} 个可继承的成功任务结果")

        task_manager.create_from_plan(
            plan,
            iteration_count=iteration_count,
            previous_task_results=previous_task_results
        )

        # 显示任务命名
        if iteration_count > 0:
            print(f"\n  [任务列表] 第 {iteration_count} 次迭代：已创建 {len(task_manager.tasks)} 个任务")
        else:
            print(f"\n  [任务列表] 已创建 {len(task_manager.tasks)} 个任务")
        print(task_manager.get_summary())

        # ========== MoER: PlanReviewer 审查 ==========
        plan_review_result = {}
        moer_plan_reviews = []
        try:
            from src.agent.moer import MoERReviewer
            moer = MoERReviewer(llm_client=llm)
            available_tool_names = [t["name"] for t in tools_simple]
            plan_review_result = moer.review_plan(
                plan, intent, available_tool_names, evidence
            )
            moer_plan_reviews = [plan_review_result]
        except Exception as moer_err:
            print(f"  [MoER] PlanReviewer 审查失败: {moer_err}")
            moer_plan_reviews = [{
                "reviewer": "PlanReviewer",
                "status": "error",
                "error": str(moer_err),
                "timestamp": datetime.now().isoformat()
            }]

        # MoER 审查摘要（详细信息由 plan_review_gate 节点的 interrupt 展示）
        if plan_review_result.get("status") == "needs_revision":
            score = plan_review_result.get("score", 0)
            n_issues = len(plan_review_result.get("issues", []))
            print(f"  [MoER:PlanReviewer] 审查评分: {score}/100, 问题: {n_issues} → 将在审查门控节点暂停")

        # 返回更新字典（合并迭代准备阶段的更新）
        return merge_state_updates(iteration_updates, {
            "plan": plan,
            "kg_enhancement": kg_enhancement,
            "plan_review": plan_review_result,
            "moer_reviews": moer_plan_reviews,
            "has_task_list": True,
            "node_history": ["generate_plan"]
        })

    except Exception as e:
        tracker.finish_step("03_generate_plan", success=False, error=str(e))
        print(f"  [错误] 计划生成失败: {e}")
        return merge_state_updates(iteration_updates, {
            "last_error": str(e),
            "error_history": [f"generate_plan: {e}"]
        })


def node_plan_review_gate(state: AgentState) -> Dict[str, Any]:
    """
    计划审查门控节点 — 检查 MoER 审查结果，必要时暂停等待用户决策。
    独立节点确保 interrupt resume 时不会重新执行计划生成（LLM调用）。
    """
    plan_review = state.get("plan_review", {})

    # 审查通过或无审查结果 → 直接放行
    if plan_review.get("status") != "needs_revision":
        return {"node_history": ["plan_review_gate:passed"]}

    # 需要修订 → 触发 interrupt 暂停等待用户决策
    from langgraph.types import interrupt
    score = plan_review.get("score", 0)
    interrupt_data = {
        "type": "plan_review",
        "score": score,
        "issues": plan_review.get("issues", []),
        "suggestions": plan_review.get("suggestions", [])
    }
    user_input = interrupt(interrupt_data)
    action = user_input.get("action", "continue") if isinstance(user_input, dict) else "continue"

    if action == "abort":
        return {
            "last_error": "用户中止：MoER计划审查未通过",
            "plan": {},
            "node_history": ["plan_review_gate:aborted"]
        }
    elif action == "fix":
        return {
            "plan": {},
            "last_error": f"MoER计划审查需要修订 (评分: {score}/100)",
            "node_history": ["plan_review_gate:fix_requested"]
        }

    # action == "continue" → 保持计划不变，继续执行
    return {"node_history": ["plan_review_gate:continued"]}


def node_map_data_fields(state: AgentState) -> Dict[str, Any]:
    """
    映射数据字段

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: map_data_fields] 映射数据字段...")

    tracker = _get_tracker()
    plan = state.get("plan", {})
    data_structure = _get_data_structure()

    tracker.start_step("04_map_data_fields", {})

    try:
        field_mapping = {
            "available_groups": list(data_structure.get("groups", {}).keys()),
            "available_modalities": data_structure.get("modalities", []),
            "required_groups": plan.get("design", {}).get("groups", []),
            "required_modalities": plan.get("modalities", [])
        }

        tracker.add_step_artifact("04_map_data_fields", "field_mapping.json", field_mapping)
        tracker.finish_step("04_map_data_fields", outputs=field_mapping)

        print(f"  可用组别: {field_mapping['available_groups']}")
        print(f"  需要模态: {field_mapping['required_modalities']}")

        return merge_state_updates({
            "field_mapping": field_mapping,
            "phase": ResearchPhase.DATA_PREPARATION.value,
            "node_history": ["map_data_fields"]
        })

    except Exception as e:
        tracker.finish_step("04_map_data_fields", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"map_data_fields: {e}"]
        })


def node_build_cohort(state: AgentState) -> Dict[str, Any]:
    """
    构建研究队列
    支持恢复模式：如果已有cohort，则跳过

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: build_cohort] 构建研究队列...")

    # 检查是否已经构建过队列（恢复模式）
    if state.get("cohort"):
        print(f"  [恢复模式] 检测到已有研究队列，跳过重新构建")
        return merge_state_updates({
            "node_history": ["build_cohort:skipped"]
        })

    tracker = _get_tracker()
    data_structure = _get_data_structure()
    field_mapping = state.get("field_mapping", {})

    tracker.start_step("05_build_cohort", {})

    try:
        groups = data_structure.get("groups", {})
        cohort = {
            "groups": {},
            "total_subjects": 0
        }

        for group_name, group_info in groups.items():
            # 使用实际扫描到的subject ID，而不是生成新的ID
            actual_subjects = group_info.get("subjects", [])

            # 去重检查
            unique_subjects = list(dict.fromkeys(actual_subjects))  # 保持顺序的去重
            if len(unique_subjects) != len(actual_subjects):
                print(f"  [警告] {group_name}组发现重复被试: {len(actual_subjects)} -> {len(unique_subjects)}")
                actual_subjects = unique_subjects

            cohort["groups"][group_name] = {
                "n": len(actual_subjects),  # 使用去重后的数量
                "subjects": actual_subjects
            }
            cohort["total_subjects"] += len(actual_subjects)

            # 打印完整被试列表（用于调试）
            print(f"  {group_name}: n={len(actual_subjects)}")
            print(f"    被试ID: {actual_subjects}")

        # 添加人口统计学数据到cohort（如果存在）
        if data_structure.get("demographics"):
            cohort["demographics"] = data_structure["demographics"]
            print(f"  [人口统计学] 已添加 {data_structure['demographics'].get('n_subjects', 0)} 个被试的人口统计学数据")

        tracker.add_step_artifact("05_build_cohort", "cohort.json", cohort)
        tracker.finish_step("05_build_cohort", outputs={"total_n": cohort["total_subjects"]})

        print(f"  总样本量: {cohort['total_subjects']}")

        return merge_state_updates({
            "cohort": cohort,
            "node_history": ["build_cohort"]
        })

    except Exception as e:
        tracker.finish_step("05_build_cohort", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"build_cohort: {e}"]
        })


def node_materialize_data(state: AgentState) -> Dict[str, Any]:
    """
    物化数据集

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: materialize_data] 物化数据集...")

    tracker = _get_tracker()
    cohort = state.get("cohort", {})

    tracker.start_step("06_materialize_data", {})

    try:
        manifest = {
            "subjects": [],
            "data_paths": {},
            "available_modalities": {}  # 记录每个subject的可用模态
        }

        # 收集所有被试ID（使用set去重）
        seen_subjects = set()
        duplicate_count = 0

        for group_name, group_data in cohort.get("groups", {}).items():
            for subject_id in group_data.get("subjects", []):
                if subject_id in seen_subjects:
                    print(f"  [警告] 发现重复被试: {subject_id} (已跳过)")
                    duplicate_count += 1
                    continue

                seen_subjects.add(subject_id)
                manifest["subjects"].append({
                    "id": subject_id,
                    "group": group_name
                })

                # 记录该subject的所有可用模态
                subject_modalities = {}
                for modality in ["anat", "dwi", "func"]:
                    modality_dir = DATA_DIR / group_name / modality / subject_id
                    if modality_dir.exists():
                        # 检查目录是否包含DICOM或NIfTI文件
                        dcm_files = list(modality_dir.glob("*.dcm"))
                        nii_files = list(modality_dir.glob("*.nii*"))
                        if dcm_files or nii_files:
                            subject_modalities[modality] = str(modality_dir)

                if subject_modalities:
                    manifest["available_modalities"][subject_id] = subject_modalities
                    manifest["data_paths"][subject_id] = subject_modalities

        if duplicate_count > 0:
            print(f"  [去重] 检测到 {duplicate_count} 个重复被试，已自动去重")

        # 打印模态统计
        modality_stats = {"anat": 0, "dwi": 0, "func": 0}
        for subj_modalities in manifest["available_modalities"].values():
            for mod in subj_modalities.keys():
                modality_stats[mod] = modality_stats.get(mod, 0) + 1
        print(f"  模态统计: anat={modality_stats['anat']}, dwi={modality_stats['dwi']}, func={modality_stats['func']}")

        # 打印实际被试列表（用于调试）
        actual_subject_ids = [s["id"] for s in manifest["subjects"]]
        print(f"  实际被试ID: {actual_subject_ids[:10]}{'...' if len(actual_subject_ids) > 10 else ''}")

        tracker.add_step_artifact("06_materialize_data", "data_manifest.json", manifest)
        tracker.finish_step("06_materialize_data", outputs={"subjects": len(manifest["subjects"])})

        print(f"  物化被试数: {len(manifest['subjects'])} (去重后)")

        return merge_state_updates({
            "data_manifest": manifest,
            "node_history": ["materialize_data"]
        })

    except Exception as e:
        tracker.finish_step("06_materialize_data", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"materialize_data: {e}"]
        })


def node_quality_control(state: AgentState) -> Dict[str, Any]:
    """
    数据质控节点 - 在 materialize_data 之后检查数据质量

    检查内容:
    1. 文件完整性（NIfTI 可读、维度合理）
    2. 模态一致性（所有被试是否都有所需模态）
    3. DWI 头动（如果有 eddy 输出）
    4. 结构像体素大小（0.5-3mm）

    QC 策略：只检查数据是否损坏，不限制样本量。
    损坏的被试数据从 manifest 中排除，只要还有可用被试即可继续。
    """
    print(f"\n[NODE: quality_control] 数据质控检查...")

    tracker = _get_tracker()
    tracker.start_step("06b_quality_control", {})

    manifest = state.get("data_manifest", {})
    subjects = manifest.get("subjects", [])
    data_paths = manifest.get("data_paths", {})

    checks = {}
    warnings = []
    failed_subjects = []
    excluded_subjects = []

    # === 1. 文件完整性检查 ===
    file_integrity = {"passed": 0, "failed": 0, "details": []}
    for subj_info in subjects:
        subj_id = subj_info["id"]
        subj_paths = data_paths.get(subj_id, {})
        subj_ok = True

        for modality, mod_path in subj_paths.items():
            mod_dir = Path(mod_path)
            if not mod_dir.exists():
                file_integrity["details"].append(f"{subj_id}/{modality}: 目录不存在")
                subj_ok = False
                continue

            nii_files = list(mod_dir.glob("*.nii")) + list(mod_dir.glob("*.nii.gz"))
            if nii_files:
                for nf in nii_files:
                    if nf.stat().st_size == 0:
                        file_integrity["details"].append(f"{subj_id}/{modality}: {nf.name} 文件为空")
                        subj_ok = False

                # 尝试用 nibabel 检查 header
                try:
                    import nibabel as nib
                    img = nib.load(str(nii_files[0]))
                    ndim = len(img.shape)
                    if ndim < 3 or ndim > 4:
                        file_integrity["details"].append(
                            f"{subj_id}/{modality}: 维度异常 ({ndim}D)")
                        subj_ok = False
                except ImportError:
                    pass  # nibabel 不可用，跳过深度检查
                except Exception as e:
                    file_integrity["details"].append(
                        f"{subj_id}/{modality}: NIfTI 读取失败 ({e})")
                    subj_ok = False

        if subj_ok:
            file_integrity["passed"] += 1
        else:
            file_integrity["failed"] += 1
            failed_subjects.append(subj_id)

    checks["file_integrity"] = file_integrity
    print(f"  [QC] 文件完整性: 通过 {file_integrity['passed']}, 失败 {file_integrity['failed']}")

    # === 2. 样本量统计（仅记录，不作为通过/失败判据）===
    group_counts = {}
    for subj_info in subjects:
        g = subj_info["group"]
        group_counts[g] = group_counts.get(g, 0) + 1
    checks["sample_size"] = {"group_counts": group_counts}
    print(f"  [QC] 样本量: 组别 {group_counts}")

    # === 3. 模态一致性检查 ===
    plan = state.get("plan", {})
    required_modalities = set()
    for step in plan.get("steps", []):
        for mod in step.get("modalities", []):
            required_modalities.add(mod)
    if not required_modalities:
        # 从 manifest 推断
        all_mods = manifest.get("available_modalities", {})
        for subj_mods in all_mods.values():
            required_modalities.update(subj_mods.keys())

    modality_check = {"passed": 0, "missing": []}
    for subj_info in subjects:
        subj_id = subj_info["id"]
        subj_mods = set(data_paths.get(subj_id, {}).keys())
        missing = required_modalities - subj_mods
        if missing:
            modality_check["missing"].append({"subject": subj_id, "missing": list(missing)})
        else:
            modality_check["passed"] += 1

    checks["modality_consistency"] = modality_check
    print(f"  [QC] 模态一致性: 完整 {modality_check['passed']}, 缺失 {len(modality_check['missing'])}")

    # === 4. DWI 头动检查（如果有 eddy 输出）===
    motion_check = {"checked": 0, "high_motion": 0, "details": []}
    try:
        from src.tools.local_tools import read_eddy_motion_parameters
        run_dir = Path(tracker.run_dir) if tracker.run_dir else None
        if run_dir:
            for subj_info in subjects:
                subj_id = subj_info["id"]
                eddy_basename = run_dir / f"{subj_id}_eddy"
                if Path(f"{eddy_basename}.eddy_movement_rms").exists():
                    try:
                        qc_data = read_eddy_motion_parameters(str(eddy_basename))
                        motion_check["checked"] += 1
                        if qc_data.get("max_movement", 0) > 2.0:
                            motion_check["high_motion"] += 1
                            motion_check["details"].append(
                                f"{subj_id}: max_movement={qc_data['max_movement']:.2f}mm")
                            if subj_id not in failed_subjects:
                                failed_subjects.append(subj_id)
                    except Exception:
                        pass
    except ImportError:
        pass

    checks["motion_qc"] = motion_check
    if motion_check["checked"] > 0:
        print(f"  [QC] DWI头动: 检查 {motion_check['checked']}, 高运动 {motion_check['high_motion']}")

    # === 5. 结构像体素大小检查 ===
    voxel_check = {"checked": 0, "abnormal": 0, "details": []}
    try:
        import nibabel as nib
        for subj_info in subjects[:10]:  # 抽样检查前10个
            subj_id = subj_info["id"]
            anat_path = data_paths.get(subj_id, {}).get("anat")
            if anat_path:
                nii_files = list(Path(anat_path).glob("*.nii")) + list(Path(anat_path).glob("*.nii.gz"))
                if nii_files:
                    try:
                        img = nib.load(str(nii_files[0]))
                        voxel_sizes = img.header.get_zooms()[:3]
                        voxel_check["checked"] += 1
                        for vs in voxel_sizes:
                            if vs < 0.5 or vs > 3.0:
                                voxel_check["abnormal"] += 1
                                voxel_check["details"].append(
                                    f"{subj_id}: voxel={voxel_sizes}")
                                break
                    except Exception:
                        pass
    except ImportError:
        pass

    checks["voxel_size"] = voxel_check
    if voxel_check["checked"] > 0:
        print(f"  [QC] 体素大小: 检查 {voxel_check['checked']}, 异常 {voxel_check['abnormal']}")

    # === 汇总 ===
    # 排除数据损坏的被试（文件不可读 + 高运动）
    excluded_subjects = list(set(failed_subjects))

    total = len(subjects)
    passed_count = total - len(excluded_subjects)
    # QC 通过条件：只要还有可用被试就继续，不限制样本量
    qc_passed = passed_count > 0

    qc_summary = f"{total}个被试中{passed_count}个通过QC"
    if excluded_subjects:
        qc_summary += f"，{len(excluded_subjects)}个数据损坏被排除"
    if warnings:
        qc_summary += f"，{len(warnings)}个警告"

    # 计算排除后各组剩余
    remaining_by_group = dict(group_counts)
    for subj_info in subjects:
        if subj_info["id"] in excluded_subjects:
            g = subj_info["group"]
            remaining_by_group[g] = remaining_by_group.get(g, 0) - 1

    qc_results = {
        "total_subjects": total,
        "passed_subjects": passed_count,
        "failed_subjects": excluded_subjects,
        "warnings": warnings,
        "checks": checks,
        "excluded_subjects": excluded_subjects,
        "remaining_by_group": remaining_by_group,
        "qc_summary": qc_summary
    }

    print(f"  [QC] 总结: {qc_summary}")
    if excluded_subjects:
        print(f"  [QC] 排除后各组: {remaining_by_group}")
    print(f"  [QC] 整体通过: {qc_passed}")

    tracker.add_step_artifact("06b_quality_control", "qc_results.json", qc_results)
    tracker.finish_step("06b_quality_control", outputs={"qc_passed": qc_passed})

    result = {
        "qc_results": qc_results,
        "qc_passed": qc_passed,
        "node_history": ["quality_control"]
    }

    # 如果有被排除的被试，更新 data_manifest 以便下游节点跳过它们
    if excluded_subjects and qc_passed:
        updated_manifest = dict(manifest)
        updated_manifest["subjects"] = [s for s in subjects if s["id"] not in excluded_subjects]
        updated_manifest["excluded_by_qc"] = excluded_subjects
        # 同步更新 data_paths
        updated_paths = {k: v for k, v in data_paths.items() if k not in excluded_subjects}
        updated_manifest["data_paths"] = updated_paths
        result["data_manifest"] = updated_manifest
        print(f"  [QC] 已从 manifest 中排除 {len(excluded_subjects)} 个被试")

    if not qc_passed:
        result["last_error"] = f"数据质控未通过: 所有被试数据均损坏，无可用数据"
        result["error_history"] = [f"quality_control: {qc_summary}"]

    return merge_state_updates(result)


def node_select_tools(state: AgentState) -> Dict[str, Any]:
    """
    选择分析工具

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: select_tools] 选择分析工具...")

    tracker = _get_tracker()
    registry = _get_registry()
    plan = state.get("plan", {})

    tracker.start_step("07_select_tools", {})

    try:
        pipeline = plan.get("pipeline", [])
        tool_chain = []

        for step in pipeline:
            tool_name = step.get("tool", "")
            if tool_name:
                tool_chain.append({
                    "step": step.get("step", ""),
                    "tool": tool_name,
                    "params": step.get("parameters", {}),
                    "status": "pending"
                })

        # 如果pipeline中没有指定工具，使用默认工具链
        if not tool_chain:
            # 默认VBM分析工具链（带必需参数）
            default_chain = [
                {"step": "VBM分割", "tool": "spm_analysis", "params": {"analysis_type": "vbm_segment"}, "status": "pending"},
                {"step": "统计分析", "tool": "python_stats", "params": {"analysis_type": "ttest"}, "status": "pending"}
            ]
            tool_chain = default_chain
            print(f"  [使用默认工具链]")

        # ========== 工具知识图谱验证 ==========
        tool_skill = _get_tool_skill()
        tool_names = [t['tool'] for t in tool_chain]

        # 验证并自动修复工具顺序
        validation_result = tool_skill.validate_and_fix(tool_names)

        if validation_result["fixed"]:
            print(f"  [知识图谱] 工具顺序已调整: {validation_result['original_tools']} -> {validation_result['fixed_tools']}")
            # 重新构建tool_chain以匹配新顺序
            tool_map = {t['tool']: t for t in tool_chain}
            tool_chain = [tool_map[tool] for tool in validation_result['fixed_tools'] if tool in tool_map]

        # 打印验证结果
        tool_skill.print_validation(validation_result)

        # 获取工具详细信息
        tool_details = tool_skill.get_tool_details([t['tool'] for t in tool_chain])
        if tool_details:
            print(f"  [知识图谱] 已加载 {len(tool_details)} 个工具的详细信息")

        # ========== 技能匹配：推荐参数 + 过程性知识 ==========
        try:
            disease = plan.get("disease", "") or plan.get("disease_context", "")
            all_modalities = plan.get("modalities", ["anat"])
            task_desc = state.get("question", "")
            for tool_item in tool_chain:
                # 根据工具的模态匹配推荐参数
                tool_info = tool_details.get(tool_item["tool"], {})
                tool_modality = tool_info.get("modality", all_modalities[0] if all_modalities else "anat")
                recommended = tool_skill.recommend_tool_params(
                    tool=tool_item["tool"], disease=disease,
                    modality=tool_modality, task_description=task_desc
                )
                if recommended:
                    # 兼容新格式(含过程性知识)和旧格式(纯参数)
                    if isinstance(recommended, dict) and "parameters" in recommended:
                        params_dict = recommended["parameters"]
                        # 将过程性知识存入 metadata
                        meta = tool_item.setdefault("metadata", {})
                        if recommended.get("tips"):
                            meta["skill_tips"] = recommended["tips"][:3]
                        if recommended.get("error_patterns"):
                            meta["known_errors"] = recommended["error_patterns"][:3]
                        if recommended.get("procedure"):
                            meta["procedure"] = recommended["procedure"]
                    else:
                        params_dict = recommended
                    for k, v in params_dict.items():
                        if k not in tool_item["params"]:
                            tool_item["params"][k] = v
                    print(f"  [技能匹配] {tool_item['tool']}({tool_modality}): 推荐 {len(params_dict)} 个参数")
        except Exception as skill_err:
            print(f"  [技能匹配] 警告: {skill_err}")

        # ========== 跨模态对齐验证 ==========
        modalities = plan.get("modalities", [])
        if len(modalities) > 1:
            has_alignment = any(
                t.get("params", {}).get("analysis_type") in ("coregister", "normalize")
                for t in tool_chain
            )
            if not has_alignment:
                print(f"  [警告] 多模态分析 ({modalities}) 但未包含对齐步骤，结果可能不可靠")

        tracker.add_step_artifact("07_select_tools", "tool_chain.json", tool_chain)
        tracker.finish_step("07_select_tools", outputs={"tools": len(tool_chain)})

        print(f"  选择工具: {[t['tool'] for t in tool_chain]}")

        return merge_state_updates({
            "tool_chain": tool_chain,
            "current_tool_index": 0,
            "tool_details": tool_details,
            "tool_validation": validation_result,
            "tool_results_iteration_offset": len(state.get("tool_results", [])),
            "phase": ResearchPhase.ANALYSIS.value,
            "node_history": ["select_tools"]
        })

    except Exception as e:
        tracker.finish_step("07_select_tools", success=False, error=str(e))
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"select_tools: {e}"]
        })


def _scan_input_files(subjects: List[Dict], modality: str, layout: str = "grouped") -> Dict[str, List[str]]:
    """
    扫描被试目录，分类NIfTI文件和DICOM目录
    优先使用已转换的NIfTI文件，避免重复转换

    Args:
        subjects: 被试列表 [{"id": "sub-001", "group": "HC"}, ...]
        modality: 影像模态 (anat/dwi/func)
        layout: 数据布局类型 ("grouped" | "bids" | "flat")

    Returns:
        字典 {"nifti_files": [...], "dicom_dirs": [...], "subject_mapping": {...}}
    """
    nifti_files = []
    dicom_dirs = []
    subject_mapping = {}  # subject_id -> file_path

    # 转换输出目录
    converted_dir = OUTPUT_DIR / "dicom_converted" / modality

    for subject in subjects:
        subject_id = subject.get("id", "")
        group = subject.get("group", "")

        # 1. 首先检查是否已有转换后的NIfTI文件（避免重复转换）
        if converted_dir.exists():
            converted_subject_dir = converted_dir / subject_id
            if converted_subject_dir.exists():
                converted_nii = list(converted_subject_dir.glob("*.nii")) + \
                               list(converted_subject_dir.glob("*.nii.gz"))
                if converted_nii:
                    nifti_files.append(str(converted_nii[0]))
                    subject_mapping[subject_id] = str(converted_nii[0])
                    print(f"  [已转换] {subject_id}: {converted_nii[0].name}")
                    continue  # 跳过后续检查

        # 2. 根据布局构建原始数据路径
        if layout == "bids":
            subject_dir = DATA_DIR / subject_id / modality
        elif layout == "flat":
            subject_dir = DATA_DIR
        else:  # grouped
            subject_dir = DATA_DIR / group / modality / subject_id

        if not subject_dir.exists():
            print(f"  警告: 被试目录不存在: {subject_dir}")
            continue

        # 3. 在原始目录查找NIfTI文件
        nii_files = list(subject_dir.glob("*.nii")) + list(subject_dir.glob("*.nii.gz"))

        if nii_files:
            # 使用第一个找到的NIfTI文件
            nifti_files.append(str(nii_files[0]))
            subject_mapping[subject_id] = str(nii_files[0])
            print(f"  [NIfTI] {subject_id}: {nii_files[0].name}")
        else:
            # 4. 如果没有NIfTI，查找DICOM文件夹（需要转换）
            dcm_files = list(subject_dir.glob("*.dcm"))
            if dcm_files:
                dicom_dirs.append(str(subject_dir))
                subject_mapping[subject_id] = str(subject_dir)
                print(f"  [需转换] {subject_id}: {len(dcm_files)} 个DICOM文件")
            else:
                # 可能是没有.dcm后缀的DICOM文件
                all_files = [f for f in subject_dir.iterdir() if f.is_file()]
                if all_files:
                    dicom_dirs.append(str(subject_dir))
                    subject_mapping[subject_id] = str(subject_dir)
                    print(f"  [需转换?] {subject_id}: {len(all_files)} 个文件（无扩展名）")
                else:
                    print(f"  警告: 未找到影像文件: {subject_dir}")

    return {
        "nifti_files": nifti_files,
        "dicom_dirs": dicom_dirs,
        "subject_mapping": subject_mapping
    }


def _convert_dicom_to_nifti(dicom_dirs: List[str], output_dir: str) -> List[str]:
    """
    将DICOM目录转换为NIfTI文件

    Args:
        dicom_dirs: DICOM目录列表
        output_dir: 输出目录

    Returns:
        转换后的NIfTI文件路径列表
    """
    from src.tools.local_tools import convert_dicom_to_nifti

    print(f"\n  [DICOM转换] 正在转换 {len(dicom_dirs)} 个DICOM目录...")

    result = convert_dicom_to_nifti(dicom_dirs, output_dir)

    if result and result.get("status") == "succeeded":
        output_files = result.get("output_files", [])
        print(f"  [DICOM转换] 成功转换 {len(output_files)} 个文件")
        for log_entry in result.get("conversion_log", []):
            print(f"    {log_entry}")
        return output_files
    else:
        error = result.get('error', '未知错误') if result else "转换函数未返回结果"
        print(f"  [DICOM转换] 转换失败: {error}")
        if result:
            for log_entry in result.get("conversion_log", []):
                print(f"    {log_entry}")
        return []


def _build_input_files(subjects: List[Dict], modality: str, plan: Dict) -> List[str]:
    """
    根据被试列表和模态构建实际的输入文件路径
    自动检测并转换DICOM格式

    Args:
        subjects: 被试列表 [{"id": "sub-001", "group": "HC"}, ...]
        modality: 影像模态 (anat/dwi/func)
        plan: 研究计划

    Returns:
        输入文件路径列表（全部为NIfTI格式）
    """
    # 扫描文件
    scan_result = _scan_input_files(subjects, modality)
    nifti_files = scan_result.get("nifti_files", []) if scan_result else []
    dicom_dirs = scan_result.get("dicom_dirs", []) if scan_result else []

    print(f"  扫描结果: {len(nifti_files)} 个NIfTI文件, {len(dicom_dirs)} 个DICOM目录")

    # 如果有DICOM目录，需要先转换
    if dicom_dirs:
        # 创建转换输出目录
        convert_output_dir = OUTPUT_DIR / "dicom_converted" / modality
        convert_output_dir.mkdir(parents=True, exist_ok=True)

        converted_files = _convert_dicom_to_nifti(dicom_dirs, str(convert_output_dir))
        nifti_files.extend(converted_files)

    return nifti_files


def _find_file_in_results(filename: str, state: AgentState) -> Optional[str]:
    """
    在 tool_results 和 DATA_DIR 中查找文件

    Args:
        filename: 文件名（可以是相对路径）
        state: Agent状态

    Returns:
        找到的文件绝对路径，找不到返回 None
    """
    from pathlib import Path

    # 1. 在 tool_results 的输出目录中查找
    tool_results = state.get("tool_results", [])
    for result in reversed(tool_results):
        if result.get("status") != "succeeded":
            continue
        outputs = result.get("outputs", {})
        output_files = outputs.get("output_files", [])
        for f in output_files:
            f_path = Path(f)
            if f_path.name == filename or str(f_path).endswith(filename):
                if f_path.exists():
                    return str(f_path)
            # 在同目录下查找
            candidate = f_path.parent / filename
            if candidate.exists():
                return str(candidate.resolve())

    # 2. 在 DATA_DIR 中递归查找
    try:
        from src.config import DATA_DIR
        data_dir = Path(DATA_DIR)
        if data_dir.exists():
            matches = list(data_dir.rglob(filename))
            if matches:
                return str(matches[0].resolve())
    except Exception:
        pass

    # 3. 在 OUTPUT_DIR 中递归查找
    try:
        from src.config import OUTPUT_DIR
        output_dir = Path(OUTPUT_DIR)
        if output_dir.exists():
            matches = list(output_dir.rglob(filename))
            if matches:
                return str(matches[0].resolve())
    except Exception:
        pass

    return None


def _find_coregister_images(state: AgentState) -> Dict[str, Optional[str]]:
    """
    为coregistration自动查找参考图像和源图像

    标准fMRI配准流程（SPM推荐）：
    - reference_image: mean功能像（realign输出，固定不动）
    - source_image: T1结构像（anat数据，要配准到功能像空间）

    这样做的好处：
    1. 功能像保持原始空间，减少插值
    2. T1配准后可用于分割，得到变形场
    3. 用变形场将功能像标准化到MNI空间

    Returns:
        包含 'reference_image' 和 'source_image' 的字典
    """
    from pathlib import Path

    result = {
        "reference_image": None,  # mean功能像
        "source_image": None      # T1结构像
    }

    tool_results = state.get("tool_results", [])

    # ========== 1. 查找 mean 功能像（作为 reference） ==========
    for tr in reversed(tool_results):
        if tr.get("status") != "succeeded":
            continue

        outputs = tr.get("outputs", {})
        analysis_type = outputs.get("analysis_type", "")
        output_files = outputs.get("output_files", [])

        if analysis_type == "realign":
            for f in output_files:
                f_path = Path(f)
                if f_path.name.startswith("mean") and f_path.suffix == ".nii":
                    if f_path.exists():
                        result["reference_image"] = str(f_path)
                        print(f"  [自动查找] mean功能像(reference): {f_path.name}")
                        break
            if result["reference_image"]:
                break

    # ========== 2. 查找 T1 结构像（作为 source） ==========
    # 2.1 从 dicom_to_nifti 的 anat 输出查找
    for tr in reversed(tool_results):
        if tr.get("status") != "succeeded":
            continue

        outputs = tr.get("outputs", {})
        modality = tr.get("modality") or outputs.get("modality", "")
        output_files = outputs.get("output_files", outputs.get("nifti_files", []))

        if modality == "anat":
            for f in output_files:
                f_path = Path(f)
                # 跳过SPM衍生文件
                if any(f_path.name.startswith(prefix) for prefix in
                       ["c1", "c2", "c3", "c4", "c5", "c6", "wc", "mwc", "y_", "iy_", "r", "s", "w"]):
                    continue
                if f_path.suffix in [".nii", ".gz"] and f_path.exists():
                    result["source_image"] = str(f_path)
                    print(f"  [自动查找] T1结构像(source): {f_path.name}")
                    break
            if result["source_image"]:
                break

    # 2.2 如果没找到，从 DATA_DIR 的 anat 目录查找
    if not result["source_image"]:
        try:
            from src.config import DATA_DIR
            data_dir = Path(DATA_DIR)
            if data_dir.exists():
                # 查找 anat 目录
                for anat_dir in data_dir.rglob("anat"):
                    if anat_dir.is_dir():
                        nii_files = list(anat_dir.glob("*.nii")) + list(anat_dir.glob("*.nii.gz"))
                        # 过滤掉SPM衍生文件
                        original_files = [f for f in nii_files if not any(
                            f.name.startswith(p) for p in ["c1", "c2", "c3", "wc", "mwc", "y_", "iy_"]
                        )]
                        if original_files:
                            result["source_image"] = str(original_files[0])
                            print(f"  [自动查找] T1结构像(source)从DATA_DIR: {original_files[0].name}")
                            break
        except Exception as e:
            print(f"  [警告] 查找DATA_DIR时出错: {e}")

    # 2.3 从 cohort 数据查找
    if not result["source_image"]:
        cohort = state.get("cohort", {})
        data_paths = state.get("data_paths", {})
        for group_name, subjects in cohort.items():
            if not subjects:
                continue
            for subject in (subjects if isinstance(subjects, list) else []):
                subject_id = subject.get("subject_id", "")
                if subject_id in data_paths:
                    anat_path = data_paths[subject_id].get("anat", "")
                    if anat_path:
                        anat_dir = Path(anat_path)
                        if anat_dir.exists():
                            nii_files = list(anat_dir.glob("*.nii")) + list(anat_dir.glob("*.nii.gz"))
                            if nii_files:
                                result["source_image"] = str(nii_files[0])
                                print(f"  [自动查找] T1结构像(source)从cohort: {nii_files[0].name}")
                                break
            if result["source_image"]:
                break

    return result


def _find_coregister_reference_image(state: AgentState) -> Optional[str]:
    """
    为coregistration自动查找参考图像（向后兼容的包装函数）

    Returns:
        参考图像的绝对路径，如果找不到则返回None
    """
    images = _find_coregister_images(state)
    # 优先返回 mean 功能像，如果没有则返回 T1
    return images.get("reference_image") or images.get("source_image")


def _extract_scan_params_from_sidecar(state: AgentState) -> Dict:
    """从 JSON sidecar（dcm2niix 生成的 BIDS 标准文件）中提取扫描参数"""
    result = {}
    try:
        data_manifest = state.get("data_manifest", {})
        # 修复：从 available_modalities 获取 func 目录路径（subjects 中没有 files 键）
        available_modalities = data_manifest.get("available_modalities", {})
        for subject_id, modalities in available_modalities.items():
            func_dir = modalities.get("func")
            if not func_dir:
                continue
            func_path = Path(func_dir)
            # 搜索范围：原始 func 目录 + 转换输出目录
            search_dirs = [func_path]
            converted_dir = OUTPUT_DIR / "dicom_converted" / "func" / subject_id
            if converted_dir.exists():
                search_dirs.append(converted_dir)
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                json_files = list(search_dir.glob("*.json"))
                for json_file in json_files:
                    import json as _json
                    with open(json_file, "r", encoding="utf-8") as jf:
                        sidecar_data = _json.load(jf)
                    if "RepetitionTime" in sidecar_data:
                        result["tr"] = float(sidecar_data["RepetitionTime"])
                    if "SliceTiming" in sidecar_data:
                        slice_timing = sidecar_data["SliceTiming"]
                        result["num_slices"] = len(slice_timing)
                        if slice_timing == sorted(slice_timing):
                            result["slice_order"] = "ascending"
                        elif slice_timing == sorted(slice_timing, reverse=True):
                            result["slice_order"] = "descending"
                        else:
                            result["slice_order"] = "interleaved_ascending"
                    if result:
                        print(f"  [sidecar] 从 {json_file.name} 读取参数: {result}")
                        return result
    except Exception as e:
        print(f"  [sidecar] 读取 JSON sidecar 失败: {e}")
    return result


def _fill_missing_params(tool_name: str, tool_step: str, params: Dict, state: AgentState) -> Dict:
    """
    智能填充缺失的工具参数

    Args:
        tool_name: 工具名称
        tool_step: 步骤名称/描述
        params: 原始参数字典
        state: Agent状态

    Returns:
        填充后的参数字典
    """
    filled_params = params.copy()
    step_lower = (tool_step or "").lower()

    # ========== SPM Analysis 参数补充 ==========
    if tool_name == "spm_analysis":
        if "analysis_type" not in filled_params:
            # 根据步骤描述推断分析类型
            # fMRI 预处理类型
            if any(keyword in step_lower for keyword in ["realign", "头动校正", "motion correction", "头动"]):
                filled_params["analysis_type"] = "realign"
                print(f"  [参数补充] analysis_type = 'realign' (fMRI头动校正)")
            elif any(keyword in step_lower for keyword in ["slice_timing", "层时间", "slice timing", "时间校正"]):
                filled_params["analysis_type"] = "slice_timing"
                print(f"  [参数补充] analysis_type = 'slice_timing' (fMRI层时间校正)")
            elif any(keyword in step_lower for keyword in ["coregister", "配准", "coregistration", "对齐"]):
                filled_params["analysis_type"] = "coregister"
                print(f"  [参数补充] analysis_type = 'coregister' (配准)")
            # 结构像处理类型
            elif any(keyword in step_lower for keyword in ["segment", "vbm", "分割", "灰质", "白质"]):
                filled_params["analysis_type"] = "vbm_segment"
                print(f"  [参数补充] analysis_type = 'vbm_segment' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["smooth", "平滑"]):
                filled_params["analysis_type"] = "smooth"
                print(f"  [参数补充] analysis_type = 'smooth' (平滑)")
            elif any(keyword in step_lower for keyword in ["normalize", "标准化"]):
                filled_params["analysis_type"] = "normalize"
                print(f"  [参数补充] analysis_type = 'normalize' (标准化)")
            elif any(keyword in step_lower for keyword in ["preprocess", "预处理"]):
                filled_params["analysis_type"] = "smooth"
                print(f"  [参数补充] analysis_type = 'smooth' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["glm", "统计", "contrast", "对比"]):
                filled_params["analysis_type"] = "glm_2nd"
                print(f"  [参数补充] analysis_type = 'glm_2nd' (从步骤描述推断)")
            else:
                # 默认使用VBM分割
                filled_params["analysis_type"] = "vbm_segment"
                print(f"  [参数补充] analysis_type = 'vbm_segment' (默认值)")

        # 根据analysis_type补充特定参数
        analysis_type = filled_params.get("analysis_type", "")

        # fMRI slice_timing 参数补充（优先从 JSON sidecar 读取）
        if analysis_type == "slice_timing":
            sidecar_params = _extract_scan_params_from_sidecar(state)
            if sidecar_params.get("tr"):
                filled_params.setdefault("tr", sidecar_params["tr"])
                print(f"  [参数补充] tr = {sidecar_params['tr']} (从 JSON sidecar 读取)")
            if sidecar_params.get("num_slices"):
                filled_params.setdefault("num_slices", sidecar_params["num_slices"])
                print(f"  [参数补充] num_slices = {sidecar_params['num_slices']} (从 JSON sidecar 读取)")
            if sidecar_params.get("slice_order"):
                filled_params.setdefault("slice_order", sidecar_params["slice_order"])
                print(f"  [参数补充] slice_order = '{sidecar_params['slice_order']}' (从 JSON sidecar 读取)")
            # 硬编码默认值作为最终 fallback
            if "tr" not in filled_params:
                filled_params["tr"] = 2.0
                print(f"  [参数补充] tr = 2.0 (默认值，请根据实际数据调整)")
            if "num_slices" not in filled_params:
                filled_params["num_slices"] = 33
                print(f"  [参数补充] num_slices = 33 (默认值，请根据实际数据调整)")
            if "slice_order" not in filled_params:
                filled_params["slice_order"] = "interleaved_ascending"
                print(f"  [参数补充] slice_order = 'interleaved_ascending' (默认值)")

        # fMRI coregister 参数检查 - 自动查找 reference_image 和 source_image
        elif analysis_type == "coregister":
            # 使用新的函数同时查找 mean 功能像和 T1 结构像
            coregister_images = _find_coregister_images(state)
            auto_ref = coregister_images.get("reference_image")  # 自动查找的 mean 功能像
            auto_src = coregister_images.get("source_image")      # 自动查找的 T1 结构像

            # ===== 处理 reference_image（mean功能像） =====
            # 优先使用自动查找的 mean 功能像，因为任务参数中的值通常不正确
            reference_image = filled_params.get("reference_image")

            if auto_ref:
                if reference_image and reference_image != auto_ref:
                    print(f"  [WARNING] 覆盖 reference_image: '{Path(reference_image).name}' → '{Path(auto_ref).name}'")
                    print(f"           原因: 自动查找的 mean 功能像比 LLM 推断的更可靠")
                filled_params["reference_image"] = auto_ref
                print(f"  [参数补充] reference_image = '{Path(auto_ref).name}' (mean功能像)")
            elif reference_image:
                # 没有自动查找到，尝试验证任务提供的 reference_image
                ref_path = Path(reference_image)
                if not ref_path.is_absolute():
                    # 尝试查找绝对路径
                    found = _find_file_in_results(reference_image, state)
                    if found:
                        filled_params["reference_image"] = found
                        print(f"  [路径转换] reference_image: {reference_image} -> {found}")
                    else:
                        # 无法找到，报错
                        filled_params["_coregister_error"] = (
                            f"reference_image '{reference_image}' 无法找到，"
                            f"且未能自动查找到 mean 功能像。请确保已完成 realign 步骤。"
                        )
                        print(f"  [错误] reference_image '{reference_image}' 无法找到，无自动替代")
                elif not ref_path.exists():
                    filled_params["_coregister_error"] = f"reference_image 文件不存在: {reference_image}"
                    print(f"  [错误] reference_image 文件不存在")

            # ===== 处理 source_image（T1结构像） =====
            # 优先使用自动查找的 T1 结构像
            source_image = filled_params.get("source_image")

            if auto_src:
                # 如果自动查找成功找到了 T1 结构像，优先使用它
                if source_image and source_image != auto_src:
                    print(f"  [覆盖] 忽略任务参数中的 source_image '{source_image}'")
                    print(f"         使用自动查找的 T1 结构像: {Path(auto_src).name}")
                filled_params["source_image"] = auto_src
                print(f"  [参数补充] source_image = '{Path(auto_src).name}' (T1结构像)")
            elif source_image:
                # 没有自动查找到，尝试验证任务提供的 source_image
                src_path = Path(source_image)
                if not src_path.is_absolute():
                    found = _find_file_in_results(source_image, state)
                    if found:
                        filled_params["source_image"] = found
                        print(f"  [路径转换] source_image: {source_image} -> {found}")
                    else:
                        print(f"  [警告] source_image '{source_image}' 无法找到，将使用 input_files")

            # ===== 验证配准所需的图像是否齐全 =====
            if "_coregister_error" not in filled_params:
                ref = filled_params.get("reference_image")
                src = filled_params.get("source_image")

                if not ref:
                    filled_params["_coregister_error"] = (
                        "coregister 缺少 reference_image（mean功能像）。"
                        "请确保已完成 realign 步骤生成 mean*.nii 文件。"
                    )
                    print(f"  [错误] 缺少 reference_image（mean功能像）")
                elif not src:
                    # source_image 不是必需的，可以使用 input_files
                    print(f"  [提示] source_image 未指定，将使用 input_files 作为源图像")
                else:
                    print(f"  [配准配置] reference={Path(ref).name}, source={Path(src).name}")

        # 补充其他可选参数
        if "smoothing_fwhm" not in filled_params:
            # fMRI 使用较小的平滑核
            if analysis_type in ["realign", "slice_timing", "coregister"]:
                filled_params["smoothing_fwhm"] = 6
            else:
                filled_params["smoothing_fwhm"] = 8

    # ========== Python Stats 参数补充 ==========
    elif tool_name == "python_stats":
        if "analysis_type" not in filled_params:
            # 根据步骤描述推断统计类型
            plan = state.get("plan", {})
            design = plan.get("design", {})
            groups = design.get("groups", [])

            if any(keyword in step_lower for keyword in ["ttest", "t检验", "t-test", "comparison", "比较"]):
                filled_params["analysis_type"] = "ttest"
                print(f"  [参数补充] analysis_type = 'ttest' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["anova", "方差分析"]):
                filled_params["analysis_type"] = "anova"
                print(f"  [参数补充] analysis_type = 'anova' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["correlation", "相关", "关联"]):
                filled_params["analysis_type"] = "correlation"
                print(f"  [参数补充] analysis_type = 'correlation' (从步骤描述推断)")
            elif len(groups) == 2:
                # 两组比较默认用t检验
                filled_params["analysis_type"] = "ttest"
                print(f"  [参数补充] analysis_type = 'ttest' (根据组数推断)")
            elif len(groups) > 2:
                # 多组比较默认用ANOVA
                filled_params["analysis_type"] = "anova"
                print(f"  [参数补充] analysis_type = 'anova' (根据组数推断)")
            else:
                # 默认t检验
                filled_params["analysis_type"] = "ttest"
                print(f"  [参数补充] analysis_type = 'ttest' (默认值)")

    # ========== Data Visualization 参数补充 ==========
    elif tool_name == "data_visualization":
        # data_files会在tool_inputs中传递，不在params中
        # 但需要确保visualization_type存在
        if "visualization_type" not in filled_params:
            # 根据步骤描述推断可视化类型
            if any(keyword in step_lower for keyword in ["slice", "切片", "brain", "脑"]):
                filled_params["visualization_type"] = "brain_slices"
                print(f"  [参数补充] visualization_type = 'brain_slices' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["statistical", "统计", "activation", "激活"]):
                filled_params["visualization_type"] = "statistical_maps"
                print(f"  [参数补充] visualization_type = 'statistical_maps' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["comparison", "比较", "contrast", "对比"]):
                filled_params["visualization_type"] = "comparison_plot"
                print(f"  [参数补充] visualization_type = 'comparison_plot' (从步骤描述推断)")
            elif any(keyword in step_lower for keyword in ["quality", "质量", "qa", "qc"]):
                filled_params["visualization_type"] = "brain_slices"
                print(f"  [参数补充] visualization_type = 'brain_slices' (质量检查)")
            else:
                # 默认自定义可视化
                filled_params["visualization_type"] = "custom"
                print(f"  [参数补充] visualization_type = 'custom' (默认值)")

        # 补充描述和标题
        if "description" not in filled_params:
            filled_params["description"] = tool_step
        if "figure_title" not in filled_params:
            filled_params["figure_title"] = tool_step or "Neuroimaging Visualization"

    # ========== DPABI 参数补充 ==========
    elif tool_name == "dpabi_analysis":
        # analysis_type 智能推断
        if "analysis_type" not in filled_params:
            if "alff" in step_lower or "低频振幅" in step_lower:
                filled_params["analysis_type"] = "alff"
                print(f"  [参数补充] analysis_type = 'alff' (低频振幅)")
            elif "falff" in step_lower or "分数" in step_lower or "fractional" in step_lower:
                filled_params["analysis_type"] = "falff"
                print(f"  [参数补充] analysis_type = 'falff' (分数低频振幅)")
            elif "reho" in step_lower or "局部一致性" in step_lower or "regional homogeneity" in step_lower:
                filled_params["analysis_type"] = "reho"
                print(f"  [参数补充] analysis_type = 'reho' (局部一致性)")
            elif "fc" in step_lower or "功能连接" in step_lower or "functional connectivity" in step_lower:
                if "voxel" in step_lower or "体素" in step_lower:
                    filled_params["analysis_type"] = "fc_voxel"
                    print(f"  [参数补充] analysis_type = 'fc_voxel' (体素级功能连接)")
                else:
                    filled_params["analysis_type"] = "fc_seed"
                    print(f"  [参数补充] analysis_type = 'fc_seed' (种子点功能连接)")
            elif "dc" in step_lower or "度中心性" in step_lower or "centrality" in step_lower:
                filled_params["analysis_type"] = "degree_centrality"
                print(f"  [参数补充] analysis_type = 'degree_centrality' (度中心性)")
            elif "rest" in step_lower or "静息" in step_lower or "resting" in step_lower:
                filled_params["analysis_type"] = "alff"
                print(f"  [参数补充] analysis_type = 'alff' (静息态分析默认)")
            else:
                filled_params["analysis_type"] = "alff"
                print(f"  [参数补充] analysis_type = 'alff' (默认值)")

        # TR 参数补充
        if "tr" not in filled_params:
            filled_params["tr"] = 2.0
            print(f"  [参数补充] tr = 2.0 (默认值，请根据实际数据调整)")

        # band_pass 参数补充
        if "band_pass" not in filled_params:
            filled_params["band_pass"] = [0.01, 0.08]
            print(f"  [参数补充] band_pass = [0.01, 0.08] (静息态默认频带)")

    # ========== DSI Studio 参数补充 ==========
    elif tool_name == "dsi_studio_analysis":
        if "analysis_type" not in filled_params:
            filled_params["analysis_type"] = "reconstruction"
            print(f"  [参数补充] analysis_type = 'reconstruction' (默认值)")

    # ========== FreeSurfer 参数补充 ==========
    elif tool_name == "freesurfer_analysis":
        # 使用统一定义的FreeSurfer命令白名单（从config_local_tools导入）

        # 检查并修正command参数
        command = filled_params.get("command", "")

        # 如果command不在有效列表中，需要智能推断
        if command not in FREESURFER_SUPPORTED_COMMANDS:
            # 检查是否是"验证/检查/复用"类型的任务（应该使用recon-all让工具自动检测）
            verify_keywords = ["verify", "check", "reuse", "复用", "验证", "检查", "existing", "已存在"]
            stats_keywords = ["stats", "table", "统计", "导出", "export"]
            convert_keywords = ["convert", "转换"]

            if any(kw in step_lower or kw in command.lower() for kw in verify_keywords):
                # 验证/复用任务：使用recon-all，FreeSurfer会自动检测recon-all.done并跳过
                filled_params["command"] = "recon-all"
                print(f"  [参数修正] command = 'recon-all' (验证/复用任务，工具会自动检测已完成的结果)")
            elif any(kw in step_lower for kw in stats_keywords):
                # 统计导出任务
                if "aparc" in step_lower or "皮层" in step_lower or "thickness" in step_lower:
                    filled_params["command"] = "aparcstats2table"
                else:
                    filled_params["command"] = "asegstats2table"
                print(f"  [参数补充] command = '{filled_params['command']}' (统计导出)")
            elif any(kw in step_lower for kw in convert_keywords):
                filled_params["command"] = "mri_convert"
                print(f"  [参数补充] command = 'mri_convert' (格式转换)")
            else:
                # 默认使用recon-all
                filled_params["command"] = "recon-all"
                print(f"  [参数修正] command = 'recon-all' (原命令'{command}'无效，使用默认值)")

        # 确保directive参数存在
        if "directive" not in filled_params and filled_params.get("command") == "recon-all":
            filled_params["directive"] = "-all"
            print(f"  [参数补充] directive = '-all' (默认值)")

    # ========== FSL 参数补充 ==========
    elif tool_name == "fsl_analysis":
        # 使用统一定义的FSL命令白名单（从config_local_tools导入）

        # 1. 如果没有 command 参数，根据步骤描述推断
        if "command" not in filled_params:
            if "bet" in step_lower or "脑提取" in step_lower:
                filled_params["command"] = "bet"
                print(f"  [参数补充] command = 'bet' (从步骤描述推断)")
            elif "fast" in step_lower or "分割" in step_lower:
                filled_params["command"] = "fast"
                print(f"  [参数补充] command = 'fast' (从步骤描述推断)")
            elif "flirt" in step_lower or "配准" in step_lower:
                filled_params["command"] = "flirt"
                print(f"  [参数补充] command = 'flirt' (从步骤描述推断)")
            elif ("质量控制" in step_lower or "qc" in step_lower) and "eddy" in step_lower:
                # eddy_quad 不可用 - 使用 python_stats 读取 eddy 输出文件
                print(f"  [信息] eddy 质量控制: 建议使用 python_stats 读取 .eddy_movement_rms 文件")
                print(f"  [信息] 或使用 vibe_coding 调用 read_eddy_motion_parameters() 函数")
                # 不设置 command - 让后续流程使用 python_stats 或 vibe_coding
                filled_params["_needs_eddy_qc"] = True
            elif "eddy" in step_lower or "涡流" in step_lower:
                filled_params["command"] = "eddy"
                print(f"  [参数补充] command = 'eddy' (从步骤描述推断)")
            elif "dtifit" in step_lower or "张量" in step_lower:
                filled_params["command"] = "dtifit"
                print(f"  [参数补充] command = 'dtifit' (从步骤描述推断)")
            # TBSS命令推断（白质分析）
            elif "tbss" in step_lower or "白质分析" in step_lower or "骨架" in step_lower or "tract" in step_lower:
                # 根据关键词推断具体的TBSS步骤
                if "preproc" in step_lower or "预处理" in step_lower or "步骤1" in step_lower:
                    filled_params["command"] = "tbss_1_preproc"
                    print(f"  [参数补充] command = 'tbss_1_preproc' (TBSS步骤1: 预处理)")
                elif "reg" in step_lower or "配准" in step_lower or "步骤2" in step_lower:
                    filled_params["command"] = "tbss_2_reg"
                    print(f"  [参数补充] command = 'tbss_2_reg' (TBSS步骤2: 配准)")
                elif "postreg" in step_lower or "投影" in step_lower or "步骤3" in step_lower:
                    filled_params["command"] = "tbss_3_postreg"
                    print(f"  [参数补充] command = 'tbss_3_postreg' (TBSS步骤3: 后处理)")
                elif "prestats" in step_lower or "统计" in step_lower or "步骤4" in step_lower:
                    filled_params["command"] = "tbss_4_prestats"
                    print(f"  [参数补充] command = 'tbss_4_prestats' (TBSS步骤4: 准备统计)")
                else:
                    # 默认从步骤1开始
                    filled_params["command"] = "tbss_1_preproc"
                    print(f"  [参数补充] command = 'tbss_1_preproc' (TBSS默认从步骤1开始)")
            # 纤维追踪命令推断
            elif "bedpostx" in step_lower or "扩散参数" in step_lower or "贝叶斯" in step_lower:
                filled_params["command"] = "bedpostx"
                print(f"  [参数补充] command = 'bedpostx' (贝叶斯扩散参数估计)")
            elif "probtrackx" in step_lower or "纤维追踪" in step_lower or "tractography" in step_lower:
                filled_params["command"] = "probtrackx"
                print(f"  [参数补充] command = 'probtrackx' (概率纤维追踪)")
            else:
                filled_params["command"] = "bet"
                print(f"  [参数补充] command = 'bet' (默认值)")

        # 2. 【关键】验证已填充的命令是否有效，如果无效则智能替换
        cmd = filled_params.get("command", "")
        if cmd and cmd not in FSL_SUPPORTED_COMMANDS:
            print(f"  [警告] 不支持的FSL命令 '{cmd}'，尝试智能替换...")
            # 尝试智能映射到有效命令
            if "motion" in cmd.lower() or "parameter" in cmd.lower() or "qc" in cmd.lower():
                print(f"  [警告] 不支持的FSL命令 '{cmd}'")
                print(f"  [建议] 运动参数提取请使用 python_stats 或 vibe_coding")
                filled_params["_skip_execution"] = True
                filled_params["_skip_reason"] = f"不支持的FSL命令: {cmd}"
            elif "extract" in cmd.lower() or "stat" in cmd.lower():
                # 统计/提取相关 → fslstats
                filled_params["command"] = "fslstats"
                print(f"  [智能替换] '{cmd}' → 'fslstats'（统计信息提取）")
            elif "math" in cmd.lower() or "calc" in cmd.lower():
                # 数学运算相关 → fslmaths
                filled_params["command"] = "fslmaths"
                print(f"  [智能替换] '{cmd}' → 'fslmaths'（图像数学运算）")
            else:
                # 无法识别 → 跳过该任务（返回无效命令让执行层报错）
                print(f"  [警告] 无法替换无效命令 '{cmd}'，将在执行时失败")

    return filled_params


def _is_pipeline_dependent(prev_tool_name: str, curr_tool_name: str) -> bool:
    """判断当前工具是否依赖前一个工具的输出"""
    # 1. 同工具名 = 管道内步骤（如 spm segment → spm smooth）
    if prev_tool_name == curr_tool_name:
        return True
    # 2. 检查知识图谱的 followed_by / depends_on
    from src.knowledge.tool_knowledge_graph import get_following_tools, get_tool_dependencies
    if curr_tool_name in get_following_tools(prev_tool_name):
        return True
    if prev_tool_name in get_tool_dependencies(curr_tool_name):
        return True
    return False


def node_execute_tool(state: AgentState) -> Dict[str, Any]:
    """
    执行当前工具 - 属于ReAct模式的Action阶段
    返回更新字典（LangGraph 标准模式），不直接修改 state。
    """
    tracker = _get_tracker()
    registry = _get_registry()

    tool_chain = state.get("tool_chain", [])
    current_index = state.get("current_tool_index", 0)

    if current_index >= len(tool_chain):
        return merge_state_updates({"node_history": ["execute_tool:no_more_tools"]})

    # 辅助函数：构建 tool_chain 副本并更新当前步骤状态
    def _updated_chain(status, error=None):
        chain_copy = list(tool_chain)
        if current_index < len(chain_copy):
            entry = {**chain_copy[current_index], "status": status}
            if error:
                entry["error"] = error
            chain_copy[current_index] = entry
        return chain_copy

    current_tool = tool_chain[current_index]
    tool_name = current_tool["tool"]
    tool_step = current_tool.get("step", "")

    print(f"\n[NODE: execute_tool] 执行工具 {current_index + 1}/{len(tool_chain)}: {tool_name}")
    print(f"  步骤: {tool_step}")

    try:
        # 构建请求
        call_id = str(uuid.uuid4())[:8]
        run_dir = tracker.run_dir if tracker.run_dir else OUTPUT_DIR / "runs" / state["run_id"]
        output_dir = str(run_dir / "tools" / f"step{current_index + 1:02d}_{tool_name}_{call_id}")

        # 获取被试列表
        subjects = state.get("data_manifest", {}).get("subjects", [])

        # 去重检查（防御性编程）
        subject_ids = [s.get("id") for s in subjects]
        unique_ids = list(dict.fromkeys(subject_ids))  # 保持顺序的去重
        if len(unique_ids) != len(subject_ids):
            print(f"  [警告] 工具执行前发现重复被试: {len(subject_ids)} -> {len(unique_ids)}")
            # 去重：只保留第一次出现的被试
            seen = set()
            unique_subjects = []
            for s in subjects:
                sid = s.get("id")
                if sid not in seen:
                    seen.add(sid)
                    unique_subjects.append(s)
            subjects = unique_subjects

        print(f"  被试数量: {len(subjects)}")
        print(f"  被试ID: {[s.get('id') for s in subjects]}")

        # 确定模态
        plan = state.get("plan", {})
        modalities = plan.get("modalities", ["anat"])
        primary_modality = modalities[0] if modalities else "anat"

        # 构建实际的输入文件路径
        print(f"  模态: {primary_modality}")
        input_files = _build_input_files(subjects, primary_modality, plan)

        if not input_files:
            raise ValueError(f"未找到任何输入文件 (模态: {primary_modality}, 被试: {len(subjects)})")

        print(f"  输入文件: {len(input_files)} 个")

        # 构建工具参数
        tool_params = current_tool.get("params", {})

        # ========== 智能填充缺失参数 ==========
        # 如果LLM生成的计划中缺少必需参数，根据工具类型和上下文智能补充
        tool_params = _fill_missing_params(tool_name, tool_step, tool_params, state)

        # ========== 检查参数填充阶段标记的跳过 ==========
        if tool_params.get("_skip_execution"):
            skip_reason = tool_params.get("_skip_reason", "参数填充阶段标记跳过")
            print(f"  [跳过] {skip_reason}")
            tool_result = {
                "tool": tool_name, "tool_name": tool_name,
                "step": tool_step, "call_id": call_id,
                "status": "skipped", "duration": 0,
                "outputs": {}, "output_dir": output_dir,
                "error": skip_reason, "modality": primary_modality
            }
            return merge_state_updates({
                "tool_results": [tool_result],
                "current_tool_index": current_index + 1,
                "tool_chain": _updated_chain("skipped"),
                "node_history": [f"execute_tool:{tool_name}:skipped_param"]
            })

        # ========== 前置依赖检查（含传递依赖） ==========
        iteration_offset = state.get("tool_results_iteration_offset", 0)
        if current_index > 0:
            # 扫描所有前驱工具，检查是否有失败的依赖
            for prev_idx in range(current_index - 1, -1, -1):
                prev_tool_entry = tool_chain[prev_idx]
                prev_status = prev_tool_entry.get("status", "")
                if prev_status in ("failed", "skipped"):
                    prev_tool_name = prev_tool_entry.get("tool", "")
                    if _is_pipeline_dependent(prev_tool_name, tool_name):
                        print(f"  [跳过] 前置工具 {prev_tool_name}(step {prev_idx}) 状态={prev_status}，跳过 {tool_name}")
                        tool_result = {
                            "tool": tool_name, "tool_name": tool_name,
                            "step": tool_step, "call_id": call_id,
                            "status": "skipped", "duration": 0,
                            "outputs": {}, "output_dir": output_dir,
                            "error": f"前置步骤 {prev_tool_name} {prev_status}",
                            "modality": primary_modality
                        }
                        return merge_state_updates({
                            "tool_results": [tool_result],
                            "current_tool_index": current_index + 1,
                            "tool_chain": _updated_chain("skipped"),
                            "node_history": [f"execute_tool:{tool_name}:skipped_dep"]
                        })

        # 如果是第一个工具之后的步骤，尝试使用前一步的输出（向后搜索最近成功的结果）
        if current_index > 0:
            prev_results = state.get("tool_results", [])
            # 使用迭代偏移，只看当前迭代的结果
            current_iter_results = prev_results[iteration_offset:]
            # 向后搜索最近一个成功的、有输出文件的结果
            prev_output_files = None
            for r in reversed(current_iter_results):
                if (r.get("status") in ("succeeded", "cached") and
                    r.get("outputs", {}).get("output_files")):
                    prev_output_files = r["outputs"]["output_files"]
                    break

            if prev_output_files:
                input_files = prev_output_files

                # FreeSurfer特殊处理：只使用原始T1文件，过滤掉SPM衍生文件
                if tool_name == "freesurfer_analysis":
                    import re
                    original_t1_files = []
                    for f in input_files:
                        basename = Path(f).name
                        if not re.match(r'^(c\d+|wc\d+|sc\d+|s[^C]|sm|m[^e]|y_)', basename, re.IGNORECASE):
                            original_t1_files.append(f)
                    if original_t1_files:
                        input_files = original_t1_files
                        print(f"  [FreeSurfer] 过滤SPM衍生文件，保留原始T1: {len(input_files)} 个")
                    else:
                        print(f"  [FreeSurfer] 警告: 未找到原始T1文件，使用所有文件")

                print(f"  使用前序步骤的输出: {len(input_files)} 个文件")

        # ========== 智能适配工具参数 ==========
        # 根据工具的schema自动调整参数名称
        tool_def = registry.get_definition(tool_name)
        if tool_def:
            input_schema = tool_def.input_schema
            required_params = input_schema.get("required", [])
            properties = input_schema.get("properties", {})

            # 准备输入数据 - 根据工具schema智能适配
            tool_inputs = {"modality": primary_modality}

            # 检查工具需要什么类型的输入
            if "input_dirs" in required_params or "input_dirs" in properties:
                # 工具需要目录列表（如DICOM转换）
                tool_inputs["input_dirs"] = input_files
                print(f"  参数适配: input_dirs (目录列表)")
            elif "data_files" in required_params or "data_files" in properties:
                # 工具需要data_files（如data_visualization）
                tool_inputs["data_files"] = input_files
                print(f"  参数适配: data_files (数据文件列表)")
            elif "input_files" in required_params or "input_files" in properties:
                # 工具需要文件列表（如SPM、DPABI）
                tool_inputs["input_files"] = input_files
                print(f"  参数适配: input_files (文件列表)")
            elif "input_file" in required_params or "input_file" in properties:
                # 工具需要单个文件（如FSL某些命令）
                tool_inputs["input_file"] = input_files[0] if input_files else ""
                print(f"  参数适配: input_file (单个文件)")
            elif "input_dir" in required_params or "input_dir" in properties:
                # 工具需要单个目录（如DPABI）
                # 特殊处理：如果input_files是文件路径列表，提取父目录
                if input_files:
                    first_input = Path(input_files[0])
                    if first_input.is_file():
                        # 如果是文件，使用其父目录
                        tool_inputs["input_dir"] = str(first_input.parent)
                        print(f"  参数适配: input_dir (从文件提取父目录: {first_input.parent.name})")
                    elif first_input.is_dir():
                        # 如果已经是目录，直接使用
                        tool_inputs["input_dir"] = str(first_input)
                        print(f"  参数适配: input_dir (目录)")
                    else:
                        # 路径不存在，尝试提取父目录
                        tool_inputs["input_dir"] = str(first_input.parent)
                        print(f"  参数适配: input_dir (路径不存在，使用父目录)")
                else:
                    tool_inputs["input_dir"] = ""
                    print(f"  警告: input_dir为空，可能导致工具执行失败")
            else:
                # 默认使用input_files
                tool_inputs["input_files"] = input_files
                print(f"  参数适配: input_files (默认)")
        else:
            # 找不到工具定义，使用默认
            tool_inputs = {
                "input_files": input_files,
                "modality": primary_modality
            }
            print(f"  警告: 未找到工具定义，使用默认参数")

        # ========== 数据复用检查 ==========
        print(f"  [数据复用] 检查是否存在可复用结果...")
        # 获取实际的输入文件/目录（用于缓存键计算）
        actual_inputs = (tool_inputs.get("input_files") or
                        tool_inputs.get("input_dirs") or
                        tool_inputs.get("data_files") or
                        tool_inputs.get("input_file") or
                        tool_inputs.get("input_dir") or
                        input_files)
        if isinstance(actual_inputs, str):
            actual_inputs = [actual_inputs]
        cached = _check_cached_result(run_dir, tool_name, actual_inputs, tool_params)

        if cached:
            # 使用缓存结果
            cached_output_dir, cached_meta = cached
            print(f"  [数据复用] [OK] 使用缓存结果，跳过执行")

            # 构建缓存结果对象
            tool_result = {
                "tool": tool_name,
                "tool_name": tool_name,  # 添加tool_name用于模态匹配
                "step": tool_step,
                "call_id": call_id,
                "status": cached_meta.get("status", "succeeded"),
                "duration": 0,  # 缓存复用耗时为0
                "outputs": {
                    "output_files": cached_meta.get("output_files", []),
                    "cached": True,
                    "original_duration": cached_meta.get("duration_seconds", 0)
                },
                "output_dir": cached_output_dir,
                "error": None,
                "modality": primary_modality  # 添加模态信息
            }

            # 累积结果
            print(f"  状态: succeeded (缓存)")
            print(f"  输出文件: {len(cached_meta.get('output_files', []))} 个")
            return merge_state_updates({
                "tool_results": [tool_result],
                "current_tool_index": current_index + 1,
                "tool_chain": _updated_chain("succeeded"),
                "node_history": [f"execute_tool:{tool_name}:cached"]
            })

        # ========== 智能跳过不必要的转换 ==========
        # 如果是DICOM转换工具，但所有输入已经是NIfTI文件，则跳过转换
        if tool_name == "dicom_to_nifti":
            input_paths = tool_inputs.get("input_dirs", [])
            # 检查所有输入是否都是NIfTI文件（不是目录）
            all_nifti = all(
                str(p).endswith(('.nii', '.nii.gz')) and Path(p).is_file()
                for p in input_paths
            )

            if all_nifti:
                print(f"  [跳过转换] 所有输入已经是NIfTI文件，无需DICOM转换")

                tool_result = {
                    "tool": tool_name,
                    "tool_name": tool_name,  # 添加tool_name用于模态匹配
                    "step": tool_step,
                    "call_id": call_id,
                    "status": "succeeded",
                    "duration": 0,
                    "outputs": {
                        "output_files": input_paths,
                        "skipped": True,
                        "reason": "All inputs are already NIfTI files"
                    },
                    "output_dir": output_dir,
                    "error": None,
                    "modality": primary_modality  # 添加模态信息
                }

                print(f"  状态: succeeded (跳过)")
                print(f"  输出文件: {len(input_paths)} 个")
                return merge_state_updates({
                    "tool_results": [tool_result],
                    "current_tool_index": current_index + 1,
                    "tool_chain": _updated_chain("succeeded"),
                    "node_history": [f"execute_tool:{tool_name}:skipped"]
                })

        # ========== 智能跳过已完成的FreeSurfer处理 ==========
        # 如果是FreeSurfer recon-all，检查是否所有被试都已完成处理
        if tool_name == "freesurfer_analysis":
            fs_command = tool_params.get("command", "recon-all")
            if fs_command in ["recon-all", "recon-all-clinical"]:
                # 检查输出目录是否已有完成的结果
                subjects = state.get("data_manifest", {}).get("subjects", [])
                subject_ids = [s.get("subject_id", s.get("id", "")) for s in subjects]

                # 检查每个被试的recon-all.done文件
                all_completed = True
                completed_subjects = []
                stats_files = []

                for subj_id in subject_ids:
                    subject_dir = Path(output_dir) / subj_id
                    done_file = subject_dir / "scripts" / "recon-all.done"
                    stats_dir = subject_dir / "stats"

                    if done_file.exists():
                        # 验证关键输出文件完整性
                        required_outputs = [
                            "surf/lh.pial", "surf/rh.pial",
                            "surf/lh.white", "surf/rh.white",
                            "stats/aseg.stats",
                            "stats/lh.aparc.stats", "stats/rh.aparc.stats"
                        ]
                        outputs_complete = all(
                            (subject_dir / req).exists() and (subject_dir / req).stat().st_size > 0
                            for req in required_outputs
                        )

                        if outputs_complete:
                            completed_subjects.append(subj_id)
                            # 收集统计文件
                            if stats_dir.exists():
                                stats_files.extend([str(f) for f in stats_dir.glob("*.stats")])
                        else:
                            print(f"  [警告] {subj_id}: recon-all.done 存在但关键输出不完整，需重新处理")
                            all_completed = False
                    else:
                        all_completed = False

                if all_completed and completed_subjects:
                    print(f"  [跳过处理] 所有 {len(completed_subjects)} 个被试的FreeSurfer recon-all已完成")

                    tool_result = {
                        "tool": tool_name,
                        "tool_name": tool_name,
                        "step": tool_step,
                        "call_id": call_id,
                        "status": "succeeded",
                        "duration": 0,
                        "outputs": {
                            "subjects_dir": str(output_dir),
                            "processed_subjects": completed_subjects,
                            "stats_files": stats_files,
                            "skipped": True,
                            "reason": "All subjects already completed recon-all"
                        },
                        "output_dir": output_dir,
                        "error": None,
                        "modality": "anat"
                    }

                    print(f"  状态: succeeded (已完成，跳过)")
                    print(f"  被试: {len(completed_subjects)} 个")
                    print(f"  统计文件: {len(stats_files)} 个")
                    return merge_state_updates({
                        "tool_results": [tool_result],
                        "current_tool_index": current_index + 1,
                        "tool_chain": _updated_chain("succeeded"),
                        "node_history": [f"execute_tool:{tool_name}:skipped"]
                    })

        # ========== 检查参数验证错误 ==========
        # 检查是否有coregister参考图像错误
        if "_coregister_error" in tool_params:
            error_msg = tool_params.pop("_coregister_error")
            print(f"  [FAILED] 参数验证失败: {error_msg}")

            tool_result = {
                "tool": tool_name,
                "tool_name": tool_name,
                "step": tool_step,
                "call_id": call_id,
                "status": "failed",
                "duration": 0,
                "outputs": {},
                "output_dir": output_dir,
                "error": error_msg,
                "modality": primary_modality
            }

            return merge_state_updates({
                "tool_results": [tool_result],
                "current_tool_index": current_index + 1,
                "tool_chain": _updated_chain("failed", error=error_msg),
                "node_history": [f"execute_tool:{tool_name}:param_error"]
            })

        # ========== 正常执行工具 ==========
        print(f"  [数据复用] 未找到缓存，正常执行工具...")

        # 准备context（包含cohort等信息）
        context = {
            "cohort": state.get("cohort", {}),
            "research_question": state.get("question", ""),
            "modality": primary_modality
        }

        request = ToolCallRequest(
            tool_name=tool_name,
            call_id=call_id,
            inputs=tool_inputs,
            params=tool_params,
            output_dir=output_dir,
            context=context
        )

        print(f"  输出目录: {output_dir}")
        print(f"  开始执行...")

        # 执行工具
        start_time = datetime.now()
        result = registry.execute(request)
        end_time = datetime.now()

        # ========== 耗时监控（仅警告，不强制终止） ==========
        elapsed_seconds = (end_time - start_time).total_seconds()
        # 预期处理时间（秒）：根据工具类型和被试数量估算
        _expected_times = {
            "freesurfer_analysis": 21600,  # 6h per subject
            "spm_analysis": 600,           # 10min
            "fsl_analysis": 1800,          # 30min
            "dpabi_analysis": 1200,        # 20min
            "dicom_to_nifti": 120,         # 2min
            "python_stats": 60,            # 1min
        }
        expected_base = _expected_times.get(tool_name, 600)
        num_subjects = len(subjects) if subjects else 1
        expected_time = expected_base * max(num_subjects, 1)
        if elapsed_seconds > expected_time * 2:
            print(f"  [警告] {tool_name} 耗时 {elapsed_seconds:.0f}s，超过预期 {expected_time:.0f}s 的 2 倍")

        # 记录结果（sanitize outputs 防止循环引用进入 state）
        tool_result = {
            "tool": tool_name,
            "tool_name": tool_name,  # 添加tool_name用于模态匹配
            "step": tool_step,
            "call_id": call_id,
            "status": result.status,
            "duration": result.duration_seconds,
            "outputs": _deep_sanitize(result.outputs, max_depth=10),
            "output_dir": output_dir,
            "error": result.error if result.status == "failed" else None,
            "modality": primary_modality  # 添加模态信息
        }

        # ========== 保存缓存元数据（用于后续复用） ==========
        if result.status == "succeeded":
            try:
                _save_cache_metadata(
                    output_dir=output_dir,
                    tool_name=tool_name,
                    input_files=actual_inputs,  # 使用适配后的输入参数
                    params=tool_params,
                    result={
                        "output_files": result.outputs.get("output_files", []),
                        "status": result.status,
                        "duration_seconds": result.duration_seconds
                    }
                )
                print(f"  [数据复用] 缓存元数据已保存")
            except Exception as cache_err:
                print(f"  [数据复用] 警告: 保存缓存元数据失败: {cache_err}")

        # ========== 技能学习：记录执行反馈 ==========
        try:
            _get_tool_skill().record_execution_feedback(
                tool=tool_name, params=tool_params,
                success=(result.status == "succeeded"),
                disease=plan.get("disease", "") or plan.get("disease_context", ""),
                modality=primary_modality,
                task_description=state.get("question", ""),
                duration_seconds=result.duration_seconds,
                output_files=result.outputs.get("output_files") if result.outputs else None,
                error_message=result.error
            )
            if result.status == "succeeded":
                print(f"  [技能学习] 已记录成功执行经验")
        except Exception as skill_err:
            print(f"  [技能学习] 警告: {skill_err}")

        # ========== 过程性知识：失败时打印已知错误恢复建议 ==========
        if result.status == "failed" and result.error:
            try:
                known_errors = current_task.get("metadata", {}).get("known_errors", [])
                for ep in known_errors:
                    pattern = ep.get("pattern", "")
                    if pattern and pattern in (result.error or ""):
                        recovery = ep.get("recovery", "")
                        if recovery:
                            print(f"  [技能提示] 已知错误 '{pattern}': {recovery}")
            except Exception:
                pass

        # ========== MoER: 工具输出规范检查 ==========
        try:
            from src.agent.moer import validate_tool_output
            analysis_type = tool_params.get("analysis_type",
                            tool_params.get("command",
                            tool_params.get("action", "")))
            spec_check = validate_tool_output(
                tool_name, analysis_type, tool_result,
                output_dir=output_dir
            )
            tool_result["output_spec_check"] = spec_check
            if not spec_check.get("passed", True):
                mf = len(spec_check.get('missing_files', []))
                print(f"  [MoER] 输出规范检查: {mf} 个缺失文件")
            acceptance = spec_check.get("acceptance_result", {})
            acc_status = acceptance.get("status", "pass")
            if acc_status == "fail":
                tool_result["acceptance_failed"] = True
                print(f"  [MoER] acceptance 判定: FAIL")
                for fc in acceptance.get("failed_criteria", []):
                    print(f"    - {fc.get('message', '')}")
            elif acc_status == "warning":
                print(f"  [MoER] acceptance 判定: WARNING")
                for w in acceptance.get("warnings", [])[:3]:
                    print(f"    - {w.get('message', '')}")
            else:
                print(f"  [MoER] acceptance 判定: PASS")
        except Exception as spec_err:
            print(f"  [MoER] 输出规范检查失败: {spec_err}")

        # 累积结果
        # 记录到tracker
        if tracker.run_dir:
            try:
                tracker.record_tool_execution(
                    tool_name=tool_name,
                    step_id=f"step{current_index + 1:02d}_{tool_name}",
                    request={"inputs": request.inputs, "params": request.params},
                    response={"status": result.status, "outputs": result.outputs},
                    started_at=start_time.isoformat(),
                    finished_at=end_time.isoformat(),
                    logs=result.logs if hasattr(result, 'logs') else ""
                )
            except Exception as track_err:
                print(f"  警告: 记录tracker失败: {track_err}")

        print(f"  状态: {result.status}")
        print(f"  耗时: {result.duration_seconds:.2f}秒")

        if result.status == "succeeded":
            output_files = result.outputs.get("output_files", [])
            print(f"  输出文件: {len(output_files)} 个")
            if output_files:
                for i, f in enumerate(output_files[:3], 1):
                    print(f"    {i}. {Path(f).name}")
                if len(output_files) > 3:
                    print(f"    ... 还有 {len(output_files) - 3} 个文件")
        else:
            print(f"  错误: {result.error}")

        return merge_state_updates({
            "tool_results": [tool_result],
            "current_tool_index": current_index + 1,
            "tool_chain": _updated_chain(result.status),
            "node_history": [f"execute_tool:{tool_name}"]
        })

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        # 修复：创建失败的 tool_result，保持与 tool_chain 的一致性
        tool_result = {
            "tool": tool_name, "tool_name": tool_name,
            "step": tool_step, "call_id": call_id,
            "status": "failed", "duration": 0,
            "outputs": {}, "output_dir": output_dir,
            "error": str(e), "modality": primary_modality
        }
        print(f"  执行失败: {e}")
        return merge_state_updates({
            "tool_results": [tool_result],
            "current_tool_index": current_index + 1,
            "tool_chain": _updated_chain("failed"),
            "last_error": error_msg,
            "error_history": [f"execute_tool[{tool_name}]: {e}"],
            "node_history": [f"execute_tool:{tool_name}:failed"]
        })


def node_validate_results(state: AgentState) -> Dict[str, Any]:
    """
    验证结果 - 使用Observation阶段的专业提示词
    属于ReAct模式的Observation阶段

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: validate_results] 质量控制和结果验证...")

    llm = _get_llm()
    tracker = _get_tracker()
    cohort = state.get("cohort", {})
    tool_results = _load_full_tool_results(state)

    tracker.start_step("09_validate_results", {})

    try:
        # ========== 使用LLM智能压缩tool_results ==========
        compressed_results = _compress_tool_results_with_llm(tool_results)

        # 使用专业的验证提示词
        prompts = get_node_prompt(
            "validate_results",
            results_summary=compressed_results,
            expectations={
                "statistical_power": "> 0.8",
                "effect_size": "medium to large",
                "multiple_comparison": "corrected",
                "sample_size": cohort.get("total_subjects", 0)
            }
        )

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]

        # 尝试使用LLM进行验证（如果可用）
        try:
            validation = llm.generate_json(messages)
        except Exception as val_err:
            # 降级到简单的规则验证
            print(f"  [验证] LLM验证失败，降级为规则验证: {val_err}")
            validation = {"checks": [], "warnings": [], "passed": True}

            # 检查样本量
            total_n = cohort.get("total_subjects", 0)
            validation["checks"].append({
                "check": "sample_size",
                "value": total_n,
                "status": "pass" if total_n >= 10 else "warning"
            })
            if total_n < 10:
                validation["warnings"].append({"type": "sample_size", "message": f"样本量较小 (n={total_n})"})

            # 检查工具执行成功率
            total_tools = len(tool_results)
            failed_tools = [r for r in tool_results if r.get("status") == "failed"]
            skipped_tools = [r for r in tool_results if r.get("status") == "skipped"]
            effective_failures = len(failed_tools) + len(skipped_tools)
            success_rate = (total_tools - effective_failures) / max(total_tools, 1)

            validation["checks"].append({
                "check": "tool_execution",
                "value": f"{total_tools - effective_failures}/{total_tools}",
                "status": "pass" if effective_failures == 0 else ("fail" if success_rate < 0.3 else "warning")
            })
            if failed_tools:
                validation["warnings"].append({"type": "tool_failures", "message": f"{len(failed_tools)} 个工具执行失败"})

            # 决策逻辑：成功率 < 30% 则拒绝
            if success_rate < 0.3:
                validation["overall_decision"] = "rejected"
                validation["warnings"].append({"type": "high_failure_rate",
                    "message": f"工具成功率仅 {success_rate:.0%}，结果不可靠"})
            elif validation["warnings"]:
                validation["overall_decision"] = "approved_with_warnings"
            else:
                validation["overall_decision"] = "approved"

        # === 聚合 acceptance_criteria 失败 ===
        acceptance_failures = []
        for r in tool_results:
            spec = r.get("output_spec_check", {})
            acc = spec.get("acceptance_result", {})
            if acc.get("status") == "fail":
                acceptance_failures.append({
                    "tool": r.get("tool_name", r.get("tool", "")),
                    "step": r.get("step", ""),
                    "failed_criteria": acc.get("failed_criteria", [])
                })

        if acceptance_failures:
            validation["checks"].append({
                "check": "acceptance_criteria",
                "value": f"{len(acceptance_failures)} 个工具未通过合格标准",
                "status": "fail",
                "details": acceptance_failures
            })
            validation["warnings"].append({
                "type": "acceptance_failure",
                "message": f"{len(acceptance_failures)} 个工具输出未通过 acceptance_criteria"
            })

        # 根据 acceptance 失败率更新 overall_decision
        acceptance_fail_rate = len(acceptance_failures) / max(len(tool_results), 1)
        if acceptance_fail_rate > 0.5:
            validation["overall_decision"] = "rejected"
        elif acceptance_failures and validation.get("overall_decision") == "approved":
            validation["overall_decision"] = "approved_with_warnings"

        tracker.add_step_artifact("09_validate_results", "validation.json", validation)
        tracker.finish_step("09_validate_results", outputs={"decision": validation.get("overall_decision")})

        print(f"  验证决策: {validation.get('overall_decision', 'approved')}")
        if validation.get("warnings"):
            print(f"  警告数量: {len(validation.get('warnings', []))}")

        # ========== MoER: StatReviewer 审查 ==========
        stat_review_result = {}
        moer_stat_reviews = []
        try:
            from src.agent.moer import MoERReviewer
            moer = MoERReviewer(llm_client=llm)
            stat_review_result = moer.review_statistics(
                tool_results, cohort, state.get("plan", {})
            )
            moer_stat_reviews = [stat_review_result]
        except Exception as moer_err:
            print(f"  [MoER] StatReviewer 审查失败: {moer_err}")
            moer_stat_reviews = [{
                "reviewer": "StatReviewer",
                "status": "error",
                "error": str(moer_err),
                "timestamp": datetime.now().isoformat()
            }]

        # === StatReviewer 结果影响 overall_decision ===
        if stat_review_result.get("status") == "rejected":
            validation["overall_decision"] = "rejected"
            validation["warnings"].append({
                "type": "stat_review_rejected",
                "message": "StatReviewer 判定统计结果不可靠"
            })
        elif (stat_review_result.get("status") == "approved_with_warnings"
              and validation.get("overall_decision") == "approved"):
            validation["overall_decision"] = "approved_with_warnings"

        return merge_state_updates({
            "validation": validation,
            "validation_passed": validation.get("overall_decision") in ["approved", "approved_with_warnings"],
            "stat_review": stat_review_result,
            "moer_reviews": moer_stat_reviews,
            "phase": ResearchPhase.REPORTING.value,
            "node_history": ["validate_results"]
        })

    except Exception as e:
        tracker.finish_step("09_validate_results", success=False, error=str(e))
        return merge_state_updates({
            "validation": {
                "overall_decision": "approved_with_warnings",
                "checks": [],
                "warnings": [{"type": "validation_error", "message": f"验证过程异常: {e}"}]
            },
            "validation_passed": True,
            "phase": ResearchPhase.REPORTING.value,
            "last_error": str(e),
            "error_history": [f"validate_results: {e}"],
            "node_history": ["validate_results"]
        })


def node_generate_report(state: AgentState) -> Dict[str, Any]:
    """
    生成报告 - 使用Reporting阶段的专业提示词
    属于ReAct模式的最终输出阶段

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: generate_report] 撰写学术报告...")

    llm = _get_llm()
    tracker = _get_tracker()

    tracker.start_step("10_generate_report", {})

    # 收集所有更新
    updates = {}

    try:
        plan = state.get("plan", {})
        cohort = state.get("cohort", {})
        validation = state.get("validation", {})
        citations = state.get("citations", [])
        tool_results = _load_full_tool_results(state)

        # ========== 使用LLM智能压缩tool_results ==========
        compressed_results = _compress_tool_results_with_llm(tool_results)

        # 使用专业的学术写作提示词
        prompts = get_node_prompt(
            "generate_report",
            question=state["question"],
            plan=plan,
            results_summary=compressed_results,
            validation=validation,
            citations=citations,
            cohort=cohort  # 传入真实被试数据
        )

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]

        # 动态调整报告 max_tokens
        num_tools = len(tool_results)
        base_tokens = 4000
        report_max_tokens = min(base_tokens + num_tools * 500, 16000)

        response = llm.chat(messages, max_tokens=report_max_tokens, temperature=0.7)
        # 从API响应中提取内容
        report_content = response["choices"][0]["message"]["content"]

        # 添加处理流程详情到报告末尾
        processing_details = _generate_processing_summary(tool_results, state)
        full_report = f"{report_content}\n\n{processing_details}"

        # ========== 报告审查 - 防止AI幻觉 + MoER聚合 ==========
        run_dir = tracker.run_dir if tracker.run_dir else OUTPUT_DIR / "runs" / state["run_id"]
        moer_reviews = state.get("moer_reviews", [])
        try:
            audit_result, audited_report = audit_report(
                report_content=full_report,
                cohort=cohort,
                tool_results=tool_results,
                plan=plan,
                run_dir=run_dir,
                llm_client=llm,
                moer_reviews=moer_reviews
            )

            # 使用审查后的报告（包含审查信息）
            full_report = audited_report

            # 保存审查结果
            report_dir = run_dir / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            audit_path = report_dir / "audit_result.json"
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(_deep_sanitize(audit_result, max_depth=12), f, indent=2, ensure_ascii=False, default=str)

            updates["audit_result"] = audit_result
            updates["audit_path"] = str(audit_path)
            tracker.add_step_artifact("10_generate_report", "audit_result.json", {"path": str(audit_path)})

            # 如果审查失败，记录警告
            if audit_result.get("status") == "FAILED":
                print(f"  [WARNING] 报告审查未通过，可能包含不准确内容")
                updates["audit_warning"] = audit_result.get("status_message", "审查未通过")
        except Exception as audit_error:
            print(f"  [WARNING] 报告审查失败: {audit_error}")
            # 审查失败不阻止流程，但记录警告
            updates["audit_error"] = str(audit_error)

        # 保存 MoER 累积审查记录（独立于 audit_report，确保始终保存）
        if moer_reviews:
            try:
                moer_report_dir = run_dir / "reports"
                moer_report_dir.mkdir(parents=True, exist_ok=True)
                moer_path = moer_report_dir / "moer_reviews.json"
                with open(moer_path, "w", encoding="utf-8") as f:
                    json.dump(_deep_sanitize(moer_reviews, max_depth=12), f, indent=2, ensure_ascii=False, default=str)
                print(f"  [MoER] 审查记录已保存: {moer_path} ({len(moer_reviews)} 条)")
                tracker.add_step_artifact("10_generate_report", "moer_reviews.json", {"path": str(moer_path)})
            except Exception as moer_save_err:
                print(f"  [MoER] 审查记录保存失败: {moer_save_err}")

        updates["report"] = full_report
        updates["phase"] = ResearchPhase.COMPLETED.value
        updates["node_history"] = ["generate_report"]

        # 保存报告
        run_dir = tracker.run_dir if tracker.run_dir else OUTPUT_DIR / "runs" / state["run_id"]
        report_dir = run_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "final_report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(full_report)

        # 同时保存处理流程的JSON文件（sanitize 防止循环引用）
        processing_log_path = report_dir / "processing_log.json"
        with open(processing_log_path, "w", encoding="utf-8") as f:
            json.dump(_deep_sanitize({
                "run_id": state["run_id"],
                "question": state["question"],
                "tool_results": tool_results,
                "cohort": cohort,
                "plan": plan
            }, max_depth=12), f, indent=2, ensure_ascii=False, default=str)

        updates["report_path"] = str(report_path)
        updates["processing_log_path"] = str(processing_log_path)

        tracker.add_step_artifact("10_generate_report", "final_report.md", {"path": str(report_path)})
        tracker.add_step_artifact("10_generate_report", "processing_log.json", {"path": str(processing_log_path)})
        tracker.finish_step("10_generate_report", outputs={"report_path": str(report_path)})

        word_count = len(full_report.split())
        print(f"  报告长度: {word_count} 词")
        print(f"  报告路径: {report_path}")
        print(f"  处理日志: {processing_log_path}")

        return merge_state_updates(updates)

    except Exception as e:
        import traceback
        tracker.finish_step("10_generate_report", success=False, error=str(e))
        return merge_state_updates({
            "last_error": f"{str(e)}\n{traceback.format_exc()}",
            "error_history": [f"generate_report: {e}"]
        })


def _compress_tool_results_with_llm(tool_results: List[Dict]) -> str:
    """
    使用LLM智能压缩tool_results，保留所有关键信息但减少token使用

    对于超大数据（>100000 tokens），先进行结构化压缩再使用LLM

    Args:
        tool_results: 完整的工具执行结果列表

    Returns:
        压缩后的总结文本
    """
    import json

    if not tool_results:
        return "无工具执行记录"

    # 估算当前token数量（粗略估算：1 token ≈ 4字符）
    try:
        full_json = json.dumps(tool_results, ensure_ascii=False, default=str)
        estimated_tokens = len(full_json) // 4
    except Exception as e:
        print(f"  [警告] JSON序列化失败: {e}，使用字符串长度估算")
        estimated_tokens = len(str(tool_results)) // 4

    # 如果小于5000 tokens，不需要压缩
    if estimated_tokens < 5000:
        try:
            return json.dumps(tool_results, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"  [警告] JSON序列化失败: {e}，返回字符串表示")
            return str(tool_results)

    print(f"  [智能压缩] tool_results过大 (~{estimated_tokens} tokens)，开始压缩...")

    # ========== 第一阶段：结构化提取关键信息 ==========
    # 无论数据多大，先提取核心信息到紧凑格式
    compact_records = []
    for i, result in enumerate(tool_results):
        try:
            record = {
                "idx": i + 1,
                "tool": result.get("tool", result.get("name", "unknown"))[:50],
                "step": result.get("step", "")[:50],
                "status": result.get("status", "unknown"),
            }

            # 提取关键参数（限制大小）
            params = result.get("params", result.get("parameters", {}))
            if isinstance(params, dict):
                record["params"] = {k: str(v)[:100] for k, v in list(params.items())[:5]}

            # 错误信息（限制长度）
            error = result.get("error", "")
            if error:
                record["error"] = str(error)[:200]

            # 输出文件数量
            outputs = result.get("outputs", {})
            if isinstance(outputs, dict):
                output_files = outputs.get("output_files", [])
                record["output_count"] = len(output_files) if isinstance(output_files, list) else 0

            # 是否缓存
            if result.get("cached") or result.get("from_cache"):
                record["cached"] = True

            compact_records.append(record)
        except Exception as e:
            compact_records.append({"idx": i + 1, "error": f"解析失败: {str(e)[:50]}"})

    # 计算压缩后大小
    try:
        compact_json = json.dumps(compact_records, ensure_ascii=False, default=str)
    except Exception as e:
        print(f"  [警告] compact_records序列化失败: {e}，使用字符串表示")
        compact_json = str(compact_records)
    compact_tokens = len(compact_json) // 4
    print(f"  [结构化压缩] {estimated_tokens} -> {compact_tokens} tokens")

    # ========== 第二阶段：如果仍然过大，进一步简化 ==========
    if compact_tokens > 50000:
        # 只保留最近30条和最早10条，中间用摘要
        if len(compact_records) > 50:
            early_records = compact_records[:10]
            recent_records = compact_records[-30:]
            skipped_count = len(compact_records) - 40

            simplified = early_records + [{"_summary": f"...省略中间 {skipped_count} 条记录..."}] + recent_records
            try:
                compact_json = json.dumps(simplified, ensure_ascii=False, default=str)
            except Exception as e:
                print(f"  [警告] simplified序列化失败: {e}")
                compact_json = str(simplified)
            compact_tokens = len(compact_json) // 4
            print(f"  [进一步简化] 保留首尾记录，压缩到 {compact_tokens} tokens")

    # ========== 第三阶段：如果仍然过大（>20000），直接使用fallback ==========
    if compact_tokens > 20000:
        print(f"  [压缩] 数据仍然过大，使用极简模式")
        return _fallback_truncate_tool_results(tool_results, max_items=20)

    # ========== 第四阶段：使用LLM进一步精炼（可选） ==========
    # 如果压缩后小于10000 tokens，尝试用LLM进一步精炼
    if compact_tokens < 10000:
        try:
            llm = _get_llm()
            prompt = f"""请将以下工具执行记录整理成简洁的执行摘要：

**输出格式：** 使用紧凑列表，每行一个工具，例如：
1. tool_name (step_name) - succeeded, 10 outputs
2. tool_name (step_name) - failed: error_message

**数据：**
{compact_json}

请输出摘要："""

            messages = [
                {"role": "system", "content": "你是一个数据整理助手，输出简洁的摘要。"},
                {"role": "user", "content": prompt}
            ]
            response = llm.chat(messages, temperature=0.2, max_tokens=3000)
            refined = response["choices"][0]["message"]["content"]

            final_tokens = len(refined) // 4
            print(f"  [LLM精炼] {compact_tokens} -> {final_tokens} tokens (减少 {100*(1-final_tokens/max(1,compact_tokens)):.1f}%)")
            return refined
        except Exception as e:
            print(f"  [LLM精炼] 失败: {e}，使用结构化结果")
            return compact_json

    return compact_json


def _fallback_truncate_tool_results(tool_results: List[Dict], max_items: int = 10) -> str:
    """
    简单截断tool_results（LLM压缩失败时的回退方案）

    Args:
        tool_results: 完整的工具执行结果列表
        max_items: 最多保留的项数

    Returns:
        截断后的JSON字符串
    """
    import json
    truncated = []
    for result in tool_results[-max_items:]:  # 只取最后N个
        truncated_result = {
            "tool": result.get("tool", ""),
            "step": result.get("step", ""),
            "status": result.get("status", ""),
            "duration": result.get("duration", 0),
            "error": result.get("error", "")[:200] if result.get("error") else None,
            "output_count": len(result.get("outputs", {}).get("output_files", [])) if "outputs" in result else 0
        }
        truncated.append(truncated_result)
    return json.dumps(truncated, ensure_ascii=False, indent=2, default=str)


def _scan_existing_processed_data(tracker=None) -> Dict[str, Any]:
    """
    扫描当前run中已存在的处理结果，避免迭代时重复处理

    Args:
        tracker: RunTracker实例，用于获取当前run目录

    Returns:
        已处理数据的摘要信息
    """
    existing_data = {
        "spm_segmentation": [],  # SPM分割结果
        "freesurfer_recon": [],  # FreeSurfer重建结果
        "dicom_converted": [],   # DICOM转换结果
        "statistics": [],        # 统计分析结果
        "completed_tasks": [],   # 已完成的任务列表
        "summary": ""
    }

    # 如果提供了tracker，优先扫描当前run的tools目录
    if tracker and tracker.run_dir:
        tools_dir = tracker.run_dir / "tools"
        if tools_dir.exists():
            # 扫描所有工具输出目录
            for tool_output_dir in tools_dir.iterdir():
                if not tool_output_dir.is_dir():
                    continue

                tool_dir_name = tool_output_dir.name  # 如: task_01_dicom_to_nifti_426d208e

                # 1. 检测DICOM转换
                if "dicom" in tool_dir_name.lower():
                    nifti_files = list(tool_output_dir.glob("**/*.nii")) + list(tool_output_dir.glob("**/*.nii.gz"))
                    if nifti_files:
                        existing_data["dicom_converted"].extend([str(f) for f in nifti_files[:20]])
                        existing_data["completed_tasks"].append("DICOM to NIfTI conversion")

                # 2. 检测SPM分割
                if "spm" in tool_dir_name.lower():
                    spm_patterns = ["c1*.nii", "c2*.nii", "c3*.nii", "wc1*.nii", "wc2*.nii", "mwc1*.nii"]
                    for pattern in spm_patterns:
                        files = list(tool_output_dir.glob(f"**/{pattern}"))
                        if files:
                            existing_data["spm_segmentation"].extend([str(f) for f in files[:10]])
                    if existing_data["spm_segmentation"]:
                        if "segment" in tool_dir_name.lower() or "vbm" in tool_dir_name.lower():
                            existing_data["completed_tasks"].append("VBM segmentation")
                        elif "normalize" in tool_dir_name.lower():
                            existing_data["completed_tasks"].append("SPM normalization")
                        elif "smooth" in tool_dir_name.lower():
                            existing_data["completed_tasks"].append("SPM smoothing")

                # 3. 检测FreeSurfer
                if "freesurfer" in tool_dir_name.lower():
                    subjects_dir = tool_output_dir
                    for subject_dir in subjects_dir.iterdir():
                        # 跳过FreeSurfer模板目录和符号链接
                        if subject_dir.name in ["fsaverage", "fsaverage_sym", "cvs_avg35",
                                                "cvs_avg35_inMNI152", "bert", "sample-001"]:
                            continue

                        try:
                            # 检查是否是目录（处理Windows符号链接错误）
                            if subject_dir.is_dir():
                                surf_dir = subject_dir / "surf"
                                if surf_dir.exists() and list(surf_dir.glob("lh.pial")):
                                    existing_data["freesurfer_recon"].append(subject_dir.name)
                        except (OSError, PermissionError) as e:
                            # Windows上访问符号链接可能失败，跳过
                            continue

                    if existing_data["freesurfer_recon"]:
                        existing_data["completed_tasks"].append("FreeSurfer reconstruction")

                # 4. 检测统计分析
                stats_files = list(tool_output_dir.glob("**/*stats*.csv")) + \
                             list(tool_output_dir.glob("**/*results*.csv")) + \
                             list(tool_output_dir.glob("**/*analysis*.csv"))
                if stats_files:
                    existing_data["statistics"].extend([str(f) for f in stats_files[:10]])
                    existing_data["completed_tasks"].append("Statistical analysis")

    # 生成摘要
    summary_parts = []
    if existing_data["completed_tasks"]:
        unique_tasks = list(set(existing_data["completed_tasks"]))
        summary_parts.append(f"已完成的步骤: {', '.join(unique_tasks)}")

    if existing_data["spm_segmentation"]:
        summary_parts.append(f"- SPM分割: {len(existing_data['spm_segmentation'])}个组织概率图文件")
    if existing_data["freesurfer_recon"]:
        summary_parts.append(f"- FreeSurfer: {len(existing_data['freesurfer_recon'])}个被试已重建")
    if existing_data["dicom_converted"]:
        summary_parts.append(f"- DICOM转换: {len(existing_data['dicom_converted'])}个NIfTI文件")
    if existing_data["statistics"]:
        summary_parts.append(f"- 统计结果: {len(existing_data['statistics'])}个结果文件")

    existing_data["summary"] = "\n".join(summary_parts) if summary_parts else "无已处理数据"

    return existing_data


def _load_full_tool_results(state: Dict) -> List[Dict]:
    """从磁盘加载完整的tool_results历史，回退到state中的数据"""
    tool_results = state.get("tool_results", [])
    tracker = _get_tracker()
    if tracker and tracker.run_dir:
        history_path = Path(tracker.run_dir) / "tool_results_history.json"
        if history_path.exists():
            try:
                full = json.loads(history_path.read_text(encoding='utf-8'))
                if len(full) > len(tool_results):
                    return full
            except Exception:
                pass
    return tool_results


def _prepare_state_for_iteration(state: AgentState) -> Dict[str, Any]:
    """
    为新迭代准备状态 - 压缩历史数据以避免token超限

    核心思想：
    - 保留规划和反馈信息（用于改进方案）
    - 压缩工具执行结果（保留关键摘要）
    - 清理不需要的中间数据
    - **扫描已处理数据，避免重复处理**

    Args:
        state: 当前Agent状态

    Returns:
        包含更新字段的字典（LangGraph 会自动合并到 state）
    """
    import json

    iteration_count = state.get("iteration_count", 0)
    print(f"\n  [迭代准备] 为第 {iteration_count + 1} 次迭代压缩历史数据...")

    # 收集所有更新
    updates = {}

    # 0. 扫描已存在的处理结果（关键！避免重复处理）
    # 获取tracker以扫描当前run的工具输出
    tracker = _get_tracker()
    existing_data = _scan_existing_processed_data(tracker)
    updates["existing_processed_data"] = existing_data
    print(f"  [已处理数据] {existing_data['summary']}")

    # 1. 压缩tool_results - 只保留摘要
    tool_results = state.get("tool_results", [])
    if tool_results:
        original_size = len(json.dumps(tool_results, ensure_ascii=False, default=str))

        # 将所有 tool_results 追加保存到磁盘（完整历史）
        tracker = _get_tracker()
        if tracker and tracker.run_dir:
            history_path = Path(tracker.run_dir) / "tool_results_history.json"
            existing = []
            if history_path.exists():
                try:
                    existing = json.loads(history_path.read_text(encoding='utf-8'))
                except Exception:
                    pass
            # 去重追加：用call_id或内容哈希去重
            def _get_result_id(r):
                cid = r.get("call_id")
                if cid:
                    return cid
                return hashlib.md5(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()

            existing_ids = {_get_result_id(r) for r in existing if isinstance(r, dict)}
            for r in tool_results:
                if isinstance(r, dict):
                    rid = _get_result_id(r)
                    if rid not in existing_ids:
                        existing.append(r)
                        existing_ids.add(rid)
            history_path.write_text(
                json.dumps(existing, ensure_ascii=False, default=str, indent=2),
                encoding='utf-8'
            )
            print(f"  [tool_results] 已保存 {len(existing)} 条完整历史到 tool_results_history.json")

        # 生成紧凑摘要
        compressed_summary = _compress_tool_results_with_llm(tool_results)

        # 保存摘要（不对 tool_results 赋值，因为 operator.add 会追加而非替换）
        updates["tool_results_summary"] = compressed_summary
        updates["tool_results_count"] = len(tool_results)

        print(f"  [tool_results] 原始 {original_size} 字符, 已保存摘要 (state中列表自然累积)")

    # 2. 压缩generated_codes - 只保留元数据
    generated_codes = state.get("generated_codes", [])
    if generated_codes and len(generated_codes) > 3:
        # 只保留代码元数据，不保留完整代码
        updates["generated_codes"] = [
            {
                "timestamp": c.get("timestamp", ""),
                "success": c.get("success", False),
                "code_path": c.get("code_path", ""),
                "attempts": c.get("attempts", 0)
            }
            for c in generated_codes[-3:]  # 只保留最近3条
        ]
        print(f"  [generated_codes] 保留最近3条元数据")

    # 3. 压缩validation结果 - 保留关键信息
    validation = state.get("validation", {})
    if validation and len(json.dumps(validation, ensure_ascii=False, default=str)) > 5000:
        # 只保留状态和主要错误
        updates["validation"] = {
            "valid": validation.get("valid", False),
            "error_count": len(validation.get("errors", [])),
            "warning_count": len(validation.get("warnings", [])),
            "key_errors": validation.get("errors", [])[:3],
            "compressed": True
        }
        print(f"  [validation] 压缩验证结果")

    # 4. 清理不需要的大字段
    large_fields_to_clear = [
        "raw_outputs",
        "intermediate_data",
        "debug_info",
        "full_response"
    ]
    for field in large_fields_to_clear:
        if field in state and state[field]:
            updates[field] = None
            print(f"  [清理] 清除 {field}")

    # 5. error_history / node_history 使用 operator.add，无法通过返回值截断
    # 磁盘持久化已保存完整历史，state 中的列表让它自然累积
    error_history = state.get("error_history", [])
    if len(error_history) > 5:
        print(f"  [error_history] 当前 {len(error_history)} 条 (operator.add, 自然累积)")

    node_history = state.get("node_history", [])
    if len(node_history) > 10:
        print(f"  [node_history] 当前 {len(node_history)} 条 (operator.add, 自然累积)")

    print(f"  [迭代准备] 完成")

    return updates


def _generate_processing_summary(tool_results: List[Dict], state: Dict) -> str:
    """
    生成处理流程的详细总结

    Args:
        tool_results: 工具执行结果列表
        state: Agent状态

    Returns:
        Markdown格式的处理总结
    """
    # 使用辅助函数加载完整历史
    tool_results = _load_full_tool_results(state)

    lines = [
        "\n\n---",
        "\n## 附录：影像处理流程详情\n",
        f"**运行ID**: `{state.get('run_id', 'N/A')}`\n",
        f"**总步骤数**: {len(tool_results)}\n"
    ]

    for i, result in enumerate(tool_results, 1):
        tool = result.get("tool", "unknown")
        step_name = result.get("step", "")
        status = result.get("status", "unknown")
        duration = result.get("duration", 0)
        output_dir = result.get("output_dir", "")

        lines.append(f"\n### 步骤 {i}: {step_name}\n")
        lines.append(f"- **工具**: `{tool}`")
        lines.append(f"- **状态**: {status}")
        lines.append(f"- **耗时**: {duration:.2f}秒")
        lines.append(f"- **输出目录**: `{output_dir}`")

        # 输出文件列表
        output_files = result.get("outputs", {}).get("output_files", [])
        if output_files:
            lines.append(f"- **输出文件** ({len(output_files)}个):")
            for j, filepath in enumerate(output_files[:5], 1):
                filename = Path(filepath).name
                lines.append(f"  {j}. `{filename}`")
            if len(output_files) > 5:
                lines.append(f"  ... 还有 {len(output_files) - 5} 个文件")
        else:
            lines.append("- **输出文件**: 无")

        # 错误信息
        if result.get("error"):
            lines.append(f"- **错误**: {result.get('error')}")

        # 其他输出信息
        other_outputs = {k: v for k, v in result.get("outputs", {}).items() if k != "output_files"}
        if other_outputs:
            lines.append(f"- **其他输出**:")
            for key, value in other_outputs.items():
                if isinstance(value, (str, int, float, bool)):
                    lines.append(f"  - {key}: `{value}`")

    # 添加数据路径说明
    lines.append("\n### 数据访问说明\n")
    lines.append("所有处理结果保存在运行目录下的 `tools/` 子目录中。")
    lines.append("每个步骤的输出都保存在独立的子目录中，可通过上述路径直接访问。")

    return "\n".join(lines)


def _collect_successful_task_results(state: AgentState) -> Dict[str, Dict[str, Any]]:
    """
    收集当前迭代成功完成的任务结果，供下次迭代继承

    Args:
        state: 当前Agent状态

    Returns:
        字典 {task_description: result_dict}
    """
    results = {}

    # 从tool_results中提取成功的任务
    tool_results = state.get("tool_results", [])
    for r in tool_results:
        if not isinstance(r, dict):
            continue

        # 检查是否成功完成
        status = r.get("status", "")
        success = r.get("success", False)
        if status == "completed" or success:
            # 使用任务描述作为key
            desc = r.get("description", "") or r.get("task_id", "") or r.get("step", "")
            if desc:
                results[desc] = {
                    "status": "completed",
                    "success": True,
                    "result": r.get("result", r.get("outputs", {})),
                    "output_files": r.get("output_files", [])
                }

    # 同时检查task_manager中的任务状态（如果有）
    tracker = _get_tracker()
    if tracker and tracker.run_dir:
        from src.agent.task_manager import TaskManager, TaskStatus
        task_manager = TaskManager(tracker.run_dir)
        if task_manager.tasks:
            for task in task_manager.tasks:
                if task.status in [TaskStatus.COMPLETED, TaskStatus.SKIPPED]:
                    if task.description and task.description not in results:
                        results[task.description] = {
                            "status": "completed",
                            "success": True,
                            "result": task.result or {},
                            "output_files": task.result.get("output_files", []) if task.result else []
                        }

    return results


def node_reflect_and_fix(state: AgentState) -> Dict[str, Any]:
    """
    反思和修复 - 使用Reflection阶段的专业提示词
    属于ReAct模式的Reflection阶段 - 最关键的学习和改进环节

    Returns:
        包含更新字段的字典（不再直接修改 state）
    """
    print(f"\n[NODE: reflect_and_fix] 深度反思和问题诊断...")

    llm = _get_llm()
    error = state.get("last_error", "")
    error_count = state.get("error_count", 0) + 1

    if error_count >= state.get("max_retries", 3):
        print(f"  超过最大重试次数({error_count})，终止执行")
        return merge_state_updates({
            "error_count": error_count,
            "phase": ResearchPhase.ERROR.value,
            "node_history": ["reflect_and_fix:max_retries"]
        })

    # ========== 使用LLM智能压缩tool_results，防止token超长 ==========
    tool_results = state.get("tool_results", [])
    compressed_results = _compress_tool_results_with_llm(tool_results)

    # 使用专业的反思提示词
    prompts = get_node_prompt(
        "reflect_and_fix",
        error=error,
        state={
            "phase": state.get("phase"),
            "completed_nodes": state.get("node_history", []),
            "tool_results_summary": compressed_results,
            "validation": state.get("validation", {})
        },
        history=state.get("error_history", [])[-5:]
    )

    messages = [
        {"role": "system", "content": prompts["system"]},
        {"role": "user", "content": prompts["user"]}
    ]

    reflection = None
    try:
        reflection = llm.generate_json(messages, temperature=0.3, max_tokens=8192)

        action = reflection.get("recommended_action", "plan_a")
        severity = reflection.get("impact_assessment", {}).get("severity", "medium")
        root_cause = reflection.get("root_cause_analysis", {}).get("root_cause", "未知")

        print(f"  根本原因: {root_cause}")
        print(f"  严重程度: {severity}")
        print(f"  推荐方案: {action}")
        print(f"  学到教训: {len(reflection.get('lessons_learned', []))} 条")

    except Exception as e:
        print(f"  [错误] 反思过程失败: {e}")

        if "400" in str(e) or "401" in str(e) or "429" in str(e) or "API" in str(e):
            print(f"  [致命] LLM API调用失败，无法继续")
            return merge_state_updates({
                "error_count": error_count,
                "phase": ResearchPhase.ERROR.value,
                "last_error": f"LLM API错误: {e}",
                "node_history": ["reflect_and_fix:api_error"]
            })

        print(f"  使用默认重试策略")
        reflection = {
            "recommended_action": "plan_a",
            "impact_assessment": {"severity": "medium"},
            "root_cause_analysis": {"root_cause": f"反思失败: {e}"}
        }

    # 返回更新字典，通过 LangGraph 正式的状态更新机制
    # 关键：清除 last_error 防止下游路由误判形成死循环
    updates = {
        "error_count": error_count,
        "last_error": "",
        "reflection": reflection,
        "lessons_learned": reflection.get("lessons_learned", []) if reflection else [],
        "node_history": ["reflect_and_fix"]
    }

    # PLANNING 阶段：清除 plan 和 plan_review，使 generate_plan 能重新生成
    phase = state.get("phase", "")
    if phase == ResearchPhase.PLANNING.value and state.get("plan"):
        updates["plan"] = {}
        updates["plan_review"] = {}
        print(f"  [清除计划] 将重新生成研究计划")

    return merge_state_updates(updates)


def node_evaluate_iteration(state: AgentState) -> Dict[str, Any]:
    """
    评估迭代质量 - 判断是否需要更深入的分析
    基于科学研究标准评估结果的完整性和深度

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: evaluate_iteration] 评估研究结果质量...")

    llm = _get_llm()
    tracker = _get_tracker()

    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)

    print(f"  当前迭代: {iteration_count + 1}/{max_iterations}")

    # 如果已达到最大迭代次数，直接结束
    if iteration_count >= max_iterations:
        print(f"  已达到最大迭代次数，结束迭代")
        return merge_state_updates({
            "needs_deeper_analysis": False,
            "iteration_feedback": "已达到最大迭代次数",
            "node_history": ["evaluate_iteration"]
        })

    tracker.start_step("11_evaluate_iteration", {"iteration": iteration_count + 1})

    try:
        # ========== 使用LLM智能压缩tool_results ==========
        tool_results = state.get("tool_results", [])
        compressed_results = _compress_tool_results_with_llm(tool_results)

        # ========== 压缩report以避免token超限 ==========
        full_report = state.get("report", "")
        # 只保留前10000字符和后2000字符（摘要和结论部分）
        if len(full_report) > 12000:
            report_summary = full_report[:10000] + "\n\n[...报告中间部分已省略...]\n\n" + full_report[-2000:]
            print(f"  [报告压缩] {len(full_report)} -> {len(report_summary)} 字符")
        else:
            report_summary = full_report

        # 构建评估提示词
        prompts = get_node_prompt(
            "evaluate_iteration",
            question=state["question"],
            plan=state.get("plan", {}),
            report=report_summary,  # 使用压缩后的报告
            results_summary=compressed_results,
            validation=state.get("validation", {}),
            iteration_count=iteration_count,
            iteration_history=state.get("iteration_history", [])
        )

        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]}
        ]

        # 设置任务类型以使用K2-Thinking推理模型
        llm.set_task_type("evaluate_iteration")

        # 使用LLM评估科学质量
        # 评估报告包含feedback、suggestions、strengths、weaknesses等详细内容，需要较大token限制
        evaluation = llm.generate_json(messages, temperature=0.3, max_tokens=8192)

        # 提取评估结果
        quality_score = evaluation.get("quality_score", 0.0)
        needs_deeper = evaluation.get("needs_deeper_analysis", False)
        feedback = evaluation.get("feedback", "")
        suggestions = evaluation.get("suggestions", [])
        strengths = evaluation.get("strengths", [])
        weaknesses = evaluation.get("weaknesses", [])

        # 记录本次迭代
        iteration_record = {
            "iteration": iteration_count + 1,
            "quality_score": quality_score,
            "needs_deeper": needs_deeper,
            "feedback": feedback,
            "suggestions": suggestions,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "timestamp": datetime.now().isoformat()
        }

        print(f"  质量评分: {quality_score:.2f}/10")
        print(f"  优势: {len(strengths)} 项")
        print(f"  不足: {len(weaknesses)} 项")
        print(f"  需要深化: {'是' if needs_deeper else '否'}")

        if needs_deeper and suggestions:
            print(f"  改进建议:")
            for i, suggestion in enumerate(suggestions[:3], 1):
                print(f"    {i}. {suggestion}")

        tracker.add_step_artifact("11_evaluate_iteration", "iteration_evaluation.json", iteration_record)
        tracker.finish_step("11_evaluate_iteration", outputs={
            "quality_score": quality_score,
            "needs_deeper": needs_deeper
        })

        # ========== Ue: Execution-driven update ==========
        # 基于工具执行结果更新动态知识图谱
        try:
            from src.knowledge.dynamic_knowledge_graph import get_dynamic_kg
            dynamic_kg = get_dynamic_kg()

            # 从state中提取疾病信息
            parsed_intent = state.get("parsed_intent", {})
            disease_info = parsed_intent.get("disease_info", {})
            disease_type = disease_info.get("disease_type") if disease_info else None

            if disease_type and disease_type.lower() not in ["null", "none", ""]:
                # 遍历tool_results，更新每个工具的有效性
                update_count = 0
                for result in tool_results:
                    tool_name = result.get("tool", result.get("name", ""))
                    status = result.get("status", "")
                    success = (status == "completed" or result.get("success", False))

                    # 使用质量评分作为质量指标
                    tool_quality = result.get("quality_score", quality_score / 10.0)

                    if tool_name:
                        dynamic_kg.update_execution_driven(
                            tool=tool_name,
                            disease=disease_type,
                            success=success,
                            quality_score=tool_quality
                        )
                        update_count += 1

                if update_count > 0:
                    print(f"  [Ue更新] 基于 {update_count} 个工具执行结果更新了知识图谱")

                    # 显示更新统计
                    stats = dynamic_kg.get_update_statistics()
                    print(f"  [KG统计] 总更新: {stats['total_updates']}, "
                          f"ROI记录: {stats['disease_roi_records']}, "
                          f"工具记录: {stats['tool_disease_records']}")
        except Exception as e:
            print(f"  [Ue更新] 失败: {e}")

        # === 收集成功任务结果，供下次迭代继承 ===
        previous_successful_results = _collect_successful_task_results(state)
        if previous_successful_results:
            print(f"  [迭代继承] 收集了 {len(previous_successful_results)} 个成功任务结果")

        # === Pipeline 复合技能学习 ===
        if not needs_deeper:
            try:
                parsed_intent = state.get("parsed_intent", {})
                disease_info = parsed_intent.get("disease_info", {})
                disease_type = disease_info.get("disease_type", "") if disease_info else ""
                _get_tool_skill().learn_pipeline_skill(
                    tool_results=tool_results,
                    disease=disease_type or "",
                    question=state.get("question", "")
                )
            except Exception as pl_err:
                print(f"  [Pipeline学习] 警告: {pl_err}")

        return merge_state_updates({
            "scientific_quality_score": quality_score,
            "needs_deeper_analysis": needs_deeper,
            "iteration_feedback": feedback,
            "iteration_suggestions": suggestions,
            "iteration_history": [iteration_record],
            "iteration_count": iteration_count + 1,
            "previous_successful_results": previous_successful_results,  # 新增：传递成功结果
            "node_history": ["evaluate_iteration"]
        })

    except Exception as e:
        print(f"  评估失败: {e}，尝试继续迭代")
        tracker.finish_step("11_evaluate_iteration", success=False, error=str(e))
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 5)
        # 如果还有迭代余量，继续；否则结束
        can_retry = (iteration_count + 1) < max_iterations
        return merge_state_updates({
            "needs_deeper_analysis": can_retry,
            "iteration_feedback": f"评估失败: {e}，{'尝试重新迭代' if can_retry else '已达最大迭代次数'}",
            "iteration_count": iteration_count + 1,
            "last_error": str(e),
            "node_history": ["evaluate_iteration"]
        })


def node_execute_next_task(state: AgentState) -> Dict[str, Any]:
    """
    执行下一个待处理任务
    这是新的任务列表执行模式，每次只执行一个任务，避免token超限

    Returns:
        包含 tool_results, node_history 等状态更新的字典
    """
    print(f"\n[NODE: execute_next_task] 执行下一个任务...")

    tracker = _get_tracker()
    registry = _get_registry()

    from src.agent.task_manager import TaskManager, TaskStatus
    task_manager = TaskManager(tracker.run_dir)

    # 获取下一个待执行任务
    next_task = task_manager.get_next_task()

    if not next_task:
        print("  没有待执行的任务")
        # 返回更新字典，不直接修改state
        return merge_state_updates({
            "node_history": ["execute_next_task:no_task"]
        })

    print(f"  [任务] {next_task.task_id}: {next_task.description}")
    print(f"  [工具] {next_task.tool_name}")

    # 更新任务状态为执行中
    task_manager.update_task_status(next_task.task_id, TaskStatus.IN_PROGRESS)

    try:
        # 准备工具调用
        tool_name = next_task.tool_name
        call_id = next_task.task_id

        # 处理特殊的概念性工具（不是实际执行的工具）
        conceptual_tools = ["reuse_existing", "skip", "manual", "placeholder"]
        if tool_name in conceptual_tools:
            print(f"  [概念性工具] {tool_name} - 自动跳过")
            # 检查是否有已存在的相关数据
            existing_data_found = False
            existing_files = []

            # 扫描之前的工具输出
            tools_dir = tracker.run_dir / "tools"
            if tools_dir.exists():
                # 根据任务描述关键词查找相关输出
                task_keywords = next_task.description.lower().split() if next_task.description else []
                relevant_keywords = ["vbm", "freesurfer", "spm", "fsl", "preprocess", "segment"]

                for subdir in tools_dir.iterdir():
                    if subdir.is_dir():
                        # 检查目录名是否匹配任务关键词
                        dir_lower = subdir.name.lower()
                        if any(kw in dir_lower for kw in relevant_keywords):
                            # 找到相关输出目录
                            nii_files = list(subdir.rglob("*.nii")) + list(subdir.rglob("*.nii.gz"))
                            if nii_files:
                                existing_data_found = True
                                existing_files = [str(f) for f in nii_files[:50]]
                                print(f"    [发现] 相关数据: {subdir.name} ({len(nii_files)} 个文件)")
                                break

            if existing_data_found:
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.COMPLETED,
                    result={"reused": True, "output_files": existing_files}
                )
                print(f"  [OK] 复用已有数据，任务完成 ({len(existing_files)} 个文件)")

                progress = task_manager.get_progress()
                print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

                # 关键修复: 将复用的数据添加到tool_results，供后续任务使用
                return merge_state_updates({
                    "tool_results": [{
                        "call_id": call_id,
                        "task_id": next_task.task_id,
                        "tool_name": f"reused_{tool_name}",
                        "modality": getattr(next_task, 'modality', None),  # 添加模态信息
                        "status": "succeeded",
                        "outputs": {"output_files": existing_files, "reused": True},
                        "error": None
                    }],
                    "node_history": [f"execute_next_task:{call_id}:reused"]
                })
            else:
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.SKIPPED,
                    error="概念性工具，无已存在数据可复用"
                )
                print(f"  [SKIP] 无已存在数据，跳过任务")

                progress = task_manager.get_progress()
                print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

                return merge_state_updates({
                    "node_history": [f"execute_next_task:{call_id}:skipped"]
                })

        # 构建输出目录
        run_id = tracker.run_dir.name  # 从run_dir路径获取run_id
        output_dir = tracker.run_dir / "tools" / f"{call_id}_{tool_name}_{run_id[:8]}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 准备context
        plan = state.get("plan", {})
        plan_modalities = plan.get("modalities", ["anat"])
        context = {
            "cohort": state.get("cohort", {}),
            "research_question": state.get("question", ""),
            "modality": plan_modalities[0] if plan_modalities else "anat"
        }

        # **关键修复**: 动态填充任务输入
        inputs = next_task.inputs.copy() if next_task.inputs else {}

        # 根据工具类型填充输入
        if tool_name == "dicom_to_nifti":
            # DICOM转换: 从cohort获取被试目录
            # **关键修复**: 多层模态推断机制

            # 1. 定义关键词（备用）
            dwi_keywords = ["dti", "dwi", "diffusion", "tractography", "tensor", "dsi", "fiber", "eddy", "bedpost"]
            anat_keywords = ["vbm", "segment", "cortical", "surface", "volumetric", "morphometry", "t1", "recon-all", "thickness", "freesurfer"]
            func_keywords = ["fmri", "functional", "bold", "resting", "task", "activation", "connectivity", "alff", "reho", "fc"]

            # 2. 多层模态推断
            target_modality = None

            # 优先级1：任务显式指定的modality
            if hasattr(next_task, 'modality') and next_task.modality:
                target_modality = next_task.modality
                print(f"    [模态检测] 任务显式指定: {target_modality}")

            # 优先级2：根据直接下游任务推断（依赖当前任务的任务）
            if not target_modality:
                current_task_id = next_task.task_id
                for task in task_manager.tasks:
                    # 检查是否依赖当前任务
                    if hasattr(task, 'depends_on') and current_task_id in (task.depends_on or []):
                        # 使用infer_required_modality函数推断
                        inferred = infer_required_modality(task.tool_name, task.params)
                        if inferred:
                            target_modality = inferred
                            print(f"    [模态检测] 下游任务 {task.tool_name} 需要 {target_modality} 模态")
                            break

            # 优先级3：遍历下游任务链（修复：只看相关任务，而非所有任务）
            if not target_modality:
                # 使用get_downstream_tasks获取完整的下游任务链
                downstream_tasks = get_downstream_tasks(next_task.task_id, task_manager.tasks)
                for task in downstream_tasks:
                    task_tool = task.tool_name or ""
                    # 使用infer_required_modality函数
                    inferred = infer_required_modality(task_tool, task.params)
                    if inferred:
                        target_modality = inferred
                        print(f"    [模态检测] 下游链任务 {task_tool} → 使用{target_modality}模态")
                        break

            # 优先级4：根据当前任务描述关键词推断
            if not target_modality:
                task_desc_lower = (next_task.description or "").lower()
                if any(kw in task_desc_lower for kw in dwi_keywords):
                    target_modality = "dwi"
                    print(f"    [模态检测] 从任务描述推断: dwi")
                elif any(kw in task_desc_lower for kw in func_keywords):
                    target_modality = "func"
                    print(f"    [模态检测] 从任务描述推断: func")
                elif any(kw in task_desc_lower for kw in anat_keywords):
                    target_modality = "anat"
                    print(f"    [模态检测] 从任务描述推断: anat")

            # 优先级5：明确的默认值（anat最常用）
            if not target_modality:
                target_modality = "anat"
                print(f"    [模态检测] 无法推断，默认使用: anat")

            # 3. 获取被试目录
            cohort = state.get("cohort", {})
            input_dirs = []
            groups = cohort.get("groups", {})

            for group_name, group_data in groups.items():
                subject_ids = group_data.get("subjects", [])
                for subject_id in subject_ids:
                    # 使用确定的目标模态
                    subject_dir = DATA_DIR / group_name / target_modality / subject_id
                    if subject_dir.exists():
                        input_dirs.append(str(subject_dir))
                    else:
                        # 如果指定模态不存在，打印警告并尝试其他模态
                        print(f"    [警告] {subject_id} 没有 {target_modality} 数据目录，尝试其他模态...")
                        # 备用：按优先级查找其他模态
                        fallback_modalities = [m for m in ["anat", "func", "dwi"] if m != target_modality]
                        for fallback in fallback_modalities:
                            fallback_dir = DATA_DIR / group_name / fallback / subject_id
                            if fallback_dir.exists():
                                input_dirs.append(str(fallback_dir))
                                print(f"    [备用] 使用 {fallback} 模态: {fallback_dir}")
                                break

            inputs["input_dirs"] = input_dirs
            print(f"    [输入] {len(input_dirs)} 个被试目录 (模态: {target_modality})")

        elif tool_name == "spm_analysis":
            # SPM分析: 从之前的任务输出获取文件
            input_files = []

            # **关键修复**: 使用infer_required_modality推断SPM所需模态
            target_modality = getattr(next_task, 'modality', None)
            current_analysis_type = next_task.params.get("analysis_type", "")
            if not target_modality:
                target_modality = infer_required_modality("spm_analysis", next_task.params)
                if target_modality:
                    print(f"    [模态推断] 根据SPM分析类型 '{current_analysis_type}' 推断需要 {target_modality} 模态")

            if target_modality:
                print(f"    [模态匹配] SPM需要 {target_modality} 模态数据")

            # 定义有效的输入工具
            # **关键修复**: 对于fMRI预处理，优先从dicom_to_nifti或前一个fMRI处理步骤获取输入
            valid_source_tools = ["dicom", "dicom_to_nifti", "mri_convert", "reused", "spm"]
            valid_extensions = [".nii", ".nii.gz"]

            # fMRI预处理步骤的有效前序步骤映射
            # 重要：强制执行标准fMRI预处理顺序，不允许跳过步骤
            # 标准顺序: slice_timing -> realign -> coregister -> normalize -> smooth
            fmri_preproc_chain = {
                "slice_timing": ["dicom_to_nifti", "dicom"],  # slice_timing是第一步，从原始数据开始
                "realign": ["slice_timing"],  # realign必须在slice_timing之后，不允许跳过
                "coregister": ["realign"],  # coregister必须在realign之后
                "normalize": ["coregister", "vbm_segment"],  # normalize在coregister之后，或VBM分割之后
                "smooth": ["normalize"]  # smooth必须在normalize之后
            }

            # 检查是否有工具结果
            if "tool_results" in state and state["tool_results"]:
                for tool_result in reversed(state["tool_results"]):
                    if tool_result.get("status") != "succeeded":
                        continue

                    source_tool = tool_result.get("tool_name", "")
                    if not source_tool or not any(valid in source_tool for valid in valid_source_tools):
                        continue

                    outputs = tool_result.get("outputs", {})

                    # **新增**: 检查上游SPM任务的analysis_type是否兼容
                    if "spm" in source_tool.lower():
                        source_analysis_type = outputs.get("analysis_type", "")
                        # 如果当前是fMRI预处理步骤，检查上游是否是有效的前序步骤
                        if current_analysis_type in fmri_preproc_chain:
                            valid_predecessors = fmri_preproc_chain[current_analysis_type]
                            # VBM分割的输出不应该用于fMRI预处理
                            if source_analysis_type in ["vbm_segment", "vbm_dartel"] and current_analysis_type in ["slice_timing", "realign", "coregister"]:
                                print(f"    [跳过] {source_tool} 是VBM分析，不适用于fMRI预处理 '{current_analysis_type}'")
                                continue
                            # 检查是否是有效的前序步骤
                            if source_analysis_type and source_analysis_type not in valid_predecessors and "dicom" not in source_tool.lower():
                                print(f"    [跳过] {source_tool} (analysis_type={source_analysis_type}) 不是 '{current_analysis_type}' 的有效前序步骤")
                                continue

                    # 多层推断源模态
                    source_modality = tool_result.get("modality")

                    # 层级1：从outputs推断
                    if not source_modality:
                        has_bvec = outputs.get("bvec_files") and len(outputs.get("bvec_files", [])) > 0
                        has_bval = outputs.get("bval_files") and len(outputs.get("bval_files", [])) > 0
                        if has_bvec or has_bval:
                            source_modality = "dwi"
                            print(f"    [模态推断] {source_tool} 包含bvec/bval文件 → dwi模态")

                    # 层级2：从task_id查找
                    if not source_modality:
                        source_task_id = tool_result.get("task_id") or tool_result.get("call_id")
                        if source_task_id and task_manager:
                            source_task = task_manager.get_task(source_task_id)
                            if source_task and hasattr(source_task, 'modality') and source_task.modality:
                                source_modality = source_task.modality
                                print(f"    [模态推断] 从任务 {source_task_id} 获取模态: {source_modality}")

                    # 层级3：从路径推断
                    if not source_modality:
                        candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                        if candidate_files:
                            sample_path = str(candidate_files[0]).lower()
                            if '/dwi/' in sample_path or '\\dwi\\' in sample_path or '_dwi' in sample_path:
                                source_modality = "dwi"
                            elif '/anat/' in sample_path or '\\anat\\' in sample_path or '_t1' in sample_path:
                                source_modality = "anat"
                            elif '/func/' in sample_path or '\\func\\' in sample_path or '_bold' in sample_path:
                                source_modality = "func"
                            if source_modality:
                                print(f"    [模态推断] 从路径推断: {source_modality}模态")

                    # 模态匹配检查
                    if target_modality and source_modality:
                        if source_modality != target_modality:
                            print(f"    [跳过] {source_tool} 模态不匹配 (源: {source_modality}, 需要: {target_modality})")
                            continue
                        print(f"    [模态匹配] {source_tool} 模态匹配: {source_modality}")
                    elif target_modality and not source_modality:
                        print(f"    [警告] {source_tool} 无法确定模态，跳过模态检查")

                    candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                    nifti_files = [f for f in candidate_files
                                   if any(str(f).lower().endswith(ext) for ext in valid_extensions)]

                    if nifti_files:
                        # **关键修复**: 对于smooth和normalize步骤，需要过滤只保留每个被试的最终预处理4D文件
                        if current_analysis_type in ["smooth", "normalize"]:
                            import re

                            # 提取被试ID的辅助函数
                            def extract_subject_id_spm(filepath):
                                """从文件名提取被试ID，去除SPM前缀"""
                                filename = Path(filepath).stem
                                # 移除常见的SPM前缀: s(smooth), r(realign), a(slice_timing), w(normalize), m(mean)
                                cleaned = re.sub(r'^[srawm]+', '', filename)
                                if len(cleaned) < 3:
                                    cleaned = filename
                                return cleaned

                            # 按被试ID分组
                            subject_files = {}
                            for f in nifti_files:
                                fname = Path(f).name.lower()
                                # 跳过mean图像（3D，不适合fMRI分析）
                                if 'mean' in fname:
                                    continue

                                subj_id = extract_subject_id_spm(f)
                                if subj_id not in subject_files:
                                    subject_files[subj_id] = []
                                subject_files[subj_id].append(f)

                            # 每个被试选择最优文件
                            final_files = []
                            for subj_id, files in subject_files.items():
                                if not files:
                                    continue

                                best_file = None
                                if current_analysis_type == "smooth":
                                    # 对于smooth，优先选择normalize后的文件（w前缀），其次是realigned文件（r前缀）
                                    # 标准fMRI预处理顺序: slice_timing -> realign -> coregister -> normalize -> smooth
                                    for f in files:
                                        fname = Path(f).name
                                        # 跳过已经smoothed的文件（s前缀但不是sw前缀）
                                        if fname.startswith('s') and not fname.startswith('sw'):
                                            continue
                                        # 最优先选择normalized的文件（w前缀，如wrHC1_0001.nii）
                                        if fname.startswith('w'):
                                            best_file = f
                                            break
                                    # 如果没有normalized文件，选择realigned文件
                                    if not best_file:
                                        for f in files:
                                            fname = Path(f).name
                                            if fname.startswith('s'):
                                                continue
                                            if fname.startswith('r') and not fname.startswith('rr'):
                                                best_file = f
                                                break
                                    if not best_file:
                                        for f in files:
                                            if not Path(f).name.startswith('s'):
                                                best_file = f
                                                break
                                elif current_analysis_type == "normalize":
                                    # 对于normalize，优先选择realigned的文件（r前缀，非rr）
                                    for f in files:
                                        fname = Path(f).name
                                        # 跳过已经normalized的文件（rr前缀表示realigned+normalized）
                                        if fname.startswith('rr') or fname.startswith('w'):
                                            continue
                                        # 优先选择realigned的文件
                                        if fname.startswith('r'):
                                            best_file = f
                                            break
                                    if not best_file:
                                        for f in files:
                                            fname = Path(f).name
                                            if not fname.startswith('rr') and not fname.startswith('w'):
                                                best_file = f
                                                break

                                if not best_file and files:
                                    best_file = files[0]
                                if best_file:
                                    final_files.append(best_file)

                            if final_files:
                                print(f"    [{current_analysis_type}过滤] 原始 {len(nifti_files)} 个文件 → 每被试1个文件 {len(final_files)} 个")
                                input_files = final_files
                            else:
                                input_files = nifti_files
                        else:
                            input_files = nifti_files

                        matched_modality = source_modality or "unknown"
                        print(f"    [输入来源] 从 {source_tool} 获取 {len(input_files)} 个NIfTI文件 (模态: {matched_modality})")
                        break

            # 如果没有找到，使用cohort中的NIfTI文件
            if not input_files:
                cohort = state.get("cohort", {})
                groups = cohort.get("groups", {})

                # **关键修复**: 优先使用target_modality对应的目录
                modality_order = [target_modality] if target_modality else []
                modality_order.extend([m for m in ["anat", "func", "dwi"] if m not in modality_order])

                for group_name, group_data in groups.items():
                    subject_ids = group_data.get("subjects", [])
                    for subject_id in subject_ids:
                        for modality in modality_order:
                            subject_dir = DATA_DIR / group_name / modality / subject_id
                            if subject_dir.exists():
                                nii_files = list(subject_dir.rglob("*.nii"))
                                input_files.extend([str(f) for f in nii_files])
                                break

            # **关键逻辑**: 智能过滤 - 区分原始扫描和SPM中间输出
            if input_files:
                import re
                # 定义SPM处理文件的前缀（完整列表）
                spm_prefixes = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6',  # 组织概率图
                               'wc1', 'wc2', 'wc3', 'wc4', 'wc5', 'wc6',  # warped组织图
                               'mwc1', 'mwc2', 'mwc3',  # modulated warped
                               'swc1', 'swc2', 'swc3',  # smoothed warped
                               'y_', 'iy_',  # 变形场
                               'w', 'm', 'r', 'a', 's']  # 其他SPM前缀

                # 检查每个文件是否有SPM前缀
                files_with_spm_prefix = 0
                files_without_spm_prefix = []
                for file_path in input_files:
                    filename = Path(file_path).name
                    if any(filename.startswith(prefix) for prefix in spm_prefixes):
                        files_with_spm_prefix += 1
                    else:
                        files_without_spm_prefix.append(file_path)

                # 判断文件类型
                if files_with_spm_prefix == len(input_files):
                    # 所有文件都有SPM前缀 → SPM中间输出 → 不过滤
                    print(f"    [检测] SPM中间输出 ({len(input_files)} 个文件，全部有SPM前缀)")
                    print(f"    [保留] 保留所有 {len(input_files)} 个文件")
                elif files_with_spm_prefix == 0:
                    # 所有文件都没有SPM前缀 → 原始扫描 → 应用过滤
                    print(f"    [检测] 原始扫描数据 ({len(input_files)} 个文件)")
                    # 只对结构像应用过滤，功能像不过滤（保留所有4D文件）
                    if target_modality == "anat":
                        filtered_files = _filter_structural_images(input_files)
                        if len(filtered_files) < len(input_files):
                            print(f"    [智能过滤] {len(input_files)} 个文件 → {len(filtered_files)} 个结构像")
                            print(f"    [过滤详情] 每个被试保留一个主要T1结构像")
                        input_files = filtered_files
                    else:
                        print(f"    [保留] 功能像/弥散像不进行过滤")
                else:
                    # 混合情况（部分有SPM前缀，部分没有）
                    print(f"    [警告] 检测到混合文件类型 ({files_with_spm_prefix}/{len(input_files)} 有SPM前缀)")
                    # **关键修复**: 对于fMRI预处理的早期步骤，只保留原始文件
                    if current_analysis_type in ["slice_timing", "realign"] and files_without_spm_prefix:
                        print(f"    [过滤] fMRI预处理 '{current_analysis_type}' 需要原始4D数据，过滤掉SPM中间输出")
                        input_files = files_without_spm_prefix
                        print(f"    [保留] 保留 {len(input_files)} 个原始文件")
                    else:
                        print(f"    [保留] 保留所有 {len(input_files)} 个文件以避免数据丢失")

            inputs["input_files"] = input_files
            print(f"    [输入] {len(input_files)} 个NIfTI文件")

        elif tool_name == "python_stats":
            # Python统计: 从cohort和之前的输出获取数据
            cohort = state.get("cohort", {})

            # 验证cohort结构
            if not cohort:
                print(f"    [警告] cohort为空，尝试从数据目录构建...")
                # 使用自动发现构建cohort
                if DATA_DIR.exists():
                    data_structure = _get_data_structure()
                    discovered = data_structure.get("groups", {})
                    cohort = {
                        "groups": list(discovered.keys()),
                        "subjects": {}
                    }
                    for group_name, group_info in discovered.items():
                        cohort["subjects"][group_name] = group_info.get("subjects", [])
                        print(f"    [构建cohort] {group_name}: {len(cohort['subjects'][group_name])} 个被试")

            # 确保cohort至少有groups和subjects字段
            if cohort and "groups" not in cohort:
                cohort["groups"] = list(cohort.get("subjects", {}).keys())
            if cohort and "subjects" not in cohort:
                cohort["subjects"] = {}

            inputs["cohort"] = cohort
            inputs["modality"] = getattr(next_task, 'modality', None) or plan_modalities[0] if plan_modalities else "anat"

            # 定义有效的处理工具和文件扩展名
            # 注意: 使用子字符串匹配，"reused" 匹配复用的数据
            valid_source_tools = ["spm", "freesurfer", "fsl", "dsi_studio", "dicom", "reused"]
            valid_extensions = [".nii", ".nii.gz", ".csv", ".stats", ".txt"]

            # 获取处理后的文件
            if "tool_results" in state and state["tool_results"]:
                for tool_result in reversed(state["tool_results"]):
                    if tool_result.get("status") == "succeeded":
                        # 检查工具名称 - 必须是有效的处理工具
                        source_tool = tool_result.get("tool_name", "")
                        # 修复: 如果source_tool为空或不匹配有效工具，跳过
                        if not source_tool or not any(valid in source_tool for valid in valid_source_tools):
                            continue

                        outputs = tool_result.get("outputs", {})
                        candidate_files = outputs.get("output_files", [])

                        # 过滤只保留有效的数据文件（排除PNG、JSON等非数据文件）
                        valid_files = [f for f in candidate_files
                                       if any(f.lower().endswith(ext) for ext in valid_extensions)]

                        if valid_files:
                            inputs["processed_files"] = valid_files
                            print(f"    [输入来源] 从 {source_tool} 获取 {len(valid_files)} 个数据文件")
                            break

            print(f"    [输入] cohort包含 {len(inputs.get('cohort', {}))} 个组")

        elif tool_name == "freesurfer_analysis":
            # FreeSurfer分析: 从之前的任务输出获取T1文件
            input_files = []
            subject_ids = []

            # **关键修复**: 获取当前任务需要的模态 (FreeSurfer需要anat)
            target_modality = getattr(next_task, 'modality', None) or "anat"  # 默认为anat
            print(f"    [模态匹配] FreeSurfer需要 {target_modality} 模态数据")

            # 定义有效的输入工具（只从这些工具的输出中获取文件）
            # 注意: FreeSurfer需要原始T1图像，不能使用SPM处理后的输出(c1*,c2*等)
            # 所以这里不包含 "spm"
            valid_source_tools = ["dicom", "mri_convert", "reused", "dicom_to_nifti"]
            # 排除SPM处理后的文件前缀
            excluded_prefixes = ["c1", "c2", "c3", "c4", "c5", "c6", "wc", "swc", "mwc", "y_", "iy_"]
            # 定义有效的NIfTI扩展名
            valid_extensions = [".nii", ".nii.gz"]

            # 检查是否有工具结果（如DICOM转换的输出）
            if "tool_results" in state and state["tool_results"]:
                for tool_result in reversed(state["tool_results"]):
                    if tool_result.get("status") == "succeeded":
                        # 检查工具名称是否是有效的输入来源
                        source_tool = tool_result.get("tool_name", "")
                        # 修复: 如果source_tool为空或不匹配有效工具，跳过
                        if not source_tool or not any(valid in source_tool for valid in valid_source_tools):
                            continue  # 跳过非相关工具的输出

                        outputs = tool_result.get("outputs", {})

                        # **关键修复**: 多层推断源模态
                        source_modality = tool_result.get("modality")

                        # 层级1：如果没有modality字段，从outputs推断
                        if not source_modality:
                            has_bvec = outputs.get("bvec_files") and len(outputs.get("bvec_files", [])) > 0
                            has_bval = outputs.get("bval_files") and len(outputs.get("bval_files", [])) > 0
                            if has_bvec or has_bval:
                                source_modality = "dwi"
                                print(f"    [模态推断] {source_tool} 包含bvec/bval文件 → dwi模态")

                        # 层级2：如果还没有，从task_id查找task_manager
                        if not source_modality:
                            source_task_id = tool_result.get("task_id") or tool_result.get("call_id")
                            if source_task_id and task_manager:
                                source_task = task_manager.get_task(source_task_id)
                                if source_task and hasattr(source_task, 'modality') and source_task.modality:
                                    source_modality = source_task.modality
                                    print(f"    [模态推断] 从任务 {source_task_id} 获取模态: {source_modality}")

                        # 层级3：如果还没有，尝试从output_files路径推断
                        if not source_modality:
                            candidate_files_check = outputs.get("nifti_files", outputs.get("output_files", []))
                            if candidate_files_check:
                                sample_path = str(candidate_files_check[0]).lower()
                                if '/dwi/' in sample_path or '\\dwi\\' in sample_path or '_dwi' in sample_path:
                                    source_modality = "dwi"
                                elif '/anat/' in sample_path or '\\anat\\' in sample_path or '_t1' in sample_path:
                                    source_modality = "anat"
                                elif '/func/' in sample_path or '\\func\\' in sample_path or '_bold' in sample_path:
                                    source_modality = "func"
                                if source_modality:
                                    print(f"    [模态推断] 从路径推断: {source_modality}模态")

                        # 进行模态匹配检查
                        if target_modality and source_modality:
                            if source_modality != target_modality:
                                print(f"    [跳过] {source_tool} 模态不匹配 (源: {source_modality}, 需要: {target_modality})")
                                continue
                            print(f"    [模态匹配] {source_tool} 模态匹配: {source_modality}")
                        elif target_modality and not source_modality:
                            print(f"    [警告] {source_tool} 无法确定模态，跳过模态检查")

                        candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))

                        # 过滤只保留NIfTI文件，并排除SPM处理后的文件
                        def is_valid_t1_file(filepath):
                            """检查是否是有效的原始T1文件"""
                            f = str(filepath).lower()
                            # 检查扩展名
                            if not any(f.endswith(ext) for ext in valid_extensions):
                                return False
                            # 检查是否是SPM处理后的文件
                            filename = Path(filepath).name.lower()
                            if any(filename.startswith(prefix) for prefix in excluded_prefixes):
                                return False
                            return True

                        nifti_files = [f for f in candidate_files if is_valid_t1_file(f)]

                        if nifti_files:
                            input_files = nifti_files
                            matched_modality = source_modality or "unknown"
                            print(f"    [输入来源] 从 {source_tool} 获取 {len(nifti_files)} 个原始T1文件 (模态: {matched_modality})")
                            break

            # 如果没有找到，检查cohort中的数据
            if not input_files:
                print(f"    [输入来源] 检查cohort原始数据...")
                cohort = state.get("cohort", {})
                groups = cohort.get("groups", {})

                dicom_found = False

                for group_name, group_data in groups.items():
                    subject_id_list = group_data.get("subjects", [])
                    for subject_id in subject_id_list:
                        # 查找anat模态的NIfTI文件
                        subject_dir = DATA_DIR / group_name / "anat" / subject_id
                        if subject_dir.exists():
                            nii_files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
                            if nii_files:
                                input_files.append(str(nii_files[0]))
                                subject_ids.append(subject_id)
                            else:
                                # 检查是否有DICOM文件
                                dcm_files = list(subject_dir.glob("*.dcm"))
                                if not dcm_files:
                                    # 可能是无后缀的DICOM文件
                                    all_files = [f for f in subject_dir.iterdir() if f.is_file()]
                                    if all_files:
                                        dicom_found = True
                                else:
                                    dicom_found = True

                if not input_files and dicom_found:
                    print(f"    [警告] 原始数据是DICOM格式，需要先执行dicom_to_nifti转换!")
                    print(f"    [建议] 请确保任务列表中包含DICOM转换步骤")

            # 如果有输入文件但没有subject_ids，从文件路径提取
            if input_files and not subject_ids:
                for f in input_files:
                    # 从路径中提取subject_id（父目录名）
                    subject_id = Path(f).parent.name
                    subject_ids.append(subject_id)

            # 如果没有指定subject_ids，从文件名生成
            if not subject_ids and input_files:
                subject_ids = [Path(f).stem.replace('.nii', '') for f in input_files]

            inputs["input_files"] = input_files
            if subject_ids:
                inputs["subject_ids"] = subject_ids

            print(f"    [输入] {len(input_files)} 个T1文件, {len(subject_ids)} 个被试")

            # 自动填充FreeSurfer参数
            if "command" not in next_task.params:
                next_task.params["command"] = "recon-all"
            if "directive" not in next_task.params:
                # 根据任务描述推断directive
                description_lower = (next_task.description or "").lower()
                if "quick" in description_lower or "fast" in description_lower or "autorecon1" in description_lower:
                    next_task.params["directive"] = "-autorecon1"
                elif "clinical" in description_lower:
                    next_task.params["command"] = "recon-all-clinical"
                    next_task.params["directive"] = ""
                else:
                    next_task.params["directive"] = "-all"  # 完整重建

        elif tool_name == "fsl_analysis":
            # FSL分析: 根据命令类型选择正确的输入源
            input_files = []
            bvec_files = []  # 额外获取bvec文件
            bval_files = []  # 额外获取bval文件
            mask_files = []  # 脑mask文件（用于eddy/dtifit）

            # **关键修复1**: 使用infer_required_modality自动推断target_modality
            fsl_command = next_task.params.get("command", "")
            target_modality = getattr(next_task, 'modality', None)
            if not target_modality:
                # 从FSL命令推断所需模态
                target_modality = infer_required_modality("fsl_analysis", next_task.params)
                if target_modality:
                    print(f"    [模态推断] 根据FSL命令 '{fsl_command}' 推断需要 {target_modality} 模态")

            if target_modality:
                print(f"    [模态匹配] 任务需要 {target_modality} 模态数据")

            # **关键修复2**: eddy/dtifit需要特殊处理
            # 这些命令需要：原始DWI文件 + bvec + bval，而不是bet处理后的文件
            dwi_commands = [
                "eddy", "dtifit", "bedpostx", "probtrackx"
            ]
            needs_original_dwi = fsl_command in dwi_commands

            # TBSS命令的特殊处理
            tbss_commands = {
                "tbss_1_preproc": "needs_fa_images",  # 需要FA图像（来自dtifit）
                "tbss_2_reg": "no_new_inputs",        # 使用TBSS工作目录
                "tbss_3_postreg": "no_new_inputs",    # 使用TBSS工作目录
                "tbss_4_prestats": "no_new_inputs"    # 使用TBSS工作目录
            }

            if needs_original_dwi:
                print(f"    [DWI命令] {fsl_command} 需要原始DWI文件和bvec/bval")

            # 定义有效的输入工具（根据命令类型调整）
            valid_extensions = [".nii", ".nii.gz"]

            # 检查是否有工具结果
            if "tool_results" in state and state["tool_results"]:
                # **关键修复3**: 对于DWI命令，先找原始数据源（dicom_to_nifti），再找mask（bet）
                if needs_original_dwi:
                    # 第一遍：从dicom_to_nifti获取原始DWI文件和bvec/bval
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue
                        source_tool = tool_result.get("tool_name", "")

                        # 只从dicom转换工具获取原始数据
                        if not any(src in source_tool for src in ["dicom", "dicom_to_nifti", "mri_convert"]):
                            continue

                        outputs = tool_result.get("outputs", {})

                        # 检查是否是DWI数据（必须有bvec/bval）
                        has_bvec = outputs.get("bvec_files") and len(outputs.get("bvec_files", [])) > 0
                        has_bval = outputs.get("bval_files") and len(outputs.get("bval_files", [])) > 0

                        if has_bvec and has_bval:
                            candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                            nifti_files = [f for f in candidate_files
                                          if any(str(f).lower().endswith(ext) for ext in valid_extensions)]

                            if nifti_files:
                                input_files = nifti_files
                                bvec_files = outputs.get("bvec_files", [])
                                bval_files = outputs.get("bval_files", [])
                                print(f"    [输入来源] 从 {source_tool} 获取 {len(nifti_files)} 个原始DWI文件")
                                print(f"    [DWI] 同时获取 {len(bvec_files)} 个bvec, {len(bval_files)} 个bval文件")
                                break

                    # 第二遍：从bet获取mask（可选）
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue
                        source_tool = tool_result.get("tool_name", "")

                        # 只从FSL工具获取mask
                        if "fsl" not in source_tool:
                            continue

                        outputs = tool_result.get("outputs", {})
                        candidate_files = outputs.get("output_files", outputs.get("nifti_files", []))

                        # 查找mask文件
                        for f in candidate_files:
                            f_lower = str(f).lower()
                            if '_mask' in f_lower or '_brain_mask' in f_lower:
                                mask_files.append(str(f))

                        if mask_files:
                            print(f"    [Mask] 从 {source_tool} 获取 {len(mask_files)} 个mask文件")
                            break

                elif fsl_command in tbss_commands:
                    # TBSS命令的特殊处理
                    input_mode = tbss_commands[fsl_command]

                    if input_mode == "needs_fa_images":
                        # tbss_1_preproc: 从dtifit的输出中获取FA图像
                        print(f"    [TBSS] {fsl_command} 需要从dtifit获取FA图像")

                        for tool_result in reversed(state["tool_results"]):
                            if tool_result.get("status") != "succeeded":
                                continue

                            source_tool = tool_result.get("tool_name", "")
                            outputs = tool_result.get("outputs", {})

                            # 检查是否是fsl_analysis (dtifit)的输出
                            if "fsl" in source_tool:
                                command_used = outputs.get("command", "")
                                if command_used == "dtifit":
                                    # 获取FA文件
                                    candidate_files = outputs.get("output_files", [])
                                    fa_files = [f for f in candidate_files
                                               if "_FA." in str(f) or "_fa." in str(f)]

                                    if fa_files:
                                        input_files = fa_files
                                        print(f"    [输入来源] 从dtifit获取 {len(fa_files)} 个FA图像")
                                        break

                        if not input_files:
                            print(f"    [错误] 未找到dtifit的FA输出。TBSS需要先运行dtifit。")

                    elif input_mode == "no_new_inputs":
                        # tbss_2/3/4: 不需要从tool_results收集输入
                        # 它们直接操作TBSS工作目录，但需要找到tbss_1_preproc创建的目录
                        print(f"    [TBSS] {fsl_command} 使用TBSS工作目录，无需新输入")
                        input_files = []  # 明确设置为空

                        # 查找tbss_1_preproc的输出目录（包含TBSS子目录）
                        tbss_base_dir = None
                        for tool_result in reversed(state["tool_results"]):
                            if tool_result.get("status") != "succeeded":
                                continue
                            outputs = tool_result.get("outputs", {})
                            command_used = outputs.get("command", "")
                            if command_used == "tbss_1_preproc":
                                # tbss_1_preproc的output_dir就是TBSS的基础目录
                                tbss_output_dir = outputs.get("output_dir")
                                if tbss_output_dir:
                                    tbss_base_dir = tbss_output_dir
                                    print(f"    [TBSS] 找到tbss_1_preproc的TBSS目录: {tbss_base_dir}")
                                    break

                        if tbss_base_dir:
                            # 将TBSS目录路径传递给工具
                            next_task.params["tbss_base_dir"] = tbss_base_dir
                        else:
                            print(f"    [警告] 未找到tbss_1_preproc的输出。{fsl_command}可能会失败。")

                else:
                    # 非DWI/TBSS命令：使用标准的模态匹配逻辑
                    valid_source_tools = ["dicom", "dicom_to_nifti", "dicom_convert", "mri_convert", "reused", "fsl"]

                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue

                        source_tool = tool_result.get("tool_name", "")
                        if not source_tool or not any(valid in source_tool for valid in valid_source_tools):
                            continue

                        outputs = tool_result.get("outputs", {})

                        # 多层推断源模态
                        source_modality = tool_result.get("modality")

                        # 层级1：从outputs推断
                        if not source_modality:
                            has_bvec = outputs.get("bvec_files") and len(outputs.get("bvec_files", [])) > 0
                            has_bval = outputs.get("bval_files") and len(outputs.get("bval_files", [])) > 0
                            if has_bvec or has_bval:
                                source_modality = "dwi"
                                print(f"    [模态推断] {source_tool} 包含bvec/bval文件 → dwi模态")

                        # 层级2：从task_id查找
                        if not source_modality:
                            source_task_id = tool_result.get("task_id") or tool_result.get("call_id")
                            if source_task_id and task_manager:
                                source_task = task_manager.get_task(source_task_id)
                                if source_task and hasattr(source_task, 'modality') and source_task.modality:
                                    source_modality = source_task.modality
                                    print(f"    [模态推断] 从任务 {source_task_id} 获取模态: {source_modality}")

                        # 层级3：从路径推断
                        if not source_modality:
                            candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                            if candidate_files:
                                sample_path = str(candidate_files[0]).lower()
                                if '/dwi/' in sample_path or '\\dwi\\' in sample_path or '_dwi' in sample_path:
                                    source_modality = "dwi"
                                elif '/anat/' in sample_path or '\\anat\\' in sample_path or '_t1' in sample_path:
                                    source_modality = "anat"
                                elif '/func/' in sample_path or '\\func\\' in sample_path or '_bold' in sample_path:
                                    source_modality = "func"
                                if source_modality:
                                    print(f"    [模态推断] 从路径推断: {source_modality}模态")

                        # 模态匹配检查
                        if target_modality and source_modality:
                            if source_modality != target_modality:
                                print(f"    [跳过] {source_tool} 模态不匹配 (源: {source_modality}, 需要: {target_modality})")
                                continue
                            print(f"    [模态匹配] {source_tool} 模态匹配: {source_modality}")
                        elif target_modality and not source_modality:
                            print(f"    [警告] {source_tool} 无法确定模态，跳过模态检查")

                        candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                        nifti_files = [f for f in candidate_files
                                       if any(str(f).lower().endswith(ext) for ext in valid_extensions)]

                        if nifti_files:
                            input_files = nifti_files
                            bvec_files = outputs.get("bvec_files", [])
                            bval_files = outputs.get("bval_files", [])
                            matched_modality = source_modality or "unknown"
                            print(f"    [输入来源] 从 {source_tool} 获取 {len(nifti_files)} 个NIfTI文件 (模态: {matched_modality})")
                            if bvec_files and bval_files:
                                print(f"    [DWI] 同时获取 {len(bvec_files)} 个bvec, {len(bval_files)} 个bval文件")
                            break

            # 如果没有找到，检查cohort中的数据
            # 注意：tbss_2/3/4 不需要从cohort收集文件，它们使用TBSS工作目录
            tbss_no_input_commands = ["tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats"]
            if not input_files and fsl_command not in tbss_no_input_commands:
                print(f"    [输入来源] 检查cohort原始数据...")
                cohort = state.get("cohort", {})
                groups = cohort.get("groups", {})
                dicom_found = False

                # **关键修复4**: 优先使用target_modality对应的目录
                modality_order = [target_modality] if target_modality else []
                modality_order.extend([m for m in ["anat", "dwi", "func"] if m not in modality_order])

                for group_name, group_data in groups.items():
                    subject_id_list = group_data.get("subjects", [])
                    for subject_id in subject_id_list:
                        for modality in modality_order:
                            subject_dir = DATA_DIR / group_name / modality / subject_id
                            if subject_dir.exists():
                                nii_files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
                                if nii_files:
                                    input_files.extend([str(f) for f in nii_files])
                                    # 同时查找bvec/bval
                                    if modality == "dwi":
                                        bvec_list = list(subject_dir.rglob("*.bvec"))
                                        bval_list = list(subject_dir.rglob("*.bval"))
                                        bvec_files.extend([str(f) for f in bvec_list])
                                        bval_files.extend([str(f) for f in bval_list])
                                    break
                                else:
                                    dcm_files = list(subject_dir.glob("*.dcm"))
                                    if dcm_files:
                                        dicom_found = True

                if not input_files and dicom_found:
                    print(f"    [警告] 原始数据是DICOM格式，需要先执行dicom_to_nifti转换!")

            inputs["input_files"] = input_files
            # 为DWI处理提供bvec/bval文件
            if bvec_files:
                inputs["bvec_files"] = bvec_files
            if bval_files:
                inputs["bval_files"] = bval_files
            # 为eddy/dtifit提供mask文件
            if mask_files:
                inputs["mask_files"] = mask_files

            print(f"    [输入] {len(input_files)} 个文件")
            if bvec_files and bval_files:
                print(f"    [DWI] 包含 {len(bvec_files)} 个bvec, {len(bval_files)} 个bval文件")
            if mask_files:
                print(f"    [Mask] 包含 {len(mask_files)} 个mask文件")

        elif tool_name == "dpabi_analysis":
            # DPABI分析: 从之前的任务输出获取预处理后的fMRI文件
            input_files = []

            # DPABI需要func模态数据
            target_modality = getattr(next_task, 'modality', None) or "func"
            print(f"    [模态匹配] DPABI需要 {target_modality} 模态数据")

            # 定义有效的输入工具（DPABI需要预处理后的fMRI数据）
            # 优先从SPM预处理输出获取，其次从DICOM转换获取
            valid_source_tools = ["spm", "dicom", "dicom_to_nifti", "mri_convert", "reused"]
            valid_extensions = [".nii", ".nii.gz"]

            # 检查是否有工具结果
            if "tool_results" in state and state["tool_results"]:
                for tool_result in reversed(state["tool_results"]):
                    if tool_result.get("status") != "succeeded":
                        continue

                    source_tool = tool_result.get("tool_name", "")
                    if not source_tool or not any(valid in source_tool for valid in valid_source_tools):
                        continue

                    outputs = tool_result.get("outputs", {})

                    # 多层推断源模态
                    source_modality = tool_result.get("modality")

                    # 层级1：从task_id查找
                    if not source_modality:
                        source_task_id = tool_result.get("task_id") or tool_result.get("call_id")
                        if source_task_id and task_manager:
                            source_task = task_manager.get_task(source_task_id)
                            if source_task and hasattr(source_task, 'modality') and source_task.modality:
                                source_modality = source_task.modality
                                print(f"    [模态推断] 从任务 {source_task_id} 获取模态: {source_modality}")

                    # 层级2：从路径推断
                    if not source_modality:
                        candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                        if candidate_files:
                            sample_path = str(candidate_files[0]).lower()
                            if '/func/' in sample_path or '\\func\\' in sample_path or '_bold' in sample_path:
                                source_modality = "func"
                            elif '/anat/' in sample_path or '\\anat\\' in sample_path or '_t1' in sample_path:
                                source_modality = "anat"
                            elif '/dwi/' in sample_path or '\\dwi\\' in sample_path or '_dwi' in sample_path:
                                source_modality = "dwi"
                            if source_modality:
                                print(f"    [模态推断] 从路径推断: {source_modality}模态")

                    # 模态匹配检查
                    if target_modality and source_modality:
                        if source_modality != target_modality:
                            print(f"    [跳过] {source_tool} 模态不匹配 (源: {source_modality}, 需要: {target_modality})")
                            continue
                        print(f"    [模态匹配] {source_tool} 模态匹配: {source_modality}")

                    candidate_files = outputs.get("nifti_files", outputs.get("output_files", []))
                    nifti_files = [f for f in candidate_files
                                   if any(str(f).lower().endswith(ext) for ext in valid_extensions)]

                    if nifti_files:
                        # **关键修复**: DPABI需要每个被试只有一个最终预处理文件
                        # 过滤规则:
                        # 1. 优先选择平滑后的文件 (s前缀)
                        # 2. 排除mean图像 (mean前缀)
                        # 3. 每个被试只保留一个文件

                        # 提取被试ID的辅助函数
                        def extract_subject_id(filepath):
                            """从文件名提取被试ID，去除前缀(s, r, a, w等)"""
                            import re
                            filename = Path(filepath).stem
                            # 移除常见的SPM前缀: s(smooth), r(realign), a(slice_timing), w(normalize)
                            # 但保留被试ID如 HC1_0001, SCA3_0002
                            cleaned = re.sub(r'^[srawm]+', '', filename)
                            # 如果清理后为空或太短，使用原始文件名
                            if len(cleaned) < 3:
                                cleaned = filename
                            return cleaned

                        # 按被试ID分组
                        subject_files = {}
                        for f in nifti_files:
                            fname = Path(f).name.lower()
                            # 跳过mean图像（3D，不适合fMRI分析）
                            if 'mean' in fname:
                                continue

                            subj_id = extract_subject_id(f)
                            if subj_id not in subject_files:
                                subject_files[subj_id] = []
                            subject_files[subj_id].append(f)

                        # 每个被试选择最优文件（优先选择处理程度最高的）
                        final_files = []
                        for subj_id, files in subject_files.items():
                            if not files:
                                continue
                            # 优先级: s前缀(smoothed) > r前缀(realigned) > 无前缀
                            best_file = None
                            for f in files:
                                fname = Path(f).name
                                if fname.startswith('s') and not fname.startswith('sr'):
                                    # 最高优先级: 平滑后的文件
                                    best_file = f
                                    break
                            if not best_file:
                                # 次优先级: 选择第一个文件
                                best_file = files[0]
                            final_files.append(best_file)

                        if final_files:
                            input_files = final_files
                            matched_modality = source_modality or "unknown"
                            print(f"    [输入来源] 从 {source_tool} 获取 {len(final_files)} 个NIfTI文件 (模态: {matched_modality})")
                            print(f"    [被试过滤] 原始 {len(nifti_files)} 个文件 → 每被试1个最终文件 {len(final_files)} 个")
                            break

            # 如果没有找到，检查cohort中的func数据
            if not input_files:
                print(f"    [输入来源] 检查cohort原始func数据...")
                cohort = state.get("cohort", {})
                groups = cohort.get("groups", {})

                for group_name, group_data in groups.items():
                    subject_id_list = group_data.get("subjects", [])
                    for subject_id in subject_id_list:
                        # 查找func模态的NIfTI文件
                        subject_dir = DATA_DIR / group_name / "func" / subject_id
                        if subject_dir.exists():
                            nii_files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
                            if nii_files:
                                input_files.extend([str(f) for f in nii_files])

            inputs["input_files"] = input_files
            print(f"    [输入] {len(input_files)} 个fMRI文件")

            # 自动填充DPABI参数
            if "analysis_type" not in next_task.params:
                # 根据任务描述推断analysis_type
                description_lower = (next_task.description or "").lower()
                if "alff" in description_lower:
                    next_task.params["analysis_type"] = "alff"
                elif "falff" in description_lower:
                    next_task.params["analysis_type"] = "falff"
                elif "reho" in description_lower:
                    next_task.params["analysis_type"] = "reho"
                elif "fc" in description_lower or "connectivity" in description_lower:
                    next_task.params["analysis_type"] = "fc_seed"
                elif "degree" in description_lower or "centrality" in description_lower:
                    next_task.params["analysis_type"] = "degree_centrality"
                else:
                    # 默认使用alff
                    next_task.params["analysis_type"] = "alff"
                print(f"    [智能参数] 推断analysis_type={next_task.params['analysis_type']}")

        elif tool_name == "dsi_studio_analysis":
            # DSI Studio分析: 根据action类型选择不同的输入源
            input_files = []
            bvec_files = []  # DWI需要bvec文件
            bval_files = []  # DWI需要bval文件

            # **自动修正**: LLM有时会错误地使用command而不是action参数
            # 如果params中有command但没有action，自动转换
            if "command" in next_task.params and "action" not in next_task.params:
                old_command = next_task.params.pop("command")
                next_task.params["action"] = old_command
                print(f"    [参数修正] DSI Studio: command=\"{old_command}\" → action=\"{old_command}\"")

            # 获取DSI Studio的action类型
            dsi_action = next_task.params.get("action", "src")
            # **关键修复**: 确保action参数在params中（如果缺失则设置默认值）
            # 否则filled_params.copy()时action不会被包含，导致"缺少必需参数: action"错误
            if "action" not in next_task.params:
                next_task.params["action"] = dsi_action
            print(f"    [DSI Studio] action={dsi_action}")

            # **关键修复**: 根据action类型选择不同的输入源
            # - src: 需要原始4D DWI + bvec + bval，只从dicom_to_nifti或eddy获取
            # - rec: 需要.src文件，从之前的dsi_studio src获取
            # - trk: 需要.fib.gz文件，从之前的dsi_studio rec获取

            if dsi_action == "src":
                # **src action需要原始DWI数据 + bvec + bval**
                # 优先从eddy输出获取（已校正的DWI + rotated bvecs）
                # 否则从dicom_to_nifti获取原始DWI
                print(f"    [DSI Studio src] 需要原始4D DWI + bvec + bval")

                # 定义src action的有效输入源 - 只接受原始DWI，不接受DTI maps
                # 注意：不包含fsl（因为fsl dtifit输出是DTI maps，不是原始DWI）
                valid_src_tools = ["dicom", "dicom_to_nifti", "dicom_convert", "mri_convert", "reused"]

                # 检查是否有eddy输出（优先使用eddy校正后的数据）
                eddy_outputs = None
                dicom_outputs = None

                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue

                        source_tool = tool_result.get("tool_name", "")
                        outputs = tool_result.get("outputs", {})

                        # 检查是否是eddy输出
                        if "fsl" in source_tool:
                            # 检查是否是eddy命令
                            source_task_id = tool_result.get("task_id") or tool_result.get("call_id")
                            if source_task_id and task_manager:
                                source_task = task_manager.get_task(source_task_id)
                                if source_task and source_task.params.get("command") == "eddy":
                                    # 找到eddy输出，获取_eddy.nii.gz文件
                                    eddy_files = [f for f in outputs.get("output_files", [])
                                                 if "_eddy.nii.gz" in str(f) and "outlier" not in str(f).lower()]
                                    if eddy_files:
                                        eddy_outputs = {
                                            "nifti_files": eddy_files,
                                            "tool_name": source_tool
                                        }
                                        print(f"    [找到eddy输出] {len(eddy_files)} 个eddy校正后的DWI文件")

                        # 检查是否是dicom_to_nifti输出
                        elif any(valid in source_tool for valid in valid_src_tools):
                            # 检查是否包含bvec/bval（表明是DWI数据）
                            has_bvec = outputs.get("bvec_files") and len(outputs.get("bvec_files", [])) > 0
                            has_bval = outputs.get("bval_files") and len(outputs.get("bval_files", [])) > 0
                            if has_bvec and has_bval:
                                dicom_outputs = {
                                    "nifti_files": outputs.get("output_files", outputs.get("nifti_files", [])),
                                    "bvec_files": outputs.get("bvec_files", []),
                                    "bval_files": outputs.get("bval_files", []),
                                    "tool_name": source_tool
                                }
                                print(f"    [找到dicom输出] {len(dicom_outputs['nifti_files'])} 个原始DWI文件")

                # 选择输入源：优先eddy（但需要配合原始bvec/bval）
                if eddy_outputs and dicom_outputs:
                    # 使用eddy校正后的DWI + 原始bval + 寻找rotated_bvecs
                    input_files = eddy_outputs["nifti_files"]
                    bval_files = dicom_outputs["bval_files"]  # bval不变

                    # 查找eddy_rotated_bvecs - 多种可能的命名方式
                    eddy_bvec_files = []
                    for eddy_file in input_files:
                        eddy_path = Path(eddy_file)
                        # 获取基础名称（去掉.nii.gz）
                        eddy_stem = eddy_path.name
                        if eddy_stem.endswith('.nii.gz'):
                            eddy_stem = eddy_stem[:-7]
                        elif eddy_stem.endswith('.nii'):
                            eddy_stem = eddy_stem[:-4]

                        # 尝试多种可能的rotated bvecs命名
                        possible_bvec_names = [
                            f"{eddy_stem}.eddy_rotated_bvecs",      # HC1_0001_eddy.eddy_rotated_bvecs
                            f"{eddy_stem}_rotated_bvecs",           # HC1_0001_eddy_rotated_bvecs
                            f"{eddy_stem.replace('_eddy', '')}.eddy_rotated_bvecs",  # HC1_0001.eddy_rotated_bvecs
                        ]

                        found_bvec = None
                        for bvec_name in possible_bvec_names:
                            candidate = eddy_path.parent / bvec_name
                            if candidate.exists():
                                found_bvec = str(candidate)
                                break

                        if found_bvec:
                            eddy_bvec_files.append(found_bvec)

                    if eddy_bvec_files:
                        bvec_files = eddy_bvec_files
                        print(f"    [输入来源] 使用eddy校正后的DWI + rotated bvecs ({len(eddy_bvec_files)}个)")
                    else:
                        # 如果没找到rotated bvecs，使用原始bvecs
                        bvec_files = dicom_outputs["bvec_files"]
                        print(f"    [输入来源] 使用eddy校正后的DWI + 原始bvecs (未找到rotated)")

                elif eddy_outputs:
                    # 只有eddy输出，尝试从eddy输出目录或原始位置查找bval/bvec
                    input_files = eddy_outputs["nifti_files"]
                    print(f"    [警告] 只有eddy输出，尝试查找bval/bvec文件...")

                    # 尝试从eddy输出目录查找相关的bval/bvec
                    for eddy_file in input_files:
                        eddy_path = Path(eddy_file)
                        eddy_stem = eddy_path.name
                        if eddy_stem.endswith('.nii.gz'):
                            eddy_stem = eddy_stem[:-7]
                        elif eddy_stem.endswith('.nii'):
                            eddy_stem = eddy_stem[:-4]

                        # 获取原始被试ID（去掉_eddy后缀）
                        original_stem = eddy_stem.replace('_eddy', '')

                        # 在eddy输出目录查找
                        for bval_name in [f"{eddy_stem}.bval", f"{original_stem}.bval"]:
                            candidate = eddy_path.parent / bval_name
                            if candidate.exists():
                                bval_files.append(str(candidate))
                                break

                        # 查找rotated bvecs
                        for bvec_name in [f"{eddy_stem}.eddy_rotated_bvecs", f"{eddy_stem}.bvec"]:
                            candidate = eddy_path.parent / bvec_name
                            if candidate.exists():
                                bvec_files.append(str(candidate))
                                break

                    if bval_files and bvec_files:
                        print(f"    [输入来源] 使用eddy输出 + 本地bval/bvec")
                    else:
                        print(f"    [警告] 未能找到bval/bvec，DSI Studio可能会失败")

                elif dicom_outputs:
                    # 只有原始DWI，直接使用
                    input_files = dicom_outputs["nifti_files"]
                    bvec_files = dicom_outputs["bvec_files"]
                    bval_files = dicom_outputs["bval_files"]
                    print(f"    [输入来源] 使用原始DWI from {dicom_outputs['tool_name']}")

            elif dsi_action == "rec":
                # **rec action需要.src文件**
                print(f"    [DSI Studio rec] 需要.src文件")

                # 1. 首先从tool_results中查找
                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue
                        if "dsi_studio" in tool_result.get("tool_name", ""):
                            outputs = tool_result.get("outputs", {})
                            src_files = [f for f in outputs.get("output_files", [])
                                        if any(str(f).endswith(ext) for ext in [".src.gz", ".src", ".sz", ".src.gz.sz"])]
                            if src_files:
                                input_files = src_files
                                print(f"    [输入来源] 从dsi_studio获取 {len(src_files)} 个.src文件")
                                break

                # 2. 回退：在运行目录中搜索.src文件（支持新旧格式）
                if not input_files:
                    print(f"    [回退] 在运行目录中搜索.src/.sz文件...")
                    tools_dir = tracker.run_dir / "tools"
                    if tools_dir.exists():
                        src_files = (
                            list(tools_dir.rglob("*.src.gz")) +
                            list(tools_dir.rglob("*.src")) +
                            list(tools_dir.rglob("*.sz")) +
                            list(tools_dir.rglob("*.src.gz.sz"))
                        )
                        if src_files:
                            input_files = [str(f) for f in src_files]
                            print(f"    [输入来源] 从运行目录找到 {len(input_files)} 个.src/.sz文件")

                # 3. 回退：在数据目录中搜索.src/.sz文件（支持新旧格式）
                if not input_files:
                    print(f"    [回退] 在数据目录中搜索.src/.sz文件...")
                    cohort = state.get("cohort", {})
                    groups = cohort.get("groups", {})
                    for group_name, group_data in groups.items():
                        for subject_id in group_data.get("subjects", []):
                            # 检查dwi子目录
                            dwi_dir = DATA_DIR / group_name / "dwi" / subject_id
                            if dwi_dir.exists():
                                src_files = (
                                    list(dwi_dir.rglob("*.src.gz")) +
                                    list(dwi_dir.rglob("*.src")) +
                                    list(dwi_dir.rglob("*.sz")) +
                                    list(dwi_dir.rglob("*.src.gz.sz"))
                                )
                                input_files.extend([str(f) for f in src_files])
                    if input_files:
                        print(f"    [输入来源] 从数据目录找到 {len(input_files)} 个.src/.sz文件")

                # 4. 最后回退：如果没有.src文件，尝试降级为src action
                if not input_files:
                    print(f"    [警告] 未找到.src文件，尝试降级为src action...")
                    # 重新设置action为src
                    dsi_action = "src"
                    next_task.params["action"] = "src"
                    print(f"    [降级] action从rec改为src")
                    # 将在后续的src回退逻辑中处理

            elif dsi_action == "trk":
                # **trk action需要.fib.gz/.fz文件**
                print(f"    [DSI Studio trk] 需要.fib.gz/.fz文件")

                # 1. 首先从tool_results中查找（支持新旧格式）
                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue
                        if "dsi_studio" in tool_result.get("tool_name", ""):
                            outputs = tool_result.get("outputs", {})
                            fib_files = [f for f in outputs.get("output_files", [])
                                        if any(str(f).endswith(ext) for ext in [".fib.gz", ".fz"])]
                            if fib_files:
                                input_files = fib_files
                                print(f"    [输入来源] 从dsi_studio获取 {len(fib_files)} 个.fib/.fz文件")
                                break

                # 2. 回退：在运行目录中搜索.fib.gz/.fz文件（支持新旧格式）
                if not input_files:
                    print(f"    [回退] 在运行目录中搜索.fib.gz/.fz文件...")
                    tools_dir = tracker.run_dir / "tools"
                    if tools_dir.exists():
                        fib_files = list(tools_dir.rglob("*.fib.gz")) + list(tools_dir.rglob("*.fz"))
                        if fib_files:
                            input_files = [str(f) for f in fib_files]
                            print(f"    [输入来源] 从运行目录找到 {len(input_files)} 个.fib/.fz文件")

                # 3. 回退：在数据目录中搜索.fib.gz/.fz文件（支持新旧格式）
                if not input_files:
                    print(f"    [回退] 在数据目录中搜索.fib.gz/.fz文件...")
                    cohort = state.get("cohort", {})
                    groups = cohort.get("groups", {})
                    for group_name, group_data in groups.items():
                        for subject_id in group_data.get("subjects", []):
                            dwi_dir = DATA_DIR / group_name / "dwi" / subject_id
                            if dwi_dir.exists():
                                fib_files = list(dwi_dir.rglob("*.fib.gz")) + list(dwi_dir.rglob("*.fz"))
                                input_files.extend([str(f) for f in fib_files])
                    if input_files:
                        print(f"    [输入来源] 从数据目录找到 {len(input_files)} 个.fib/.fz文件")

                # 4. 最后回退：如果没有.fib.gz文件，尝试降级为rec action
                if not input_files:
                    print(f"    [警告] 未找到.fib.gz文件，尝试降级为rec action...")
                    dsi_action = "rec"
                    next_task.params["action"] = "rec"
                    print(f"    [降级] action从trk改为rec")
                    # 注意：此时需要重新走rec的回退逻辑

            else:
                # 其他action（如ana, exp等）- 根据具体需求处理
                print(f"    [DSI Studio {dsi_action}] 从之前的dsi_studio输出获取")
                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        if tool_result.get("status") != "succeeded":
                            continue
                        if "dsi_studio" in tool_result.get("tool_name", ""):
                            outputs = tool_result.get("outputs", {})
                            output_files = outputs.get("output_files", [])
                            if output_files:
                                input_files = output_files
                                print(f"    [输入来源] 从dsi_studio获取 {len(output_files)} 个文件")
                                break

            # ========== 降级链处理 ==========
            # 处理 trk → rec → src 的降级链
            # 当 trk 降级为 rec 时，需要执行 rec 的回退逻辑

            # 如果 rec action（包括从trk降级的）没有找到.src/.sz文件，继续降级为src
            if dsi_action == "rec" and not input_files:
                print(f"    [降级链] rec action没有.src/.sz文件，继续降级为src...")
                # 搜索.src/.sz文件（支持新旧格式）
                tools_dir = tracker.run_dir / "tools"
                if tools_dir.exists():
                    src_files = (
                        list(tools_dir.rglob("*.src.gz")) +
                        list(tools_dir.rglob("*.src")) +
                        list(tools_dir.rglob("*.sz")) +
                        list(tools_dir.rglob("*.src.gz.sz"))
                    )
                    if src_files:
                        input_files = [str(f) for f in src_files]
                        print(f"    [输入来源] 从运行目录找到 {len(input_files)} 个.src/.sz文件")

                if not input_files:
                    # 在数据目录中搜索（支持新旧格式）
                    cohort = state.get("cohort", {})
                    groups = cohort.get("groups", {})
                    for group_name, group_data in groups.items():
                        for subject_id in group_data.get("subjects", []):
                            dwi_dir = DATA_DIR / group_name / "dwi" / subject_id
                            if dwi_dir.exists():
                                src_files = (
                                    list(dwi_dir.rglob("*.src.gz")) +
                                    list(dwi_dir.rglob("*.src")) +
                                    list(dwi_dir.rglob("*.sz")) +
                                    list(dwi_dir.rglob("*.src.gz.sz"))
                                )
                                input_files.extend([str(f) for f in src_files])
                    if input_files:
                        print(f"    [输入来源] 从数据目录找到 {len(input_files)} 个.src/.sz文件")

                if not input_files:
                    # 降级为src action
                    dsi_action = "src"
                    next_task.params["action"] = "src"
                    print(f"    [降级] action从rec改为src")

            # 如果src action没有找到数据，从cohort获取原始DWI
            if dsi_action == "src" and not input_files:
                print(f"    [输入来源] 检查cohort原始DWI数据...")
                cohort = state.get("cohort", {})
                groups = cohort.get("groups", {})
                dicom_dirs = []

                for group_name, group_data in groups.items():
                    subject_id_list = group_data.get("subjects", [])
                    for subject_id in subject_id_list:
                        subject_dir = DATA_DIR / group_name / "dwi" / subject_id
                        if subject_dir.exists():
                            nii_files = list(subject_dir.rglob("*.nii")) + list(subject_dir.rglob("*.nii.gz"))
                            if nii_files:
                                input_files.extend([str(f) for f in nii_files])
                            else:
                                # DSI Studio可以直接读取DICOM目录
                                dcm_files = list(subject_dir.glob("*.dcm"))
                                if dcm_files:
                                    dicom_dirs.append(str(subject_dir))

                # 如果没有NIfTI但有DICOM，传递DICOM目录（DSI Studio支持）
                if not input_files and dicom_dirs:
                    input_files = dicom_dirs
                    print(f"    [输入来源] 使用DICOM目录（DSI Studio原生支持）")

            inputs["input_files"] = input_files
            # 为src action提供bvec/bval文件
            if bvec_files:
                inputs["bvec_files"] = bvec_files
            if bval_files:
                inputs["bval_files"] = bval_files
            print(f"    [输入] {len(input_files)} 个文件/目录")

        # **关键修复**: 填充缺失的params（如analysis_type）
        # 智能填充工具参数（特别是analysis_type等关键参数）
        filled_params = next_task.params.copy()

        # 如果是SPM分析，根据任务描述推断analysis_type
        if tool_name == "spm_analysis" and "analysis_type" not in filled_params:
            description_lower = (next_task.description or "").lower()
            if "segment" in description_lower or "vbm" in description_lower:
                filled_params["analysis_type"] = "vbm_segment"
            elif "smooth" in description_lower:
                filled_params["analysis_type"] = "smooth"
            elif "normal" in description_lower:
                filled_params["analysis_type"] = "normalize"
            elif "dartel" in description_lower:
                filled_params["analysis_type"] = "vbm_dartel"

        # 如果是统计分析，根据研究问题推断analysis_type
        elif tool_name == "python_stats" and "analysis_type" not in filled_params:
            description_lower = (next_task.description or "").lower()
            research_question = state.get("question", "").lower()

            # 合并描述和研究问题进行分析
            combined_text = f"{description_lower} {research_question}"

            # 检测分析类型
            if "相关" in combined_text or "correlation" in combined_text:
                filled_params["analysis_type"] = "correlation"
            elif "anova" in combined_text or "方差分析" in combined_text:
                filled_params["analysis_type"] = "anova"
            elif any(kw in combined_text for kw in ["比较", "compare", "comparison", "difference", "between", "vs", "versus"]):
                # 默认使用t检验进行组间比较
                filled_params["analysis_type"] = "ttest"
            else:
                # 如果无法明确判断，但是research_question提到了多个组，默认使用t检验
                # 检查是否有组的概念
                if any(kw in combined_text for kw in ["组", "group", "患者", "patient", "健康", "healthy", "control", "病人", "hc", "sca"]):
                    filled_params["analysis_type"] = "ttest"
                    print(f"  [智能参数] 根据研究问题推断analysis_type=ttest（检测到组间比较）")
                else:
                    # 最后的默认值
                    filled_params["analysis_type"] = "ttest"
                    print(f"  [智能参数] 使用默认analysis_type=ttest")

        # **关键逻辑**: 检测后处理任务（数据提取、统计分析、可视化），使用Vibe Coding生成脚本
        task_description_lower = (next_task.description or "").lower()

        # 检测任务类型
        extraction_keywords = [
            "extract", "提取", "导出", "export", "stats", "table", "指标",
            "脑区", "roi", "volume", "体积", "测量", "measure", "metric"
        ]
        analysis_keywords = [
            # 统计分析
            "分析", "analyze", "analysis", "比较", "compare", "统计", "statistic",
            "t检验", "ttest", "anova", "bootstrap", "效应量", "置信区间", "置信",
            "bayes", "贝叶斯", "相关", "correlation", "回归", "regression",
            "检验", "test", "差异", "difference", "显著", "significant",
            "p值", "p-value", "均值", "mean", "标准差", "std", "方差", "variance",
            # 质量评估
            "质量", "quality", "评分", "score", "rating", "assessment", "evaluation",
            "qc", "qa", "mriqc", "质控",
            # 算法实现
            "算法", "algorithm", "计算", "calculate", "computation", "处理", "process",
            "implement", "实现", "方法", "method", "pipeline", "流程"
        ]
        visualization_keywords = ["可视化", "visualiz", "画图", "plot", "图表", "chart", "figure", "绑图"]

        # 排除主处理任务的关键词（这些任务需要运行原有工具，不是后处理）
        # 注意: 使用完整词匹配避免substring误匹配(如"tract"匹配"extract")
        primary_processing_keywords = [
            # FreeSurfer
            "recon-all", "recon all", "皮层重建", "reconstruction",
            # SPM/VBM
            "segment", "分割", "vbm",
            # FSL
            "bet", "brain extraction", "颅骨剥离",
            "eddy", "eddy_correct", "涡流校正",
            "dtifit", "tensor fitting", "张量拟合",
            "fast", "组织分割",
            "flirt", "配准", "registration",
            "tbss", "白质分析",
            # DSI Studio
            "tractography", "fiber tract", "纤维追踪", "tracking",
            "src", "source file", "源文件",
            "rec", "gqi", "qsdr",
            # DTI/DWI通用
            "dti", "dwi", "diffusion",
            "preprocessing", "预处理",
            "motion correction", "运动校正"
        ]
        # 使用正则表达式进行词边界匹配
        import re
        def check_keyword_match(text, keywords):
            for kw in keywords:
                # 对于包含空格或特殊字符的关键词直接检查
                if ' ' in kw or '-' in kw:
                    if kw in text:
                        return True
                else:
                    # 对于单词关键词，使用词边界匹配
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, text):
                        return True
            return False
        is_primary_processing = check_keyword_match(task_description_lower, primary_processing_keywords)

        is_extraction_task = any(kw in task_description_lower for kw in extraction_keywords)
        is_analysis_task = any(kw in task_description_lower for kw in analysis_keywords)
        is_visualization_task = any(kw in task_description_lower for kw in visualization_keywords)

        # 确定任务类型（排除主处理任务）
        post_processing_task_type = None
        if not is_primary_processing:
            if is_extraction_task:
                post_processing_task_type = "extraction"
            elif is_analysis_task:
                post_processing_task_type = "analysis"
            elif is_visualization_task:
                post_processing_task_type = "visualization"

        # 检测是否应该使用Vibe Coding
        # 新策略：任何后处理任务（提取、分析、可视化）都直接使用Vibe Coding编写代码实现
        # 不再依赖python_stats等固定模式工具，而是根据研究计划智能生成代码
        should_use_vibe_coding = (
            post_processing_task_type is not None and
            tool_name in ["freesurfer_analysis", "spm_analysis", "dsi_studio_analysis", "fsl_analysis", "python_stats", "python_scripting"]
        )

        # **核心策略**：python_stats工具功能有限，所有统计分析任务直接使用Vibe Coding
        # Vibe Coding可以智能生成代码处理各种数据格式和复杂分析需求
        if tool_name == "python_stats":
            print(f"  [策略] python_stats任务自动切换到Vibe Coding（功能更强大）")
            should_use_vibe_coding = True
            # 如果还没有确定任务类型，默认为分析任务
            if post_processing_task_type is None:
                post_processing_task_type = "analysis"

        # 如果检测到后处理任务，但工具是python_stats，优先切换到vibe_coding
        # python_stats只能处理标准SPM格式，灵活性不足
        if post_processing_task_type is not None and tool_name == "python_stats":
            print(f"  [策略调整] 检测到{post_processing_task_type}任务，优先使用Vibe Coding而非python_stats")
            should_use_vibe_coding = True

        # 特殊情况1：任务被标记为强制使用vibe_coding（通常是因为之前的执行失败）
        if next_task.params.get("force_vibe_coding", False):
            vibe_reason = next_task.params.get("vibe_coding_reason", "未知原因")
            print(f"  [强制切换] 任务已标记强制使用Vibe Coding（原因: {vibe_reason}）")
            should_use_vibe_coding = True
            if post_processing_task_type is None:
                post_processing_task_type = "extraction"  # 默认为数据提取任务

        if should_use_vibe_coding:
            print(f"\n  [Vibe Coding] 检测到后处理任务: {post_processing_task_type}")

            # 查找上游工具的输出
            source_tool = None
            tool_outputs = {}

            # 映射当前工具到对应的处理工具
            tool_mapping = {
                "freesurfer_analysis": "freesurfer",
                "spm_analysis": "spm",
                "dsi_studio_analysis": "dsi_studio",
                "fsl_analysis": "fsl",
                "python_stats": None  # python_stats需要从其他工具获取数据
            }

            if tool_name == "python_stats":
                # 对于统计分析，查找之前的处理工具输出
                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        result_tool_name = tool_result.get("tool_name", "")
                        if tool_result.get("status") == "succeeded":
                            if "freesurfer" in result_tool_name:
                                source_tool = "freesurfer"
                            elif "spm" in result_tool_name:
                                source_tool = "spm"
                            elif "dsi_studio" in result_tool_name:
                                source_tool = "dsi_studio"
                            elif "fsl" in result_tool_name:
                                source_tool = "fsl"

                            if source_tool:
                                tool_outputs = tool_result.get("outputs", {})
                                break
            else:
                source_tool = tool_mapping.get(tool_name)
                # 查找同类工具之前的输出
                if "tool_results" in state and state["tool_results"]:
                    for tool_result in reversed(state["tool_results"]):
                        if (tool_result.get("tool_name") == tool_name and
                            tool_result.get("status") == "succeeded"):
                            tool_outputs = tool_result.get("outputs", {})
                            break

            # 如果没有从state找到，尝试从tools目录查找
            # **关键改进**：严格区分处理输出和分析结果，优先使用原始处理目录
            if not tool_outputs and source_tool:
                tools_dir = tracker.run_dir / "tools"
                if tools_dir.exists():
                    # 获取工具专用扩展名配置
                    ext_config = get_tool_required_extensions(source_tool)
                    required_extensions = ext_config["required"]
                    all_extensions = ext_config["all"]

                    # 收集所有匹配的目录
                    matching_dirs = []
                    for subdir in tools_dir.iterdir():
                        if source_tool in subdir.name.lower():
                            matching_dirs.append(subdir)

                    # 分类：原始处理目录 vs 迭代任务目录
                    # 原始目录格式: task_XX_toolname
                    # 迭代目录格式: N_task_XX_toolname (N为迭代次数)
                    original_dirs = []
                    iteration_dirs = []
                    for d in matching_dirs:
                        # 检查是否以数字开头且包含_task_（迭代任务标志）
                        if d.name[0].isdigit() and '_task_' in d.name:
                            iteration_dirs.append(d)
                        else:
                            original_dirs.append(d)

                    # 按修改时间排序（最新优先）
                    original_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
                    iteration_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

                    print(f"  [Vibe Coding] 搜索 {source_tool} 输出目录:")
                    print(f"    - 原始处理目录: {len(original_dirs)} 个")
                    print(f"    - 迭代任务目录: {len(iteration_dirs)} 个")
                    print(f"    - 必需文件类型: {required_extensions}")

                    # **优先搜索原始处理目录**
                    search_order = original_dirs + iteration_dirs

                    for subdir in search_order:
                        # **关键验证**：检查是否包含必须的文件类型（排除只有CSV/JSON的目录）
                        has_required_files = False
                        for ext in required_extensions:
                            if list(subdir.rglob(ext)):
                                has_required_files = True
                                break

                        if not has_required_files:
                            print(f"    - [跳过] {subdir.name} (缺少必需文件)")
                            continue

                        # 收集所有输出文件
                        output_files = []
                        for ext in all_extensions:
                            output_files.extend([str(f) for f in subdir.rglob(ext)])

                        if output_files:
                            print(f"  [Vibe Coding] ✓ 使用目录: {subdir.name}")
                            print(f"    - 包含 {len(output_files)} 个有效输出文件")

                            tool_outputs = {
                                "output_dir": str(subdir),
                                "output_files": output_files[:50],
                                "subjects_dir": str(subdir)
                            }

                            # FreeSurfer特殊处理：提取被试列表
                            if source_tool == "freesurfer":
                                possible_subjects = []
                                for d in subdir.iterdir():
                                    try:
                                        if d.name in ["fsaverage", "fsaverage5", "fsaverage6"]:
                                            continue
                                        if d.is_dir() and (d / "stats").exists():
                                            possible_subjects.append(d.name)
                                    except OSError:
                                        continue
                                if possible_subjects:
                                    tool_outputs["processed_subjects"] = possible_subjects
                                    print(f"    - FreeSurfer被试: {len(possible_subjects)} 个")

                            # DSI Studio特殊处理：提取纤维束文件（支持新旧格式）
                            if source_tool == "dsi_studio":
                                fib_files = list(subdir.rglob("*.fib.gz")) + list(subdir.rglob("*.fz"))
                                tract_files = list(subdir.rglob("*.tt.gz")) + list(subdir.rglob("*.trk*"))
                                if fib_files:
                                    tool_outputs["fib_files"] = [str(f) for f in fib_files]
                                    print(f"    - FIB文件: {len(fib_files)} 个")
                                if tract_files:
                                    tool_outputs["tract_files"] = [str(f) for f in tract_files]
                                    print(f"    - 纤维束文件: {len(tract_files)} 个")

                            break

                    if not tool_outputs:
                        print(f"  [警告] 未找到包含有效 {source_tool} 输出的目录")
                        print(f"    - 搜索了 {len(matching_dirs)} 个目录，均不包含必需文件")

            # 对于python_stats，即使没有找到特定上游工具，也尝试扫描整个tools目录
            if tool_name == "python_stats" and (not source_tool or not tool_outputs):
                print(f"  [Vibe Coding] 未找到特定上游工具，扫描所有可用的处理输出...")
                tools_dir = tracker.run_dir / "tools"
                if tools_dir.exists():
                    all_outputs = {}
                    found_tools = []
                    for subdir in sorted(tools_dir.iterdir()):
                        if not subdir.is_dir():
                            continue

                        # **改进**：跳过迭代任务目录，只扫描原始处理目录
                        if subdir.name[0].isdigit() and '_task_' in subdir.name:
                            continue

                        # 扫描各种处理工具的输出
                        for tool_key in ["freesurfer", "spm", "fsl", "dsi_studio", "ants"]:
                            if tool_key in subdir.name.lower():
                                # 使用工具专用的必需扩展名验证
                                ext_config = get_tool_required_extensions(tool_key)
                                required_exts = ext_config["required"]
                                all_exts = ext_config["all"]

                                # 验证目录包含必需文件
                                has_required = False
                                for ext in required_exts:
                                    if list(subdir.rglob(ext)):
                                        has_required = True
                                        break

                                if not has_required:
                                    continue

                                # 收集输出文件
                                output_files = []
                                # 收集原始工具输出
                                for ext in all_exts:
                                    output_files.extend([str(f) for f in subdir.rglob(ext)])

                                # 【修复3】同时收集分析产出文件（JSON/CSV）
                                for analysis_ext in ["*.json", "*.csv"]:
                                    output_files.extend([str(f) for f in subdir.rglob(analysis_ext)])

                                if output_files:
                                    all_outputs[tool_key] = {
                                        "output_dir": str(subdir),
                                        "output_files": output_files[:100]
                                    }

                                    # 【修复2】为 all_outputs 也添加工具特定字段
                                    if tool_key == "freesurfer":
                                        try:
                                            possible_subjects = [
                                                d.name for d in subdir.iterdir()
                                                if d.is_dir() and (d / "stats").exists()
                                                and d.name not in ["fsaverage", "fsaverage5", "fsaverage6"]
                                            ]
                                            if possible_subjects:
                                                all_outputs[tool_key]["processed_subjects"] = possible_subjects
                                                all_outputs[tool_key]["subjects_dir"] = str(subdir)
                                        except OSError:
                                            pass
                                    elif tool_key == "dsi_studio":
                                        # 支持新旧格式
                                        fib_files = list(subdir.rglob("*.fib.gz")) + list(subdir.rglob("*.fz"))
                                        tract_files = list(subdir.rglob("*.tt.gz")) + list(subdir.rglob("*.trk*"))
                                        if fib_files:
                                            all_outputs[tool_key]["fib_files"] = [str(f) for f in fib_files]
                                        if tract_files:
                                            all_outputs[tool_key]["tract_files"] = [str(f) for f in tract_files]

                                    found_tools.append(tool_key)
                                    print(f"    - 发现 {tool_key} 输出: {len(output_files)} 个文件")

                    if all_outputs:
                        # 使用找到的第一个工具作为source_tool
                        source_tool = found_tools[0] if found_tools else "unknown"
                        tool_outputs = all_outputs.get(source_tool, {})
                        tool_outputs["all_tool_outputs"] = all_outputs
                        print(f"  [Vibe Coding] 将使用 {source_tool} 的输出进行分析")

            if not tool_outputs and tool_name != "python_stats":
                # 非python_stats工具且没有找到上游工具输出，回退到原有工具执行
                print(f"  [INFO] 未找到上游工具输出，将执行实际的 {tool_name} 工具")
                # **关键修复**：清除 source_tool 以允许代码回退到实际工具执行
                source_tool = None
            elif tool_name == "python_stats" and not source_tool and not tool_outputs:
                # python_stats没有任何可用数据，使用vibe_coding生成数据扫描脚本
                print(f"  [Vibe Coding] 未找到任何处理输出，将使用Vibe Coding扫描并处理原始数据")
                source_tool = "scan_and_extract"  # 特殊标记
                tool_outputs = {"run_dir": str(tracker.run_dir)}

            # 【修复1】从依赖任务获取前序输出文件
            if task_manager and next_task.depends_on:
                previous_outputs = []
                for dep_task_id in next_task.depends_on:
                    dep_task = task_manager.get_task(dep_task_id)
                    if dep_task and dep_task.result:
                        dep_output_files = dep_task.result.get("output_files", [])
                        if dep_output_files:
                            previous_outputs.extend(dep_output_files)
                            print(f"  [Vibe Coding] 从依赖任务 {dep_task_id} 获取 {len(dep_output_files)} 个输出文件")

                if previous_outputs:
                    if not tool_outputs:
                        tool_outputs = {}
                    # 将前序任务的输出文件添加到 output_files（合并，不覆盖）
                    existing = tool_outputs.get("output_files", [])
                    tool_outputs["output_files"] = existing + previous_outputs
                    tool_outputs["previous_task_outputs"] = previous_outputs
                    print(f"  [Vibe Coding] 共获取 {len(previous_outputs)} 个前序任务输出文件")

            if source_tool or tool_outputs:
                # **关键修复**：将从tools目录搜索到的输出添加到state["tool_results"]
                # 这样后续的代码生成和分析可以使用这些路径
                if tool_outputs:
                    # **修复**: 根据source_tool推断模态，避免后续模态检查警告
                    source_modality = None
                    if source_tool:
                        tool_modality_map = {
                            "spm": "anat",       # SPM通常处理结构像
                            "freesurfer": "anat", # FreeSurfer处理T1
                            "fsl": None,         # FSL可能处理多种模态，需要进一步判断
                            "dsi_studio": "dwi",  # DSI Studio处理DWI
                            "ants": "anat",       # ANTs通常处理结构像
                            "scan_and_extract": None  # 特殊模式，不设置模态
                        }
                        source_modality = tool_modality_map.get(source_tool.lower())

                        # 对于FSL，尝试从输出文件名推断模态
                        if source_tool.lower() == "fsl" and tool_outputs.get("output_files"):
                            sample_files = tool_outputs.get("output_files", [])[:3]
                            for f in sample_files:
                                fname = str(f).lower()
                                if any(x in fname for x in ["_fa", "_md", "_l1", "_v1", "_eddy", "dti"]):
                                    source_modality = "dwi"
                                    break
                                elif any(x in fname for x in ["_bold", "_func", "fmri"]):
                                    source_modality = "func"
                                    break
                            if not source_modality:
                                source_modality = "anat"  # 默认为anat

                    temp_tool_result = {
                        "tool_name": f"{source_tool}_output_reference",
                        "status": "succeeded",
                        "outputs": tool_outputs,
                        "modality": source_modality,  # 添加模态字段
                        "timestamp": datetime.now().isoformat(),
                        "note": "从tools目录搜索到的上游工具输出（用于迭代任务）"
                    }
                    if "tool_results" not in state:
                        state["tool_results"] = []
                    state["tool_results"].append(temp_tool_result)
                    modality_info = f" (模态: {source_modality})" if source_modality else ""
                    print(f"  [Vibe Coding] 已将 {source_tool} 输出路径添加到 tool_results{modality_info}")

                # 使用Vibe Coding生成后处理脚本
                from src.agent.vibe_coding import create_vibe_coding_engine
                vibe_engine = create_vibe_coding_engine()

                # **关键修复**: 处理 source_tool 为 None 的情况
                # 当没有找到上游工具时，使用 "scan_and_extract" 策略让 Vibe Coding 智能扫描
                if source_tool is None:
                    print(f"  [警告] 未找到上游工具，使用通用扫描模式")
                    source_tool = "scan_and_extract"
                    # 收集所有可用的工具输出
                    tool_outputs["run_dir"] = str(tracker.run_dir)
                    tool_outputs["all_tool_outputs"] = {}
                    if "tool_results" in state and state["tool_results"]:
                        for tr in state["tool_results"]:
                            if tr.get("status") == "succeeded":
                                tool_outputs["all_tool_outputs"][tr.get("tool_name", "unknown")] = tr.get("outputs", {})

                vibe_result = vibe_engine.generate_post_processing_script(
                    task_type=post_processing_task_type,
                    source_tool=source_tool,
                    tool_outputs=tool_outputs,
                    task_description=next_task.description,
                    output_dir=output_dir,
                    run_dir=Path(tracker.run_dir),
                    cohort=state.get("cohort", {})
                )

                # 构建结果
                from src.tools.registry import ToolCallResult
                if vibe_result.get("success"):
                    result = ToolCallResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        status="succeeded",
                        outputs={
                            "output_files": vibe_result.get("output_files", []),
                            "script_path": vibe_result.get("save_path", ""),
                            "task_type": post_processing_task_type,
                            "source_tool": source_tool
                        },
                        error=None
                    )
                else:
                    result = ToolCallResult(
                        call_id=call_id,
                        tool_name=tool_name,
                        status="failed",
                        outputs={},
                        error=vibe_result.get("error", "Vibe Coding脚本执行失败")
                    )

                # 更新任务状态
                if result.status == "succeeded":
                    task_manager.update_task_status(
                        next_task.task_id,
                        TaskStatus.COMPLETED,
                        result=result.outputs
                    )
                    print(f"  [OK] 后处理任务完成 ({post_processing_task_type})")
                    if "task_retry_counts" not in state:
                        state["task_retry_counts"] = {}
                    state["task_retry_counts"][next_task.task_id] = 0
                else:
                    # **Vibe Coding失败重试逻辑**
                    # 获取重试计数
                    if "task_retry_counts" not in state:
                        state["task_retry_counts"] = {}
                    retry_count = state["task_retry_counts"].get(next_task.task_id, 0)
                    max_retries = 2  # Vibe Coding已有内部10次重试，任务级最多再重试2次

                    print(f"  [FAILED] Vibe Coding任务失败: {result.error}")
                    print(f"  [重试] 当前重试次数: {retry_count}/{max_retries}")

                    if retry_count < max_retries:
                        # 增加重试计数
                        state["task_retry_counts"][next_task.task_id] = retry_count + 1

                        # 调整策略：在下次重试时提供更详细的错误上下文
                        # 这样vibe_coding可以生成更健壮的代码
                        print(f"\n  [策略调整] 准备第 {retry_count + 1} 次任务级重试")
                        print(f"  [提示] 将在下次执行时生成更健壮的代码")

                        # 重置任务状态为PENDING，允许重试
                        task_manager.update_task_status(
                            next_task.task_id,
                            TaskStatus.PENDING,
                            error=None
                        )

                        # 显示进度
                        progress = task_manager.get_progress()
                        print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

                        # 返回状态，允许系统继续尝试
                        return merge_state_updates({
                            "tool_results": [_sanitize_tool_result(result)],
                            "task_retry_counts": dict(state.get("task_retry_counts", {})),
                            "node_history": [f"execute_next_task:{call_id}:retry_{retry_count}"]
                        })
                    else:
                        # 超过最大重试次数，标记为失败
                        print(f"\n  [放弃] 已达到最大重试次数，任务最终失败")
                        task_manager.update_task_status(
                            next_task.task_id,
                            TaskStatus.FAILED,
                            error=result.error
                        )

                        # 标记被阻塞的任务
                        blocked_count = task_manager.mark_blocked_tasks()
                        if blocked_count > 0:
                            print(f"  [警告] {blocked_count} 个依赖任务被阻塞")

                # 显示进度
                progress = task_manager.get_progress()
                print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

                # ========== MoER: Vibe Coding 输出规范检查 ==========
                vc_result_dict = None
                if result.status == "succeeded":
                    try:
                        from src.agent.moer import validate_tool_output
                        vc_spec_check = validate_tool_output(
                            tool_name, post_processing_task_type or "",
                            {"result": result.outputs, "outputs": result.outputs},
                            output_dir=str(output_dir)
                        )
                        vc_result_dict = _sanitize_tool_result(result)
                        vc_result_dict["output_spec_check"] = vc_spec_check

                        acceptance = vc_spec_check.get("acceptance_result", {})
                        acc_status = acceptance.get("status", "pass")
                        if not vc_spec_check.get("passed", True):
                            mf = len(vc_spec_check.get('missing_files', []))
                            qi = len(vc_spec_check.get('quality_issues', []))
                            print(f"  [MoER] Vibe Coding 输出规范检查: {mf} 个缺失文件, {qi} 个质量问题")
                        if acc_status == "fail":
                            print(f"  [MoER] acceptance 判定: FAIL")
                            for fc in acceptance.get("failed_criteria", []):
                                print(f"    - {fc.get('message', '')}")
                        elif acc_status == "warning":
                            print(f"  [MoER] acceptance 判定: WARNING")
                            for w in acceptance.get("warnings", [])[:3]:
                                print(f"    - {w.get('message', '')}")
                        else:
                            print(f"  [MoER] acceptance 判定: PASS")
                    except Exception as vc_spec_err:
                        print(f"  [MoER] Vibe Coding 输出规范检查失败: {vc_spec_err}")

                # 返回更新字典（tool_results使用operator.add会自动累积）
                vc_moer_reviews = []
                vc_return_result = _sanitize_tool_result(result)
                # 如果有 spec_check 结果，使用包含它的 vc_result_dict
                if vc_result_dict is not None:
                    vc_return_result = vc_result_dict
                    spec_check = vc_result_dict.get("output_spec_check")
                    if spec_check:
                        vc_moer_reviews.append({
                            "reviewer": "ToolOutputValidator",
                            "tool_name": tool_name,
                            "passed": spec_check.get("passed", True),
                            "acceptance_status": spec_check.get("acceptance_result", {}).get("status", "pass"),
                            "timestamp": datetime.now().isoformat()
                        })
                return merge_state_updates({
                    "tool_results": [vc_return_result],
                    "moer_reviews": vc_moer_reviews,
                    "node_history": [f"execute_next_task:{call_id}:{result.status}"]
                })

        # ========== 调用 _fill_missing_params 补充缺失参数 ==========
        # 特别是 coregister 的 reference_image 和 source_image
        filled_params = _fill_missing_params(
            tool_name=tool_name,
            tool_step=next_task.description or "",
            params=filled_params,
            state=state
        )

        # 检查是否有参数验证错误
        if "_coregister_error" in filled_params:
            error_msg = filled_params.pop("_coregister_error")
            print(f"  [FAILED] 参数验证失败: {error_msg}")
            task_manager.update_task_status(
                next_task.task_id,
                TaskStatus.FAILED,
                error=error_msg
            )
            return merge_state_updates({
                "tool_results": [{
                    "tool_name": tool_name,
                    "call_id": call_id,
                    "status": "failed",
                    "error": error_msg,
                    "outputs": {}
                }],
                "node_history": [f"execute_next_task:{call_id}:param_error"]
            })

        # 创建工具调用请求
        from src.tools.registry import ToolCallRequest
        request = ToolCallRequest(
            tool_name=tool_name,
            call_id=call_id,
            inputs=inputs,
            params=filled_params,  # 使用填充后的params
            output_dir=str(output_dir),
            context=context
        )

        # 执行工具
        print(f"  开始执行...")
        result = registry.execute(request)

        # 更新任务状态
        if result.status == "succeeded":
            task_manager.update_task_status(
                next_task.task_id,
                TaskStatus.COMPLETED,
                result=result.outputs
            )
            print(f"  [OK] 任务完成")
            # 清除重试计数
            if "task_retry_counts" not in state:
                state["task_retry_counts"] = {}
            state["task_retry_counts"][next_task.task_id] = 0
        else:
            # **错误处理**: 使用智能错误处理器
            _extra_updates = {}  # 收集额外的 state 更新（避免直接修改 state）
            from src.agent.error_handler import get_error_handler, RecoveryStrategy
            error_handler = get_error_handler()

            # 获取重试计数
            if "task_retry_counts" not in state:
                state["task_retry_counts"] = {}
            retry_count = state["task_retry_counts"].get(next_task.task_id, 0)

            # 分析错误
            error_context = {
                "task_id": next_task.task_id,
                "tool_name": tool_name,
                "retry_count": retry_count
            }
            analysis = error_handler.analyze_error(result.error, error_context)
            error_handler.log_error(analysis, next_task.task_id)

            # 显示错误信息和建议
            print(f"  [FAILED] 任务失败: {result.error}")
            print(f"  [错误分析] 类别: {analysis.category.value}, 严重程度: {analysis.severity.value}")
            print(f"  [建议策略] {analysis.suggested_strategy.value}")
            if analysis.fix_suggestions:
                print(f"  [修复建议]:")
                for i, suggestion in enumerate(analysis.fix_suggestions, 1):
                    print(f"    {i}. {suggestion}")

            # 特殊处理：python_stats找不到所需文件格式时，自动切换到vibe_coding
            should_switch_to_vibe = False
            if tool_name == "python_stats":
                # 检查错误是否由于找不到灰质概率图文件
                error_msg = str(result.error).lower()
                if ("找到 0 个灰质概率图" in error_msg or
                    "无法提取脑指标" in error_msg or
                    "组1有0个数据" in error_msg or
                    "组2有0个数据" in error_msg):
                    print(f"  [自动切换] python_stats无法处理当前数据格式，切换到Vibe Coding模式")
                    should_switch_to_vibe = True

                    # 标记任务强制使用vibe_coding
                    current_task = task_manager.get_task(next_task.task_id)
                    if current_task:
                        current_task.params["force_vibe_coding"] = True
                        current_task.params["vibe_coding_reason"] = "python_stats无法处理当前数据格式"
                        task_manager.save_tasks()
                        print(f"  [任务更新] 已标记任务 {next_task.task_id} 强制使用Vibe Coding")

            # 根据策略处理
            if should_switch_to_vibe:
                # 强制切换到vibe_coding模式，重置任务为PENDING
                print(f"  [重试] 使用Vibe Coding模式重新执行...")
                state["task_retry_counts"][next_task.task_id] = retry_count + 1
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.PENDING
                )
            elif analysis.suggested_strategy == RecoveryStrategy.RETRY:
                # 简单重试 - 保持任务为PENDING状态
                print(f"  [重试] 第 {retry_count + 1} 次重试...")
                state["task_retry_counts"][next_task.task_id] = retry_count + 1
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.PENDING  # 重置为PENDING以便重试
                )

            elif analysis.suggested_strategy == RecoveryStrategy.RETRY_WITH_UPGRADE:
                # 使用更高级模型重试
                print(f"  [模型升级] 使用更高级模型重试...")
                state["task_retry_counts"][next_task.task_id] = retry_count + 1
                _extra_updates["use_advanced_model"] = True  # 标记使用高级模型
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.PENDING
                )

            elif analysis.suggested_strategy == RecoveryStrategy.HUMAN_INTERVENTION:
                # 需要人工介入
                print(f"  [人工介入] 错误需要人工处理")
                _extra_updates["needs_human_intervention"] = True
                _extra_updates["intervention_task"] = next_task.task_id
                _extra_updates["intervention_error"] = result.error
                _extra_updates["intervention_suggestions"] = analysis.fix_suggestions
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.FAILED,
                    error=f"需要人工介入: {result.error}"
                )

            elif analysis.suggested_strategy == RecoveryStrategy.SKIP_TASK:
                # 跳过任务
                print(f"  [跳过] 跳过该任务，继续执行")
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.SKIPPED
                )

            else:
                # 默认标记为失败
                task_manager.update_task_status(
                    next_task.task_id,
                    TaskStatus.FAILED,
                    error=result.error
                )

            # 记录错误分析（error_analyses不在AgentState中，暂时不更新）

        # 显示进度
        progress = task_manager.get_progress()
        print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

        # 返回更新字典（tool_results使用operator.add会自动累积）
        # **关键修复**: 添加task_id和modality到tool_result，用于多模态输入选择
        tool_result_with_meta = _sanitize_tool_result(result)
        tool_result_with_meta["task_id"] = next_task.task_id
        tool_result_with_meta["modality"] = getattr(next_task, 'modality', None)
        tool_result_with_meta["tool_name"] = tool_name  # 确保有tool_name

        # ========== MoER: 工具输出规范检查 ==========
        if result.status == "succeeded":
            try:
                from src.agent.moer import validate_tool_output
                analysis_type = filled_params.get("analysis_type",
                                filled_params.get("command",
                                filled_params.get("action", "")))
                spec_check = validate_tool_output(
                    tool_name, analysis_type,
                    {"result": result.outputs, "outputs": result.outputs},
                    output_dir=str(output_dir)
                )
                tool_result_with_meta["output_spec_check"] = spec_check

                acceptance = spec_check.get("acceptance_result", {})
                acc_status = acceptance.get("status", "pass")
                if not spec_check.get("passed", True):
                    mf = len(spec_check.get('missing_files', []))
                    qi = len(spec_check.get('quality_issues', []))
                    print(f"  [MoER] 输出规范检查: {mf} 个缺失文件, {qi} 个质量问题")
                if acc_status == "fail":
                    print(f"  [MoER] acceptance 判定: FAIL")
                    for fc in acceptance.get("failed_criteria", []):
                        print(f"    - {fc.get('message', '')}")
                elif acc_status == "warning":
                    print(f"  [MoER] acceptance 判定: WARNING")
                    for w in acceptance.get("warnings", [])[:3]:
                        print(f"    - {w.get('message', '')}")
                else:
                    print(f"  [MoER] acceptance 判定: PASS")
            except Exception as spec_err:
                print(f"  [MoER] 输出规范检查失败: {spec_err}")

        # 收集 MoER 审查记录，累积到 moer_reviews（operator.add reducer）
        moer_task_reviews = []
        spec_check = tool_result_with_meta.get("output_spec_check")
        if spec_check:
            moer_task_reviews.append({
                "reviewer": "ToolOutputValidator",
                "task_id": next_task.task_id,
                "tool_name": tool_name,
                "passed": spec_check.get("passed", True),
                "acceptance_status": spec_check.get("acceptance_result", {}).get("status", "pass"),
                "missing_files": spec_check.get("missing_files", []),
                "quality_issues": spec_check.get("quality_issues", []),
                "timestamp": datetime.now().isoformat()
            })

        # 构建返回更新（包含 task_retry_counts 和错误处理产生的额外更新）
        result_updates = {
            "tool_results": [tool_result_with_meta],
            "task_retry_counts": dict(state.get("task_retry_counts", {})),
            "moer_reviews": moer_task_reviews,
            "node_history": [f"execute_next_task:{call_id}:{result.status}"]
        }
        # 合并错误处理阶段收集的额外更新（如 needs_human_intervention 等）
        if '_extra_updates' in dir() and _extra_updates:
            result_updates.update(_extra_updates)
        return merge_state_updates(result_updates)

    except Exception as e:
        print(f"  [ERROR] 执行失败: {e}")
        # 同样应用错误处理逻辑
        from src.agent.error_handler import get_error_handler
        error_handler = get_error_handler()

        # task_retry_counts 已在 AgentState 中定义，通过返回 dict 更新
        retry_count = state.get("task_retry_counts", {}).get(next_task.task_id, 0)

        error_context = {
            "task_id": next_task.task_id,
            "tool_name": tool_name,
            "retry_count": retry_count
        }
        analysis = error_handler.analyze_error(str(e), error_context)
        error_handler.log_error(analysis, next_task.task_id)

        # 根据策略决定是否重试
        if retry_count < 2:
            task_manager.update_task_status(
                next_task.task_id,
                TaskStatus.PENDING
            )
            print(f"  [重试] 异常重试")
        else:
            task_manager.update_task_status(
                next_task.task_id,
                TaskStatus.FAILED,
                error=str(e)
            )

        # 显示进度
        progress = task_manager.get_progress()
        print(f"\n  [进度] {progress['completed']}/{progress['total']} ({progress['progress_pct']:.1f}%)")

        # 返回错误更新
        return merge_state_updates({
            "last_error": str(e),
            "error_history": [f"execute_next_task:{next_task.task_id}: {e}"],
            "node_history": [f"execute_next_task:{next_task.task_id}:failed"]
        })


def node_check_tasks_complete(state: AgentState) -> Dict[str, Any]:
    """
    检查所有任务是否完成
    用于条件路由：如果未完成，返回execute_next_task；如果完成，进入generate_algorithm_code

    Returns:
        包含 tasks_complete, has_task_failures 等状态更新的字典
    """
    print(f"\n[NODE: check_tasks_complete] 检查任务完成状态...")

    tracker = _get_tracker()
    from src.agent.task_manager import TaskManager
    task_manager = TaskManager(tracker.run_dir)

    # 首先检查是否有失败的任务，如果有，标记其依赖任务为BLOCKED
    if task_manager.has_failures():
        blocked_count = task_manager.mark_blocked_tasks()
        if blocked_count > 0:
            print(f"  已标记 {blocked_count} 个任务因依赖失败而阻塞")

    progress = task_manager.get_progress()
    is_complete = task_manager.is_complete()
    has_failures = task_manager.has_failures()

    print(f"  进度: {progress['completed']}/{progress['total']}")
    print(f"  失败: {progress['failed']}")
    if progress.get('blocked', 0) > 0:
        print(f"  阻塞: {progress['blocked']}")

    # 构建更新字典
    updates: Dict[str, Any] = {
        "node_history": ["check_tasks_complete"]
    }

    if is_complete:
        print(f"  [COMPLETE] 所有任务已完成 → 将路由到 generate_algorithm_code 节点")
        updates["tasks_complete"] = True
    else:
        print(f"  [PENDING] 还有 {progress['pending']} 个任务待执行 → 将继续执行任务")
        updates["tasks_complete"] = False

    if has_failures:
        print(f"  [WARNING] 有 {progress['failed']} 个任务失败")
        updates["has_task_failures"] = True

    return merge_state_updates(updates)


def _execute_generated_code(code_path: str, run_dir: Path, tool_results: List[Dict]) -> Dict:
    """
    执行生成的算法代码

    Args:
        code_path: 生成的代码文件路径
        run_dir: 运行目录
        tool_results: 工具执行结果（用于提供数据路径）

    Returns:
        执行结果字典 {success, output_files, stdout, stderr, error}
    """
    import subprocess
    import sys
    from glob import glob

    try:
        # 检查代码文件是否存在
        if not os.path.exists(code_path):
            return {
                "success": False,
                "error": f"代码文件不存在: {code_path}",
                "output_files": [],
                "stdout": "",
                "stderr": ""
            }

        print(f"    执行代码: {code_path}")
        print(f"    工作目录: {run_dir}")

        # 获取执行前的文件列表（用于检测新生成的文件）
        before_files = set()
        for pattern in ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.csv', '**/*.json', '**/*.txt']:
            before_files.update(glob(str(run_dir / pattern), recursive=True))

        # 运行Python脚本
        result = subprocess.run(
            [sys.executable, code_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            cwd=str(run_dir),  # 在运行目录中执行
            encoding='utf-8',
            errors='replace'  # 处理编码错误
        )

        # 获取执行后的文件列表
        after_files = set()
        for pattern in ['**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.csv', '**/*.json', '**/*.txt']:
            after_files.update(glob(str(run_dir / pattern), recursive=True))

        # 找出新生成的文件
        new_files = sorted(list(after_files - before_files))

        # 返回执行结果
        return {
            "success": result.returncode == 0,
            "output_files": new_files,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "代码执行超时（超过10分钟）",
            "output_files": [],
            "stdout": "",
            "stderr": ""
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"执行异常: {str(e)}",
            "output_files": [],
            "stdout": "",
            "stderr": traceback.format_exc()
        }


def node_generate_algorithm_code(state: AgentState) -> Dict[str, Any]:
    """
    Vibe Coding节点 - 使用GLM-4.6生成分析算法代码

    在任务完成后，根据研究计划和提取的脑区指标生成算法代码

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: generate_algorithm_code] Vibe Coding - 生成分析算法代码")

    tracker = _get_tracker()
    if not tracker.run_dir:
        print("  [跳过] 无运行目录，跳过代码生成")
        return merge_state_updates({
            "node_history": ["generate_algorithm_code:skipped"]
        })

    # 获取研究计划
    plan = state.get("plan", {})
    if not plan:
        print("  [跳过] 无研究计划，跳过代码生成")
        return merge_state_updates({
            "node_history": ["generate_algorithm_code:skipped"]
        })

    # 获取工具执行结果
    tool_results = state.get("tool_results", [])
    if not tool_results:
        print("  [跳过] 无工具结果，跳过代码生成")
        return merge_state_updates({
            "node_history": ["generate_algorithm_code:skipped"]
        })

    # 检查是否需要生成算法代码（根据计划中的分析类型判断）
    # 如果研究计划中明确需要算法分析，才生成代码
    plan_str = json.dumps(plan, ensure_ascii=False, default=str).lower()
    needs_algorithm = any(keyword in plan_str for keyword in [
        "算法", "algorithm", "分析脚本", "analysis script",
        "统计分析", "statistical analysis", "相关分析", "correlation",
        "比较分析", "comparison", "回归", "regression"
    ])

    if not needs_algorithm:
        print("  [跳过] 研究计划不需要算法代码生成")
        return merge_state_updates({
            "node_history": ["generate_algorithm_code:skipped"]
        })

    try:
        # 创建Vibe Coding引擎
        from src.agent.vibe_coding import create_vibe_coding_engine
        vibe_engine = create_vibe_coding_engine()

        # 获取脑区建议（用于ROI优先级分析）
        brain_region_suggestions = state.get("brain_region_suggestions", {})

        # 构建任务描述
        task_description = f"""
        根据研究计划，为以下研究问题生成分析算法代码：
        {state.get('question', '')}

        研究方法：{', '.join(plan.get('methods', []))}
        预期输出：{plan.get('expected_output', '')}
        """

        # 生成算法代码（包含ROI优先级信息）
        result = vibe_engine.generate_algorithm_code(
            plan=plan,
            tool_results=tool_results,
            task_description=task_description,
            run_dir=Path(tracker.run_dir),
            brain_region_suggestions=brain_region_suggestions
        )

        # 构建生成记录
        code_record = {
            "timestamp": datetime.now().isoformat(),
            "success": result.get("execution_success", False),
            "code_path": result.get("save_path", ""),
            "code_length": result.get("code_length", 0),
            "attempts": result.get("attempts", 0),
            "output_files": result.get("output_files", []),
            "execution": result.get("execution_result", {})
        }

        # 打印结果（vibe_coding内部已经打印了详细信息）
        if result.get("execution_success"):
            print(f"\n  [OK] 算法代码生成并执行成功")
            print(f"  代码路径: {result.get('save_path', '')}")
            print(f"  尝试次数: {result.get('attempts', 0)}")
            if result.get("output_files"):
                print(f"  输出文件: {len(result.get('output_files', []))} 个")
                for f in result.get("output_files", [])[:5]:
                    print(f"    - {Path(f).name}")
        else:
            print(f"\n  [WARNING] 代码执行未成功完成")
            print(f"  尝试次数: {result.get('attempts', 0)}")
            if result.get("save_path"):
                print(f"  代码已保存: {result.get('save_path', '')}")

        return merge_state_updates({
            "generated_codes": [code_record],
            "node_history": ["generate_algorithm_code"]
        })

    except Exception as e:
        print(f"  [ERROR] 代码生成异常: {e}")
        import traceback
        traceback.print_exc()

        # 即使失败也继续流程，不阻塞后续节点
        return merge_state_updates({
            "last_error": f"代码生成失败: {str(e)}",
            "node_history": ["generate_algorithm_code:error"]
        })


def node_end(state: AgentState) -> Dict[str, Any]:
    """
    结束节点

    Returns:
        包含更新字段的字典
    """
    print(f"\n[NODE: end] 任务完成")

    tracker = _get_tracker()
    is_success = state.get("phase") == ResearchPhase.COMPLETED.value

    if tracker.run_dir:
        tracker.finish_run(success=is_success)

    # ========== 保存到长期记忆 ==========
    try:
        memory = _get_memory()

        # 设置工作记忆的关键信息
        memory.working.session_id = state.get("run_id", state.get("session_id", "unknown"))
        memory.working.question = state.get("question", "")
        memory.working.current_plan = state.get("plan", {})

        # 从state中提取步骤结果
        if state.get("plan"):
            # 从真实state构建步骤结果记录
            step_result = StepResult(
                step_id="workflow",
                step_name="complete_workflow",
                status="succeeded" if is_success else "failed",
                inputs={"question": state.get("question", "")},
                outputs={"report": state.get("report", "")[:500] if state.get("report") else ""},
                artifacts=[str(tracker.run_dir)] if tracker.run_dir else [],
                started_at=state.get("created_at", datetime.now().isoformat()),
                finished_at=datetime.now().isoformat()
            )
            memory.working.add_step_result(step_result)

        # 保存到长期记忆
        save_result = memory.save_to_long_term(force=is_success)
        print(f"\n[Memory] 记忆保存结果: 案例={save_result['saved_run']}, 压缩={save_result['compressed']}")

        # 显示记忆状态
        status = memory.get_memory_status()
        print(f"[Memory] 当前容量: {status['capacity']['usage_percent']}")

    except Exception as e:
        print(f"[Memory] 记忆保存失败: {e}")
        import traceback
        traceback.print_exc()

    # 打印迭代总结
    iteration_count = state.get("iteration_count", 0)
    if iteration_count > 0:
        print(f"\n迭代总结:")
        print(f"  总迭代次数: {iteration_count}")
        print(f"  最终质量评分: {state.get('scientific_quality_score', 0.0):.2f}/10")
        if state.get("iteration_history"):
            print(f"  迭代历史: {len(state.get('iteration_history', []))} 次")

    # 显示任务列表总结
    try:
        from src.agent.task_manager import TaskManager
        task_manager = TaskManager(tracker.run_dir)
        print(f"\n任务执行总结:")
        print(task_manager.get_summary())
    except Exception:
        pass

    return merge_state_updates({
        "node_history": ["end"]
    })


def node_human_review(state: AgentState) -> dict:
    """人工审查节点 - 迭代完成后暂停等待用户决策"""
    from langgraph.types import interrupt

    summary = {
        "iteration": state.get("iteration_count", 0),
        "quality_score": state.get("scientific_quality_score", 0),
        "feedback": state.get("iteration_feedback", ""),
        "suggestions": state.get("iteration_suggestions", [])[:3],
        "needs_deeper_analysis": state.get("needs_deeper_analysis", False),
        "needs_human_intervention": state.get("needs_human_intervention", False),
    }
    if state.get("needs_human_intervention"):
        summary["intervention_task"] = state.get("intervention_task")
        summary["intervention_error"] = state.get("intervention_error")
        summary["intervention_suggestions"] = state.get("intervention_suggestions", [])

    user_input = interrupt(summary)

    action = user_input.get("action", "continue") if isinstance(user_input, dict) else "continue"
    user_feedback = user_input.get("feedback", "") if isinstance(user_input, dict) else ""

    updates = {
        "needs_human_intervention": False,
        "intervention_task": "",
        "intervention_error": "",
        "intervention_suggestions": [],
        "node_history": ["human_review"],
    }

    if user_feedback:
        updates["iteration_feedback"] = user_feedback

    if action == "stop":
        updates["needs_deeper_analysis"] = False
        updates["iteration_count"] = state.get("max_iterations", 5)
    # action="continue" 时保持 evaluate_iteration 的判断不变

    return updates
