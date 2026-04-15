"""
NEURA Config

"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
PAPER_DIR = PROJECT_ROOT / "paper"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 确保输出目录存在
OUTPUT_DIR.mkdir(exist_ok=True)

# SILICONFLOW_API 配置
# 公开仓库中不要硬编码密钥；请通过环境变量注入。
SILICONFLOW_API_BASE = os.environ.get("SILICONFLOW_API_BASE", "https://api.siliconflow.cn/v1")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

# LLM 模型配置 - 支持多模型切换
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 基础模型：用于简单任务（解析、分析等）
LLM_MODEL_ADVANCED = "Qwen/Qwen3-235B-A22B-Instruct-2507"  # 高端模型：用于代码生成、脚本编写等复杂任务（2507最新版本）
LLM_MODEL_CODE_SPECIALIST = "zai-org/GLM-4.6"  # 专业代码模型：用于分析算法代码生成
LLM_MODEL_REASONING = "Qwen/Qwen3-235B-A22B-Instruct-2507" # 推理专用模型：用于研究规划、实验设计等需要深度推理的任务（Thinking版本，推理能力强，支持256K上下文）
# 备选高端模型：Qwen/Qwen2.5-14B-Instruct, Qwen/QwQ-32B-Preview, Qwen/Qwen3-235B-A22B-Thinking-2507

# 任务类型与模型映射
TASK_MODEL_MAPPING = {
    # 推理密集型任务 - 使用推理专用模型
    "planning": LLM_MODEL_REASONING,  # 研究规划（核心步骤，使用推理模型）
    "planning_iteration": LLM_MODEL_REASONING,  # 迭代规划（需要反思和改进）
    "evaluate_iteration": LLM_MODEL_REASONING,  # 结果评估（需要科学判断）
    "reflection": LLM_MODEL_REASONING,  # 反思分析（需要深度思考）
    "parse_question": LLM_MODEL_REASONING,  # 问题解析（需要识别疾病、预测脑区）

    # 代码生成任务 - 统一使用GLM-4.6专业代码模型
    "code_generation": LLM_MODEL_CODE_SPECIALIST,  # 代码生成（画图、工具脚本等）
    "script_generation": LLM_MODEL_CODE_SPECIALIST,  # 脚本生成（MATLAB、Python等）
    "data_visualization": LLM_MODEL_CODE_SPECIALIST,  # 数据可视化（也需要写代码）
    "algorithm_code_generation": LLM_MODEL_CODE_SPECIALIST,  # 分析算法代码生成
    "code_review": LLM_MODEL_CODE_SPECIALIST,  # 代码审查验证

    # 简单任务 - 使用快速基础模型（7B）
    "general": LLM_MODEL,  # 一般任务
    "analysis": LLM_MODEL,  # 结果分析
    "parse": LLM_MODEL,  # 简单解析（保留用于其他简单解析任务）
    "mapping": LLM_MODEL,  # 字段映射
}

# Embedding 模型配置
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# RAG 配置
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# PUBMED 配置
PUBMED_EMAIL = "research@example.com"
PUBMED_MAX_RESULTS = 10

# 数据组描述（可选，仅用于报告显示；实际组别从数据目录自动发现）
DATA_GROUP_DESCRIPTIONS = {
    "HC": "健康对照组",
    "SCA3": "小脑共济失调组",
    "ST": "卒中患者组",
}
DATA_GROUPS = DATA_GROUP_DESCRIPTIONS  # 向后兼容

# 数据布局: "auto" | "grouped" | "bids" | "flat"
DATA_LAYOUT = "auto"

# 已知模态目录名（用于布局检测）
KNOWN_MODALITY_DIRS = {"anat", "func", "dwi"}

# 扫描时跳过的目录/文件
SCAN_SKIP_PATTERNS = {"__pycache__", ".db", ".git", ".DS_Store"}

# 影像模态
MODALITIES = {
    "anat": "解剖学T1加权像",
    "dwi": "扩散加权成像",
    "func": "功能磁共振成像"
}

# 脑图谱配置 (ICBM152_adult MNI空间图谱)
ATLAS_DIR = PROJECT_ROOT / "ICBM152_adult"

# 关键脑图谱（用于脑区体积提取）
BRAIN_ATLASES = {
    "cerebellum": {
        "atlas_file": ATLAS_DIR / "Cerebellum-SUIT.nii.gz",
        "label_file": ATLAS_DIR / "Cerebellum-SUIT.txt",
        "description": "SUIT小脑图谱（28个亚区）"
    },
    "subcortical": {
        "atlas_file": ATLAS_DIR / "FreeSurferDKT_Subcortical.nii.gz",
        "label_file": ATLAS_DIR / "FreeSurferDKT_Subcortical.txt",
        "description": "FreeSurfer DKT皮下结构图谱"
    },
    "basal_ganglia": {
        "atlas_file": ATLAS_DIR / "ATAG_basal_ganglia.nii.gz",
        "label_file": ATLAS_DIR / "ATAG_basal_ganglia.txt",
        "description": "基底节图谱"
    },
    "cortical": {
        "atlas_file": ATLAS_DIR / "FreeSurferDKT_Cortical.nii.gz",
        "label_file": ATLAS_DIR / "FreeSurferDKT_Cortical.txt",
        "description": "FreeSurfer DKT皮层图谱"
    },
    "brainseg": {
        "atlas_file": ATLAS_DIR / "BrainSeg.nii.gz",
        "label_file": ATLAS_DIR / "BrainSeg.txt",
        "description": "脑组织分割图谱"
    }
}

# SCA3研究优先使用的图谱（小脑和脑干是重点）
SCA3_PRIORITY_ATLASES = ["cerebellum", "subcortical", "basal_ganglia"]
