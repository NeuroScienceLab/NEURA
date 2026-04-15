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
    # 工具节点定义
    # =========================================================================
    "tools": {
        "freesurfer_analysis": {
            "name": "FreeSurfer",
            "category": "structural",
            "modality": "anat",
            "function": "皮层重建、厚度分析、海马分割、皮下结构体积测量、海马亚区分割",
            "inputs": ["T1w NIfTI"],
            "outputs": ["皮层厚度", "皮下体积", "表面积", "曲率", "海马体积", "海马亚区体积", "杏仁核体积"],
            "depends_on": [],
            "followed_by": ["python_stats", "roi_extraction"],
            "best_for": [
                "皮层厚度分析", "海马体积", "皮层表面分析",
                "皮下结构体积", "杏仁核体积", "丘脑体积",
                "海马亚区分割", "内嗅皮层", "纵向分析"
            ],
            "not_for": ["白质纤维", "功能连接", "DTI分析"],
            "typical_params": {
                "command": "recon-all",
                "directive": "-all",
                "parallel": True,
                "hippocampal_subfields": True,
                "longitudinal": False
            },
            "processing_time": "6-12小时/被试",
            "software_version": "FreeSurfer 7.x",
            "quality_control": "ENIGMA QC protocol recommended",
            "evidence_level": "high",
            "confidence": 0.95,
            "references": [
                "FreeSurfer 7.x documentation",
                "ENIGMA consortium QC guidelines",
                "Fischl et al. 2012 NeuroImage"
            ]
        },

        "spm_analysis": {
            "name": "SPM VBM",
            "category": "structural",
            "modality": "anat",
            "function": "体素形态学分析、灰白质分割、空间标准化、调制分析",
            "inputs": ["T1w NIfTI"],
            "outputs": ["灰质体积图", "白质体积图", "标准化图像", "调制图像", "雅可比行列式图"],
            "depends_on": [],
            "followed_by": ["python_stats"],
            "best_for": [
                "VBM分析", "全脑体素比较", "灰质密度",
                "灰质体积", "白质体积", "全脑萎缩分析"
            ],
            "not_for": ["皮层厚度", "纤维追踪", "功能连接"],
            "typical_params": {
                "analysis_type": "vbm_segment",
                "template": "TPM.nii",
                "modulation": True,
                "smoothing_fwhm": 8,
                "dartel": True
            },
            "processing_time": "10-30分钟/被试",
            "software_version": "SPM12",
            "quality_control": "Visual check of normalization and segmentation",
            "evidence_level": "high",
            "confidence": 0.92,
            "references": [
                "SPM12 Manual",
                "Ashburner & Friston 2000 NeuroImage",
                "CAT12 Toolbox documentation"
            ]
        },

        "fsl_analysis": {
            "name": "FSL DTI",
            "category": "diffusion",
            "modality": "dwi",
            "function": "DTI张量分析、涡流校正、脑提取、白质分析、TBSS分析",
            "inputs": ["DWI NIfTI", "bvec", "bval"],
            "outputs": ["FA图", "MD图", "AD图", "RD图", "V1图", "MO图", "S0图"],
            "depends_on": ["dicom_to_nifti"],
            "followed_by": ["python_stats", "dsi_studio_analysis"],
            "best_for": [
                "白质完整性", "FA分析", "MD分析",
                "DTI指标", "TBSS分析", "AD分析", "RD分析"
            ],
            "not_for": ["灰质体积", "功能连接", "皮层厚度"],
            "typical_params": {
                "command": "dtifit",
                "bet_f": 0.3,
                "eddy_correct": True,
                "tbss": False,
                "min_directions": 6
            },
            "processing_time": "15-30分钟/被试",
            "software_version": "FSL 6.x",
            "quality_control": "Check for eddy current artifacts and motion",
            "evidence_level": "high",
            "confidence": 0.93,
            "references": [
                "FSL 6.0 documentation",
                "Smith et al. 2006 NeuroImage (TBSS)",
                "Basser et al. 1994 Biophysical Journal"
            ]
        },

        "dsi_studio_analysis": {
            "name": "DSI Studio",
            "category": "diffusion",
            "modality": "dwi",
            "function": "纤维追踪、连接组分析、高级扩散模型、T2R连接组映射",
            "inputs": ["DWI NIfTI", "bvec", "bval"],
            "outputs": ["纤维束", "连接矩阵", "GFA图", "QA图", "T2R连接组"],
            "depends_on": ["dicom_to_nifti"],
            "followed_by": ["python_stats"],
            "best_for": [
                "纤维追踪", "白质束分析", "连接组",
                "HARDI分析", "DSI分析", "GQI分析", "弓状束", "胼胝体束"
            ],
            "not_for": ["灰质体积", "功能连接"],
            "typical_params": {
                "method": 4,  # GQI
                "param0": 1.25,
                "fiber_tracking": "deterministic",
                "min_length": 20,
                "max_length": 300,
                "turning_angle": 60
            },
            "processing_time": "30-60分钟/被试",
            "software_version": "DSI Studio 2024",
            "quality_control": "Check fiber bundle anatomical plausibility",
            "evidence_level": "high",
            "confidence": 0.90,
            "references": [
                "DSI Studio documentation 2024",
                "Yeh et al. 2013 NeuroImage",
                "HCP Fiber Data Hub integration"
            ]
        },

        "dpabi_analysis": {
            "name": "DPABI",
            "category": "functional",
            "modality": "func",
            "function": "静息态fMRI分析、局部指标、功能连接、网络分析",
            "inputs": ["fMRI NIfTI", "T1w NIfTI"],
            "outputs": ["ALFF", "fALFF", "ReHo", "FC矩阵", "DC", "VMHC", "网络分析"],
            "depends_on": [],
            "followed_by": ["python_stats"],
            "best_for": [
                "功能连接", "ALFF分析", "ReHo分析",
                "静息态网络", "度中心性", "默认模式网络", "边缘系统功能连接"
            ],
            "not_for": ["结构体积", "白质纤维", "皮层厚度"],
            "typical_params": {
                "TR": 2.0,
                "slice_timing": True,
                "smoothing_fwhm": 6,
                "bandpass": [0.01, 0.1],
                "nuisance_regression": True,
                "global_signal_regression": False
            },
            "processing_time": "30-60分钟/被试",
            "software_version": "DPABI V7.0",
            "quality_control": "Check head motion, framewise displacement",
            "evidence_level": "high",
            "confidence": 0.91,
            "references": [
                "DPABI V7.0 documentation",
                "Yan et al. 2016 Neuroinformatics",
                "Power et al. 2012 NeuroImage (motion correction)"
            ]
        },

        "python_stats": {
            "name": "统计分析",
            "category": "statistics",
            "modality": "any",
            "function": "组间比较、相关分析、回归分析",
            "inputs": ["数值数据", "CSV", "表格"],
            "outputs": ["统计结果", "p值", "效应量", "图表"],
            "depends_on": ["freesurfer_analysis", "spm_analysis", "fsl_analysis", "dpabi_analysis"],
            "followed_by": [],
            "best_for": [
                "t检验", "方差分析", "相关分析",
                "回归分析", "多重比较校正"
            ],
            "not_for": [],
            "typical_params": {
                "alpha": 0.05,
                "correction": "fdr_bh"
            },
            "processing_time": "秒级",
            "software_version": "scipy, statsmodels"
        },

        "roi_extraction": {
            "name": "ROI提取",
            "category": "extraction",
            "modality": "any",
            "function": "感兴趣区提取、掩膜应用",
            "inputs": ["NIfTI图像", "Atlas"],
            "outputs": ["ROI数值", "掩膜图像"],
            "depends_on": ["freesurfer_analysis", "spm_analysis"],
            "followed_by": ["python_stats"],
            "best_for": ["ROI分析", "区域提取", "掩膜应用"],
            "not_for": [],
            "typical_params": {
                "atlas": "Desikan-Killiany"
            },
            "processing_time": "秒级",
            "software_version": "nibabel, nilearn"
        },

        "dicom_to_nifti": {
            "name": "DICOM转换",
            "category": "preprocessing",
            "modality": "any",
            "function": "DICOM到NIfTI格式转换",
            "inputs": ["DICOM目录"],
            "outputs": ["NIfTI文件", "bvec", "bval", "JSON"],
            "depends_on": [],
            "followed_by": ["freesurfer_analysis", "spm_analysis", "fsl_analysis", "dpabi_analysis"],
            "best_for": ["格式转换", "数据预处理"],
            "not_for": [],
            "typical_params": {
                "compress": True
            },
            "processing_time": "秒级",
            "software_version": "dcm2niix"
        }
    },

    # =========================================================================
    # 任务-工具映射
    # =========================================================================
    "task_tool_mapping": {
        # 结构分析任务
        "皮层厚度": ["freesurfer_analysis"],
        "皮层厚度分析": ["freesurfer_analysis"],
        "cortical thickness": ["freesurfer_analysis"],
        "表面分析": ["freesurfer_analysis"],
        "皮层表面": ["freesurfer_analysis"],

        "灰质体积": ["spm_analysis", "freesurfer_analysis"],
        "灰质密度": ["spm_analysis"],
        "gray matter volume": ["spm_analysis", "freesurfer_analysis"],
        "grey matter": ["spm_analysis", "freesurfer_analysis"],
        "VBM": ["spm_analysis"],
        "VBM分析": ["spm_analysis"],
        "体素形态学": ["spm_analysis"],
        "萎缩分析": ["spm_analysis", "freesurfer_analysis"],
        "脑萎缩": ["spm_analysis", "freesurfer_analysis"],

        "海马体积": ["freesurfer_analysis", "spm_analysis"],
        "海马分割": ["freesurfer_analysis"],
        "海马亚区": ["freesurfer_analysis"],
        "hippocampus": ["freesurfer_analysis"],
        "hippocampal subfields": ["freesurfer_analysis"],
        "内嗅皮层": ["freesurfer_analysis"],
        "entorhinal": ["freesurfer_analysis"],

        "杏仁核": ["freesurfer_analysis"],
        "amygdala": ["freesurfer_analysis"],
        "丘脑": ["freesurfer_analysis"],
        "thalamus": ["freesurfer_analysis"],
        "基底节": ["freesurfer_analysis", "spm_analysis"],
        "basal ganglia": ["freesurfer_analysis", "spm_analysis"],
        "尾状核": ["freesurfer_analysis"],
        "caudate": ["freesurfer_analysis"],
        "壳核": ["freesurfer_analysis"],
        "putamen": ["freesurfer_analysis"],
        "纹状体": ["freesurfer_analysis"],
        "striatum": ["freesurfer_analysis"],
        "小脑": ["freesurfer_analysis", "spm_analysis"],
        "cerebellum": ["freesurfer_analysis", "spm_analysis"],
        "脑干": ["freesurfer_analysis"],
        "brainstem": ["freesurfer_analysis"],
        "胼胝体": ["freesurfer_analysis", "fsl_analysis"],
        "corpus callosum": ["freesurfer_analysis", "fsl_analysis"],

        # 扩散分析任务
        "白质纤维": ["fsl_analysis", "dsi_studio_analysis"],
        "纤维追踪": ["dsi_studio_analysis", "fsl_analysis"],
        "fiber tracking": ["dsi_studio_analysis", "fsl_analysis"],
        "tractography": ["dsi_studio_analysis", "fsl_analysis"],
        "DTI": ["fsl_analysis"],
        "DTI分析": ["fsl_analysis"],
        "扩散张量": ["fsl_analysis"],
        "diffusion tensor": ["fsl_analysis"],
        "FA": ["fsl_analysis"],
        "fractional anisotropy": ["fsl_analysis"],
        "MD": ["fsl_analysis"],
        "mean diffusivity": ["fsl_analysis"],
        "AD": ["fsl_analysis"],  # axial diffusivity
        "RD": ["fsl_analysis"],  # radial diffusivity
        "白质完整性": ["fsl_analysis"],
        "white matter integrity": ["fsl_analysis"],
        "TBSS": ["fsl_analysis"],
        "皮质脊髓束": ["fsl_analysis", "dsi_studio_analysis"],
        "弓状束": ["dsi_studio_analysis"],
        "连接组": ["dsi_studio_analysis"],
        "connectome": ["dsi_studio_analysis"],

        # 功能分析任务
        "功能连接": ["dpabi_analysis"],
        "functional connectivity": ["dpabi_analysis"],
        "ALFF": ["dpabi_analysis"],
        "fALFF": ["dpabi_analysis"],
        "ReHo": ["dpabi_analysis"],
        "静息态": ["dpabi_analysis"],
        "resting state": ["dpabi_analysis"],
        "fMRI": ["dpabi_analysis"],
        "度中心性": ["dpabi_analysis"],
        "默认模式网络": ["dpabi_analysis"],
        "DMN": ["dpabi_analysis"],

        # 统计任务
        "组间比较": ["python_stats"],
        "t检验": ["python_stats"],
        "t-test": ["python_stats"],
        "相关分析": ["python_stats"],
        "correlation": ["python_stats"],
        "回归分析": ["python_stats"],
        "regression": ["python_stats"],
        "方差分析": ["python_stats"],
        "ANOVA": ["python_stats"],
        "多重比较校正": ["python_stats"],
        "FDR": ["python_stats"]
    },

    # =========================================================================
    # 疾病-脑区映射（基于文献证据）
    # =========================================================================
    "disease_roi_mapping": {
        "阿尔茨海默病": {
            "primary": ["海马", "内嗅皮层", "颞叶", "后扣带回", "杏仁核", "海马旁回"],
            "secondary": ["顶叶", "前扣带回", "额叶", "楔前叶"],
            "evidence": "Braak staging, medial temporal lobe atrophy, 2024 diagnostic criteria update",
            "typical_findings": "海马和内嗅皮层萎缩(可早于临床诊断数年)，颞顶叶皮层变薄，杏仁核萎缩",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "内嗅皮层体积", "皮层厚度", "灰质密度"],
            "confidence": 0.95,
            "references": [
                "Braak & Braak 1991 Acta Neuropathol",
                "Jack et al. 2018 Alzheimers Dement (NIA-AA criteria)",
                "2024 Alzheimer's diagnostic criteria update"
            ],
            "progression_sequence": ["海马", "杏仁核", "颞中回", "内嗅皮层", "海马旁回"],
            "early_detection": True
        },
        "AD": {
            "primary": ["海马", "内嗅皮层", "颞叶", "后扣带回", "杏仁核", "海马旁回"],
            "secondary": ["顶叶", "前扣带回", "额叶", "楔前叶"],
            "evidence": "Braak staging, medial temporal lobe atrophy, 2024 diagnostic criteria update",
            "typical_findings": "海马和内嗅皮层萎缩(可早于临床诊断数年)，颞顶叶皮层变薄，杏仁核萎缩",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "内嗅皮层体积", "皮层厚度", "灰质密度"],
            "confidence": 0.95,
            "references": [
                "Braak & Braak 1991 Acta Neuropathol",
                "Jack et al. 2018 Alzheimers Dement (NIA-AA criteria)",
                "2024 Alzheimer's diagnostic criteria update"
            ],
            "progression_sequence": ["海马", "杏仁核", "颞中回", "内嗅皮层", "海马旁回"],
            "early_detection": True
        },

        "帕金森病": {
            "primary": ["黑质", "壳核", "苍白球", "尾状核", "黑质致密部"],
            "secondary": ["额叶", "小脑", "丘脑", "蓝斑"],
            "evidence": "Dopaminergic pathway degeneration, nigrostriatal pathway, 7T MRI neuromelanin imaging",
            "typical_findings": "基底节萎缩，黑质铁沉积，黑质致密部神经黑色素信号减低，DTI显示FA降低",
            "recommended_tools": ["freesurfer_analysis", "fsl_analysis"],
            "recommended_metrics": ["基底节体积", "FA", "MD", "黑质信号", "铁沉积"],
            "confidence": 0.92,
            "references": [
                "Braak et al. 2003 Neurobiol Aging",
                "2025 7T MRI substantia nigra mapping studies",
                "Neuromelanin-sensitive MRI research 2024"
            ],
            "early_detection": True
        },
        "PD": {
            "primary": ["黑质", "壳核", "苍白球", "尾状核", "黑质致密部"],
            "secondary": ["额叶", "小脑", "丘脑", "蓝斑"],
            "evidence": "Dopaminergic pathway degeneration, nigrostriatal pathway, 7T MRI neuromelanin imaging",
            "typical_findings": "基底节萎缩，黑质铁沉积，黑质致密部神经黑色素信号减低，DTI显示FA降低",
            "recommended_tools": ["freesurfer_analysis", "fsl_analysis"],
            "recommended_metrics": ["基底节体积", "FA", "MD", "黑质信号", "铁沉积"],
            "confidence": 0.92,
            "references": [
                "Braak et al. 2003 Neurobiol Aging",
                "2025 7T MRI substantia nigra mapping studies",
                "Neuromelanin-sensitive MRI research 2024"
            ],
            "early_detection": True
        },

        "脊髓小脑共济失调": {
            "primary": ["小脑", "脑干", "脑桥", "延髓", "小脑脚"],
            "secondary": ["丘脑", "基底节", "大脑皮层", "小脑蚓部"],
            "evidence": "Cerebellar and brainstem degeneration, cerebellar tract involvement",
            "typical_findings": "小脑萎缩，脑干变细，小脑脚白质异常，小脑蚓部和半球体积减少",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["小脑体积", "脑干体积", "FA", "小脑脚白质体积"],
            "confidence": 0.91,
            "references": [
                "Jacobi et al. 2015 Brain",
                "Reetz et al. 2013 Mov Disord",
                "2024 SCA neuroimaging biomarker studies"
            ]
        },
        "SCA": {
            "primary": ["小脑", "脑干", "脑桥", "延髓", "小脑脚"],
            "secondary": ["丘脑", "基底节", "大脑皮层", "小脑蚓部"],
            "evidence": "Cerebellar and brainstem degeneration, cerebellar tract involvement",
            "typical_findings": "小脑萎缩，脑干变细，小脑脚白质异常，小脑蚓部和半球体积减少",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["小脑体积", "脑干体积", "FA", "小脑脚白质体积"],
            "confidence": 0.91,
            "references": [
                "Jacobi et al. 2015 Brain",
                "Reetz et al. 2013 Mov Disord",
                "2024 SCA neuroimaging biomarker studies"
            ]
        },
        "SCA3": {
            "primary": ["小脑", "脑干", "脑桥", "齿状核", "小脑脚"],
            "secondary": ["丘脑", "基底节", "额叶", "小脑蚓部"],
            "evidence": "Polyglutamine expansion, Machado-Joseph disease, widespread degeneration",
            "typical_findings": "小脑蚓部和半球萎缩，脑桥萎缩，白质体积减少(小脑半球、蚓部、小脑脚)",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["小脑体积", "脑干体积", "FA", "皮层厚度", "白质体积"],
            "confidence": 0.92,
            "references": [
                "Mascalchi et al. 2018 NeuroImage Clin",
                "D'Abreu et al. 2012 Cerebellum",
                "2024 SCA3 cerebellar-cerebral network studies"
            ]
        },

        "抑郁症": {
            "primary": ["前额叶", "扣带回", "海马", "杏仁核", "背外侧前额叶"],
            "secondary": ["岛叶", "纹状体", "丘脑", "眶额皮层"],
            "evidence": "Limbic-cortical dysregulation, DLPFC-amygdala connectivity, hippocampal volume reduction",
            "typical_findings": "前额叶和海马体积减小，杏仁核过度激活，前额叶-杏仁核功能连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["海马体积", "前额叶厚度", "功能连接", "杏仁核活动"],
            "confidence": 0.89,
            "references": [
                "Drevets et al. 2008 Nat Rev Neurosci",
                "2024 MDD AI early detection studies",
                "Malhi & Mann 2018 Lancet"
            ]
        },
        "MDD": {
            "primary": ["前额叶", "扣带回", "海马", "杏仁核", "背外侧前额叶"],
            "secondary": ["岛叶", "纹状体", "丘脑", "眶额皮层"],
            "evidence": "Limbic-cortical dysregulation, DLPFC-amygdala connectivity, hippocampal volume reduction",
            "typical_findings": "前额叶和海马体积减小，杏仁核过度激活，前额叶-杏仁核功能连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["海马体积", "前额叶厚度", "功能连接", "杏仁核活动"],
            "confidence": 0.89,
            "references": [
                "Drevets et al. 2008 Nat Rev Neurosci",
                "2024 MDD AI early detection studies",
                "Malhi & Mann 2018 Lancet"
            ]
        },

        "精神分裂症": {
            "primary": ["前额叶", "颞叶", "海马", "丘脑", "颞上回"],
            "secondary": ["扣带回", "岛叶", "基底节", "背外侧前额叶"],
            "evidence": "Neurodevelopmental disconnection, gray matter reduction, thalamocortical dysfunction",
            "typical_findings": "广泛皮层变薄，灰质减少(颞上回、前额叶)，白质异常，丘脑皮质连接异常",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["皮层厚度", "灰质体积", "FA", "丘脑体积"],
            "confidence": 0.90,
            "references": [
                "van Erp et al. 2018 Biol Psychiatry (ENIGMA)",
                "2024 EOS meta-analysis",
                "Fornito et al. 2012 Schizophr Bull"
            ]
        },
        "SCZ": {
            "primary": ["前额叶", "颞叶", "海马", "丘脑", "颞上回"],
            "secondary": ["扣带回", "岛叶", "基底节", "背外侧前额叶"],
            "evidence": "Neurodevelopmental disconnection, gray matter reduction, thalamocortical dysfunction",
            "typical_findings": "广泛皮层变薄，灰质减少(颞上回、前额叶)，白质异常，丘脑皮质连接异常",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["皮层厚度", "灰质体积", "FA", "丘脑体积"],
            "confidence": 0.90,
            "references": [
                "van Erp et al. 2018 Biol Psychiatry (ENIGMA)",
                "2024 EOS meta-analysis",
                "Fornito et al. 2012 Schizophr Bull"
            ]
        },

        "多发性硬化": {
            "primary": ["白质", "胼胝体", "脑室周围", "半卵圆中心"],
            "secondary": ["皮层", "丘脑", "小脑", "脑干"],
            "evidence": "Demyelination and axonal loss, corpus callosum involvement, 2024 MAGNIMS-CMSC-NAIMS guidelines",
            "typical_findings": "白质病变(FLAIR高信号)，胼胝体病变(高度特异性)，脑室周围病变，进行性脑萎缩",
            "recommended_tools": ["fsl_analysis", "spm_analysis"],
            "recommended_metrics": ["白质病变体积", "FA", "脑体积", "胼胝体FA"],
            "confidence": 0.93,
            "references": [
                "2024 MAGNIMS-CMSC-NAIMS consensus guidelines",
                "Filippi et al. 2019 Lancet Neurol",
                "Thompson et al. 2018 Lancet Neurol"
            ]
        },
        "MS": {
            "primary": ["白质", "胼胝体", "脑室周围", "半卵圆中心"],
            "secondary": ["皮层", "丘脑", "小脑", "脑干"],
            "evidence": "Demyelination and axonal loss, corpus callosum involvement, 2024 MAGNIMS-CMSC-NAIMS guidelines",
            "typical_findings": "白质病变(FLAIR高信号)，胼胝体病变(高度特异性)，脑室周围病变，进行性脑萎缩",
            "recommended_tools": ["fsl_analysis", "spm_analysis"],
            "recommended_metrics": ["白质病变体积", "FA", "脑体积", "胼胝体FA"],
            "confidence": 0.93,
            "references": [
                "2024 MAGNIMS-CMSC-NAIMS consensus guidelines",
                "Filippi et al. 2019 Lancet Neurol",
                "Thompson et al. 2018 Lancet Neurol"
            ]
        },

        "癫痫": {
            "primary": ["海马", "颞叶", "杏仁核", "海马旁回"],
            "secondary": ["额叶", "丘脑", "岛叶"],
            "evidence": "Mesial temporal sclerosis, network-based disorder",
            "typical_findings": "海马硬化，颞叶萎缩，海马T2/FLAIR高信号",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "皮层厚度", "杏仁核体积"],
            "confidence": 0.90,
            "references": [
                "Bernasconi et al. 2004 Neurology",
                "2024 surface-based morphometry studies",
                "MTLE network studies 2024"
            ]
        },

        "自闭症": {
            "primary": ["额叶", "颞叶", "杏仁核", "小脑", "内侧前额叶"],
            "secondary": ["扣带回", "基底节", "胼胝体", "伏隔核"],
            "evidence": "Abnormal brain development and connectivity, mPFC-amygdala pathway",
            "typical_findings": "早期脑过度生长，杏仁核发育异常，mPFC-杏仁核连接减弱，小脑蚓部异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["皮层厚度", "功能连接", "杏仁核体积", "小脑蚓部体积"],
            "confidence": 0.86,
            "references": [
                "Amaral et al. 2008 Trends Neurosci",
                "2024 social brain circuitry studies",
                "Hazlett et al. 2017 Nature"
            ]
        },
        "ASD": {
            "primary": ["额叶", "颞叶", "杏仁核", "小脑", "内侧前额叶"],
            "secondary": ["扣带回", "基底节", "胼胝体", "伏隔核"],
            "evidence": "Abnormal brain development and connectivity, mPFC-amygdala pathway",
            "typical_findings": "早期脑过度生长，杏仁核发育异常，mPFC-杏仁核连接减弱，小脑蚓部异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["皮层厚度", "功能连接", "杏仁核体积", "小脑蚓部体积"],
            "confidence": 0.86,
            "references": [
                "Amaral et al. 2008 Trends Neurosci",
                "2024 social brain circuitry studies",
                "Hazlett et al. 2017 Nature"
            ]
        },

        "ADHD": {
            "primary": ["前额叶", "基底节", "小脑", "尾状核"],
            "secondary": ["扣带回", "顶叶", "壳核"],
            "evidence": "Frontostriatal dysfunction, delayed cortical maturation",
            "typical_findings": "前额叶和基底节发育延迟，尾状核体积减小，皮层成熟延迟",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["皮层厚度", "尾状核体积", "功能连接"],
            "confidence": 0.85,
            "references": [
                "Shaw et al. 2007 PNAS",
                "Castellanos et al. 2002 JAMA",
                "ENIGMA-ADHD Working Group 2017 Lancet Psychiatry"
            ]
        },

        "脑卒中": {
            "primary": [],  # 取决于病灶位置
            "secondary": [],
            "evidence": "Location-dependent",
            "typical_findings": "局灶性病变，周围水肿",
            "recommended_tools": ["spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["病灶体积", "灰质体积"]
        },

        "脑肿瘤": {
            "primary": [],  # 取决于肿瘤位置
            "secondary": [],
            "evidence": "Location-dependent",
            "typical_findings": "占位性病变，周围水肿",
            "recommended_tools": ["spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["肿瘤体积", "周围组织"],
            "confidence": 0.70
        },

        # =====================================================================
        # 新增疾病 - 基于2024-2025年最新神经影像学研究
        # =====================================================================

        "亨廷顿病": {
            "primary": ["尾状核", "壳核", "纹状体", "苍白球"],
            "secondary": ["额叶", "运动皮层", "岛叶", "丘脑"],
            "evidence": "Striatal atrophy is most reliable biomarker, CAG repeat correlation",
            "typical_findings": "尾状核和壳核显著萎缩(30-52%体积丢失)，侧脑室额角扩大呈'盒状'外观",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["尾状核体积", "壳核体积", "纹状体体积", "灰质体积"],
            "confidence": 0.95,
            "references": ["Tabrizi et al. 2009 Lancet Neurology", "Aylward et al. 2011 Brain"],
            "progression_markers": ["尾状核体积变化率", "壳核体积变化率"],
            "early_detection": True
        },
        "HD": {
            "primary": ["尾状核", "壳核", "纹状体", "苍白球"],
            "secondary": ["额叶", "运动皮层", "岛叶", "丘脑"],
            "evidence": "Striatal atrophy is most reliable biomarker, CAG repeat correlation",
            "typical_findings": "尾状核和壳核显著萎缩(30-52%体积丢失)，侧脑室额角扩大呈'盒状'外观",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["尾状核体积", "壳核体积", "纹状体体积", "灰质体积"],
            "confidence": 0.95,
            "references": ["Tabrizi et al. 2009 Lancet Neurology", "Aylward et al. 2011 Brain"],
            "progression_markers": ["尾状核体积变化率", "壳核体积变化率"],
            "early_detection": True
        },

        "双相障碍": {
            "primary": ["杏仁核", "前额叶", "前扣带回", "眶额皮层"],
            "secondary": ["纹状体", "丘脑", "岛叶", "腹侧纹状体"],
            "evidence": "Corticolimbic circuit dysfunction, amygdala-PFC connectivity",
            "typical_findings": "杏仁核激活异常，前额叶-杏仁核功能连接减弱，白质完整性下降",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis", "fsl_analysis"],
            "recommended_metrics": ["杏仁核体积", "前额叶厚度", "功能连接", "FA"],
            "confidence": 0.88,
            "references": ["Phillips & Swartz 2014 Am J Psychiatry", "Strakowski et al. 2012 Mol Psychiatry"],
            "state_dependent": True
        },
        "BD": {
            "primary": ["杏仁核", "前额叶", "前扣带回", "眶额皮层"],
            "secondary": ["纹状体", "丘脑", "岛叶", "腹侧纹状体"],
            "evidence": "Corticolimbic circuit dysfunction, amygdala-PFC connectivity",
            "typical_findings": "杏仁核激活异常，前额叶-杏仁核功能连接减弱，白质完整性下降",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis", "fsl_analysis"],
            "recommended_metrics": ["杏仁核体积", "前额叶厚度", "功能连接", "FA"],
            "confidence": 0.88,
            "references": ["Phillips & Swartz 2014 Am J Psychiatry", "Strakowski et al. 2012 Mol Psychiatry"],
            "state_dependent": True
        },

        "创伤后应激障碍": {
            "primary": ["海马", "杏仁核", "内侧前额叶", "前扣带回"],
            "secondary": ["岛叶", "眶额皮层", "丘脑"],
            "evidence": "Amygdala hyperactivity, hippocampal and mPFC volume reduction",
            "typical_findings": "杏仁核过度激活，海马和前额叶体积减小，杏仁核-前额叶连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["海马体积", "杏仁核体积", "前额叶厚度", "功能连接"],
            "confidence": 0.87,
            "references": ["Pitman et al. 2012 Nat Rev Neurosci", "Shin et al. 2006 Ann NY Acad Sci"]
        },
        "PTSD": {
            "primary": ["海马", "杏仁核", "内侧前额叶", "前扣带回"],
            "secondary": ["岛叶", "眶额皮层", "丘脑"],
            "evidence": "Amygdala hyperactivity, hippocampal and mPFC volume reduction",
            "typical_findings": "杏仁核过度激活，海马和前额叶体积减小，杏仁核-前额叶连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["海马体积", "杏仁核体积", "前额叶厚度", "功能连接"],
            "confidence": 0.87,
            "references": ["Pitman et al. 2012 Nat Rev Neurosci", "Shin et al. 2006 Ann NY Acad Sci"]
        },

        "强迫症": {
            "primary": ["眶额皮层", "前扣带回", "尾状核", "丘脑"],
            "secondary": ["壳核", "苍白球", "背外侧前额叶"],
            "evidence": "Corticostriatothalamocortical (CSTC) circuit dysfunction",
            "typical_findings": "眶额皮层过度激活，纹状体功能连接异常，皮质-纹状体环路失调",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis", "spm_analysis"],
            "recommended_metrics": ["眶额皮层厚度", "尾状核体积", "功能连接"],
            "confidence": 0.89,
            "references": ["Menzies et al. 2008 Neurosci Biobehav Rev", "Saxena et al. 2001 Arch Gen Psychiatry"]
        },
        "OCD": {
            "primary": ["眶额皮层", "前扣带回", "尾状核", "丘脑"],
            "secondary": ["壳核", "苍白球", "背外侧前额叶"],
            "evidence": "Corticostriatothalamocortical (CSTC) circuit dysfunction",
            "typical_findings": "眶额皮层过度激活，纹状体功能连接异常，皮质-纹状体环路失调",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis", "spm_analysis"],
            "recommended_metrics": ["眶额皮层厚度", "尾状核体积", "功能连接"],
            "confidence": 0.89,
            "references": ["Menzies et al. 2008 Neurosci Biobehav Rev", "Saxena et al. 2001 Arch Gen Psychiatry"]
        },

        "肌萎缩侧索硬化": {
            "primary": ["运动皮层", "中央前回", "皮质脊髓束", "胼胝体"],
            "secondary": ["前额叶", "颞叶", "小脑"],
            "evidence": "Motor cortex atrophy, corticospinal tract degeneration, iron accumulation",
            "typical_findings": "运动皮层灰质萎缩，皮质脊髓束FA降低，运动带征(Motor band sign)",
            "recommended_tools": ["spm_analysis", "fsl_analysis", "freesurfer_analysis"],
            "recommended_metrics": ["运动皮层厚度", "FA", "灰质体积", "皮质脊髓束完整性"],
            "confidence": 0.85,
            "references": ["Turner et al. 2012 Lancet Neurol", "Agosta et al. 2010 Brain"],
            "network_disease": True
        },
        "ALS": {
            "primary": ["运动皮层", "中央前回", "皮质脊髓束", "胼胝体"],
            "secondary": ["前额叶", "颞叶", "小脑"],
            "evidence": "Motor cortex atrophy, corticospinal tract degeneration, iron accumulation",
            "typical_findings": "运动皮层灰质萎缩，皮质脊髓束FA降低，运动带征(Motor band sign)",
            "recommended_tools": ["spm_analysis", "fsl_analysis", "freesurfer_analysis"],
            "recommended_metrics": ["运动皮层厚度", "FA", "灰质体积", "皮质脊髓束完整性"],
            "confidence": 0.85,
            "references": ["Turner et al. 2012 Lancet Neurol", "Agosta et al. 2010 Brain"],
            "network_disease": True
        },

        "额颞叶痴呆": {
            "primary": ["额叶", "颞叶前部", "眶额皮层", "前扣带回"],
            "secondary": ["岛叶", "基底节", "尾状核"],
            "evidence": "Frontal and anterior temporal atrophy, anterior-posterior gradient",
            "typical_findings": "额叶和颞叶前部显著萎缩，与AD区分的前后梯度特征",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["额叶体积", "颞叶体积", "皮层厚度", "尾状核体积"],
            "confidence": 0.88,
            "references": ["Rosen et al. 2002 Neurology", "Rohrer et al. 2011 Lancet Neurol"],
            "subtypes": ["行为变异型(bvFTD)", "非流利变异型(nfvPPA)", "语义变异型(svPPA)"]
        },
        "FTD": {
            "primary": ["额叶", "颞叶前部", "眶额皮层", "前扣带回"],
            "secondary": ["岛叶", "基底节", "尾状核"],
            "evidence": "Frontal and anterior temporal atrophy, anterior-posterior gradient",
            "typical_findings": "额叶和颞叶前部显著萎缩，与AD区分的前后梯度特征",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["额叶体积", "颞叶体积", "皮层厚度", "尾状核体积"],
            "confidence": 0.88,
            "references": ["Rosen et al. 2002 Neurology", "Rohrer et al. 2011 Lancet Neurol"],
            "subtypes": ["行为变异型(bvFTD)", "非流利变异型(nfvPPA)", "语义变异型(svPPA)"]
        },

        "轻度认知障碍": {
            "primary": ["海马", "内嗅皮层", "颞叶内侧", "后扣带回"],
            "secondary": ["顶叶", "楔前叶", "扣带回"],
            "evidence": "Early hippocampal and entorhinal cortex atrophy, predictive of AD conversion",
            "typical_findings": "海马和内嗅皮层体积减小(预测AD转化的关键指标)",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "内嗅皮层体积", "颞叶体积", "皮层厚度"],
            "confidence": 0.90,
            "references": ["Petersen et al. 2010 Arch Neurol", "Jack et al. 2010 Lancet Neurol"],
            "progression_to_AD": True,
            "early_detection": True
        },
        "MCI": {
            "primary": ["海马", "内嗅皮层", "颞叶内侧", "后扣带回"],
            "secondary": ["顶叶", "楔前叶", "扣带回"],
            "evidence": "Early hippocampal and entorhinal cortex atrophy, predictive of AD conversion",
            "typical_findings": "海马和内嗅皮层体积减小(预测AD转化的关键指标)",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "内嗅皮层体积", "颞叶体积", "皮层厚度"],
            "confidence": 0.90,
            "references": ["Petersen et al. 2010 Arch Neurol", "Jack et al. 2010 Lancet Neurol"],
            "progression_to_AD": True,
            "early_detection": True
        },

        "SCA2": {
            "primary": ["小脑", "脑干", "脑桥", "橄榄核"],
            "secondary": ["额叶", "纹状体", "大脑皮层"],
            "evidence": "Cerebellar cortex, brainstem, and cerebellar tracts atrophy",
            "typical_findings": "小脑皮层和白质束萎缩，脑干变细，疾病晚期可累及皮层",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis", "fsl_analysis"],
            "recommended_metrics": ["小脑体积", "脑干体积", "小脑脚FA", "皮层厚度"],
            "confidence": 0.90,
            "references": ["Jacobi et al. 2015 Brain", "Reetz et al. 2013 Mov Disord"]
        },

        "颞叶癫痫": {
            "primary": ["海马", "杏仁核", "颞叶内侧", "海马旁回"],
            "secondary": ["丘脑", "额叶", "Papez环路"],
            "evidence": "Mesial temporal sclerosis, hippocampal hyperintensity on T2/FLAIR",
            "typical_findings": "海马硬化(T2/FLAIR高信号)，海马萎缩，颞角扩大，同侧穹窿和乳头体萎缩",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "杏仁核体积", "皮层厚度"],
            "confidence": 0.92,
            "references": ["Bernasconi et al. 2004 Neurology", "Thom 2014 Neuropathol Appl Neurobiol"],
            "network_disease": True
        },
        "TLE": {
            "primary": ["海马", "杏仁核", "颞叶内侧", "海马旁回"],
            "secondary": ["丘脑", "额叶", "Papez环路"],
            "evidence": "Mesial temporal sclerosis, hippocampal hyperintensity on T2/FLAIR",
            "typical_findings": "海马硬化(T2/FLAIR高信号)，海马萎缩，颞角扩大，同侧穹窿和乳头体萎缩",
            "recommended_tools": ["freesurfer_analysis", "spm_analysis"],
            "recommended_metrics": ["海马体积", "杏仁核体积", "皮层厚度"],
            "confidence": 0.92,
            "references": ["Bernasconi et al. 2004 Neurology", "Thom 2014 Neuropathol Appl Neurobiol"],
            "network_disease": True
        },

        "焦虑症": {
            "primary": ["杏仁核", "前扣带回", "岛叶", "内侧前额叶"],
            "secondary": ["海马", "眶额皮层", "丘脑"],
            "evidence": "Amygdala hyperactivity, altered prefrontal-limbic connectivity",
            "typical_findings": "杏仁核过度激活，前额叶-边缘系统连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["杏仁核体积", "前额叶厚度", "功能连接"],
            "confidence": 0.82,
            "references": ["Etkin & Wager 2007 Am J Psychiatry", "Martin et al. 2009 J Affect Disord"]
        },
        "GAD": {
            "primary": ["杏仁核", "前扣带回", "岛叶", "内侧前额叶"],
            "secondary": ["海马", "眶额皮层", "丘脑"],
            "evidence": "Amygdala hyperactivity, altered prefrontal-limbic connectivity",
            "typical_findings": "杏仁核过度激活，前额叶-边缘系统连接异常",
            "recommended_tools": ["freesurfer_analysis", "dpabi_analysis"],
            "recommended_metrics": ["杏仁核体积", "前额叶厚度", "功能连接"],
            "confidence": 0.82,
            "references": ["Etkin & Wager 2007 Am J Psychiatry", "Martin et al. 2009 J Affect Disord"]
        }
    },

    # =========================================================================
    # 工具等价性映射（用于评估时的功能等价判断）
    # =========================================================================
    "tool_equivalences": {
        "cortical_analysis": ["freesurfer_analysis", "cat12_surface", "civet"],
        "vbm_analysis": ["spm_analysis", "fsl_vbm", "cat12_vbm"],
        "dti_analysis": ["fsl_analysis", "dsi_studio_analysis", "mrtrix", "dipy"],
        "fiber_tracking": ["dsi_studio_analysis", "mrtrix_tckgen", "dipy_tracking"],
        "functional_analysis": ["dpabi_analysis", "conn", "afni_3dtcorrelate"],
        "statistics": ["python_stats", "spm_stats", "fsl_randomise"]
    }
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
