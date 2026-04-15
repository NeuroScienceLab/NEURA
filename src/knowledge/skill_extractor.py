"""
Skill Extractor - Extracting analytical skills from literature and experiences

5阶段提取流程:
1. Section Identification: 识别方法学章节
2. Tool NER: 识别工具名称
3. Parameter Extraction: 提取参数配置
4. Context Encoding: 编码适用上下文
5. Quality Assessment: 评估技能质量
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

from src.knowledge.pdf_parser import PDFParser, ParsedDocument, DocumentSection
from src.agent.skill_learning.skill_library import (
    AnalysisSkill,
    SkillContext,
    SkillQuality,
    SkillLibrary,
    get_skill_library
)


class SkillExtractor:
    """
    技能提取器

    从神经影像学文献中提取可复用的分析技能
    """

    def __init__(self, skill_library: SkillLibrary = None):
        """
        初始化技能提取器

        Args:
            skill_library: 技能库实例
        """
        self.skill_library = skill_library or get_skill_library()
        self.pdf_parser = PDFParser()

        # 工具名称映射（用于NER）
        self.tool_patterns = {
            "freesurfer": r"freesurfer|recon-all",
            "spm": r"spm\d*|statistical parametric mapping",
            "fsl": r"fsl|fmrib|dtifit|tbss",
            "dpabi": r"dpabi|dparsf|rest",
            "dsi_studio": r"dsi\s*studio",
            "ants": r"ants|advanced normalization tools",
            "afni": r"afni",
            "cat12": r"cat12|computational anatomy toolbox"
        }

        # 参数关键词
        self.param_keywords = [
            "fwhm", "smoothing", "threshold", "alpha", "correction",
            "voxel size", "slice timing", "motion correction",
            "normalization", "template", "atlas", "roi"
        ]

    def extract_from_pdf(self, pdf_path: str) -> List[AnalysisSkill]:
        """
        从PDF文献中提取技能

        Args:
            pdf_path: PDF文件路径

        Returns:
            提取的技能列表
        """
        print(f"正在提取技能: {pdf_path}")

        # 1. 解析PDF
        try:
            doc = self.pdf_parser.parse(pdf_path)
        except Exception as e:
            print(f"PDF解析失败: {e}")
            return []

        # 2. 识别方法学章节
        method_sections = self._identify_method_sections(doc)
        if not method_sections:
            print("未找到方法学章节")
            return []

        print(f"找到 {len(method_sections)} 个方法学章节")

        # 3. 从每个章节提取技能
        skills = []
        for section in method_sections:
            extracted = self._extract_skills_from_section(section, doc)
            skills.extend(extracted)

        print(f"提取了 {len(skills)} 个技能")

        # 4. 保存到技能库
        for skill in skills:
            self.skill_library.add_skill(skill)

        return skills

    def _identify_method_sections(self, doc: ParsedDocument) -> List[DocumentSection]:
        """
        识别方法学章节

        Args:
            doc: 解析后的文档

        Returns:
            方法学章节列表
        """
        method_sections = []

        for section in doc.sections:
            section_type = section.section_type.lower()
            title = section.title.lower()

            # 检查是否为方法学章节
            if section_type == "methods" or "method" in title:
                method_sections.append(section)
            elif any(keyword in title for keyword in [
                "image acquisition", "image processing", "data analysis",
                "statistical analysis", "preprocessing", "mri protocol"
            ]):
                method_sections.append(section)

        return method_sections

    def _extract_skills_from_section(self,
                                    section: DocumentSection,
                                    doc: ParsedDocument) -> List[AnalysisSkill]:
        """
        从章节中提取技能

        Args:
            section: 章节对象
            doc: 文档对象

        Returns:
            技能列表
        """
        skills = []

        # 1. Tool NER: 识别工具
        tools = self._identify_tools(section.content)
        if not tools:
            return []

        # 2. 为每个工具提取技能
        for tool in tools:
            # 提取参数
            parameters = self._extract_parameters(section.content, tool)

            # 编码上下文
            context = self._encode_context(section.content, doc)

            # 评估质量
            quality = self._assess_quality(section.content, doc)

            # 生成技能ID
            skill_id = self._generate_skill_id(tool, doc.title)

            # 创建技能对象
            skill = AnalysisSkill(
                skill_id=skill_id,
                tool=tool,
                parameters=parameters,
                context=context,
                quality=quality,
                source=doc.title or "Unknown",
                description=self._generate_description(tool, parameters, context),
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )

            skills.append(skill)

        return skills

    def _identify_tools(self, text: str) -> List[str]:
        """
        识别文本中提到的工具

        Args:
            text: 文本内容

        Returns:
            工具名称列表
        """
        text_lower = text.lower()
        identified_tools = []

        for tool_name, pattern in self.tool_patterns.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                identified_tools.append(tool_name)

        return identified_tools

    def _extract_parameters(self, text: str, tool: str) -> Dict[str, Any]:
        """
        提取工具参数

        Args:
            text: 文本内容
            tool: 工具名称

        Returns:
            参数字典
        """
        parameters = {}

        # 提取FWHM平滑参数
        fwhm_match = re.search(r"(\d+)\s*mm\s*(fwhm|gaussian|smooth)", text, re.IGNORECASE)
        if fwhm_match:
            parameters["smoothing_fwhm"] = int(fwhm_match.group(1))

        # 提取阈值
        threshold_match = re.search(r"(p|alpha)\s*[<>=]\s*(0\.\d+)", text, re.IGNORECASE)
        if threshold_match:
            parameters["alpha"] = float(threshold_match.group(2))

        # 提取校正方法
        if re.search(r"fdr|false discovery rate", text, re.IGNORECASE):
            parameters["correction"] = "fdr_bh"
        elif re.search(r"bonferroni", text, re.IGNORECASE):
            parameters["correction"] = "bonferroni"
        elif re.search(r"fwe|family-wise error", text, re.IGNORECASE):
            parameters["correction"] = "fwe"

        # 提取体素大小
        voxel_match = re.search(r"(\d+)\s*mm\s*isotropic|(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*mm", text, re.IGNORECASE)
        if voxel_match:
            if voxel_match.group(1):
                parameters["voxel_size"] = int(voxel_match.group(1))
            else:
                parameters["voxel_size"] = [
                    int(voxel_match.group(2)),
                    int(voxel_match.group(3)),
                    int(voxel_match.group(4))
                ]

        # 工具特定参数
        if tool == "freesurfer":
            if "hippocampal subfield" in text.lower():
                parameters["hippocampal_subfields"] = True
            if "longitudinal" in text.lower():
                parameters["longitudinal"] = True

        elif tool == "spm":
            if "dartel" in text.lower():
                parameters["dartel"] = True
            if "modulation" in text.lower() or "modulated" in text.lower():
                parameters["modulation"] = True

        elif tool == "fsl":
            if "tbss" in text.lower():
                parameters["tbss"] = True
            if "eddy" in text.lower():
                parameters["eddy_correct"] = True

        return parameters

    def _encode_context(self, text: str, doc: ParsedDocument) -> SkillContext:
        """
        编码技能适用上下文

        Args:
            text: 文本内容
            doc: 文档对象

        Returns:
            技能上下文
        """
        # 识别疾病
        diseases = self._identify_diseases(text, doc)

        # 识别模态
        modalities = self._identify_modalities(text)

        # 识别分析类型
        analysis_types = self._identify_analysis_types(text)

        return SkillContext(
            diseases=diseases,
            modalities=modalities,
            analysis_types=analysis_types,
            embedding=None  # 可以后续添加语义嵌入
        )

    def _identify_diseases(self, text: str, doc: ParsedDocument) -> List[str]:
        """识别疾病"""
        diseases = []
        text_combined = (text + " " + doc.title + " " + doc.abstract).lower()

        disease_patterns = {
            "Alzheimer's disease": r"alzheimer|ad\b",
            "Parkinson's disease": r"parkinson|pd\b",
            "Multiple sclerosis": r"multiple sclerosis|ms\b",
            "Schizophrenia": r"schizophrenia|scz",
            "Depression": r"depression|mdd|major depressive",
            "Autism": r"autism|asd",
            "ADHD": r"adhd|attention deficit",
            "Epilepsy": r"epilepsy|seizure",
            "Stroke": r"stroke|cerebrovascular",
            "Brain tumor": r"tumor|glioma|glioblastoma"
        }

        for disease, pattern in disease_patterns.items():
            if re.search(pattern, text_combined, re.IGNORECASE):
                diseases.append(disease)

        return diseases

    def _identify_modalities(self, text: str) -> List[str]:
        """识别影像模态"""
        modalities = []
        text_lower = text.lower()

        if re.search(r"t1|structural|anatomical|mprage", text_lower):
            modalities.append("anat")
        if re.search(r"dwi|dti|diffusion", text_lower):
            modalities.append("dwi")
        if re.search(r"fmri|functional|bold|resting.?state", text_lower):
            modalities.append("func")
        if re.search(r"t2|flair", text_lower):
            modalities.append("t2")

        return modalities

    def _identify_analysis_types(self, text: str) -> List[str]:
        """识别分析类型"""
        analysis_types = []
        text_lower = text.lower()

        if re.search(r"vbm|voxel.?based morphometry", text_lower):
            analysis_types.append("VBM")
        if re.search(r"dti|tensor", text_lower):
            analysis_types.append("DTI")
        if re.search(r"functional connectivity|fc\b", text_lower):
            analysis_types.append("FC")
        if re.search(r"cortical thickness|surface.?based", text_lower):
            analysis_types.append("cortical_thickness")
        if re.search(r"volumetry|volume", text_lower):
            analysis_types.append("volumetry")
        if re.search(r"tractography|fiber tracking", text_lower):
            analysis_types.append("tractography")

        return analysis_types

    def _assess_quality(self, text: str, doc: ParsedDocument) -> SkillQuality:
        """
        评估技能质量

        Args:
            text: 文本内容
            doc: 文档对象

        Returns:
            质量指标
        """
        # 初始置信度基于文献质量指标
        confidence = 0.5

        # 如果有详细的参数描述，提升置信度
        if len(text) > 500:
            confidence += 0.1

        # 如果提到了验证或复现，提升置信度
        if re.search(r"validated|replicated|reproduced", text, re.IGNORECASE):
            confidence += 0.1

        # 如果是高影响因子期刊（从标题推断），提升置信度
        if any(keyword in doc.title.lower() for keyword in [
            "nature", "science", "lancet", "jama", "nejm", "brain"
        ]):
            confidence += 0.2

        # 提取效应量
        effect_sizes = self._extract_effect_sizes(text)

        return SkillQuality(
            confidence=min(1.0, confidence),
            success_rate=0.5,  # 初始成功率
            effect_sizes=effect_sizes,
            validation_count=0
        )

    def _extract_effect_sizes(self, text: str) -> List[float]:
        """提取效应量"""
        effect_sizes = []

        # Cohen's d
        d_matches = re.findall(r"cohen'?s?\s*d\s*=\s*([\d.]+)", text, re.IGNORECASE)
        effect_sizes.extend([float(d) for d in d_matches])

        # Eta squared
        eta_matches = re.findall(r"eta\s*squared\s*=\s*([\d.]+)", text, re.IGNORECASE)
        effect_sizes.extend([float(e) for e in eta_matches])

        return effect_sizes

    def _generate_skill_id(self, tool: str, source: str) -> str:
        """生成唯一技能ID"""
        content = f"{tool}_{source}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_description(self,
                             tool: str,
                             parameters: Dict[str, Any],
                             context: SkillContext) -> str:
        """生成技能描述"""
        parts = [f"使用 {tool}"]

        if context.analysis_types:
            parts.append(f"进行 {', '.join(context.analysis_types)} 分析")

        if context.diseases:
            parts.append(f"用于 {', '.join(context.diseases[:2])}")

        if parameters:
            param_strs = []
            for key, value in list(parameters.items())[:3]:
                param_strs.append(f"{key}={value}")
            if param_strs:
                parts.append(f"参数: {', '.join(param_strs)}")

        return " ".join(parts)

    def extract_from_directory(self, paper_dir: str) -> int:
        """
        批量提取目录下所有PDF的技能

        Args:
            paper_dir: 论文目录路径

        Returns:
            提取的技能总数
        """
        paper_path = Path(paper_dir)
        if not paper_path.exists():
            print(f"目录不存在: {paper_dir}")
            return 0

        pdf_files = list(paper_path.glob("*.pdf"))
        print(f"找到 {len(pdf_files)} 个PDF文件")

        total_skills = 0
        for pdf_file in pdf_files:
            try:
                skills = self.extract_from_pdf(str(pdf_file))
                total_skills += len(skills)
            except Exception as e:
                print(f"处理失败 {pdf_file.name}: {e}")

        print(f"\n总计提取 {total_skills} 个技能")
        return total_skills


# ========== 便捷函数 ==========

def extract_skills_from_pdf(pdf_path: str) -> List[AnalysisSkill]:
    """
    从PDF提取技能（便捷函数）

    Args:
        pdf_path: PDF文件路径

    Returns:
        技能列表
    """
    extractor = SkillExtractor()
    return extractor.extract_from_pdf(pdf_path)


def extract_skills_from_directory(paper_dir: str) -> int:
    """
    从目录批量提取技能（便捷函数）

    Args:
        paper_dir: 论文目录路径

    Returns:
        提取的技能总数
    """
    extractor = SkillExtractor()
    return extractor.extract_from_directory(paper_dir)
