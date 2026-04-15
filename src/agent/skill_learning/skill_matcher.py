"""
Skill Matcher - Matching and Adapting Suitable Skills for Tasks


- Semantic Matching: Score = α * sim(task, skill) + (1-α) * applicability
- Context-Aware Parameter Adaptation
- Quality Update After Execution
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from src.agent.skill_learning.skill_library import AnalysisSkill, SkillLibrary, get_skill_library


@dataclass
class SkillMatch:
    """技能匹配结果"""
    skill: AnalysisSkill
    score: float  # 匹配分数 (0.0-1.0)
    semantic_similarity: float  # 语义相似度
    applicability: float  # 适用性分数
    adapted_parameters: Dict[str, Any]  # 适配后的参数


class SkillMatcher:
    """
    技能匹配器

    功能:
    1. 为给定任务匹配最合适的技能
    2. 适配技能参数到具体任务
    3. 更新技能质量基于执行反馈

    匹配使用向量嵌入（优先）或关键词回退。
    """

    def __init__(self, skill_library: SkillLibrary = None, alpha: float = 0.6):
        """
        初始化技能匹配器

        Args:
            skill_library: 技能库实例
            alpha: 语义相似度权重 (0.0-1.0)，剩余权重给适用性
        """
        self.skill_library = skill_library or get_skill_library()
        self.alpha = alpha  # 语义相似度权重
        self._embedding_client = None  # 懒加载
        self._embedding_available = None  # None=未检测, True/False

    def _get_embedding_client(self):
        """懒加载 embedding 客户端，失败时缓存结果避免重复尝试"""
        if self._embedding_available is False:
            return None
        if self._embedding_client is not None:
            return self._embedding_client
        try:
            from src.utils.embedding import get_embedding_client
            client = get_embedding_client()
            # 简单验证连通性：不实际调用，只检查对象可用
            self._embedding_client = client
            self._embedding_available = True
            return client
        except Exception:
            self._embedding_available = False
            return None

    def match_skills(self,
                    task_description: str,
                    disease: str = None,
                    modality: str = None,
                    tool: str = None,
                    top_k: int = 5) -> List[SkillMatch]:
        """
        为任务匹配技能

        Args:
            task_description: 任务描述
            disease: 疾病名称（可选）
            modality: 模态（可选）
            tool: 工具名称（可选）
            top_k: 返回前K个匹配

        Returns:
            匹配结果列表，按分数降序排列
        """
        # 1. 从技能库检索候选技能
        candidate_skills = self.skill_library.search_skills(
            tool=tool,
            disease=disease,
            modality=modality,
            min_confidence=0.3  # 过滤低置信度技能
        )

        if not candidate_skills:
            return []

        # 2. 计算每个技能的匹配分数
        matches = []
        for skill in candidate_skills:
            # 计算语义相似度
            semantic_sim = self._compute_semantic_similarity(task_description, skill)

            # 计算适用性分数
            applicability = self._compute_applicability(
                skill=skill,
                disease=disease,
                modality=modality,
                tool=tool
            )

            # 综合分数: Score = α * sim + (1-α) * applicability
            score = self.alpha * semantic_sim + (1 - self.alpha) * applicability

            # 适配参数
            adapted_params = self._adapt_parameters(skill, task_description)

            matches.append(SkillMatch(
                skill=skill,
                score=score,
                semantic_similarity=semantic_sim,
                applicability=applicability,
                adapted_parameters=adapted_params
            ))

        # 3. 按分数排序并返回Top-K
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def _compute_semantic_similarity(self, task_description: str, skill: AnalysisSkill) -> float:
        """
        计算任务与技能的语义相似度

        优先使用向量余弦相似度（如果 embedding 可用），
        否则回退到关键词匹配。

        Args:
            task_description: 任务描述
            skill: 技能对象

        Returns:
            相似度分数 (0.0-1.0)
        """
        # 优先：向量嵌入余弦相似度
        client = self._get_embedding_client()
        if client and skill.context.embedding:
            try:
                task_emb = client.embed(task_description)[0]
                skill_emb = np.array(skill.context.embedding)
                dot = np.dot(task_emb, skill_emb)
                norm = np.linalg.norm(task_emb) * np.linalg.norm(skill_emb) + 1e-8
                cos_sim = float(dot / norm)
                return max(0.0, min(1.0, cos_sim))
            except Exception:
                pass  # 回退到关键词

        # 回退：关键词匹配
        return self._keyword_similarity(task_description, skill)

    def _keyword_similarity(self, task_description: str, skill: AnalysisSkill) -> float:
        """关键词匹配回退方案"""

        task_lower = task_description.lower()
        skill_desc = skill.description.lower()

        # 检查工具名称匹配
        tool_match = 1.0 if skill.tool.lower() in task_lower else 0.0

        # 检查疾病匹配
        disease_match = 0.0
        for disease in skill.context.diseases:
            if disease.lower() in task_lower:
                disease_match = 1.0
                break

        # 检查分析类型匹配
        analysis_match = 0.0
        for analysis_type in skill.context.analysis_types:
            if analysis_type.lower() in task_lower:
                analysis_match = 1.0
                break

        # 检查描述相似度（简单的词重叠）
        if skill_desc:
            task_words = set(task_lower.split())
            skill_words = set(skill_desc.split())
            if task_words and skill_words:
                overlap = len(task_words & skill_words)
                desc_match = overlap / max(len(task_words), len(skill_words))
            else:
                desc_match = 0.0
        else:
            desc_match = 0.0

        # 综合相似度
        similarity = (
            0.3 * tool_match +
            0.3 * disease_match +
            0.2 * analysis_match +
            0.2 * desc_match
        )

        return min(1.0, similarity)

    def _compute_applicability(self,
                               skill: AnalysisSkill,
                               disease: str = None,
                               modality: str = None,
                               tool: str = None) -> float:
        """
        计算技能的适用性分数

        Args:
            skill: 技能对象
            disease: 疾病名称
            modality: 模态
            tool: 工具名称

        Returns:
            适用性分数 (0.0-1.0)
        """
        score = 0.0
        weight_sum = 0.0

        # 1. 疾病匹配 (权重: 0.4)
        if disease:
            weight_sum += 0.4
            if disease in skill.context.diseases:
                score += 0.4
            elif any(disease.lower() in d.lower() or d.lower() in disease.lower()
                    for d in skill.context.diseases):
                score += 0.2  # 部分匹配

        # 2. 模态匹配 (权重: 0.3)
        if modality:
            weight_sum += 0.3
            if modality in skill.context.modalities:
                score += 0.3

        # 3. 工具匹配 (权重: 0.3)
        if tool:
            weight_sum += 0.3
            if tool == skill.tool:
                score += 0.3

        # 4. 技能质量加权
        quality_weight = skill.quality.confidence * skill.quality.success_rate
        score = score * quality_weight if weight_sum > 0 else quality_weight

        # 归一化
        if weight_sum > 0:
            score = score / weight_sum

        return min(1.0, score)

    def _adapt_parameters(self, skill: AnalysisSkill, task_description: str) -> Dict[str, Any]:
        """
        适配技能参数到具体任务

        Args:
            skill: 技能对象
            task_description: 任务描述

        Returns:
            适配后的参数字典
        """
        # 复制原始参数
        adapted = dict(skill.parameters)

        # 简单的参数适配规则
        # 在实际应用中，可以使用更复杂的规则或LLM辅助

        task_lower = task_description.lower()

        # 示例：根据任务描述调整平滑参数
        if "smooth" in task_lower or "平滑" in task_lower:
            if "smoothing_fwhm" in adapted:
                # 如果任务强调平滑，可能需要更大的FWHM
                if "more" in task_lower or "larger" in task_lower or "更大" in task_lower:
                    adapted["smoothing_fwhm"] = min(adapted.get("smoothing_fwhm", 8) * 1.5, 12)

        # 示例：根据任务描述调整阈值
        if "threshold" in task_lower or "阈值" in task_lower:
            if "strict" in task_lower or "严格" in task_lower:
                if "alpha" in adapted:
                    adapted["alpha"] = min(adapted.get("alpha", 0.05) * 0.5, 0.01)

        # 示例：根据任务描述调整校正方法
        if "correction" in task_lower or "校正" in task_lower:
            if "fdr" in task_lower:
                adapted["correction"] = "fdr_bh"
            elif "bonferroni" in task_lower:
                adapted["correction"] = "bonferroni"

        return adapted

    def update_skill_from_execution(self,
                                   skill_id: str,
                                   success: bool,
                                   quality_score: float = None,
                                   task_description: str = None):
        """
        基于执行结果更新技能质量

        Args:
            skill_id: 技能ID
            success: 执行是否成功
            quality_score: 质量评分 (0.0-1.0)
            task_description: 任务描述（可选）
        """
        self.skill_library.update_skill_quality(
            skill_id=skill_id,
            success=success,
            quality_score=quality_score
        )

    def get_best_skill(self,
                      task_description: str,
                      disease: str = None,
                      modality: str = None,
                      tool: str = None) -> Optional[SkillMatch]:
        """
        获取最佳匹配技能

        Args:
            task_description: 任务描述
            disease: 疾病名称
            modality: 模态
            tool: 工具名称

        Returns:
            最佳匹配，如果没有匹配返回None
        """
        matches = self.match_skills(
            task_description=task_description,
            disease=disease,
            modality=modality,
            tool=tool,
            top_k=1
        )

        return matches[0] if matches else None

    def recommend_parameters(self,
                           tool: str,
                           disease: str = None,
                           modality: str = None,
                           task_description: str = "") -> Dict[str, Any]:
        """
        为工具推荐参数配置

        Args:
            tool: 工具名称
            disease: 疾病名称
            modality: 模态
            task_description: 任务描述

        Returns:
            推荐的参数配置
        """
        # 获取匹配的技能
        matches = self.match_skills(
            task_description=task_description,
            disease=disease,
            modality=modality,
            tool=tool,
            top_k=3
        )

        if not matches:
            return {}

        # 如果有高分匹配，直接使用
        if matches[0].score > 0.8:
            return matches[0].adapted_parameters

        # 否则，综合多个技能的参数（加权平均）
        recommended = {}
        total_weight = sum(m.score for m in matches)

        if total_weight == 0:
            return matches[0].adapted_parameters if matches else {}

        # 收集所有参数键
        all_keys = set()
        for match in matches:
            all_keys.update(match.adapted_parameters.keys())

        # 对每个参数进行加权
        for key in all_keys:
            values = []
            weights = []
            for match in matches:
                if key in match.adapted_parameters:
                    values.append(match.adapted_parameters[key])
                    weights.append(match.score)

            if values:
                # 对数值参数进行加权平均
                if all(isinstance(v, (int, float)) for v in values):
                    weighted_sum = sum(v * w for v, w in zip(values, weights))
                    recommended[key] = weighted_sum / sum(weights)
                else:
                    # 对非数值参数，选择权重最高的
                    max_idx = weights.index(max(weights))
                    recommended[key] = values[max_idx]

        return recommended


# ========== 全局实例 ==========

_skill_matcher = None

def get_skill_matcher() -> SkillMatcher:
    """获取全局技能匹配器实例"""
    global _skill_matcher
    if _skill_matcher is None:
        _skill_matcher = SkillMatcher()
    return _skill_matcher
