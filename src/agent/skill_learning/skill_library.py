"""
Skill Library - Storing and managing analytical skills extracted from literature

Skill Structure: S = (C, π, T, R) — Refer to the Anthropic Skill quadruplet
- C: Applicability Condition (applicability condition — preconditions)
- π: Executable Policy (executable policy — parameters + procedure + tips)
- T: Termination Condition (termination condition — postconditions)
- R: Callable Interface (callable interface — tool + context)

Additional Learning Dimensions:
- error_patterns: Error patterns and recovery strategies accumulated from execution failures
- tips: Domain experience tips accumulated from successful executions
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SkillContext:
    """
    技能适用上下文

    定义技能在什么情况下适用
    """
    diseases: List[str]  # 适用疾病列表
    modalities: List[str]  # 适用模态: anat, dwi, func
    analysis_types: List[str]  # 分析类型: VBM, DTI, FC等
    embedding: Optional[List[float]] = None  # 语义嵌入向量（用于相似度匹配）

    def to_dict(self) -> Dict:
        return {
            "diseases": self.diseases,
            "modalities": self.modalities,
            "analysis_types": self.analysis_types,
            "embedding": self.embedding
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillContext':
        return cls(
            diseases=data.get("diseases", []),
            modalities=data.get("modalities", []),
            analysis_types=data.get("analysis_types", []),
            embedding=data.get("embedding")
        )


@dataclass
class SkillQuality:
    """
    技能质量指标

    记录技能的可靠性和有效性
    """
    confidence: float  # 初始置信度 (0.0-1.0)
    success_rate: float  # 历史成功率
    effect_sizes: List[float]  # 来源文献报告的效应量
    validation_count: int  # 验证次数

    def to_dict(self) -> Dict:
        return {
            "confidence": self.confidence,
            "success_rate": self.success_rate,
            "effect_sizes": self.effect_sizes,
            "validation_count": self.validation_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SkillQuality':
        return cls(
            confidence=data.get("confidence", 0.5),
            success_rate=data.get("success_rate", 0.5),
            effect_sizes=data.get("effect_sizes", []),
            validation_count=data.get("validation_count", 0)
        )


@dataclass
class AnalysisSkill:
    """
    分析技能 — S = (C, π, T, R)

    C (适用条件): preconditions — 使用此技能前必须满足的条件
    π (执行策略): parameters + procedure + tips — 怎么用
    T (终止条件): postconditions — 怎么判断成功
    R (调用接口): tool + context — 怎么触发
    """
    skill_id: str  # 唯一标识符
    tool: str  # 工具名称
    parameters: Dict[str, Any]  # 参数配置
    context: SkillContext  # 适用上下文
    quality: SkillQuality  # 质量指标
    source: str  # 来源文献
    created_at: str  # 创建时间
    updated_at: str  # 更新时间
    description: str = ""  # 技能描述

    # === 过程性知识 (Anthropic Skill 四元组扩展) ===
    preconditions: List[str] = field(default_factory=list)
    # 适用条件 C — 例: ["需要T1加权像", "依赖preprocessing完成"]

    procedure: List[str] = field(default_factory=list)
    # 执行策略 π — 例: ["1. 检查输入NIfTI格式", "2. 设置smoothing_fwhm=8"]

    postconditions: List[str] = field(default_factory=list)
    # 终止条件 T — 例: ["输出包含统计图", "输出文件>1KB"]

    error_patterns: List[Dict[str, str]] = field(default_factory=list)
    # 从执行失败中学习的错误模式
    # 例: [{"pattern": "内存不足", "recovery": "降低分辨率", "count": 3}]

    tips: List[str] = field(default_factory=list)
    # 从经验中积累的领域提示
    # 例: ["AD研究中smoothing_fwhm=6比8效果更好"]

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "tool": self.tool,
            "parameters": self.parameters,
            "context": self.context.to_dict(),
            "quality": self.quality.to_dict(),
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "description": self.description,
            "preconditions": self.preconditions,
            "procedure": self.procedure,
            "postconditions": self.postconditions,
            "error_patterns": self.error_patterns,
            "tips": self.tips
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisSkill':
        return cls(
            skill_id=data["skill_id"],
            tool=data["tool"],
            parameters=data.get("parameters", {}),
            context=SkillContext.from_dict(data.get("context", {})),
            quality=SkillQuality.from_dict(data.get("quality", {})),
            source=data.get("source", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            description=data.get("description", ""),
            preconditions=data.get("preconditions", []),
            procedure=data.get("procedure", []),
            postconditions=data.get("postconditions", []),
            error_patterns=data.get("error_patterns", []),
            tips=data.get("tips", [])
        )


class SkillLibrary:
    """
    技能库 - 管理所有分析技能

    功能:
    1. 存储技能到SQLite数据库
    2. 检索匹配的技能
    3. 更新技能质量
    4. 导出/导入技能
    """

    def __init__(self, db_path: str = None):
        """
        初始化技能库

        Args:
            db_path: 数据库路径，默认为 data/skill_library.db
        """
        if db_path is None:
            from src.config import OUTPUT_DIR
            data_dir = OUTPUT_DIR.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "skill_library.db")

        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 技能表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                tool TEXT NOT NULL,
                parameters TEXT NOT NULL,
                context TEXT NOT NULL,
                quality TEXT NOT NULL,
                source TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 技能-疾病映射表（用于快速检索）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_disease_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                disease TEXT NOT NULL,
                FOREIGN KEY (skill_id) REFERENCES skills(skill_id),
                UNIQUE(skill_id, disease)
            )
        """)

        # 技能-模态映射表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_modality_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                modality TEXT NOT NULL,
                FOREIGN KEY (skill_id) REFERENCES skills(skill_id),
                UNIQUE(skill_id, modality)
            )
        """)

        # 技能使用历史表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skill_usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                task_description TEXT,
                success BOOLEAN,
                quality_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_tool
            ON skills(tool)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_disease
            ON skill_disease_mapping(disease)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_skill_modality
            ON skill_modality_mapping(modality)
        """)

        # 工具执行档案表 — 累积每个工具的使用经验
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_execution_profiles (
                tool_name TEXT PRIMARY KEY,
                total_executions INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                avg_duration_seconds REAL DEFAULT 0,
                param_stats TEXT DEFAULT '{}',
                common_errors TEXT DEFAULT '{}',
                output_patterns TEXT DEFAULT '[]',
                best_for_tasks TEXT DEFAULT '[]',
                updated_at TEXT
            )
        """)

        # Pipeline 复合技能表 — 保存成功的多工具流水线
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_skills (
                pipeline_id TEXT PRIMARY KEY,
                tool_chain TEXT NOT NULL,
                disease TEXT DEFAULT '',
                modality TEXT DEFAULT '',
                research_type TEXT DEFAULT '',
                description TEXT DEFAULT '',
                success_count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 安全迁移：为旧数据库添加 procedural_knowledge 列
        try:
            cursor.execute("ALTER TABLE skills ADD COLUMN procedural_knowledge TEXT DEFAULT '{}'")
        except sqlite3.OperationalError:
            pass  # 列已存在

        conn.commit()
        conn.close()

    def add_skill(self, skill: AnalysisSkill) -> bool:
        """
        添加技能到库中

        如果技能没有 embedding，会自动调用 EmbeddingClient 生成。

        Args:
            skill: 分析技能对象

        Returns:
            是否成功添加
        """
        # 自动生成 embedding（如果还没有）
        if not skill.context.embedding:
            try:
                from src.utils.embedding import get_embedding_client
                parts = [skill.tool, skill.description]
                parts.extend(skill.context.diseases)
                parts.extend(skill.context.analysis_types)
                parts.extend(skill.context.modalities)
                text = " ".join(p for p in parts if p)
                emb = get_embedding_client().embed(text)
                skill.context.embedding = emb[0].tolist()
            except Exception:
                pass  # embedding 生成失败不阻塞存储

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 插入技能（含过程性知识）
            procedural_knowledge = json.dumps({
                "preconditions": skill.preconditions,
                "procedure": skill.procedure,
                "postconditions": skill.postconditions,
                "error_patterns": skill.error_patterns,
                "tips": skill.tips
            }, ensure_ascii=False)

            cursor.execute("""
                INSERT OR REPLACE INTO skills
                (skill_id, tool, parameters, context, quality, source, description,
                 created_at, updated_at, procedural_knowledge)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                skill.skill_id,
                skill.tool,
                json.dumps(skill.parameters, ensure_ascii=False),
                json.dumps(skill.context.to_dict(), ensure_ascii=False),
                json.dumps(skill.quality.to_dict(), ensure_ascii=False),
                skill.source,
                skill.description,
                skill.created_at,
                skill.updated_at,
                procedural_knowledge
            ))

            # 插入疾病映射
            for disease in skill.context.diseases:
                cursor.execute("""
                    INSERT OR IGNORE INTO skill_disease_mapping (skill_id, disease)
                    VALUES (?, ?)
                """, (skill.skill_id, disease))

            # 插入模态映射
            for modality in skill.context.modalities:
                cursor.execute("""
                    INSERT OR IGNORE INTO skill_modality_mapping (skill_id, modality)
                    VALUES (?, ?)
                """, (skill.skill_id, modality))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"添加技能失败: {e}")
            return False

    def get_skill(self, skill_id: str) -> Optional[AnalysisSkill]:
        """
        获取指定技能

        Args:
            skill_id: 技能ID

        Returns:
            技能对象，如果不存在返回None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT skill_id, tool, parameters, context, quality, source, description,
                   created_at, updated_at, procedural_knowledge
            FROM skills
            WHERE skill_id = ?
        """, (skill_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            # 解析过程性知识（兼容旧数据库中无此列的情况）
            pk = {}
            if len(row) > 9 and row[9]:
                try:
                    pk = json.loads(row[9])
                except (json.JSONDecodeError, TypeError):
                    pass

            return AnalysisSkill(
                skill_id=row[0],
                tool=row[1],
                parameters=json.loads(row[2]),
                context=SkillContext.from_dict(json.loads(row[3])),
                quality=SkillQuality.from_dict(json.loads(row[4])),
                source=row[5],
                description=row[6] or "",
                created_at=row[7],
                updated_at=row[8],
                preconditions=pk.get("preconditions", []),
                procedure=pk.get("procedure", []),
                postconditions=pk.get("postconditions", []),
                error_patterns=pk.get("error_patterns", []),
                tips=pk.get("tips", [])
            )
        return None

    def search_skills(self,
                     tool: str = None,
                     disease: str = None,
                     modality: str = None,
                     min_confidence: float = 0.0) -> List[AnalysisSkill]:
        """
        搜索匹配的技能

        Args:
            tool: 工具名称（可选）
            disease: 疾病名称（可选）
            modality: 模态（可选）
            min_confidence: 最小置信度

        Returns:
            匹配的技能列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 构建查询
        query = "SELECT DISTINCT s.skill_id FROM skills s"
        conditions = []
        params = []

        if disease:
            query += " JOIN skill_disease_mapping sdm ON s.skill_id = sdm.skill_id"
            conditions.append("sdm.disease = ?")
            params.append(disease)

        if modality:
            query += " JOIN skill_modality_mapping smm ON s.skill_id = smm.skill_id"
            conditions.append("smm.modality = ?")
            params.append(modality)

        if tool:
            conditions.append("s.tool = ?")
            params.append(tool)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        skill_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        # 获取完整技能并过滤置信度
        skills = []
        for skill_id in skill_ids:
            skill = self.get_skill(skill_id)
            if skill and skill.quality.confidence >= min_confidence:
                skills.append(skill)

        # 按置信度排序
        skills.sort(key=lambda s: s.quality.confidence, reverse=True)
        return skills

    def update_skill_quality(self, skill_id: str, success: bool, quality_score: float = None):
        """
        更新技能质量（基于使用反馈）

        Args:
            skill_id: 技能ID
            success: 是否成功
            quality_score: 质量评分（可选）
        """
        skill = self.get_skill(skill_id)
        if not skill:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 记录使用历史
        cursor.execute("""
            INSERT INTO skill_usage_history (skill_id, success, quality_score)
            VALUES (?, ?, ?)
        """, (skill_id, success, quality_score))

        # 更新成功率（指数移动平均）
        beta = 0.9
        success_value = 1.0 if success else 0.0
        new_success_rate = beta * skill.quality.success_rate + (1 - beta) * success_value

        # 更新置信度
        if quality_score is not None:
            new_confidence = beta * skill.quality.confidence + (1 - beta) * quality_score
        else:
            new_confidence = skill.quality.confidence

        # 更新验证次数
        new_validation_count = skill.quality.validation_count + 1

        # 更新数据库
        skill.quality.success_rate = new_success_rate
        skill.quality.confidence = new_confidence
        skill.quality.validation_count = new_validation_count
        skill.updated_at = datetime.now().isoformat()

        cursor.execute("""
            UPDATE skills
            SET quality = ?, updated_at = ?
            WHERE skill_id = ?
        """, (
            json.dumps(skill.quality.to_dict(), ensure_ascii=False),
            skill.updated_at,
            skill_id
        ))

        conn.commit()
        conn.close()

    def _update_procedural_knowledge(self, skill: AnalysisSkill) -> None:
        """更新技能的过程性知识列"""
        pk = {
            "preconditions": skill.preconditions,
            "procedure": skill.procedure,
            "postconditions": skill.postconditions,
            "error_patterns": skill.error_patterns,
            "tips": skill.tips
        }
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE skills SET procedural_knowledge=?, updated_at=? WHERE skill_id=?",
            (json.dumps(pk, ensure_ascii=False), datetime.now().isoformat(), skill.skill_id)
        )
        conn.commit()
        conn.close()

    def update_execution_profile(self, tool_name: str, success: bool,
                                  duration: float = 0, params: Dict = None,
                                  error: str = None, output_files: list = None,
                                  task_description: str = "") -> None:
        """
        更新工具执行档案（累积学习工具的使用经验）

        每次工具执行后调用，积累参数统计、常见错误、输出模式等。

        Args:
            tool_name: 工具名称
            success: 是否成功
            duration: 执行时长（秒）
            params: 使用的参数
            error: 错误信息（如果失败）
            output_files: 输出文件列表
            task_description: 任务描述
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取现有档案
        cursor.execute("SELECT * FROM tool_execution_profiles WHERE tool_name = ?", (tool_name,))
        row = cursor.fetchone()

        now = datetime.now().isoformat()

        if row:
            total = row[1] + 1
            sc = row[2] + (1 if success else 0)
            # 增量更新平均时长
            old_avg = row[3] or 0
            new_avg = old_avg + (duration - old_avg) / total if duration > 0 else old_avg

            # 更新参数统计
            param_stats = json.loads(row[4] or "{}")
            if params:
                for k, v in params.items():
                    key_stats = param_stats.setdefault(k, {})
                    v_str = str(v)
                    key_stats[v_str] = key_stats.get(v_str, 0) + 1

            # 更新常见错误
            common_errors = json.loads(row[5] or "{}")
            if error and not success:
                # 取错误前100字符作为key
                err_key = error[:100]
                common_errors[err_key] = common_errors.get(err_key, 0) + 1

            # 更新输出模式
            output_patterns = json.loads(row[6] or "[]")
            if output_files and success:
                import os
                exts = sorted(set(os.path.splitext(f)[1] for f in output_files if f))
                ext_str = ",".join(exts)
                if ext_str and ext_str not in output_patterns:
                    output_patterns.append(ext_str)
                    output_patterns = output_patterns[-10:]  # 只保留最近10种

            # 更新 best_for_tasks
            best_for = json.loads(row[7] or "[]")
            if success and task_description:
                label = task_description.strip()[:80]
                if label and label not in best_for:
                    best_for.append(label)
                    best_for = best_for[-15:]  # 只保留最近15个

            cursor.execute("""
                UPDATE tool_execution_profiles
                SET total_executions=?, success_count=?, avg_duration_seconds=?,
                    param_stats=?, common_errors=?, output_patterns=?,
                    best_for_tasks=?, updated_at=?
                WHERE tool_name=?
            """, (total, sc, new_avg,
                  json.dumps(param_stats, ensure_ascii=False),
                  json.dumps(common_errors, ensure_ascii=False),
                  json.dumps(output_patterns, ensure_ascii=False),
                  json.dumps(best_for, ensure_ascii=False),
                  now, tool_name))
        else:
            param_stats = {}
            if params:
                for k, v in params.items():
                    param_stats[k] = {str(v): 1}
            common_errors = {}
            if error and not success:
                common_errors[error[:100]] = 1
            output_patterns = []
            if output_files and success:
                import os
                exts = sorted(set(os.path.splitext(f)[1] for f in output_files if f))
                ext_str = ",".join(exts)
                if ext_str:
                    output_patterns.append(ext_str)
            best_for = []
            if success and task_description:
                best_for.append(task_description.strip()[:80])

            cursor.execute("""
                INSERT INTO tool_execution_profiles
                (tool_name, total_executions, success_count, avg_duration_seconds,
                 param_stats, common_errors, output_patterns, best_for_tasks, updated_at)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
            """, (tool_name, 1 if success else 0, duration if duration > 0 else 0,
                  json.dumps(param_stats, ensure_ascii=False),
                  json.dumps(common_errors, ensure_ascii=False),
                  json.dumps(output_patterns, ensure_ascii=False),
                  json.dumps(best_for, ensure_ascii=False),
                  now))

        conn.commit()
        conn.close()

    def get_execution_profile(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具的执行档案"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM tool_execution_profiles WHERE tool_name = ?", (tool_name,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "tool_name": row[0],
            "total_executions": row[1],
            "success_count": row[2],
            "success_rate": row[2] / row[1] if row[1] > 0 else 0,
            "avg_duration_seconds": row[3],
            "param_stats": json.loads(row[4] or "{}"),
            "common_errors": json.loads(row[5] or "{}"),
            "output_patterns": json.loads(row[6] or "[]"),
            "best_for_tasks": json.loads(row[7] or "[]"),
        }

    def save_pipeline_skill(self, tool_chain: List[Dict], disease: str = "",
                            question: str = "") -> bool:
        """
        保存成功的 pipeline 作为复合技能

        Args:
            tool_chain: 工具链 [{"tool": "spm", "params": {...}}, ...]
            disease: 相关疾病
            question: 研究问题

        Returns:
            是否成功保存
        """
        import hashlib
        # 用工具序列生成唯一ID
        chain_key = "|".join(t["tool"] for t in tool_chain)
        pid = f"pipeline_{hashlib.md5(chain_key.encode()).hexdigest()[:12]}"

        # 推断 research_type
        tools_str = chain_key.lower()
        if "spm" in tools_str and "vbm" in tools_str:
            rtype = "VBM分析"
        elif "fsl" in tools_str and ("dti" in tools_str or "dwi" in tools_str):
            rtype = "DTI分析"
        elif "dpabi" in tools_str:
            rtype = "功能连接分析"
        elif "freesurfer" in tools_str:
            rtype = "皮层分析"
        else:
            rtype = "综合分析"

        # 推断 modality
        modality = "anat"
        for t in tool_chain:
            tn = t["tool"].lower()
            if "fsl" in tn or "dsi" in tn or "dipy" in tn:
                modality = "dwi"
                break
            elif "dpabi" in tn:
                modality = "func"
                break

        now = datetime.now().isoformat()
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 检查是否已存在
            cursor.execute("SELECT success_count, confidence FROM pipeline_skills WHERE pipeline_id = ?", (pid,))
            existing = cursor.fetchone()

            if existing:
                # 已存在，增加成功计数和置信度
                new_count = existing[0] + 1
                new_conf = min(0.95, existing[1] + 0.05)
                cursor.execute("""
                    UPDATE pipeline_skills
                    SET success_count=?, confidence=?, updated_at=?
                    WHERE pipeline_id=?
                """, (new_count, new_conf, now, pid))
            else:
                cursor.execute("""
                    INSERT INTO pipeline_skills
                    (pipeline_id, tool_chain, disease, modality, research_type,
                     description, success_count, confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 1, 0.5, ?, ?)
                """, (pid,
                      json.dumps(tool_chain, ensure_ascii=False),
                      disease, modality, rtype,
                      question[:200] if question else "",
                      now, now))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[SkillLibrary] 保存pipeline失败: {e}")
            return False

    def search_pipeline_skills(self, disease: str = None, modality: str = None,
                                research_type: str = None,
                                min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        """
        搜索匹配的 pipeline 复合技能

        Args:
            disease: 疾病名称
            modality: 模态
            research_type: 研究类型
            min_confidence: 最小置信度

        Returns:
            匹配的 pipeline 列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT * FROM pipeline_skills WHERE confidence >= ?"
        params_list: list = [min_confidence]

        if disease:
            query += " AND disease LIKE ?"
            params_list.append(f"%{disease}%")
        if modality:
            query += " AND modality = ?"
            params_list.append(modality)
        if research_type:
            query += " AND research_type LIKE ?"
            params_list.append(f"%{research_type}%")

        query += " ORDER BY confidence DESC, success_count DESC LIMIT 5"
        cursor.execute(query, params_list)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            results.append({
                "pipeline_id": row[0],
                "tool_chain": json.loads(row[1]),
                "disease": row[2],
                "modality": row[3],
                "research_type": row[4],
                "description": row[5],
                "success_count": row[6],
                "confidence": row[7],
            })
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """获取技能库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 总技能数
        cursor.execute("SELECT COUNT(*) FROM skills")
        total_skills = cursor.fetchone()[0]

        # 按工具分组
        cursor.execute("""
            SELECT tool, COUNT(*)
            FROM skills
            GROUP BY tool
        """)
        skills_by_tool = dict(cursor.fetchall())

        # 按疾病分组
        cursor.execute("""
            SELECT disease, COUNT(DISTINCT skill_id)
            FROM skill_disease_mapping
            GROUP BY disease
        """)
        skills_by_disease = dict(cursor.fetchall())

        # 使用统计
        cursor.execute("""
            SELECT COUNT(*), AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END)
            FROM skill_usage_history
        """)
        usage_stats = cursor.fetchone()

        conn.close()

        return {
            "total_skills": total_skills,
            "skills_by_tool": skills_by_tool,
            "skills_by_disease": skills_by_disease,
            "total_usage": usage_stats[0] if usage_stats else 0,
            "overall_success_rate": usage_stats[1] if usage_stats else 0.0
        }

    def export_to_json(self, output_path: str = None) -> str:
        """
        导出技能库到JSON

        Args:
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        if output_path is None:
            from src.config import OUTPUT_DIR
            output_path = str(OUTPUT_DIR.parent / "data" / "skill_library_export.json")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT skill_id FROM skills")
        skill_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        skills = [self.get_skill(sid).to_dict() for sid in skill_ids]

        export_data = {
            "export_time": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "skills": skills
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return output_path


# ========== 全局实例 ==========

_skill_library = None

def get_skill_library() -> SkillLibrary:
    """获取全局技能库实例"""
    global _skill_library
    if _skill_library is None:
        _skill_library = SkillLibrary()
    return _skill_library
