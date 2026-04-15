"""
Dynamic Knowledge Graph - Weight Update Mechanism Based on Execution Feedback


- Uq: Query-driven updates (基于研究问题模式)
- Ur: Retrieval-driven updates (基于文献检索结果)
- Ue: Execution-driven updates (基于工具执行反馈)

权重更新公式: w'(d,r) = β * w(d,r) + (1-β) * c_new
其中 β = 0.95 (衰减因子)
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.knowledge.tool_knowledge_graph import TOOL_KNOWLEDGE_GRAPH


@dataclass
class WeightUpdate:
    """权重更新记录"""
    entity_type: str  # 'disease_roi' or 'tool_disease'
    entity_key: str
    old_weight: float
    new_weight: float
    evidence_source: str
    timestamp: str


class DynamicKnowledgeGraph:
    """
    动态知识图谱 - 支持基于反馈的权重更新

    核心功能:
    1. 查询疾病-ROI映射（带动态置信度权重）
    2. 查询工具-疾病有效性
    3. 三种更新机制：Uq, Ur, Ue
    4. 持久化存储（SQLite）
    """

    def __init__(self, db_path: str = None):
        """
        初始化动态知识图谱

        Args:
            db_path: 数据库路径，默认为 data/knowledge_graph.db
        """
        if db_path is None:
            from src.config import OUTPUT_DIR
            data_dir = OUTPUT_DIR.parent / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "knowledge_graph.db")

        self.db_path = db_path
        self.static_kg = TOOL_KNOWLEDGE_GRAPH  # 静态知识图谱作为基础

        # 不同更新类型的衰减因子
        # Uq: 用户明确指定，更信任用户输入
        # Ur: 文献检索，中等信任度
        # Ue: 执行反馈，需要更多证据才能改变
        self.beta_query = 0.90      # Uq: Query-driven updates
        self.beta_retrieval = 0.95  # Ur: Retrieval-driven updates
        self.beta_execution = 0.98  # Ue: Execution-driven updates
        self.beta = 0.95  # 默认衰减因子（向后兼容）

        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Enable WAL mode for better concurrent access
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")  # 5 second timeout for locks

        # 1. 疾病-ROI权重表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS disease_roi_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease TEXT NOT NULL,
                roi TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                update_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(disease, roi)
            )
        """)

        # 2. 工具-疾病有效性表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_disease_effectiveness (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool TEXT NOT NULL,
                disease TEXT NOT NULL,
                success_rate REAL DEFAULT 0.5,
                avg_quality_score REAL DEFAULT 0.5,
                execution_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tool, disease)
            )
        """)

        # 3. 更新历史表（用于审计和分析）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kg_update_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_type TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                old_weight REAL,
                new_weight REAL,
                evidence_source TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引以加速查询
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_disease_roi
            ON disease_roi_weights(disease, roi)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_disease
            ON tool_disease_effectiveness(tool, disease)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_update_history
            ON kg_update_history(entity_type, entity_key, timestamp)
        """)

        conn.commit()
        conn.close()

    # ========== 查询方法 ==========

    def _normalize_disease_name(self, disease: str) -> str:
        """
        Normalize disease name for consistent matching

        Args:
            disease: Disease name to normalize

        Returns:
            Normalized disease name (returns the canonical key used in static KG)
        """
        if not disease:
            return ""
        # Remove apostrophes and normalize
        normalized = disease.lower().replace("'", "").replace("'", "").strip()

        # Comprehensive aliases including Chinese names
        # Maps to canonical keys used in TOOL_KNOWLEDGE_GRAPH["disease_roi_mapping"]
        aliases = {
            # Alzheimer's Disease / 阿尔茨海默病
            "alzheimers disease": "阿尔茨海默病",
            "alzheimer disease": "阿尔茨海默病",
            "alzheimers": "阿尔茨海默病",
            "ad": "AD",
            "阿尔茨海默病": "阿尔茨海默病",
            "阿尔茨海默症": "阿尔茨海默病",
            "老年痴呆": "阿尔茨海默病",
            "老年痴呆症": "阿尔茨海默病",
            "老年性痴呆": "阿尔茨海默病",

            # Parkinson's Disease / 帕金森病
            "parkinsons disease": "帕金森病",
            "parkinson disease": "帕金森病",
            "parkinsons": "帕金森病",
            "pd": "PD",
            "帕金森病": "帕金森病",
            "帕金森症": "帕金森病",
            "帕金森氏病": "帕金森病",
            "震颤麻痹": "帕金森病",

            # Spinocerebellar Ataxia / 脊髓小脑共济失调
            "spinocerebellar ataxia": "脊髓小脑共济失调",
            "spinocerebellar ataxia type 3": "SCA3",
            "spinocerebellar ataxia type 2": "SCA2",
            "sca": "SCA",
            "sca2": "SCA2",
            "sca3": "SCA3",
            "脊髓小脑共济失调": "脊髓小脑共济失调",
            "脊髓小脑性共济失调": "脊髓小脑共济失调",
            "小脑共济失调": "脊髓小脑共济失调",
            "共济失调": "脊髓小脑共济失调",

            # Multiple Sclerosis / 多发性硬化
            "multiple sclerosis": "多发性硬化",
            "ms": "MS",
            "多发性硬化": "多发性硬化",
            "多发性硬化症": "多发性硬化",

            # Major Depressive Disorder / 抑郁症
            "major depressive disorder": "抑郁症",
            "depression": "抑郁症",
            "mdd": "MDD",
            "抑郁症": "抑郁症",
            "抑郁障碍": "抑郁症",
            "重度抑郁症": "抑郁症",
            "重性抑郁障碍": "抑郁症",

            # Autism Spectrum Disorder / 自闭症
            "autism spectrum disorder": "自闭症",
            "autism": "自闭症",
            "asd": "ASD",
            "自闭症": "自闭症",
            "孤独症": "自闭症",
            "自闭症谱系障碍": "自闭症",

            # Schizophrenia / 精神分裂症
            "schizophrenia": "精神分裂症",
            "scz": "SCZ",
            "精神分裂症": "精神分裂症",
            "精神分裂": "精神分裂症",

            # Epilepsy / 癫痫
            "epilepsy": "癫痫",
            "temporal lobe epilepsy": "颞叶癫痫",
            "tle": "TLE",
            "癫痫": "癫痫",
            "癫痫症": "癫痫",
            "颞叶癫痫": "颞叶癫痫",

            # ADHD
            "attention deficit hyperactivity disorder": "ADHD",
            "adhd": "ADHD",
            "注意力缺陷多动障碍": "ADHD",
            "多动症": "ADHD",

            # Huntington's Disease / 亨廷顿病
            "huntingtons disease": "亨廷顿病",
            "huntington disease": "亨廷顿病",
            "hd": "HD",
            "亨廷顿病": "亨廷顿病",
            "亨廷顿舞蹈病": "亨廷顿病",
            "舞蹈病": "亨廷顿病",

            # Bipolar Disorder / 双相障碍
            "bipolar disorder": "双相障碍",
            "bd": "BD",
            "双相障碍": "双相障碍",
            "双相情感障碍": "双相障碍",
            "躁郁症": "双相障碍",

            # PTSD / 创伤后应激障碍
            "post traumatic stress disorder": "创伤后应激障碍",
            "ptsd": "PTSD",
            "创伤后应激障碍": "创伤后应激障碍",
            "创伤后应激": "创伤后应激障碍",

            # OCD / 强迫症
            "obsessive compulsive disorder": "强迫症",
            "ocd": "OCD",
            "强迫症": "强迫症",
            "强迫障碍": "强迫症",

            # ALS / 肌萎缩侧索硬化
            "amyotrophic lateral sclerosis": "肌萎缩侧索硬化",
            "als": "ALS",
            "肌萎缩侧索硬化": "肌萎缩侧索硬化",
            "渐冻症": "肌萎缩侧索硬化",
            "渐冻人": "肌萎缩侧索硬化",

            # FTD / 额颞叶痴呆
            "frontotemporal dementia": "额颞叶痴呆",
            "ftd": "FTD",
            "额颞叶痴呆": "额颞叶痴呆",
            "额颞叶变性": "额颞叶痴呆",

            # MCI / 轻度认知障碍
            "mild cognitive impairment": "轻度认知障碍",
            "mci": "MCI",
            "轻度认知障碍": "轻度认知障碍",
            "轻度认知损害": "轻度认知障碍",

            # Anxiety / 焦虑症
            "generalized anxiety disorder": "焦虑症",
            "anxiety disorder": "焦虑症",
            "gad": "GAD",
            "焦虑症": "焦虑症",
            "焦虑障碍": "焦虑症",
            "广泛性焦虑障碍": "焦虑症",

            # Stroke / 脑卒中
            "stroke": "脑卒中",
            "脑卒中": "脑卒中",
            "中风": "脑卒中",
            "脑梗": "脑卒中",
            "脑梗塞": "脑卒中",

            # Brain Tumor / 脑肿瘤
            "brain tumor": "脑肿瘤",
            "脑肿瘤": "脑肿瘤",
            "脑瘤": "脑肿瘤",
        }
        return aliases.get(normalized, disease)

    def get_disease_rois(self, disease: str) -> Dict[str, Any]:
        """
        获取疾病相关的ROI，带动态置信度权重

        Args:
            disease: 疾病名称

        Returns:
            包含ROI列表和动态权重的字典
        """
        # Normalize disease name for consistent matching
        normalized_disease = self._normalize_disease_name(disease)
        matched_disease = disease  # Default to original

        # 1. 从静态知识图谱获取基础信息
        static_info = self.static_kg["disease_roi_mapping"].get(disease, {})

        if not static_info:
            # 尝试模糊匹配 using normalization
            for key, value in self.static_kg["disease_roi_mapping"].items():
                # Use normalized comparison
                if self._normalize_disease_name(key) == normalized_disease:
                    static_info = value
                    matched_disease = key  # Store the matched key for DB query
                    break
                # Fallback to substring matching
                elif key.lower() in disease.lower() or disease.lower() in key.lower():
                    static_info = value
                    matched_disease = key
                    break

        if not static_info:
            return {
                "primary": [],
                "secondary": [],
                "evidence": "无疾病-脑区映射",
                "dynamic_weights": {}
            }

        # 2. 获取动态权重
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        dynamic_weights = {}
        all_rois = static_info.get("primary", []) + static_info.get("secondary", [])

        for roi in all_rois:
            # Use matched_disease for database query (not original disease)
            cursor.execute("""
                SELECT weight, update_count, last_updated
                FROM disease_roi_weights
                WHERE disease = ? AND roi = ?
            """, (matched_disease, roi))

            result = cursor.fetchone()
            if result:
                dynamic_weights[roi] = {
                    "weight": result[0],
                    "update_count": result[1],
                    "last_updated": result[2],
                    "source": "dynamic"
                }
            else:
                # 使用静态置信度作为初始值
                dynamic_weights[roi] = {
                    "weight": static_info.get("confidence", 0.5),
                    "update_count": 0,
                    "last_updated": None,
                    "source": "static"
                }

        conn.close()

        # 3. 合并静态和动态信息
        result = dict(static_info)
        result["dynamic_weights"] = dynamic_weights

        # 按权重排序ROI
        primary_sorted = sorted(
            static_info.get("primary", []),
            key=lambda r: dynamic_weights.get(r, {}).get("weight", 0.5),
            reverse=True
        )
        secondary_sorted = sorted(
            static_info.get("secondary", []),
            key=lambda r: dynamic_weights.get(r, {}).get("weight", 0.5),
            reverse=True
        )

        result["primary"] = primary_sorted
        result["secondary"] = secondary_sorted

        return result

    def get_tool_confidence(self, tool: str, disease: str = None) -> float:
        """
        获取工具的置信度（可选疾病特定）

        Args:
            tool: 工具名称
            disease: 疾病名称（可选）

        Returns:
            置信度分数 (0.0-1.0)
        """
        # 1. 从静态知识图谱获取基础置信度
        tool_info = self.static_kg["tools"].get(tool, {})
        base_confidence = tool_info.get("confidence", 0.5)

        # 2. 如果指定了疾病，查询动态有效性
        if disease:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT success_rate, avg_quality_score, execution_count
                FROM tool_disease_effectiveness
                WHERE tool = ? AND disease = ?
            """, (tool, disease))

            result = cursor.fetchone()
            conn.close()

            if result and result[2] > 0:  # 有执行记录
                # 综合成功率和质量分数
                dynamic_confidence = (result[0] + result[1]) / 2.0
                # 根据执行次数调整权重（执行越多，动态权重越高）
                execution_weight = min(result[2] / 10.0, 0.8)  # 最多80%动态权重
                return base_confidence * (1 - execution_weight) + dynamic_confidence * execution_weight

        return base_confidence

    def get_tool_effectiveness(self, tool: str, disease: str) -> Dict[str, Any]:
        """
        获取工具对特定疾病的有效性统计

        Args:
            tool: 工具名称
            disease: 疾病名称

        Returns:
            有效性统计字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT success_rate, avg_quality_score, execution_count, last_updated
            FROM tool_disease_effectiveness
            WHERE tool = ? AND disease = ?
        """, (tool, disease))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "success_rate": result[0],
                "avg_quality_score": result[1],
                "execution_count": result[2],
                "last_updated": result[3],
                "confidence": (result[0] + result[1]) / 2.0
            }
        else:
            return {
                "success_rate": 0.5,
                "avg_quality_score": 0.5,
                "execution_count": 0,
                "last_updated": None,
                "confidence": 0.5
            }

    # ========== 更新方法 ==========

    def update_query_driven(self, query: str, disease: str, selected_rois: List[str]) -> None:
        """
        Uq: 基于研究问题模式的更新

        当用户明确指定某些ROI时，提升这些ROI的权重

        Args:
            query: 研究问题
            disease: 疾病名称
            selected_rois: 用户选择或问题中提到的ROI列表
        """
        if not selected_rois or not disease:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 对每个选中的ROI提升权重
        for roi in selected_rois:
            # 获取当前权重
            cursor.execute("""
                SELECT weight FROM disease_roi_weights
                WHERE disease = ? AND roi = ?
            """, (disease, roi))

            result = cursor.fetchone()
            current_weight = result[0] if result else 0.5

            # 用户明确选择，给予高置信度
            new_evidence_confidence = 0.9
            new_weight = self._apply_weight_update(current_weight, new_evidence_confidence, 'query')

            # 更新或插入
            cursor.execute("""
                INSERT INTO disease_roi_weights (disease, roi, weight, update_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(disease, roi) DO UPDATE SET
                    weight = ?,
                    update_count = update_count + 1,
                    last_updated = ?
            """, (disease, roi, new_weight, datetime.now().isoformat(),
                  new_weight, datetime.now().isoformat()))

            # 记录更新历史
            cursor.execute("""
                INSERT INTO kg_update_history
                (update_type, entity_type, entity_key, old_weight, new_weight, evidence_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("query", "disease_roi", f"{disease}:{roi}",
                  current_weight, new_weight, f"query:{query[:100]}"))

        conn.commit()
        conn.close()

    def update_retrieval_driven(self, disease: str, literature_findings: Dict) -> None:
        """
        Ur: 基于文献检索结果的更新

        根据检索到的文献中提到的ROI，更新权重

        Args:
            disease: 疾病名称
            literature_findings: 文献发现，格式:
                {
                    "primary_rois": ["roi1", "roi2"],
                    "secondary_rois": ["roi3"],
                    "confidence": 0.8
                }
        """
        if not disease or not literature_findings:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 处理主要ROI
        primary_rois = literature_findings.get("primary_rois", [])
        for roi in primary_rois:
            cursor.execute("""
                SELECT weight FROM disease_roi_weights
                WHERE disease = ? AND roi = ?
            """, (disease, roi))

            result = cursor.fetchone()
            current_weight = result[0] if result else 0.5

            # 文献支持的主要ROI，给予较高置信度
            literature_confidence = literature_findings.get("confidence", 0.8)
            new_weight = self._apply_weight_update(current_weight, literature_confidence, 'retrieval')

            cursor.execute("""
                INSERT INTO disease_roi_weights (disease, roi, weight, update_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(disease, roi) DO UPDATE SET
                    weight = ?,
                    update_count = update_count + 1,
                    last_updated = ?
            """, (disease, roi, new_weight, datetime.now().isoformat(),
                  new_weight, datetime.now().isoformat()))

            cursor.execute("""
                INSERT INTO kg_update_history
                (update_type, entity_type, entity_key, old_weight, new_weight, evidence_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("retrieval", "disease_roi", f"{disease}:{roi}",
                  current_weight, new_weight, "literature_primary"))

        # 处理次要ROI（权重稍低）
        secondary_rois = literature_findings.get("secondary_rois", [])
        for roi in secondary_rois:
            cursor.execute("""
                SELECT weight FROM disease_roi_weights
                WHERE disease = ? AND roi = ?
            """, (disease, roi))

            result = cursor.fetchone()
            current_weight = result[0] if result else 0.5

            # 次要ROI，置信度降低
            literature_confidence = literature_findings.get("confidence", 0.8) * 0.7
            new_weight = self._apply_weight_update(current_weight, literature_confidence, 'retrieval')

            cursor.execute("""
                INSERT INTO disease_roi_weights (disease, roi, weight, update_count, last_updated)
                VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(disease, roi) DO UPDATE SET
                    weight = ?,
                    update_count = update_count + 1,
                    last_updated = ?
            """, (disease, roi, new_weight, datetime.now().isoformat(),
                  new_weight, datetime.now().isoformat()))

            cursor.execute("""
                INSERT INTO kg_update_history
                (update_type, entity_type, entity_key, old_weight, new_weight, evidence_source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ("retrieval", "disease_roi", f"{disease}:{roi}",
                  current_weight, new_weight, "literature_secondary"))

        conn.commit()
        conn.close()

    def update_execution_driven(self, tool: str, disease: str,
                               success: bool, quality_score: float = None) -> None:
        """
        Ue: 基于工具执行结果的更新

        根据工具执行的成功/失败和质量评分，更新工具-疾病有效性

        Args:
            tool: 工具名称
            disease: 疾病名称
            success: 是否执行成功
            quality_score: 质量评分 (0.0-1.0)，可选
        """
        if not tool or not disease:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当前统计
        cursor.execute("""
            SELECT success_rate, avg_quality_score, execution_count
            FROM tool_disease_effectiveness
            WHERE tool = ? AND disease = ?
        """, (tool, disease))

        result = cursor.fetchone()

        if result:
            current_success_rate = result[0]
            current_quality = result[1]
            execution_count = result[2]
        else:
            current_success_rate = 0.5
            current_quality = 0.5
            execution_count = 0

        # 计算新的成功率（指数移动平均）
        success_value = 1.0 if success else 0.0
        new_success_rate = self._apply_weight_update(current_success_rate, success_value, 'execution')

        # 计算新的质量分数
        if quality_score is not None:
            new_quality = self._apply_weight_update(current_quality, quality_score, 'execution')
        else:
            # 如果没有质量分数，使用成功/失败作为质量指标
            new_quality = self._apply_weight_update(current_quality, success_value, 'execution')

        # 更新数据库
        cursor.execute("""
            INSERT INTO tool_disease_effectiveness
            (tool, disease, success_rate, avg_quality_score, execution_count, last_updated)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(tool, disease) DO UPDATE SET
                success_rate = ?,
                avg_quality_score = ?,
                execution_count = execution_count + 1,
                last_updated = ?
        """, (tool, disease, new_success_rate, new_quality, datetime.now().isoformat(),
              new_success_rate, new_quality, datetime.now().isoformat()))

        # 记录更新历史
        cursor.execute("""
            INSERT INTO kg_update_history
            (update_type, entity_type, entity_key, old_weight, new_weight, evidence_source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("execution", "tool_disease", f"{tool}:{disease}",
              current_success_rate, new_success_rate,
              f"success={success},quality={quality_score}"))

        conn.commit()
        conn.close()

    def _apply_weight_update(self, current_weight: float,
                            new_evidence_confidence: float,
                            update_type: str = None) -> float:
        """
        应用权重更新公式

        w'(d,r) = β * w(d,r) + (1-β) * c_new

        Args:
            current_weight: 当前权重
            new_evidence_confidence: 新证据的置信度
            update_type: 更新类型 ('query', 'retrieval', 'execution')，用于选择衰减因子

        Returns:
            更新后的权重
        """
        # 根据更新类型选择衰减因子
        if update_type == 'query':
            beta = self.beta_query
        elif update_type == 'retrieval':
            beta = self.beta_retrieval
        elif update_type == 'execution':
            beta = self.beta_execution
        else:
            beta = self.beta  # 默认

        new_weight = beta * current_weight + (1 - beta) * new_evidence_confidence
        # 确保权重在 [0, 1] 范围内
        return max(0.0, min(1.0, new_weight))

    # ========== 统计和分析方法 ==========

    def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 统计各类更新次数
        cursor.execute("""
            SELECT update_type, COUNT(*)
            FROM kg_update_history
            GROUP BY update_type
        """)
        update_counts = dict(cursor.fetchall())

        # 统计疾病-ROI记录数
        cursor.execute("SELECT COUNT(*) FROM disease_roi_weights")
        roi_count = cursor.fetchone()[0]

        # 统计工具-疾病记录数
        cursor.execute("SELECT COUNT(*) FROM tool_disease_effectiveness")
        tool_count = cursor.fetchone()[0]

        # 最近更新时间
        cursor.execute("""
            SELECT MAX(timestamp) FROM kg_update_history
        """)
        last_update = cursor.fetchone()[0]

        conn.close()

        return {
            "update_counts": update_counts,
            "disease_roi_records": roi_count,
            "tool_disease_records": tool_count,
            "last_update": last_update,
            "total_updates": sum(update_counts.values())
        }

    def cleanup_update_history(self, max_age_days: int = 30, max_records: int = 10000) -> int:
        """
        清理更新历史记录，防止表无限增长

        Args:
            max_age_days: 保留最近N天的记录，默认30天
            max_records: 最多保留N条记录，默认10000条

        Returns:
            删除的记录数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        deleted_count = 0

        try:
            # 1. 删除超过max_age_days天的记录
            cursor.execute("""
                DELETE FROM kg_update_history
                WHERE timestamp < datetime('now', ?)
            """, (f'-{max_age_days} days',))
            deleted_count += cursor.rowcount

            # 2. 如果记录数仍然超过max_records，删除最旧的记录
            cursor.execute("SELECT COUNT(*) FROM kg_update_history")
            current_count = cursor.fetchone()[0]

            if current_count > max_records:
                # 找到要保留的最小ID
                cursor.execute("""
                    SELECT id FROM kg_update_history
                    ORDER BY timestamp DESC
                    LIMIT 1 OFFSET ?
                """, (max_records - 1,))
                result = cursor.fetchone()

                if result:
                    min_id_to_keep = result[0]
                    cursor.execute("""
                        DELETE FROM kg_update_history
                        WHERE id < ?
                    """, (min_id_to_keep,))
                    deleted_count += cursor.rowcount

            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return deleted_count

    def get_top_rois(self, disease: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        获取疾病的Top-K ROI（按权重排序）

        Args:
            disease: 疾病名称
            top_k: 返回数量

        Returns:
            [(roi, weight), ...] 列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT roi, weight
            FROM disease_roi_weights
            WHERE disease = ?
            ORDER BY weight DESC
            LIMIT ?
        """, (disease, top_k))

        results = cursor.fetchall()
        conn.close()

        return results

    def export_to_json(self, output_path: str = None) -> str:
        """
        导出知识图谱到JSON文件

        Args:
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        if output_path is None:
            from src.config import OUTPUT_DIR
            output_path = str(OUTPUT_DIR.parent / "data" / "dynamic_kg_export.json")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 导出疾病-ROI权重
        cursor.execute("SELECT * FROM disease_roi_weights")
        roi_weights = [
            {
                "disease": row[1],
                "roi": row[2],
                "weight": row[3],
                "update_count": row[4],
                "last_updated": row[5]
            }
            for row in cursor.fetchall()
        ]

        # 导出工具-疾病有效性
        cursor.execute("SELECT * FROM tool_disease_effectiveness")
        tool_effectiveness = [
            {
                "tool": row[1],
                "disease": row[2],
                "success_rate": row[3],
                "avg_quality_score": row[4],
                "execution_count": row[5],
                "last_updated": row[6]
            }
            for row in cursor.fetchall()
        ]

        conn.close()

        export_data = {
            "export_time": datetime.now().isoformat(),
            "statistics": self.get_update_statistics(),
            "disease_roi_weights": roi_weights,
            "tool_disease_effectiveness": tool_effectiveness
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return output_path


# ========== 全局实例 ==========

_dynamic_kg = None

def get_dynamic_kg() -> DynamicKnowledgeGraph:
    """获取全局动态知识图谱实例"""
    global _dynamic_kg
    if _dynamic_kg is None:
        _dynamic_kg = DynamicKnowledgeGraph()
    return _dynamic_kg
