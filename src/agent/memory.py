"""
Memory Module - Short-term Memory and Long-term Memory
Supports capacity limits and LLM compression
"""
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import deque

from src.config import OUTPUT_DIR

# ========== 记忆系统容量配置 ==========
MEMORY_CONFIG = {
    # 各表最大记录数
    "max_research_patterns": 50,      # 研究模式最多保留50条
    "max_tool_experiences": 200,      # 工具经验最多保留200条
    "max_data_summaries": 100,        # 数据摘要最多保留100条
    "max_successful_runs": 30,        # 成功案例最多保留30条

    # 压缩阈值（超过此比例时触发压缩）
    "compress_threshold": 0.8,        # 达到容量80%时开始压缩

    # 单条记录最大长度（字符）
    "max_summary_length": 2000,       # 摘要最大2000字符
    "max_plan_length": 5000,          # 计划最大5000字符
}


@dataclass
class Message:
    """对话消息"""
    role: str           # user/assistant/system/tool
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    """工具调用记录"""
    call_id: str
    tool_name: str
    inputs: Dict
    outputs: Dict
    status: str
    timestamp: str
    duration: float = 0


@dataclass
class StepResult:
    """步骤结果"""
    step_id: str
    step_name: str
    status: str
    inputs: Dict
    outputs: Dict
    artifacts: List[str]
    started_at: str
    finished_at: str


class WorkingMemory:
    """
    短时记忆 (Working Memory)
    存储当前会话的上下文信息
    """

    def __init__(self, max_conversation_length: int = 20):
        self.session_id: str = ""
        self.question: str = ""
        self.parsed_intent: Dict = {}

        # 对话历史（限制长度）
        self._conversation: deque = deque(maxlen=max_conversation_length)

        # 当前执行计划
        self.current_plan: Dict = {}
        self.current_step: int = 0

        # 步骤结果
        self.step_results: Dict[str, StepResult] = {}

        # 工具调用历史
        self.tool_calls: List[ToolCallRecord] = []

        # 临时变量
        self.variables: Dict[str, Any] = {}

        # 知识证据
        self.evidence: str = ""
        self.citations: List[Dict] = []

    def init_session(self, session_id: str, question: str):
        """初始化会话"""
        self.session_id = session_id
        self.question = question
        self._conversation.clear()
        self.current_plan = {}
        self.current_step = 0
        self.step_results = {}
        self.tool_calls = []
        self.variables = {}

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """添加消息到对话历史"""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self._conversation.append(msg)

    def get_conversation(self, last_n: int = None) -> List[Dict]:
        """获取对话历史"""
        msgs = list(self._conversation)
        if last_n:
            msgs = msgs[-last_n:]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def get_context_summary(self) -> str:
        """获取上下文摘要（用于LLM）"""
        summary = f"""## 当前会话上下文

### 研究问题
{self.question}

### 解析意图
{json.dumps(self.parsed_intent, ensure_ascii=False, indent=2)}

### 执行计划
{json.dumps(self.current_plan, ensure_ascii=False, indent=2) if self.current_plan else '尚未生成'}

### 当前进度
步骤 {self.current_step} / {len(self.current_plan.get('steps', []))}

### 已完成步骤
"""
        for step_id, result in self.step_results.items():
            summary += f"- {result.step_name}: {result.status}\n"

        summary += f"\n### 临时变量\n"
        for k, v in self.variables.items():
            summary += f"- {k}: {str(v)[:100]}...\n" if len(str(v)) > 100 else f"- {k}: {v}\n"

        return summary

    def set_variable(self, name: str, value: Any):
        """设置临时变量"""
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """获取临时变量"""
        return self.variables.get(name, default)

    def add_step_result(self, result: StepResult):
        """添加步骤结果"""
        self.step_results[result.step_id] = result

    def add_tool_call(self, record: ToolCallRecord):
        """添加工具调用记录"""
        self.tool_calls.append(record)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "question": self.question,
            "parsed_intent": self.parsed_intent,
            "conversation": self.get_conversation(),
            "current_plan": self.current_plan,
            "current_step": self.current_step,
            "step_results": {k: asdict(v) for k, v in self.step_results.items()},
            "tool_calls": [asdict(t) for t in self.tool_calls],
            "variables": {k: str(v)[:500] for k, v in self.variables.items()},
            "evidence": self.evidence[:1000] if self.evidence else "",
            "citations": self.citations
        }


class LongTermMemory:
    """
    长时记忆 (Long-term Memory)
    持久化存储，支持跨会话使用
    """

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else OUTPUT_DIR / "memory" / "long_term.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 研究模式表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                question_template TEXT,
                plan_template TEXT,
                success_rate REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 工具经验表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_experiences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT,
                input_pattern TEXT,
                success INTEGER,
                error_message TEXT,
                solution TEXT,
                created_at TEXT
            )
        """)

        # 数据摘要表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_path TEXT UNIQUE,
                summary TEXT,
                statistics TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # 成功案例表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS successful_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE,
                question TEXT,
                plan TEXT,
                result_summary TEXT,
                rating INTEGER,
                created_at TEXT
            )
        """)

        # 用户偏好表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_research_pattern(self, pattern_type: str, question_template: str, plan_template: str):
        """保存研究模式"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO research_patterns
            (pattern_type, question_template, plan_template, success_rate, usage_count, created_at, updated_at)
            VALUES (?, ?, ?, 1.0, 1, ?, ?)
        """, (pattern_type, question_template, plan_template, now, now))

        conn.commit()
        conn.close()

    def get_similar_patterns(self, question: str, limit: int = 3) -> List[Dict]:
        """获取相似的研究模式"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 简单的关键词匹配（后续可改为向量相似度）
        keywords = question.split()
        patterns = []

        for keyword in keywords:
            cursor.execute("""
                SELECT pattern_type, question_template, plan_template, success_rate
                FROM research_patterns
                WHERE question_template LIKE ?
                ORDER BY success_rate DESC, usage_count DESC
                LIMIT ?
            """, (f"%{keyword}%", limit))
            patterns.extend(cursor.fetchall())

        conn.close()

        return [
            {
                "pattern_type": p[0],
                "question_template": p[1],
                "plan_template": p[2],
                "success_rate": p[3]
            }
            for p in patterns[:limit]
        ]

    def save_tool_experience(self, tool_name: str, input_pattern: str,
                             success: bool, error_message: str = "", solution: str = ""):
        """保存工具使用经验"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tool_experiences
            (tool_name, input_pattern, success, error_message, solution, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (tool_name, input_pattern, int(success), error_message, solution, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_tool_experiences(self, tool_name: str, limit: int = 5) -> List[Dict]:
        """获取工具使用经验"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT input_pattern, success, error_message, solution
            FROM tool_experiences
            WHERE tool_name = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (tool_name, limit))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "input_pattern": r[0],
                "success": bool(r[1]),
                "error_message": r[2],
                "solution": r[3]
            }
            for r in results
        ]

    def save_data_summary(self, data_path: str, summary: Dict, statistics: Dict):
        """保存数据摘要"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        now = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO data_summaries
            (data_path, summary, statistics, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (data_path, json.dumps(summary), json.dumps(statistics), now, now))

        conn.commit()
        conn.close()

    def get_data_summary(self, data_path: str) -> Optional[Dict]:
        """获取数据摘要"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT summary, statistics FROM data_summaries WHERE data_path = ?
        """, (data_path,))

        result = cursor.fetchone()
        conn.close()

        if result:
            return {
                "summary": json.loads(result[0]),
                "statistics": json.loads(result[1])
            }
        return None

    def save_successful_run(self, run_id: str, question: str, plan: Dict,
                           result_summary: str, rating: int = 5):
        """保存成功案例"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO successful_runs
            (run_id, question, plan, result_summary, rating, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (run_id, question, json.dumps(plan), result_summary, rating, datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_successful_runs(self, limit: int = 10) -> List[Dict]:
        """获取成功案例"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT run_id, question, plan, result_summary, rating
            FROM successful_runs
            ORDER BY rating DESC, created_at DESC
            LIMIT ?
        """, (limit,))

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "run_id": r[0],
                "question": r[1],
                "plan": json.loads(r[2]),
                "result_summary": r[3],
                "rating": r[4]
            }
            for r in results
        ]

    def set_preference(self, key: str, value: Any):
        """设置用户偏好"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (key, json.dumps(value), datetime.now().isoformat()))

        conn.commit()
        conn.close()

    def get_preference(self, key: str, default: Any = None) -> Any:
        """获取用户偏好"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT value FROM user_preferences WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()

        if result:
            return json.loads(result[0])
        return default

    # ========== 容量管理方法 ==========

    def get_table_counts(self) -> Dict[str, int]:
        """获取各表的记录数"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        counts = {}
        tables = ["research_patterns", "tool_experiences", "data_summaries", "successful_runs"]
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cursor.fetchone()[0]

        conn.close()
        return counts

    def get_capacity_status(self) -> Dict[str, Any]:
        """获取容量状态"""
        counts = self.get_table_counts()
        status = {
            "counts": counts,
            "limits": {
                "research_patterns": MEMORY_CONFIG["max_research_patterns"],
                "tool_experiences": MEMORY_CONFIG["max_tool_experiences"],
                "data_summaries": MEMORY_CONFIG["max_data_summaries"],
                "successful_runs": MEMORY_CONFIG["max_successful_runs"],
            },
            "usage_percent": {},
            "needs_compress": False
        }

        for table, count in counts.items():
            limit = status["limits"][table]
            percent = (count / limit) * 100 if limit > 0 else 0
            status["usage_percent"][table] = round(percent, 1)
            if percent >= MEMORY_CONFIG["compress_threshold"] * 100:
                status["needs_compress"] = True

        return status

    def _delete_oldest_records(self, table: str, keep_count: int):
        """删除最旧的记录，保留指定数量"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 根据表选择合适的排序字段
        if table == "research_patterns":
            order_field = "updated_at"
        elif table == "successful_runs":
            order_field = "created_at"
        else:
            order_field = "created_at"

        # 删除最旧的记录
        cursor.execute(f"""
            DELETE FROM {table}
            WHERE id NOT IN (
                SELECT id FROM {table}
                ORDER BY {order_field} DESC
                LIMIT ?
            )
        """, (keep_count,))

        deleted = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted

    def compress_with_llm(self, llm_client=None) -> Dict[str, Any]:
        """
        使用LLM压缩旧记忆，合并相似内容
        返回压缩结果统计
        """
        results = {
            "compressed": False,
            "tables_processed": [],
            "records_removed": 0,
            "records_merged": 0
        }

        status = self.get_capacity_status()
        if not status["needs_compress"]:
            print("[Memory] 容量正常，无需压缩")
            return results

        print(f"[Memory] 开始压缩记忆... 当前状态: {status['usage_percent']}")

        # 1. 压缩tool_experiences：删除旧的失败记录
        if status["usage_percent"].get("tool_experiences", 0) >= MEMORY_CONFIG["compress_threshold"] * 100:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # 先删除所有失败的旧记录（保留最近20条失败记录用于学习）
            cursor.execute("""
                DELETE FROM tool_experiences
                WHERE success = 0 AND id NOT IN (
                    SELECT id FROM tool_experiences
                    WHERE success = 0
                    ORDER BY created_at DESC
                    LIMIT 20
                )
            """)
            deleted = cursor.rowcount
            results["records_removed"] += deleted
            conn.commit()
            conn.close()

            # 再删除最旧的记录到容量的60%
            target = int(MEMORY_CONFIG["max_tool_experiences"] * 0.6)
            deleted = self._delete_oldest_records("tool_experiences", target)
            results["records_removed"] += deleted
            results["tables_processed"].append("tool_experiences")

        # 2. 压缩successful_runs：使用LLM合并相似案例（如果提供了LLM）
        if status["usage_percent"].get("successful_runs", 0) >= MEMORY_CONFIG["compress_threshold"] * 100:
            if llm_client:
                merged = self._merge_similar_runs_with_llm(llm_client)
                results["records_merged"] += merged
            else:
                # 没有LLM，直接删除最旧的
                target = int(MEMORY_CONFIG["max_successful_runs"] * 0.6)
                deleted = self._delete_oldest_records("successful_runs", target)
                results["records_removed"] += deleted
            results["tables_processed"].append("successful_runs")

        # 3. 压缩research_patterns
        if status["usage_percent"].get("research_patterns", 0) >= MEMORY_CONFIG["compress_threshold"] * 100:
            target = int(MEMORY_CONFIG["max_research_patterns"] * 0.6)
            deleted = self._delete_oldest_records("research_patterns", target)
            results["records_removed"] += deleted
            results["tables_processed"].append("research_patterns")

        # 4. 压缩data_summaries
        if status["usage_percent"].get("data_summaries", 0) >= MEMORY_CONFIG["compress_threshold"] * 100:
            target = int(MEMORY_CONFIG["max_data_summaries"] * 0.6)
            deleted = self._delete_oldest_records("data_summaries", target)
            results["records_removed"] += deleted
            results["tables_processed"].append("data_summaries")

        results["compressed"] = True
        print(f"[Memory] 压缩完成: 删除{results['records_removed']}条, 合并{results['records_merged']}条")
        return results

    def _merge_similar_runs_with_llm(self, llm_client) -> int:
        """使用LLM合并相似的成功案例"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # 获取所有成功案例
        cursor.execute("""
            SELECT id, question, plan, result_summary
            FROM successful_runs
            ORDER BY created_at ASC
        """)
        runs = cursor.fetchall()

        if len(runs) <= 5:
            conn.close()
            return 0

        # 找出相似的案例（简单的关键词匹配）
        merged_count = 0
        to_delete = []

        # 将旧案例分组并压缩
        old_runs = runs[:-5]  # 保留最近5条不压缩
        if len(old_runs) > 3:
            # 使用LLM压缩旧案例摘要
            summaries = [f"问题: {r[1]}\n摘要: {r[3][:200]}" for r in old_runs[:10]]
            compress_prompt = f"""请将以下{len(summaries)}个研究案例压缩成一个通用的经验总结（不超过500字）：

{chr(10).join(summaries)}

输出格式：
一段简洁的经验总结，提取共同的模式和教训。"""

            try:
                response = llm_client.chat([
                    {"role": "system", "content": "你是一个研究助手，擅长总结和压缩信息。"},
                    {"role": "user", "content": compress_prompt}
                ], max_tokens=600, temperature=0.3)

                compressed_summary = response["choices"][0]["message"]["content"]

                # 保存压缩后的摘要，删除原始记录
                now = datetime.now().isoformat()
                cursor.execute("""
                    INSERT INTO successful_runs
                    (run_id, question, plan, result_summary, rating, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    f"compressed_{now}",
                    "多个历史案例的压缩摘要",
                    json.dumps({"type": "compressed", "original_count": len(old_runs[:10])}),
                    compressed_summary,
                    4,
                    now
                ))

                # 删除被压缩的原始记录
                for r in old_runs[:10]:
                    to_delete.append(r[0])
                merged_count = len(old_runs[:10])

            except Exception as e:
                print(f"[Memory] LLM压缩失败: {e}")

        # 执行删除
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            cursor.execute(f"DELETE FROM successful_runs WHERE id IN ({placeholders})", to_delete)

        conn.commit()
        conn.close()

        return merged_count

    def cleanup(self):
        """清理数据库，释放空间"""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("VACUUM")
        conn.close()
        print("[Memory] 数据库清理完成")


class MemoryManager:
    """
    记忆管理器 - 统一管理短时和长时记忆
    支持容量检查和自动压缩
    """

    def __init__(self, llm_client=None):
        self.working = WorkingMemory()
        self.long_term = LongTermMemory()
        self._llm_client = llm_client  # 用于压缩时调用

    def set_llm_client(self, llm_client):
        """设置LLM客户端（用于压缩）"""
        self._llm_client = llm_client

    def init_session(self, session_id: str, question: str):
        """初始化新会话"""
        self.working.init_session(session_id, question)

        # 从长时记忆中检索相关模式
        patterns = self.long_term.get_similar_patterns(question)
        if patterns:
            self.working.set_variable("similar_patterns", patterns)

        # 检查容量状态（不阻塞会话初始化）
        status = self.long_term.get_capacity_status()
        if status["needs_compress"]:
            print(f"[Memory] 警告: 记忆容量较高 {status['usage_percent']}")

    def save_to_long_term(self, force: bool = False) -> Dict[str, Any]:
        """
        将当前会话保存到长时记忆

        Args:
            force: 是否强制保存（即使步骤未全部成功）

        Returns:
            保存结果统计
        """
        result = {
            "saved_run": False,
            "saved_tool_experiences": 0,
            "compressed": False
        }

        # 检查是否需要保存
        has_plan = bool(self.working.current_plan)
        all_succeeded = all(
            r.status == "succeeded" for r in self.working.step_results.values()
        ) if self.working.step_results else False

        # 保存成功案例
        if has_plan and (all_succeeded or force):
            try:
                self.long_term.save_successful_run(
                    run_id=self.working.session_id,
                    question=self.working.question,
                    plan=self.working.current_plan,
                    result_summary=json.dumps(
                        {k: v.outputs for k, v in self.working.step_results.items()},
                        ensure_ascii=False
                    )[:MEMORY_CONFIG["max_summary_length"]],
                    rating=5 if all_succeeded else 3
                )
                result["saved_run"] = True
                print(f"[Memory] 保存成功案例: {self.working.session_id}")
            except Exception as e:
                print(f"[Memory] 保存案例失败: {e}")

        # 保存工具使用经验
        for tool_call in self.working.tool_calls:
            try:
                self.long_term.save_tool_experience(
                    tool_name=tool_call.tool_name,
                    input_pattern=json.dumps(tool_call.inputs)[:500],
                    success=(tool_call.status == "succeeded"),
                    error_message="" if tool_call.status == "succeeded" else str(tool_call.outputs.get("error", ""))[:500]
                )
                result["saved_tool_experiences"] += 1
            except Exception as e:
                print(f"[Memory] 保存工具经验失败: {e}")

        # 检查并执行容量压缩
        compress_result = self.check_and_compress()
        if compress_result["compressed"]:
            result["compressed"] = True

        return result

    def check_and_compress(self) -> Dict[str, Any]:
        """检查容量并在需要时压缩"""
        status = self.long_term.get_capacity_status()
        if status["needs_compress"]:
            print(f"[Memory] 触发自动压缩，当前容量: {status['usage_percent']}")
            return self.long_term.compress_with_llm(self._llm_client)
        return {"compressed": False}

    def get_context_for_llm(self) -> str:
        """获取用于LLM的上下文信息"""
        context = self.working.get_context_summary()

        # 添加相关的长时记忆
        similar = self.working.get_variable("similar_patterns", [])
        if similar:
            context += "\n\n### 相似研究模式\n"
            for p in similar[:2]:
                context += f"- {p['pattern_type']}: {p['question_template'][:100]}...\n"

        return context

    def get_memory_status(self) -> Dict[str, Any]:
        """获取记忆系统状态"""
        return {
            "session_id": self.working.session_id,
            "has_plan": bool(self.working.current_plan),
            "step_count": len(self.working.step_results),
            "tool_call_count": len(self.working.tool_calls),
            "capacity": self.long_term.get_capacity_status()
        }

    def reset(self):
        """重置工作记忆（保留长时记忆）"""
        self.working = WorkingMemory()


# 全局实例
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """获取全局记忆管理器"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
