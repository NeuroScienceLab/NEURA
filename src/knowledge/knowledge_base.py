"""
Knowledge Base Management Module - Integration of Local PDF and PUBMED Search
"""
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from src.config import PAPER_DIR, OUTPUT_DIR
from src.knowledge.pdf_parser import PDFParser, ParsedDocument
from src.knowledge.vector_store import VectorStore, TextChunk, get_vector_store
from src.knowledge.pubmed import PubMedSearcher, PubMedArticle, search_pubmed


@dataclass
class SearchResult:
    """搜索结果"""
    content: str
    source: str
    section_type: str
    score: float
    metadata: Dict


class KnowledgeBase:
    """知识库"""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or get_vector_store()
        self.pdf_parser = PDFParser()
        self.indexed_docs: Dict[str, ParsedDocument] = {}

    def index_local_papers(self, paper_dir: Optional[str] = None):
        """
        索引本地论文

        Args:
            paper_dir: 论文目录路径
        """
        paper_path = Path(paper_dir) if paper_dir else PAPER_DIR

        if not paper_path.exists():
            print(f"论文目录不存在: {paper_path}")
            return

        pdf_files = list(paper_path.glob("*.pdf"))
        print(f"找到 {len(pdf_files)} 个PDF文件")

        for pdf_file in pdf_files:
            print(f"正在索引: {pdf_file.name}")
            try:
                doc = self.pdf_parser.parse(str(pdf_file))
                self.indexed_docs[pdf_file.name] = doc

                # 添加摘要
                if doc.abstract:
                    self.vector_store.add_document(
                        doc_id=f"{pdf_file.stem}_abstract",
                        text=doc.abstract,
                        source=pdf_file.name,
                        section_type="abstract",
                        metadata={"title": doc.title}
                    )

                # 添加各章节
                for i, section in enumerate(doc.sections):
                    self.vector_store.add_document(
                        doc_id=f"{pdf_file.stem}_section_{i}",
                        text=section.content,
                        source=pdf_file.name,
                        section_type=section.section_type,
                        metadata={"title": doc.title, "section_title": section.title}
                    )

            except Exception as e:
                print(f"索引失败 {pdf_file.name}: {e}")

        # 构建索引
        if self.vector_store.chunks:
            print("正在构建向量索引...")
            self.vector_store.build_index()
            self.vector_store.save()
            print(f"索引完成，共 {len(self.vector_store.chunks)} 个文本块")

    def search_local(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        搜索本地知识库

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            搜索结果列表
        """
        try:
            results = self.vector_store.search(query, top_k)
            return [
                SearchResult(
                    content=chunk.content,
                    source=chunk.source,
                    section_type=chunk.section_type,
                    score=score,
                    metadata=chunk.metadata
                )
                for chunk, score in results
            ]
        except Exception as e:
            print(f"本地搜索失败: {e}")
            return []

    def search_pubmed(self, query: str, max_results: int = 5) -> List[PubMedArticle]:
        """
        搜索PubMed

        Args:
            query: 搜索词
            max_results: 最大结果数

        Returns:
            PubMed文章列表
        """
        try:
            return search_pubmed(query, max_results)
        except Exception as e:
            print(f"PubMed搜索失败: {e}")
            return []

    def search(self, query: str, include_pubmed: bool = True) -> Dict:
        """
        综合搜索

        Args:
            query: 查询文本
            include_pubmed: 是否包含PubMed搜索

        Returns:
            搜索结果字典
        """
        results = {
            "local": self.search_local(query),
            "pubmed": []
        }

        if include_pubmed:
            results["pubmed"] = self.search_pubmed(query)

        return results

    def get_evidence(self, query: str) -> str:
        """
        获取证据文本（用于Agent）

        Args:
            query: 查询

        Returns:
            格式化的证据文本
        """
        results = self.search(query)
        evidence_parts = []

        # 本地文献证据
        if results["local"]:
            evidence_parts.append("## 本地文献证据\n")
            for i, r in enumerate(results["local"], 1):
                evidence_parts.append(f"### 证据 {i} (来源: {r.source}, 类型: {r.section_type})")
                evidence_parts.append(f"{r.content[:500]}...")
                evidence_parts.append("")

        # PubMed证据
        if results["pubmed"]:
            evidence_parts.append("\n## PubMed文献\n")
            for article in results["pubmed"]:
                evidence_parts.append(f"### {article.title}")
                evidence_parts.append(f"- 作者: {', '.join(article.authors[:3])}{'等' if len(article.authors) > 3 else ''}")
                evidence_parts.append(f"- 期刊: {article.journal} ({article.year})")
                evidence_parts.append(f"- PMID: {article.pmid}")
                if article.abstract:
                    evidence_parts.append(f"- 摘要: {article.abstract[:300]}...")
                evidence_parts.append("")

        if not evidence_parts:
            return "未找到相关文献证据"

        return "\n".join(evidence_parts)

    def load(self):
        """加载已保存的知识库"""
        try:
            self.vector_store.load()
            print(f"已加载知识库，共 {len(self.vector_store.chunks)} 个文本块")
        except Exception as e:
            print(f"加载知识库失败: {e}")


# 全局实例
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """获取全局知识库"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base
