"""
PDF parsing module
"""
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class DocumentSection:
    """文档章节"""
    title: str
    content: str
    section_type: str  # abstract, introduction, methods, results, discussion, other
    page_num: int = 0


@dataclass
class ParsedDocument:
    """解析后的文档"""
    file_path: str
    title: str
    authors: List[str] = field(default_factory=list)
    abstract: str = ""
    sections: List[DocumentSection] = field(default_factory=list)
    full_text: str = ""
    metadata: Dict = field(default_factory=dict)


class PDFParser:
    """PDF解析器"""

    # 章节标题模式
    SECTION_PATTERNS = {
        "abstract": r"(?i)^(abstract|摘要)",
        "introduction": r"(?i)^(introduction|引言|背景|1\.\s*introduction)",
        "methods": r"(?i)^(methods?|materials?\s+and\s+methods?|方法|材料与方法|2\.\s*methods?)",
        "results": r"(?i)^(results?|结果|3\.\s*results?)",
        "discussion": r"(?i)^(discussion|讨论|4\.\s*discussion)",
        "conclusion": r"(?i)^(conclusions?|结论|5\.\s*conclusions?)",
        "references": r"(?i)^(references?|参考文献)"
    }

    def __init__(self):
        self._pymupdf = None

    def _get_pymupdf(self):
        """延迟加载pymupdf"""
        if self._pymupdf is None:
            # PyMuPDF 1.24+ 使用 pymupdf，旧版使用 fitz
            try:
                import pymupdf
                if hasattr(pymupdf, 'open'):
                    self._pymupdf = pymupdf
                else:
                    raise ImportError("pymupdf.open not found")
            except ImportError:
                try:
                    import fitz
                    if hasattr(fitz, 'open'):
                        self._pymupdf = fitz
                    else:
                        raise ImportError("fitz.open not found")
                except ImportError:
                    raise ImportError("无法导入 pymupdf 或 fitz，请安装: pip install pymupdf")
        return self._pymupdf

    def parse(self, pdf_path: str) -> ParsedDocument:
        """
        解析PDF文档

        Args:
            pdf_path: PDF文件路径

        Returns:
            解析后的文档对象
        """
        pymupdf = self._get_pymupdf()
        pdf_path = Path(pdf_path)

        # 确保static目录存在（某些PDF需要）
        static_dir = Path("static")
        if not static_dir.exists():
            static_dir.mkdir(exist_ok=True)

        doc = pymupdf.open(str(pdf_path))

        # 提取全文
        full_text = ""
        pages_text = []
        for page_num, page in enumerate(doc):
            try:
                text = page.get_text()
                # 清理特殊字符，避免编码问题
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
            except Exception as e:
                # 某些页面可能提取失败，跳过
                print(f"  警告: 页面 {page_num + 1} 提取失败: {e}")
                text = ""
            pages_text.append((page_num + 1, text))
            full_text += text + "\n"

        doc.close()

        # 提取标题（通常在第一页开头）
        title = self._extract_title(pages_text[0][1] if pages_text else "")

        # 提取摘要
        abstract = self._extract_abstract(full_text)

        # 分割章节
        sections = self._split_sections(full_text)

        return ParsedDocument(
            file_path=str(pdf_path),
            title=title,
            abstract=abstract,
            sections=sections,
            full_text=full_text,
            metadata={"file_name": pdf_path.name}
        )

    def _extract_title(self, first_page_text: str) -> str:
        """提取标题"""
        lines = first_page_text.strip().split("\n")
        # 假设标题在前几行中，是最长的非空行之一
        title_candidates = []
        for line in lines[:10]:
            line = line.strip()
            if len(line) > 20 and len(line) < 300:
                title_candidates.append(line)

        if title_candidates:
            return title_candidates[0]
        return "Unknown Title"

    def _extract_abstract(self, text: str) -> str:
        """提取摘要"""
        # 查找Abstract部分
        abstract_match = re.search(
            r"(?i)abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|1\.|background))",
            text,
            re.DOTALL
        )
        if abstract_match:
            return abstract_match.group(1).strip()
        return ""

    def _split_sections(self, text: str) -> List[DocumentSection]:
        """分割章节"""
        sections = []
        lines = text.split("\n")

        current_section = None
        current_content = []
        current_type = "other"

        for line in lines:
            line_stripped = line.strip()

            # 检查是否是章节标题
            new_section_type = self._detect_section_type(line_stripped)

            if new_section_type and new_section_type != current_type:
                # 保存当前章节
                if current_section:
                    sections.append(DocumentSection(
                        title=current_section,
                        content="\n".join(current_content),
                        section_type=current_type
                    ))

                current_section = line_stripped
                current_content = []
                current_type = new_section_type
            else:
                current_content.append(line)

        # 保存最后一个章节
        if current_section and current_content:
            sections.append(DocumentSection(
                title=current_section,
                content="\n".join(current_content),
                section_type=current_type
            ))

        return sections

    def _detect_section_type(self, line: str) -> Optional[str]:
        """检测章节类型"""
        for section_type, pattern in self.SECTION_PATTERNS.items():
            if re.match(pattern, line):
                return section_type
        return None


def parse_pdf(pdf_path: str) -> ParsedDocument:
    """便捷函数：解析PDF"""
    parser = PDFParser()
    return parser.parse(pdf_path)
