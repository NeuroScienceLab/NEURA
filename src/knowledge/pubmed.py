"""
PUBMED search module
"""
import re
import httpx
from typing import List, Dict, Optional
from dataclasses import dataclass
from xml.etree import ElementTree

from src.config import PUBMED_EMAIL, PUBMED_MAX_RESULTS


@dataclass
class PubMedArticle:
    """PubMed文章"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    year: str
    doi: Optional[str] = None


class PubMedSearcher:
    """PubMed检索器"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, email: str = PUBMED_EMAIL):
        self.email = email
        self.client = httpx.Client(timeout=30.0)

    def search(self, query: str, max_results: int = PUBMED_MAX_RESULTS) -> List[PubMedArticle]:
        """
        搜索PubMed

        Args:
            query: 搜索词
            max_results: 最大结果数

        Returns:
            文章列表
        """
        # 1. 搜索获取PMID列表
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "email": self.email
        }

        response = self.client.get(search_url, params=search_params)
        response.raise_for_status()

        search_result = response.json()
        pmids = search_result.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # 2. 获取文章详情
        return self._fetch_articles(pmids)

    def _fetch_articles(self, pmids: List[str]) -> List[PubMedArticle]:
        """获取文章详情"""
        fetch_url = f"{self.BASE_URL}/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "email": self.email
        }

        response = self.client.get(fetch_url, params=fetch_params)
        response.raise_for_status()

        return self._parse_xml(response.text)

    def _parse_xml(self, xml_text: str) -> List[PubMedArticle]:
        """解析XML响应"""
        articles = []

        try:
            root = ElementTree.fromstring(xml_text)

            for article_elem in root.findall(".//PubmedArticle"):
                pmid = self._get_text(article_elem, ".//PMID")
                title = self._get_text(article_elem, ".//ArticleTitle")
                abstract = self._get_text(article_elem, ".//AbstractText")
                journal = self._get_text(article_elem, ".//Journal/Title")
                year = self._get_text(article_elem, ".//PubDate/Year") or \
                       self._get_text(article_elem, ".//PubDate/MedlineDate", "")[:4]

                # 获取作者
                authors = []
                for author_elem in article_elem.findall(".//Author"):
                    lastname = self._get_text(author_elem, "LastName", "")
                    forename = self._get_text(author_elem, "ForeName", "")
                    if lastname:
                        authors.append(f"{lastname} {forename}".strip())

                # 获取DOI
                doi = None
                for id_elem in article_elem.findall(".//ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text

                if pmid and title:
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=title,
                        abstract=abstract or "",
                        authors=authors,
                        journal=journal or "",
                        year=year or "",
                        doi=doi
                    ))
        except ElementTree.ParseError as e:
            print(f"XML解析错误: {e}")

        return articles

    def _get_text(self, elem, path: str, default: str = "") -> str:
        """获取XML元素文本"""
        found = elem.find(path)
        if found is not None and found.text:
            return found.text.strip()
        return default

    def close(self):
        """关闭客户端"""
        self.client.close()


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS) -> List[PubMedArticle]:
    """便捷函数：搜索PubMed"""
    searcher = PubMedSearcher()
    try:
        return searcher.search(query, max_results)
    finally:
        searcher.close()
