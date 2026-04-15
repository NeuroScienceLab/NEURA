"""
Statistical analysis tool module
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.config import OUTPUT_DIR


@dataclass
class StatResult:
    """统计结果"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    description: str = ""


class StatsTools:
    """统计分析工具"""

    def __init__(self):
        self._np = None
        self._scipy = None
        self._stats = None

    def _get_numpy(self):
        if self._np is None:
            import numpy as np
            self._np = np
        return self._np

    def _get_scipy_stats(self):
        if self._stats is None:
            from scipy import stats
            self._stats = stats
        return self._stats

    def t_test_independent(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict[str, Any]:
        """
        独立样本t检验

        Args:
            group1: 第一组数据
            group2: 第二组数据
            group1_name: 第一组名称
            group2_name: 第二组名称

        Returns:
            统计结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        g1 = np.array(group1)
        g2 = np.array(group2)

        # t检验
        t_stat, p_value = stats.ttest_ind(g1, g2)

        # Cohen's d 效应量
        pooled_std = np.sqrt(((len(g1)-1)*np.std(g1, ddof=1)**2 +
                              (len(g2)-1)*np.std(g2, ddof=1)**2) /
                             (len(g1)+len(g2)-2))
        cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

        return {
            "test": "Independent Samples T-Test",
            "groups": {
                group1_name: {
                    "n": len(g1),
                    "mean": float(np.mean(g1)),
                    "std": float(np.std(g1, ddof=1)),
                    "min": float(np.min(g1)),
                    "max": float(np.max(g1))
                },
                group2_name: {
                    "n": len(g2),
                    "mean": float(np.mean(g2)),
                    "std": float(np.std(g2, ddof=1)),
                    "min": float(np.min(g2)),
                    "max": float(np.max(g2))
                }
            },
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05,
            "interpretation": self._interpret_effect_size(cohens_d)
        }

    def mann_whitney_u(
        self,
        group1: List[float],
        group2: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict[str, Any]:
        """
        Mann-Whitney U检验（非参数）

        Args:
            group1: 第一组数据
            group2: 第二组数据
            group1_name: 第一组名称
            group2_name: 第二组名称

        Returns:
            统计结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        g1 = np.array(group1)
        g2 = np.array(group2)

        u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')

        # 计算效应量 r = Z / sqrt(N)
        n1, n2 = len(g1), len(g2)
        z = stats.norm.ppf(1 - p_value/2)
        r = z / np.sqrt(n1 + n2)

        return {
            "test": "Mann-Whitney U Test",
            "groups": {
                group1_name: {
                    "n": n1,
                    "median": float(np.median(g1)),
                    "iqr": float(np.percentile(g1, 75) - np.percentile(g1, 25))
                },
                group2_name: {
                    "n": n2,
                    "median": float(np.median(g2)),
                    "iqr": float(np.percentile(g2, 75) - np.percentile(g2, 25))
                }
            },
            "u_statistic": float(u_stat),
            "p_value": float(p_value),
            "effect_size_r": float(r),
            "significant": p_value < 0.05
        }

    def correlation(
        self,
        x: List[float],
        y: List[float],
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        相关性分析

        Args:
            x: 变量X
            y: 变量Y
            method: pearson 或 spearman

        Returns:
            统计结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        x_arr = np.array(x)
        y_arr = np.array(y)

        if method == "pearson":
            r, p = stats.pearsonr(x_arr, y_arr)
        else:
            r, p = stats.spearmanr(x_arr, y_arr)

        return {
            "test": f"{method.capitalize()} Correlation",
            "n": len(x_arr),
            "correlation": float(r),
            "p_value": float(p),
            "r_squared": float(r**2),
            "significant": p < 0.05,
            "interpretation": self._interpret_correlation(r)
        }

    def anova_oneway(
        self,
        groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        单因素方差分析

        Args:
            groups: 组名到数据的字典

        Returns:
            统计结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        group_data = [np.array(v) for v in groups.values()]
        group_names = list(groups.keys())

        f_stat, p_value = stats.f_oneway(*group_data)

        # 计算eta squared
        all_data = np.concatenate(group_data)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        group_stats = {}
        for name, data in zip(group_names, group_data):
            group_stats[name] = {
                "n": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1))
            }

        return {
            "test": "One-Way ANOVA",
            "groups": group_stats,
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "eta_squared": float(eta_squared),
            "significant": p_value < 0.05,
            "interpretation": self._interpret_eta_squared(eta_squared)
        }

    def descriptive_stats(self, data: List[float], name: str = "Variable") -> Dict[str, Any]:
        """
        描述性统计

        Args:
            data: 数据列表
            name: 变量名

        Returns:
            描述性统计结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        arr = np.array(data)

        return {
            "variable": name,
            "n": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "se": float(np.std(arr, ddof=1) / np.sqrt(len(arr))),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "skewness": float(stats.skew(arr)),
            "kurtosis": float(stats.kurtosis(arr))
        }

    def normality_test(self, data: List[float]) -> Dict[str, Any]:
        """
        正态性检验

        Args:
            data: 数据列表

        Returns:
            正态性检验结果
        """
        np = self._get_numpy()
        stats = self._get_scipy_stats()

        arr = np.array(data)

        # Shapiro-Wilk检验（样本量<5000）
        if len(arr) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(arr)
        else:
            shapiro_stat, shapiro_p = None, None

        # Kolmogorov-Smirnov检验
        ks_stat, ks_p = stats.kstest(arr, 'norm', args=(np.mean(arr), np.std(arr)))

        return {
            "test": "Normality Tests",
            "n": len(arr),
            "shapiro_wilk": {
                "statistic": float(shapiro_stat) if shapiro_stat else None,
                "p_value": float(shapiro_p) if shapiro_p else None,
                "normal": shapiro_p > 0.05 if shapiro_p else None
            },
            "kolmogorov_smirnov": {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "normal": ks_p > 0.05
            },
            "recommendation": "使用参数检验" if (shapiro_p and shapiro_p > 0.05) else "考虑使用非参数检验"
        }

    def _interpret_effect_size(self, d: float) -> str:
        """解释Cohen's d"""
        d = abs(d)
        if d < 0.2:
            return "效应量极小 (negligible)"
        elif d < 0.5:
            return "效应量小 (small)"
        elif d < 0.8:
            return "效应量中等 (medium)"
        else:
            return "效应量大 (large)"

    def _interpret_correlation(self, r: float) -> str:
        """解释相关系数"""
        r = abs(r)
        if r < 0.1:
            return "无相关 (negligible)"
        elif r < 0.3:
            return "弱相关 (weak)"
        elif r < 0.5:
            return "中等相关 (moderate)"
        elif r < 0.7:
            return "强相关 (strong)"
        else:
            return "极强相关 (very strong)"

    def _interpret_eta_squared(self, eta2: float) -> str:
        """解释eta squared"""
        if eta2 < 0.01:
            return "效应量极小"
        elif eta2 < 0.06:
            return "效应量小"
        elif eta2 < 0.14:
            return "效应量中等"
        else:
            return "效应量大"


def get_tool_definitions() -> List[Dict]:
    """返回统计工具的函数定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "t_test",
                "description": "进行独立样本t检验，比较两组数据的均值差异",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group1": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "第一组数据"
                        },
                        "group2": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "第二组数据"
                        },
                        "group1_name": {
                            "type": "string",
                            "description": "第一组名称"
                        },
                        "group2_name": {
                            "type": "string",
                            "description": "第二组名称"
                        }
                    },
                    "required": ["group1", "group2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "mann_whitney",
                "description": "进行Mann-Whitney U检验（非参数），用于非正态分布数据的两组比较",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group1": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "第一组数据"
                        },
                        "group2": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "第二组数据"
                        }
                    },
                    "required": ["group1", "group2"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "correlation",
                "description": "计算两个变量之间的相关性",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "变量X"
                        },
                        "y": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "变量Y"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["pearson", "spearman"],
                            "description": "相关系数类型"
                        }
                    },
                    "required": ["x", "y"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "descriptive_stats",
                "description": "计算描述性统计（均值、标准差、中位数等）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "数据列表"
                        },
                        "name": {
                            "type": "string",
                            "description": "变量名称"
                        }
                    },
                    "required": ["data"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "normality_test",
                "description": "进行正态性检验，判断数据是否服从正态分布",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "数据列表"
                        }
                    },
                    "required": ["data"]
                }
            }
        }
    ]


# 全局实例
_stats_tools = None

def get_stats_tools() -> StatsTools:
    """获取全局统计工具实例"""
    global _stats_tools
    if _stats_tools is None:
        _stats_tools = StatsTools()
    return _stats_tools
