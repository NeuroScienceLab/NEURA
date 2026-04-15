"""
Mixture-of-Experts Review (MoER) 模块

将不同类型的输出路由到专业审查器
- PlanReviewer: 研究计划可行性审查
- CodeReviewer: 代码正确性审查（委托给 vibe_coding._llm_code_review）
- StatReviewer: 统计严谨性审查
- ReportReviewer: 报告质量审查（委托给 report_auditor.ReportAuditor）
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.agent.standards import load_tool_standard, load_review_criteria
from src.agent.prompts import MOER_PROMPTS


# ============== LLM 审查辅助函数 ==============

# 统计相关字段 — 与 stats_tools 输出结构对齐
_STAT_KEYS = {
    "test", "p_value", "t_statistic", "f_statistic", "cohens_d",
    "eta_squared", "correlation", "r_squared", "u_statistic",
    "effect_size", "significant", "groups", "n", "mean", "std",
    "median", "iqr", "interpretation", "correction_method",
    "corrected_p", "fdr_corrected", "fwe_corrected",
    "shapiro_p", "levene_p", "normality_test", "homogeneity_test",
}
# QC 相关字段 — 与神经影像工具输出对齐
_QC_KEYS = {
    "gm_volume_ml", "wm_volume_ml", "tiv_ml", "max_motion",
    "mean_fd", "tsnr", "outlier_ratio", "nonzero_ratio",
    "cross_correlation", "fa_mean", "md_mean",
}


def _extract_statistical_evidence(tool_results: List[Dict],
                                   max_items: int = 15) -> List[Dict]:
    """
    从 tool_results 中提取科学相关字段供 LLM 审查。

    替代原来只取 tool/status/step 三个元数据字段的做法，
    保留实际统计值（p_value、t_statistic、cohens_d 等）和 QC 指标。
    """
    evidence = []
    for r in tool_results[:max_items]:
        record = {
            "tool": str(r.get("tool", ""))[:50],
            "step": str(r.get("step", ""))[:50],
            "status": r.get("status", ""),
        }
        # 从 outputs / result 中提取统计和 QC 字段
        outputs = r.get("outputs", r.get("result", {}))
        if isinstance(outputs, dict):
            for key in _STAT_KEYS | _QC_KEYS:
                if key in outputs:
                    val = outputs[key]
                    if isinstance(val, dict):
                        record[key] = {k: v for k, v in list(val.items())[:5]}
                    elif isinstance(val, list):
                        record[key] = val[:10]
                    else:
                        record[key] = val
        # 提取 spec_check 失败信息
        spec = r.get("output_spec_check", {})
        if spec and not spec.get("passed", True):
            record["spec_issues"] = {
                "missing_files": spec.get("missing_files", [])[:3],
                "quality_issues": spec.get("quality_issues", [])[:3],
                "acceptance": spec.get("acceptance_result", {}).get("status", ""),
            }
        # 提取科学相关参数
        params = r.get("params", {})
        if isinstance(params, dict):
            for pk in ("fwhm", "threshold", "correction", "alpha",
                        "contrast", "covariates"):
                if pk in params:
                    record.setdefault("params", {})[pk] = params[pk]
        # 只保留有实际数据的记录
        if len(record) > 3:
            evidence.append(record)
    return evidence


def _validate_review_schema(response: Dict, review_type: str) -> Dict:
    """
    校验并规范化 LLM 审查输出的 JSON 结构。

    确保 issues/suggestions/score 字段存在且类型正确，
    防止 LLM 返回非预期字段名时静默丢失数据。
    """
    if not isinstance(response, dict):
        return {"issues": [], "suggestions": [], "score": 50}

    if not isinstance(response.get("issues"), list):
        response["issues"] = []
    if not isinstance(response.get("suggestions"), list):
        response["suggestions"] = []
    if not isinstance(response.get("score"), (int, float)):
        response["score"] = 50

    response["score"] = max(0, min(100, int(response["score"])))

    validated_issues = []
    for issue in response["issues"]:
        if isinstance(issue, dict) and issue.get("message"):
            issue.setdefault("severity", "warning")
            issue.setdefault("category", "methodology")
            if issue["severity"] not in ("error", "warning", "info"):
                issue["severity"] = "warning"
            validated_issues.append(issue)
        elif isinstance(issue, str):
            validated_issues.append({
                "severity": "warning",
                "category": "methodology",
                "message": issue,
            })
    response["issues"] = validated_issues
    return response


class MoERReviewer:
    """MoER 统一审查器"""

    def __init__(self, llm_client=None):
        self.llm = llm_client
        self.criteria = load_review_criteria()

    # --- PlanReviewer ---
    def review_plan(self, plan: Dict, parsed_intent: Dict,
                    available_tools: List[str], evidence: str = "") -> Dict:
        """
        审查研究计划的完整性和可行性

        宽松模式：只记录结果，不阻止流程
        """
        print("  [MoER:PlanReviewer] 审查研究计划...")
        issues = []
        suggestions = []
        checked = {"completeness": False, "feasibility": False, "methodology": False}
        score = 100

        # === 1. 结构完整性（规则检查） ===
        plan_criteria = self.criteria.get("plan_review", {})
        required_fields = plan_criteria.get("required_plan_fields",
                                            ["title", "pipeline"])
        for field in required_fields:
            # pipeline 可能在 plan 的不同位置
            if field == "pipeline":
                has_pipeline = bool(plan.get("pipeline")) or bool(plan.get("steps"))
                if not has_pipeline:
                    issues.append({
                        "severity": "error",
                        "category": "completeness",
                        "message": f"计划缺少必要字段: {field}"
                    })
                    score -= 15
            elif field == "hypothesis":
                # hypothesis 可能在 design 或顶层
                has_hyp = bool(plan.get("hypothesis")) or bool(
                    plan.get("design", {}).get("hypotheses"))
                if not has_hyp:
                    issues.append({
                        "severity": "warning",
                        "category": "completeness",
                        "message": "计划缺少明确的研究假设"
                    })
                    score -= 5
            elif field == "statistical_methods":
                has_stats = bool(plan.get("statistics")) or bool(
                    plan.get("statistical_methods"))
                if not has_stats:
                    issues.append({
                        "severity": "warning",
                        "category": "completeness",
                        "message": "计划缺少统计方法说明"
                    })
                    score -= 5
            elif not plan.get(field):
                issues.append({
                    "severity": "warning",
                    "category": "completeness",
                    "message": f"计划缺少字段: {field}"
                })
                score -= 5

        checked["completeness"] = True

        # === 2. 工具可行性（规则检查） ===
        pipeline = plan.get("pipeline", plan.get("steps", []))
        if isinstance(pipeline, list):
            for step in pipeline:
                if isinstance(step, dict):
                    tool = step.get("tool", "")
                    if tool and available_tools and tool not in available_tools:
                        issues.append({
                            "severity": "error",
                            "category": "feasibility",
                            "message": f"Pipeline引用了不存在的工具: {tool}"
                        })
                        score -= 10
        checked["feasibility"] = True

        # === 3. 流程合理性 + ROI合理性（LLM审查） ===
        if self.llm:
            try:
                llm_review = self._llm_plan_review(plan, parsed_intent, evidence)
                if llm_review.get("issues"):
                    for issue in llm_review["issues"]:
                        issues.append(issue)
                    # 注意：不从 score 中扣除 LLM issues，因为下面的加权混合
                    # 已经通过 LLM score 反映了这些问题，避免双重惩罚
                if llm_review.get("suggestions"):
                    suggestions.extend(llm_review["suggestions"])
                if llm_review.get("score"):
                    # 取LLM评分和规则评分的加权平均
                    score = int(score * 0.6 + llm_review["score"] * 0.4)
            except Exception as e:
                print(f"    [PlanReviewer] LLM审查失败: {e}")

        checked["methodology"] = True
        score = max(0, min(100, score))

        # 确定状态
        error_count = sum(1 for i in issues if i.get("severity") == "error")
        if error_count > 0:
            status = "needs_revision"
        elif score < 60:
            status = "needs_revision"
        else:
            status = "approved"

        result = {
            "reviewer": "PlanReviewer",
            "status": status,
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "checked_items": checked,
            "timestamp": datetime.now().isoformat()
        }

        print(f"    状态: {status}, 评分: {score}/100, "
              f"问题: {len(issues)}")
        return result

    def _llm_plan_review(self, plan: Dict, parsed_intent: Dict,
                         evidence: str) -> Dict:
        """使用LLM进行计划审查 — 注入神经影像标准"""
        prompt = MOER_PROMPTS["plan_reviewer"]

        # 从 review_criteria.json 注入具体标准
        neuro = self.criteria.get("neuroimaging_criteria", {})
        reporting = self.criteria.get("reporting_guidelines", {})
        plan_weights = self.criteria.get("plan_review", {}).get("weights", {})

        stat_thresh = neuro.get("statistical_thresholds", {})
        motion_thresh = neuro.get("motion_thresholds", {})

        standards_ctx = f"""
## 审查标准参考（必须依据以下标准评判）

### 统计阈值
- Cluster-forming: p < {stat_thresh.get('cluster_forming_p', 0.001)} ({stat_thresh.get('cluster_forming_note', 'Woo et al. 2014')})
- FWE cluster: p < {stat_thresh.get('fwe_cluster_p', 0.05)}
- FDR: q < {stat_thresh.get('fdr_q', 0.05)}
- TFCE: {stat_thresh.get('tfce_note', '不需要cluster-forming threshold')}

### 运动参数
- fMRI FD理想值: < {motion_thresh.get('fmri_fd_ideal_mm', 0.2)}mm
- fMRI FD排除: > {motion_thresh.get('fmri_fd_exclude_mm', 0.5)}mm
- 参考: {motion_thresh.get('reference', 'Power et al. 2012')}

### COBIDAS要求
{json.dumps(reporting.get('cobidas', {}).get('required_items', []), ensure_ascii=False)}
参考: {reporting.get('cobidas', {}).get('reference', '')}

### OHBM最佳实践
{json.dumps(reporting.get('ohbm_best_practices', {}).get('key_points', []), ensure_ascii=False)}

### 评分权重
完整性: {plan_weights.get('completeness', 25)}%, 可行性: {plan_weights.get('feasibility', 25)}%, 方法学: {plan_weights.get('methodology', 30)}%, ROI: {plan_weights.get('roi_rationale', 20)}%
"""

        # 智能截断：保留 pipeline 结构
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2)[:5000]
        intent_text = json.dumps(parsed_intent, ensure_ascii=False)[:1500]

        messages = [
            {"role": "system", "content": prompt + "\n" + standards_ctx},
            {"role": "user", "content": f"""请审查以下研究计划：

## 研究意图
{intent_text}

## 研究计划
{plan_text}

## 文献证据
{evidence[:1500] if evidence else '无'}

请严格依据上述标准审查，以JSON格式返回审查结果。"""}
        ]

        response = self.llm.generate_json(messages)
        return _validate_review_schema(response, "plan")

    # --- CodeReviewer ---
    def review_code(self, code: str, task_context: str = "") -> Dict:
        """
        代码正确性审查 - 委托给 VibeCodingEngine._llm_code_review
        """
        print("  [MoER:CodeReviewer] 审查代码...")
        try:
            from src.agent.vibe_coding import VibeCodingEngine
            engine = VibeCodingEngine()
            review = engine._llm_code_review(code)
            return {
                "reviewer": "CodeReviewer",
                "status": "approved" if not review.get("has_issues") else "needs_revision",
                "score": review.get("quality_score", 70),
                "issues": review.get("issues", []),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    [CodeReviewer] 审查失败: {e}")
            return {
                "reviewer": "CodeReviewer",
                "status": "error",
                "score": 0,
                "issues": [{"severity": "error", "category": "review_failed",
                            "message": f"代码审查异常: {e}"}],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # --- StatReviewer ---
    def review_statistics(self, tool_results: List[Dict],
                          cohort: Dict, plan: Dict) -> Dict:
        """
        统计严谨性审查

        阻断模式：当 status="rejected"（工具成功率<30%）时会影响
        validation 的 overall_decision，触发 reflect_and_fix 流程
        """
        print("  [MoER:StatReviewer] 审查统计严谨性...")
        issues = []
        score = 100
        stat_checks = {
            "multiple_comparison": {"applied": False, "method": "none"},
            "effect_sizes": {"reported": False, "types": []},
            "assumptions_tested": {"normality": False, "homogeneity": False},
            "sample_adequacy": {"min_group_n": 0, "adequate": True}
        }

        if not tool_results:
            return {
                "reviewer": "StatReviewer",
                "status": "approved",
                "score": 50,
                "issues": [{"severity": "warning", "category": "no_data",
                            "message": "无工具结果可供审查"}],
                "stat_checks": stat_checks,
                "timestamp": datetime.now().isoformat()
            }

        # 将所有结果转为文本用于关键词搜索
        # 注意：只在统计相关工具的输出中搜索，避免 motion_correction 等误匹配
        stat_tool_types = {"t_test", "ttest", "anova", "correlation", "regression",
                           "mann_whitney", "wilcoxon", "chi_square", "kruskal",
                           "vbm", "tbss", "glm", "statistical", "stats"}

        # === 1. 多重比较校正 ===
        correction_keywords = ["fdr", "fwe", "bonferroni", "holm",
                               "benjamini", "hochberg",
                               "multiple_comparison", "corrected_p"]
        for r in tool_results:
            tool = str(r.get("tool", "")).lower()
            step = str(r.get("step", "")).lower()
            outputs = r.get("outputs", r.get("result", {}))
            if not isinstance(outputs, dict):
                outputs = {}
            # 只检查统计相关工具的输出，或包含统计字段的输出
            is_stat_tool = any(st in tool or st in step for st in stat_tool_types)
            if is_stat_tool or any(k in outputs for k in _STAT_KEYS):
                outputs_text = json.dumps(outputs, ensure_ascii=False,
                                          default=str).lower()
                params_text = json.dumps(r.get("params", {}),
                                         ensure_ascii=False, default=str).lower()
                check_text = outputs_text + " " + params_text
                for kw in correction_keywords:
                    if kw in check_text:
                        stat_checks["multiple_comparison"]["applied"] = True
                        stat_checks["multiple_comparison"]["method"] = kw
                        break
            if stat_checks["multiple_comparison"]["applied"]:
                break

        # 计算统计检验数量（基于工具名和步骤名）
        stat_test_keywords = ["t_test", "ttest", "anova", "correlation",
                              "regression", "mann_whitney", "wilcoxon",
                              "chi_square", "kruskal"]
        stat_test_count = 0
        for r in tool_results:
            tool = str(r.get("tool", "")).lower()
            step = str(r.get("step", "")).lower()
            tool_step = tool + " " + step
            for kw in stat_test_keywords:
                if kw in tool_step:
                    stat_test_count += 1
                    break

        if stat_test_count > 1 and not stat_checks["multiple_comparison"]["applied"]:
            issues.append({
                "severity": "warning",
                "category": "multiple_comparison",
                "message": f"执行了{stat_test_count}个统计检验但未检测到多重比较校正"
            })
            score -= 10

        # === 2. 效应量报告（检查输出字段） ===
        effect_keywords = {
            "cohen": "Cohen's d", "eta_squared": "eta²",
            "eta-squared": "eta²", "r_squared": "R²",
            "effect_size": "effect_size", "odds_ratio": "OR",
            "cohens_d": "Cohen's d"
        }
        for r in tool_results:
            outputs = r.get("outputs", r.get("result", {}))
            if not isinstance(outputs, dict):
                continue
            outputs_text = json.dumps(outputs, ensure_ascii=False,
                                      default=str).lower()
            for kw, name in effect_keywords.items():
                if kw in outputs_text:
                    stat_checks["effect_sizes"]["reported"] = True
                    if name not in stat_checks["effect_sizes"]["types"]:
                        stat_checks["effect_sizes"]["types"].append(name)

        if not stat_checks["effect_sizes"]["reported"]:
            issues.append({
                "severity": "warning",
                "category": "effect_size",
                "message": "未检测到效应量报告"
            })
            score -= 5

        # === 3. 假设检验（检查输出字段） ===
        for r in tool_results:
            outputs = r.get("outputs", r.get("result", {}))
            if not isinstance(outputs, dict):
                continue
            outputs_text = json.dumps(outputs, ensure_ascii=False,
                                      default=str).lower()
            if "shapiro" in outputs_text or "normality" in outputs_text:
                stat_checks["assumptions_tested"]["normality"] = True
            if "levene" in outputs_text or "homogeneity" in outputs_text:
                stat_checks["assumptions_tested"]["homogeneity"] = True

        # === 4. 样本量充分性 ===
        thresholds = self.criteria.get("stat_review", {}).get("thresholds", {})
        min_warn = thresholds.get("min_group_n_warning", 10)
        min_err = thresholds.get("min_group_n_error", 5)

        groups = cohort.get("groups", {})
        min_n = float('inf')
        for group_data in groups.values():
            if isinstance(group_data, dict):
                n = group_data.get("n", len(group_data.get("subjects", [])))
            elif isinstance(group_data, list):
                n = len(group_data)
            else:
                continue
            min_n = min(min_n, n)

        if min_n < float('inf'):
            stat_checks["sample_adequacy"]["min_group_n"] = min_n
            if min_n < min_err:
                issues.append({
                    "severity": "error",
                    "category": "sample_size",
                    "message": f"最小组样本量n={min_n}，结果可能不可靠"
                })
                stat_checks["sample_adequacy"]["adequate"] = False
                score -= 20
            elif min_n < min_warn:
                issues.append({
                    "severity": "warning",
                    "category": "sample_size",
                    "message": f"最小组样本量n={min_n}，统计效能可能不足"
                })
                score -= 10

        # === 5. 结果一致性（工具成功率） ===
        total = len(tool_results)
        failed = sum(1 for r in tool_results
                     if r.get("status") in ("failed", "skipped"))
        success_rate = (total - failed) / max(total, 1)

        rate_rejected = thresholds.get("success_rate_rejected", 0.3)
        rate_warning = thresholds.get("success_rate_warning", 0.7)

        if success_rate < rate_rejected:
            status = "rejected"
            score -= 30
        elif success_rate < rate_warning:
            status = "approved_with_warnings"
            score -= 10
        else:
            status = "approved"

        # === 6. 汇总工具输出规范检查 ===
        spec_failures = []
        for result in tool_results:
            spec_check = result.get("output_spec_check", {})
            if not spec_check.get("passed", True):
                spec_failures.append({
                    "tool": spec_check.get("tool"),
                    "operation": spec_check.get("operation"),
                    "missing_files": spec_check.get("missing_files", []),
                    "quality_issues": spec_check.get("quality_issues", [])
                })
        if spec_failures:
            issues.append({
                "severity": "warning",
                "category": "output_spec",
                "message": f"{len(spec_failures)}个工具输出未通过规范检查",
                "details": spec_failures
            })
            score -= len(spec_failures) * 3

        # === 7. LLM统计审查（可选） ===
        if self.llm:
            try:
                llm_stat = self._llm_stat_review(tool_results, cohort, plan)
                if llm_stat.get("issues"):
                    issues.extend(llm_stat["issues"])
                if llm_stat.get("score"):
                    score = int(score * 0.6 + llm_stat["score"] * 0.4)
            except Exception as e:
                print(f"    [StatReviewer] LLM审查失败: {e}")

        score = max(0, min(100, score))
        error_count = sum(1 for i in issues if i.get("severity") == "error")
        if error_count > 0 and status == "approved":
            status = "approved_with_warnings"

        result = {
            "reviewer": "StatReviewer",
            "status": status,
            "score": score,
            "issues": issues,
            "stat_checks": stat_checks,
            "spec_failures": spec_failures,
            "timestamp": datetime.now().isoformat()
        }

        print(f"    状态: {status}, 评分: {score}/100, "
              f"问题: {len(issues)}")
        return result

    def _llm_stat_review(self, tool_results: List[Dict],
                         cohort: Dict, plan: Dict) -> Dict:
        """使用LLM进行统计审查 — 提供实际统计数据和标准"""
        prompt = MOER_PROMPTS["stat_reviewer"]

        # 提取实际统计证据（而非仅元数据）
        stat_evidence = _extract_statistical_evidence(tool_results, max_items=15)

        # 注入神经影像统计标准
        neuro = self.criteria.get("neuroimaging_criteria", {})
        stat_thresh = neuro.get("statistical_thresholds", {})
        stat_weights = self.criteria.get("stat_review", {}).get("weights", {})

        standards_block = f"""
## 统计审查标准（必须依据）

### 神经影像统计阈值
- Cluster-forming: p < {stat_thresh.get('cluster_forming_p', 0.001)} (Woo et al. 2014)
- FWE cluster: p < {stat_thresh.get('fwe_cluster_p', 0.05)}
- FDR: q < {stat_thresh.get('fdr_q', 0.05)}
- TFCE: {stat_thresh.get('tfce_note', '不需要cluster-forming threshold')}

### 评分权重
多重比较: {stat_weights.get('multiple_comparison', 25)}%, 效应量: {stat_weights.get('effect_sizes', 20)}%, 假设检验: {stat_weights.get('assumptions_tested', 25)}%, 样本充分性: {stat_weights.get('sample_adequacy', 15)}%, 结果一致性: {stat_weights.get('result_consistency', 15)}%

### 关键判定规则
- 未校正p值报告为显著 → error
- 多个检验无多重比较校正 → warning
- 缺少效应量 → warning
- 每组n<5 → error, n<10 → warning
- circular analysis (Kriegeskorte et al. 2009) → error
"""

        # 构建组样本量摘要
        groups = cohort.get("groups", {})
        group_summary = {}
        for gname, gdata in groups.items():
            if isinstance(gdata, dict):
                group_summary[gname] = {
                    "n": gdata.get("n", len(gdata.get("subjects", []))),
                }
            elif isinstance(gdata, list):
                group_summary[gname] = {"n": len(gdata)}

        messages = [
            {"role": "system", "content": prompt + "\n" + standards_block},
            {"role": "user", "content": f"""请审查以下统计分析结果：

## 队列信息
总样本: {cohort.get('total_subjects', 'N/A')}
分组: {json.dumps(group_summary, ensure_ascii=False)}

## 统计分析结果（实际数据）
{json.dumps(stat_evidence, ensure_ascii=False, default=str)[:4000]}

## 研究计划统计方法
{json.dumps(plan.get('statistics', plan.get('statistical_methods', {})), ensure_ascii=False)[:1000]}

请严格依据上述标准审查实际统计数据，以JSON格式返回审查结果。"""}
        ]

        response = self.llm.generate_json(messages)
        return _validate_review_schema(response, "stat")

    # --- ReportReviewer (委托) ---
    def review_report(self, report: str, cohort: Dict,
                      tool_results: List[Dict], plan: Dict,
                      run_dir: str = None) -> Dict:
        """
        报告质量审查 - 委托给 ReportAuditor
        """
        print("  [MoER:ReportReviewer] 审查报告...")
        try:
            from src.agent.report_auditor import ReportAuditor
            auditor = ReportAuditor(Path(run_dir) if run_dir else None)
            audit = auditor.audit_report(
                report, cohort, tool_results, plan, self.llm)
            return {
                "reviewer": "ReportReviewer",
                "status": "approved" if audit.get("status") == "PASSED" else "needs_revision",
                "score": audit.get("credibility_score", 70),
                "issues": audit.get("issues", []),
                "warnings": audit.get("warnings", []),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"    [ReportReviewer] 审查失败: {e}")
            return {
                "reviewer": "ReportReviewer",
                "status": "error",
                "score": 0,
                "issues": [{"severity": "error", "category": "review_failed",
                            "message": f"报告审查异常: {e}"}],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# ============== 工具输出规范验证 ==============

def _run_quality_check(qc: Dict, tool_result: Dict,
                       output_dir: str = None) -> Optional[Dict]:
    """执行单个质量检查"""
    check_type = qc.get("check", "")
    params = qc.get("params", {})
    field = qc.get("field", "")
    result_data = tool_result.get("outputs", tool_result.get("result", {}))
    if not isinstance(result_data, dict):
        result_data = {}

    # --- 基础文件检查 ---
    if check_type == "file_readable":
        return None  # 由文件存在性检查覆盖

    # --- 数值范围检查 ---
    elif check_type == "value_range":
        min_val = params.get("min")
        max_val = params.get("max")
        if field:
            val = result_data.get(field)
            if val is not None:
                try:
                    val = float(val)
                    if min_val is not None and val < min_val:
                        return {"check": check_type, "field": field,
                                "message": f"{field}值{val}低于最小值{min_val}"}
                    if max_val is not None and val > max_val:
                        return {"check": check_type, "field": field,
                                "message": f"{field}值{val}超过最大值{max_val}"}
                except (ValueError, TypeError):
                    pass
        return None

    # --- FWHM 范围检查 ---
    elif check_type == "fwhm_in_range":
        fwhm = tool_result.get("params", {}).get("fwhm")
        if fwhm is not None:
            try:
                fwhm_val = fwhm[0] if isinstance(fwhm, list) else float(fwhm)
                min_f = params.get("min", 4)
                max_f = params.get("max", 12)
                if fwhm_val < min_f or fwhm_val > max_f:
                    return {"check": check_type,
                            "message": f"FWHM={fwhm_val}不在推荐范围[{min_f},{max_f}]mm"}
            except (ValueError, TypeError):
                pass
        return None

    # --- 正值检查 ---
    elif check_type == "value_positive":
        if field:
            val = result_data.get(field)
            if val is not None:
                try:
                    if float(val) <= 0:
                        return {"check": check_type, "field": field,
                                "message": f"{field}值{val}应为正数"}
                except (ValueError, TypeError):
                    pass
        return None

    # --- 掩模非零体素占比 ---
    elif check_type == "mask_nonzero_ratio":
        ratio = result_data.get("nonzero_ratio", result_data.get("mask_ratio"))
        if ratio is not None:
            try:
                ratio = float(ratio)
                min_r = params.get("min", 0.1)
                max_r = params.get("max", 0.8)
                if ratio < min_r or ratio > max_r:
                    return {"check": check_type,
                            "message": f"掩模非零占比{ratio:.3f}不在合理范围[{min_r},{max_r}]"}
            except (ValueError, TypeError):
                pass
        return None

    # --- 运动参数阈值 ---
    elif check_type == "motion_threshold":
        motion_fields = {
            "max_mm": ("max_motion", "max_displacement", "max_abs_motion"),
            "mean_rms_warn": ("mean_rms", "mean_motion", "mean_fd"),
            "max_translation_mm": ("max_translation",),
            "max_rotation_deg": ("max_rotation",),
        }
        issues = []
        for param_key, result_keys in motion_fields.items():
            threshold = params.get(param_key)
            if threshold is None:
                continue
            for rk in result_keys:
                val = result_data.get(rk)
                if val is not None:
                    try:
                        if float(val) > float(threshold):
                            issues.append(f"{rk}={val}超过阈值{threshold}")
                    except (ValueError, TypeError):
                        pass
                    break
        # 兼容简单 max_mm 模式
        if not issues and "max_mm" in params:
            for key in ("mean_motion", "max_motion", "fd_mean"):
                val = result_data.get(key)
                if val is not None:
                    try:
                        if float(val) > float(params["max_mm"]):
                            issues.append(f"{key}={val}超过阈值{params['max_mm']}mm")
                    except (ValueError, TypeError):
                        pass
        if issues:
            return {"check": check_type, "message": "; ".join(issues)}
        return None

    # --- 异常比例 ---
    elif check_type == "outlier_ratio":
        max_ratio = params.get("max_ratio", 0.1)
        ratio = result_data.get("outlier_ratio", result_data.get("outlier_percentage"))
        if ratio is not None:
            try:
                ratio = float(ratio)
                # 如果是百分比形式，转换
                if ratio > 1:
                    ratio = ratio / 100.0
                if ratio > max_ratio:
                    return {"check": check_type,
                            "message": f"异常比例{ratio:.3f}超过阈值{max_ratio}"}
            except (ValueError, TypeError):
                pass
        return None

    # --- p值/r值/R²有效性（委托给 value_range） ---
    elif check_type == "p_value_valid":
        return _run_quality_check(
            {"check": "value_range", "field": field or "p_value",
             "params": {"min": 0, "max": 1}}, tool_result, output_dir)

    elif check_type == "r_value_valid":
        return _run_quality_check(
            {"check": "value_range", "field": field or "r_value",
             "params": {"min": -1, "max": 1}}, tool_result, output_dir)

    elif check_type == "r_squared_valid":
        return _run_quality_check(
            {"check": "value_range", "field": field or "r_squared",
             "params": {"min": 0, "max": 1}}, tool_result, output_dir)

    # --- 样本量充分性 ---
    elif check_type == "sample_size_adequate":
        min_n = params.get("min_per_group", params.get("min_n", 5))
        for key in ("n", "sample_size", "n_per_group", "min_group_n"):
            val = result_data.get(key)
            if val is not None:
                try:
                    if int(val) < min_n:
                        return {"check": check_type,
                                "message": f"样本量{val}低于最小要求{min_n}"}
                except (ValueError, TypeError):
                    pass
        return None

    # --- 文件最小大小 ---
    elif check_type == "file_size_min":
        min_bytes = params.get("min_bytes", 1024)
        if output_dir:
            out_path = Path(output_dir)
            pattern = qc.get("file_pattern", "*")
            for f in out_path.rglob(pattern):
                if f.is_file() and f.stat().st_size < min_bytes:
                    return {"check": check_type,
                            "message": f"文件{f.name}大小{f.stat().st_size}B低于最小值{min_bytes}B"}
        return None

    # --- bvec/bval 匹配 ---
    elif check_type == "bvec_bval_match":
        n_volumes = result_data.get("n_volumes", result_data.get("dwi_volumes"))
        n_bvals = result_data.get("n_bvals")
        n_bvecs = result_data.get("n_bvecs")
        if n_volumes is not None and n_bvals is not None:
            try:
                if int(n_volumes) != int(n_bvals):
                    return {"check": check_type,
                            "message": f"DWI体积数({n_volumes})与bval数({n_bvals})不匹配"}
            except (ValueError, TypeError):
                pass
        if n_volumes is not None and n_bvecs is not None:
            try:
                if int(n_volumes) != int(n_bvecs):
                    return {"check": check_type,
                            "message": f"DWI体积数({n_volumes})与bvec方向数({n_bvecs})不匹配"}
            except (ValueError, TypeError):
                pass
        return None

    # --- 维度一致性 ---
    elif check_type == "dimensions_preserved":
        input_dims = result_data.get("input_dimensions", result_data.get("input_shape"))
        output_dims = result_data.get("output_dimensions", result_data.get("output_shape"))
        if input_dims is not None and output_dims is not None:
            if str(input_dims) != str(output_dims):
                return {"check": check_type,
                        "message": f"输入维度{input_dims}与输出维度{output_dims}不一致"}
        return None

    # --- 表格非空 ---
    elif check_type == "table_not_empty":
        for key in ("row_count", "n_rows", "table_rows"):
            val = result_data.get(key)
            if val is not None:
                try:
                    if int(val) == 0:
                        return {"check": check_type,
                                "message": "输出表格为空（0行数据）"}
                except (ValueError, TypeError):
                    pass
        return None

    # --- tSNR 检查 ---
    elif check_type == "tsnr_check":
        tsnr = result_data.get("tsnr", result_data.get("tSNR"))
        if tsnr is not None:
            try:
                tsnr_val = float(tsnr)
                min_tsnr = params.get("min", 40)
                if tsnr_val < min_tsnr:
                    return {"check": check_type,
                            "message": f"tSNR={tsnr_val:.1f}低于阈值{min_tsnr}"}
            except (ValueError, TypeError):
                pass
        return None

    # --- 设计矩阵秩检查 ---
    elif check_type == "design_matrix_rank":
        rank = result_data.get("design_matrix_rank")
        n_regressors = result_data.get("n_regressors", result_data.get("n_columns"))
        if rank is not None and n_regressors is not None:
            try:
                if int(rank) < int(n_regressors):
                    return {"check": check_type,
                            "message": f"设计矩阵秩({rank})小于回归量数({n_regressors})，矩阵不满秩"}
            except (ValueError, TypeError):
                pass
        return None

    # --- acceptance_criteria 条件检查 ---
    elif check_type == "acceptance_check":
        # 由 validate_tool_output 的第4步统一处理
        return None

    # --- 复杂检查（需要 nibabel 等依赖）→ 返回建议 ---
    else:
        desc = qc.get("description", check_type)
        return {"check": check_type, "status": "advisory",
                "message": f"建议手动检查: {desc}"}


def _check_acceptance_criteria(spec: Dict, tool_result: Dict) -> Dict:
    """
    根据 acceptance_criteria 判定工具输出是否合格

    返回:
        {
            "status": "pass" | "warning" | "fail",
            "failed_criteria": [...],
            "warnings": [...],
            "passed_criteria": [...]
        }
    """
    criteria = spec.get("acceptance_criteria", {})
    if not criteria:
        return {"status": "pass", "failed_criteria": [],
                "warnings": [], "passed_criteria": []}

    result_data = tool_result.get("outputs", tool_result.get("result", {}))
    if not isinstance(result_data, dict):
        result_data = {}

    failed = []
    warnings = []
    passed = []

    # --- fail_conditions: 任一触发则 status=fail ---
    for cond in criteria.get("fail_conditions", []):
        outcome = _evaluate_condition(cond, result_data)
        if outcome == "triggered":
            failed.append({
                "condition": cond,
                "message": cond.get("message", f"fail条件触发: {cond.get('field', cond.get('check', ''))}")
            })
        elif outcome == "not_triggered":
            passed.append(cond)
        # outcome == "skipped" → 字段不存在，跳过

    # --- pass_conditions: 未满足时降级为 warning（不直接 fail） ---
    for cond in criteria.get("pass_conditions", []):
        outcome = _evaluate_condition(cond, result_data)
        if outcome == "triggered":
            # pass_condition 未满足 → 记为 warning（只有 fail_conditions 才能导致 status=fail）
            field_name = cond.get('field', '')
            expected = cond.get('expected_range', cond.get('operator', ''))
            warnings.append({
                "condition": cond,
                "message": cond.get("note", f"pass条件未满足: {field_name} (期望: {expected})")
            })
        elif outcome == "not_triggered":
            passed.append(cond)

    # --- warning_conditions ---
    for cond in criteria.get("warning_conditions", []):
        outcome = _evaluate_condition(cond, result_data)
        if outcome == "triggered":
            warnings.append({
                "condition": cond,
                "message": cond.get("message", f"warning: {cond.get('field', '')}")
            })

    if failed:
        status = "fail"
    elif warnings:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "failed_criteria": failed,
        "warnings": warnings,
        "passed_criteria": [{"field": c.get("field", c.get("check", ""))} for c in passed]
    }


def _evaluate_condition(cond: Dict, result_data: Dict) -> str:
    """
    评估单个 acceptance 条件

    返回:
        "triggered" - 条件被触发（对fail/warning=坏事发生，对pass=不满足）
        "not_triggered" - 条件未触发
        "skipped" - 字段不存在，无法评估
    """
    field = cond.get("field", "")

    # --- 基于 range 的检查 ---
    # 注意：range 和 range_outside 的逻辑完全等价（都在值超出范围时触发）。
    # 语义区分由 _check_acceptance_criteria 层处理：
    #   - pass_conditions 中的 range: triggered → warning（条件未满足）
    #   - fail_conditions 中的 range_outside: triggered → fail（违规条件触发）
    if "range" in cond and field:
        val = result_data.get(field)
        if val is None:
            return "skipped"
        try:
            val = float(val)
            r = cond["range"]
            if val < r[0] or val > r[1]:
                return "triggered"  # 超出范围
            return "not_triggered"
        except (ValueError, TypeError, IndexError):
            return "skipped"

    # --- 基于 range_outside 的检查（fail条件常用） ---
    if "range_outside" in cond and field:
        val = result_data.get(field)
        if val is None:
            return "skipped"
        try:
            val = float(val)
            r = cond["range_outside"]
            if val < r[0] or val > r[1]:
                return "triggered"  # 在范围外 → 触发
            return "not_triggered"
        except (ValueError, TypeError, IndexError):
            return "skipped"

    # --- 基于 threshold + operator 的检查 ---
    if "threshold" in cond and "operator" in cond and field:
        val = result_data.get(field)
        if val is None:
            return "skipped"
        try:
            val = float(val)
            threshold = float(cond["threshold"])
            op = cond["operator"]
            if op == ">" and val > threshold:
                return "triggered"
            elif op == "<" and val < threshold:
                return "triggered"
            elif op == ">=" and val >= threshold:
                return "triggered"
            elif op == "<=" and val <= threshold:
                return "triggered"
            return "not_triggered"
        except (ValueError, TypeError):
            return "skipped"

    # --- 基于 expected 的精确匹配 ---
    if "expected" in cond and field:
        val = result_data.get(field)
        if val is None:
            return "skipped"
        if val != cond["expected"]:
            return "triggered"
        return "not_triggered"

    # --- 基于 check 类型的检查（文件存在性等） ---
    check = cond.get("check", "")
    if check in ("required_files_missing", "required_files_exist",
                  "output_file_exists", "output_file_missing",
                  "file_empty", "table_empty", "matrix_empty",
                  "bvec_bval_match", "bvec_bval_mismatch",
                  "dimensions_preserved", "table_not_empty",
                  "file_size_min", "seed_roi_in_mask",
                  "feat_dir_incomplete", "full_rank",
                  "fwhm_voxel_ratio", "ad_le_rd",
                  "post_hoc_missing", "not_reported"):
        # 这些检查需要更多上下文（文件系统、完整数据），
        # 在 _run_quality_check 中已有部分覆盖，此处标记为跳过
        return "skipped"

    return "skipped"


def validate_tool_output(tool_name: str, operation: str,
                         tool_result: Dict,
                         output_dir: str = None) -> Dict:
    """
    根据标准规范验证工具输出
    仅记录模式：返回检查结果但不影响工具执行状态
    """
    spec = load_tool_standard(tool_name, operation)
    if not spec:
        return {"passed": True,
                "note": f"无{tool_name}/{operation}的标准规范",
                "tool": tool_name, "operation": operation}

    missing_fields = []
    missing_files = []
    quality_issues = []

    # 1. 检查必要输出字段
    result_data = tool_result.get("outputs", tool_result.get("result", {}))
    if isinstance(result_data, dict):
        for field in spec.get("expected_output_fields", []):
            if field not in result_data:
                missing_fields.append(field)

    # 2. 检查预期输出文件
    if output_dir:
        out_path = Path(output_dir)
        if out_path.exists():
            for file_spec in spec.get("expected_output_files", []):
                pattern = file_spec["pattern"]
                required = file_spec.get("required", False)
                matched = list(out_path.rglob(pattern))
                if not matched and required:
                    missing_files.append({
                        "pattern": pattern,
                        "description": file_spec.get("description", "")
                    })

    # 3. 执行质量检查
    for qc in spec.get("quality_checks", []):
        issue = _run_quality_check(qc, tool_result, output_dir)
        if issue:
            quality_issues.append(issue)

    # 4. 检查 acceptance_criteria
    acceptance_result = _check_acceptance_criteria(spec, tool_result)

    passed = len(missing_files) == 0 and acceptance_result["status"] != "fail"
    return {
        "passed": passed,
        "tool": tool_name,
        "operation": operation,
        "missing_fields": missing_fields,
        "missing_files": missing_files,
        "quality_issues": quality_issues,
        "acceptance_result": acceptance_result
    }
