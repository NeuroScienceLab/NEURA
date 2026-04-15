"""
Report Review Module - Preventing AI Hallucinations, Ensuring Scientific Rigor

Review Content:
1. Sample Size Verification: Whether the number of samples in the report is consistent with the cohort
2. File Path Verification: Whether the files referenced in the report exist
3. Statistical Data Verification: Whether the figures in the report are supported by tool outputs
4. Content Consistency Check: Using LLM to detect potential hallucination content
"""
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class ReportAuditor:
    """报告审查器"""

    def __init__(self, run_dir: Path = None):
        self.run_dir = run_dir
        self.issues = []  # 发现的问题
        self.warnings = []  # 警告
        self.verified_facts = []  # 已验证的事实

    def audit_report(
        self,
        report_content: str,
        cohort: Dict,
        tool_results: List[Dict],
        plan: Dict,
        llm_client=None,
        moer_reviews: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        执行完整的报告审查

        Args:
            report_content: 报告内容
            cohort: 队列信息
            tool_results: 工具执行结果
            plan: 研究计划
            llm_client: LLM客户端（用于内容审查）
            moer_reviews: MoER历史审查记录（来自state["moer_reviews"]）

        Returns:
            审查结果
        """
        self.issues = []
        self.warnings = []
        self.verified_facts = []

        print("\n[Report Audit] 开始报告审查...")

        # 1. 验证样本数量
        self._verify_sample_counts(report_content, cohort)

        # 2. 验证文件引用
        self._verify_file_references(report_content)

        # 3. 验证统计数据
        self._verify_statistics(report_content, tool_results)

        # 4. 检测可能的幻觉内容
        self._detect_hallucinations(report_content, cohort, tool_results, plan)

        # 5. 验证数据来源（防止幻觉）
        self._verify_data_sources(report_content, cohort, tool_results, plan)

        # 6. LLM辅助审查（如果提供了客户端）
        if llm_client:
            self._llm_content_audit(report_content, cohort, tool_results, llm_client)

        # 7. IMRAD结构合规检查
        self._verify_imrad_structure(report_content)

        # 8. 声明-证据对齐检查
        if llm_client:
            self._verify_claim_evidence_alignment(report_content, tool_results, llm_client)

        # 9. 聚合MoER历史审查结果
        if moer_reviews:
            self._aggregate_moer_reviews(moer_reviews)

        # 生成审查报告
        audit_result = self._generate_audit_report()

        return audit_result

    def _verify_sample_counts(self, report: str, cohort: Dict):
        """验证样本数量"""
        print("  [1/5] 验证样本数量...")

        if not cohort:
            self.warnings.append({
                "type": "missing_cohort",
                "message": "无法验证样本数量：cohort数据为空"
            })
            return

        # 从cohort获取真实数据
        total_subjects = cohort.get("total_subjects", 0)
        groups = cohort.get("groups", {})

        # 提取报告中的数字
        numbers_in_report = re.findall(r'\b(\d+)\s*(?:名|例|个|位|人|subjects?|participants?|patients?|controls?)\b', report, re.IGNORECASE)

        # 检查总样本数是否匹配
        if str(total_subjects) in report:
            self.verified_facts.append({
                "type": "sample_count",
                "message": f"总样本数 {total_subjects} 验证通过"
            })
        else:
            # 检查是否有其他数字可能是样本数
            for num in numbers_in_report:
                if int(num) == total_subjects:
                    self.verified_facts.append({
                        "type": "sample_count",
                        "message": f"总样本数 {total_subjects} 验证通过"
                    })
                    break

        # 检查各组样本数
        for group_name, group_data in groups.items():
            expected_n = group_data.get("n", 0)
            if str(expected_n) not in report and expected_n > 0:
                # 不是严重问题，只是警告
                pass
            else:
                self.verified_facts.append({
                    "type": "group_count",
                    "message": f"组 {group_name} 样本数 {expected_n} 验证通过"
                })

    def _verify_file_references(self, report: str):
        """验证文件引用"""
        print("  [2/5] 验证文件引用...")

        # 匹配可能的文件路径
        path_patterns = [
            r'(?:path|路径|文件)[:\s]*([A-Za-z]:\\[^\s\n]+)',  # Windows路径
            r'(?:path|路径|文件)[:\s]*(/[^\s\n]+)',  # Unix路径
            r'`([A-Za-z]:\\[^`]+)`',  # 代码块中的Windows路径
            r'`(/[^`]+)`',  # 代码块中的Unix路径
        ]

        referenced_files = []
        for pattern in path_patterns:
            matches = re.findall(pattern, report, re.IGNORECASE)
            referenced_files.extend(matches)

        # 验证文件是否存在
        for file_path in referenced_files:
            path = Path(file_path.strip())
            if path.exists():
                self.verified_facts.append({
                    "type": "file_exists",
                    "message": f"文件存在: {file_path}"
                })
            else:
                # 检查是否是相对于run_dir的路径
                if self.run_dir:
                    relative_path = self.run_dir / file_path
                    if relative_path.exists():
                        self.verified_facts.append({
                            "type": "file_exists",
                            "message": f"文件存在（相对路径）: {file_path}"
                        })
                        continue

                self.issues.append({
                    "type": "file_not_found",
                    "severity": "warning",
                    "message": f"报告引用的文件不存在: {file_path}"
                })

    def _verify_statistics(self, report: str, tool_results: List[Dict]):
        """验证统计数据"""
        print("  [3/5] 验证统计数据...")

        if not tool_results:
            self.warnings.append({
                "type": "no_tool_results",
                "message": "无法验证统计数据：工具结果为空"
            })
            return

        # 从工具结果中提取数值
        tool_numbers = set()
        for result in tool_results:
            self._extract_numbers_from_dict(result, tool_numbers)

        # 提取报告中的统计数值（p值、t值、F值等）
        stat_patterns = [
            (r'[pP]\s*[=<>]\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)', "p值"),
            (r'[tT]\s*[=<>]\s*(-?\d+\.?\d*)', "t值"),
            (r'[fF]\s*[=<>]\s*(\d+\.?\d*)', "F值"),
            (r'[rR]\s*[=<>]\s*(-?\d+\.?\d*)', "相关系数"),
            (r'(?:mean|均值|平均)[:\s]*(-?\d+\.?\d*)', "均值"),
            (r'(?:SD|std|标准差)[:\s]*(\d+\.?\d*)', "标准差"),
        ]

        for pattern, stat_type in stat_patterns:
            matches = re.findall(pattern, report)
            for match in matches:
                try:
                    value = float(match)
                    # 检查这个值是否在工具结果中
                    value_found = False
                    for tool_num in tool_numbers:
                        try:
                            if abs(float(tool_num) - value) < 0.001:  # 允许小误差
                                value_found = True
                                break
                        except:
                            continue

                    if value_found:
                        self.verified_facts.append({
                            "type": "statistic_verified",
                            "message": f"{stat_type} {value} 在工具输出中找到对应值"
                        })
                except:
                    pass

    def _extract_numbers_from_dict(self, d: Any, numbers: set, depth: int = 0):
        """递归提取字典中的数值"""
        if depth > 10:  # 防止过深递归
            return

        if isinstance(d, dict):
            for v in d.values():
                self._extract_numbers_from_dict(v, numbers, depth + 1)
        elif isinstance(d, list):
            for item in d[:100]:  # 限制列表长度
                self._extract_numbers_from_dict(item, numbers, depth + 1)
        elif isinstance(d, (int, float)):
            numbers.add(str(d))
        elif isinstance(d, str):
            # 尝试从字符串中提取数字
            found = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', d)
            for f in found[:10]:
                numbers.add(f)

    def _detect_hallucinations(self, report: str, cohort: Dict, tool_results: List[Dict], plan: Dict):
        """检测可能的幻觉内容"""
        print("  [4/5] 检测潜在幻觉...")

        # 检查是否有未在cohort中的组名
        if cohort and "groups" in cohort:
            valid_groups = set(cohort["groups"].keys())
            # 常见的组名模式
            group_patterns = [
                r'(?:组|group|Group)\s*[：:]\s*(\w+)',
                r'(\w+)\s*(?:组|group)',
                r'(?:患者|病人|对照)\s*[（(](\w+)[）)]'
            ]
            for pattern in group_patterns:
                matches = re.findall(pattern, report)
                for match in matches:
                    if match.upper() not in [g.upper() for g in valid_groups]:
                        # 可能是幻觉的组名
                        pass  # 暂时不报告，因为可能是描述性文字

        # 检查是否有明显编造的数据特征
        hallucination_patterns = [
            # 过于精确的p值可能是编造的
            (r'[pP]\s*=\s*0\.0{4,}\d+', "警告：过于精确的p值可能需要验证"),
            # 异常的样本数
            (r'[nN]\s*=\s*(\d+)', None),  # 需要与cohort对比
        ]

        for pattern, warning in hallucination_patterns:
            if warning:
                matches = re.findall(pattern, report)
                if matches:
                    self.warnings.append({
                        "type": "potential_hallucination",
                        "message": warning,
                        "matches": matches[:3]
                    })

        # 检查样本数是否异常
        if cohort:
            total_n = cohort.get("total_subjects", 0)
            # 查找报告中的n=数字
            n_values = re.findall(r'[nN]\s*=\s*(\d+)', report)
            for n in n_values:
                n_int = int(n)
                if n_int > total_n * 2:  # 如果报告中的n远大于实际样本
                    self.issues.append({
                        "type": "sample_count_mismatch",
                        "severity": "error",
                        "message": f"报告中的样本数 n={n_int} 远超实际样本数 {total_n}，可能是幻觉"
                    })

    def _verify_data_sources(self, report: str, cohort: Dict, tool_results: List[Dict], plan: Dict):
        """
        验证数据来源 - 确保报告中提到的数据确实存在于处理流程中

        关键检查：
        1. 灰质/白质数据 → 必须有分割处理步骤
        2. 量表数据 → 必须有对应输入数据
        3. 特定分析结果 → 必须有对应工具执行
        """
        print("  [5/6] 验证数据来源...")

        # 提取已执行的工具/步骤名称和完整信息
        executed_tools = set()
        tool_outputs = {}
        tool_full_info = []  # 保存完整的工具信息用于深度检查

        if tool_results:
            for result in tool_results:
                # 提取工具名称
                tool_name = result.get("tool", result.get("name", ""))
                if tool_name:
                    executed_tools.add(tool_name.lower())

                # 提取任务ID和描述
                task_id = result.get("task_id", result.get("id", ""))
                if task_id:
                    executed_tools.add(task_id.lower())

                # 记录工具输出
                output = str(result.get("result", result.get("output", "")))
                tool_outputs[tool_name] = output

                # 保存完整信息（转为字符串以便搜索）
                tool_full_info.append(str(result).lower())

        # 从计划中提取预期步骤
        planned_steps = set()
        plan_full_text = ""
        if plan:
            plan_full_text = str(plan).lower()
            steps = plan.get("steps", plan.get("analysis_steps", plan.get("tasks", [])))
            for step in steps:
                if isinstance(step, dict):
                    step_name = step.get("name", step.get("step", step.get("description", ""))) or ""
                    step_tool = step.get("tool", "") or ""
                    if step_name:
                        planned_steps.add(step_name.lower())
                    if step_tool:
                        planned_steps.add(step_tool.lower())
                elif isinstance(step, str) and step:
                    planned_steps.add(step.lower())

        # 合并所有已知步骤
        all_known_steps = executed_tools | planned_steps
        # 将所有工具信息合并为一个大字符串，用于关键词搜索
        all_tool_info_text = " ".join(tool_full_info) + " " + plan_full_text

        # ===== 1. 灰质数据验证 =====
        gray_matter_patterns = [
            r'灰质', r'gray\s*matter', r'\bGM\b', r'灰质体积', r'gray\s*matter\s*volume',
            r'皮层厚度', r'cortical\s*thickness', r'灰质密度', r'gray\s*matter\s*density'
        ]
        gray_matter_mentioned = any(
            re.search(p, report, re.IGNORECASE) for p in gray_matter_patterns
        )

        if gray_matter_mentioned:
            # 扩展分割关键词列表，包括SPM相关的各种表述
            segmentation_keywords = [
                'segment', '分割', 'segmentation', 'cat12', 'freesurfer',
                'spm_segment', 'vbm', 'tissue', 'gm_volume', 'spm_analysis',
                'spm', 'c1', 'c2', 'c3', 'wc1', 'wc2', 'mwc1', 'normalize',
                'smoothing', 'smooth', 'dartel', 'unified_segmentation'
            ]

            # 检查步骤名称
            has_segmentation = any(
                kw in step for kw in segmentation_keywords for step in all_known_steps
            )

            # 检查工具输出中是否有灰质相关内容
            has_gm_output = any(
                re.search(r'gray|gm|灰质|segment|c1|c2|wc1|tissue', output, re.IGNORECASE)
                for output in tool_outputs.values()
            )

            # 检查完整工具信息中是否有分割相关内容
            has_spm_in_info = any(
                kw in all_tool_info_text for kw in segmentation_keywords
            )

            if has_segmentation or has_gm_output or has_spm_in_info:
                self.verified_facts.append({
                    "type": "data_source_verified",
                    "message": "灰质数据来源验证通过：检测到分割处理步骤"
                })
            else:
                # 降级为警告而非错误，因为可能存在检测遗漏
                self.warnings.append({
                    "type": "data_source_unverified",
                    "message": "报告提到灰质数据，未能自动确认灰质分割步骤（建议人工核实）"
                })

        # ===== 2. 白质数据验证 =====
        white_matter_patterns = [
            r'白质', r'white\s*matter', r'\bWM\b', r'白质体积', r'white\s*matter\s*volume',
            r'髓鞘', r'myelin', r'白质纤维', r'white\s*matter\s*tract'
        ]
        white_matter_mentioned = any(
            re.search(p, report, re.IGNORECASE) for p in white_matter_patterns
        )

        if white_matter_mentioned:
            # 扩展分割关键词列表
            segmentation_keywords = [
                'segment', '分割', 'segmentation', 'cat12', 'freesurfer',
                'spm_segment', 'wm_volume', 'dti', 'diffusion', 'spm_analysis',
                'spm', 'c1', 'c2', 'c3', 'wc1', 'wc2', 'mwc2', 'normalize',
                'smoothing', 'smooth', 'dartel', 'tissue'
            ]

            has_segmentation = any(
                kw in step for kw in segmentation_keywords for step in all_known_steps
            )

            has_wm_output = any(
                re.search(r'white|wm|白质|segment|c2|wc2|tissue', output, re.IGNORECASE)
                for output in tool_outputs.values()
            )

            # 检查完整工具信息中是否有分割相关内容
            has_spm_in_info = any(
                kw in all_tool_info_text for kw in segmentation_keywords
            )

            if has_segmentation or has_wm_output or has_spm_in_info:
                self.verified_facts.append({
                    "type": "data_source_verified",
                    "message": "白质数据来源验证通过：检测到分割处理步骤"
                })
            else:
                # 降级为警告而非错误
                self.warnings.append({
                    "type": "data_source_unverified",
                    "message": "报告提到白质数据，未能自动确认白质分割步骤（建议人工核实）"
                })

        # ===== 3. 量表数据验证 =====
        scale_patterns = {
            'SARA': (r'\bSARA\b', '小脑共济失调评分'),
            'ICARS': (r'\bICARS\b', '国际共济失调评定量表'),
            'CAG': (r'\bCAG\b', 'CAG重复数'),
            'MMSE': (r'\bMMSE\b', '简易精神状态检查'),
            'MoCA': (r'\bMoCA\b', '蒙特利尔认知评估'),
            'BDI': (r'\bBDI\b', '贝克抑郁量表'),
            'HAMD': (r'\bHAMD\b', '汉密尔顿抑郁量表'),
            'ADL': (r'\bADL\b', '日常生活活动能力量表'),
            'EDSS': (r'\bEDSS\b', '扩展残疾状态量表'),
        }

        # 获取cohort中的实际变量
        cohort_variables = set()
        if cohort:
            # 从cohort中提取所有变量名
            for key in cohort.keys():
                cohort_variables.add(key.upper())
            # 检查groups中的数据
            groups = cohort.get("groups", {})
            for group_data in groups.values():
                if isinstance(group_data, dict):
                    for var_name in group_data.keys():
                        cohort_variables.add(var_name.upper())
            # 检查subjects中的数据
            subjects = cohort.get("subjects", [])
            if subjects and len(subjects) > 0:
                for subj in subjects[:1]:  # 只检查第一个subject的变量名
                    if isinstance(subj, dict):
                        for var_name in subj.keys():
                            cohort_variables.add(var_name.upper())

            # 【修复】检查demographics中的列名（来自data.xlsx）
            demographics = cohort.get("demographics", {})
            if isinstance(demographics, dict):
                columns = demographics.get("columns", [])
                for col in columns:
                    cohort_variables.add(col.upper())
                # 也检查 numeric_columns
                numeric_cols = demographics.get("numeric_columns", [])
                for col in numeric_cols:
                    cohort_variables.add(col.upper())

        for scale_name, (pattern, desc) in scale_patterns.items():
            if re.search(pattern, report, re.IGNORECASE):
                # 检查cohort中是否有这个量表数据
                has_scale_data = any(
                    scale_name.upper() in var.upper() for var in cohort_variables
                )

                # 也检查工具输出中是否有相关内容
                has_scale_output = any(
                    re.search(pattern, output, re.IGNORECASE)
                    for output in tool_outputs.values()
                )

                if has_scale_data or has_scale_output:
                    self.verified_facts.append({
                        "type": "scale_data_verified",
                        "message": f"{scale_name}({desc})数据验证通过"
                    })
                else:
                    self.issues.append({
                        "type": "scale_data_missing",
                        "severity": "error",
                        "message": f"报告提到{scale_name}量表数据，但输入数据中未发现该量表，可能是幻觉内容"
                    })

        # ===== 4. 分析方法验证 =====
        analysis_methods = {
            'VBM': (r'\bVBM\b|体素形态学', ['vbm', 'spm', 'cat12']),
            'DTI': (r'\bDTI\b|扩散张量', ['dti', 'diffusion', 'fsl']),
            'fMRI': (r'\bfMRI\b|功能磁共振', ['fmri', 'bold', 'functional']),
            'ROI': (r'\bROI\b|感兴趣区', ['roi', 'region', 'extract']),
            'ICA': (r'\bICA\b|独立成分', ['ica', 'melodic', 'gift']),
            'SVM': (r'\bSVM\b|支持向量机', ['svm', 'classify', 'machine_learning']),
            'RandomForest': (r'随机森林|Random\s*Forest', ['random_forest', 'rf', 'sklearn']),
        }

        for method_name, (pattern, required_tools) in analysis_methods.items():
            if re.search(pattern, report, re.IGNORECASE):
                has_method_tool = any(
                    any(tool in step for tool in required_tools)
                    for step in all_known_steps
                )

                has_method_output = any(
                    re.search(pattern, output, re.IGNORECASE)
                    for output in tool_outputs.values()
                )

                if has_method_tool or has_method_output:
                    self.verified_facts.append({
                        "type": "analysis_method_verified",
                        "message": f"{method_name}分析方法验证通过"
                    })
                else:
                    self.warnings.append({
                        "type": "analysis_method_unverified",
                        "message": f"报告提到{method_name}分析方法，但未找到对应工具执行记录"
                    })

        # ===== 5. 脑区验证 =====
        brain_regions = {
            '小脑': r'小脑|cerebellum',
            '海马': r'海马|hippocampus',
            '丘脑': r'丘脑|thalamus',
            '基底节': r'基底节|basal\s*ganglia',
            '脑干': r'脑干|brainstem',
            '额叶': r'额叶|frontal\s*lobe',
            '顶叶': r'顶叶|parietal\s*lobe',
            '颞叶': r'颞叶|temporal\s*lobe',
            '枕叶': r'枕叶|occipital\s*lobe',
        }

        mentioned_regions = []
        for region_name, pattern in brain_regions.items():
            if re.search(pattern, report, re.IGNORECASE):
                mentioned_regions.append(region_name)

        if mentioned_regions:
            # 检查是否有脑区分析相关步骤
            roi_keywords = ['roi', 'region', 'atlas', 'parcellation', 'extract', 'label']
            has_roi_analysis = any(
                any(kw in step for kw in roi_keywords)
                for step in all_known_steps
            )

            if has_roi_analysis:
                self.verified_facts.append({
                    "type": "brain_region_verified",
                    "message": f"脑区分析验证通过，提到的脑区：{', '.join(mentioned_regions[:5])}"
                })
            else:
                # 如果有VBM或其他体素分析，也可能涉及脑区
                has_voxel_analysis = any(
                    kw in step for kw in ['vbm', 'spm', 'fsl', 'anova', 'ttest']
                    for step in all_known_steps
                )
                if not has_voxel_analysis:
                    self.warnings.append({
                        "type": "brain_region_unverified",
                        "message": f"报告提到特定脑区（{', '.join(mentioned_regions[:3])}等），但未找到明确的脑区分析步骤"
                    })

    def _llm_content_audit(self, report: str, cohort: Dict, tool_results: List[Dict], llm_client):
        """使用LLM辅助审查内容 — 提供实际数据用于交叉验证"""
        print("  [6/6] LLM内容审查...")

        # 提取实际统计证据作为 ground truth
        from src.agent.moer import _extract_statistical_evidence
        stat_evidence = _extract_statistical_evidence(tool_results, max_items=10)

        # 构建组样本量信息
        groups_info = {}
        if cohort and "groups" in cohort:
            for gname, gdata in cohort["groups"].items():
                if isinstance(gdata, dict):
                    groups_info[gname] = {"n": gdata.get("n", len(gdata.get("subjects", [])))}
                elif isinstance(gdata, list):
                    groups_info[gname] = {"n": len(gdata)}

        context = f"""
实际数据摘要（用于交叉验证报告内容）：
- 总样本数：{cohort.get('total_subjects', 'N/A') if cohort else 'N/A'}
- 分组及样本量：{json.dumps(groups_info, ensure_ascii=False)}
- 工具执行数：{len(tool_results) if tool_results else 0}

实际统计结果（ground truth）：
{json.dumps(stat_evidence, ensure_ascii=False, default=str)[:3000]}
"""

        # 智能截断：优先提取 Results 和 Methods 部分
        report_excerpt = report
        if len(report) > 5000:
            results_match = re.search(
                r'(?:#{1,3}\s*(?:Results?|结果))(.*?)(?=#{1,3}\s|\Z)',
                report, re.DOTALL | re.IGNORECASE
            )
            methods_match = re.search(
                r'(?:#{1,3}\s*(?:Methods?|方法))(.*?)(?=#{1,3}\s|\Z)',
                report, re.DOTALL | re.IGNORECASE
            )
            parts = []
            if results_match:
                parts.append("## Results\n" + results_match.group(1)[:2500])
            if methods_match:
                parts.append("## Methods\n" + methods_match.group(1)[:1500])
            report_excerpt = "\n\n".join(parts) if parts else report[:5000]

        audit_prompt = f"""作为科研审查专家，请对比报告内容与实际数据，检查是否存在数据不一致或虚构内容。

{context}

报告内容（节选）：
{report_excerpt}

请检查：
1. 报告中的样本数是否与实际数据一致
2. 报告中的统计数值（p值、t值、效应量等）是否与实际统计结果匹配
3. 是否有明显编造的统计数值（如过于精确或不合理的p值）
4. 是否有未在数据中出现的组别或变量

请用JSON格式输出审查结果：
{{
    "has_issues": true/false,
    "issues": [
        {{"type": "类型", "description": "描述（引用具体数值）", "severity": "error/warning"}}
    ],
    "summary": "简短总结"
}}

只输出JSON，不要其他内容。"""

        try:
            response = llm_client.chat([
                {"role": "system", "content": "你是一个严谨的科研审查专家，专门检测AI生成报告中的幻觉内容。你将收到实际统计结果作为ground truth，请逐项对比报告声明与实际数据。"},
                {"role": "user", "content": audit_prompt}
            ], max_tokens=1000, temperature=0.1)

            result_text = response["choices"][0]["message"]["content"]

            try:
                result_text = result_text.strip()
                if result_text.startswith("```"):
                    result_text = re.sub(r'^```\w*\n?', '', result_text)
                    result_text = re.sub(r'\n?```$', '', result_text)

                audit_result = json.loads(result_text)

                if audit_result.get("has_issues"):
                    for issue in audit_result.get("issues", []):
                        self.issues.append({
                            "type": f"llm_audit_{issue.get('type', 'unknown')}",
                            "severity": issue.get("severity", "warning"),
                            "message": issue.get("description", "LLM检测到问题")
                        })

                if audit_result.get("summary"):
                    self.verified_facts.append({
                        "type": "llm_summary",
                        "message": audit_result["summary"]
                    })

            except json.JSONDecodeError:
                if "问题" in result_text or "issue" in result_text.lower():
                    self.warnings.append({
                        "type": "llm_audit_raw",
                        "message": f"LLM审查结果（原文）: {result_text[:200]}"
                    })

        except Exception as e:
            self.warnings.append({
                "type": "llm_audit_failed",
                "message": f"LLM审查失败: {str(e)}"
            })

    def _verify_imrad_structure(self, report: str):
        """验证IMRAD结构合规性"""
        print("  [7/9] 验证IMRAD结构...")

        # 检查四个主要部分
        sections = {
            "Introduction": [r'(?i)#*\s*introduction', r'(?i)#*\s*引言', r'(?i)#*\s*背景'],
            "Methods": [r'(?i)#*\s*methods?', r'(?i)#*\s*方法', r'(?i)#*\s*材料与方法'],
            "Results": [r'(?i)#*\s*results?', r'(?i)#*\s*结果'],
            "Discussion": [r'(?i)#*\s*discussion', r'(?i)#*\s*讨论']
        }

        found_sections = {}
        for section_name, patterns in sections.items():
            found = any(re.search(p, report) for p in patterns)
            found_sections[section_name] = found
            if not found:
                self.warnings.append({
                    "type": "imrad_missing_section",
                    "message": f"报告缺少IMRAD结构中的 {section_name} 部分"
                })

        # 检查Methods子部分
        if found_sections.get("Methods"):
            methods_subsections = {
                "Participants": [r'(?i)participant', r'(?i)被试', r'(?i)subject', r'(?i)sample'],
                "Data Acquisition": [r'(?i)acqui', r'(?i)数据采集', r'(?i)MRI\s*param', r'(?i)scanner'],
                "Preprocessing": [r'(?i)preprocess', r'(?i)预处理'],
                "Statistical Analysis": [r'(?i)statistic', r'(?i)统计分析']
            }
            for sub_name, patterns in methods_subsections.items():
                found = any(re.search(p, report) for p in patterns)
                if not found:
                    self.warnings.append({
                        "type": "imrad_missing_methods_sub",
                        "message": f"Methods部分可能缺少 {sub_name} 描述"
                    })

        if all(found_sections.values()):
            self.verified_facts.append({
                "type": "imrad_complete",
                "message": "IMRAD结构完整"
            })

    def _verify_claim_evidence_alignment(self, report: str,
                                          tool_results: List[Dict],
                                          llm_client):
        """使用LLM检查声明-证据对齐 — 提供实际统计数据"""
        print("  [8/9] 检查声明-证据对齐...")

        # 提取报告中的统计声明
        stat_claims = re.findall(
            r'[^.]*(?:significant|显著|p\s*[<>=]|t\s*[<>=]|F\s*[<>=]|r\s*[<>=])[^.]*\.',
            report, re.IGNORECASE
        )

        if not stat_claims:
            return

        claims_to_check = stat_claims[:8]
        claims_text = "\n".join([f"- {c.strip()}" for c in claims_to_check])

        # 提取实际统计证据（而非仅元数据）
        from src.agent.moer import _extract_statistical_evidence
        stat_evidence = _extract_statistical_evidence(tool_results, max_items=10)

        prompt = f"""检查以下报告中的统计声明是否有实际数据支持。

## 统计声明（来自报告）
{claims_text}

## 实际统计结果（ground truth — 来自工具执行）
{json.dumps(stat_evidence, ensure_ascii=False, default=str)[:3000]}

请逐条对比每个声明与实际数据：
1. 声明中的p值/统计量是否与实际结果匹配
2. 声明中的显著性判断是否与实际p值一致
3. 声明中的效应量描述是否与实际效应量一致

请以JSON格式返回：
{{"unsupported_claims": ["无数据支持的声明及原因"], "overstatements": ["过度解释的声明及原因"], "verified_claims": ["与数据一致的声明"]}}

只输出JSON。"""

        try:
            response = llm_client.chat([
                {"role": "system", "content": "你是科研审查专家，逐条对比报告声明与实际统计数据，检查是否一致。"},
                {"role": "user", "content": prompt}
            ], max_tokens=800, temperature=0.1)

            result_text = response["choices"][0]["message"]["content"].strip()
            if result_text.startswith("```"):
                result_text = re.sub(r'^```\w*\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)

            result = json.loads(result_text)

            for claim in result.get("unsupported_claims", []):
                self.warnings.append({
                    "type": "unsupported_claim",
                    "message": f"可能缺乏数据支持: {claim[:150]}"
                })
            for claim in result.get("overstatements", []):
                self.warnings.append({
                    "type": "overstatement",
                    "message": f"可能过度解释: {claim[:150]}"
                })
            for claim in result.get("verified_claims", []):
                self.verified_facts.append({
                    "type": "claim_verified",
                    "message": f"声明与数据一致: {claim[:150]}"
                })
        except Exception as e:
            self.warnings.append({
                "type": "claim_check_failed",
                "message": f"声明-证据对齐检查失败: {str(e)}"
            })

    def _aggregate_moer_reviews(self, moer_reviews: List[Dict]):
        """聚合MoER历史审查结果到最终报告"""
        print("  [9/9] 聚合MoER历史审查...")

        for review in moer_reviews:
            try:
                # 校验必要字段
                if not isinstance(review, dict):
                    continue
                reviewer = review.get("reviewer")
                if not reviewer or not isinstance(reviewer, str):
                    continue
                review_issues = review.get("issues", [])
                if not isinstance(review_issues, list):
                    review_issues = []
                review_status = review.get("status", "")

                for issue in review_issues:
                    if not isinstance(issue, dict):
                        continue
                    severity = issue.get("severity", "info")
                    message = issue.get("message", "")
                    if not message:
                        continue
                    prefixed_msg = f"[{reviewer}] {message}"

                    if severity == "error":
                        self.issues.append({
                            "type": f"moer_{reviewer.lower()}",
                            "severity": "warning",  # 降级为warning，因为是历史记录
                            "message": prefixed_msg
                        })
                    elif severity == "warning":
                        self.warnings.append({
                            "type": f"moer_{reviewer.lower()}",
                            "message": prefixed_msg
                        })

                if review_status in ("needs_revision", "rejected"):
                    self.warnings.append({
                        "type": f"moer_{reviewer.lower()}_status",
                        "message": f"[{reviewer}] 审查状态: {review_status} (评分: {review.get('score', 'N/A')})"
                    })
            except Exception as e:
                print(f"    [MoER聚合] 跳过格式异常的审查记录: {e}")

    def _generate_audit_report(self) -> Dict[str, Any]:
        """生成审查报告"""
        # 计算审查分数
        error_count = sum(1 for i in self.issues if i.get("severity") == "error")
        warning_count = len(self.warnings) + sum(1 for i in self.issues if i.get("severity") == "warning")
        verified_count = len(self.verified_facts)

        # 计算可信度分数 (0-100)
        base_score = 100
        base_score -= error_count * 20  # 每个错误扣20分
        base_score -= warning_count * 5  # 每个警告扣5分
        base_score = max(0, min(100, base_score))  # 限制在0-100

        # 根据验证事实调整
        if verified_count > 0:
            base_score = min(100, base_score + verified_count * 2)

        # 确定审查状态
        if error_count > 0:
            status = "FAILED"
            status_message = f"发现 {error_count} 个严重问题，报告可能包含虚假内容"
        elif warning_count > 3:
            status = "WARNING"
            status_message = f"发现 {warning_count} 个警告，建议人工复核"
        else:
            status = "PASSED"
            status_message = "审查通过，未发现明显问题"

        result = {
            "status": status,
            "status_message": status_message,
            "credibility_score": base_score,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "errors": error_count,
                "warnings": warning_count,
                "verified_facts": verified_count
            },
            "issues": self.issues,
            "warnings": self.warnings,
            "verified_facts": self.verified_facts
        }

        # 打印摘要
        print(f"\n[Report Audit] 审查完成")
        print(f"  状态: {status}")
        print(f"  可信度: {base_score}/100")
        print(f"  错误: {error_count}, 警告: {warning_count}, 已验证: {verified_count}")

        if self.issues:
            print(f"\n  发现的问题:")
            for issue in self.issues[:3]:
                print(f"    - [{issue.get('severity', 'unknown')}] {issue.get('message', '')}")

        return result


def audit_report(
    report_content: str,
    cohort: Dict,
    tool_results: List[Dict],
    plan: Dict,
    run_dir: Path = None,
    llm_client=None,
    moer_reviews: List[Dict] = None
) -> Tuple[Dict[str, Any], str]:
    """
    审查报告并返回审查结果和修改后的报告

    Args:
        report_content: 原始报告内容
        cohort: 队列信息
        tool_results: 工具执行结果
        plan: 研究计划
        run_dir: 运行目录
        llm_client: LLM客户端
        moer_reviews: MoER历史审查记录

    Returns:
        (审查结果, 带审查标记的报告)
    """
    auditor = ReportAuditor(run_dir)
    audit_result = auditor.audit_report(
        report_content, cohort, tool_results, plan, llm_client,
        moer_reviews=moer_reviews
    )

    # 在报告末尾添加审查信息
    audit_section = f"""

---

## 报告审查信息

| 项目 | 结果 |
|------|------|
| 审查状态 | {audit_result['status']} |
| 可信度评分 | {audit_result['credibility_score']}/100 |
| 审查时间 | {audit_result['timestamp'][:19]} |
| 错误数 | {audit_result['statistics']['errors']} |
| 警告数 | {audit_result['statistics']['warnings']} |
| 已验证项 | {audit_result['statistics']['verified_facts']} |

"""

    if audit_result['issues']:
        audit_section += "\n### 发现的问题\n\n"
        for issue in audit_result['issues']:
            severity_icon = "X" if issue.get('severity') == 'error' else "!"
            audit_section += f"- [{severity_icon}] {issue.get('message', '')}\n"

    if audit_result['warnings']:
        audit_section += "\n### 警告\n\n"
        for warning in audit_result['warnings'][:5]:
            audit_section += f"- {warning.get('message', '')}\n"

    if audit_result['status'] == "FAILED":
        audit_section += "\n> **注意**: 本报告存在可能的虚假内容，请谨慎使用。建议人工核实关键数据。\n"
    elif audit_result['status'] == "WARNING":
        audit_section += "\n> **提示**: 本报告已通过基本审查，但建议人工复核标记的警告项。\n"

    modified_report = report_content + audit_section

    return audit_result, modified_report
