"""
Vibe Coding Module - Generating and Reviewing Analysis Algorithm Code Using GLM-4.6

This module is responsible for:
1. Generating analysis algorithm code based on research plans and image data metrics
2. Running the code and fixing errors based on actual runtime issues
3. Saving all generated code for user review and reproduction

Process: Generate code -> Syntax check -> Run -> Fix based on runtime errors -> Loop until successful
"""

import json
import ast
import re
import subprocess
import sys
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from glob import glob

import numpy as np

from src.utils.llm import get_llm_client
from src.config import BRAIN_ATLASES, ATLAS_DIR


class VibeCodingEngine:
    """算法代码生成引擎"""

    def __init__(self):
        self.llm = get_llm_client()
        # Session内代码缓存：key为任务hash，value为执行结果
        self._code_cache: Dict[str, Dict[str, Any]] = {}

    def _get_task_hash(self, task_description: str, plan_pipeline: List[Dict]) -> str:
        """
        生成任务唯一标识用于缓存

        Args:
            task_description: 任务描述
            plan_pipeline: 研究计划的pipeline部分

        Returns:
            16位hash字符串
        """
        # 将pipeline转为稳定的字符串形式
        pipeline_str = json.dumps(plan_pipeline, sort_keys=True, ensure_ascii=False)
        content = f"{task_description}:{pipeline_str}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def generate_algorithm_code(
        self,
        plan: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        task_description: str,
        run_dir: Path,
        brain_region_suggestions: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        根据研究计划和工具结果生成分析算法代码

        Args:
            plan: 研究计划
            tool_results: 工具执行结果（包含提取的脑区指标或mask）
            task_description: 任务描述
            run_dir: 运行目录
            brain_region_suggestions: 脑区选择建议（来自search_knowledge）

        Returns:
            生成结果字典，包含代码、审查结果、保存路径等
        """
        # === 缓存检查：避免重复生成相同任务的代码 ===
        task_hash = self._get_task_hash(task_description, plan.get("pipeline", []))

        if task_hash in self._code_cache:
            cached = self._code_cache[task_hash]
            if cached.get("success"):
                print(f"\n[缓存命中] 复用已成功执行的代码")
                print(f"  任务hash: {task_hash}")
                print(f"  代码路径: {cached.get('save_path', 'N/A')}")
                return cached
            else:
                print(f"\n[缓存跳过] 之前执行失败，重新生成代码 (hash: {task_hash})")

        print("\n" + "="*60)
        print("Vibe Coding - 算法代码生成")
        print("="*60)

        # 显示ROI选择策略
        roi_selection = plan.get("roi_selection", {}) or brain_region_suggestions or {}
        if roi_selection:
            strategy = roi_selection.get("strategy") or roi_selection.get("analysis_priority", "exploratory")
            primary_rois = roi_selection.get("primary_rois", [])
            print(f"  [ROI策略] 分析模式: {strategy}")
            if primary_rois:
                print(f"  [ROI策略] Primary ROIs: {primary_rois[:5]}{'...' if len(primary_rois) > 5 else ''}")

        # 从研究计划推断目标数据模态
        target_modality = self._infer_target_modality(plan)
        if target_modality:
            print(f"  [数据模态] 研究计划指定使用: {target_modality.upper()}")

        # 1. 准备输入数据文件（从NIfTI提取脑区体积到CSV）
        data_prep_result = self._prepare_input_data(tool_results, run_dir, target_modality=target_modality)

        # 2. 提取数据指标信息
        data_indicators = self._extract_indicators_from_results(tool_results)

        # 将数据准备结果添加到指标中
        data_indicators["input_data_file"] = data_prep_result.get("data_file", "")
        data_indicators["data_summary"] = data_prep_result.get("summary", {})
        data_indicators["n_subjects"] = data_prep_result.get("n_subjects", 0)
        data_indicators["groups"] = data_prep_result.get("groups", [])

        # 3. 构建代码生成提示（包含ROI优先级信息和工具格式文档）
        code_prompt = self._build_code_generation_prompt(
            plan, data_indicators, task_description, brain_region_suggestions, tool_results
        )

        # 3. 使用GLM-4.6生成代码
        print("\n[步骤 1/4] 使用 GLM-4.6 生成算法代码...")
        self.llm.set_task_type("algorithm_code_generation")

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self._get_code_generation_system_prompt()},
                    {"role": "user", "content": code_prompt}
                ],
                temperature=0.3,
                max_tokens=32768  # 大幅增加token限制，确保复杂算法代码完整生成
            )

            generated_content = response["choices"][0]["message"]["content"]
            code = self._extract_code_from_response(generated_content)

            print(f"  [OK] 代码生成完成 ({len(code)} 字符)")

        except Exception as e:
            print(f"  [ERROR] 代码生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "review": {}
            }

        # 4. 新流程：生成 -> 语法检查 -> 运行 -> 根据运行错误修复 -> 循环
        max_fix_attempts = 10 # 减少到5次，避免过多重复尝试
        attempt = 0
        execution_success = False
        execution_result = None
        save_result = None
        validation_result = None  # 验证结果
        error_history = []  # 记录错误历史，防止重复错误

        while attempt < max_fix_attempts:
            attempt += 1

            # 4.1 语法检查
            print(f"\n[步骤 2/5] 语法检查... (尝试 {attempt}/{max_fix_attempts})")
            syntax_check = self._check_syntax(code)

            if syntax_check["has_error"]:
                print(f"  [ERROR] 语法错误: {syntax_check['error']}")
                if attempt < max_fix_attempts:
                    print(f"\n[步骤 3/5] 修复语法错误...")
                    code = self._fix_syntax_error(code, syntax_check)
                    continue
                else:
                    print(f"  [FAIL] 语法错误无法修复")
                    break

            print("  [OK] 语法检查通过")

            # 4.2 保存代码
            print(f"\n[步骤 3/5] 保存代码...")
            save_result = self._save_generated_code(
                code=code,
                task_description=task_description,
                plan=plan,
                review_result={"has_errors": False, "errors": [], "warnings": []},
                run_dir=run_dir
            )
            print(f"  [OK] 代码已保存: {save_result['code_path']}")

            # 4.3 运行代码
            print(f"\n[步骤 4/5] 运行代码...")
            execution_result = self._execute_code(
                code_path=save_result["code_path"],
                run_dir=run_dir
            )

            if execution_result["success"]:
                print(f"  [OK] 代码执行成功!")
                if execution_result.get("output_files"):
                    print(f"  [输出] 生成 {len(execution_result['output_files'])} 个文件")

                # 验证分析结果
                output_files = execution_result.get("output_files", [])
                if output_files:
                    validation_result = self._validate_analysis_results(output_files, run_dir)
                    if not validation_result["valid"]:
                        print(f"  [WARNING] 结果验证发现问题:")
                        for err in validation_result.get("errors", []):
                            print(f"    ❌ {err['type']}: {err['message'][:100]}")
                        # 如果是ROI值重复问题，尝试重新生成代码
                        if any(e["type"] == "roi_value_duplication" for e in validation_result.get("errors", [])):
                            print(f"  [重试] 检测到ROI值重复错误，将修复代码...")
                            roi_error_msg = "ROI值重复错误 - 所有ROI的统计值相同，表明代码使用了全脑均值而非ROI特定值。必须为每个ROI单独提取数据。"
                            error_history.append({
                                "attempt": attempt,
                                "error": roi_error_msg
                            })
                            # 构造虚拟 execution_result 传递 ROI 错误信息
                            roi_execution_result = {
                                "success": False,
                                "stderr": roi_error_msg,
                                "error": roi_error_msg
                            }
                            previous_code = code
                            code = self._fix_runtime_error(
                                code=code,
                                execution_result=roi_execution_result,
                                task_description=task_description,
                                data_indicators=data_indicators,
                                error_history=error_history
                            )
                            if len(code) < 500:
                                print(f"  [WARNING] 修复后代码过短 ({len(code)} 字符)，恢复原版本")
                                code = previous_code
                            continue
                    else:
                        print(f"  [验证] 结果验证通过")

                execution_success = True
                break
            else:
                print(f"  [ERROR] 代码执行失败")
                error_msg = execution_result.get("stderr", "") or execution_result.get("error", "")
                # 处理Unicode编码问题
                try:
                    print(f"  [错误信息] {error_msg[:500]}")
                except UnicodeEncodeError:
                    # 替换无法编码的字符
                    safe_msg = error_msg[:500].encode('gbk', errors='replace').decode('gbk')
                    print(f"  [错误信息] {safe_msg}")

                # 记录错误历史
                error_history.append({
                    "attempt": attempt,
                    "error": error_msg[:1000]  # 保留前1000字符
                })

                if attempt < max_fix_attempts:
                    # 4.4 根据运行错误修复代码
                    print(f"\n[步骤 5/5] 根据运行错误修复代码... (第 {attempt} 次)")
                    previous_code = code
                    code = self._fix_runtime_error(
                        code=code,
                        execution_result=execution_result,
                        task_description=task_description,
                        data_indicators=data_indicators,
                        error_history=error_history
                    )

                    # 检查代码完整性 - 防止LLM返回代码片段
                    if len(code) < 500:
                        print(f"  [WARNING] 修复后的代码过短 ({len(code)} 字符)，可能不完整")
                        print(f"  [WARNING] 恢复到之前的版本并重试")
                        code = previous_code
                        # 使用更高温度重新生成
                        print(f"  [重试] 使用不同策略重新修复...")
                        code = self._fix_runtime_error_with_retry(
                            code=code,
                            execution_result=execution_result,
                            task_description=task_description,
                            data_indicators=data_indicators,
                            error_history=error_history
                        )
                else:
                    print(f"  [FAIL] 已达到最大修复次数 ({max_fix_attempts})")
                    print(f"  [建议] 检查数据文件路径、依赖库安装和代码逻辑")

        # 5. 返回结果
        result = {
            "success": execution_success,
            "code": code,
            "save_path": save_result["code_path"] if save_result else "",
            "metadata_path": save_result["metadata_path"] if save_result else "",
            "code_length": len(code),
            "execution_success": execution_success,
            "execution_result": execution_result,
            "validation_result": validation_result,  # 结果验证
            "attempts": attempt,
            "output_files": execution_result.get("output_files", []) if execution_result else [],
            "has_syntax_errors": False,  # 如果到这里，语法是正确的
            "errors": [],
            "warnings": []
        }

        # === 缓存保存：成功执行的代码保存到缓存 ===
        if execution_success:
            self._code_cache[task_hash] = result
            print(f"  [缓存保存] 成功执行的代码已缓存 (hash: {task_hash})")

        return result

    def _extract_indicators_from_results(self, tool_results: List[Dict]) -> Dict[str, Any]:
        """从工具结果中提取脑区指标信息"""
        indicators = {
            "brain_regions": [],
            "masks": [],
            "metrics": [],
            "data_files": [],
            "stats_results": [],
            "segmentation_files": [],
            "smoothed_files": []
        }

        for result in tool_results:
            if not isinstance(result, dict):
                continue

            # 提取脑区信息
            if "brain_regions" in result:
                indicators["brain_regions"].extend(result["brain_regions"])

            # 提取mask文件
            if "mask" in result or "mask_file" in result:
                mask_file = result.get("mask") or result.get("mask_file")
                if mask_file:
                    indicators["masks"].append(mask_file)

            # 提取指标
            if "metrics" in result:
                indicators["metrics"].extend(result["metrics"])

            # 从outputs中提取文件
            outputs = result.get("outputs", {})
            if isinstance(outputs, dict):
                output_files = outputs.get("output_files", [])
                for f in output_files:
                    if isinstance(f, str):
                        if f.endswith(".nii") or f.endswith(".nii.gz"):
                            indicators["data_files"].append(f)
                            # 分类文件
                            fname = Path(f).name.lower()
                            if fname.startswith(('c1', 'c2', 'c3', 'wc1', 'wc2')):
                                indicators["segmentation_files"].append(f)
                            if fname.startswith(('s', 'sw')):
                                indicators["smoothed_files"].append(f)
                        elif f.endswith(".json"):
                            indicators["stats_results"].append(f)

            # 从result字段提取（task_list格式）
            task_result = result.get("result", {})
            if isinstance(task_result, dict):
                output_files = task_result.get("output_files", [])
                for f in output_files:
                    if isinstance(f, str) and f.endswith((".nii", ".nii.gz")):
                        indicators["data_files"].append(f)
                        fname = Path(f).name.lower()
                        if fname.startswith(('c1', 'c2', 'c3', 'wc1', 'wc2')):
                            indicators["segmentation_files"].append(f)

            # 提取统计结果
            if "stats" in result or "statistics" in result:
                stats = result.get("stats") or result.get("statistics")
                if stats:
                    indicators["stats_results"].append(stats)

        # 去重
        for key in ["data_files", "segmentation_files", "smoothed_files", "stats_results"]:
            if isinstance(indicators[key], list):
                seen = set()
                unique = []
                for item in indicators[key]:
                    if isinstance(item, str) and item not in seen:
                        seen.add(item)
                        unique.append(item)
                    elif isinstance(item, dict):
                        unique.append(item)
                indicators[key] = unique

        return indicators

    def _get_tool_output_documentation(self, tool_name: str) -> str:
        """
        获取特定工具的输出格式详细文档

        Args:
            tool_name: 工具名称 (freesurfer, spm, fsl等)

        Returns:
            该工具输出格式的详细说明
        """
        docs = {
            "freesurfer": """
## FreeSurfer 输出格式详细说明

### 目录结构
FreeSurfer recon-all 处理后，每个被试的目录结构如下：
```
SUBJECTS_DIR/
  subject_id/
    ├── mri/               # 影像文件（.mgz格式）
    │   ├── orig.mgz       # 原始T1影像
    │   ├── brain.mgz      # 剥离后的脑组织
    │   ├── aseg.mgz       # 皮下结构分割
    │   ├── norm.mgz       # 归一化影像
    │   └── ...
    ├── surf/              # 表面文件
    │   ├── lh.pial        # 左半球皮层表面
    │   ├── rh.pial        # 右半球皮层表面
    │   ├── lh.white       # 左半球白质表面
    │   ├── rh.white       # 右半球白质表面
    │   └── ...
    └── stats/             # 统计结果文件（重要！数据提取的主要来源）
        ├── aseg.stats     # 皮下结构体积统计
        ├── lh.aparc.stats # 左半球皮层分区统计（DK atlas）
        ├── rh.aparc.stats # 右半球皮层分区统计
        ├── lh.aparc.a2009s.stats  # Destrieux atlas
        ├── rh.aparc.a2009s.stats
        └── ...
```

### aseg.stats 文件格式（皮下结构体积）

**文件保存路径**: `SUBJECTS_DIR/subject_id/stats/aseg.stats`

**格式**: 空格分隔的表格，包含以下列（注意：第一列是Index！）：

```
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   4     15676    15939.7  Left-Lateral-Ventricle            33.9264    11.4073    12.0000    83.0000    71.0000
  2   5       289      334.7  Left-Inf-Lat-Vent                 48.4810    13.9027    19.0000    83.0000    64.0000
  3   7     13614    14291.0  Left-Cerebellum-White-Matter      85.3890     6.1163    38.0000   103.0000    65.0000
  4   8     46651    46379.2  Left-Cerebellum-Cortex            64.1554    10.5700    12.0000   117.0000   105.0000
  5  10      7529     7302.8  Left-Thalamus                     85.3829    11.0551    28.0000   110.0000    82.0000
  6  11      5068     5011.1  Left-Caudate                      83.7608     7.9405    49.0000   107.0000    58.0000
  7  12      6098     5926.2  Left-Putamen                      88.5187     6.4889    44.0000   109.0000    65.0000
  8  13      2728     2750.4  Left-Pallidum                     96.4593     8.6540    52.0000   123.0000    71.0000
  9  17      4166     3987.2  Left-Hippocampus                  71.7283     9.2010    26.0000   101.0000    75.0000
 10  18      1794     1779.4  Left-Amygdala                     75.9950     6.7588    30.0000    96.0000    66.0000
...
```

**⚠️ 重要：列索引定义（按空格分隔后的parts数组）**:
- **`parts[0]`**: Index - 行号（1, 2, 3, ...）
- **`parts[1]`**: SegId - FreeSurfer分割ID（4, 5, 7, 8, ...）
- **`parts[2]`**: NVoxels - 体素数量
- **`parts[3]`**: Volume_mm3 - **体积（立方毫米）**（这是我们需要提取的数值！）
- **`parts[4]`**: StructName - **结构名称**（如 Left-Hippocampus，这是我们需要的列名！）
- `parts[5]`: normMean - 归一化影像的平均强度
- `parts[6]`: normStdDev - 归一化影像的标准差
- `parts[7-9]`: normMin, normMax, normRange - 强度范围

**⚠️ 常见错误（必须避免）**:
1. ❌ **列索引错误**：
   - 错误：`volume = float(parts[2])`（这是NVoxels，不是体积！）
   - 正确：`volume = float(parts[3])`（这才是Volume_mm3）

2. ❌ **结构名称提取错误**：
   - 错误：`struct_name = parts[3]`（这是体积数值，不是名称！）
   - 正确：`struct_name = parts[4]`（这才是StructName）

3. ❌ **使用parts[0]作为SegId**：
   - 错误：`seg_id = parts[0]`（这是Index行号）
   - 正确：`seg_id = parts[1]`（这才是SegId）

**✅ 正确的Python提取代码**:
```python
import pandas as pd
from pathlib import Path

def extract_aseg_volumes(subject_dir):
    \"\"\"从aseg.stats提取皮下结构体积\"\"\"
    stats_file = Path(subject_dir) / 'stats' / 'aseg.stats'

    if not stats_file.exists():
        raise FileNotFoundError(f"aseg.stats not found: {stats_file}")

    volumes = {}
    with open(stats_file, 'r') as f:
        for line in f:
            # 跳过注释行和空行
            if line.startswith('#') or not line.strip():
                continue

            parts = line.split()  # 按空格分隔
            if len(parts) >= 5:
                seg_id = parts[1]        # SegId（第2列）
                volume = float(parts[3])  # Volume_mm3（第4列）✅
                struct_name = parts[4]   # StructName（第5列）✅
                volumes[struct_name] = volume

    return volumes

# 使用示例
subject_volumes = extract_aseg_volumes('/path/to/SUBJECTS_DIR/subject_001')
print(f"Left Hippocampus: {subject_volumes['Left-Hippocampus']} mm³")
print(f"Left Cerebellum Cortex: {subject_volumes['Left-Cerebellum-Cortex']} mm³")
```

**重要的脑区（SCA3等小脑性共济失调研究）**:
- Left-Cerebellum-White-Matter (SegId=7)
- Left-Cerebellum-Cortex (SegId=8)
- Right-Cerebellum-White-Matter (SegId=46)
- Right-Cerebellum-Cortex (SegId=47)
- Brain-Stem (SegId=16)
- Left-Thalamus (SegId=10), Right-Thalamus (SegId=49)
- Left-Hippocampus (SegId=17), Right-Hippocampus (SegId=53)
- Left-Amygdala (SegId=18), Right-Amygdala (SegId=54)
- Left-Caudate (SegId=11), Right-Caudate (SegId=50)
- Left-Putamen (SegId=12), Right-Putamen (SegId=51)
- Left-Pallidum (SegId=13), Right-Pallidum (SegId=52)

### 全脑体积指标（从aseg.stats注释中提取）

aseg.stats文件开头的注释行包含全脑体积摘要：
```
# Measure BrainSeg, BrainSegVol, Brain Segmentation Volume, 1131575.000000, mm^3
# Measure BrainSegNotVent, BrainSegVolNotVent, Brain Segmentation Volume Without Ventricles, 1094666.000000, mm^3
# Measure lhCortex, lhCortexVol, Left hemisphere cortical gray matter volume, 221714.682884, mm^3
# Measure rhCortex, rhCortexVol, Right hemisphere cortical gray matter volume, 220134.770618, mm^3
# Measure Cortex, CortexVol, Total cortical gray matter volume, 441849.453502, mm^3
# Measure SubCortGray, SubCortGrayVol, Subcortical gray matter volume, 63757.000000, mm^3
# Measure TotalGray, TotalGrayVol, Total gray matter volume, 600551.453502, mm^3
```

**提取全脑指标**:
```python
def extract_brain_volumes(stats_file):
    \"\"\"从aseg.stats注释中提取全脑体积\"\"\"
    volumes = {}
    with open(stats_file, 'r') as f:
        for line in f:
            if line.startswith('# Measure'):
                parts = line.strip().split(', ')
                if len(parts) >= 5:
                    measure_name = parts[1]  # 如 BrainSegVol
                    measure_value = float(parts[3])  # 体积值
                    volumes[measure_name] = measure_value
    return volumes
```

### lh.aparc.stats / rh.aparc.stats 文件格式（皮层分区）

**文件路径**:
- 左半球：`SUBJECTS_DIR/subject_id/stats/lh.aparc.stats`
- 右半球：`SUBJECTS_DIR/subject_id/stats/rh.aparc.stats`

**格式**: 空格分隔

```
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
bankssts                                 1346     882     2894  2.510  0.598     0.140     0.030       19     1.6
caudalanteriorcingulate                  1156     830     2656  2.712  0.682     0.137     0.028       18     1.3
```

**列定义**:
- `parts[0]`: StructName - 脑区名称
- `parts[1]`: NumVert - 顶点数
- `parts[2]`: SurfArea - 表面积（mm²）
- `parts[3]`: GrayVol - 灰质体积（mm³）
- `parts[4]`: ThickAvg - 平均皮层厚度（mm）
""",

            "spm": """
## SPM (Statistical Parametric Mapping) 输出格式详细说明

### VBM (Voxel-Based Morphometry) 分析输出

**目录结构**:
```
output_dir/
  subject_id/
    ├── c1*.nii        # 灰质组织概率图（原始空间，0-1范围）
    ├── c2*.nii        # 白质组织概率图（原始空间）
    ├── c3*.nii        # 脑脊液组织概率图（原始空间）
    ├── wc1*.nii       # 归一化的灰质（MNI空间）
    ├── wc2*.nii       # 归一化的白质（MNI空间）
    ├── mwc1*.nii      # 调制+归一化的灰质（用于VBM统计）
    ├── mwc2*.nii      # 调制+归一化的白质
    ├── swc1*.nii      # 平滑后的归一化灰质（8mm FWHM）
    └── *_seg8.mat     # 分割参数文件
```

**文件前缀含义**:
- `c*`: native space（原始空间）
- `w`: warped（归一化到MNI空间）
- `m`: modulated（调制，保留体积信息）
- `s`: smoothed（平滑，通常8mm FWHM）

**Python读取代码**:
```python
import nibabel as nib
import numpy as np

def calculate_tissue_volume(tissue_file):
    \"\"\"计算SPM组织概率图的总体积\"\"\"
    img = nib.load(tissue_file)
    data = img.get_fdata()  # 概率图，值范围0-1
    affine = img.affine
    voxel_vol = np.abs(np.linalg.det(affine[:3, :3]))  # mm³
    total_volume_ml = np.sum(data) * voxel_vol / 1000  # 转换为mL
    return total_volume_ml

# 使用示例
gm_volume = calculate_tissue_volume('c1subject001.nii')
wm_volume = calculate_tissue_volume('c2subject001.nii')
```
"""
        }

        return docs.get(tool_name.lower(), f"# {tool_name} 输出格式\n\n暂无详细文档。\n")

    def _detect_upstream_tools(self, tool_results: List[Dict]) -> List[str]:
        """
        检测上游使用了哪些工具

        Args:
            tool_results: 工具执行结果列表

        Returns:
            工具名称列表
        """
        detected_tools = set()

        for result in tool_results:
            if not isinstance(result, dict):
                continue

            tool_name = result.get("tool_name", "").lower()

            # 检测FreeSurfer
            if "freesurfer" in tool_name:
                detected_tools.add("freesurfer")

            # 检测SPM
            elif "spm" in tool_name or "vbm" in tool_name:
                detected_tools.add("spm")

            # 从outputs中检测
            outputs = result.get("outputs", {})
            if isinstance(outputs, dict):
                subjects_dir = outputs.get("subjects_dir", "")
                if subjects_dir:
                    detected_tools.add("freesurfer")

                output_files = outputs.get("output_files", [])
                for f in output_files:
                    if isinstance(f, str):
                        if any(prefix in f for prefix in ['c1', 'c2', 'wc1', 'wc2', 'mwc1']):
                            detected_tools.add("spm")

        return list(detected_tools)

    def _infer_target_modality(self, plan: Dict) -> Optional[str]:
        """
        从研究计划推断目标数据模态

        Args:
            plan: 研究计划字典

        Returns:
            目标模态: "dwi", "anat", "func", 或 None
        """
        # 1. 检查plan顶级modalities字段
        modalities = plan.get("modalities", [])

        # 2. 如果只有一个模态，直接使用
        if len(modalities) == 1:
            return modalities[0]

        # 3. 多模态时，检查pipeline中的统计分析步骤需要什么数据
        pipeline = plan.get("pipeline", [])
        for step in pipeline:
            tool = step.get("tool", "")
            if tool == "python_stats":
                # 检查参数中是否提到DTI/FA/MD等
                params = step.get("parameters", {})
                params_str = str(params).lower()
                if any(k in params_str for k in ["fa", "md", "rd", "ad", "dti", "diffusion", "tract"]):
                    return "dwi"
                if any(k in params_str for k in ["bold", "activation", "fmri", "functional"]):
                    return "func"

        # 4. 检查研究问题/标题中是否有模态关键词
        title = plan.get("title", "").lower()
        question = plan.get("research_question", "").lower()
        text_to_check = f"{title} {question}"

        if any(k in text_to_check for k in ["dti", "dwi", "扩散", "diffusion", "白质", "fa", "md"]):
            return "dwi"
        if any(k in text_to_check for k in ["fmri", "功能", "bold", "activation", "functional"]):
            return "func"

        # 5. 默认返回第一个模态（如果有的话）
        return modalities[0] if modalities else None

    def _group_results_by_modality(self, tool_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        按模态分组工具结果

        Args:
            tool_results: 工具执行结果列表

        Returns:
            按模态分组的字典: {"dwi": [...], "anat": [...], "func": [...], "unknown": [...]}
        """
        grouped = {"dwi": [], "anat": [], "func": [], "unknown": []}

        for result in tool_results:
            # 1. 首先检查outputs中的modality字段（工具直接返回的）
            outputs = result.get("outputs", {}) or result.get("result", {})
            modality = outputs.get("modality")

            if modality and modality in grouped:
                grouped[modality].append(result)
                continue

            # 2. 如果没有显式modality，根据工具名称和输出推断
            tool_name = result.get("tool_name", "").lower()

            # FSL DTI相关命令
            if "fsl" in tool_name:
                output_files = outputs.get("output_files", [])
                commands = outputs.get("commands", [])
                # 检查是否有DTI相关输出
                has_dti = any(
                    any(x in str(f).upper() for x in ["_FA", "_MD", "_RD", "_AD", "EDDY", "DTIFIT"])
                    for f in output_files
                )
                if has_dti or any("eddy" in str(c) or "dtifit" in str(c) for c in commands):
                    grouped["dwi"].append(result)
                else:
                    grouped["anat"].append(result)
            elif "freesurfer" in tool_name:
                grouped["anat"].append(result)
            elif "spm" in tool_name:
                analysis_type = outputs.get("analysis_type", "")
                if analysis_type in ["slice_timing", "realign", "coregister", "normalize"]:
                    grouped["func"].append(result)
                else:
                    grouped["anat"].append(result)
            elif "dsi" in tool_name or "dsi_studio" in tool_name:
                grouped["dwi"].append(result)
            else:
                grouped["unknown"].append(result)

        return grouped

    def _extract_dti_files_from_results(self, dwi_results: List[Dict]) -> Dict[str, List[str]]:
        """
        从DWI/DTI工具结果中提取DTI指标文件

        Args:
            dwi_results: DWI模态的工具结果列表

        Returns:
            DTI指标文件字典: {"FA": [...], "MD": [...], ...}
        """
        dti_files = {"FA": [], "MD": [], "RD": [], "AD": [], "L1": [], "L2": [], "L3": []}

        for result in dwi_results:
            outputs = result.get("outputs", {}) or result.get("result", {})
            if not isinstance(outputs, dict):
                continue

            output_files = outputs.get("output_files", [])
            for f in output_files:
                if not isinstance(f, str):
                    continue
                fname = Path(f).name
                for metric in dti_files.keys():
                    if f"_{metric}" in fname and fname.endswith('.nii.gz'):
                        dti_files[metric].append(f)

        return dti_files

    def _prepare_input_data(self, tool_results: List[Dict], run_dir: Path, target_modality: Optional[str] = None) -> Dict[str, Any]:
        """
        从工具结果准备输入数据文件（CSV格式）
        支持SPM、FreeSurfer和FSL DTI输出

        基于研究计划指定的模态选择数据源：
        - target_modality="dwi" → 使用FSL DTI数据（FA/MD等）
        - target_modality="anat" → 使用FreeSurfer或SPM数据
        - target_modality=None → 使用默认优先级

        Args:
            tool_results: 工具执行结果
            run_dir: 运行目录
            target_modality: 目标数据模态（"dwi", "anat", "func", None）

        Returns:
            包含数据文件路径和元信息的字典
        """
        import pandas as pd
        import nibabel as nib
        import numpy as np
        from scipy import ndimage
        from src.config import BRAIN_ATLASES, SCA3_PRIORITY_ATLASES

        print("  [数据准备] 从工具结果提取数据...")
        if target_modality:
            print(f"  [目标模态] 根据研究计划使用 {target_modality.upper()} 数据")

        # ========== 按模态分组工具结果 ==========
        results_by_modality = self._group_results_by_modality(tool_results)
        for mod, results in results_by_modality.items():
            if results:
                print(f"  [检测] {mod} 模态: {len(results)} 个工具结果")

        # ========== 基于目标模态选择数据源 ==========
        # 如果指定了目标模态，优先使用该模态的数据

        if target_modality == "dwi" and results_by_modality.get("dwi"):
            print(f"  [选择] 使用DWI/DTI数据（根据研究计划）")
            # 从DWI结果中提取DTI文件
            dti_files = self._extract_dti_files_from_results(results_by_modality["dwi"])
            if any(files for files in dti_files.values()):
                return self._extract_fsl_dti_data(dti_files, run_dir)

        if target_modality == "anat" and results_by_modality.get("anat"):
            print(f"  [选择] 使用解剖结构数据（根据研究计划）")
            # 从anat结果中选择FreeSurfer（优先）或SPM
            freesurfer_result = None
            spm_results = []

            for result in results_by_modality["anat"]:
                tool_name = result.get("tool_name", "").lower()
                if "freesurfer" in tool_name:
                    outputs = result.get("outputs", {})
                    if outputs.get("subjects_dir") and outputs.get("processed_subjects"):
                        freesurfer_result = {
                            "subjects_dir": outputs["subjects_dir"],
                            "processed_subjects": outputs["processed_subjects"]
                        }
                elif "spm" in tool_name:
                    spm_results.append(result)

            # 优先使用FreeSurfer
            if freesurfer_result:
                print(f"  [选择] 使用FreeSurfer数据")
                return self._extract_freesurfer_data(freesurfer_result, run_dir)

            # 没有FreeSurfer时使用SPM（继续到下方SPM处理逻辑）
            if spm_results:
                print(f"  [选择] 使用SPM数据")
                # SPM数据将在下方的默认逻辑中处理

        # ========== 无指定模态或指定模态数据不足时使用默认优先级 ==========
        # 策略: DTI数据优先（包含FA/MD等扩散指标），然后FreeSurfer，最后SPM
        # 注意: 如果已指定anat模态且有anat数据，跳过DTI检测
        skip_dti_check = (target_modality == "anat" and results_by_modality.get("anat"))

        if target_modality and not results_by_modality.get(target_modality):
            print(f"  [警告] 未找到 {target_modality.upper()} 模态数据，使用默认优先级")
        elif not target_modality:
            print(f"  [回退] 使用默认数据源优先级")
        elif skip_dti_check:
            print(f"  [流程] 已指定anat模态，跳过DTI检测，直接处理解剖结构数据")

        # 1. 检测FSL DTI输出（优先级最高 - 包含FA/MD等扩散指标）
        # 但如果已指定anat模态，跳过此步骤
        if not skip_dti_check:
            has_fsl_dti = False
            dti_files = {"FA": [], "MD": [], "RD": [], "AD": [], "L1": [], "L2": [], "L3": []}

            for result in tool_results:
                if not isinstance(result, dict):
                    continue

                outputs = result.get("outputs", {}) or result.get("result", {})

                if isinstance(outputs, dict):
                    output_files = outputs.get("output_files", [])
                else:
                    output_files = []

                # 检测DTI指标文件
                for f in output_files:
                    if not isinstance(f, str):
                        continue
                    fname = Path(f).name
                    for metric in dti_files.keys():
                        if f"_{metric}" in fname and fname.endswith('.nii.gz'):
                            dti_files[metric].append(f)
                            has_fsl_dti = True

            # 如果是FSL DTI输出，优先使用DTI数据提取逻辑
            if has_fsl_dti:
                print(f"  [检测] 发现FSL DTI输出（优先使用扩散指标数据）")
                for metric, files in dti_files.items():
                    if files:
                        print(f"    - {metric}: {len(files)} 个文件")
                return self._extract_fsl_dti_data(dti_files, run_dir)

        # 2. 检测FreeSurfer输出（次优先级 - 结构体积数据）
        has_freesurfer = False
        freesurfer_info = {}

        for result in tool_results:
            if not isinstance(result, dict):
                continue

            tool_name = result.get("tool_name", "")
            if "freesurfer" in tool_name.lower():
                outputs = result.get("outputs", {})
                subjects_dir = outputs.get("subjects_dir", "")
                processed_subjects = outputs.get("processed_subjects", [])

                if subjects_dir and processed_subjects:
                    has_freesurfer = True
                    freesurfer_info = {
                        "subjects_dir": subjects_dir,
                        "processed_subjects": processed_subjects
                    }
                    print(f"  [检测] 发现FreeSurfer输出: {len(processed_subjects)} 个被试")
                    print(f"  [检测] SUBJECTS_DIR: {subjects_dir}")
                    break

        # 如果是FreeSurfer输出，使用数据提取脚本
        if has_freesurfer:
            return self._extract_freesurfer_data(freesurfer_info, run_dir)

        # ========== 否则使用原有的SPM NIfTI处理逻辑 ==========
        # 查找分割后的灰质/白质文件
        gm_files = []  # 灰质文件 (c1*, wc1*, swc1*)
        wm_files = []  # 白质文件 (c2*, wc2*, swc2*)

        # 从tool_results提取文件
        for result in tool_results:
            if not isinstance(result, dict):
                continue

            # 检查outputs
            outputs = result.get("outputs", {})
            if isinstance(outputs, dict):
                output_files = outputs.get("output_files", [])
            else:
                output_files = []

            # 检查result字段
            task_result = result.get("result", {})
            if isinstance(task_result, dict):
                output_files.extend(task_result.get("output_files", []))

            for f in output_files:
                if not isinstance(f, str):
                    continue
                fname = Path(f).name.lower()
                # 优先使用标准化后的文件（wc1/wc2），因为它们与MNI空间对齐
                if fname.startswith('wc1') or fname.startswith('swc1') or fname.startswith('c1'):
                    gm_files.append(f)
                elif fname.startswith('wc2') or fname.startswith('swc2') or fname.startswith('c2'):
                    wm_files.append(f)

        # 按优先级排序（wc > swc > c，因为wc是标准化后的，与图谱空间一致）
        def sort_priority(f):
            fname = Path(f).name.lower()
            if fname.startswith('wc'):
                return 0  # 标准化后的最优先
            elif fname.startswith('swc'):
                return 1  # 平滑后的次优先
            else:
                return 2  # 原始分割

        gm_files = sorted(set(gm_files), key=sort_priority)
        wm_files = sorted(set(wm_files), key=sort_priority)

        if not gm_files:
            print("  [警告] 未找到灰质分割文件")
            return {"success": False, "error": "未找到灰质分割文件", "data_file": None}

        print(f"  [数据准备] 找到 {len(gm_files)} 个灰质文件, {len(wm_files)} 个白质文件")

        # 选择最高优先级的文件类型
        best_gm_prefix = Path(gm_files[0]).name[:3].lower() if gm_files else ""
        best_wm_prefix = Path(wm_files[0]).name[:3].lower() if wm_files else ""

        print(f"  [数据准备] 使用文件类型: 灰质={best_gm_prefix}*, 白质={best_wm_prefix}*")

        # 只使用同一类型的文件
        gm_to_use = [f for f in gm_files if Path(f).name.lower().startswith(best_gm_prefix)]
        wm_to_use = [f for f in wm_files if Path(f).name.lower().startswith(best_wm_prefix)]

        # 加载脑图谱
        atlases = self._load_brain_atlases(SCA3_PRIORITY_ATLASES)
        if atlases:
            print(f"  [数据准备] 已加载 {len(atlases)} 个脑图谱")
            for name, info in atlases.items():
                print(f"    - {name}: {len(info['labels'])} 个脑区")

        # 提取体积数据
        data_records = []
        processed_subjects = set()  # 用于去重：跟踪已处理的被试ID

        for gm_file in gm_to_use:
            try:
                # 提取被试ID和组别
                fname = Path(gm_file).stem
                # 【修复】移除.nii后缀（防止.nii.gz文件的stem保留.nii）
                if fname.endswith('.nii'):
                    fname = fname[:-4]

                # 移除前缀 (c1, wc1, swc1等)
                subject_part = fname
                for prefix in ['swc1', 'wc1', 'c1', 'swc2', 'wc2', 'c2']:
                    if subject_part.lower().startswith(prefix):
                        subject_part = subject_part[len(prefix):]
                        break

                subject_id = subject_part

                # 去重检查：如果该被试已处理，跳过
                if subject_id in processed_subjects:
                    continue
                processed_subjects.add(subject_id)

                # 确定组别
                if 'HC' in subject_id.upper() or 'CON' in subject_id.upper():
                    group = 'HC'
                elif 'SCA3' in subject_id.upper() or 'PAT' in subject_id.upper():
                    group = 'SCA3'
                else:
                    group = 'Unknown'

                # 初始化记录
                record = {
                    'subject_id': subject_id,
                    'group': group
                }

                # 加载灰质图像
                try:
                    gm_img = nib.load(gm_file)
                    gm_data = gm_img.get_fdata()
                    voxel_vol = np.prod(gm_img.header.get_zooms()[:3])  # mm³
                    gm_volume = np.sum(gm_data > 0.1) * voxel_vol / 1000  # mL
                    record['gray_matter_volume'] = round(gm_volume, 2)
                except Exception as e:
                    print(f"  [警告] 无法读取 {Path(gm_file).name}: {e}")
                    record['gray_matter_volume'] = 0
                    gm_data = None

                # 计算白质体积
                wm_volume = 0
                wm_file = None
                for wf in wm_to_use:
                    wf_stem = Path(wf).stem
                    if wf_stem == subject_id or wf_stem.startswith(subject_id + '_') or ('_' + subject_id + '_') in wf_stem or wf_stem.endswith('_' + subject_id):
                        wm_file = wf
                        break

                if wm_file:
                    try:
                        wm_img = nib.load(wm_file)
                        wm_data = wm_img.get_fdata()
                        voxel_vol = np.prod(wm_img.header.get_zooms()[:3])
                        wm_volume = np.sum(wm_data > 0.1) * voxel_vol / 1000
                    except Exception as e:
                        print(f"  [WARNING] 加载白质文件失败 ({subject_id}): {e}")

                record['white_matter_volume'] = round(wm_volume, 2)
                record['total_brain_volume'] = round(record['gray_matter_volume'] + wm_volume, 2)

                # 使用脑图谱提取各脑区体积
                if atlases and gm_data is not None:
                    region_volumes = self._extract_roi_volumes(
                        gm_data, gm_img.affine, atlases, voxel_vol
                    )
                    record.update(region_volumes)

                data_records.append(record)

            except Exception as e:
                print(f"  [警告] 处理文件失败 {gm_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not data_records:
            print("  [错误] 无法提取任何数据")
            return {"success": False, "error": "无法提取数据", "data_file": None}

        # 创建DataFrame
        df = pd.DataFrame(data_records)

        # 合并人口统计学数据
        df = self._merge_demographics_data(df)

        # 保存合并后的数据
        data_file = run_dir / "brain_region_data.csv"
        df.to_csv(data_file, index=False, encoding='utf-8')

        # 统计提取的脑区数量
        region_cols = [c for c in df.columns if c not in ['subject_id', 'group', 'gray_matter_volume',
                                                           'white_matter_volume', 'total_brain_volume']]

        print(f"  [数据准备] 完成! 保存到 {data_file.name}")
        print(f"  [数据准备] 共 {len(df)} 个被试, 组别分布: {df['group'].value_counts().to_dict()}")
        print(f"  [数据准备] 提取了 {len(region_cols)} 个脑区体积指标")

        return {
            "success": True,
            "data_file": str(data_file),
            "n_subjects": len(df),
            "groups": df['group'].unique().tolist(),
            "columns": df.columns.tolist(),
            "n_regions": len(region_cols),
            "summary": {
                "gray_matter": {
                    "mean": round(df['gray_matter_volume'].mean(), 2),
                    "std": round(df['gray_matter_volume'].std(), 2)
                },
                "white_matter": {
                    "mean": round(df['white_matter_volume'].mean(), 2),
                    "std": round(df['white_matter_volume'].std(), 2)
                }
            }
        }

    def _merge_demographics_data(self, df, demographics_file: Path = None):
        """
        合并人口统计学和量表数据到影像数据DataFrame

        Args:
            df: 影像数据DataFrame（必须包含subject_id列）
            demographics_file: 人口统计学数据文件路径（默认为data/data.xlsx）

        Returns:
            合并后的DataFrame
        """
        import pandas as pd
        from src.config import DATA_DIR

        if demographics_file is None:
            demographics_file = DATA_DIR / "data.xlsx"

        if not demographics_file.exists():
            print(f"  [提示] 未找到人口统计学数据文件: {demographics_file}")
            return df

        try:
            # 读取人口统计学数据
            demo_df = pd.read_excel(demographics_file)

            # 识别ID列
            id_column = None
            for col in demo_df.columns:
                if col.lower() in ['id', 'subject', 'subjectid', 'subject_id']:
                    id_column = col
                    break

            if id_column is None:
                print(f"  [警告] 人口统计学数据文件中未找到ID列")
                return df

            # 标准化ID列名为subject_id
            if id_column != 'subject_id':
                demo_df = demo_df.rename(columns={id_column: 'subject_id'})

            # 标准化subject_id格式（移除空格、统一大小写等）
            df['subject_id'] = df['subject_id'].astype(str).str.strip()
            demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.strip()

            # 合并数据（左连接，保留所有影像数据记录）
            original_cols = df.columns.tolist()
            merged_df = pd.merge(df, demo_df, on='subject_id', how='left', suffixes=('', '_demo'))

            # 处理group列冲突（如果人口统计学数据也有group列）
            if 'group_demo' in merged_df.columns:
                # 优先使用人口统计学数据的分组信息
                merged_df['group'] = merged_df['group_demo'].fillna(merged_df['group'])
                merged_df = merged_df.drop(columns=['group_demo'])

            # 统计合并成功的记录数
            n_matched = merged_df[demo_df.columns[demo_df.columns != 'subject_id']].notna().any(axis=1).sum()
            n_total = len(merged_df)

            print(f"  [数据合并] 成功合并人口统计学数据: {n_matched}/{n_total} 个被试匹配")

            # 显示添加的新列
            new_cols = [c for c in merged_df.columns if c not in original_cols]
            if new_cols:
                print(f"  [数据合并] 添加的变量: {', '.join(new_cols)}")

            return merged_df

        except Exception as e:
            print(f"  [警告] 合并人口统计学数据失败: {e}")
            import traceback
            traceback.print_exc()
            return df

    def _extract_fsl_dti_data(self, dti_files: Dict[str, List[str]], run_dir: Path) -> Dict[str, Any]:
        """
        从FSL dtifit输出提取DTI指标数据

        Args:
            dti_files: DTI指标文件字典 {metric: [file_paths]}
            run_dir: 运行目录

        Returns:
            包含数据文件路径和元信息的字典
        """
        import pandas as pd
        import nibabel as nib
        import numpy as np
        from src.config import BRAIN_ATLASES, SCA3_PRIORITY_ATLASES

        print("  [DTI数据提取] 从FSL dtifit输出提取数据...")

        # 确定被试列表（从FA文件获取）
        fa_files = dti_files.get("FA", [])
        if not fa_files:
            print("  [错误] 未找到FA文件")
            return {"success": False, "error": "未找到FA文件", "data_file": None}

        # 加载脑图谱（用于计算ROI平均值）
        atlases = self._load_brain_atlases(SCA3_PRIORITY_ATLASES)
        if atlases:
            print(f"  [DTI数据提取] 已加载 {len(atlases)} 个脑图谱")

        # 提取数据
        data_records = []

        for fa_file in fa_files:
            try:
                # 提取被试ID和组别
                fname = Path(fa_file).stem  # 如 HC1_0001_dti_FA.nii (Path.stem只移除最后一个后缀.gz)
                # 【修复】移除.nii后缀（因为.nii.gz文件的stem会保留.nii）
                if fname.endswith('.nii'):
                    fname = fname[:-4]  # 现在 fname = HC1_0001_dti_FA

                # 移除DTI后缀
                subject_part = fname
                for suffix in ['_FA', '_MD', '_RD', '_AD', '_L1', '_L2', '_L3', '_dti']:
                    subject_part = subject_part.replace(suffix, '')

                subject_id = subject_part

                # 确定组别
                if 'HC' in subject_id.upper() or 'CON' in subject_id.upper():
                    group = 'HC'
                elif 'SCA3' in subject_id.upper() or 'PAT' in subject_id.upper():
                    group = 'SCA3'
                else:
                    group = 'Unknown'

                # 初始化记录
                record = {
                    'subject_id': subject_id,
                    'group': group
                }

                # 加载各DTI指标
                for metric, metric_files in dti_files.items():
                    # 找到当前被试对应的文件
                    metric_file = None
                    for mf in metric_files:
                        mf_stem = Path(mf).stem
                        if mf_stem == subject_part or mf_stem.startswith(subject_part + '_') or ('_' + subject_part + '_') in mf_stem or mf_stem.endswith('_' + subject_part) or mf_stem == subject_id or mf_stem.startswith(subject_id + '_'):
                            metric_file = mf
                            break

                    if metric_file and Path(metric_file).exists():
                        try:
                            img = nib.load(metric_file)
                            data = img.get_fdata()

                            # 计算全脑平均值（排除背景和异常值）
                            mask = (data > 0) & ((data < 1) if metric == "FA" else (data < 0.01))
                            if mask.sum() > 0:
                                mean_val = np.mean(data[mask])
                                record[f'{metric}_mean'] = round(float(mean_val), 6)
                            else:
                                record[f'{metric}_mean'] = np.nan

                            # 如果有图谱，计算各ROI的平均值
                            if atlases and metric in ["FA", "MD", "RD", "AD"]:
                                roi_values = self._extract_dti_roi_values(
                                    data, img.affine, atlases, metric
                                )
                                record.update(roi_values)

                        except Exception as e:
                            print(f"  [警告] 无法读取 {Path(metric_file).name}: {e}")
                            record[f'{metric}_mean'] = np.nan
                    else:
                        record[f'{metric}_mean'] = np.nan

                # **关键修复**: FSL dtifit不输出RD和AD，需要从L1/L2/L3计算
                # AD (Axial Diffusivity) = L1
                # RD (Radial Diffusivity) = (L2 + L3) / 2
                if 'RD_mean' not in record or pd.isna(record.get('RD_mean')):
                    l2_val = record.get('L2_mean')
                    l3_val = record.get('L3_mean')
                    if l2_val is not None and l3_val is not None and not (np.isnan(l2_val) or np.isnan(l3_val)):
                        record['RD_mean'] = round((l2_val + l3_val) / 2, 6)
                        print(f"  [计算] {subject_id}: RD_mean 从 L2/L3 计算")

                if 'AD_mean' not in record or pd.isna(record.get('AD_mean')):
                    l1_val = record.get('L1_mean')
                    if l1_val is not None and not np.isnan(l1_val):
                        record['AD_mean'] = round(l1_val, 6)
                        print(f"  [计算] {subject_id}: AD_mean = L1_mean")

                data_records.append(record)
                fa_val = record.get('FA_mean', None)
                fa_str = f"{fa_val:.4f}" if fa_val is not None and not np.isnan(fa_val) else "N/A"
                print(f"  [OK] {subject_id}: FA={fa_str}")

            except Exception as e:
                print(f"  [警告] 处理文件失败 {fa_file}: {e}")
                continue

        if not data_records:
            print("  [错误] 无法提取任何DTI数据")
            return {"success": False, "error": "无法提取DTI数据", "data_file": None}

        # 创建DataFrame
        df = pd.DataFrame(data_records)

        # 合并人口统计学数据
        df = self._merge_demographics_data(df)

        # 保存数据
        data_file = run_dir / "dti_metrics_data.csv"
        df.to_csv(data_file, index=False, encoding='utf-8')

        # 统计
        metric_cols = [c for c in df.columns if ('_mean' in c or '_' in c) and c not in ['subject_id', 'group']]

        print(f"  [DTI数据提取] 完成! 保存到 {data_file.name}")
        print(f"  [DTI数据提取] 共 {len(df)} 个被试, 组别分布: {df['group'].value_counts().to_dict()}")

        return {
            "success": True,
            "data_file": str(data_file),
            "n_subjects": len(df),
            "groups": df['group'].unique().tolist(),
            "columns": df.columns.tolist(),
            "summary": {
                "FA_mean": round(df['FA_mean'].mean(), 4) if 'FA_mean' in df.columns else None,
                "MD_mean": round(df['MD_mean'].mean(), 6) if 'MD_mean' in df.columns else None
            }
        }

    def _extract_dti_roi_values(self, data: np.ndarray, affine: np.ndarray,
                                 atlases: Dict, metric: str) -> Dict[str, float]:
        """
        使用脑图谱提取DTI指标的ROI平均值

        Args:
            data: DTI指标数据（3D数组）
            affine: 仿射变换矩阵
            atlases: 脑图谱字典
            metric: DTI指标名称（FA, MD等）

        Returns:
            各ROI的平均值字典
        """
        import nibabel as nib
        from scipy import ndimage
        import numpy as np

        roi_values = {}

        for atlas_name, atlas_info in atlases.items():
            atlas_data = atlas_info.get('data')
            labels = atlas_info.get('labels', {})

            if atlas_data is None:
                continue

            # 确保数据形状一致（可能需要重采样）
            if atlas_data.shape != data.shape:
                # 简单处理：跳过形状不匹配的图谱
                continue

            # 提取各ROI的平均值
            for label_id, label_name in labels.items():
                if label_id == 0:  # 跳过背景
                    continue

                roi_mask = (atlas_data == label_id)
                if roi_mask.sum() == 0:
                    continue

                # 计算ROI内的平均值
                roi_data = data[roi_mask]
                # 过滤无效值
                valid_mask = (roi_data > 0) & np.isfinite(roi_data)
                if metric == "FA":
                    valid_mask = valid_mask & (roi_data < 1)
                elif metric in ["MD", "RD", "AD"]:
                    valid_mask = valid_mask & (roi_data < 0.01)

                if valid_mask.sum() > 0:
                    mean_val = np.mean(roi_data[valid_mask])
                    # 列名格式：metric_atlasname_roi
                    col_name = f"{metric}_{label_name.replace(' ', '_').replace('-', '_')}"
                    roi_values[col_name] = round(float(mean_val), 6)

        return roi_values

    def _extract_freesurfer_data(self, freesurfer_info: Dict, run_dir: Path) -> Dict[str, Any]:
        """
        从FreeSurfer输出提取数据

        Args:
            freesurfer_info: FreeSurfer信息（subjects_dir, processed_subjects）
            run_dir: 运行目录

        Returns:
            包含数据文件路径和元信息的字典
        """
        import pandas as pd

        print("  [FreeSurfer数据提取] 生成并运行数据提取脚本...")

        subjects_dir = freesurfer_info["subjects_dir"]
        subject_ids = freesurfer_info["processed_subjects"]

        # 创建输出目录
        output_dir = run_dir / "freesurfer_extracted_data"
        output_dir.mkdir(exist_ok=True)

        # 使用generate_data_extraction_code生成并运行脚本
        extraction_result = self.generate_data_extraction_code(
            subjects_dir=subjects_dir,
            subject_ids=subject_ids,
            extraction_type="subcortical_volumes",  # 提取皮下结构体积
            output_dir=output_dir,
            run_dir=run_dir,
            task_description="从FreeSurfer stats文件中提取皮下结构体积（海马、杏仁核、基底节、丘脑、小脑等）"
        )

        if not extraction_result.get("success"):
            error_msg = extraction_result.get("error", "未知错误")
            print(f"  [错误] FreeSurfer数据提取失败: {error_msg}")
            return {"success": False, "error": f"FreeSurfer数据提取失败: {error_msg}", "data_file": None}

        # 查找生成的CSV文件
        output_files = extraction_result.get("output_files", [])
        csv_file = None
        for f in output_files:
            if f.endswith('.csv'):
                csv_file = f
                print(f"  [成功] 数据已提取到: {csv_file}")
                break

        if not csv_file:
            print(f"  [错误] 未找到提取的CSV文件")
            return {"success": False, "error": "未找到提取的CSV文件", "data_file": None}

        # 读取CSV获取摘要信息
        try:
            df = pd.read_csv(csv_file)

            # 合并人口统计学数据
            df = self._merge_demographics_data(df)

            # 保存合并后的数据（覆盖原文件）
            df.to_csv(csv_file, index=False, encoding='utf-8')

            # 统计脑区数量（排除subject_id和group列以及人口统计学列）
            demo_cols = ['age', 'sex', 'gender', 'education', 'moca', 'mmse', 'updrs']  # 常见人口统计学列
            region_cols = [c for c in df.columns if c not in ['subject_id', 'group', 'Subject'] and c.lower() not in demo_cols]

            print(f"  [数据摘要] 共 {len(df)} 个被试")
            if 'group' in df.columns:
                print(f"  [数据摘要] 组别分布: {df['group'].value_counts().to_dict()}")
            print(f"  [数据摘要] 提取了 {len(region_cols)} 个脑区指标")

            return {
                "success": True,
                "data_file": csv_file,
                "n_subjects": len(df),
                "groups": df['group'].unique().tolist() if 'group' in df.columns else [],
                "columns": df.columns.tolist(),
                "n_regions": len(region_cols),
                "summary": {}  # FreeSurfer数据不需要计算总体积摘要
            }
        except Exception as e:
            print(f"  [错误] 读取CSV文件失败: {e}")
            return {"success": False, "error": f"读取CSV文件失败: {str(e)}", "data_file": None}

    def _load_brain_atlases(self, atlas_names: List[str]) -> Dict[str, Any]:
        """
        加载脑图谱和标签文件

        Args:
            atlas_names: 要加载的图谱名称列表

        Returns:
            包含图谱数据和标签的字典
        """
        import nibabel as nib
        from src.config import BRAIN_ATLASES

        atlases = {}

        for name in atlas_names:
            if name not in BRAIN_ATLASES:
                continue

            atlas_config = BRAIN_ATLASES[name]
            atlas_file = atlas_config["atlas_file"]
            label_file = atlas_config["label_file"]

            if not atlas_file.exists() or not label_file.exists():
                print(f"  [警告] 图谱文件不存在: {name}")
                continue

            try:
                # 加载图谱NIfTI
                atlas_img = nib.load(str(atlas_file))
                atlas_data = atlas_img.get_fdata()

                # 加载标签文件
                labels = {}
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            try:
                                label_id = int(parts[0])
                                label_name = parts[1].strip()
                                labels[label_id] = label_name
                            except ValueError:
                                continue

                atlases[name] = {
                    "data": atlas_data,
                    "affine": atlas_img.affine,
                    "labels": labels,
                    "description": atlas_config["description"]
                }

            except Exception as e:
                print(f"  [警告] 加载图谱 {name} 失败: {e}")
                continue

        return atlases

    def _extract_roi_volumes(
        self,
        gm_data: np.ndarray,
        gm_affine: np.ndarray,
        atlases: Dict[str, Any],
        voxel_vol: float
    ) -> Dict[str, float]:
        """
        使用脑图谱提取各脑区的灰质体积

        Args:
            gm_data: 灰质概率图数据
            gm_affine: 灰质图像的仿射矩阵
            atlases: 脑图谱字典
            voxel_vol: 体素体积 (mm³)

        Returns:
            各脑区体积的字典
        """
        import numpy as np
        from scipy import ndimage

        region_volumes = {}

        for atlas_name, atlas_info in atlases.items():
            atlas_data = atlas_info["data"]
            labels = atlas_info["labels"]

            # 检查图谱和灰质图像尺寸是否匹配
            if atlas_data.shape != gm_data.shape:
                # 需要重采样图谱到灰质图像空间
                try:
                    atlas_data = self._resample_atlas(
                        atlas_data, atlas_info["affine"],
                        gm_data.shape, gm_affine
                    )
                except Exception as e:
                    print(f"  [警告] 图谱 {atlas_name} 重采样失败: {e}")
                    continue

            # 计算每个脑区的灰质体积
            for label_id, label_name in labels.items():
                # 创建脑区mask
                roi_mask = (atlas_data == label_id)

                if not np.any(roi_mask):
                    continue

                # 计算该脑区内的灰质体积（概率加权）
                # 使用灰质概率图与ROI mask的乘积
                roi_gm = gm_data * roi_mask

                # 体积 = 概率值之和 * 体素体积
                volume_mm3 = np.sum(roi_gm) * voxel_vol
                volume_ml = volume_mm3 / 1000  # 转换为mL

                # 生成列名（图谱名_脑区名）
                col_name = f"{atlas_name}_{label_name}"
                # 清理列名中的特殊字符
                col_name = col_name.replace(' ', '_').replace('-', '_').replace('/', '_')

                region_volumes[col_name] = round(volume_ml, 4)

        return region_volumes

    def _resample_atlas(
        self,
        atlas_data: np.ndarray,
        atlas_affine: np.ndarray,
        target_shape: tuple,
        target_affine: np.ndarray
    ) -> np.ndarray:
        """
        将图谱重采样到目标空间

        Args:
            atlas_data: 图谱数据
            atlas_affine: 图谱仿射矩阵
            target_shape: 目标形状
            target_affine: 目标仿射矩阵

        Returns:
            重采样后的图谱数据
        """
        import numpy as np
        from scipy import ndimage

        # 计算从目标空间到图谱空间的变换
        # target_voxel -> world -> atlas_voxel
        target_to_world = target_affine
        world_to_atlas = np.linalg.inv(atlas_affine)
        target_to_atlas = world_to_atlas @ target_to_world

        # 生成目标空间的坐标网格
        coords = np.meshgrid(
            np.arange(target_shape[0]),
            np.arange(target_shape[1]),
            np.arange(target_shape[2]),
            indexing='ij'
        )
        coords = np.array(coords).reshape(3, -1)
        # 添加齐次坐标
        coords_homo = np.vstack([coords, np.ones(coords.shape[1])])

        # 变换到图谱空间
        atlas_coords = target_to_atlas @ coords_homo
        atlas_coords = atlas_coords[:3, :]

        # 使用最近邻插值（保持标签值）
        resampled = ndimage.map_coordinates(
            atlas_data,
            atlas_coords,
            order=0,  # 最近邻插值
            mode='constant',
            cval=0
        )

        return resampled.reshape(target_shape)

    def _build_code_generation_prompt(
        self,
        plan: Dict[str, Any],
        data_indicators: Dict[str, Any],
        task_description: str,
        brain_region_suggestions: Dict[str, Any] = None,
        tool_results: List[Dict] = None
    ) -> str:
        """构建代码生成提示"""

        # 提取关键信息
        research_question = plan.get("research_question", "")
        methods = plan.get("methods", [])
        expected_output = plan.get("expected_output", "")

        # 提取ROI选择策略（来自plan或brain_region_suggestions）
        roi_selection = plan.get("roi_selection", {})
        if not roi_selection and brain_region_suggestions:
            roi_selection = brain_region_suggestions

        # 提取数据指标摘要
        brain_regions = data_indicators.get("brain_regions", [])
        metrics = data_indicators.get("metrics", [])
        data_files = data_indicators.get("data_files", [])

        # 获取准备好的输入数据文件
        input_data_file = data_indicators.get("input_data_file", "")
        data_summary = data_indicators.get("data_summary", {})
        n_subjects = data_indicators.get("n_subjects", 0)
        groups = data_indicators.get("groups", [])

        # 检测上游工具并获取格式文档
        detected_tools = self._detect_upstream_tools(tool_results or [])
        tool_docs = ""
        if detected_tools:
            tool_docs = "\n\n## 上游工具输出格式参考\n\n"
            tool_docs += "以下是上游工具的输出格式说明，供理解数据来源和结构：\n\n"
            for tool in detected_tools[:2]:  # 最多显示2个工具文档，避免prompt过长
                tool_docs += self._get_tool_output_documentation(tool)
                tool_docs += "\n\n---\n\n"

        # 构建数据说明
        data_description = ""
        if input_data_file:
            # 读取CSV文件获取实际的列信息
            try:
                import pandas as pd
                temp_df = pd.read_csv(input_data_file, nrows=0)  # 只读取列名
                actual_columns = temp_df.columns.tolist()

                # 识别人口统计学列（排除影像指标列）
                imaging_cols = {'subject_id', 'group', 'gray_matter_volume', 'white_matter_volume',
                               'total_brain_volume', 'gm_file', 'wm_file'}
                demographics_cols = [c for c in actual_columns if (c not in imaging_cols
                                    and not c.startswith('Left_') and not c.startswith('Right_')
                                    and not c.startswith('lh_') and not c.startswith('rh_')
                                    and '_' not in c.lower()) or c.lower() in ['age', 'sex', 'gender', 'education', 'moca', 'mmse']]

                demographics_info = ""
                if demographics_cols:
                    demographics_info = f"""

**人口统计学和量表数据（已合并）**:
{', '.join(demographics_cols)}

**重要提示**:
- 这些人口统计学变量可以用作协变量进行统计分析（如ANCOVA）
- 可以分析临床量表与影像指标的相关性
- 在组间比较时应考虑人口统计学变量的匹配性
"""

            except Exception as e:
                demographics_info = ""
                actual_columns = ['subject_id', 'group', 'gray_matter_volume', 'white_matter_volume',
                                 'total_brain_volume', 'gm_file', 'wm_file']

            # 根据数据类型构建摘要
            if 'FA_mean' in data_summary or 'dti' in input_data_file.lower():
                # DTI数据摘要
                imaging_summary = f"""**DTI数据摘要**:
- FA均值: {data_summary.get('FA_mean', 'N/A')}
- MD均值: {data_summary.get('MD_mean', 'N/A')}"""
            else:
                # VBM数据摘要
                imaging_summary = f"""**影像数据摘要**:
- 灰质体积: 均值={data_summary.get('gray_matter', {}).get('mean', 'N/A')} mL, 标准差={data_summary.get('gray_matter', {}).get('std', 'N/A')} mL
- 白质体积: 均值={data_summary.get('white_matter', {}).get('mean', 'N/A')} mL, 标准差={data_summary.get('white_matter', {}).get('std', 'N/A')} mL"""

            data_description = f"""
### 已准备的输入数据文件（重要！）
**文件路径**: `{input_data_file}`
**被试数量**: {n_subjects}
**组别**: {', '.join(groups)}
**数据列**: {', '.join(actual_columns[:15])}{'...' if len(actual_columns) > 15 else ''}
{demographics_info}
{imaging_summary}

## 数据加载强制要求 (CRITICAL)

**必须使用以下模板加载数据**:
```python
import pandas as pd

# 强制使用预处理好的CSV文件
DATA_FILE = r'{input_data_file}'

def load_prepared_data():
    \"\"\"Load pre-extracted data from CSV file (MANDATORY)\"\"\"
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"Required data file not found: {{DATA_FILE}}")

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {{len(df)}} records from {{DATA_FILE}}")
    return df
```

**绝对禁止** (违反将导致分析结果无效):
1. ❌ 禁止直接从NIfTI文件计算数据 - 数据已经提取好在CSV中
2. ❌ 禁止对所有ROI使用相同的全脑均值（如 `np.mean(data)` 然后复制给所有ROI）
3. ❌ 禁止设置 ROI mask = None 然后跳过真实提取
4. ❌ 禁止在循环中给所有ROI赋相同值

**数据验证要求**:
- 代码运行后，不同ROI的统计值必须不同（如果所有ROI的mean完全相同，说明代码有bug）
- 必须从CSV的各列直接读取每个ROI的值，而不是重新计算
{tool_docs}
"""
        else:
            data_description = f"""
### 数据文件
{json.dumps(data_files[:5], ensure_ascii=False, indent=2) if data_files else "无具体文件路径"}
{tool_docs}
"""

        # 构建ROI选择策略描述
        roi_strategy_description = ""
        if roi_selection:
            strategy = roi_selection.get("strategy") or roi_selection.get("analysis_priority", "exploratory")
            primary_rois = roi_selection.get("primary_rois", [])
            secondary_rois = roi_selection.get("secondary_rois", [])
            primary_rationale = roi_selection.get("primary_rationale", "")
            expected_findings = roi_selection.get("expected_findings", "")

            if primary_rois:
                roi_strategy_description = f"""
## 脑区选择策略（重要！基于疾病-脑区映射）

**分析策略**: {strategy}

**Primary ROIs（主要分析区域，必须优先分析）**:
{json.dumps(primary_rois, ensure_ascii=False, indent=2)}
理由: {primary_rationale}

**Secondary ROIs（次要分析区域）**:
{json.dumps(secondary_rois, ensure_ascii=False, indent=2) if secondary_rois else "无"}

**预期发现**:
{expected_findings if expected_findings else "根据疾病病理学，预期在Primary ROIs中观察到显著差异"}

**分析要求**:
1. **对Primary ROIs进行详细分析**：
   - 计算每个区域的描述性统计量（均值、标准差、范围）
   - 进行组间比较（t检验或非参数检验）
   - 计算效应量（Cohen's d）和95%置信区间
   - 生成专门的可视化图表（箱线图、小提琴图）

2. **对Secondary ROIs进行探索性分析**：
   - 快速筛查是否有异常
   - 记录但不深入解释

3. **全脑筛查（如果是exploratory策略）**：
   - 对所有可用ROI进行FDR校正的多重比较
   - 按效应量排序输出Top 10区域

4. **结果呈现**：
   - Primary ROIs的结果应该单独成节，详细报告
   - 生成Primary ROIs的热力图或脑区可视化
"""

        prompt = f"""
# 神经影像分析算法代码生成任务

## 研究背景
研究问题：{research_question}
分析方法：{', '.join(methods) if methods else '统计分析'}
预期输出：{expected_output}
{roi_strategy_description}
## 可用数据资源
{data_description}

### 脑区指标
{json.dumps(brain_regions[:10], ensure_ascii=False, indent=2) if brain_regions else "从CSV文件中读取"}

### 计算指标
{json.dumps(metrics[:10], ensure_ascii=False, indent=2) if metrics else ("FA_mean, MD_mean, RD_mean, AD_mean" if 'dti' in (input_data_file or '').lower() else "gray_matter_volume, white_matter_volume, total_brain_volume")}

## 代码生成要求

### 绝对禁止（违反此规则将导致分析无效）
1. **禁止生成任何模拟数据**：不允许使用np.random生成脑区数据、不允许创建demo数据函数
2. **禁止使用use_demo_data配置**：不允许任何fallback到模拟数据的逻辑
3. **文件不存在必须报错**：如果输入文件不存在，直接raise FileNotFoundError，不能继续

### 基本要求
1. **完整性**：包含所有必要的import语句，代码可直接运行
2. **数据读取**：**必须使用上述提供的CSV文件路径**，直接用pandas.read_csv读取
3. **分析方法**：根据研究计划实现相应的统计分析方法
4. **结果输出**：生成清晰的分析结果（数值+可视化）
5. **真实数据**：所有分析必须基于工具输出的真实数据，不允许任何模拟或伪造

### 技术规范
1. **库选择**：
   - 数据处理：pandas, numpy
   - 统计分析：scipy.stats, statsmodels
   - 可视化：matplotlib, seaborn

2. **配置参数（重要！必须使用绝对路径）**：
   ```python
   CONFIG = {{
       'input_file': r'{input_data_file if input_data_file else "brain_region_data.csv"}',  # 输入文件绝对路径
       'output_dir': r'{str(Path(input_data_file).parent / "analysis_results") if input_data_file else "analysis_results"}',  # 输出目录绝对路径
       ...
   }}
   ```
   **注意**：output_dir必须使用绝对路径，确保图表保存到正确位置！

3. **代码结构**：
   ```python
   # 1. 导入必要的库
   # 2. 定义配置参数（使用上述实际路径）
   # 3. 数据加载函数
   # 4. 数据预处理函数
   # 5. 统计分析函数
   # 6. 结果可视化函数
   # 7. 主函数（调用上述函数）
   # 8. 执行入口
   ```

4. **错误处理**：
   - 检查文件是否存在
   - 验证数据格式和维度
   - 捕获并记录异常信息

5. **输出规范**：
   - 打印分析步骤和中间结果
   - 保存统计结果到JSON/CSV文件
   - **必须保存所有生成的可视化图表**：
     * 使用 `plt.savefig(output_path, dpi=300, bbox_inches='tight')` 保存每个图表
     * 调用 `plt.tight_layout()` 优化布局后再保存
     * 图表文件名应具有描述性（如 'group_comparison_boxplot.png'）
     * 保存后使用 `plt.close()` 释放内存
     * 图表格式：PNG（推荐300 DPI）
   - **所有输出必须保存到output_dir指定的绝对路径目录**

6. **多重比较校正**：
   - **推荐使用statsmodels**：`from statsmodels.stats.multitest import multipletests`
   - 调用方式：`reject, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')`
   - **不要手动实现FDR校正算法**，容易出错

### 特定分析建议
{self._get_analysis_suggestions(research_question, methods)}

## 输出格式
请直接输出完整的Python代码，使用```python代码块包裹。
代码应该是生产级别的，包含完整的文档字符串和注释。
"""
        return prompt

    def _get_analysis_suggestions(self, research_question: str, methods: List[str]) -> str:
        """根据研究问题和方法提供具体的分析建议"""
        suggestions = []

        question_lower = (research_question or "").lower()

        # 组间比较分析
        if any(kw in question_lower for kw in ["区别", "差异", "比较", "difference", "compare", "vs"]):
            suggestions.append("""
- **组间比较分析**：
  - 使用独立样本t检验（scipy.stats.ttest_ind）比较两组均值
  - 计算效应量（Cohen's d）评估差异程度
  - 使用箱线图或小提琴图可视化组间差异
  - 考虑多重比较校正（Bonferroni或FDR）
""")

        # 相关分析
        if any(kw in question_lower for kw in ["相关", "关联", "correlation", "relationship"]):
            suggestions.append("""
- **相关性分析**：
  - 使用Pearson或Spearman相关系数
  - 绘制散点图展示相关关系
  - 计算p值评估相关性显著性
  - 使用热图展示多变量相关矩阵
""")

        # 脑区分析
        if any(kw in question_lower for kw in ["脑区", "brain region", "roi", "灰质", "白质"]):
            suggestions.append("""
- **脑区分析**：
  - 提取每个脑区的平均值/体积
  - 使用雷达图比较多个脑区的指标
  - 识别显著差异的脑区（多重比较校正）
  - 生成脑区差异热图
""")

        if not suggestions:
            suggestions.append("""
- **通用统计分析**：
  - 描述性统计（均值、标准差、范围）
  - 组间差异检验（t检验或ANOVA）
  - 结果可视化（柱状图、箱线图）
""")

        return "\n".join(suggestions)

    def _get_code_generation_system_prompt(self) -> str:
        """获取代码生成的系统提示"""
        return """你是一位资深的神经影像分析专家和Python开发者，专精于医学影像数据处理和统计分析。

## 你的专业背景
- 神经影像学研究经验：熟悉fMRI、VBM、DTI等分析方法
- Python开发专家：精通numpy、scipy、pandas、matplotlib、nibabel等科学计算库
- 统计学功底：掌握t检验、ANOVA、相关分析、回归分析等统计方法
- 代码质量意识：遵循PEP 8规范，编写可维护、可复用的代码

## 绝对禁止事项（Critical）
1. **禁止生成任何模拟数据**：
   - 绝对不能使用np.random生成模拟脑区数据
   - 绝对不能创建create_demo_data()或类似函数
   - 绝对不能使用use_demo_data配置项
   - 绝对不能在数据文件不存在时自动生成替代数据

2. **禁止伪造结果**：
   - 所有分析必须基于真实的工具输出数据
   - 如果数据文件不存在，必须报错并退出，不能继续
   - 不允许使用任何placeholder或dummy数据

3. **数据来源要求**：
   - 代码只能读取已存在的CSV/NIfTI等数据文件
   - 数据来源于工具（FreeSurfer/SPM等）的真实输出
   - 必须在文件不存在时抛出明确的FileNotFoundError

## 代码生成原则
1. **科学严谨性**：
   - 统计方法选择合理，符合数据分布特征
   - 正确处理多重比较问题
   - 计算并报告效应量和置信区间
   - 检验统计假设（正态性、方差齐性等）
   - **所有结果必须来自真实数据，不能是生成的**

2. **工程质量**：
   - 模块化设计，每个函数职责单一
   - 完整的错误处理和边界条件检查
   - 详细的文档字符串（docstring）
   - 清晰的变量命名和代码注释

3. **可重复性**：
   - 使用配置参数而非硬编码
   - 设置随机种子确保结果可复现（仅用于统计抽样等，不用于生成数据）
   - 保存完整的分析参数和中间结果
   - 生成详细的分析日志

4. **实用性**：
   - 代码可直接运行，无需额外修改
   - 自动创建输出目录
   - 生成易于理解的可视化结果
   - 输出结构化的分析报告

## 语言要求（Critical for Windows compatibility）
- **Use English** for ALL logging messages, print statements, and code comments
- This ensures proper display on Windows consoles (GBK encoding compatibility)
- Examples:
  - DO: logger.info("Starting analysis..."), print("Loading data...")
  - DON'T: logger.info("开始分析..."), print("正在加载数据...")

## 输出要求
- 使用```python代码块包裹完整代码
- 代码长度：200-500行（根据复杂度调整）
- 包含完整的main函数和if __name__ == "__main__"入口
- 注释覆盖率：15-25%（关键逻辑必须注释）
- **数据加载函数必须在文件不存在时raise FileNotFoundError**"""

    def _extract_code_from_response(self, response: str) -> str:
        """从LLM响应中提取代码（增强版：支持多种格式+AST验证）"""
        # 清理重复代码块标记
        response_cleaned = re.sub(r'```python\s*\n\s*```python', '```python', response)
        response_cleaned = re.sub(r'```\s*\n\s*```', '```', response_cleaned)

        # 尝试多种代码块格式
        patterns = [
            r'```python\s*(.*?)\s*```',      # ```python ... ```
            r'```py\s*(.*?)\s*```',          # ```py ... ```
            r'```\s*(.*?)\s*```',            # ``` ... ```
        ]

        for pattern in patterns:
            code_match = re.search(pattern, response_cleaned, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                if code:
                    # AST验证是否为有效Python代码
                    try:
                        ast.parse(code)
                        return code
                    except SyntaxError:
                        continue  # 尝试下一个匹配

        # 智能提取：从import/def/class等关键字开始
        lines = response_cleaned.split('\n')
        code_lines = []
        in_code = False
        for line in lines:
            if re.match(r'^(import |from |def |class |#|@)', line.strip()):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            code = '\n'.join(code_lines)
            try:
                ast.parse(code)
                return code
            except SyntaxError:
                pass

        # 最后回退：返回整个响应但记录警告
        print("  [WARNING] 无法提取有效代码块，使用原始响应")
        return response_cleaned.strip()

    def _review_code(self, code: str) -> Dict[str, Any]:
        """
        审查代码质量和安全性

        Returns:
            审查结果字典
        """
        review_result = {
            "has_errors": False,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        # 1. 语法检查
        try:
            ast.parse(code)
        except SyntaxError as e:
            review_result["has_errors"] = True
            review_result["errors"].append({
                "type": "syntax_error",
                "message": str(e),
                "line": e.lineno if hasattr(e, 'lineno') else None
            })
            return review_result  # 语法错误时直接返回

        # 2. 安全检查
        dangerous_patterns = [
            (r'\bos\.system\b', "使用os.system可能不安全"),
            (r'\beval\b', "使用eval可能不安全"),
            (r'\bexec\b', "使用exec可能不安全"),
            (r'\b__import__\b', "动态导入可能不安全"),
            (r'\bopen\([^)]*[\'"]w[\'"]', "文件写入操作需要确认输出路径")
        ]

        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                review_result["warnings"].append({
                    "type": "security",
                    "message": message
                })

        # 3. 导入检查
        tree = ast.parse(code)
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])

        # 检查是否有必要的导入
        if not imports:
            review_result["warnings"].append({
                "type": "missing_imports",
                "message": "代码中没有导入任何模块，可能不完整"
            })

        # 4. 使用GLM-4.6进行深度审查
        review_result["llm_review"] = self._llm_code_review(code)

        # 合并LLM审查结果
        if review_result["llm_review"].get("issues"):
            for issue in review_result["llm_review"]["issues"]:
                if issue.get("severity") == "error":
                    review_result["has_errors"] = True
                    review_result["errors"].append(issue)
                elif issue.get("severity") == "warning":
                    review_result["warnings"].append(issue)

        return review_result

    def _llm_code_review(self, code: str) -> Dict[str, Any]:
        """使用LLM进行代码审查"""
        self.llm.set_task_type("code_review")

        review_prompt = f"""
# 神经影像分析代码审查任务

请对以下Python代码进行全面审查，重点关注科学严谨性和工程质量。

## 待审查代码
```python
{code}
```

## 审查维度

### 1. 科学正确性（Critical）
- 统计方法是否适用于数据类型
- 是否正确处理多重比较问题
- 是否检验统计假设
- 是否计算效应量

### 2. 运行时安全（Critical）
- 文件路径是否存在性检查
- 数据维度是否验证
- 异常是否妥善捕获
- 边界条件是否处理

### 3. 代码质量（Important）
- 函数是否模块化
- 变量命名是否清晰
- 是否有足够的注释
- 是否遵循PEP 8规范

### 4. 实用性（Important）
- 是否能直接运行
- 输出是否完整
- 可视化是否清晰
- 日志是否详细

## 输出格式（JSON）
{{
    "has_issues": true/false,
    "issues": [
        {{
            "severity": "error/warning/info",
            "type": "scientific/runtime/quality/usability",
            "message": "具体问题描述",
            "line": 行号（如能定位）,
            "suggestion": "详细的修复建议"
        }}
    ],
    "overall_quality": "excellent/good/fair/poor",
    "quality_score": 0-100,
    "comments": "总体评价（包括亮点和不足）",
    "scientific_rigor": "对科学严谨性的评价",
    "code_maintainability": "对代码可维护性的评价"
}}

请严格审查，对于任何可能导致错误结果的问题都标记为"error"级别。
"""

        try:
            review_response = self.llm.generate_json(
                review_prompt,
                temperature=0.2
            )
            return review_response
        except Exception as e:
            print(f"  [WARNING] LLM代码审查失败: {e}")
            return {"has_issues": False, "issues": [], "overall_quality": "unknown"}

    def _fix_code_errors(self, code: str, review_result: Dict[str, Any]) -> str:
        """尝试修复代码错误"""
        self.llm.set_task_type("algorithm_code_generation")

        errors_description = "\n".join([
            f"- {err.get('message', str(err))}"
            for err in review_result.get("errors", [])
        ])

        fix_prompt = f"""
以下代码存在错误，请修复：

错误列表：
{errors_description}

原始代码：
```python
{code}
```

请输出修复后的完整代码（使用```python代码块包裹）。
"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self._get_code_generation_system_prompt()},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.2,
                max_tokens=32768  # 大幅增加token限制，确保修复后代码完整
            )

            fixed_code = self._extract_code_from_response(
                response["choices"][0]["message"]["content"]
            )
            print(f"  [OK] 代码错误已修复")
            return fixed_code

        except Exception as e:
            print(f"  [ERROR] 代码修复失败: {e}")
            return code  # 返回原始代码

    def _save_generated_code(
        self,
        code: str,
        task_description: str,
        plan: Dict[str, Any],
        review_result: Dict[str, Any],
        run_dir: Path
    ) -> Dict[str, Any]:
        """
        保存生成的代码和元数据

        Returns:
            保存结果字典
        """
        # 创建generated_code目录
        code_dir = run_dir / "generated_code"
        code_dir.mkdir(exist_ok=True)

        # 生成文件名（使用时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_filename = f"algorithm_{timestamp}.py"
        metadata_filename = f"algorithm_{timestamp}_metadata.json"

        code_path = code_dir / code_filename
        metadata_path = code_dir / metadata_filename

        # 清理代码格式 - 移除markdown包装器
        cleaned_code = code.strip()
        if cleaned_code.startswith('```python'):
            cleaned_code = cleaned_code[len('```python'):].lstrip('\n')
        elif cleaned_code.startswith('```'):
            cleaned_code = cleaned_code[len('```'):].lstrip('\n')
        if cleaned_code.endswith('```'):
            cleaned_code = cleaned_code[:-len('```')].rstrip('\n')

        # 保存代码
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_code)

        # 保存元数据
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "task_description": task_description,
            "plan_summary": {
                "research_question": plan.get("research_question", ""),
                "methods": plan.get("methods", []),
                "expected_output": plan.get("expected_output", "")
            },
            "code_length": len(code),
            "review_result": {
                "has_errors": review_result.get("has_errors", False),
                "errors_count": len(review_result.get("errors", [])),
                "warnings_count": len(review_result.get("warnings", [])),
                "overall_quality": review_result.get("llm_review", {}).get("overall_quality", "unknown")
            },
            "code_file": code_filename
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            "code_path": str(code_path),
            "metadata_path": str(metadata_path)
        }

    def _check_syntax(self, code: str) -> Dict[str, Any]:
        """语法检查"""
        try:
            ast.parse(code)
            return {"has_error": False, "error": None, "line": None}
        except SyntaxError as e:
            return {
                "has_error": True,
                "error": str(e),
                "line": e.lineno if hasattr(e, 'lineno') else None,
                "offset": e.offset if hasattr(e, 'offset') else None
            }

    def _execute_code(self, code_path: str, run_dir: Path, is_retry: bool = False, output_dir: Path = None) -> Dict[str, Any]:
        """执行生成的代码"""
        import os

        try:
            if not os.path.exists(code_path):
                return {
                    "success": False,
                    "error": f"代码文件不存在: {code_path}",
                    "output_files": [],
                    "stdout": "",
                    "stderr": ""
                }

            # 获取执行前的文件列表（排除generated_code目录）
            before_files = set()
            for pattern in ['**/*.png', '**/*.jpg', '**/*.csv', '**/*.json', '**/*.txt', '**/*.html']:
                for f in glob(str(run_dir / pattern), recursive=True):
                    if 'generated_code' not in f:
                        before_files.add(f)

            # 运行Python脚本
            result = subprocess.run(
                [sys.executable, code_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5分钟超时
                cwd=str(run_dir),
                encoding='utf-8',
                errors='replace'
            )

            # 获取执行后的文件列表（排除generated_code目录）
            after_files = set()
            for pattern in ['**/*.png', '**/*.jpg', '**/*.csv', '**/*.json', '**/*.txt', '**/*.html']:
                for f in glob(str(run_dir / pattern), recursive=True):
                    if 'generated_code' not in f:
                        after_files.add(f)

            new_files = sorted(list(after_files - before_files))

            # 如果是重试且没有检测到新文件，列出所有分析结果文件
            if not new_files and result.returncode == 0:
                analysis_dir = run_dir / "analysis_results"
                if analysis_dir.exists():
                    for pattern in ['*.png', '*.jpg', '*.csv', '*.json', '*.txt']:
                        new_files.extend([str(f) for f in analysis_dir.glob(pattern)])

                # 同时扫描传入的 output_dir（post_processing 脚本的输出目录）
                if not new_files and output_dir:
                    output_path = Path(output_dir)
                    if output_path.exists() and output_path != (run_dir / "analysis_results"):
                        for pattern in ['*.png', '*.jpg', '*.csv', '*.json', '*.txt']:
                            new_files.extend([str(f) for f in output_path.glob(pattern)])

            return {
                "success": result.returncode == 0,
                "output_files": new_files,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "代码执行超时（超过5分钟）",
                "output_files": [],
                "stdout": "",
                "stderr": ""
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"执行异常: {str(e)}",
                "output_files": [],
                "stdout": "",
                "stderr": str(e)
            }

    def _validate_analysis_results(self, output_files: List[str], run_dir: Path) -> Dict[str, Any]:
        """
        验证分析结果，检测常见错误如ROI值重复

        Args:
            output_files: 输出文件列表
            run_dir: 运行目录

        Returns:
            验证结果字典
        """
        import json

        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # 查找统计结果JSON文件
        stats_files = [f for f in output_files if f.endswith('.json') and
                       ('statistic' in f.lower() or 'descriptive' in f.lower() or 'comparison' in f.lower())]

        for stats_file in stats_files:
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not isinstance(data, dict):
                    continue

                # 收集所有ROI的mean值
                mean_values = []
                roi_names = []

                for roi_name, roi_data in data.items():
                    if isinstance(roi_data, dict):
                        # 检查是否有group数据
                        for group_name, group_data in roi_data.items():
                            if isinstance(group_data, dict) and 'mean' in group_data:
                                mean_values.append(group_data['mean'])
                                roi_names.append(f"{roi_name}_{group_name}")
                        # 检查直接的mean字段
                        if 'group1_mean' in roi_data:
                            mean_values.append(roi_data['group1_mean'])
                            roi_names.append(f"{roi_name}_group1")
                        if 'group2_mean' in roi_data:
                            mean_values.append(roi_data['group2_mean'])
                            roi_names.append(f"{roi_name}_group2")

                # 检测重复值
                if len(mean_values) >= 4:  # 至少4个值才有意义检测
                    unique_values = set(mean_values)
                    # 计算预期的最小唯一值数量
                    # 如果有N个ROI和M个组，预期至少有 min(N, M*some_factor) 个唯一值
                    # 但如果唯一值 <= 2（只有组间差异，ROI间无差异），几乎肯定是错误
                    num_rois = len(set(r.rsplit('_', 1)[0] for r in roi_names))
                    min_expected_unique = max(3, num_rois // 2)  # 至少3个唯一值，或ROI数量的一半

                    # 条件1: 唯一值太少（<= 2通常表示全脑均值错误）
                    # 条件2: 唯一值少于预期的30%
                    is_suspicious = len(unique_values) <= 2 or len(unique_values) < len(mean_values) * 0.30

                    if is_suspicious:
                        validation_result["valid"] = False
                        validation_result["errors"].append({
                            "type": "roi_value_duplication",
                            "file": stats_file,
                            "message": f"检测到ROI值重复问题：{len(mean_values)}个值中只有{len(unique_values)}个唯一值（{num_rois}个ROI）。"
                                       f"这通常表示代码使用了全脑均值代替ROI特定值。",
                            "unique_values": list(unique_values)[:5],
                            "affected_rois": roi_names[:10]
                        })
                        print(f"  [验证失败] ROI值重复: {len(mean_values)} values, {len(unique_values)} unique, {num_rois} ROIs")

            except Exception as e:
                validation_result["warnings"].append({
                    "type": "validation_error",
                    "file": stats_file,
                    "message": f"无法验证文件: {str(e)}"
                })

        if validation_result["valid"]:
            print("  [验证通过] 分析结果通过ROI值多样性检查")

        return validation_result

    def _fix_syntax_error(self, code: str, syntax_check: Dict) -> str:
        """修复语法错误"""
        self.llm.set_task_type("algorithm_code_generation")

        fix_prompt = f"""
以下Python代码存在语法错误，请修复：

语法错误信息：
- 错误: {syntax_check.get('error', '')}
- 行号: {syntax_check.get('line', 'unknown')}

原始代码：
```python
{code}
```

请输出修复后的完整代码（使用```python代码块包裹）。只修复语法错误，不要改变代码逻辑。
"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个Python专家，专门修复代码语法错误。"},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.1,
                max_tokens=32768
            )
            return self._extract_code_from_response(response["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"  [WARNING] 语法修复失败: {e}")
            return code

    def _fix_runtime_error(
        self,
        code: str,
        execution_result: Dict,
        task_description: str = "",
        data_indicators: Dict = None,
        error_history: List[Dict] = None
    ) -> str:
        """根据运行时错误修复代码（增强版，包含更多上下文）"""
        self.llm.set_task_type("algorithm_code_generation")

        stderr = execution_result.get("stderr", "")
        stdout = execution_result.get("stdout", "")
        error = execution_result.get("error", "")

        # 提取错误的关键信息（保留开头和结尾以获取完整错误上下文）
        error_info = stderr or error
        if len(error_info) > 3000:
            # 保留开头1000字符（import错误）+ 结尾2000字符（堆栈跟踪）
            error_info = error_info[:1000] + "\n...[truncated]...\n" + error_info[-2000:]

        # 构建错误历史说明
        error_history_text = ""
        if error_history and len(error_history) > 1:
            error_history_text = "\n## 之前的错误历史\n"
            error_history_text += "以下是之前的修复尝试中遇到的错误，请避免重复这些错误：\n"
            for h in error_history[-5:]:  # 显示最近5次错误
                error_history_text += f"- 尝试 {h['attempt']}: {h['error'][:200]}\n"

        # 构建数据上下文说明
        data_context = ""
        if data_indicators:
            input_file = data_indicators.get("input_data_file", "")
            n_subjects = data_indicators.get("n_subjects", 0)
            groups = data_indicators.get("groups", [])
            if input_file:
                data_context = f"""
## 可用的数据文件
**输入文件路径**: `{input_file}`
**被试数量**: {n_subjects}
**分组**: {', '.join(groups) if groups else '未知'}

**重要提示**:
- 这是真实数据文件，必须使用此路径
- 使用 `pd.read_csv(r'{input_file}')` 读取数据
- 如果文件不存在，应该报错而不是生成模拟数据
"""

        fix_prompt = f"""
以下Python代码在运行时出现错误，请根据错误信息修复代码。

## 任务描述
{task_description}
{data_context}
{error_history_text}

## 当前运行时错误信息
```
{error_info}
```

## 标准输出（部分）
```
{stdout[-500:] if stdout else '无输出'}
```

## 需要修复的代码
```python
{code}
```

## 修复要求

### 关键要求（必须遵守）
1. **输出完整代码**：必须返回完整的Python代码，包含所有import、所有函数定义和主执行逻辑
2. **不要返回代码片段**：不要只返回修改的部分，必须返回整个文件的完整代码
3. **代码必须可执行**：修复后的代码应该能够直接运行，无需任何额外修改

### 错误分析和修复
1. 仔细分析错误信息，找出问题的根本原因
2. 如果是 NameError（变量未定义），检查：
   - 变量是否在使用前定义
   - 是否在正确的作用域内
   - 循环变量是否正确初始化
3. 如果是文件路径问题，使用上述提供的实际文件路径
4. 如果是模块导入问题，检查并修正导入语句
5. 如果是数据处理问题，添加适当的错误处理和数据验证

### 绝对禁止
- **禁止生成模拟数据**：不能使用np.random生成任何脑区数据
- **禁止创建demo数据函数**：不能有create_demo_data()或类似函数
- **禁止使用use_demo_data配置**：不能有任何fallback到模拟数据的逻辑
- **禁止返回不完整代码**：不能只返回几行代码片段，必须是完整可运行的程序
- 如果数据文件不存在，代码应该raise FileNotFoundError并退出，而不是生成替代数据

### 输出格式
请输出修复后的**完整Python代码**，使用```python代码块包裹。
代码应该包含完整的文档字符串和注释。
确保代码长度至少500字符以上（完整的程序不可能太短）。
"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self._get_code_generation_system_prompt()},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.2,
                max_tokens=32768
            )
            fixed_code = self._extract_code_from_response(response["choices"][0]["message"]["content"])

            # 验证代码不是过短的片段
            if len(fixed_code) < 500:
                print(f"  [WARNING] LLM返回的代码过短 ({len(fixed_code)} 字符)，可能不完整")
                return code  # 返回原代码而不是不完整的修复

            print(f"  [OK] 代码已根据运行错误修复 ({len(fixed_code)} 字符)")
            return fixed_code
        except Exception as e:
            print(f"  [WARNING] 运行错误修复失败: {e}")
            return code

    def _fix_runtime_error_with_retry(
        self,
        code: str,
        execution_result: Dict,
        task_description: str = "",
        data_indicators: Dict = None,
        error_history: list = None
    ) -> str:
        """使用更高温度和不同策略重新修复代码"""
        self.llm.set_task_type("algorithm_code_generation")

        stderr = execution_result.get("stderr", "")
        error = execution_result.get("error", "")
        error_info = stderr or error
        if len(error_info) > 1500:
            error_info = error_info[-1500:]

        # 构建数据上下文
        data_context = ""
        if data_indicators:
            input_file = data_indicators.get("input_data_file", "")
            if input_file:
                data_context = f"可用数据文件: `{input_file}`"

        # 构建错误历史上下文
        error_history_text = ""
        if error_history:
            error_history_text = "\n之前的修复尝试（请避免重复同样的修复策略）:\n" + "\n".join([f"- 第{e['attempt']}次: {e['error'][:200]}" for e in error_history])

        fix_prompt = f"""
你正在修复一个神经影像分析代码。{data_context}{error_history_text}

错误信息:
```
{error_info}
```

原始代码:
```python
{code}
```

请生成修复后的**完整代码**。注意：
1. 必须包含所有import语句
2. 必须包含所有函数定义
3. 必须包含主执行逻辑
4. 代码必须是完整的、可直接运行的
5. 不要生成模拟数据，使用真实数据文件

输出完整的Python代码（```python包裹）：
"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个Python专家，擅长修复代码错误。始终返回完整的、可运行的代码。"},
                    {"role": "user", "content": fix_prompt}
                ],
                temperature=0.4,  # 更高温度，增加多样性
                max_tokens=32768
            )
            fixed_code = self._extract_code_from_response(response["choices"][0]["message"]["content"])

            if len(fixed_code) < 500:
                print(f"  [WARNING] 重试修复仍然返回过短代码 ({len(fixed_code)} 字符)")
                return code

            print(f"  [OK] 重试修复完成 ({len(fixed_code)} 字符)")
            return fixed_code
        except Exception as e:
            print(f"  [WARNING] 重试修复失败: {e}")
            return code

    def generate_data_extraction_code(
        self,
        subjects_dir: str,
        subject_ids: List[str],
        extraction_type: str,
        output_dir: Path,
        run_dir: Path,
        task_description: str = ""
    ) -> Dict[str, Any]:
        """
        生成数据提取脚本（用于从FreeSurfer/SPM等工具输出中提取数据）

        Args:
            subjects_dir: FreeSurfer SUBJECTS_DIR或SPM输出目录
            subject_ids: 被试ID列表
            extraction_type: 提取类型 (subcortical_volumes, cortical_thickness, etc.)
            output_dir: 输出目录
            run_dir: 运行目录
            task_description: 任务描述

        Returns:
            生成结果字典
        """
        print("\n" + "="*60)
        print("Vibe Coding - 数据提取脚本生成")
        print("="*60)
        print(f"  SUBJECTS_DIR: {subjects_dir}")
        print(f"  被试数量: {len(subject_ids)}")
        print(f"  提取类型: {extraction_type}")

        # 构建提取脚本生成提示
        prompt = self._build_extraction_prompt(
            subjects_dir, subject_ids, extraction_type, output_dir, task_description
        )

        # 使用高级模型生成代码
        print("\n[步骤 1/4] 使用高级模型生成数据提取脚本...")
        self.llm.set_task_type("script_generation")

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self._get_extraction_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=16384
            )

            generated_content = response["choices"][0]["message"]["content"]

            # 调试：打印原始返回内容长度
            print(f"  [调试] LLM返回内容长度: {len(generated_content)} 字符")
            if len(generated_content) < 100:
                print(f"  [调试] LLM返回内容: {generated_content}")
            else:
                print(f"  [调试] LLM返回内容前200字符: {generated_content[:200]}")

            code = self._extract_code_from_response(generated_content)

            # 调试：打印提取后的代码长度
            print(f"  [调试] 提取后代码长度: {len(code)} 字符")
            if len(code) == 0:
                print(f"  [警告] 代码提取失败！原始内容: {generated_content[:500]}")

            print(f"  [OK] 脚本生成完成 ({len(code)} 字符)")

        except Exception as e:
            print(f"  [ERROR] 脚本生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "output_files": []
            }

        # 运行-审查-修复流程
        max_fix_attempts = 10  # 最多10次修复尝试
        attempt = 0
        execution_success = False
        execution_result = None
        save_result = None
        error_history = []

        while attempt < max_fix_attempts:
            attempt += 1

            # 语法检查
            print(f"\n[步骤 2/4] 语法检查... (尝试 {attempt}/{max_fix_attempts})")
            syntax_check = self._check_syntax(code)

            if syntax_check["has_error"]:
                print(f"  [ERROR] 语法错误: {syntax_check['error']}")
                if attempt < max_fix_attempts:
                    code = self._fix_syntax_error(code, syntax_check)
                    continue
                else:
                    break

            print("  [OK] 语法检查通过")

            # 保存代码
            print(f"\n[步骤 3/4] 保存脚本...")
            save_result = self._save_extraction_script(
                code=code,
                extraction_type=extraction_type,
                run_dir=run_dir
            )
            print(f"  [OK] 脚本已保存: {save_result['code_path']}")

            # 运行代码
            print(f"\n[步骤 4/4] 运行脚本...")
            execution_result = self._execute_code(
                code_path=save_result["code_path"],
                run_dir=run_dir
            )

            if execution_result["success"]:
                print(f"  [OK] 脚本执行成功!")
                if execution_result.get("output_files"):
                    print(f"  [输出] 生成 {len(execution_result['output_files'])} 个文件")
                    for f in execution_result["output_files"][:5]:
                        print(f"    - {Path(f).name}")
                execution_success = True
                break
            else:
                print(f"  [ERROR] 脚本执行失败")
                error_msg = execution_result.get("stderr", "") or execution_result.get("error", "")
                try:
                    print(f"  [错误信息] {error_msg[:500]}")
                except UnicodeEncodeError:
                    safe_msg = error_msg[:500].encode('gbk', errors='replace').decode('gbk')
                    print(f"  [错误信息] {safe_msg}")

                if attempt < max_fix_attempts:
                    print(f"\n[修复] 根据运行错误修复脚本... (第 {attempt} 次)")
                    error_history.append({
                        "attempt": attempt,
                        "error": error_msg[:1000]
                    })
                    previous_code = code
                    code = self._fix_runtime_error(
                        code=code,
                        execution_result=execution_result,
                        task_description=task_description,
                        error_history=error_history
                    )
                    # 代码完整性检查
                    if len(code) < 500:
                        print(f"  [WARNING] 修复后代码过短 ({len(code)} 字符)，恢复原版本")
                        code = previous_code

        return {
            "success": execution_success,
            "code": code,
            "save_path": save_result["code_path"] if save_result else "",
            "output_files": execution_result.get("output_files", []) if execution_result else [],
            "attempts": attempt,
            "execution_result": execution_result
        }

    def _build_extraction_prompt(
        self,
        subjects_dir: str,
        subject_ids: List[str],
        extraction_type: str,
        output_dir: Path,
        task_description: str
    ) -> str:
        """构建数据提取脚本生成提示"""

        # 将路径转换为正斜杠格式（避免某些LLM无法正确处理反斜杠）
        subjects_dir = subjects_dir.replace('\\', '/')
        output_dir = str(output_dir).replace('\\', '/')

        # 根据提取类型确定要读取的文件
        file_patterns = {
            "subcortical_volumes": {
                "stats_file": "stats/aseg.stats",
                "description": "皮下结构体积（如海马、杏仁核、丘脑等）"
            },
            "cortical_thickness": {
                "stats_file": "stats/lh.aparc.stats, stats/rh.aparc.stats",
                "description": "皮层厚度（DK atlas分区）"
            },
            "cortical_volume": {
                "stats_file": "stats/lh.aparc.stats, stats/rh.aparc.stats",
                "description": "皮层体积"
            },
            "cortical_area": {
                "stats_file": "stats/lh.aparc.stats, stats/rh.aparc.stats",
                "description": "皮层表面积"
            },
            "brain_volumes": {
                "stats_file": "stats/aseg.stats",
                "description": "全脑体积指标（灰质、白质、脑脊液等）"
            }
        }

        type_info = file_patterns.get(extraction_type, file_patterns["subcortical_volumes"])

        prompt = f"""
## 任务：生成FreeSurfer数据提取脚本

### 任务描述
{task_description if task_description else f"从FreeSurfer输出中提取{type_info['description']}"}

### 输入信息
- SUBJECTS_DIR: {subjects_dir}
- 被试列表: {subject_ids}
- 提取类型: {extraction_type}
- 要读取的stats文件: {type_info['stats_file']}
- 输出目录: {output_dir}

### 要求
1. **读取FreeSurfer stats文件**：
   - 遍历每个被试的stats目录
   - 解析.stats文件格式（类似表格格式）
   - 提取脑区名称、体积/厚度值等

2. **数据整理**：
   - 将所有被试的数据合并为一个DataFrame
   - 行：被试ID
   - 列：脑区名称
   - 值：对应的测量值

3. **输出**：
   - 保存为CSV文件到输出目录
   - 文件名: {extraction_type}_data.csv
   - 打印提取的数据摘要

4. **错误处理**：
   - 检查文件是否存在
   - 处理缺失数据
   - 提供清晰的错误信息

### FreeSurfer stats文件格式说明

**重要：aseg.stats文件每行是空格分隔的，列索引如下：**

aseg.stats文件格式示例：
```
# ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
  1   4     15676    15939.7  Left-Lateral-Ventricle            33.9264    11.4073    12.0000    83.0000    71.0000
  2   5       289      334.7  Left-Inf-Lat-Vent                 48.4810    13.9027    19.0000    83.0000    64.0000
  3   7     13614    14291.0  Left-Cerebellum-White-Matter      85.3890     6.1163    38.0000   103.0000    65.0000
...
```

**提取逻辑（按空格分隔）：**
- `parts[0]`: Index（第1列，行号）
- `parts[1]`: SegId（第2列，分割ID）
- `parts[2]`: NVoxels（第3列，体素数）
- `parts[3]`: Volume_mm3（第4列，体积，这是我们需要的！）
- `parts[4]`: StructName（第5列，结构名称，这是我们需要的！）

**示例提取代码：**
```python
for line in f:
    if line.startswith('#') or not line.strip():
        continue
    parts = line.split()
    if len(parts) >= 5:
        seg_id = parts[1]  # SegId
        volume = float(parts[3])  # Volume_mm3
        struct_name = parts[4]  # StructName
        data[struct_name] = volume
```

aparc.stats文件格式示例：
```
# ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ...
bankssts    1346    882    2894    2.510    ...
caudalanteriorcingulate    1156    830    2656    2.712    ...
...
```

### 代码结构
```python
import os
import pandas as pd
from pathlib import Path

# 配置
CONFIG = {{
    'subjects_dir': r'{subjects_dir}',
    'subjects': {subject_ids},
    'output_dir': r'{output_dir}',
    'extraction_type': '{extraction_type}'
}}

def parse_stats_file(stats_path):
    '''解析FreeSurfer stats文件'''
    pass

def extract_data(config):
    '''提取所有被试的数据'''
    pass

def main():
    '''主函数'''
    pass

if __name__ == '__main__':
    main()
```

请生成完整的Python脚本，使用```python代码块包裹。
"""
        return prompt

    def _get_extraction_system_prompt(self) -> str:
        """获取数据提取脚本生成的系统提示"""
        return """你是一位神经影像数据处理专家，擅长编写FreeSurfer和SPM数据提取脚本。

## 专业背景
- 熟悉FreeSurfer输出结构（recon-all生成的stats目录）
- 了解各种脑区指标（体积、厚度、面积等）
- 精通Python数据处理（pandas、numpy）

##  绝对禁止
- **禁止生成任何模拟数据**：不能使用np.random生成脑区数据
- **禁止创建demo数据函数**：不能有create_demo_data()或类似函数
- **禁止使用use_demo_data配置**：不能有任何fallback到模拟数据的逻辑
- 如果文件不存在，必须raise FileNotFoundError，不能继续
- 所有提取的数据必须来自真实的FreeSurfer/SPM输出文件

## 代码要求
1. **健壮性**：处理缺失文件、格式错误等异常（缺失时报错而非生成替代数据）
2. **可读性**：清晰的注释和变量命名
3. **可复现**：固定路径配置，便于重复使用
4. **输出完整**：包含数据摘要和保存确认
5. **真实数据**：所有数据必须来自工具的真实输出

## 语言要求（Windows兼容性）
- **Use English** for all logging messages and print statements
- Avoid Chinese characters to prevent console encoding issues
- Example: print("Loading data...") NOT print("正在加载数据...")

## 输出格式
- 使用```python代码块包裹完整代码
- 代码应可直接运行
- 包含详细的打印输出说明进度"""

    def _save_extraction_script(
        self,
        code: str,
        extraction_type: str,
        run_dir: Path
    ) -> Dict[str, Any]:
        """保存数据提取脚本"""
        code_dir = run_dir / "generated_code"
        code_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        code_filename = f"extract_{extraction_type}_{timestamp}.py"
        code_path = code_dir / code_filename

        # 清理代码格式
        cleaned_code = code.strip()
        if cleaned_code.startswith('```python'):
            cleaned_code = cleaned_code[len('```python'):].lstrip('\n')
        elif cleaned_code.startswith('```'):
            cleaned_code = cleaned_code[len('```'):].lstrip('\n')
        if cleaned_code.endswith('```'):
            cleaned_code = cleaned_code[:-len('```')].rstrip('\n')

        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_code)

        return {"code_path": str(code_path)}

    def generate_post_processing_script(
        self,
        task_type: str,
        source_tool: str,
        tool_outputs: Dict[str, Any],
        task_description: str,
        output_dir: Path,
        run_dir: Path,
        cohort: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        生成后处理脚本（数据提取、分析、可视化）

        这是一个统一的入口，根据上游工具类型和任务类型生成合适的脚本。

        Args:
            task_type: 任务类型 (extraction, analysis, visualization)
            source_tool: 上游工具 (freesurfer, spm, dsi_studio, fsl)
            tool_outputs: 上游工具的输出（文件列表、目录等）
            task_description: 任务描述
            output_dir: 输出目录
            run_dir: 运行目录
            cohort: 队列信息（分组等）

        Returns:
            生成结果字典
        """
        print("\n" + "="*60)
        print(f"Vibe Coding - 后处理脚本生成")
        print("="*60)
        print(f"  任务类型: {task_type}")
        print(f"  上游工具: {source_tool}")
        print(f"  任务描述: {task_description[:50]}...")

        # 构建提示
        prompt = self._build_post_processing_prompt(
            task_type=task_type,
            source_tool=source_tool,
            tool_outputs=tool_outputs,
            task_description=task_description,
            output_dir=output_dir,
            cohort=cohort
        )

        # 使用高级模型生成代码
        print("\n[步骤 1/4] 使用高级模型生成脚本...")
        self.llm.set_task_type("script_generation")

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": self._get_post_processing_system_prompt(source_tool)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=16384
            )

            generated_content = response["choices"][0]["message"]["content"]

            # 调试：打印原始返回内容长度
            print(f"  [调试] LLM返回内容长度: {len(generated_content)} 字符")
            if len(generated_content) < 100:
                print(f"  [调试] LLM返回内容: {generated_content}")
            else:
                print(f"  [调试] LLM返回内容前200字符: {generated_content[:200]}")

            code = self._extract_code_from_response(generated_content)

            # 调试：打印提取后的代码长度
            print(f"  [调试] 提取后代码长度: {len(code)} 字符")
            if len(code) == 0:
                print(f"  [警告] 代码提取失败！原始内容: {generated_content[:500]}")

            print(f"  [OK] 脚本生成完成 ({len(code)} 字符)")

        except Exception as e:
            print(f"  [ERROR] 脚本生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "code": "",
                "output_files": []
            }

        # 运行-审查-修复流程
        max_fix_attempts = 10  # 最多10次修复尝试
        attempt = 0
        execution_success = False
        execution_result = None
        save_result = None
        error_history = []  # 记录错误历史，避免重复修复

        while attempt < max_fix_attempts:
            attempt += 1

            # 语法检查
            print(f"\n[步骤 2/4] 语法检查... (尝试 {attempt}/{max_fix_attempts})")
            syntax_check = self._check_syntax(code)

            if syntax_check["has_error"]:
                print(f"  [ERROR] 语法错误: {syntax_check['error']}")
                if attempt < max_fix_attempts:
                    code = self._fix_syntax_error(code, syntax_check)
                    continue
                else:
                    break

            print("  [OK] 语法检查通过")

            # 保存代码
            print(f"\n[步骤 3/4] 保存脚本...")
            script_name = f"{source_tool}_{task_type}"
            save_result = self._save_extraction_script(
                code=code,
                extraction_type=script_name,
                run_dir=run_dir
            )
            print(f"  [OK] 脚本已保存: {save_result['code_path']}")

            # 运行代码
            print(f"\n[步骤 4/4] 运行脚本...")
            execution_result = self._execute_code(
                code_path=save_result["code_path"],
                run_dir=run_dir,
                output_dir=output_dir
            )

            if execution_result["success"]:
                # 如果 output_files 为空，主动扫描 output_dir
                if not execution_result.get("output_files") and output_dir:
                    output_path = Path(output_dir)
                    if output_path.exists():
                        collected = []
                        for pattern in ['*.png', '*.jpg', '*.csv', '*.json', '*.txt']:
                            collected.extend([str(f) for f in output_path.glob(pattern)])
                        if collected:
                            execution_result["output_files"] = collected
                            print(f"  [补充扫描] 从 output_dir 收集到 {len(collected)} 个文件")

                print(f"  [OK] 脚本执行成功!")
                if execution_result.get("output_files"):
                    print(f"  [输出] 生成 {len(execution_result['output_files'])} 个文件")
                    for f in execution_result["output_files"][:5]:
                        print(f"    - {Path(f).name}")
                execution_success = True
                break
            else:
                print(f"  [ERROR] 脚本执行失败")
                error_msg = execution_result.get("stderr", "") or execution_result.get("error", "")
                try:
                    print(f"  [错误信息] {error_msg[:500]}")
                except UnicodeEncodeError:
                    safe_msg = error_msg[:500].encode('gbk', errors='replace').decode('gbk')
                    print(f"  [错误信息] {safe_msg}")

                # 记录错误历史
                error_history.append({
                    "attempt": attempt,
                    "error": error_msg[:1000]
                })

                if attempt < max_fix_attempts:
                    print(f"\n[修复] 根据运行错误修复脚本... (第 {attempt} 次)")
                    previous_code = code
                    code = self._fix_runtime_error(
                        code=code,
                        execution_result=execution_result,
                        task_description=task_description,
                        error_history=error_history
                    )

                    # 代码完整性检查
                    if len(code) < 500:
                        print(f"  [WARNING] 修复后代码过短 ({len(code)} 字符)，恢复原版本")
                        code = previous_code

        return {
            "success": execution_success,
            "code": code,
            "save_path": save_result["code_path"] if save_result else "",
            "output_files": execution_result.get("output_files", []) if execution_result else [],
            "attempts": attempt,
            "execution_result": execution_result
        }

    def _build_post_processing_prompt(
        self,
        task_type: str,
        source_tool: str,
        tool_outputs: Dict[str, Any],
        task_description: str,
        output_dir: Path,
        cohort: Dict[str, Any]
    ) -> str:
        """构建后处理脚本生成提示"""

        # 提取输出文件信息
        output_files = tool_outputs.get("output_files", [])
        output_dir_str = tool_outputs.get("output_dir", str(output_dir))

        # 提取源文件所在目录（根据工具类型适配）
        source_dir = output_dir_str  # 默认使用output_dir
        subjects_dir = None  # FreeSurfer专用

        if output_files:
            first_file = Path(output_files[0])

            if source_tool in ["freesurfer", "freesurfer_analysis"]:
                # FreeSurfer: 需要找到 SUBJECTS_DIR
                # 结构: SUBJECTS_DIR/subject/stats/*.stats 或 SUBJECTS_DIR/subject/mri/*.mgz
                if "stats" in str(first_file):
                    # stats目录的祖父目录是 SUBJECTS_DIR
                    subjects_dir = str(first_file.parent.parent.parent)
                    source_dir = subjects_dir
                elif "mri" in str(first_file) or "surf" in str(first_file):
                    # mri/surf 目录的祖父目录是 SUBJECTS_DIR
                    subjects_dir = str(first_file.parent.parent.parent)
                    source_dir = subjects_dir
                else:
                    source_dir = str(first_file.parent) if first_file.parent.exists() else output_dir_str
            else:
                # SPM/FSL/DSI Studio/DPABI等: 直接取文件所在目录
                if first_file.exists():
                    source_dir = str(first_file.parent)
                elif first_file.parent.exists():
                    source_dir = str(first_file.parent)

        # 【修复】对于 FSL 工具，尝试从 all_tool_outputs 获取包含 DTI 文件的目录
        # 这是因为 python_stats 任务可能依赖的是 fslmeants 输出（task_08），
        # 而 DTI 文件（FA/MD）实际在 dtifit 输出目录（task_05）中
        if source_tool in ["fsl", "fsl_analysis"]:
            all_outputs = tool_outputs.get("all_tool_outputs", {})
            fsl_outputs = all_outputs.get("fsl", {})
            fsl_files = fsl_outputs.get("output_files", [])

            # 查找包含 FA/MD 文件的路径
            for f in fsl_files:
                if "_FA" in f or "_MD" in f or "_dti_" in f:
                    dti_dir = str(Path(f).parent)
                    if Path(dti_dir).exists():
                        source_dir = dti_dir
                        print(f"  [修正] FSL source_dir 指向 DTI 输出目录: {source_dir}")
                        break

            # 如果 all_tool_outputs 中没有，直接扫描 tools 目录
            if "_FA" not in source_dir and "_MD" not in source_dir:
                run_dir_path = tool_outputs.get("run_dir") or str(output_dir).rsplit("/tools/", 1)[0]
                if run_dir_path:
                    tools_dir = Path(run_dir_path) / "tools"
                    if tools_dir.exists():
                        for subdir in sorted(tools_dir.iterdir(), reverse=True):  # 按时间倒序
                            if "fsl" in subdir.name.lower():
                                # 查找包含 DTI 文件的目录
                                fa_files = list(subdir.glob("*_FA*.nii*")) + list(subdir.glob("*_dti_FA*.nii*"))
                                if fa_files:
                                    source_dir = str(subdir)
                                    # 同时更新 output_files 以包含 DTI 文件
                                    dti_files = list(subdir.glob("*_dti_*.nii*"))
                                    if dti_files:
                                        output_files = [str(f) for f in dti_files]
                                        print(f"  [修正] 更新 output_files 包含 {len(dti_files)} 个 DTI 文件")
                                    print(f"  [修正] FSL source_dir 通过扫描找到: {source_dir}")
                                    break

        # 【修复】对于 FSL 工具，如果 output_files 没有 DTI 文件，从 source_dir 收集
        if source_tool in ["fsl", "fsl_analysis"]:
            has_dti_files = any("_FA" in f or "_MD" in f or "_dti_" in f for f in output_files)
            if not has_dti_files and source_dir:
                source_dir_path = Path(source_dir)
                if source_dir_path.exists():
                    dti_files = list(source_dir_path.glob("*_dti_*.nii*"))
                    if dti_files:
                        output_files = [str(f) for f in dti_files]
                        print(f"  [修正] 从 source_dir 收集到 {len(dti_files)} 个 DTI 文件")

        # 将路径转换为正斜杠格式（避免某些LLM无法正确处理反斜杠）
        source_dir = source_dir.replace('\\', '/')
        output_dir_str = output_dir_str.replace('\\', '/')
        output_dir_prompt = str(output_dir).replace('\\', '/')  # Path对象转为正斜杠字符串

        # 预处理 subjects_dir 行（避免嵌套f-string导致花括号不匹配）
        subjects_dir_line = ""
        if subjects_dir:
            subjects_dir = subjects_dir.replace('\\', '/')
            subjects_dir_line = f"\n    'subjects_dir': r'{subjects_dir}',"

        # 将 output_files 中的路径也转换为正斜杠格式
        output_files = [f.replace('\\', '/') for f in output_files]

        # 分组信息
        group_info = ""
        cohort_subjects = []
        if cohort:
            groups = cohort.get("groups", {})
            for group_name, group_data in groups.items():
                subjects = group_data.get("subjects", [])
                cohort_subjects.extend(subjects)
                group_info += f"  - {group_name}: {len(subjects)} 个被试 ({subjects[:3]}...)\n"

        # 【关键修复】构建被试-文件映射，确保每个被试使用正确的文件
        subject_file_mapping = {}
        for subject_id in cohort_subjects:
            subject_files = [f for f in output_files if subject_id in f]
            if subject_files:
                # 按DTI指标类型分类
                fa_files = [f for f in subject_files if '_FA' in f or '_fa' in f]
                md_files = [f for f in subject_files if '_MD' in f or '_md' in f]
                ad_files = [f for f in subject_files if '_AD' in f or '_ad' in f or '_L1' in f]
                rd_files = [f for f in subject_files if '_RD' in f or '_rd' in f]
                subject_file_mapping[subject_id] = {
                    "FA": fa_files[0] if fa_files else None,
                    "MD": md_files[0] if md_files else None,
                    "AD": ad_files[0] if ad_files else None,
                    "RD": rd_files[0] if rd_files else None,
                    "all": subject_files
                }

        # 格式化被试-文件映射信息用于提示词
        subject_mapping_str = ""
        if subject_file_mapping:
            subject_mapping_str = "\n### 被试与文件的对应关系（必须严格遵守！）\n"
            subject_mapping_str += "**重要**：处理每个被试时，必须使用下表中该被试对应的文件路径，禁止使用通用glob模式！\n\n"
            subject_mapping_str += "```python\n"
            subject_mapping_str += "# 直接在代码中使用此映射\n"
            subject_mapping_str += "SUBJECT_FILES = {\n"
            for subj_id, files in subject_file_mapping.items():
                subject_mapping_str += f"    '{subj_id}': {{\n"
                if files.get('FA'):
                    subject_mapping_str += f"        'FA': r'{files['FA']}',\n"
                if files.get('MD'):
                    subject_mapping_str += f"        'MD': r'{files['MD']}',\n"
                if files.get('AD'):
                    subject_mapping_str += f"        'AD': r'{files['AD']}',\n"
                if files.get('RD'):
                    subject_mapping_str += f"        'RD': r'{files['RD']}',\n"
                subject_mapping_str += f"    }},\n"
            subject_mapping_str += "}\n```\n"

        # 根据上游工具构建特定提示
        tool_specific_info = self._get_tool_specific_info(source_tool, tool_outputs)

        # 检查是否有多个工具的输出
        all_tool_outputs = tool_outputs.get("all_tool_outputs", {})
        all_tools_info = ""
        if all_tool_outputs:
            all_tools_info = "\n### 所有可用的工具输出\n"
            for tool_name, outputs in all_tool_outputs.items():
                tool_dir = outputs.get("output_dir", "")
                tool_files = outputs.get("output_files", [])
                all_tools_info += f"\n**{tool_name}**:\n"
                all_tools_info += f"- 目录: {tool_dir}\n"
                all_tools_info += f"- 文件数: {len(tool_files)}\n"
                if tool_files:
                    all_tools_info += f"- 示例: {tool_files[:3]}\n"

        prompt = f"""
## 任务：生成{source_tool.upper()}后处理脚本

### 任务描述
{task_description}

### 任务类型
{task_type} (extraction=数据提取, analysis=统计分析, visualization=可视化)

### 上游工具输出
- 主要工具: {source_tool}
- 源文件目录: {source_dir}
- 输出目录: {output_dir_str}
- 输出文件数量: {len(output_files)}
{f"- 示例文件: {output_files[:5]}" if output_files else ""}
{all_tools_info}

### 源文件路径列表（请直接使用这些路径）
{chr(10).join(output_files[:10]) if output_files else "无文件"}
{subject_mapping_str}
### 分组信息
{group_info if group_info else "无分组信息"}

### 工具特定信息
{tool_specific_info}

### 代码要求（严格遵守，减少错误）

1. **输入处理（关键）**：
   - **文件路径**：使用绝对路径，使用 Path 对象处理路径
   - **文件检查**：在读取前检查文件是否存在，不存在立即抛出 FileNotFoundError
   - **错误提示**：文件不存在时，清晰列出预期路径和实际扫描结果
   - **Windows路径**：使用 raw string (r'path') 或 Path 对象，避免转义问题
   - **示例代码**：
     ```python
     from pathlib import Path

     def find_gray_matter_files(input_dir):
         \"\"\"查找灰质文件\"\"\"
         input_path = Path(input_dir)
         if not input_path.exists():
             raise FileNotFoundError(f"输入目录不存在: {{input_path}}")

         # 查找文件（提供多种可能的命名模式）
         patterns = ['wc1*.nii', 'c1*.nii', 'swc1*.nii']
         files = []
         for pattern in patterns:
             files.extend(list(input_path.rglob(pattern)))

         if not files:
             # 详细的错误信息
             available = list(input_path.rglob('*.nii'))
             raise FileNotFoundError(
                 f"未找到灰质文件！\\n"
                 f"搜索目录: {{input_path}}\\n"
                 f"搜索模式: {{patterns}}\\n"
                 f"目录中的.nii文件: {{[f.name for f in available[:10]]}}\\n"
                 f"请检查文件命名或处理流程"
             )

         return files
     ```

2. **数据处理（详细算法）**：
   - **数据提取**：
     * FreeSurfer: 解析 stats 文件，提取每行的体积/厚度数值
     * SPM: 加载 NIfTI 文件，计算概率图的体素和（sum(data) * voxel_volume）
     * 整理为 DataFrame，列: [subject_id, group, region1, region2, ...]

   - **统计分析**：
     * 组间比较: scipy.stats.ttest_ind(group1, group2)
     * 效应量: cohen_d = (mean1 - mean2) / pooled_std
     * 多重比较: statsmodels.stats.multitest.multipletests(p_values, method='fdr_bh')
     * 相关分析: scipy.stats.pearsonr(x, y) 或 spearmanr

   - **可视化**：
     * 必须生成的图表类型（根据任务）：
       - extraction: 无需图表，专注数据提取
       - analysis: 箱线图、小提琴图、散点图、热力图
       - visualization: 根据具体需求

3. **输出规范（必须严格执行）**：
   - **结果保存**：
     * 统计结果: 保存为 CSV（统计量表格）和 JSON（详细结果）
     * 图表文件: PNG 格式，300 DPI，文件名描述性强
     * 示例: 'group_comparison_boxplot.png', 'correlation_heatmap.png'

   - **图表保存（完整流程）**：
     ```python
     import matplotlib.pyplot as plt
     import seaborn as sns

     # 设置样式
     sns.set_style("whitegrid")
     plt.rcParams['figure.dpi'] = 300
     plt.rcParams['savefig.dpi'] = 300
     plt.rcParams['font.family'] = 'sans-serif'

     # 创建图表
     fig, ax = plt.subplots(figsize=(10, 6))
     # ... 绘图代码 ...

     # 优化布局
     plt.tight_layout()

     # 保存（使用绝对路径）
     output_path = Path(CONFIG['output_dir']) / 'figure_name.png'
     output_path.parent.mkdir(parents=True, exist_ok=True)
     plt.savefig(output_path, dpi=300, bbox_inches='tight')
     plt.close(fig)  # 释放内存

     print(f"图表已保存: {{output_path}}")
     ```

   - **输出目录**: {output_dir_prompt}
   - **进度打印**: 每个主要步骤都打印进度（加载数据、处理、保存等）

### 代码结构
```python
import os
import pandas as pd
import numpy as np
from pathlib import Path

# 配置
CONFIG = {{
    'input_dir': r'{source_dir}',           # 源文件所在目录
    'output_dir': r'{output_dir_prompt}',          # 输出目录
    'source_tool': '{source_tool}',
    'source_files': {output_files[:10]!r}{subjects_dir_line}
}}

def main():
    '''主函数'''
    pass

if __name__ == '__main__':
    main()
```

请生成完整的Python脚本，使用```python代码块包裹。
"""
        return prompt

    def _get_tool_specific_info(self, source_tool: str, tool_outputs: Dict) -> str:
        """获取工具特定的信息"""

        # 获取前序任务实际输出文件
        output_files = tool_outputs.get("output_files", [])
        json_files = [Path(f).name for f in output_files if f.endswith('.json')]
        csv_files = [Path(f).name for f in output_files if f.endswith('.csv')]

        if source_tool in ["freesurfer", "freesurfer_analysis"]:
            subjects_dir = tool_outputs.get("subjects_dir", tool_outputs.get("output_dir", ""))
            subjects_dir = subjects_dir.replace('\\', '/')  # 转换为正斜杠格式
            processed_subjects = tool_outputs.get("processed_subjects", [])
            return f"""
**FreeSurfer原始输出结构**：
- SUBJECTS_DIR: {subjects_dir}
- 被试列表: {processed_subjects}
- stats/aseg.stats: 皮下结构体积
- stats/lh.aparc.stats, stats/rh.aparc.stats: 皮层分区指标

**【aseg.stats 列索引 - 必须严格遵守！】**

aseg.stats 数据行格式示例：
```
1   4     15349    15689.4  Left-Lateral-Ventricle  33.0035  10.9615
```

列索引对应（使用 `parts = line.split()` 后）：
- `parts[0]` → Index (行号)
- `parts[1]` → SegId (分割ID)
- `parts[2]` → NVoxels (体素数量)
- `parts[3]` → **Volume_mm3 (体积)** ✅ 使用这个提取体积！
- `parts[4]` → **StructName (结构名称)** ✅ 使用这个作为字典键！
- `parts[5]` → normMean (平均强度)
- `parts[6]` → normStdDev (标准差) ❌ 不要用于结构名称！

**必须使用以下代码解析 aseg.stats（直接复制使用）**：
```python
def parse_aseg_stats(stats_file):
    \"\"\"解析aseg.stats文件，返回结构名称到体积的映射\"\"\"
    volumes = {{}}
    with open(stats_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                volume = float(parts[3])    # Volume_mm3 (第4列)
                struct_name = parts[4]      # StructName (第5列)
                volumes[struct_name] = volume
    return volumes
```

**常见错误（禁止！）**：
- ❌ `struct_name = parts[6]` → 这会得到数字如 "10.9615"
- ✅ `struct_name = parts[4]` → 这会得到 "Left-Lateral-Ventricle"

**前序任务已生成的文件（请直接使用）**：
- JSON文件: {json_files if json_files else '无'}
- CSV文件: {csv_files if csv_files else '无'}

**后处理脚本标准输出格式（你的代码应该读取这些格式）**：

1. **描述性统计JSON** (xxx_descriptive_stats.json, xxx_roi_results.json):
```json
{{
  "region_name": {{
    "HC": {{"n": 2, "mean": 4.0, "std": 0.1, "min": 3.9, "max": 4.1}},
    "SCA3": {{"n": 2, "mean": 3.8, "std": 0.2}}
  }}
}}
```

2. **组间比较JSON** (xxx_comparison_results.json):
```json
{{
  "region_name": {{
    "t_statistic": 1.5, "p_value": 0.23, "cohens_d": 0.8,
    "HC_mean": 4.0, "SCA3_mean": 3.8
  }}
}}
```

3. **体积CSV** (extracted_volumes.csv, brain_region_data.csv):
```csv
subject_id,group,region1,region2,...
HC1_0001,HC,4.0,3.5,...
```

**重要**：优先读取上面列出的已存在文件，不要假设存在其他格式的文件。
"""

        elif source_tool in ["spm", "spm_analysis"]:
            return f"""
**SPM原始输出结构**：
- c1*.nii: 灰质概率图
- c2*.nii: 白质概率图
- c3*.nii: 脑脊液概率图
- wc1*.nii: 标准化灰质
- swc1*.nii: 平滑后标准化灰质

**前序任务已生成的文件（请直接使用）**：
- JSON文件: {json_files if json_files else '无'}
- CSV文件: {csv_files if csv_files else '无'}

**后处理脚本标准输出格式**：

1. **体积CSV** (extracted_volumes.csv, gray_matter_volumes.csv):
```csv
subject_id,group,gm_volume,wm_volume,csf_volume
HC1_0001,HC,650.5,580.2,120.3
SCA3_0001,SCA3,620.3,560.1,115.2
```

2. **统计结果CSV** (group_comparison_stats.csv):
```csv
variable,t_stat,p_value,cohens_d,HC_mean,SCA3_mean
gm_volume,1.8,0.15,0.9,650.5,620.3
```

3. **描述性统计JSON** (xxx_descriptive_stats.json):
```json
{{
  "gm_volume": {{
    "HC": {{"mean": 650.5, "std": 30.2}},
    "SCA3": {{"mean": 620.3, "std": 25.1}}
  }}
}}
```

**重要**：读取已存在的CSV/JSON文件，不要重新从NIfTI提取。
"""

        elif source_tool in ["dsi_studio", "dsi_studio_analysis"]:
            # 获取 DSI Studio 路径
            try:
                from src.config_local_tools import DSI_STUDIO_PATH
                dsi_path = str(DSI_STUDIO_PATH).replace('\\', '/') if DSI_STUDIO_PATH else "未配置"
            except ImportError:
                dsi_path = "未配置"

            return f"""
**DSI Studio标准工作流程**：
1. action=src: 创建源文件 (NIfTI/DICOM → .src.gz)
2. action=rec: 重建 (.src.gz → .fib.gz, 包含FA/MD/AD/RD等指标)
3. action=trk: 纤维追踪 (.fib.gz → .tt.gz/.trk.gz)

**DSI Studio安装路径**: {dsi_path}

**原始输出结构**：
- .src.gz/.sz: 源数据文件（预处理后的DWI）
- .fib.gz: 纤维方向文件（包含FA, MD, AD, RD等DTI指标）
- .tt.gz/.trk.gz: 纤维束追踪结果
- connectivity_matrix.csv: 连接矩阵

**从.fib.gz文件提取DTI指标**：
```python
import scipy.io as sio
import numpy as np

# 加载.fib.gz文件（MATLAB格式）
fib_data = sio.loadmat(fib_path)

# 提取DTI指标（全脑平均值）
fa = fib_data.get('fa0', fib_data.get('fa', None))
if fa is not None:
    fa_mean = np.mean(fa[fa > 0])  # 排除0值

# 其他指标类似: md, ad, rd
```

**前序任务已生成的文件（请直接使用）**：
- FIB文件: {tool_outputs.get('fib_files', '无')}
- 纤维束文件: {tool_outputs.get('tract_files', '无')}
- JSON文件: {json_files if json_files else '无'}
- CSV文件: {csv_files if csv_files else '无'}

**后处理脚本标准输出格式**：

1. **DTI指标CSV** (dti_metrics.csv, fa_values.csv):
```csv
subject_id,group,FA_mean,MD_mean,AD_mean,RD_mean
HC1_0001,HC,0.45,0.0008,0.0012,0.0006
```

2. **组间比较JSON** (dti_comparison_results.json):
```json
{{
  "FA_mean": {{
    "t_statistic": 2.1, "p_value": 0.12, "cohens_d": 0.95,
    "HC_mean": 0.45, "SCA3_mean": 0.38
  }}
}}
```

**代码要求**：
1. 优先读取已存在的CSV/JSON文件
2. 如需从.fib.gz提取，使用scipy.io.loadmat()
3. DTI指标单位: FA无单位(0-1), MD/AD/RD单位mm²/s
"""

        elif source_tool in ["fsl", "fsl_analysis"]:
            # 获取atlas路径
            try:
                cerebellum_atlas = str(BRAIN_ATLASES["cerebellum"]["atlas_file"]).replace('\\', '/')
                cerebellum_label = str(BRAIN_ATLASES["cerebellum"]["label_file"]).replace('\\', '/')
                subcortical_atlas = str(BRAIN_ATLASES["subcortical"]["atlas_file"]).replace('\\', '/')
                atlas_dir = str(ATLAS_DIR).replace('\\', '/')
            except (KeyError, NameError):
                cerebellum_atlas = "未配置"
                cerebellum_label = "未配置"
                subcortical_atlas = "未配置"
                atlas_dir = "未配置"

            return f"""
**FSL原始输出结构**：
- bet输出: *_brain.nii.gz (颅骨剥离)
- fast输出: *_seg.nii.gz (分割), *_pve_*.nii.gz (概率图)
- flirt输出: *_registered.nii.gz (配准)
- first输出: *_all_fast_firstseg.nii.gz (皮下结构分割)
- **dtifit输出**: {{subject_id}}_dti_FA.nii.gz, {{subject_id}}_dti_MD.nii.gz, {{subject_id}}_dti_AD.nii.gz, {{subject_id}}_dti_RD.nii.gz

**【DTI处理特别说明 - 必读！】**：
1. **每个被试有独立的FA/MD/AD/RD文件**，文件名包含被试ID（如 HC1_0001_dti_FA.nii.gz）
2. **禁止使用通用glob模式**如 `glob('*_FA*.nii*')` - 这会匹配所有被试的文件！
3. **必须为每个被试分别读取其对应的文件**

**正确的被试文件处理方式**：
```python
# 正确示例：使用被试ID过滤文件
def get_subject_dti_files(all_files: list, subject_id: str) -> tuple:
    \"\"\"根据被试ID精确获取该被试的DTI文件\"\"\"
    fa = next((f for f in all_files if subject_id in f and '_FA' in f), None)
    md = next((f for f in all_files if subject_id in f and '_MD' in f), None)
    if not fa or not md:
        raise FileNotFoundError(f"被试 {{subject_id}} 的DTI文件未找到")
    return fa, md

# 处理所有被试
results = []
for subject_id, group in subjects_with_groups:
    fa_file, md_file = get_subject_dti_files(source_files, subject_id)
    fa_data = nib.load(fa_file).get_fdata()  # 该被试的FA数据
    md_data = nib.load(md_file).get_fdata()  # 该被试的MD数据
    # 提取该被试的ROI指标...
    results.append({{'subject_id': subject_id, 'group': group, ...}})
```

**错误示例（禁止使用！）**：
```python
# 错误！这会匹配所有被试的文件，导致每个被试读取相同数据
fa_files = glob.glob('*_FA*.nii*')
return fa_files[0]  # 每个被试都返回同一个文件！
```

**【重要】可用的脑图谱（请使用以下绝对路径）**：
- 小脑SUIT图谱: {cerebellum_atlas}
- 小脑标签文件: {cerebellum_label}
- 皮下结构图谱: {subcortical_atlas}
- 图谱目录: {atlas_dir}

**前序任务已生成的文件（请直接使用）**：
- JSON文件: {json_files if json_files else '无'}
- CSV文件: {csv_files if csv_files else '无'}

**后处理脚本标准输出格式**：

1. **体积CSV** (brain_volumes.csv, segmentation_volumes.csv):
```csv
subject_id,group,gm_volume,wm_volume,csf_volume,total_brain_volume
HC1_0001,HC,650.5,580.2,120.3,1350.0
```

2. **皮下结构/小脑ROI CSV** (subcortical_volumes.csv, cerebellar_volumes.csv):
```csv
subject_id,group,L_Thal,R_Thal,L_Caud,R_Caud,L_Puta,R_Puta
HC1_0001,HC,8.2,8.1,3.5,3.4,5.2,5.1
```

3. **组间比较JSON** (fsl_comparison_results.json):
```json
{{
  "gm_volume": {{
    "t_statistic": 1.8, "p_value": 0.15, "cohens_d": 0.82,
    "HC_mean": 650.5, "SCA3_mean": 620.3
  }}
}}
```

**代码要求**：
1. 使用上述绝对路径加载图谱，不要在输出目录内查找
2. 优先读取已存在的CSV/JSON文件，不要重新从NIfTI提取
3. 如果需要从NIfTI提取，确保FSL输出文件存在
"""

        elif source_tool == "scan_and_extract":
            # 当没有找到特定上游工具时，需要扫描整个run_dir
            run_dir = tool_outputs.get("run_dir", "")
            all_tool_outputs = tool_outputs.get("all_tool_outputs", {})

            available_tools_info = ""
            if all_tool_outputs:
                for tool_name, outputs in all_tool_outputs.items():
                    available_tools_info += f"\n- {tool_name}:"
                    available_tools_info += f"\n  目录: {outputs.get('output_dir', '')}"
                    files = outputs.get('output_files', [])
                    if files:
                        available_tools_info += f"\n  文件数: {len(files)}"
                        available_tools_info += f"\n  示例: {files[:3]}"

            return f"""
**数据扫描模式**：
需要扫描run_dir目录查找可用的处理输出

run_dir: {run_dir}

**已发现的工具输出**：
{available_tools_info if available_tools_info else "尚未发现工具输出，需要扫描"}

**扫描策略**：
1. 扫描 run_dir/tools/ 目录查找处理工具输出
2. 扫描 run_dir/data/ 目录查找原始数据
3. 查找以下类型的文件:
   - *.nii, *.nii.gz: NIfTI影像文件
   - *.csv, *.txt: 数据文件
   - *.stats: FreeSurfer统计文件
   - *.mat: MATLAB/SPM数据文件

**代码要求**：
1. 首先扫描目录结构，打印发现的文件
2. 根据发现的文件类型决定处理策略
3. 提取可用的脑影像指标（体积、表面积等）
4. 按被试和分组整理数据
"""

        else:
            return "通用神经影像处理输出"

    def _get_post_processing_system_prompt(self, source_tool: str) -> str:
        """获取后处理脚本生成的系统提示"""
        return f"""你是一位神经影像数据处理专家，擅长编写{source_tool.upper()}后处理脚本。

## 专业背景
- 熟悉{source_tool.upper()}的输出结构和文件格式
- 精通Python数据处理（pandas、numpy、nibabel）
- 了解统计分析方法（scipy.stats）
- 擅长数据可视化（matplotlib、seaborn）

##  绝对禁止
- **禁止生成任何模拟数据**：不能使用np.random生成脑区数据
- **禁止创建demo数据函数**：不能有create_demo_data()或类似函数
- **禁止使用use_demo_data配置**：不能有任何fallback到模拟数据的逻辑
- 如果输入文件不存在，必须raise FileNotFoundError，不能继续
- 所有处理的数据必须来自工具的真实输出

## 代码要求
1. **健壮性**：处理缺失文件、格式错误等异常（缺失时报错而非生成替代数据）
2. **可读性**：清晰的注释和变量命名
3. **可复现**：固定路径配置，便于重复使用
4. **输出完整**：包含数据摘要和保存确认
5. **真实数据**：所有分析必须基于工具的真实输出

## 特别注意
- Windows路径使用原始字符串 r'path'
- 检查文件是否存在再处理（不存在则报错）
- 打印详细的处理进度
- 保存所有结果到指定输出目录

## 输出格式
- 使用```python代码块包裹完整代码
- 代码应可直接运行"""


def create_vibe_coding_engine() -> VibeCodingEngine:
    """创建Vibe Coding引擎实例"""
    return VibeCodingEngine()
