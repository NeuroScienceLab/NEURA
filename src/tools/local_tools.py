"""
Local tool adapter - directly call installed neuroimaging analysis tools
Supported Tools: SPM, DPABI, DSI Studio, FreeSurfer, FSL
"""
import os
import json
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from src.tools.registry import (
    BaseTool, ToolDefinition, ToolCallRequest, ToolCallResult,
    Modality, ExecutorType
)
from src.config import OUTPUT_DIR, DATA_DIR
from src.config_local_tools import (
    run_matlab_script, run_dsi_studio, run_freesurfer, run_fsl, run_dcm2niix,
    windows_to_wsl_path, SPM_PATH, DPABI_PATH, DCM2NIIX_PATH, check_tools_availability,
    FSL_SUPPORTED_COMMANDS, FREESURFER_SUPPORTED_COMMANDS,
    FMRI_BATCH_SIZE
)

# 导出列表
__all__ = [
    'LocalSPMTool',
    'LocalDPABITool',
    'LocalDSIStudioTool',
    'LocalDiPyTool',
    'LocalFreeSurferTool',
    'LocalFSLTool',
    'LocalPythonStatsTool',
    'LocalDataVisualizationTool',
    'LocalDICOMConverterTool',
    'read_eddy_motion_parameters',  # FSL eddy QC 辅助函数
]


class LocalSPMTool(BaseTool):
    """本地SPM工具 - 直接调用MATLAB执行SPM分析"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spm_analysis",
            description="使用本地安装的SPM25进行神经影像分析，支持VBM、GLM统计分析",
            category="analysis",
            supported_modalities=[Modality.ANAT, Modality.FUNC],
            executor_type=ExecutorType.MATLAB,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入NIfTI文件路径列表"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": [
                            "vbm_segment",   # 结构像VBM分割
                            "smooth",        # 平滑
                            "normalize",     # 标准化
                            "realign",       # 头动校正（功能性MRI必需）
                            "slice_timing",  # 层时间校正（功能性MRI必需）
                            "coregister"     # 配准（结构像与功能像配准）
                        ],
                        "description": "分析类型：vbm_segment(VBM分割), smooth(平滑), normalize(标准化), realign(头动校正), slice_timing(层时间校正), coregister(配准)"
                    },
                    "reference_image": {
                        "type": "string",
                        "description": "配准参考图像路径（用于coregister）"
                    },
                    "slice_order": {
                        "type": "string",
                        "enum": ["ascending", "descending", "interleaved_ascending", "interleaved_descending"],
                        "default": "ascending",
                        "description": "层采集顺序（用于slice_timing）"
                    },
                    "tr": {
                        "type": "number",
                        "description": "重复时间TR（秒，用于slice_timing）"
                    },
                    "num_slices": {
                        "type": "integer",
                        "description": "层数（用于slice_timing）"
                    },
                    "smoothing_fwhm": {
                        "type": "number",
                        "default": 8,
                        "description": "平滑核大小(mm)"
                    },
                    "template": {
                        "type": "string",
                        "default": "TPM",
                        "description": "模板类型"
                    }
                },
                "required": ["input_files", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "log": {"type": "string"}
                }
            },
            version="SPM25",
            dependencies=["MATLAB R2019b", "SPM25"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行SPM分析 - 使用单被试逐个处理策略"""
        import shutil
        from src.config_local_tools import MATLAB_EXE
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # MATLAB 可用性预检查
        if not MATLAB_EXE or not MATLAB_EXE.exists():
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"MATLAB 未找到 (配置路径: {MATLAB_EXE})。请设置 MATLAB_ROOT 环境变量。",
                outputs={}, duration_seconds=0
            )

        analysis_type = request.params.get("analysis_type", "vbm_segment")
        input_files = request.inputs.get("input_files", [])
        smoothing_fwhm = request.params.get("smoothing_fwhm", 8)

        # 输入文件存在性检查
        missing = [f for f in input_files if not Path(f).exists()]
        if missing:
            return ToolCallResult(
                call_id=request.call_id, tool_name=self.definition.name,
                status="failed",
                error=f"输入文件不存在: {[str(f) for f in missing[:3]]}{'...' if len(missing) > 3 else ''}",
                outputs={}, duration_seconds=0
            )

        # 根据分析类型确定数据模态
        func_analysis_types = {"slice_timing", "realign", "coregister", "normalize"}
        modality = "func" if analysis_type in func_analysis_types else "anat"

        # 过滤掉已处理的文件（对于VBM分割，只使用原始T1图像）
        if analysis_type == "vbm_segment":
            filtered_inputs = self._filter_raw_t1_images([str(f) for f in input_files])
            if not filtered_inputs:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="没有找到原始T1图像（所有输入文件都已被SPM处理过）",
                    outputs={"modality": modality}
                )
            print(f"  [文件过滤] 过滤后剩余 {len(filtered_inputs)} 个原始T1图像")
            input_files = filtered_inputs

        # **新策略**: VBM分割使用单被试逐个处理，其他分析类型保持批处理
        if analysis_type == "vbm_segment":
            return self._execute_vbm_segment_sequential(request, input_files, output_dir, start_time)
        else:
            return self._execute_batch_analysis(request, input_files, output_dir, start_time, analysis_type, smoothing_fwhm)

    def _execute_vbm_segment_sequential(self, request: ToolCallRequest, input_files: List[str],
                                        output_dir: Path, start_time: datetime) -> ToolCallResult:
        """小批量处理VBM分割 - 平衡速度与稳定性的策略"""
        import shutil

        BATCH_SIZE = 8  # 每批处理8个被试
        total_subjects = len(input_files)
        processed_subjects = []
        skipped_subjects = []
        failed_subjects = []
        processing_log = []
        total_processing_time = 0  # 累积实际处理时间（不含检查跳过等开销）

        # 先检查哪些被试需要处理（断点续传）
        subjects_to_process = []
        for src_file in input_files:
            src_path = Path(src_file)

            # **关键修复**: 正确处理.nii.gz文件名
            # .nii.gz的stem是 HC1_0001.nii，需要再去掉.nii
            if src_path.suffix == '.gz' and src_path.stem.endswith('.nii'):
                # 例如: HC1_0001.nii.gz → subject_id=HC1_0001, nii_name=HC1_0001.nii
                nii_name = src_path.stem  # HC1_0001.nii
                subject_id = Path(nii_name).stem  # HC1_0001
            else:
                nii_name = src_path.name  # HC1_0001.nii
                subject_id = src_path.stem  # HC1_0001

            # c1输出文件基于.nii文件名（不含.gz）
            c1_output = output_dir / f"c1{nii_name}"

            if c1_output.exists():
                skipped_subjects.append(subject_id)
                processing_log.append(f"[跳过] {subject_id}: 检测到已有c1输出文件")
            else:
                subjects_to_process.append((src_file, subject_id))

        # 计算批次数
        num_batches = (len(subjects_to_process) + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'='*60}")
        print(f"  [VBM分割] 小批量处理策略 (每批{BATCH_SIZE}个被试)")
        print(f"  [总数] {total_subjects} 个被试")
        print(f"  [已完成] {len(skipped_subjects)} 个被试 (跳过)")
        print(f"  [待处理] {len(subjects_to_process)} 个被试")
        print(f"  [批次数] {num_batches} 批")
        print(f"{'='*60}\n")

        # 分批处理
        for batch_idx in range(num_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(subjects_to_process))
            batch = subjects_to_process[batch_start:batch_end]

            batch_num = batch_idx + 1
            batch_subjects = [subj_id for _, subj_id in batch]

            print(f"  [批次 {batch_num}/{num_batches}] 处理 {len(batch)} 个被试: {', '.join(batch_subjects[:3])}{'...' if len(batch) > 3 else ''}")

            # 复制文件并准备批处理输入
            # **关键修复**: SPM不能直接处理.nii.gz文件，需要解压为.nii
            import gzip
            batch_inputs = []
            batch_paths = []
            for src_file, subject_id in batch:
                src_path = Path(src_file)

                # 如果是.nii.gz文件，需要解压
                if src_path.suffix == '.gz' and src_path.stem.endswith('.nii'):
                    # 解压后的文件名（去掉.gz后缀）
                    dst_path = output_dir / src_path.stem  # 例如 HC1_0001.nii
                    if not dst_path.exists():
                        print(f"    [解压] {src_path.name} → {dst_path.name}")
                        with gzip.open(src_path, 'rb') as f_in:
                            with open(dst_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                else:
                    # 普通.nii文件直接复制
                    dst_path = output_dir / src_path.name
                    if not dst_path.exists():
                        shutil.copy2(src_path, dst_path)

                matlab_input = str(dst_path).replace("\\", "/")
                batch_inputs.append(matlab_input)
                batch_paths.append((dst_path, subject_id))

            # 生成批处理脚本
            matlab_output = str(output_dir).replace("\\", "/")
            script = self._generate_vbm_segment_script(batch_inputs, matlab_output)

            # 保存脚本
            script_path = output_dir / f"spm_vbm_segment_batch_{batch_num}.m"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            # 执行MATLAB批处理
            print(f"    执行中...", end=" ", flush=True)
            batch_start_time = datetime.now()

            try:
                result = run_matlab_script(script, str(output_dir))

                # 验证每个被试的输出文件
                batch_success = 0
                batch_failed = 0
                for dst_path, subject_id in batch_paths:
                    c1_output = output_dir / f"c1{dst_path.name}"
                    if c1_output.exists():
                        processed_subjects.append(subject_id)
                        processing_log.append(f"[成功] {subject_id}: VBM分割完成 (批次{batch_num})")
                        batch_success += 1
                    else:
                        failed_subjects.append(subject_id)
                        processing_log.append(f"[失败] {subject_id}: 未生成c1输出文件 (批次{batch_num})")
                        batch_failed += 1

                batch_duration = (datetime.now() - batch_start_time).total_seconds()
                total_processing_time += batch_duration  # 累积处理时间
                avg_time = batch_duration / len(batch)

                if batch_failed == 0:
                    print(f"✓ 全部成功 ({batch_duration:.1f}秒, 平均{avg_time:.1f}秒/被试)")
                else:
                    print(f"⚠ 部分失败 (成功:{batch_success}, 失败:{batch_failed}, 耗时:{batch_duration:.1f}秒)")

            except Exception as e:
                print(f"✗ 批次失败")
                batch_duration = (datetime.now() - batch_start_time).total_seconds()
                total_processing_time += batch_duration  # 累积处理时间

                # 标记整批失败
                for _, subject_id in batch:
                    failed_subjects.append(subject_id)
                    processing_log.append(f"[失败] {subject_id}: 批处理失败 - {str(e)}")

                print(f"    错误: {str(e)[:100]}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 收集所有输出文件
        output_files = []
        output_files.extend(list(output_dir.glob("*.nii")))
        output_files.extend(list(output_dir.glob("*.nii.gz")))
        output_files.extend(list(output_dir.glob("*.mat")))

        # 生成处理报告
        print(f"\n{'='*60}")
        print(f"  [处理完成]")
        print(f"  总数: {total_subjects} | 成功: {len(processed_subjects)} | "
              f"跳过: {len(skipped_subjects)} | 失败: {len(failed_subjects)}")
        print(f"  总耗时: {duration:.1f}秒 ({duration/60:.1f}分钟)")

        # 使用实际处理时间计算平均值（更精确）
        if len(processed_subjects) > 0:
            avg_per_subject = total_processing_time / len(processed_subjects)
            print(f"  平均处理时间: {avg_per_subject:.1f}秒/被试 (纯处理时间: {total_processing_time:.1f}秒)")
        elif len(subjects_to_process) > 0:
            # 有待处理的被试但都失败了
            avg_per_subject = total_processing_time / len(subjects_to_process)
            print(f"  平均处理时间: {avg_per_subject:.1f}秒/被试 (全部失败, 处理时间: {total_processing_time:.1f}秒)")

        print(f"{'='*60}\n")

        # 保存处理日志
        log_path = output_dir / "processing_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(processing_log))

        # 决定整体状态
        # 判断标准：看待处理的被试是否都失败，而不是看总数
        if len(subjects_to_process) > 0 and len(failed_subjects) == len(subjects_to_process):
            # 所有待处理的被试都失败了
            status = "failed"
            error = f"所有待处理的{len(subjects_to_process)}个被试处理失败"
        elif len(subjects_to_process) == 0:
            # 没有需要处理的被试（全部已完成）
            status = "succeeded"
            error = None
        elif len(failed_subjects) > 0:
            # 部分成功部分失败
            status = "succeeded"
            error = f"警告: {len(failed_subjects)}个被试处理失败: {', '.join(failed_subjects[:5])}"
        else:
            # 全部成功
            status = "succeeded"
            error = None

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status=status,
            duration_seconds=duration,
            outputs={
                "output_files": [str(f) for f in output_files],
                "modality": "anat",  # VBM分割始终处理解剖结构像
                "log": f"VBM分割完成。成功: {len(processed_subjects)}, 跳过: {len(skipped_subjects)}, 失败: {len(failed_subjects)}",
                "processed_count": len(processed_subjects),
                "skipped_count": len(skipped_subjects),
                "failed_count": len(failed_subjects),
                "failed_subjects": failed_subjects,
                "total_subjects": total_subjects,
                "processing_log": str(log_path)
            },
            error=error
        )

    def _execute_batch_analysis(self, request: ToolCallRequest, input_files: List[str],
                                output_dir: Path, start_time: datetime, analysis_type: str,
                                smoothing_fwhm: float) -> ToolCallResult:
        """批处理模式执行分析（用于非VBM分割任务）"""
        import shutil
        import gzip

        # 根据分析类型确定数据模态
        func_analysis_types = {"slice_timing", "realign", "coregister", "normalize"}
        modality = "func" if analysis_type in func_analysis_types else "anat"

        # **关键修复**: 将输入文件复制到output_dir，避免SPM输出污染源目录
        # **新增**: 自动解压.nii.gz文件，因为SPM不支持直接读取压缩格式
        copied_files = []
        for src_file in input_files:
            src_path = Path(src_file)

            # 检查是否为gzip压缩的NIfTI文件
            if src_path.suffix == '.gz' and src_path.stem.endswith('.nii'):
                # .nii.gz 文件需要解压
                decompressed_name = src_path.stem  # 去掉 .gz 后缀，保留 .nii
                dst_path = output_dir / decompressed_name

                if not dst_path.exists():
                    print(f"  [解压] {src_path.name} -> {decompressed_name} (SPM需要未压缩的NIfTI)")
                    try:
                        with gzip.open(src_path, 'rb') as f_in:
                            with open(dst_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    except Exception as e:
                        print(f"  [警告] 解压失败: {e}，尝试直接复制")
                        shutil.copy2(src_path, output_dir / src_path.name)
                        dst_path = output_dir / src_path.name

                copied_files.append(str(dst_path))
            else:
                # 普通文件直接复制
                dst_path = output_dir / src_path.name
                if not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    print(f"  [复制] {src_path.name} -> {output_dir.name}/")
                copied_files.append(str(dst_path))

        # 转换路径格式（用于MATLAB）
        matlab_inputs = [str(f).replace("\\", "/") for f in copied_files]
        matlab_output = str(output_dir).replace("\\", "/")

        # 生成SPM MATLAB脚本
        if analysis_type == "smooth":
            script = self._generate_smooth_script(matlab_inputs, matlab_output, smoothing_fwhm)
        elif analysis_type == "normalize":
            # 大批量fMRI文件分批处理，避免超时
            if len(matlab_inputs) > FMRI_BATCH_SIZE:
                return self._execute_fmri_batched(
                    request, matlab_inputs, output_dir, start_time,
                    analysis_type, modality, matlab_output
                )
            script = self._generate_normalize_script(matlab_inputs, matlab_output)
        elif analysis_type == "realign":
            # 大批量fMRI文件分批处理，避免SPM内存溢出
            if len(matlab_inputs) > FMRI_BATCH_SIZE:
                return self._execute_fmri_batched(
                    request, matlab_inputs, output_dir, start_time,
                    analysis_type, modality, matlab_output
                )
            script = self._generate_realign_script(matlab_inputs, matlab_output)
        elif analysis_type == "slice_timing":
            # 获取用户指定的参数（可能是默认值）
            user_tr = request.params.get("tr")
            user_num_slices = request.params.get("num_slices")
            slice_order = request.params.get("slice_order", "ascending")

            # 扫描所有文件的元数据，验证层数一致性
            tr = None
            num_slices = None

            if copied_files:
                from collections import Counter
                slice_counts = {}  # file -> num_slices
                tr_values = []

                print(f"  [slice_timing] 扫描所有 {len(copied_files)} 个文件的元数据...")
                for f in copied_files:
                    meta = self._get_nifti_fmri_metadata(f)
                    if meta.get("num_slices"):
                        slice_counts[f] = meta["num_slices"]
                    if meta.get("tr") and meta["tr"] not in tr_values:
                        tr_values.append(meta["tr"])

                # 确定TR（从第一个有效值）
                if tr_values:
                    tr = tr_values[0]
                    print(f"  [自动检测] TR = {tr:.3f}s (从NIfTI头读取)")
                    if user_tr and abs(user_tr - tr) > 0.1:
                        print(f"  [警告] 用户指定TR={user_tr}s 与NIfTI头信息不符，使用NIfTI值")

                # 检查层数一致性
                if slice_counts:
                    count_freq = Counter(slice_counts.values())
                    unique_counts = sorted(count_freq.keys())

                    if len(unique_counts) == 1:
                        # 所有文件层数一致
                        num_slices = unique_counts[0]
                        print(f"  [自动检测] 层数 = {num_slices} (所有 {len(slice_counts)} 个文件一致)")
                    else:
                        # 层数不一致，使用多数文件的层数
                        majority_slices = count_freq.most_common(1)[0][0]
                        majority_count = count_freq.most_common(1)[0][1]
                        num_slices = majority_slices
                        print(f"  [警告] 文件层数不一致! 分布: {dict(count_freq)}")
                        print(f"  [警告] 使用多数文件的层数 = {majority_slices} ({majority_count}/{len(slice_counts)} 个文件)")

                        # 过滤掉层数不匹配的文件
                        excluded = []
                        filtered_matlab_inputs = []
                        filtered_copied_files = []
                        for f, matlab_f in zip(copied_files, matlab_inputs):
                            if slice_counts.get(f) == majority_slices:
                                filtered_matlab_inputs.append(matlab_f)
                                filtered_copied_files.append(f)
                            else:
                                excluded.append((Path(f).name, slice_counts.get(f, "unknown")))

                        for name, sc in excluded:
                            print(f"  [排除] {name} (层数={sc}, 期望={majority_slices})")

                        matlab_inputs = filtered_matlab_inputs
                        copied_files = filtered_copied_files
                        print(f"  [slice_timing] 过滤后保留 {len(matlab_inputs)} 个文件")

                    if user_num_slices and user_num_slices != num_slices:
                        print(f"  [警告] 用户指定层数={user_num_slices} 与NIfTI头信息不符，使用NIfTI值")

            # 如果NIfTI读取失败，使用用户提供的参数作为后备
            if tr is None:
                tr = user_tr
                if tr:
                    print(f"  [后备] 使用用户指定TR = {tr}s")
            if num_slices is None:
                num_slices = user_num_slices
                if num_slices:
                    print(f"  [后备] 使用用户指定层数 = {num_slices}")

            # 验证必需参数
            missing_params = []
            if tr is None:
                missing_params.append("tr (重复时间，秒)")
            if num_slices is None:
                missing_params.append("num_slices (层数)")

            if missing_params:
                error_msg = (
                    f"slice_timing需要以下参数但无法自动获取:\n"
                    f"  - {', '.join(missing_params)}\n"
                    f"请在parameters中明确指定这些参数，或确保NIfTI文件包含正确的头信息。"
                )
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"modality": "func"}
                )

            if not matlab_inputs:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="过滤层数不一致的文件后，没有剩余文件可处理",
                    outputs={"modality": "func"}
                )

            print(f"  [slice_timing] 使用参数: TR={tr}s, 层数={num_slices}, 顺序={slice_order}")

            # 大批量fMRI文件分批处理，避免SPM内存溢出
            if len(matlab_inputs) > FMRI_BATCH_SIZE:
                return self._execute_fmri_batched(
                    request, matlab_inputs, output_dir, start_time,
                    analysis_type, modality, matlab_output,
                    tr=tr, num_slices=num_slices, slice_order=slice_order
                )
            script = self._generate_slice_timing_script(matlab_inputs, matlab_output, tr, num_slices, slice_order)
        elif analysis_type == "coregister":
            reference_image = request.params.get("reference_image") or request.inputs.get("reference_image")
            source_image = request.params.get("source_image") or request.inputs.get("source_image")

            if not reference_image:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="coregister需要reference_image参数（mean功能像）",
                    outputs={}
                )

            # 验证参考图像路径
            ref_path = Path(reference_image)
            if not ref_path.is_absolute():
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"reference_image必须是绝对路径，当前值: {reference_image}",
                    outputs={}
                )

            if not ref_path.exists():
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"reference_image文件不存在: {reference_image}",
                    outputs={}
                )

            # 处理参考图像（复制/解压到output_dir）
            if ref_path.suffix == '.gz' and ref_path.stem.endswith('.nii'):
                decompressed_name = ref_path.stem
                ref_dst_path = output_dir / decompressed_name
                if not ref_dst_path.exists():
                    print(f"  [解压参考图像] {ref_path.name} -> {decompressed_name}")
                    try:
                        with gzip.open(ref_path, 'rb') as f_in:
                            with open(ref_dst_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                    except Exception as e:
                        return ToolCallResult(
                            call_id=request.call_id,
                            tool_name=self.definition.name,
                            status="failed",
                            error=f"解压参考图像失败: {e}",
                            outputs={}
                        )
            else:
                ref_dst_path = output_dir / ref_path.name
                if not ref_dst_path.exists():
                    shutil.copy2(ref_path, ref_dst_path)
                    print(f"  [复制参考图像] {ref_path.name}")

            ref_matlab = str(ref_dst_path).replace("\\", "/")

            # 处理源图像（T1结构像）
            source_matlab = None
            if source_image:
                src_path = Path(source_image)
                if src_path.exists():
                    # 复制/解压源图像
                    if src_path.suffix == '.gz' and src_path.stem.endswith('.nii'):
                        src_decompressed = src_path.stem
                        src_dst_path = output_dir / src_decompressed
                        if not src_dst_path.exists():
                            print(f"  [解压源图像] {src_path.name} -> {src_decompressed}")
                            with gzip.open(src_path, 'rb') as f_in:
                                with open(src_dst_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                    else:
                        src_dst_path = output_dir / src_path.name
                        if not src_dst_path.exists():
                            shutil.copy2(src_path, src_dst_path)
                            print(f"  [复制源图像] {src_path.name}")
                    source_matlab = str(src_dst_path).replace("\\", "/")
                    print(f"  [coregister] 源图像(T1): {src_path.name}")

            print(f"  [coregister] 参考图像(mean): {ref_path.name}")

            # 生成配准脚本
            if source_matlab:
                # 有T1源图像：将T1配准到mean功能像
                script = self._generate_coregister_script_with_source(
                    ref_matlab, source_matlab, matlab_inputs, matlab_output
                )
            else:
                # 无T1源图像：将input_files配准到参考图像
                script = self._generate_coregister_script(matlab_inputs, matlab_output, ref_matlab)
        else:
            script = self._generate_basic_script(matlab_inputs, matlab_output, analysis_type)

        # 保存脚本
        script_path = output_dir / f"spm_{analysis_type}.m"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        # 执行MATLAB脚本
        result = run_matlab_script(script, str(output_dir))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 收集输出文件 - 现在所有输出都在output_dir中
        output_files = []

        # 从output_dir收集所有SPM生成的文件
        output_files.extend(list(output_dir.glob("*.nii")))
        output_files.extend(list(output_dir.glob("*.nii.gz")))
        output_files.extend(list(output_dir.glob("*.mat")))  # SPM也生成.mat文件

        # 转换为字符串路径
        output_files = [str(f) for f in output_files]

        # ========== 改进的状态判断逻辑 ==========
        # 不仅检查MATLAB退出码，还要检查是否生成了输出文件
        matlab_succeeded = result and result.get("status") == "succeeded"
        has_output = len(output_files) > 0

        # 检查stderr中是否有SPM错误
        stderr = result.get("stderr", "") if result else ""
        has_spm_error = "Job execution failed" in stderr or "Error" in stderr

        # 综合判断：MATLAB成功 且 有输出文件 且 无SPM错误
        if matlab_succeeded and has_output and not has_spm_error:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": output_files,
                    "modality": modality,
                    "analysis_type": analysis_type,
                    "script_path": str(script_path)
                },
                artifacts=[
                    {"name": "spm_script.m", "path": str(script_path)},
                    {"name": "matlab_log.txt", "path": str(output_dir / "matlab_log.txt")}
                ]
            )
        else:
            # 构建详细的错误信息
            error_parts = []
            if not matlab_succeeded:
                error_parts.append("MATLAB执行失败")
            if not has_output:
                error_parts.append(f"未生成输出文件(期望{analysis_type}的输出)")
            if has_spm_error:
                error_parts.append("SPM批处理失败")

            error_msg = "; ".join(error_parts)
            if stderr:
                error_msg += f"\n\nMATLAB错误:\n{stderr[:500]}"

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=error_msg,
                outputs={"modality": modality, "script_path": str(script_path), "output_files": output_files}
            )

    def _execute_fmri_batched(self, request: ToolCallRequest, matlab_inputs: List[str],
                               output_dir: Path, start_time: datetime,
                               analysis_type: str, modality: str, matlab_output: str,
                               **kwargs) -> ToolCallResult:
        """分批执行fMRI预处理（slice_timing/realign），避免SPM内存溢出。

        将大批量文件拆分为每批FMRI_BATCH_SIZE个，逐批调用SPM。
        每批完成后更新processing_log.json记录进度。
        """
        import time as _time

        total_files = len(matlab_inputs)
        total_batches = (total_files + FMRI_BATCH_SIZE - 1) // FMRI_BATCH_SIZE
        print(f"\n  [分批处理] {analysis_type}: {total_files}个文件 -> {total_batches}批 (每批最多{FMRI_BATCH_SIZE}个)")

        batch_log = []  # 每批次的处理记录
        succeeded_batches = 0
        failed_batches = 0
        all_script_paths = []

        for batch_idx in range(total_batches):
            batch_start = batch_idx * FMRI_BATCH_SIZE
            batch_end = min(batch_start + FMRI_BATCH_SIZE, total_files)
            batch_inputs = matlab_inputs[batch_start:batch_end]
            batch_num = batch_idx + 1
            batch_time_start = _time.time()

            print(f"\n  [批次 {batch_num}/{total_batches}] 处理 {len(batch_inputs)} 个文件...")

            # 生成该批次的脚本
            if analysis_type == "slice_timing":
                batch_script = self._generate_slice_timing_script(
                    batch_inputs, matlab_output,
                    kwargs["tr"], kwargs["num_slices"], kwargs["slice_order"]
                )
            elif analysis_type == "realign":
                batch_script = self._generate_realign_script(batch_inputs, matlab_output)
            elif analysis_type == "normalize":
                batch_script = self._generate_normalize_script(batch_inputs, matlab_output)
            else:
                batch_script = self._generate_basic_script(batch_inputs, matlab_output, analysis_type)

            # 保存并执行
            batch_script_path = output_dir / f"spm_{analysis_type}_batch{batch_num}.m"
            with open(batch_script_path, "w", encoding="utf-8") as f:
                f.write(batch_script)
            all_script_paths.append(str(batch_script_path))

            batch_result = run_matlab_script(batch_script, str(output_dir))
            batch_duration = _time.time() - batch_time_start

            # 判断批次结果
            batch_ok = batch_result and batch_result.get("status") == "succeeded"
            batch_stderr = batch_result.get("stderr", "") if batch_result else ""
            batch_has_error = "Job execution failed" in batch_stderr or "Error" in batch_stderr

            entry = {
                "batch": batch_num,
                "total_batches": total_batches,
                "files_count": len(batch_inputs),
                "duration_seconds": round(batch_duration, 1),
                "status": "succeeded" if (batch_ok and not batch_has_error) else "failed",
            }

            if batch_ok and not batch_has_error:
                succeeded_batches += 1
                print(f"  [批次 {batch_num}/{total_batches}] 完成 ({batch_duration:.0f}s)")
            else:
                failed_batches += 1
                error_detail = batch_result.get("error", batch_stderr[:200]) if batch_result else "unknown"
                entry["error"] = error_detail
                print(f"  [批次 {batch_num}/{total_batches}] 失败: {error_detail[:100]}")

            batch_log.append(entry)

            # 每批完成后写入进度日志
            log_path = output_dir / "processing_log.json"
            progress = {
                "analysis_type": analysis_type,
                "total_files": total_files,
                "batch_size": FMRI_BATCH_SIZE,
                "completed_batches": succeeded_batches + failed_batches,
                "total_batches": total_batches,
                "succeeded_batches": succeeded_batches,
                "failed_batches": failed_batches,
                "batches": batch_log
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)

        # 收集所有输出文件
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        output_files = []
        output_files.extend(list(output_dir.glob("*.nii")))
        output_files.extend(list(output_dir.glob("*.nii.gz")))
        output_files.extend(list(output_dir.glob("*.mat")))
        output_files = [str(f) for f in output_files]

        summary = f"分批处理完成: {succeeded_batches}/{total_batches}批成功, {failed_batches}批失败"
        print(f"\n  [分批处理] {summary}")

        if succeeded_batches > 0:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded" if failed_batches == 0 else "partial",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": output_files,
                    "modality": modality,
                    "analysis_type": analysis_type,
                    "batch_summary": summary,
                    "succeeded_batches": succeeded_batches,
                    "failed_batches": failed_batches,
                    "total_batches": total_batches,
                    "processing_log": str(log_path),
                    "script_paths": all_script_paths
                },
                artifacts=[
                    {"name": "processing_log.json", "path": str(log_path)}
                ]
            )
        else:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"所有{total_batches}个批次均失败",
                outputs={
                    "modality": modality,
                    "processing_log": str(log_path),
                    "output_files": output_files
                }
            )

    def _get_nifti_fmri_metadata(self, nii_file: str) -> dict:
        """
        从NIfTI文件头读取fMRI元数据（TR和层数）

        Args:
            nii_file: NIfTI文件路径

        Returns:
            dict: 包含tr（秒）和num_slices的字典，无法读取时为None
        """
        try:
            import nibabel as nib
            img = nib.load(nii_file)
            header = img.header

            # 获取TR（第4维的时间间隔，单位通常是秒或毫秒）
            tr = None
            if len(img.shape) >= 4:
                zooms = header.get_zooms()
                if len(zooms) >= 4:
                    tr_value = float(zooms[3])
                    # 检查单位：如果TR > 100，可能是毫秒
                    if tr_value > 100:
                        tr = tr_value / 1000.0  # 转换为秒
                    elif tr_value > 0:
                        tr = tr_value

            # 获取层数（第3维）
            num_slices = None
            if len(img.shape) >= 3:
                num_slices = img.shape[2]

            return {
                "tr": tr,
                "num_slices": num_slices,
                "shape": img.shape
            }
        except Exception as e:
            print(f"  [警告] 无法读取NIfTI元数据: {e}")
            return {"tr": None, "num_slices": None, "shape": None}

    def _filter_raw_t1_images(self, input_files: List[str]) -> List[str]:
        """
        过滤出原始T1图像，排除已被SPM处理过的文件

        SPM处理过的文件通常有以下前缀：
        - c1, c2, c3, c4, c5, c6 (分割后的组织概率图)
        - wc1, wc2, wc3 (标准化后的组织概率图)
        - sw (平滑+标准化的图像)
        - sc1, sc2, sc3 (平滑后的组织概率图)
        - rp_ (重定向参数文件)

        注意：dcm2niix会添加's'前缀表示series number（如sC00003.nii），
        这是原始文件，不应该被过滤！
        """
        from pathlib import Path

        raw_images = []

        for file_path in input_files:
            filename = Path(file_path).name
            is_processed = False

            # 检查SPM组织概率图（c1, c2, c3等开头，但不是以s开头的原始文件）
            if filename.startswith('c1') or filename.startswith('c2') or filename.startswith('c3') or \
               filename.startswith('c4') or filename.startswith('c5') or filename.startswith('c6'):
                # 但如果前面还有s（如sc1），则是平滑后的，需要过滤
                is_processed = True

            # 检查标准化的组织概率图
            elif filename.startswith('wc1') or filename.startswith('wc2') or filename.startswith('wc3'):
                is_processed = True

            # 检查平滑后的文件（sw = smoothed+warped, sc1 = smoothed c1）
            elif filename.startswith('sw') or filename.startswith('sc1') or \
                 filename.startswith('sc2') or filename.startswith('sc3'):
                is_processed = True

            # 检查重定向参数文件
            elif filename.startswith('rp_'):
                is_processed = True

            # 检查多次处理（文件名中有多个c1/c2前缀）
            elif filename.count('c1') > 1 or filename.count('c2') > 1:
                is_processed = True

            if not is_processed:
                raw_images.append(file_path)
                print(f"  [保留原始] {filename}")
            else:
                print(f"  [跳过已处理] {filename}")

        return raw_images

    def _generate_vbm_segment_script(self, inputs: List[str], output_dir: str) -> str:
        """生成VBM分割脚本"""
        files_cell = "{\n" + "\n".join([f"    '{f},1'" for f in inputs]) + "\n}"

        return f"""% SPM VBM Segmentation Script
% Generated by Research Agent

spm('defaults', 'fmri');
spm_jobman('initcfg');

% Define batch job
matlabbatch{{1}}.spm.spatial.preproc.channel.vols = {files_cell};
matlabbatch{{1}}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{{1}}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{{1}}.spm.spatial.preproc.channel.write = [0 1];

% Tissue probability maps
tpm_path = fullfile(spm('Dir'), 'tpm', 'TPM.nii');
for i = 1:6
    matlabbatch{{1}}.spm.spatial.preproc.tissue(i).tpm = {{[tpm_path ',' num2str(i)]}};
    matlabbatch{{1}}.spm.spatial.preproc.tissue(i).ngaus = i;
    matlabbatch{{1}}.spm.spatial.preproc.tissue(i).native = [1 0];  % 写入native space组织图
    if i <= 2
        % 对灰质和白质，写入调制标准化的组织图（标准VBM）
        matlabbatch{{1}}.spm.spatial.preproc.tissue(i).warped = [1 0];  % [modulated, unmodulated]
    else
        % 其他组织类型不需要标准化
        matlabbatch{{1}}.spm.spatial.preproc.tissue(i).warped = [0 0];
    end
end

matlabbatch{{1}}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{{1}}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{{1}}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{{1}}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{{1}}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{{1}}.spm.spatial.preproc.warp.write = [0 1];

% Run batch
spm_jobman('run', matlabbatch);

disp('VBM Segmentation completed successfully!');
"""

    def _generate_smooth_script(self, inputs: List[str], output_dir: str, fwhm: float) -> str:
        """生成平滑脚本 - 正确处理4D fMRI数据"""
        # 构建输入文件列表（不带volume索引，让MATLAB自动展开4D文件）
        files_list = "\n".join([f"    '{f}'" for f in inputs])

        return f"""% SPM Smoothing Script - 支持4D fMRI数据
spm('defaults', 'fmri');
spm_jobman('initcfg');

% 输入文件列表
input_files = {{
{files_list}
}};

% 展开4D文件为所有volumes
all_volumes = {{}};
for i = 1:length(input_files)
    f = input_files{{i}};
    if ~exist(f, 'file')
        fprintf('Warning: File not found: %s\\n', f);
        continue;
    end

    % 使用spm_vol获取volume数量
    try
        V = spm_vol(f);
        n_vols = length(V);
        fprintf('File %s has %d volumes\\n', f, n_vols);

        if n_vols > 1
            % 4D文件：展开所有volumes
            for v = 1:n_vols
                all_volumes{{end+1, 1}} = sprintf('%s,%d', f, v);
            end
        else
            % 3D文件：直接添加
            all_volumes{{end+1, 1}} = sprintf('%s,1', f);
        end
    catch ME
        fprintf('Error reading %s: %s\\n', f, ME.message);
        % 尝试直接添加
        all_volumes{{end+1, 1}} = sprintf('%s,1', f);
    end
end

fprintf('Total volumes to smooth: %d\\n', length(all_volumes));

if isempty(all_volumes)
    error('No valid input files found!');
end

% 配置平滑任务
matlabbatch{{1}}.spm.spatial.smooth.data = all_volumes;
matlabbatch{{1}}.spm.spatial.smooth.fwhm = [{fwhm} {fwhm} {fwhm}];
matlabbatch{{1}}.spm.spatial.smooth.dtype = 0;
matlabbatch{{1}}.spm.spatial.smooth.im = 0;
matlabbatch{{1}}.spm.spatial.smooth.prefix = 's';

spm_jobman('run', matlabbatch);
disp('Smoothing completed!');
"""

    def _generate_normalize_script(self, inputs: List[str], output_dir: str) -> str:
        """
        生成标准化脚本 - 支持两种模式:
        1. fMRI模式: 使用SPM Normalise: Estimate & Write，直接将功能像标准化到MNI空间
        2. VBM模式: 使用分割产生的形变场(y_*.nii)应用到组织概率图

        自动检测输入类型:
        - 如果输入是4D fMRI数据(r*.nii等)，使用fMRI模式
        - 如果输入包含c1*文件，使用VBM模式
        """
        # 检查是否已有调制标准化的灰质文件（mwc1*）- 这意味着分割时已经写入了标准化输出
        mwc1_files = [f for f in inputs if Path(f).name.startswith('mwc1')]

        if mwc1_files:
            # 已经有标准化文件，无需再做normalize.write
            return f"""% SPM Normalization - Skipped (already normalized)
% 分割步骤已输出调制标准化的组织图(mwc1*, mwc2*)
disp('标准化步骤已在分割时完成，跳过单独的标准化');
disp('找到 {len(mwc1_files)} 个mwc1文件');
"""

        # 如果没有mwc1文件，检查是否有wc1文件（未调制的标准化文件）
        wc1_files = [f for f in inputs if Path(f).name.startswith('wc1') and not Path(f).name.startswith('mwc1')]
        if wc1_files:
            return f"""% SPM Normalization - Skipped (already normalized)
% 找到未调制的标准化文件(wc1*)
disp('找到 {len(wc1_files)} 个wc1文件，标准化已完成');
"""

        # 检查是否有VBM分割文件
        c1_files = [f for f in inputs if Path(f).name.startswith('c1')]

        # 检查是否是fMRI数据（4D文件，通常有r前缀表示realigned）
        # fMRI文件特征: r*.nii, 不是c1/c2/c3开头，不是mean开头
        fmri_files = []
        for f in inputs:
            fname = Path(f).name.lower()
            # 排除VBM相关文件和mean文件
            if fname.startswith(('c1', 'c2', 'c3', 'c4', 'c5', 'mwc', 'wc', 'y_', 'mean')):
                continue
            # 包含realigned文件(r前缀)或原始功能像
            if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                fmri_files.append(f)

        # 如果有fMRI文件但没有c1文件，使用fMRI标准化模式
        if fmri_files and not c1_files:
            return self._generate_fmri_normalize_script(fmri_files, output_dir)

        # VBM模式：使用形变场进行标准化
        if not c1_files:
            # 既没有c1文件也没有有效的fMRI文件
            return """% SPM Normalization - Skipped
% 未找到可标准化的文件（需要c1*分割文件或fMRI数据）
disp('未找到可标准化的文件');
"""

        # 构建MATLAB脚本 - 使用单个batch job with多个subjects (VBM模式)
        script_lines = [
            "% SPM Normalization Script (VBM - Apply Deformations)",
            "spm('defaults', 'fmri');",
            "spm_jobman('initcfg');",
            "",
            "% 创建单个normalise.write任务with多个被试"
        ]

        # 为每个文件添加一个subject条目到同一个batch job
        for idx, file_path in enumerate(c1_files, 1):
            # 获取对应的形变场文件（y_开头的文件）
            file_dir = Path(file_path).parent
            file_stem = Path(file_path).stem.replace('c1', '')  # 移除c1前缀
            deform_file = file_dir / f"y_{file_stem}.nii"

            # 转换路径为MATLAB格式（使用正斜杠）
            deform_path_matlab = str(deform_file).replace(chr(92), '/')
            file_path_matlab = file_path.replace(chr(92), '/')

            script_lines.extend([
                f"",
                f"% 被试 {idx}: {Path(file_path).name}",
                f"matlabbatch{{1}}.spm.spatial.normalise.write.subj({idx}).def = {{'{deform_path_matlab}'}};",
                f"matlabbatch{{1}}.spm.spatial.normalise.write.subj({idx}).resample = {{'{file_path_matlab}'}};"
            ])

        # 添加通用的写入选项（对所有被试都适用）
        script_lines.extend([
            "",
            "% 写入选项（对所有被试适用）",
            "matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70; 78 76 85];",
            "matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];",
            "matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;",
            "matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'w';",
            "",
            "% 运行批处理",
            "spm_jobman('run', matlabbatch);",
            "disp('Normalization completed!');"
        ])

        return "\n".join(script_lines)

    def _generate_fmri_normalize_script(self, inputs: List[str], output_dir: str) -> str:
        """
        生成fMRI标准化脚本 - 使用SPM Normalise: Estimate & Write
        直接将功能像标准化到MNI空间，不需要分割步骤

        这是fMRI预处理流程中的标准化步骤:
        slice_timing -> realign -> coregister -> **normalize** -> smooth

        使用SPM12的EPI模板进行标准化
        """
        files_cell = "{\n" + "\n".join([f"    '{f}'" for f in inputs]) + "\n}"

        return f"""% SPM Normalization Script (fMRI - Estimate & Write)
% Generated by Research Agent
% 用于fMRI预处理流程的标准化步骤
% 使用SPM12的Normalise: Estimate & Write直接标准化到MNI空间
% 参考文档: https://www.fil.ion.ucl.ac.uk/spm/docs/manual/preprocessing/normalise

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 输入的4D fMRI文件列表
input_files = {files_cell};
n_subjects = length(input_files);

fprintf('=== SPM fMRI Normalization ===\\n');
fprintf('被试数量: %d\\n', n_subjects);

% 获取SPM路径以找到EPI模板
spm_path = spm('Dir');
epi_template = fullfile(spm_path, 'toolbox', 'OldNorm', 'EPI.nii');

% 检查模板是否存在
if ~exist(epi_template, 'file')
    % 尝试其他可能的位置
    epi_template = fullfile(spm_path, 'templates', 'EPI.nii');
end
if ~exist(epi_template, 'file')
    % SPM12新位置
    epi_template = fullfile(spm_path, 'canonical', 'avg152T1.nii');
    fprintf('使用T1模板作为参考\\n');
end

fprintf('使用模板: %s\\n', epi_template);

% 为每个被试创建标准化任务
for subj = 1:n_subjects
    file_path = input_files{{subj}};
    fprintf('\\n处理被试 %d: %s\\n', subj, file_path);

    % 检查文件是否存在
    if ~exist(file_path, 'file')
        fprintf('  [警告] 文件不存在，跳过: %s\\n', file_path);
        continue;
    end

    % 获取4D文件中的所有volumes
    V = spm_vol(file_path);
    n_vols = length(V);
    fprintf('  Volumes数量: %d\\n', n_vols);

    % 构建该被试的volumes列表
    vol_list = cell(n_vols, 1);
    for v = 1:n_vols
        vol_list{{v}} = sprintf('%s,%d', file_path, v);
    end

    % 配置Normalise: Estimate & Write
    clear matlabbatch;

    % 估计选项 - 使用第一个volume作为源图像
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.subj.vol = {{sprintf('%s,1', file_path)}};
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.subj.resample = vol_list;

    % 估计选项
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.0001;
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.tpm = {{fullfile(spm_path, 'tpm', 'TPM.nii')}};
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.eoptions.samp = 3;

    % 写入选项 - fMRI标准体素大小3mm
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70; 78 76 85];
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.woptions.vox = [3 3 3];  % fMRI常用3mm体素
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.woptions.interp = 4;
    matlabbatch{{1}}.spm.spatial.normalise.estwrite.woptions.prefix = 'w';

    % 运行该被试的标准化
    fprintf('  开始标准化...\\n');
    spm_jobman('run', matlabbatch);
    fprintf('  被试 %d 标准化完成\\n', subj);
end

fprintf('\\n=== 所有被试标准化完成 ===\\n');
disp('Normalization completed!');
"""

    def _generate_basic_script(self, inputs: List[str], output_dir: str, analysis_type: str) -> str:
        """生成基本分析脚本"""
        return f"""% SPM Basic Script - {analysis_type}
spm('defaults', 'fmri');
disp('SPM initialized for {analysis_type}');
disp('Input files:');
disp({inputs});
disp('Output directory: {output_dir}');
"""

    def _generate_realign_script(self, inputs: List[str], output_dir: str) -> str:
        """
        生成头动校正脚本（Realign: Estimate & Reslice）
        用于功能性MRI的头动校正，输出：
        - r*.nii: 重新对齐的图像
        - rp_*.txt: 六参数头动文件
        - mean*.nii: 平均图像

        注意：每个4D文件作为独立的session处理，不会混合不同被试的volumes
        """
        # 构建4D文件列表（每个文件是一个session）
        files_cell = "{\n" + "\n".join([f"    '{f}'" for f in inputs]) + "\n}"

        return f"""% SPM Realign: Estimate & Reslice Script
% Generated by Research Agent
% 用于功能性MRI头动校正
% 每个4D文件作为独立session处理
% 参考文档: https://www.fil.ion.ucl.ac.uk/spm/docs/manual/preprocessing/realign

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 输入的4D文件列表（每个文件是一个session/被试）
input_files = {files_cell};
n_sessions = length(input_files);

% ===== 文件存在性验证 =====
fprintf('检查输入文件...\\n');
for i = 1:n_sessions
    if ~exist(input_files{{i}}, 'file')
        error('输入文件不存在: %s', input_files{{i}});
    end
    fprintf('  [OK] Session %d: %s\\n', i, input_files{{i}});
end
fprintf('文件验证通过。共 %d 个sessions。\\n\\n', n_sessions);

% 为每个session构建volumes列表
% SPM realign.estwrite.data 需要格式: {{{{session1_vols}}, {{session2_vols}}, ...}}
session_data = cell(1, n_sessions);

for sess = 1:n_sessions
    file_path = input_files{{sess}};

    % 获取4D文件中的所有volumes
    V = spm_vol(file_path);
    n_vols = length(V);
    fprintf('Session %d: %s 包含 %d 个volumes\\n', sess, file_path, n_vols);

    % 检查是否为4D数据
    if n_vols < 2
        error('文件 %s 不是4D fMRI数据（只有 %d 个volume）。请确保输入为4D NIfTI文件。', file_path, n_vols);
    end

    % 构建该session的volumes列表
    session_vols = cell(n_vols, 1);
    for v = 1:n_vols
        session_vols{{v}} = sprintf('%s,%d', file_path, v);
    end
    session_data{{sess}} = session_vols;
end

% Realign设置 - 每个session独立处理
matlabbatch{{1}}.spm.spatial.realign.estwrite.data = session_data;

% 估计选项（按照SPM12默认参数）
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;    % 质量（0-1）
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.sep = 4;          % 采样间隔(mm)
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;         % 平滑核大小
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.rtm = 1;          % 配准到平均图像
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.interp = 2;       % 插值方法（2阶B样条）
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];   % 不环绕
matlabbatch{{1}}.spm.spatial.realign.estwrite.eoptions.weight = '';      % 无加权

% 重采样选项
matlabbatch{{1}}.spm.spatial.realign.estwrite.roptions.which = [2 1];    % [所有图像 + 平均图像]
matlabbatch{{1}}.spm.spatial.realign.estwrite.roptions.interp = 4;       % 4阶B样条插值
matlabbatch{{1}}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
matlabbatch{{1}}.spm.spatial.realign.estwrite.roptions.mask = 1;         % 掩码处理
matlabbatch{{1}}.spm.spatial.realign.estwrite.roptions.prefix = 'r';     % 输出前缀

% 执行
fprintf('开始执行头动校正...\\n');
try
    spm_jobman('run', matlabbatch);
    disp('Realignment completed successfully!');
    fprintf('处理了 %d 个sessions\\n', n_sessions);
    disp('输出文件:');
    disp('  - r*.nii: 头动校正后的图像');
    disp('  - rp_*.txt: 六参数头动文件');
    disp('  - mean*.nii: 平均图像');
catch ME
    fprintf('Error in realignment: %s\\n', ME.message);
    fprintf('详细信息: %s\\n', getReport(ME, 'extended'));
    rethrow(ME);
end
"""

    def _generate_slice_timing_script(self, inputs: List[str], output_dir: str,
                                       tr: float, num_slices: int, slice_order: str) -> str:
        """
        生成层时间校正脚本（Slice Timing Correction）
        用于校正fMRI不同层的采集时间差异，输出：
        - a*.nii: 层时间校正后的图像

        注意：每个4D文件作为独立的session处理
        """
        # 构建4D文件列表（每个文件是一个session）
        files_cell = "{\n" + "\n".join([f"    '{f}'" for f in inputs]) + "\n}"

        # 根据slice_order生成MATLAB层顺序数组
        slice_order_map = {
            "ascending": f"1:{num_slices}",  # 升序 1,2,3,...,N
            "descending": f"{num_slices}:-1:1",  # 降序 N,...,3,2,1
            "interleaved_ascending": f"[1:2:{num_slices} 2:2:{num_slices}]",  # 交错升序 1,3,5,...,2,4,6,...
            "interleaved_descending": f"[{num_slices}:-2:1 {num_slices-1}:-2:1]"  # 交错降序
        }
        matlab_slice_order = slice_order_map.get(slice_order, f"1:{num_slices}")

        # 计算TA（采集时间）= TR - TR/nslices
        ta = tr - tr / num_slices

        return f"""% SPM Slice Timing Correction Script
% Generated by Research Agent
% 用于校正fMRI不同层的采集时间差异
% 每个4D文件作为独立session处理
% 参考文档: https://www.fil.ion.ucl.ac.uk/spm/docs/manual/preprocessing/slicetiming

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 参数
TR = {tr};           % 重复时间（秒）
nslices = {num_slices};  % 层数
TA = {ta:.6f};       % 采集时间 = TR - TR/nslices

% 层顺序: {slice_order}
slice_order = {matlab_slice_order};

% 参考层（使用中间层）
refslice = round(nslices / 2);

% 输入的4D文件列表（每个文件是一个session/被试）
input_files = {files_cell};
n_sessions = length(input_files);

% ===== 文件存在性验证 =====
fprintf('检查输入文件...\\n');
for i = 1:n_sessions
    if ~exist(input_files{{i}}, 'file')
        error('输入文件不存在: %s', input_files{{i}});
    end
    fprintf('  [OK] Session %d: %s\\n', i, input_files{{i}});
end
fprintf('文件验证通过。共 %d 个sessions。\\n\\n', n_sessions);

% 为每个session构建volumes列表，同时验证层数一致性
session_data = cell(1, n_sessions);
valid_sessions = true(1, n_sessions);

for sess = 1:n_sessions
    file_path = input_files{{sess}};

    try
        V = spm_vol(file_path);
        n_vols = length(V);

        % 验证层数与预期一致
        actual_slices = V(1).dim(3);
        if actual_slices ~= nslices
            fprintf('  [跳过] Session %d: %s 层数=%d (期望=%d)\\n', sess, file_path, actual_slices, nslices);
            valid_sessions(sess) = false;
            continue;
        end

        fprintf('Session %d: %s 包含 %d 个volumes (层数=%d)\\n', sess, file_path, n_vols, actual_slices);

        % 检查是否为4D数据
        if n_vols < 2
            error('文件 %s 不是4D fMRI数据（只有 %d 个volume）。请确保输入为4D NIfTI文件。', file_path, n_vols);
        end

        % 构建该session的volumes列表
        session_vols = cell(n_vols, 1);
        for v = 1:n_vols
            session_vols{{v}} = sprintf('%s,%d', file_path, v);
        end
        session_data{{sess}} = session_vols;
    catch ME
        fprintf('错误: 无法读取文件 %s: %s\\n', file_path, ME.message);
        rethrow(ME);
    end
end

% 移除层数不匹配的session
session_data = session_data(valid_sessions);
n_valid = sum(valid_sessions);
n_skipped = n_sessions - n_valid;
if n_skipped > 0
    fprintf('\\n[警告] 跳过 %d 个层数不匹配的文件，保留 %d 个文件\\n', n_skipped, n_valid);
end
if n_valid == 0
    error('没有层数匹配的文件可处理（期望层数=%d）', nslices);
end

% Slice Timing设置 - 每个session独立处理
matlabbatch{{1}}.spm.temporal.st.scans = session_data;
matlabbatch{{1}}.spm.temporal.st.nslices = nslices;
matlabbatch{{1}}.spm.temporal.st.tr = TR;
matlabbatch{{1}}.spm.temporal.st.ta = TA;
matlabbatch{{1}}.spm.temporal.st.so = slice_order;
matlabbatch{{1}}.spm.temporal.st.refslice = refslice;
matlabbatch{{1}}.spm.temporal.st.prefix = 'a';  % 输出前缀

% 执行
fprintf('开始执行层时间校正...\\n');
try
    spm_jobman('run', matlabbatch);
    disp('Slice timing correction completed successfully!');
    fprintf('处理了 %d 个sessions (跳过 %d 个层数不匹配)\\n', n_valid, n_skipped);
    disp('输出文件: a*.nii (层时间校正后的图像)');
    fprintf('使用参数: TR=%.2fs, 层数=%d, 层顺序=%s, 参考层=%d\\n', TR, nslices, '{slice_order}', refslice);
catch ME
    fprintf('Error in slice timing: %s\\n', ME.message);
    fprintf('详细信息: %s\\n', getReport(ME, 'extended'));
    rethrow(ME);
end
"""

    def _generate_coregister_script(self, inputs: List[str], output_dir: str, reference_image: str) -> str:
        """
        生成配准脚本（Coregister: Estimate & Reslice）
        用于将功能像与结构像配准，或多模态配准
        输出：r*.nii（配准后的图像）

        fMRI配准工作流说明：
        - 参考图像（reference）：通常是平均功能像(mean*.nii)或结构像
        - 源图像（source）：要配准的图像，通常是第一个功能像
        - 其他图像（other）：与source一起变换的其他功能像

        参考SPM12官方文档：
        https://www.fil.ion.ucl.ac.uk/spm/docs/manual/preprocessing/coregistration
        """
        # 第一个输入文件作为source（要移动的图像）
        source_file = inputs[0] if inputs else ""
        # 其他输入文件作为other（跟随source一起移动）
        other_files = inputs[1:] if len(inputs) > 1 else []

        other_cell = ""
        if other_files:
            other_cell = "{\n" + "\n".join([f"    '{f},1'" for f in other_files]) + "\n}"
        else:
            other_cell = "{''}"

        return f"""% SPM Coregister: Estimate & Reslice Script
% Generated by Research Agent
% 用于图像配准（如功能像到结构像）
% 参考文档: https://www.fil.ion.ucl.ac.uk/spm/docs/manual/preprocessing/coregistration

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 参考图像（固定不动）
ref_image_path = '{reference_image}';
ref_image = [ref_image_path ',1'];

% 源图像（要移动以匹配参考图像）
source_image_path = '{source_file}';
source_image = [source_image_path ',1'];

% 其他图像（跟随源图像一起变换）
other_images = {other_cell};

% ===== 文件存在性验证 =====
fprintf('检查输入文件...\\n');

if ~exist(ref_image_path, 'file')
    error('参考图像不存在: %s', ref_image_path);
end
fprintf('  [OK] 参考图像: %s\\n', ref_image_path);

if ~exist(source_image_path, 'file')
    error('源图像不存在: %s', source_image_path);
end
fprintf('  [OK] 源图像: %s\\n', source_image_path);

% 检查其他图像
for i = 1:length(other_images)
    other_path = other_images{{i}};
    % 移除 ,1 后缀获取文件路径
    comma_idx = strfind(other_path, ',');
    if ~isempty(comma_idx)
        other_file = other_path(1:comma_idx(1)-1);
    else
        other_file = other_path;
    end
    if ~isempty(other_file) && ~exist(other_file, 'file')
        error('其他图像不存在: %s', other_file);
    end
end
fprintf('  [OK] 其他图像: %d 个\\n', length(other_images));
fprintf('文件验证通过。\\n\\n');

% ===== Coregister: Estimate & Reslice 设置 =====
matlabbatch{{1}}.spm.spatial.coreg.estwrite.ref = {{ref_image}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.source = {{source_image}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.other = other_images;

% 估计选项（按照SPM12默认参数）
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';   % 归一化互信息
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];        % 采样间隔（粗到细）
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];       % 高斯平滑核FWHM

% 重采样选项
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.interp = 4;         % 4阶B样条插值
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];     % 不环绕
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.mask = 0;           % 不使用掩码
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';       % 输出前缀

% 执行配准
fprintf('开始执行配准...\\n');
try
    spm_jobman('run', matlabbatch);
    disp('Coregistration completed successfully!');
    disp('输出文件: r*.nii (配准后的图像)');
    fprintf('参考图像: %s\\n', ref_image_path);
    fprintf('源图像: %s\\n', source_image_path);
catch ME
    fprintf('Error in coregistration: %s\\n', ME.message);
    fprintf('详细信息: %s\\n', getReport(ME, 'extended'));
    rethrow(ME);
end
"""

    def _generate_coregister_script_with_source(self, reference_image: str, source_image: str,
                                                  other_inputs: List[str], output_dir: str) -> str:
        """
        生成配准脚本 - 将T1结构像配准到mean功能像

        标准fMRI预处理流程：
        - reference (固定): mean功能像 (realign输出)
        - source (移动): T1结构像
        - other (跟随source移动): 其他需要配准的图像

        这样做后T1就在功能像空间了，后续可以：
        1. 对配准后的T1进行分割，获得变形场
        2. 用变形场将功能像标准化到MNI空间
        """
        other_cell = ""
        if other_inputs:
            other_cell = "{\n" + "\n".join([f"    '{f},1'" for f in other_inputs]) + "\n}"
        else:
            other_cell = "{''}"

        return f"""% SPM Coregister: T1 to Mean Functional Image
% Generated by Research Agent
% 标准fMRI配准流程：将T1结构像配准到mean功能像空间

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 参考图像（固定不动）: mean功能像
ref_image_path = '{reference_image}';
ref_image = [ref_image_path ',1'];

% 源图像（要移动以匹配参考图像）: T1结构像
source_image_path = '{source_image}';
source_image = [source_image_path ',1'];

% 其他图像（跟随T1一起变换）
other_images = {other_cell};

% ===== 文件存在性验证 =====
fprintf('检查输入文件...\\n');

if ~exist(ref_image_path, 'file')
    error('参考图像(mean功能像)不存在: %s', ref_image_path);
end
fprintf('  [OK] 参考图像(mean): %s\\n', ref_image_path);

if ~exist(source_image_path, 'file')
    error('源图像(T1结构像)不存在: %s', source_image_path);
end
fprintf('  [OK] 源图像(T1): %s\\n', source_image_path);

fprintf('配准方向: T1 -> mean功能像\\n');
fprintf('文件验证通过。\\n\\n');

% ===== Coregister: Estimate & Reslice 设置 =====
matlabbatch{{1}}.spm.spatial.coreg.estwrite.ref = {{ref_image}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.source = {{source_image}};
matlabbatch{{1}}.spm.spatial.coreg.estwrite.other = other_images;

% 估计选项（T1-功能像配准优化参数）
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.cost_fun = 'nmi';   % 归一化互信息（多模态最佳）
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.sep = [4 2];        % 采样间隔（粗到细）
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
matlabbatch{{1}}.spm.spatial.coreg.estwrite.eoptions.fwhm = [7 7];       % 高斯平滑核FWHM

% 重采样选项
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.interp = 4;         % 4阶B样条插值
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.wrap = [0 0 0];     % 不环绕
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.mask = 0;           % 不使用掩码
matlabbatch{{1}}.spm.spatial.coreg.estwrite.roptions.prefix = 'r';       % 输出前缀

% 执行配准
fprintf('开始执行T1到功能像配准...\\n');
try
    spm_jobman('run', matlabbatch);
    disp('T1 to Functional Coregistration completed successfully!');
    disp('输出文件: rT1*.nii (配准后的T1结构像)');
    fprintf('参考图像(mean功能像): %s\\n', ref_image_path);
    fprintf('源图像(T1结构像): %s\\n', source_image_path);
    fprintf('\\n后续步骤: 对配准后的T1进行分割，获取变形场用于功能像标准化\\n');
catch ME
    fprintf('Error in T1-functional coregistration: %s\\n', ME.message);
    fprintf('详细信息: %s\\n', getReport(ME, 'extended'));
    rethrow(ME);
end
"""


class LocalDPABITool(BaseTool):
    """本地DPABI工具 - 静息态fMRI分析"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dpabi_analysis",
            description="使用本地DPABI进行静息态fMRI分析，支持ALFF、ReHo、FC等",
            category="analysis",
            supported_modalities=[Modality.FUNC],
            executor_type=ExecutorType.MATLAB,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入的4D fMRI NIfTI文件列表"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["alff", "falff", "reho", "fc_seed", "degree_centrality", "extract_roi_values", "roi_analysis"],
                        "description": "分析类型: alff(低频振幅), falff(分数低频振幅), reho(局部一致性), fc_seed(种子点功能连接), degree_centrality(度中心性), extract_roi_values(提取ROI值), roi_analysis(ROI分析，计算多个指标)"
                    },
                    "roi_atlas": {
                        "type": "string",
                        "description": "ROI图谱路径（用于extract_roi_values）"
                    },
                    "tr": {
                        "type": "number",
                        "default": 2.0,
                        "description": "TR时间(秒)"
                    },
                    "band_pass": {
                        "type": "array",
                        "default": [0.01, 0.08],
                        "description": "带通滤波范围"
                    }
                },
                "required": ["input_files", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "metrics": {"type": "object"}
                }
            },
            version="V90",
            dependencies=["MATLAB R2019b", "DPABI V90", "SPM25"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DPABI分析 - 支持MATLAB和Python双模式"""
        from src.config_local_tools import MATLAB_EXE
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # MATLAB 可用性预检查
        if not MATLAB_EXE or not MATLAB_EXE.exists():
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"MATLAB 未找到 (配置路径: {MATLAB_EXE})。请设置 MATLAB_ROOT 环境变量。",
                outputs={}, duration_seconds=0
            )

        analysis_type = request.params.get("analysis_type", "alff")
        input_files = request.inputs.get("input_files", [])
        tr = request.params.get("tr", 2.0)
        band_pass = request.params.get("band_pass", [0.01, 0.08])

        # 验证输入文件
        if not input_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="未提供输入文件列表 (input_files)"
            )

        # ========== 过滤输入文件：只保留4D fMRI数据 ==========
        valid_4d_files = self._filter_4d_files(input_files)
        if not valid_4d_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="未找到有效的4D fMRI文件。DPABI分析需要4D时间序列数据。"
            )

        print(f"  [输入过滤] {len(input_files)} 个文件 -> {len(valid_4d_files)} 个有效4D文件")

        # ========== 尝试MATLAB/DPABI ==========
        version_check = self._check_dpabi_version()
        matlab_success = False
        matlab_error = None

        if version_check["available"]:
            if version_check.get("warning"):
                print(f"  [DPABI警告] {version_check['warning']}")
            print(f"  [DPABI版本] {version_check.get('version', 'unknown')}")

            # 转换文件路径为MATLAB格式
            matlab_files = [str(f).replace("\\", "/") for f in valid_4d_files]
            matlab_output = str(output_dir).replace("\\", "/")

            # 生成DPABI脚本
            script = self._generate_dpabi_script(
                analysis_type, matlab_files, matlab_output, tr, band_pass
            )

            script_path = output_dir / f"dpabi_{analysis_type}.m"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)

            # 执行MATLAB
            print(f"  [MATLAB] 尝试使用DPABI执行 {analysis_type} 分析...")
            result = run_matlab_script(script, str(output_dir))

            if result and result.get("status") == "succeeded":
                # 检查是否有实际输出文件
                output_files = []
                for subdir in output_dir.iterdir():
                    if subdir.is_dir():
                        output_files.extend(list(subdir.glob("*.nii*")))
                output_files.extend(list(output_dir.glob("*.nii*")))

                if output_files:
                    matlab_success = True
                    print(f"  [MATLAB] DPABI分析成功，生成 {len(output_files)} 个输出文件")
                else:
                    matlab_error = "MATLAB执行完成但未生成输出文件"
                    print(f"  [MATLAB] {matlab_error}")
            else:
                matlab_error = result.get("error") or result.get("stderr", "Unknown MATLAB error")
                print(f"  [MATLAB] DPABI执行失败: {matlab_error[:200]}...")
        else:
            matlab_error = f"DPABI不可用: {version_check['error']}"
            print(f"  [MATLAB] {matlab_error}")

        # ========== 如果MATLAB成功，返回结果 ==========
        if matlab_success:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            output_files = []
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    output_files.extend([str(f) for f in subdir.glob("*.nii*")])
            output_files.extend([str(f) for f in output_dir.glob("*.nii*")])

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": output_files,
                    "analysis_type": analysis_type,
                    "method": "MATLAB/DPABI"
                }
            )

        # ========== MATLAB失败，尝试Python备选方案 ==========
        print(f"\n  [Python备选] MATLAB/DPABI失败，尝试使用Python计算fMRI指标...")

        try:
            python_result = self._compute_fmri_metrics_python(
                valid_4d_files, analysis_type, tr, band_pass, output_dir
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if python_result["success"]:
                print(f"  [Python备选] 成功计算 {len(python_result['output_files'])} 个输出文件")
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded",
                    started_at=start_time.isoformat(),
                    finished_at=end_time.isoformat(),
                    duration_seconds=duration,
                    outputs={
                        "output_files": python_result["output_files"],
                        "analysis_type": analysis_type,
                        "method": "Python/nibabel",
                        "matlab_error": matlab_error
                    }
                )
            else:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"MATLAB失败: {matlab_error}\nPython备选也失败: {python_result['error']}"
                )

        except Exception as e:
            import traceback
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"MATLAB失败: {matlab_error}\nPython备选异常: {str(e)}\n{traceback.format_exc()}"
            )

    def _filter_4d_files(self, input_files: List[str]) -> List[str]:
        """过滤输入文件，只保留4D fMRI数据"""
        import nibabel as nib

        valid_files = []
        for f in input_files:
            try:
                img = nib.load(f)
                shape = img.shape
                if len(shape) >= 4 and shape[3] > 1:
                    valid_files.append(f)
                    print(f"    [OK] 4D数据 ({shape[3]}卷): {Path(f).name}")
                else:
                    print(f"    [跳过] 3D数据: {Path(f).name}")
            except Exception as e:
                print(f"    [跳过] 无法读取: {Path(f).name} ({e})")
        return valid_files

    def _compute_fmri_metrics_python(self, input_files: List[str], analysis_type: str,
                                      tr: float, band_pass: List[float], output_dir: Path) -> Dict:
        """使用Python计算fMRI指标（ALFF, fALFF, ReHo）

        这是MATLAB/DPABI的备选方案，使用nibabel和scipy实现。
        """
        import nibabel as nib
        import numpy as np
        from scipy import signal
        from scipy.ndimage import uniform_filter

        output_files = []
        errors = []

        for input_file in input_files:
            try:
                print(f"    [Python] 处理: {Path(input_file).name}")

                # 加载4D数据
                img = nib.load(input_file)
                data = img.get_fdata()
                affine = img.affine
                header = img.header

                if len(data.shape) != 4:
                    print(f"      [跳过] 不是4D数据: {data.shape}")
                    continue

                n_timepoints = data.shape[3]
                print(f"      [数据] 形状: {data.shape}, TR: {tr}s")

                # 创建脑掩模（简单阈值）
                mean_data = np.mean(data, axis=3)
                mask = mean_data > (np.mean(mean_data) * 0.2)

                # 提取被试ID
                subject_id = Path(input_file).stem.replace('.nii', '')

                # 创建被试输出目录
                subject_output = output_dir / subject_id
                subject_output.mkdir(parents=True, exist_ok=True)

                if analysis_type in ["alff", "falff"]:
                    # ========== ALFF/fALFF 计算 ==========
                    print(f"      [计算] ALFF/fALFF...")

                    # 计算采样频率
                    fs = 1.0 / tr
                    low_freq, high_freq = band_pass

                    # 初始化输出
                    alff_map = np.zeros(data.shape[:3])
                    falff_map = np.zeros(data.shape[:3])

                    # 对每个体素计算ALFF
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            for k in range(data.shape[2]):
                                if not mask[i, j, k]:
                                    continue

                                ts = data[i, j, k, :]

                                # 去趋势
                                ts = signal.detrend(ts)

                                # FFT
                                fft_result = np.fft.fft(ts)
                                freqs = np.fft.fftfreq(n_timepoints, tr)

                                # 计算功率谱
                                power = np.abs(fft_result) ** 2

                                # 找到低频范围的索引
                                low_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                                all_idx = np.where(freqs >= 0)[0]

                                # ALFF: 低频功率的平方根
                                if len(low_idx) > 0:
                                    alff_map[i, j, k] = np.sqrt(np.mean(power[low_idx]))

                                # fALFF: 低频功率 / 全频功率
                                total_power = np.sum(power[all_idx])
                                if total_power > 0 and len(low_idx) > 0:
                                    falff_map[i, j, k] = np.sum(power[low_idx]) / total_power

                    # 保存ALFF
                    alff_img = nib.Nifti1Image(alff_map, affine, header)
                    alff_path = subject_output / f"{subject_id}_ALFF.nii"
                    nib.save(alff_img, str(alff_path))
                    output_files.append(str(alff_path))
                    print(f"      [保存] {alff_path.name}")

                    # 保存fALFF
                    falff_img = nib.Nifti1Image(falff_map, affine, header)
                    falff_path = subject_output / f"{subject_id}_fALFF.nii"
                    nib.save(falff_img, str(falff_path))
                    output_files.append(str(falff_path))
                    print(f"      [保存] {falff_path.name}")

                elif analysis_type == "reho":
                    # ========== ReHo 计算 ==========
                    print(f"      [计算] ReHo (Kendall's W)...")

                    # ReHo使用27个邻域体素
                    reho_map = np.zeros(data.shape[:3])

                    # 对每个体素计算ReHo
                    for i in range(1, data.shape[0] - 1):
                        for j in range(1, data.shape[1] - 1):
                            for k in range(1, data.shape[2] - 1):
                                if not mask[i, j, k]:
                                    continue

                                # 提取27个邻域体素的时间序列
                                neighborhood = data[i-1:i+2, j-1:j+2, k-1:k+2, :]
                                neighborhood = neighborhood.reshape(-1, n_timepoints)

                                # 只使用掩模内的体素
                                mask_neighborhood = mask[i-1:i+2, j-1:j+2, k-1:k+2].flatten()
                                valid_voxels = neighborhood[mask_neighborhood]

                                if len(valid_voxels) < 7:  # 至少需要7个有效体素
                                    continue

                                # 计算Kendall's W
                                try:
                                    # 对每个时间点排序
                                    n_voxels = len(valid_voxels)
                                    ranks = np.zeros_like(valid_voxels)
                                    for t in range(n_timepoints):
                                        ranks[:, t] = np.argsort(np.argsort(valid_voxels[:, t])) + 1

                                    # Kendall's W = 12 * S / (k^2 * (n^3 - n))
                                    # S = sum((Ri - R_mean)^2)
                                    R = np.sum(ranks, axis=0)
                                    R_mean = np.mean(R)
                                    S = np.sum((R - R_mean) ** 2)
                                    k = n_voxels
                                    n = n_timepoints
                                    W = 12 * S / (k ** 2 * (n ** 3 - n))
                                    reho_map[i, j, k] = W
                                except Exception:
                                    pass

                    # 保存ReHo
                    reho_img = nib.Nifti1Image(reho_map, affine, header)
                    reho_path = subject_output / f"{subject_id}_ReHo.nii"
                    nib.save(reho_img, str(reho_path))
                    output_files.append(str(reho_path))
                    print(f"      [保存] {reho_path.name}")

                elif analysis_type == "fc_seed":
                    # ========== 种子点功能连接 ==========
                    print(f"      [计算] 种子点FC (PCC: [0, -53, 26])...")

                    # PCC种子点MNI坐标
                    seed_mni = np.array([0, -53, 26, 1])

                    # 转换到体素坐标
                    inv_affine = np.linalg.inv(affine)
                    seed_voxel = np.round(inv_affine @ seed_mni).astype(int)[:3]

                    print(f"      [种子点] MNI: {seed_mni[:3]} -> 体素: {seed_voxel}")

                    # 检查种子点是否在图像范围内
                    if all(0 <= seed_voxel[i] < data.shape[i] for i in range(3)):
                        # 提取种子点时间序列（6mm球形ROI）
                        radius_voxels = int(6 / np.mean(np.abs(np.diag(affine)[:3])))
                        seed_ts = []

                        for di in range(-radius_voxels, radius_voxels + 1):
                            for dj in range(-radius_voxels, radius_voxels + 1):
                                for dk in range(-radius_voxels, radius_voxels + 1):
                                    ni, nj, nk = seed_voxel + np.array([di, dj, dk])
                                    if (0 <= ni < data.shape[0] and
                                        0 <= nj < data.shape[1] and
                                        0 <= nk < data.shape[2]):
                                        if di**2 + dj**2 + dk**2 <= radius_voxels**2:
                                            seed_ts.append(data[ni, nj, nk, :])

                        if seed_ts:
                            seed_ts = np.mean(seed_ts, axis=0)
                            seed_ts = signal.detrend(seed_ts)

                            # 计算每个体素与种子点的相关
                            fc_map = np.zeros(data.shape[:3])

                            for i in range(data.shape[0]):
                                for j in range(data.shape[1]):
                                    for k in range(data.shape[2]):
                                        if not mask[i, j, k]:
                                            continue

                                        voxel_ts = signal.detrend(data[i, j, k, :])
                                        r = np.corrcoef(seed_ts, voxel_ts)[0, 1]
                                        fc_map[i, j, k] = r

                            # Fisher Z变换
                            fc_map = np.arctanh(np.clip(fc_map, -0.999, 0.999))

                            # 保存FC图
                            fc_img = nib.Nifti1Image(fc_map, affine, header)
                            fc_path = subject_output / f"{subject_id}_FC_PCC.nii"
                            nib.save(fc_img, str(fc_path))
                            output_files.append(str(fc_path))
                            print(f"      [保存] {fc_path.name}")
                        else:
                            errors.append(f"{subject_id}: 种子点ROI为空")
                    else:
                        errors.append(f"{subject_id}: 种子点超出图像范围")

                elif analysis_type == "degree_centrality":
                    # ========== 度中心性 ==========
                    print(f"      [计算] 度中心性 (r > 0.25)...")

                    r_threshold = 0.25
                    dc_map = np.zeros(data.shape[:3])

                    # 提取所有掩模内体素的时间序列
                    mask_indices = np.where(mask)
                    n_voxels = len(mask_indices[0])
                    all_ts = np.zeros((n_voxels, n_timepoints))

                    for idx, (i, j, k) in enumerate(zip(*mask_indices)):
                        all_ts[idx] = signal.detrend(data[i, j, k, :])

                    # 计算相关矩阵（分块计算以节省内存）
                    print(f"      [计算] 计算 {n_voxels} 个体素的度中心性...")

                    for idx, (i, j, k) in enumerate(zip(*mask_indices)):
                        # 计算当前体素与所有其他体素的相关
                        correlations = np.corrcoef(all_ts[idx], all_ts)[0, 1:]
                        # 度中心性 = 超过阈值的连接数
                        dc_map[i, j, k] = np.sum(np.abs(correlations) > r_threshold)

                    # 保存DC图
                    dc_img = nib.Nifti1Image(dc_map, affine, header)
                    dc_path = subject_output / f"{subject_id}_DC.nii"
                    nib.save(dc_img, str(dc_path))
                    output_files.append(str(dc_path))
                    print(f"      [保存] {dc_path.name}")

                elif analysis_type == "extract_roi_values":
                    # ========== 提取ROI值 ==========
                    # 先计算ALFF、fALFF、ReHo，然后从ROI中提取值
                    print(f"      [计算] 提取ROI值 (先计算ALFF/fALFF/ReHo)...")

                    # 计算采样频率
                    fs = 1.0 / tr
                    low_freq, high_freq = band_pass

                    # 初始化输出
                    alff_map = np.zeros(data.shape[:3])
                    falff_map = np.zeros(data.shape[:3])
                    reho_map = np.zeros(data.shape[:3])

                    # 计算ALFF/fALFF
                    print(f"        [步骤1] 计算ALFF/fALFF...")
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            for k in range(data.shape[2]):
                                if not mask[i, j, k]:
                                    continue

                                ts = data[i, j, k, :]
                                ts = signal.detrend(ts)

                                fft_result = np.fft.fft(ts)
                                freqs = np.fft.fftfreq(n_timepoints, tr)
                                power = np.abs(fft_result) ** 2

                                low_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                                all_idx = np.where(freqs >= 0)[0]

                                if len(low_idx) > 0:
                                    alff_map[i, j, k] = np.sqrt(np.mean(power[low_idx]))

                                total_power = np.sum(power[all_idx])
                                if total_power > 0 and len(low_idx) > 0:
                                    falff_map[i, j, k] = np.sum(power[low_idx]) / total_power

                    # 计算ReHo
                    print(f"        [步骤2] 计算ReHo...")
                    for i in range(1, data.shape[0] - 1):
                        for j in range(1, data.shape[1] - 1):
                            for k in range(1, data.shape[2] - 1):
                                if not mask[i, j, k]:
                                    continue

                                neighborhood = data[i-1:i+2, j-1:j+2, k-1:k+2, :]
                                neighborhood = neighborhood.reshape(-1, n_timepoints)
                                mask_neighborhood = mask[i-1:i+2, j-1:j+2, k-1:k+2].flatten()
                                valid_voxels = neighborhood[mask_neighborhood]

                                if len(valid_voxels) < 7:
                                    continue

                                try:
                                    n_voxels = len(valid_voxels)
                                    ranks = np.zeros_like(valid_voxels)
                                    for t in range(n_timepoints):
                                        ranks[:, t] = np.argsort(np.argsort(valid_voxels[:, t])) + 1

                                    R = np.sum(ranks, axis=0)
                                    R_mean = np.mean(R)
                                    S = np.sum((R - R_mean) ** 2)
                                    k_val = n_voxels
                                    n_val = n_timepoints
                                    W = 12 * S / (k_val ** 2 * (n_val ** 3 - n_val))
                                    reho_map[i, j, k] = W
                                except Exception:
                                    pass

                    # 保存ALFF、fALFF、ReHo图
                    alff_img = nib.Nifti1Image(alff_map, affine, header)
                    alff_path = subject_output / f"{subject_id}_ALFF.nii"
                    nib.save(alff_img, str(alff_path))
                    output_files.append(str(alff_path))

                    falff_img = nib.Nifti1Image(falff_map, affine, header)
                    falff_path = subject_output / f"{subject_id}_fALFF.nii"
                    nib.save(falff_img, str(falff_path))
                    output_files.append(str(falff_path))

                    reho_img = nib.Nifti1Image(reho_map, affine, header)
                    reho_path = subject_output / f"{subject_id}_ReHo.nii"
                    nib.save(reho_img, str(reho_path))
                    output_files.append(str(reho_path))

                    print(f"      [保存] {subject_id}_ALFF.nii, {subject_id}_fALFF.nii, {subject_id}_ReHo.nii")

                    # 提取ROI值（使用简单的脑区划分）
                    print(f"        [步骤3] 提取ROI平均值...")

                    # 创建简单的ROI划分（基于坐标）
                    roi_results = {
                        "subject_id": subject_id,
                        "global_alff": float(np.mean(alff_map[mask])) if np.any(mask) else 0,
                        "global_falff": float(np.mean(falff_map[mask])) if np.any(mask) else 0,
                        "global_reho": float(np.mean(reho_map[mask])) if np.any(mask) else 0,
                    }

                    # 简单的前后左右划分
                    mid_x, mid_y, mid_z = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2

                    regions = {
                        "frontal": (slice(mid_x, None), slice(None), slice(mid_z, None)),
                        "parietal": (slice(None, mid_x), slice(None), slice(mid_z, None)),
                        "temporal_left": (slice(None), slice(None, mid_y), slice(None, mid_z)),
                        "temporal_right": (slice(None), slice(mid_y, None), slice(None, mid_z)),
                        "occipital": (slice(None, mid_x), slice(None), slice(None, mid_z)),
                        "cerebellum": (slice(None), slice(None), slice(None, data.shape[2]//3)),
                    }

                    for region_name, region_slice in regions.items():
                        region_mask = np.zeros_like(mask)
                        region_mask[region_slice] = mask[region_slice]

                        if np.any(region_mask):
                            roi_results[f"{region_name}_alff"] = float(np.mean(alff_map[region_mask]))
                            roi_results[f"{region_name}_falff"] = float(np.mean(falff_map[region_mask]))
                            roi_results[f"{region_name}_reho"] = float(np.mean(reho_map[region_mask]))

                    # 保存ROI结果到JSON
                    import json
                    roi_json_path = subject_output / f"{subject_id}_roi_values.json"
                    with open(roi_json_path, 'w') as f:
                        json.dump(roi_results, f, indent=2)
                    output_files.append(str(roi_json_path))
                    print(f"      [保存] {roi_json_path.name}")

                elif analysis_type == "roi_analysis":
                    # ========== ROI分析 (与extract_roi_values相同) ==========
                    # 计算ALFF、fALFF、ReHo，然后从ROI中提取值
                    print(f"      [计算] ROI分析 (ALFF/fALFF/ReHo + ROI提取)...")

                    # 计算采样频率
                    fs = 1.0 / tr
                    low_freq, high_freq = band_pass

                    # 初始化输出
                    alff_map = np.zeros(data.shape[:3])
                    falff_map = np.zeros(data.shape[:3])
                    reho_map = np.zeros(data.shape[:3])

                    # 计算ALFF/fALFF
                    print(f"        [步骤1] 计算ALFF/fALFF...")
                    for i in range(data.shape[0]):
                        for j in range(data.shape[1]):
                            for k in range(data.shape[2]):
                                if not mask[i, j, k]:
                                    continue

                                ts = data[i, j, k, :]
                                ts = signal.detrend(ts)

                                fft_result = np.fft.fft(ts)
                                freqs = np.fft.fftfreq(n_timepoints, tr)
                                power = np.abs(fft_result) ** 2

                                low_idx = np.where((freqs >= low_freq) & (freqs <= high_freq))[0]
                                all_idx = np.where(freqs >= 0)[0]

                                if len(low_idx) > 0:
                                    alff_map[i, j, k] = np.sqrt(np.mean(power[low_idx]))

                                total_power = np.sum(power[all_idx])
                                if total_power > 0 and len(low_idx) > 0:
                                    falff_map[i, j, k] = np.sum(power[low_idx]) / total_power

                    # 计算ReHo
                    print(f"        [步骤2] 计算ReHo...")
                    for i in range(1, data.shape[0] - 1):
                        for j in range(1, data.shape[1] - 1):
                            for k in range(1, data.shape[2] - 1):
                                if not mask[i, j, k]:
                                    continue

                                neighborhood = data[i-1:i+2, j-1:j+2, k-1:k+2, :]
                                neighborhood = neighborhood.reshape(-1, n_timepoints)
                                mask_neighborhood = mask[i-1:i+2, j-1:j+2, k-1:k+2].flatten()
                                valid_voxels = neighborhood[mask_neighborhood]

                                if len(valid_voxels) < 7:
                                    continue

                                try:
                                    n_voxels = len(valid_voxels)
                                    ranks = np.zeros_like(valid_voxels)
                                    for t in range(n_timepoints):
                                        ranks[:, t] = np.argsort(np.argsort(valid_voxels[:, t])) + 1

                                    R = np.sum(ranks, axis=0)
                                    R_mean = np.mean(R)
                                    S = np.sum((R - R_mean) ** 2)
                                    k_val = n_voxels
                                    n_val = n_timepoints
                                    W = 12 * S / (k_val ** 2 * (n_val ** 3 - n_val))
                                    reho_map[i, j, k] = W
                                except Exception:
                                    pass

                    # 保存ALFF、fALFF、ReHo图
                    alff_img = nib.Nifti1Image(alff_map, affine, header)
                    alff_path = subject_output / f"{subject_id}_ALFF.nii"
                    nib.save(alff_img, str(alff_path))
                    output_files.append(str(alff_path))

                    falff_img = nib.Nifti1Image(falff_map, affine, header)
                    falff_path = subject_output / f"{subject_id}_fALFF.nii"
                    nib.save(falff_img, str(falff_path))
                    output_files.append(str(falff_path))

                    reho_img = nib.Nifti1Image(reho_map, affine, header)
                    reho_path = subject_output / f"{subject_id}_ReHo.nii"
                    nib.save(reho_img, str(reho_path))
                    output_files.append(str(reho_path))

                    print(f"      [保存] {subject_id}_ALFF.nii, {subject_id}_fALFF.nii, {subject_id}_ReHo.nii")

                    # 提取ROI值
                    print(f"        [步骤3] 提取ROI平均值...")

                    roi_results = {
                        "subject_id": subject_id,
                        "global_alff": float(np.mean(alff_map[mask])) if np.any(mask) else 0,
                        "global_falff": float(np.mean(falff_map[mask])) if np.any(mask) else 0,
                        "global_reho": float(np.mean(reho_map[mask])) if np.any(mask) else 0,
                    }

                    # 简单的脑区划分
                    mid_x, mid_y, mid_z = data.shape[0]//2, data.shape[1]//2, data.shape[2]//2

                    regions = {
                        "frontal": (slice(mid_x, None), slice(None), slice(mid_z, None)),
                        "parietal": (slice(None, mid_x), slice(None), slice(mid_z, None)),
                        "temporal_left": (slice(None), slice(None, mid_y), slice(None, mid_z)),
                        "temporal_right": (slice(None), slice(mid_y, None), slice(None, mid_z)),
                        "occipital": (slice(None, mid_x), slice(None), slice(None, mid_z)),
                        "cerebellum": (slice(None), slice(None), slice(None, data.shape[2]//3)),
                    }

                    for region_name, region_slice in regions.items():
                        region_mask = np.zeros_like(mask)
                        region_mask[region_slice] = mask[region_slice]

                        if np.any(region_mask):
                            roi_results[f"{region_name}_alff"] = float(np.mean(alff_map[region_mask]))
                            roi_results[f"{region_name}_falff"] = float(np.mean(falff_map[region_mask]))
                            roi_results[f"{region_name}_reho"] = float(np.mean(reho_map[region_mask]))

                    # 保存ROI结果到JSON
                    import json
                    roi_json_path = subject_output / f"{subject_id}_roi_analysis.json"
                    with open(roi_json_path, 'w') as f:
                        json.dump(roi_results, f, indent=2)
                    output_files.append(str(roi_json_path))
                    print(f"      [保存] {roi_json_path.name}")

                else:
                    errors.append(f"不支持的分析类型: {analysis_type}")

            except Exception as e:
                import traceback
                errors.append(f"{Path(input_file).name}: {str(e)}")
                print(f"      [错误] {e}")
                traceback.print_exc()

        if output_files:
            return {
                "success": True,
                "output_files": output_files,
                "errors": errors if errors else None
            }
        else:
            return {
                "success": False,
                "output_files": [],
                "error": "; ".join(errors) if errors else "未生成任何输出文件"
            }

    def _check_dpabi_version(self) -> Dict[str, Any]:
        """检查 DPABI 版本和可用性

        Returns:
            包含以下键的字典:
            - available: bool, DPABI是否可用
            - version: str, DPABI版本号
            - error: str, 错误信息（如果不可用）
            - warning: str, 警告信息（如果版本不匹配）
        """
        result = {
            "available": False,
            "version": "unknown",
            "error": None,
            "warning": None
        }

        # 检查 DPABI 路径是否存在
        if not DPABI_PATH or not Path(DPABI_PATH).exists():
            result["error"] = f"DPABI 路径不存在: {DPABI_PATH}"
            return result

        dpabi_path = Path(DPABI_PATH)

        # 检查关键目录结构
        dparsf_path = dpabi_path / "DPARSF"
        if not dparsf_path.exists():
            result["error"] = f"DPABI 缺少 DPARSF 目录: {dparsf_path}"
            return result

        subfunctions_path = dparsf_path / "Subfunctions"
        if not subfunctions_path.exists():
            result["error"] = f"DPABI 缺少 Subfunctions 目录: {subfunctions_path}"
            return result

        # 检查关键函数文件是否存在
        required_functions = {
            "y_alff_falff.m": "ALFF/fALFF 分析",
            "y_reho.m": "ReHo 分析",
            "y_SCA.m": "种子点功能连接分析",
            "y_DegreeCentrality.m": "度中心性分析"
        }

        missing_functions = []
        for func_file, func_desc in required_functions.items():
            if not (subfunctions_path / func_file).exists():
                missing_functions.append(f"{func_file} ({func_desc})")

        if missing_functions:
            result["error"] = f"DPABI 缺少关键函数文件: {', '.join(missing_functions)}"
            return result

        # 尝试从版本文件读取版本号
        version = "unknown"
        version_file = dpabi_path / "DPABI_VERSION.txt"
        if version_file.exists():
            try:
                version = version_file.read_text().strip()
            except Exception:
                pass

        # 如果没有版本文件，尝试从目录名或其他来源推断
        if version == "unknown":
            # 检查 Contents.m 文件
            contents_file = dpabi_path / "Contents.m"
            if contents_file.exists():
                try:
                    content = contents_file.read_text()
                    # 尝试提取版本号
                    import re
                    version_match = re.search(r'Version\s*[:\s]*([VvRr]?\d+[\.\d]*)', content)
                    if version_match:
                        version = version_match.group(1)
                except:
                    pass

        # 如果还是未知，假设是目录名中的版本
        if version == "unknown":
            dir_name = dpabi_path.name.upper()
            if "V" in dir_name:
                import re
                version_match = re.search(r'V(\d+)', dir_name)
                if version_match:
                    version = f"V{version_match.group(1)}"

        result["version"] = version
        result["available"] = True

        # 检查版本兼容性
        expected_version = "V90"
        if version != "unknown" and expected_version not in version.upper():
            result["warning"] = (
                f"当前 DPABI 版本 ({version}) 可能与预期版本 ({expected_version}) 不兼容。"
                f"函数签名可能不同，建议使用 DPABI {expected_version}。"
            )

        return result

    def _generate_dpabi_script(self, analysis_type: str, input_files: List[str],
                               output_dir: str, tr: float, band_pass: List[float]) -> str:
        """生成DPABI分析脚本 - DPABI V90 (基于官方源码验证)

        DPABI V90 函数位置: DPARSF/Subfunctions/
        - y_alff_falff.m: 同时计算 ALFF 和 fALFF
        - y_reho.m: 计算 ReHo
        - y_SCA.m: 种子点功能连接 (不是 y_FC!)
        - y_DegreeCentrality.m: 度中心性

        Args:
            analysis_type: 分析类型
            input_files: 输入文件列表（MATLAB格式路径）
            output_dir: 输出目录
            tr: TR时间
            band_pass: 带通滤波范围
        """
        # 添加 DPARSF/Subfunctions 到路径
        dpabi_subfunctions = str(DPABI_PATH / "DPARSF" / "Subfunctions").replace("\\", "/")

        # 构建MATLAB文件列表字符串
        files_cell = "{\n" + ",\n".join([f"    '{f}'" for f in input_files]) + "\n}"

        return f"""% DPABI Analysis Script - {analysis_type.upper()}
% Generated by Research Agent
% DPABI Version: V90 (verified from source code)

% Initialize DPABI
DPABI_Path = '{str(DPABI_PATH).replace(chr(92), "/")}';
addpath(genpath(DPABI_Path));

% Ensure DPARSF Subfunctions are in path (contains y_alff_falff, y_reho, y_SCA, y_DegreeCentrality)
addpath('{dpabi_subfunctions}');

% Parameters
InputFiles = {files_cell};
OutputDir = '{output_dir}';
TR = {tr};
LowCutoff = {band_pass[0]};
HighCutoff = {band_pass[1]};
BandPass = [{band_pass[0]}, {band_pass[1]}];

% Create output directory
if ~exist(OutputDir, 'dir')
    mkdir(OutputDir);
end

% Filter input files: only keep 4D fMRI data (exclude mean, smoothed 3D images)
ValidFiles = {{}};
fprintf('Checking input files for 4D data...\\n');
for i = 1:length(InputFiles)
    input_file = InputFiles{{i}};
    if ~exist(input_file, 'file')
        fprintf('  [SKIP] File not found: %s\\n', input_file);
        continue;
    end

    % Check if file is 4D
    try
        hdr = spm_vol(input_file);
        if length(hdr) > 1
            % 4D file (multiple volumes)
            ValidFiles{{end+1}} = input_file;
            fprintf('  [OK] 4D data (%d volumes): %s\\n', length(hdr), input_file);
        else
            % 3D file (single volume) - skip for fMRI analysis
            fprintf('  [SKIP] 3D data (not suitable for fMRI analysis): %s\\n', input_file);
        end
    catch
        % If spm_vol fails, try niftiinfo
        try
            info = niftiinfo(input_file);
            if length(info.ImageSize) >= 4 && info.ImageSize(4) > 1
                ValidFiles{{end+1}} = input_file;
                fprintf('  [OK] 4D data (%d volumes): %s\\n', info.ImageSize(4), input_file);
            else
                fprintf('  [SKIP] 3D data: %s\\n', input_file);
            end
        catch ME2
            fprintf('  [SKIP] Cannot read file: %s (%s)\\n', input_file, ME2.message);
        end
    end
end

if isempty(ValidFiles)
    error('No valid 4D fMRI files found! DPABI analysis requires 4D time series data.');
end

fprintf('\\nProcessing %d valid 4D NIfTI files\\n', length(ValidFiles));

% Process each valid file
for i = 1:length(ValidFiles)
    input_file = ValidFiles{{i}};

    [~, filename, ~] = fileparts(input_file);
    % Remove .nii extension if present (for .nii.gz files)
    subject_id = strrep(filename, '.nii', '');

    fprintf('Processing [%d/%d]: %s\\n', i, length(ValidFiles), subject_id);

    % Create subject output directory
    subject_output = fullfile(OutputDir, subject_id);
    if ~exist(subject_output, 'dir')
        mkdir(subject_output);
    end

    try
        switch '{analysis_type}'
            case {{'alff', 'falff'}}
                % ALFF/fALFF Analysis - 使用 y_alff_falff (同时计算两者)
                fprintf('  Running ALFF/fALFF analysis (y_alff_falff)...\\n');
                alff_output = fullfile(subject_output, [subject_id, '_ALFF.nii']);
                [ALFFBrain, fALFFBrain, Header] = y_alff_falff(input_file, TR, HighCutoff, LowCutoff, '', alff_output);
                fprintf('  ALFF/fALFF completed for %s\\n', subject_id);

            case 'reho'
                % ReHo Analysis - 使用 y_reho
                fprintf('  Running ReHo analysis (y_reho)...\\n');
                reho_output = fullfile(subject_output, [subject_id, '_ReHo.nii']);
                [ReHoBrain, Header] = y_reho(input_file, 27, '', reho_output, 1, BandPass, TR);
                fprintf('  ReHo completed for %s\\n', subject_id);

            case 'fc_seed'
                % Seed-based FC - 使用 y_SCA
                % ROIDef format: cell array, each element can be:
                %   - [x, y, z, radius]: sphere ROI in MNI coordinates
                %   - 'path/to/mask.nii': mask file
                % PCC (Posterior Cingulate Cortex) coordinates: [0, -53, 26], radius=6mm
                fprintf('  Running seed-based FC analysis (y_SCA)...\\n');
                ROIDef = {{[0, -53, 26, 6]}}; % MNI coordinates for PCC with 6mm radius
                fc_output = fullfile(subject_output, [subject_id, '_FC.nii']);
                [FCBrain, Header] = y_SCA(input_file, ROIDef, fc_output, '', 0, [], 1, BandPass, TR);
                fprintf('  FC completed for %s\\n', subject_id);

            case 'degree_centrality'
                % Degree Centrality - 使用 y_DegreeCentrality
                fprintf('  Running degree centrality analysis (y_DegreeCentrality)...\\n');
                rThreshold = 0.25;
                dc_output = fullfile(subject_output, [subject_id, '_DC.nii']);
                [DC_Weighted, DC_Binarized, Header] = y_DegreeCentrality(input_file, rThreshold, dc_output, '', 1, BandPass, TR);
                fprintf('  DC completed for %s\\n', subject_id);

            case 'extract_roi_values'
                % Extract ROI values - 计算ALFF/fALFF/ReHo并提取ROI值
                fprintf('  Running extract_roi_values (ALFF + fALFF + ReHo)...\\n');

                % Step 1: Calculate ALFF/fALFF
                alff_output = fullfile(subject_output, [subject_id, '_ALFF.nii']);
                [ALFFBrain, fALFFBrain, Header] = y_alff_falff(input_file, TR, HighCutoff, LowCutoff, '', alff_output);
                fprintf('    ALFF/fALFF completed\\n');

                % Step 2: Calculate ReHo
                reho_output = fullfile(subject_output, [subject_id, '_ReHo.nii']);
                [ReHoBrain, Header] = y_reho(input_file, 27, '', reho_output, 1, BandPass, TR);
                fprintf('    ReHo completed\\n');

                fprintf('  extract_roi_values completed for %s\\n', subject_id);

            case 'roi_analysis'
                % ROI Analysis - 与extract_roi_values相同
                fprintf('  Running roi_analysis (ALFF + fALFF + ReHo)...\\n');

                % Step 1: Calculate ALFF/fALFF
                alff_output = fullfile(subject_output, [subject_id, '_ALFF.nii']);
                [ALFFBrain, fALFFBrain, Header] = y_alff_falff(input_file, TR, HighCutoff, LowCutoff, '', alff_output);
                fprintf('    ALFF/fALFF completed\\n');

                % Step 2: Calculate ReHo
                reho_output = fullfile(subject_output, [subject_id, '_ReHo.nii']);
                [ReHoBrain, Header] = y_reho(input_file, 27, '', reho_output, 1, BandPass, TR);
                fprintf('    ReHo completed\\n');

                fprintf('  roi_analysis completed for %s\\n', subject_id);

            otherwise
                error('Unknown analysis type: {analysis_type}');
        end
    catch ME
        fprintf('  Error processing %s: %s\\n', subject_id, ME.message);
        continue;
    end
end

fprintf('DPABI %s analysis completed!\\n', upper('{analysis_type}'));
"""


class LocalDSIStudioTool(BaseTool):
    """本地DSI Studio工具 - 弥散张量成像分析

    完整分析流程：src → rec → trk
    1. src: 从NIfTI+bvec/bval生成.sz源文件
    2. rec: 重建（GQI/QSDR），输出.fz纤维方向文件
    3. trk: 纤维追踪，输出.tt.gz纤维束文件
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dsi_studio_analysis",
            description="""使用DSI Studio进行弥散成像分析。

完整流程: src → rec → trk
- src: 从NIfTI+bvec/bval生成.sz源文件
- rec: 重建（GQI/QSDR），输出.fz纤维方向文件
- trk: 纤维追踪，输出.tt.gz纤维束文件

重建方法:
- method=4: GQI (本地空间，推荐)
- method=7: QSDR (MNI空间，群组分析)

纤维追踪参数:
- fa_threshold: FA阈值 (默认0.15)
- turning_angle: 最大转角 (默认55度)
- fiber_count: 纤维数量 (默认100000)""",
            category="analysis",
            supported_modalities=[Modality.DWI],
            executor_type=ExecutorType.CLI,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入文件路径列表（NIfTI, .sz, .fz）"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["src", "rec", "trk", "ana", "exp"],
                        "description": "操作类型: src=创建源文件, rec=重建, trk=追踪, ana=分析, exp=导出"
                    },
                    "method": {
                        "type": "integer",
                        "enum": [0, 1, 4, 7],
                        "default": 4,
                        "description": "重建方法: 0=DSI, 1=DTI, 4=GQI(推荐), 7=QSDR(MNI空间)"
                    },
                    "param0": {
                        "type": "number",
                        "default": 1.25,
                        "description": "扩散采样长度比（GQI/QSDR参数，范围0.3-2.0，推荐1.25）"
                    },
                    "fiber_count": {
                        "type": "integer",
                        "default": 100000,
                        "description": "纤维追踪数量"
                    },
                    "fa_threshold": {
                        "type": "number",
                        "default": 0.15,
                        "description": "FA阈值（纤维追踪暂停阈值，范围0-2.0）"
                    },
                    "turning_angle": {
                        "type": "number",
                        "default": 55,
                        "description": "最大转角（度，范围0-90）"
                    },
                    "min_length": {
                        "type": "number",
                        "default": 10,
                        "description": "最小纤维长度（mm）"
                    },
                    "max_length": {
                        "type": "number",
                        "default": 400,
                        "description": "最大纤维长度（mm）"
                    },
                    "bval_file": {
                        "type": "string",
                        "description": "b值文件路径（src时需要，可自动查找）"
                    },
                    "bvec_file": {
                        "type": "string",
                        "description": "b向量文件路径（src时需要，可自动查找）"
                    },
                    # 新增可配置参数（之前为硬编码）
                    "step_size": {
                        "type": "number",
                        "default": 1.0,
                        "description": "追踪步长（mm），默认1.0"
                    },
                    "smoothing": {
                        "type": "number",
                        "default": 0.0,
                        "description": "纤维平滑参数，默认0.0"
                    },
                    "thread_count": {
                        "type": "integer",
                        "default": 4,
                        "description": "重建使用的线程数"
                    },
                    "memory_release_interval": {
                        "type": "integer",
                        "default": 10,
                        "description": "每处理多少文件后释放一次内存"
                    }
                },
                "required": ["input_files", "action"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "src_files": {"type": "array", "description": ".sz源文件"},
                    "fib_files": {"type": "array", "description": ".fz纤维方向文件"},
                    "trk_files": {"type": "array", "description": ".tt.gz纤维束文件"}
                }
            },
            version="2024",
            dependencies=["DSI Studio 2024"]
        )

    def _find_auxiliary_files(self, input_file: str) -> dict:
        """自动查找DWI的伴随文件"""
        input_path = Path(input_file)
        base = input_path.stem.replace('.nii', '').replace('.gz', '')
        parent = input_path.parent

        files = {'bval': None, 'bvec': None}

        for ext in ['.bval', '_bval.txt', '.bvals']:
            candidate = parent / f"{base}{ext}"
            if candidate.exists():
                files['bval'] = str(candidate)
                break

        for ext in ['.bvec', '_bvec.txt', '.bvecs']:
            candidate = parent / f"{base}{ext}"
            if candidate.exists():
                files['bvec'] = str(candidate)
                break

        return files

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DSI Studio分析（支持批量处理）"""
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取输入文件列表（兼容单文件和多文件）
        input_files = request.inputs.get("input_files", [])
        if not input_files:
            # 兼容旧的单文件参数
            single_file = request.inputs.get("input_file", "")
            if single_file:
                input_files = [single_file]

        if not input_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="未提供输入文件"
            )

        action = request.params.get("action", "rec")
        method = request.params.get("method", 4)
        param0 = request.params.get("param0", 1.25)
        fiber_count = request.params.get("fiber_count", 100000)
        fa_threshold = request.params.get("fa_threshold", 0.15)
        turning_angle = request.params.get("turning_angle", 55)
        min_length = request.params.get("min_length", 10)
        max_length = request.params.get("max_length", 400)
        # 单个bval/bvec文件（旧接口）
        bval_file = request.params.get("bval_file")
        bvec_file = request.params.get("bvec_file")
        # bval/bvec文件数组（新接口，与input_files对应）
        bval_files = request.inputs.get("bval_files", [])
        bvec_files = request.inputs.get("bvec_files", [])
        # 新增可配置参数
        step_size = request.params.get("step_size", 1.0)
        smoothing = request.params.get("smoothing", 0.0)
        thread_count = request.params.get("thread_count", 4)
        memory_release_interval = request.params.get("memory_release_interval", 10)

        all_output_files = []
        all_commands = []
        failed_files = []

        total_files = len(input_files)
        print(f"  [DSI Studio] 开始处理 {total_files} 个文件, action={action}")

        # 批量处理每个输入文件
        for file_idx, input_file in enumerate(input_files):
            if not input_file:
                continue

            # 为每个文件创建子目录（避免输出覆盖）
            # **关键修复**: DSI Studio 2025使用.sz/.fz扩展名，需要特殊处理
            input_path_for_basename = Path(input_file)
            input_basename = input_path_for_basename.name
            # 移除所有已知的DSI Studio和NIfTI扩展名
            dsi_extensions = ['.src.gz.sz', '.qsdr.fz', '.gqi.fz', '.fib.gz', '.sz', '.fz',
                              '.nii.gz', '.nii', '.src.gz', '.gz']
            for ext in dsi_extensions:
                if input_basename.lower().endswith(ext):
                    input_basename = input_basename[:-len(ext)]
                    break
            # 如果basename仍是'data'（DSI Studio默认输出名），使用父目录名作为被试标识
            if input_basename.lower() in ['data', '']:
                input_basename = input_path_for_basename.parent.name
            print(f"    [输出目录] {input_basename}")
            file_output_dir = output_dir / input_basename
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # 构建命令行参数
            args = [f"--action={action}"]

            if action == "src":
                args.append(f"--source={input_file}")

                # 为当前文件查找匹配的bval/bvec
                current_bval = bval_file  # 单个文件参数优先
                current_bvec = bvec_file

                # 获取输入文件的基础名称用于匹配
                input_path = Path(input_file)
                base_name = input_path.name
                if base_name.endswith('.nii.gz'):
                    base_name = base_name[:-7]
                elif base_name.endswith('.nii'):
                    base_name = base_name[:-4]
                # 处理eddy输出的特殊命名（如HC1_0001_eddy -> HC1_0001）
                match_base = base_name.replace('_eddy', '').replace('_brain', '')

                # 从bval_files数组中查找匹配的文件
                if not current_bval and bval_files:
                    for bf in bval_files:
                        bf_stem = Path(bf).stem.replace('.bval', '')
                        if bf_stem == match_base or match_base.startswith(bf_stem):
                            current_bval = bf
                            print(f"    [bval匹配] {Path(bf).name} -> {base_name}")
                            break
                    # 如果没找到匹配，使用第一个
                    if not current_bval and len(bval_files) > 0:
                        current_bval = bval_files[min(file_idx, len(bval_files)-1)]
                        print(f"    [bval备选] 使用索引 {min(file_idx, len(bval_files)-1)}")

                # 从bvec_files数组中查找匹配的文件
                if not current_bvec and bvec_files:
                    for vf in bvec_files:
                        vf_stem = Path(vf).stem
                        # 处理rotated bvecs的特殊命名
                        if '.eddy_rotated_bvecs' in vf_stem or '_eddy.eddy_rotated_bvecs' in str(vf):
                            vf_stem = vf_stem.replace('.eddy_rotated_bvecs', '').replace('_eddy', '')
                        vf_stem = vf_stem.replace('.bvec', '')
                        if vf_stem == match_base or match_base.startswith(vf_stem):
                            current_bvec = vf
                            print(f"    [bvec匹配] {Path(vf).name} -> {base_name}")
                            break
                    # 如果没找到匹配，使用第一个
                    if not current_bvec and len(bvec_files) > 0:
                        current_bvec = bvec_files[min(file_idx, len(bvec_files)-1)]
                        print(f"    [bvec备选] 使用索引 {min(file_idx, len(bvec_files)-1)}")

                if current_bval:
                    args.append(f"--bval={current_bval}")
                if current_bvec:
                    args.append(f"--bvec={current_bvec}")

                # 如果仍未找到bval/bvec，尝试从输入文件同目录查找
                if not current_bval or not current_bvec:
                    parent_dir = input_path.parent

                    # 查找bval文件 - 使用match_base（去掉_eddy等后缀）而非base_name
                    if not current_bval:
                        # 尝试多种可能的文件名
                        for search_base in [match_base, base_name]:
                            found = False
                            for ext in ['.bval', '_bval.txt', '.bvals']:
                                auto_bval = parent_dir / f"{search_base}{ext}"
                                if auto_bval.exists():
                                    args.append(f"--bval={auto_bval}")
                                    print(f"    [自动发现] bval: {auto_bval.name}")
                                    found = True
                                    break
                            if found:
                                break

                    # 查找bvec文件 - 优先查找rotated bvecs
                    if not current_bvec:
                        # 先查找rotated bvecs
                        rotated_patterns = [
                            f"{base_name}.eddy_rotated_bvecs",  # HC1_0001_eddy.eddy_rotated_bvecs
                            f"{match_base}.eddy_rotated_bvecs", # HC1_0001.eddy_rotated_bvecs
                        ]
                        found = False
                        for pattern in rotated_patterns:
                            auto_bvec = parent_dir / pattern
                            if auto_bvec.exists():
                                args.append(f"--bvec={auto_bvec}")
                                print(f"    [自动发现] bvec: {auto_bvec.name}")
                                found = True
                                break

                        # 如果没找到rotated，查找普通bvec
                        if not found:
                            for search_base in [match_base, base_name]:
                                for ext in ['.bvec', '_bvec.txt', '.bvecs']:
                                    auto_bvec = parent_dir / f"{search_base}{ext}"
                                    if auto_bvec.exists():
                                        args.append(f"--bvec={auto_bvec}")
                                        print(f"    [自动发现] bvec: {auto_bvec.name}")
                                        found = True
                                        break
                                if found:
                                    break

                args.append(f"--output={file_output_dir / 'data.src.gz'}")
            elif action == "rec":
                # 重建步骤：src -> fib
                # **关键修复**: 使用--other_output在重建时导出DTI指标为NIfTI
                # DSI Studio的exp action是用于connectometry数据库，不是单个FIB文件
                other_output = request.params.get("other_output", "dti_fa,md,ad,rd")
                args.extend([
                    f"--source={input_file}",
                    f"--method={method}",
                    f"--param0={param0}",
                    f"--thread={thread_count}",
                    f"--other_output={other_output}",
                    f"--output={file_output_dir}"
                ])
            elif action == "trk":
                # 纤维追踪步骤：fib -> trk
                fib_file = request.params.get("fib_file", input_file)
                args.extend([
                    f"--source={fib_file}",
                    f"--fiber_count={fiber_count}",
                    f"--fa_threshold={fa_threshold}",
                    f"--turning_angle={turning_angle}",
                    f"--min_length={min_length}",
                    f"--max_length={max_length}",
                    f"--step_size={step_size}",
                    f"--smoothing={smoothing}",
                    f"--output={file_output_dir / 'tracks.tt.gz'}"
                ])
            elif action == "ana":
                args.extend([
                    f"--source={input_file}",
                    f"--output={file_output_dir}"
                ])
            elif action == "exp":
                # **关键修复**: DSI Studio的exp action仅用于connectometry数据库导出
                # 对于单个FIB文件，指标应该在rec步骤通过--other_output导出
                # 这里改为：直接收集rec步骤已导出的NIfTI文件
                print(f"  [DSI Studio] exp: 收集已导出的指标文件 (来自rec --other_output)")

                # 查找输入文件所在目录中的已导出NIfTI文件
                input_dir = Path(input_file).parent
                exported_niftis = list(input_dir.glob("*.nii.gz"))

                if exported_niftis:
                    # 复制已导出的文件到输出目录
                    import shutil
                    for nii_file in exported_niftis:
                        dest = file_output_dir / nii_file.name
                        if not dest.exists():
                            shutil.copy2(nii_file, dest)
                            print(f"    [复制] {nii_file.name}")
                    all_output_files.extend([file_output_dir / f.name for f in exported_niftis])
                    result = {"status": "succeeded", "command": f"copy from {input_dir}"}
                else:
                    print(f"    [警告] 未找到已导出的NIfTI文件，尝试DSI Studio exp命令...")
                    # 如果没找到预导出的文件，尝试传统的exp命令（可能失败）
                    export_metrics = request.params.get("export_metrics", "dti_fa,md,ad,rd")
                    args.extend([
                        f"--source={input_file}",
                        f"--export={export_metrics}",
                        f"--output={file_output_dir}"
                    ])
                    result = None  # 让后面的代码执行DSI Studio命令

                # 如果已经处理完成，跳过DSI Studio执行
                if result and result.get("status") == "succeeded":
                    all_commands.append(result.get("command", ""))
                    continue

            print(f"  [DSI Studio] 执行: {action} on {Path(input_file).name}")
            result = run_dsi_studio(args, str(file_output_dir))
            all_commands.append(result.get("command", "") if result else "")

            # 收集该文件的输出（包括新旧格式）
            file_outputs = (
                list(file_output_dir.glob("*.fib.gz")) +  # 旧格式纤维方向文件
                list(file_output_dir.glob("*.fz")) +       # 新格式纤维方向文件
                list(file_output_dir.glob("*.trk.gz")) +   # 旧格式纤维束文件
                list(file_output_dir.glob("*.tt.gz")) +    # 新格式纤维束文件
                list(file_output_dir.glob("*.nii.gz")) +   # 导出的NIfTI文件
                list(file_output_dir.glob("*.src.gz")) +   # 旧格式源文件
                list(file_output_dir.glob("*.sz"))         # 新格式源文件
            )
            all_output_files.extend(file_outputs)

            if result and result.get("status") == "succeeded":
                print(f"  [OK] {Path(input_file).name}")
            else:
                failed_files.append(input_file)
                # 修复：同时检查error、stderr和stdout字段（DSI Studio错误输出到stdout）
                error_msg = "执行未返回结果"
                if result:
                    error_msg = result.get('error') or result.get('stderr')
                    # DSI Studio把错误信息输出到stdout，需要提取
                    if not error_msg and result.get('stdout'):
                        stdout = result.get('stdout', '')
                        # 查找常见错误模式
                        import re
                        error_patterns = [
                            r'file does not exist[^\n]*',
                            r'cannot open[^\n]*',
                            r'error[^\n]*',
                            r'failed[^\n]*',
                            r'invalid[^\n]*'
                        ]
                        for pattern in error_patterns:
                            match = re.search(pattern, stdout, re.IGNORECASE)
                            if match:
                                error_msg = match.group(0).strip()
                                break
                    if not error_msg:
                        error_msg = f"returncode={result.get('returncode')}"
                    if isinstance(error_msg, str) and len(error_msg) > 200:
                        error_msg = error_msg[:200] + "..."
                print(f"  [FAILED] {Path(input_file).name}: {error_msg}")

            # 周期性内存释放（参考FreeSurfer实现）
            if memory_release_interval > 0 and (file_idx + 1) % memory_release_interval == 0:
                import gc
                gc.collect()
                print(f"  [DSI Studio] 内存释放 ({file_idx + 1}/{total_files})")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 判断整体成功/失败
        if len(failed_files) < len(input_files) or all_output_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": [str(f) for f in all_output_files],
                    "action": action,
                    "commands": all_commands,
                    "processed_count": len(input_files) - len(failed_files),
                    "failed_count": len(failed_files)
                }
            )
        else:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"所有 {len(input_files)} 个文件处理失败"
            )


class LocalDiPyTool(BaseTool):
    """本地DiPy工具 - 基于Python的弥散MRI分析

    DiPy (Diffusion Imaging in Python) 提供完整的DTI/DWI分析功能:
    1. DTI拟合: 计算FA, MD, AD, RD等指标
    2. 纤维追踪: 确定性和概率性追踪
    3. 连接组分析: 构建结构连接矩阵

    优势:
    - 纯Python实现，无需外部软件
    - 支持多种重建模型 (DTI, CSD, DKI等)
    - 可作为DSI Studio的备选方案
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dipy_analysis",
            description="""使用DiPy进行弥散MRI分析。

功能:
- dti_fit: DTI拟合，计算FA/MD/AD/RD指标
- tractography: 纤维追踪（确定性/概率性）
- connectivity: 构建结构连接矩阵

DTI指标说明:
- FA (Fractional Anisotropy): 各向异性分数，反映白质完整性
- MD (Mean Diffusivity): 平均扩散率
- AD (Axial Diffusivity): 轴向扩散率
- RD (Radial Diffusivity): 径向扩散率

适用场景:
- DSI Studio不可用时的备选方案
- 需要Python集成的分析流程
- 自定义DTI分析需求""",
            category="analysis",
            supported_modalities=[Modality.DWI],
            executor_type=ExecutorType.PYTHON,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入DWI NIfTI文件列表"
                    },
                    "bval_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "b值文件列表（与input_files对应）"
                    },
                    "bvec_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "b向量文件列表（与input_files对应）"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["dti_fit", "tractography", "connectivity", "all"],
                        "default": "dti_fit",
                        "description": "分析类型: dti_fit(DTI拟合), tractography(纤维追踪), connectivity(连接组), all(全部)"
                    },
                    "mask_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "脑mask文件列表（可选，自动生成）"
                    },
                    "fa_threshold": {
                        "type": "number",
                        "default": 0.2,
                        "description": "FA阈值（用于追踪停止条件）"
                    },
                    "tracking_method": {
                        "type": "string",
                        "enum": ["deterministic", "probabilistic"],
                        "default": "deterministic",
                        "description": "追踪方法: deterministic(确定性), probabilistic(概率性)"
                    },
                    "seed_density": {
                        "type": "integer",
                        "default": 1,
                        "description": "种子点密度（每体素种子数）"
                    },
                    "step_size": {
                        "type": "number",
                        "default": 0.5,
                        "description": "追踪步长（mm）"
                    },
                    "max_angle": {
                        "type": "number",
                        "default": 30,
                        "description": "最大转角（度）"
                    },
                    "atlas_file": {
                        "type": "string",
                        "description": "脑区图谱文件（用于连接组分析）"
                    }
                },
                "required": ["input_files", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "fa_files": {"type": "array", "description": "FA图像文件"},
                    "md_files": {"type": "array", "description": "MD图像文件"},
                    "tractography_files": {"type": "array", "description": "纤维束文件"},
                    "connectivity_matrices": {"type": "array", "description": "连接矩阵文件"}
                }
            },
            version="1.9.0",
            dependencies=["dipy", "nibabel", "numpy", "scipy"]
        )

    def _find_auxiliary_files(self, input_file: str) -> dict:
        """自动查找DWI的伴随文件（bval, bvec）"""
        input_path = Path(input_file)
        base = input_path.stem.replace('.nii', '').replace('.gz', '')
        parent = input_path.parent

        files = {'bval': None, 'bvec': None}

        for ext in ['.bval', '_bval.txt', '.bvals', '_bvals']:
            candidate = parent / f"{base}{ext}"
            if candidate.exists():
                files['bval'] = str(candidate)
                break

        for ext in ['.bvec', '_bvec.txt', '.bvecs', '_bvecs']:
            candidate = parent / f"{base}{ext}"
            if candidate.exists():
                files['bvec'] = str(candidate)
                break

        return files

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DiPy分析"""
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 检查DiPy是否可用
        try:
            import dipy
            from dipy.io.image import load_nifti, save_nifti
            from dipy.io.gradients import read_bvals_bvecs
            from dipy.core.gradients import gradient_table
            print(f"  [DiPy] 版本: {dipy.__version__}")
        except ImportError as e:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"DiPy未安装: {e}. 请运行: pip install dipy"
            )

        # 获取参数
        input_files = request.inputs.get("input_files", [])
        bval_files = request.inputs.get("bval_files", [])
        bvec_files = request.inputs.get("bvec_files", [])
        mask_files = request.inputs.get("mask_files", [])
        analysis_type = request.params.get("analysis_type", "dti_fit")
        fa_threshold = request.params.get("fa_threshold", 0.2)
        tracking_method = request.params.get("tracking_method", "deterministic")
        seed_density = request.params.get("seed_density", 1)
        step_size = request.params.get("step_size", 0.5)
        max_angle = request.params.get("max_angle", 30)
        atlas_file = request.params.get("atlas_file")

        if not input_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="未提供输入文件"
            )

        all_output_files = []
        fa_files = []
        md_files = []
        tractography_files = []
        connectivity_matrices = []
        errors = []

        print(f"  [DiPy] 开始处理 {len(input_files)} 个文件, 分析类型: {analysis_type}")

        for idx, input_file in enumerate(input_files):
            if not input_file or not Path(input_file).exists():
                errors.append(f"文件不存在: {input_file}")
                continue

            # 获取被试ID
            input_path = Path(input_file)
            subject_id = input_path.stem.replace('.nii', '').replace('.gz', '')
            # 处理特殊命名
            subject_id = subject_id.replace('_eddy', '').replace('_brain', '')

            print(f"  [DiPy] 处理 [{idx+1}/{len(input_files)}]: {subject_id}")

            # 创建被试输出目录
            subject_output = output_dir / subject_id
            subject_output.mkdir(parents=True, exist_ok=True)

            try:
                # 加载DWI数据
                data, affine, img = load_nifti(input_file, return_img=True)
                print(f"    [数据] 形状: {data.shape}")

                # 获取bval/bvec文件
                current_bval = bval_files[idx] if idx < len(bval_files) else None
                current_bvec = bvec_files[idx] if idx < len(bvec_files) else None

                # 自动查找bval/bvec
                if not current_bval or not current_bvec:
                    aux_files = self._find_auxiliary_files(input_file)
                    if not current_bval:
                        current_bval = aux_files['bval']
                    if not current_bvec:
                        current_bvec = aux_files['bvec']

                if not current_bval or not current_bvec:
                    errors.append(f"{subject_id}: 未找到bval/bvec文件")
                    continue

                # 读取梯度信息
                bvals, bvecs = read_bvals_bvecs(current_bval, current_bvec)
                gtab = gradient_table(bvals, bvecs)
                print(f"    [梯度] b值范围: {bvals.min():.0f} - {bvals.max():.0f}, 方向数: {len(bvals)}")

                # 获取或生成mask
                current_mask = mask_files[idx] if idx < len(mask_files) else None
                if current_mask and Path(current_mask).exists():
                    mask_data, _ = load_nifti(current_mask)
                    mask = mask_data > 0
                else:
                    # 自动生成mask（使用median_otsu）
                    from dipy.segment.mask import median_otsu
                    _, mask = median_otsu(data, median_radius=2, numpass=1,
                                         vol_idx=range(min(10, data.shape[-1])))
                    print(f"    [Mask] 自动生成，体素数: {mask.sum()}")

                # ========== DTI拟合 ==========
                if analysis_type in ["dti_fit", "all"]:
                    print(f"    [DTI] 开始拟合...")
                    from dipy.reconst.dti import TensorModel

                    tensor_model = TensorModel(gtab)
                    tensor_fit = tensor_model.fit(data, mask=mask)

                    # 计算DTI指标
                    fa = tensor_fit.fa
                    md = tensor_fit.md
                    ad = tensor_fit.ad
                    rd = tensor_fit.rd

                    # 保存FA
                    fa_path = subject_output / f"{subject_id}_FA.nii.gz"
                    save_nifti(str(fa_path), fa.astype(np.float32), affine)
                    fa_files.append(str(fa_path))
                    all_output_files.append(str(fa_path))

                    # 保存MD
                    md_path = subject_output / f"{subject_id}_MD.nii.gz"
                    save_nifti(str(md_path), md.astype(np.float32), affine)
                    md_files.append(str(md_path))
                    all_output_files.append(str(md_path))

                    # 保存AD
                    ad_path = subject_output / f"{subject_id}_AD.nii.gz"
                    save_nifti(str(ad_path), ad.astype(np.float32), affine)
                    all_output_files.append(str(ad_path))

                    # 保存RD
                    rd_path = subject_output / f"{subject_id}_RD.nii.gz"
                    save_nifti(str(rd_path), rd.astype(np.float32), affine)
                    all_output_files.append(str(rd_path))

                    # 保存RGB方向图
                    from dipy.reconst.dti import color_fa
                    rgb = color_fa(fa, tensor_fit.evecs)
                    rgb_path = subject_output / f"{subject_id}_RGB.nii.gz"
                    save_nifti(str(rgb_path), (rgb * 255).astype(np.uint8), affine)
                    all_output_files.append(str(rgb_path))

                    print(f"    [DTI] 完成: FA={fa[mask].mean():.3f}±{fa[mask].std():.3f}")

                # ========== 纤维追踪 ==========
                if analysis_type in ["tractography", "all"]:
                    print(f"    [追踪] 开始 {tracking_method} 追踪...")

                    # 需要先进行DTI拟合
                    if analysis_type != "all":
                        from dipy.reconst.dti import TensorModel
                        tensor_model = TensorModel(gtab)
                        tensor_fit = tensor_model.fit(data, mask=mask)
                        fa = tensor_fit.fa

                    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
                    from dipy.tracking import utils
                    from dipy.tracking.streamline import Streamlines
                    from dipy.io.streamline import save_trk
                    from dipy.io.stateful_tractogram import StatefulTractogram, Space

                    # 创建停止条件
                    stopping_criterion = ThresholdStoppingCriterion(fa, fa_threshold)

                    # 创建种子点
                    seed_mask = fa > fa_threshold
                    seeds = utils.seeds_from_mask(seed_mask, affine, density=seed_density)
                    print(f"    [追踪] 种子点数: {len(seeds)}")

                    if tracking_method == "deterministic":
                        # 确定性追踪
                        from dipy.direction import peaks_from_model
                        from dipy.tracking.local_tracking import LocalTracking

                        # 获取峰值方向
                        peaks = peaks_from_model(tensor_model, data, sphere=None,
                                                relative_peak_threshold=0.5,
                                                min_separation_angle=25,
                                                mask=mask)

                        # 执行追踪
                        streamline_generator = LocalTracking(
                            peaks, stopping_criterion, seeds, affine,
                            step_size=step_size, max_cross=1
                        )
                        streamlines = Streamlines(streamline_generator)

                    else:
                        # 概率性追踪
                        from dipy.reconst.csdeconv import auto_response_ssst, ConstrainedSphericalDeconvModel
                        from dipy.direction import ProbabilisticDirectionGetter
                        from dipy.tracking.local_tracking import LocalTracking
                        from dipy.data import default_sphere

                        # CSD重建
                        response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
                        csd_model = ConstrainedSphericalDeconvModel(gtab, response)
                        csd_fit = csd_model.fit(data, mask=mask)

                        # 概率方向获取器
                        prob_dg = ProbabilisticDirectionGetter.from_shcoeff(
                            csd_fit.shm_coeff, max_angle=max_angle, sphere=default_sphere
                        )

                        # 执行追踪
                        streamline_generator = LocalTracking(
                            prob_dg, stopping_criterion, seeds, affine,
                            step_size=step_size, max_cross=1
                        )
                        streamlines = Streamlines(streamline_generator)

                    print(f"    [追踪] 生成 {len(streamlines)} 条纤维")

                    # 保存纤维束
                    trk_path = subject_output / f"{subject_id}_tracks.trk"
                    sft = StatefulTractogram(streamlines, img, Space.RASMM)
                    save_trk(sft, str(trk_path))
                    tractography_files.append(str(trk_path))
                    all_output_files.append(str(trk_path))

                    print(f"    [追踪] 保存: {trk_path.name}")

                # ========== 连接组分析 ==========
                if analysis_type in ["connectivity", "all"] and atlas_file:
                    print(f"    [连接组] 构建连接矩阵...")

                    # 加载图谱
                    if Path(atlas_file).exists():
                        atlas_data, atlas_affine = load_nifti(atlas_file)

                        # 需要先有纤维束
                        if analysis_type != "all" and not tractography_files:
                            errors.append(f"{subject_id}: 连接组分析需要先进行纤维追踪")
                            continue

                        from dipy.tracking import utils

                        # 计算连接矩阵
                        labels = np.unique(atlas_data[atlas_data > 0]).astype(int)
                        n_labels = len(labels)

                        connectivity_matrix, _ = utils.connectivity_matrix(
                            streamlines, affine, atlas_data,
                            return_mapping=True, mapping_as_streamlines=False
                        )

                        # 保存连接矩阵
                        conn_path = subject_output / f"{subject_id}_connectivity.npy"
                        np.save(str(conn_path), connectivity_matrix)
                        connectivity_matrices.append(str(conn_path))
                        all_output_files.append(str(conn_path))

                        # 保存为CSV（便于查看）
                        csv_path = subject_output / f"{subject_id}_connectivity.csv"
                        np.savetxt(str(csv_path), connectivity_matrix, delimiter=',', fmt='%d')
                        all_output_files.append(str(csv_path))

                        print(f"    [连接组] 完成: {n_labels}个脑区, 连接数: {(connectivity_matrix > 0).sum()}")
                    else:
                        errors.append(f"{subject_id}: 图谱文件不存在: {atlas_file}")

                print(f"  [OK] {subject_id}")

            except Exception as e:
                import traceback
                errors.append(f"{subject_id}: {str(e)}")
                print(f"  [错误] {subject_id}: {e}")
                traceback.print_exc()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if all_output_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": all_output_files,
                    "fa_files": fa_files,
                    "md_files": md_files,
                    "tractography_files": tractography_files,
                    "connectivity_matrices": connectivity_matrices,
                    "analysis_type": analysis_type,
                    "processed_count": len(input_files) - len(errors),
                    "errors": errors if errors else None
                }
            )
        else:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="; ".join(errors) if errors else "未生成任何输出文件"
            )


class LocalFreeSurferTool(BaseTool):
    """
    本地FreeSurfer工具 - 皮层重建和分析（通过WSL2执行）

    FreeSurfer 7.4.1 主要功能:

    1. 基础皮层重建（必须首先运行）:
       - recon-all: 完整皮层重建管道（约6-24小时）
         - directive选项: -all, -autorecon1, -autorecon2, -autorecon3
       - recon-all-clinical: 临床快速版本（约1小时）

    2. 亚结构精细分割（需要先完成recon-all）:
       - segmentBS: 脑干亚结构分割
       - segmentHA: 海马亚结构分割
       - segmentThalamus: 丘脑核团分割

    3. 统计数据导出（需要先完成recon-all）:
       - asegstats2table: 皮下结构体积统计导出
       - aparcstats2table: 皮层分区统计导出

    注意: segmentBS/segmentHA/segmentThalamus 必须在 recon-all 完成后才能运行！

    参考: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="freesurfer_analysis",
            description="使用FreeSurfer 7.4.1进行皮层重建和分析。支持: recon-all(基础皮层重建), segmentBS/segmentHA/segmentThalamus(亚结构分割,需先完成recon-all), asegstats2table/aparcstats2table(统计导出)",
            category="analysis",
            supported_modalities=[Modality.ANAT],
            executor_type=ExecutorType.CLI,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入T1文件路径列表（支持批量处理）"
                    },
                    "subject_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "被试ID列表（与input_files一一对应）"
                    },
                    "command": {
                        "type": "string",
                        "enum": [
                            "recon-all",           # 完整重建
                            "recon-all-clinical",  # 临床快速版本
                            "segmentBS",           # 脑干亚结构分割（需要先完成recon-all）
                            "segmentHA",           # 海马亚结构分割（需要先完成recon-all）
                            "segmentThalamus",     # 丘脑核团分割（需要先完成recon-all）
                            "mri_convert",         # 格式转换
                            "mri_vol2vol",         # 体积配准
                            "mri_surf2surf",       # 表面配准
                            "mris_anatomical_stats", # 解剖统计
                            "asegstats2table",     # 皮下结构统计导出
                            "aparcstats2table",    # 皮层分区统计导出
                            "mri_segstats",        # 分割统计
                            "mris_preproc"         # 表面预处理（用于组分析）
                        ],
                        "description": "FreeSurfer命令（segmentBS/segmentHA/segmentThalamus需要先完成recon-all）"
                    },
                    "directive": {
                        "type": "string",
                        "enum": ["-all", "-autorecon1", "-autorecon2", "-autorecon3",
                                 "-autorecon2-cp", "-autorecon2-wm", "-qcache"],
                        "default": "-all",
                        "description": "recon-all指令"
                    },
                    "hemi": {
                        "type": "string",
                        "enum": ["lh", "rh", "both"],
                        "default": "both",
                        "description": "处理的半球"
                    },
                    "parcellation": {
                        "type": "string",
                        "enum": ["aparc", "aparc.a2009s", "aparc.DKTatlas"],
                        "default": "aparc",
                        "description": "皮层分区图谱"
                    },
                    "subjects_dir": {
                        "type": "string",
                        "description": "FreeSurfer SUBJECTS_DIR路径（用于segmentBS/segmentHA/segmentThalamus等需要已有recon-all结果的命令）"
                    },
                    "parallel": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否并行处理多个被试"
                    },
                    "openmp_threads": {
                        "type": "integer",
                        "default": 4,
                        "description": "OpenMP并行线程数"
                    }
                },
                "required": ["command"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "subjects_dir": {"type": "string"},
                    "stats_files": {"type": "array"},
                    "processed_subjects": {"type": "array"},
                    "cortical_thickness": {"type": "object"},
                    "subcortical_volumes": {"type": "object"}
                }
            },
            version="7.4.1",
            dependencies=["FreeSurfer 7.4.1", "WSL2 Ubuntu-22.04"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行FreeSurfer分析"""
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 获取参数 - 同时检查inputs和params
        input_files = request.inputs.get("input_files", [])
        # 兼容单文件输入
        if not input_files and request.inputs.get("input_file"):
            input_files = [request.inputs.get("input_file")]
        # 也检查params中的input_files（兼容性）
        if not input_files:
            input_files = request.params.get("input_files", [])

        # subject_ids同时检查inputs和params
        subject_ids = request.inputs.get("subject_ids", []) or request.params.get("subject_ids", [])
        # 兼容单被试输入
        if not subject_ids and (request.inputs.get("subject_id") or request.params.get("subject_id")):
            subject_ids = [request.inputs.get("subject_id") or request.params.get("subject_id")]
        # 如果没有指定subject_ids，从文件名生成
        if not subject_ids and input_files:
            subject_ids = [Path(f).stem.replace('.nii', '').replace('.gz', '') for f in input_files]

        command = request.params.get("command", "recon-all")
        directive = request.params.get("directive", "-all")

        # 验证directive参数仅适用于recon-all命令
        if "directive" in request.params and command not in ["recon-all", "recon-all-clinical"]:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"directive参数仅适用于recon-all/recon-all-clinical命令，{command}命令不支持directive参数。"
                      f"对于{command}，请直接调用而无需directive参数。",
                outputs={"modality": "anat", "command": command, "invalid_param": "directive"}
            )

        # 获取或查找 SUBJECTS_DIR（用于需要已有recon-all结果的命令）
        subjects_dir_param = request.params.get("subjects_dir") or request.inputs.get("subjects_dir")
        if subjects_dir_param:
            subjects_dir = Path(subjects_dir_param)
        else:
            subjects_dir = output_dir  # 默认使用当前输出目录

        # 验证directive参数（只对recon-all有效）
        valid_directives = ["-all", "-autorecon1", "-autorecon2", "-autorecon3",
                           "-autorecon2-cp", "-autorecon2-wm", "-qcache"]
        if command == "recon-all" and directive not in valid_directives:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"无效的recon-all指令: {directive}。有效指令: {valid_directives}。"
                      f"如果需要脑干/海马/丘脑分割，请使用 segmentBS/segmentHA/segmentThalamus 命令（需要先完成recon-all）。",
                outputs={"modality": "anat", "invalid_directive": directive, "valid_directives": valid_directives}
            )

        # 检查是否有输入文件（对于recon-all必需）
        if command in ["recon-all", "recon-all-clinical", "mri_convert"] and not input_files:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"FreeSurfer {command} 需要输入文件，但未找到任何T1图像。请确保cohort中包含anat模态的NIfTI文件。",
                outputs={"modality": "anat"}
            )
        hemi = request.params.get("hemi", "both")
        parcellation = request.params.get("parcellation", "aparc")
        parallel = request.params.get("parallel", True)  # 默认启用并行处理
        openmp_threads = request.params.get("openmp_threads", 4)

        # 转换路径
        wsl_output = windows_to_wsl_path(str(output_dir))

        # 执行日志
        execution_log = []
        processed_subjects = []
        failed_subjects = []
        stats_files = []

        print(f"  [FreeSurfer] 命令: {command}, 指令: {directive}")
        print(f"  [FreeSurfer] SUBJECTS_DIR: {output_dir}")
        print(f"  [FreeSurfer] 被试数量: {len(subject_ids)}")

        # 检查FreeSurfer license文件
        license_check = run_freesurfer("test -f $FREESURFER_HOME/license.txt && echo 'LICENSE_OK' || test -f $FREESURFER_HOME/.license && echo 'LICENSE_OK'", str(output_dir))
        if "LICENSE_OK" not in license_check.get("stdout", ""):
            print(f"  [警告] FreeSurfer license文件可能不存在，请检查 $FREESURFER_HOME/license.txt")
            # 继续执行，让FreeSurfer自己报错

        # 根据命令类型执行
        if command == "recon-all":
            # 皮层重建 - 支持并行处理
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            import shutil
            import time
            from src.config_local_tools import run_wsl_command as _run_wsl_command, FREESURFER_HOME as _FS_HOME

            # 最大并行数（用户可通过parallel参数控制，默认8）
            max_parallel = min(request.params.get("max_parallel", 8), 8) if parallel else 1

            # 用于线程安全的列表操作
            lock = threading.Lock()
            # 用于控制启动间隔，避免同时启动太多WSL进程
            start_semaphore = threading.Semaphore(max_parallel)

            def process_single_subject(args):
                """处理单个被试的函数（用于并行执行）"""
                idx, input_file, subject_id = args
                result_info = {
                    "subject_id": subject_id,
                    "status": None,
                    "log": None,
                    "stats_files": []
                }

                # 首先检查输入文件是否存在
                input_path = Path(input_file)
                if not input_path.exists():
                    print(f"    [{subject_id}] 错误: 输入文件不存在 - {input_file}")
                    result_info["status"] = "failed"
                    result_info["log"] = f"[失败] {subject_id}: 输入文件不存在 - {input_file}"
                    return result_info

                # 检查文件大小（NIfTI文件至少应该有几MB）
                file_size_mb = input_path.stat().st_size / (1024 * 1024)
                if file_size_mb < 1:
                    print(f"    [{subject_id}] 警告: 输入文件过小 ({file_size_mb:.2f} MB)，可能转换失败")

                print(f"  [FreeSurfer] 开始处理被试 {idx+1}/{len(subject_ids)}: {subject_id} (文件大小: {file_size_mb:.1f} MB)")

                wsl_input = windows_to_wsl_path(input_file)
                subject_dir = output_dir / subject_id

                # 检查被试文件夹是否已存在
                if subject_dir.exists():
                    # 检查是否已完成处理（存在recon-all.done文件）
                    done_file = subject_dir / "scripts" / "recon-all.done"
                    if done_file.exists():
                        print(f"    [跳过] {subject_id} 已完成处理，复用现有结果")
                        result_info["status"] = "reused"
                        result_info["log"] = f"[复用] {subject_id}: 已完成处理"
                        # 收集输出统计文件
                        subject_stats_dir = subject_dir / "stats"
                        if subject_stats_dir.exists():
                            result_info["stats_files"] = [str(f) for f in subject_stats_dir.glob("*.stats")]
                        return result_info
                    else:
                        # 文件夹存在但未完成 - 删除并重新开始
                        print(f"    [清理] {subject_id} 存在但未完成，删除并重新处理")
                        try:
                            shutil.rmtree(subject_dir)
                            time.sleep(0.5)  # 等待文件系统同步
                        except Exception as e:
                            print(f"    [警告] 无法删除 {subject_dir}: {e}")
                            result_info["status"] = "failed"
                            result_info["log"] = f"[失败] {subject_id}: 无法删除旧文件夹 - {e}"
                            return result_info

                # 构建recon-all命令
                threads_per_subject = max(1, openmp_threads // max_parallel) if parallel else openmp_threads

                # 自定义重试逻辑
                max_retries = 2
                result = None

                # 预先构建环境设置（避免在循环中重复）
                wsl_subjects_dir = windows_to_wsl_path(str(output_dir))
                env_setup = f"export FREESURFER_HOME={_FS_HOME} && source {_FS_HOME}/SetUpFreeSurfer.sh && export SUBJECTS_DIR={wsl_subjects_dir}"

                for retry in range(max_retries + 1):
                    # 每次重试前检查并清理可能存在的未完成文件夹
                    if retry > 0 and subject_dir.exists():
                        done_file = subject_dir / "scripts" / "recon-all.done"
                        if not done_file.exists():
                            print(f"    [{subject_id}] 重试前清理未完成的文件夹...")
                            try:
                                shutil.rmtree(subject_dir)
                                time.sleep(0.5)
                            except Exception as e:
                                print(f"    [{subject_id}] 清理失败: {e}")

                    fs_cmd = f"recon-all -s {subject_id} -i {wsl_input} {directive} -sd {wsl_output}"

                    if threads_per_subject > 1:
                        fs_cmd = f"export OMP_NUM_THREADS={threads_per_subject} && {fs_cmd}"

                    if hemi != "both":
                        fs_cmd += f" -hemi {hemi}"

                    if retry > 0:
                        print(f"    [{subject_id}] 重试 {retry}/{max_retries}...")
                        time.sleep(2)  # 重试前等待2秒

                    print(f"    [{subject_id}] 命令: {fs_cmd[:80]}...")

                    # 执行命令
                    result = _run_wsl_command(fs_cmd, env_setup, timeout=86400, retries=0)

                    # 检查是否是 "re-run existing subject" 错误
                    stdout = result.get("stdout", "")
                    if "re-run an existing subject" in stdout and retry < max_retries:
                        print(f"    [{subject_id}] 检测到'existing subject'错误，清理后重试...")
                        continue

                    # 如果成功或其他类型失败，退出重试循环
                    if result and result.get("status") == "succeeded":
                        break
                    elif result and retry < max_retries and "timeout" in str(result.get("error", "")).lower():
                        print(f"    [{subject_id}] 超时，将重试...")
                        continue
                    else:
                        break

                # 安全检查：确保result不为None
                if result is None:
                    result = {"status": "failed", "error": "执行未返回结果", "returncode": -1}

                print(f"    [{subject_id}] 状态: {result.get('status')}, returncode={result.get('returncode')}")

                if result.get("status") == "succeeded":
                    result_info["status"] = "succeeded"
                    result_info["log"] = f"[成功] {subject_id}"
                    subject_stats_dir = output_dir / subject_id / "stats"
                    if subject_stats_dir.exists():
                        result_info["stats_files"] = [str(f) for f in subject_stats_dir.glob("*.stats")]
                else:
                    result_info["status"] = "failed"
                    # 提取错误信息
                    stderr_msg = result.get("stderr", "") or ""
                    stdout_msg = result.get("stdout", "") or ""
                    error_field = result.get("error", "") or ""

                    error_lines = []
                    if stdout_msg:
                        for line in stdout_msg.split('\n'):
                            line_lower = line.lower()
                            if 'error' in line_lower or 'cannot' in line_lower or 'failed' in line_lower or 'license' in line_lower:
                                error_lines.append(line.strip())

                    if error_lines:
                        error_msg = "; ".join(error_lines[:5])
                    elif stderr_msg:
                        error_msg = stderr_msg
                    elif error_field:
                        error_msg = error_field
                    else:
                        error_msg = f"Unknown error, returncode={result.get('returncode')}"

                    result_info["log"] = f"[失败] {subject_id}: {error_msg[:500]}"
                    print(f"    [{subject_id}] 错误: {error_msg[:300]}")

                return result_info

            # 准备任务列表
            tasks = [(i, input_file, subject_id) for i, (input_file, subject_id) in enumerate(zip(input_files, subject_ids))]

            if parallel and len(tasks) > 1:
                print(f"  [FreeSurfer] 并行处理模式: 最多 {max_parallel} 个被试同时处理")

                # 分批处理 - 每批完成后再启动下一批，避免WSL资源耗尽
                total_batches = (len(tasks) + max_parallel - 1) // max_parallel

                for batch_idx in range(total_batches):
                    batch_start = batch_idx * max_parallel
                    batch_end = min(batch_start + max_parallel, len(tasks))
                    batch_tasks = tasks[batch_start:batch_end]

                    print(f"  [FreeSurfer] 批次 {batch_idx + 1}/{total_batches}: 处理被试 {batch_start + 1}-{batch_end}")

                    # 如果不是第一批，等待一段时间让WSL释放资源
                    if batch_idx > 0:
                        wait_time = 30  # 增加到30秒，确保内存和资源完全释放
                        print(f"  [FreeSurfer] 等待 {wait_time} 秒让WSL释放资源...")
                        time.sleep(wait_time)
                        # 强制垃圾回收
                        import gc
                        gc.collect()

                    # 并行执行当前批次
                    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                        future_to_subject = {executor.submit(process_single_subject, task): task[2] for task in batch_tasks}

                        for future in as_completed(future_to_subject):
                            subject_id = future_to_subject[future]
                            try:
                                result_info = future.result()
                                with lock:
                                    if result_info["status"] in ["succeeded", "reused"]:
                                        processed_subjects.append(result_info["subject_id"])
                                    else:
                                        failed_subjects.append(result_info["subject_id"])
                                    execution_log.append(result_info["log"])
                                    stats_files.extend(result_info["stats_files"])
                            except Exception as e:
                                print(f"    [{subject_id}] 异常: {e}")
                                with lock:
                                    failed_subjects.append(subject_id)
                                    execution_log.append(f"[异常] {subject_id}: {str(e)[:200]}")

                    print(f"  [FreeSurfer] 批次 {batch_idx + 1} 完成")
            else:
                # 串行执行
                print(f"  [FreeSurfer] 串行处理模式")
                for task in tasks:
                    result_info = process_single_subject(task)
                    if result_info["status"] in ["succeeded", "reused"]:
                        processed_subjects.append(result_info["subject_id"])
                    else:
                        failed_subjects.append(result_info["subject_id"])
                    execution_log.append(result_info["log"])
                    stats_files.extend(result_info["stats_files"])

        elif command == "recon-all-clinical":
            # 临床快速版本（约1小时）
            for i, (input_file, subject_id) in enumerate(zip(input_files, subject_ids)):
                print(f"  [FreeSurfer Clinical] 处理被试 {i+1}/{len(subject_ids)}: {subject_id}")

                wsl_input = windows_to_wsl_path(input_file)
                fs_cmd = f"recon-all-clinical.sh {wsl_input} {subject_id} {openmp_threads} {wsl_output}"

                result = run_freesurfer(fs_cmd, str(output_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                else:
                    failed_subjects.append(subject_id)

        elif command == "mri_convert":
            # 格式转换
            for input_file, subject_id in zip(input_files, subject_ids):
                wsl_input = windows_to_wsl_path(input_file)
                output_file = f"{wsl_output}/{subject_id}.nii.gz"
                fs_cmd = f"mri_convert {wsl_input} {output_file}"

                result = run_freesurfer(fs_cmd, str(output_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                    stats_files.append(str(output_dir / f"{subject_id}.nii.gz"))
                else:
                    failed_subjects.append(subject_id)

        elif command == "segmentBS":
            # 脑干亚结构分割（需要先完成recon-all）
            # FreeSurfer 7.x: segmentBS.sh SUBJECT_ID [SUBJECTS_DIR]
            print(f"  [FreeSurfer] 脑干亚结构分割（需要recon-all已完成）")

            # 确定 SUBJECTS_DIR - 优先使用参数指定的目录，否则搜索
            fs_subjects_dir = subjects_dir
            if not subjects_dir_param:
                # 搜索 tools 目录下的 recon-all 输出
                tools_parent = output_dir.parent
                for task_dir in tools_parent.iterdir():
                    if task_dir.is_dir() and "freesurfer" in task_dir.name.lower():
                        # 检查是否有已完成的 recon-all 被试
                        for subdir in task_dir.iterdir():
                            if subdir.is_dir() and (subdir / "scripts" / "recon-all.done").exists():
                                fs_subjects_dir = task_dir
                                print(f"  [FreeSurfer] 找到 recon-all 结果: {task_dir}")
                                break
                        if fs_subjects_dir != output_dir:
                            break

            wsl_subjects_dir = windows_to_wsl_path(str(fs_subjects_dir))

            # 自动检测已完成recon-all的被试
            if not subject_ids:
                subject_ids = [d.name for d in fs_subjects_dir.iterdir()
                              if d.is_dir() and (d / "scripts" / "recon-all.done").exists()]

            if not subject_ids:
                potential_subjects = [d.name for d in fs_subjects_dir.iterdir() if d.is_dir()]
                if potential_subjects:
                    error_msg = (f"segmentBS 失败：找到 {len(potential_subjects)} 个被试目录，"
                                f"但没有任何完成 recon-all 的被试。请先运行 recon-all 完成皮层重建。")
                else:
                    error_msg = "segmentBS 失败：未找到任何被试目录。请先运行 recon-all 完成皮层重建。"
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"command": command, "prerequisite": "recon-all", "searched_dir": str(fs_subjects_dir)}
                )

            print(f"  [FreeSurfer] SUBJECTS_DIR: {fs_subjects_dir}")

            for i, subject_id in enumerate(subject_ids):
                print(f"  [FreeSurfer] 处理被试 {i+1}/{len(subject_ids)}: {subject_id}")

                # 检查是否已经有脑干分割结果
                bs_stats_file = fs_subjects_dir / subject_id / "mri" / "brainstemSsLabels.v13.FSvoxelSpace.mgz"
                if bs_stats_file.exists():
                    print(f"    [跳过] {subject_id} 已有脑干分割结果")
                    processed_subjects.append(subject_id)
                    stats_files.append(str(bs_stats_file))
                    continue

                fs_cmd = f"segmentBS.sh {subject_id} {wsl_subjects_dir}"
                result = run_freesurfer(fs_cmd, str(fs_subjects_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                    if bs_stats_file.exists():
                        stats_files.append(str(bs_stats_file))
                    execution_log.append(f"[成功] {subject_id}: 脑干分割完成")
                else:
                    failed_subjects.append(subject_id)
                    error_msg = result.get('stderr', '')[:200] if result else "执行未返回结果"
                    execution_log.append(f"[失败] {subject_id}: {error_msg}")

        elif command == "segmentHA":
            # 海马亚结构分割（需要先完成recon-all）
            # FreeSurfer 7.x: segmentHA_T1.sh SUBJECT_ID [SUBJECTS_DIR]
            print(f"  [FreeSurfer] 海马亚结构分割（需要recon-all已完成）")

            # 确定 SUBJECTS_DIR - 优先使用参数指定的目录，否则搜索
            fs_subjects_dir = subjects_dir
            if not subjects_dir_param:
                tools_parent = output_dir.parent
                for task_dir in tools_parent.iterdir():
                    if task_dir.is_dir() and "freesurfer" in task_dir.name.lower():
                        for subdir in task_dir.iterdir():
                            if subdir.is_dir() and (subdir / "scripts" / "recon-all.done").exists():
                                fs_subjects_dir = task_dir
                                print(f"  [FreeSurfer] 找到 recon-all 结果: {task_dir}")
                                break
                        if fs_subjects_dir != output_dir:
                            break

            wsl_subjects_dir = windows_to_wsl_path(str(fs_subjects_dir))

            if not subject_ids:
                subject_ids = [d.name for d in fs_subjects_dir.iterdir()
                              if d.is_dir() and (d / "scripts" / "recon-all.done").exists()]

            if not subject_ids:
                potential_subjects = [d.name for d in fs_subjects_dir.iterdir() if d.is_dir()]
                if potential_subjects:
                    error_msg = (f"segmentHA 失败：找到 {len(potential_subjects)} 个被试目录，"
                                f"但没有任何完成 recon-all 的被试。请先运行 recon-all 完成皮层重建。")
                else:
                    error_msg = "segmentHA 失败：未找到任何被试目录。请先运行 recon-all 完成皮层重建。"
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"command": command, "prerequisite": "recon-all", "searched_dir": str(fs_subjects_dir)}
                )

            print(f"  [FreeSurfer] SUBJECTS_DIR: {fs_subjects_dir}")

            for i, subject_id in enumerate(subject_ids):
                print(f"  [FreeSurfer] 处理被试 {i+1}/{len(subject_ids)}: {subject_id}")

                # 检查是否已经有海马分割结果
                ha_stats_file = fs_subjects_dir / subject_id / "mri" / "lh.hippoAmygLabels-T1.v22.FSvoxelSpace.mgz"
                if ha_stats_file.exists():
                    print(f"    [跳过] {subject_id} 已有海马分割结果")
                    processed_subjects.append(subject_id)
                    stats_files.append(str(ha_stats_file))
                    continue

                fs_cmd = f"segmentHA_T1.sh {subject_id} {wsl_subjects_dir}"
                result = run_freesurfer(fs_cmd, str(fs_subjects_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                    if ha_stats_file.exists():
                        stats_files.append(str(ha_stats_file))
                    execution_log.append(f"[成功] {subject_id}: 海马分割完成")
                else:
                    failed_subjects.append(subject_id)
                    error_msg = result.get('stderr', '')[:200] if result else "执行未返回结果"
                    execution_log.append(f"[失败] {subject_id}: {error_msg}")

        elif command == "segmentThalamus":
            # 丘脑核团分割（需要先完成recon-all）
            # FreeSurfer 7.x: segmentThalamicNuclei.sh SUBJECT_ID [SUBJECTS_DIR]
            print(f"  [FreeSurfer] 丘脑核团分割（需要recon-all已完成）")

            # 确定 SUBJECTS_DIR - 优先使用参数指定的目录，否则搜索
            fs_subjects_dir = subjects_dir
            if not subjects_dir_param:
                tools_parent = output_dir.parent
                for task_dir in tools_parent.iterdir():
                    if task_dir.is_dir() and "freesurfer" in task_dir.name.lower():
                        for subdir in task_dir.iterdir():
                            if subdir.is_dir() and (subdir / "scripts" / "recon-all.done").exists():
                                fs_subjects_dir = task_dir
                                print(f"  [FreeSurfer] 找到 recon-all 结果: {task_dir}")
                                break
                        if fs_subjects_dir != output_dir:
                            break

            wsl_subjects_dir = windows_to_wsl_path(str(fs_subjects_dir))

            if not subject_ids:
                subject_ids = [d.name for d in fs_subjects_dir.iterdir()
                              if d.is_dir() and (d / "scripts" / "recon-all.done").exists()]

            if not subject_ids:
                potential_subjects = [d.name for d in fs_subjects_dir.iterdir() if d.is_dir()]
                if potential_subjects:
                    error_msg = (f"segmentThalamus 失败：找到 {len(potential_subjects)} 个被试目录，"
                                f"但没有任何完成 recon-all 的被试。请先运行 recon-all 完成皮层重建。")
                else:
                    error_msg = "segmentThalamus 失败：未找到任何被试目录。请先运行 recon-all 完成皮层重建。"
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"command": command, "prerequisite": "recon-all", "searched_dir": str(fs_subjects_dir)}
                )

            print(f"  [FreeSurfer] SUBJECTS_DIR: {fs_subjects_dir}")

            for i, subject_id in enumerate(subject_ids):
                print(f"  [FreeSurfer] 处理被试 {i+1}/{len(subject_ids)}: {subject_id}")

                # 检查是否已经有丘脑分割结果
                thal_stats_file = fs_subjects_dir / subject_id / "mri" / "ThalamicNuclei.v13.T1.FSvoxelSpace.mgz"
                if thal_stats_file.exists():
                    print(f"    [跳过] {subject_id} 已有丘脑分割结果")
                    processed_subjects.append(subject_id)
                    stats_files.append(str(thal_stats_file))
                    continue

                fs_cmd = f"segmentThalamicNuclei.sh {subject_id} {wsl_subjects_dir}"
                result = run_freesurfer(fs_cmd, str(fs_subjects_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                    if thal_stats_file.exists():
                        stats_files.append(str(thal_stats_file))
                    execution_log.append(f"[成功] {subject_id}: 丘脑分割完成")
                else:
                    failed_subjects.append(subject_id)
                    error_msg = result.get('stderr', '')[:200] if result else "执行未返回结果"
                    execution_log.append(f"[失败] {subject_id}: {error_msg}")

        elif command == "asegstats2table":
            # 皮下结构统计导出（批量处理所有被试）
            # 注意：此命令依赖于 recon-all 的输出，必须先完成皮层重建

            if not subject_ids:
                # 自动检测SUBJECTS_DIR下的被试
                subject_ids = [d.name for d in output_dir.iterdir()
                              if d.is_dir() and (d / "stats" / "aseg.stats").exists()]

            # 检查是否找到有效的被试
            if not subject_ids:
                # 检查是否有被试目录但没有 aseg.stats（说明 recon-all 未完成）
                potential_subjects = [d.name for d in output_dir.iterdir() if d.is_dir()]
                if potential_subjects:
                    error_msg = (f"asegstats2table 失败：找到 {len(potential_subjects)} 个被试目录，"
                                f"但没有任何 stats/aseg.stats 文件。请先运行 recon-all 完成皮层重建。")
                else:
                    error_msg = "asegstats2table 失败：未找到任何被试目录。请先运行 recon-all 完成皮层重建。"

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"command": command, "prerequisite": "recon-all"}
                )

            if subject_ids:
                subjects_str = " ".join(subject_ids)
                output_table = f"{wsl_output}/aseg_stats_table.txt"

                fs_cmd = f"asegstats2table --subjects {subjects_str} --tablefile {output_table} --sd {wsl_output}"
                result = run_freesurfer(fs_cmd, str(output_dir))

                if result and result.get("status") == "succeeded":
                    stats_files.append(str(output_dir / "aseg_stats_table.txt"))
                    processed_subjects = subject_ids
                    execution_log.append(f"[成功] 导出 {len(subject_ids)} 个被试的皮下结构统计")
                else:
                    error_msg = result.get('stderr', '')[:200] if result else "执行未返回结果"
                    execution_log.append(f"[失败] asegstats2table: {error_msg}")

        elif command == "aparcstats2table":
            # 皮层分区统计导出
            # 注意：此命令依赖于 recon-all 的输出，必须先完成皮层重建

            if not subject_ids:
                subject_ids = [d.name for d in output_dir.iterdir()
                              if d.is_dir() and (d / "stats" / f"lh.{parcellation}.stats").exists()]

            # 检查是否找到有效的被试
            if not subject_ids:
                potential_subjects = [d.name for d in output_dir.iterdir() if d.is_dir()]
                if potential_subjects:
                    error_msg = (f"aparcstats2table 失败：找到 {len(potential_subjects)} 个被试目录，"
                                f"但没有任何 stats/lh.{parcellation}.stats 文件。请先运行 recon-all 完成皮层重建。")
                else:
                    error_msg = "aparcstats2table 失败：未找到任何被试目录。请先运行 recon-all 完成皮层重建。"

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"command": command, "prerequisite": "recon-all"}
                )

            if subject_ids:
                subjects_str = " ".join(subject_ids)

                # 导出左右半球的厚度、面积、体积
                for hemi_name in ["lh", "rh"]:
                    for measure in ["thickness", "area", "volume"]:
                        output_table = f"{wsl_output}/aparc_{hemi_name}_{measure}_table.txt"
                        fs_cmd = f"aparcstats2table --subjects {subjects_str} --hemi {hemi_name} " \
                                f"--meas {measure} --parc {parcellation} --tablefile {output_table} --sd {wsl_output}"

                        result = run_freesurfer(fs_cmd, str(output_dir))

                        if result and result.get("status") == "succeeded":
                            stats_files.append(str(output_dir / f"aparc_{hemi_name}_{measure}_table.txt"))

                processed_subjects = subject_ids
                execution_log.append(f"[成功] 导出 {len(subject_ids)} 个被试的皮层分区统计")

        elif command == "mris_anatomical_stats":
            # 解剖统计（单个被试）
            for subject_id in subject_ids:
                for hemi_name in ["lh", "rh"] if hemi == "both" else [hemi]:
                    fs_cmd = f"mris_anatomical_stats -sd {wsl_output} {subject_id} {hemi_name}"
                    result = run_freesurfer(fs_cmd, str(output_dir))

                    if result and result.get("status") == "succeeded":
                        processed_subjects.append(subject_id)

        elif command == "mri_segstats":
            # 分割统计
            for subject_id in subject_ids:
                seg_file = f"{wsl_output}/{subject_id}/mri/aseg.mgz"
                brain_file = f"{wsl_output}/{subject_id}/mri/brain.mgz"
                output_stats = f"{wsl_output}/{subject_id}/stats/custom_segstats.stats"

                fs_cmd = f"mri_segstats --seg {seg_file} --ctab $FREESURFER_HOME/FreeSurferColorLUT.txt " \
                        f"--i {brain_file} --sum {output_stats}"

                result = run_freesurfer(fs_cmd, str(output_dir))

                if result and result.get("status") == "succeeded":
                    processed_subjects.append(subject_id)
                    stats_files.append(str(output_dir / subject_id / "stats" / "custom_segstats.stats"))

        elif command == "mris_preproc":
            # 表面预处理（用于组分析）
            if subject_ids:
                subjects_str = " ".join([f"--s {s}" for s in subject_ids])

                for hemi_name in ["lh", "rh"]:
                    output_file = f"{wsl_output}/group_{hemi_name}_thickness.mgh"
                    fs_cmd = f"mris_preproc {subjects_str} --hemi {hemi_name} " \
                            f"--meas thickness --target fsaverage --out {output_file} --sd {wsl_output}"

                    result = run_freesurfer(fs_cmd, str(output_dir))

                    if result and result.get("status") == "succeeded":
                        stats_files.append(str(output_dir / f"group_{hemi_name}_thickness.mgh"))

                processed_subjects = subject_ids

        else:
            # 自定义命令
            fs_cmd = command
            result = run_freesurfer(fs_cmd, str(output_dir))
            status = result.get("status", "unknown") if result else "failed"
            execution_log.append(f"[自定义] {command}: {status}")

        # 保存执行日志
        log_path = output_dir / "freesurfer_execution.log"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(execution_log))

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 尝试提取统计数据摘要
        stats_summary = self._extract_stats_summary(output_dir, processed_subjects)

        # 判断整体状态
        if len(processed_subjects) > 0:
            status = "succeeded" if not failed_subjects else "partial"
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status=status,
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "subjects_dir": str(output_dir),
                    "modality": "anat",  # FreeSurfer处理解剖结构像
                    "processed_subjects": processed_subjects,
                    "failed_subjects": failed_subjects,
                    "stats_files": stats_files,
                    "stats_summary": stats_summary,
                    "command": command,
                    "directive": directive
                },
                artifacts=[
                    {"name": "freesurfer_execution.log", "path": str(log_path)}
                ]
            )
        else:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=f"所有被试处理失败。日志: {execution_log[:3]}",
                outputs={"modality": "anat", "execution_log": execution_log}
            )

    def _extract_stats_summary(self, subjects_dir: Path, subject_ids: List[str]) -> Dict[str, Any]:
        """从FreeSurfer输出中提取统计数据摘要"""
        summary = {
            "subcortical_volumes": {},
            "cortical_thickness": {},
            "total_brain_volume": {}
        }

        for subject_id in subject_ids[:5]:  # 只处理前5个被试作为摘要
            # 读取aseg.stats
            aseg_file = subjects_dir / subject_id / "stats" / "aseg.stats"
            if aseg_file.exists():
                try:
                    volumes = self._parse_aseg_stats(aseg_file)
                    summary["subcortical_volumes"][subject_id] = volumes
                except Exception as e:
                    print(f"  [警告] 解析 {subject_id} aseg.stats 失败: {e}")

            # 读取aparc.stats（皮层厚度）
            for hemi in ["lh", "rh"]:
                aparc_file = subjects_dir / subject_id / "stats" / f"{hemi}.aparc.stats"
                if aparc_file.exists():
                    try:
                        thickness = self._parse_aparc_stats(aparc_file)
                        if subject_id not in summary["cortical_thickness"]:
                            summary["cortical_thickness"][subject_id] = {}
                        summary["cortical_thickness"][subject_id][hemi] = thickness
                    except Exception as e:
                        print(f"  [警告] 解析 {subject_id} {hemi}.aparc.stats 失败: {e}")

        return summary

    def _parse_aseg_stats(self, stats_file: Path) -> Dict[str, float]:
        """解析aseg.stats文件，提取皮下结构体积"""
        volumes = {}

        with open(stats_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    # 提取全脑测量值
                    if "BrainSegVol," in line:
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            volumes["BrainSegVol"] = float(parts[3])
                    elif "eTIV" in line:
                        parts = line.strip().split(",")
                        if len(parts) >= 4:
                            volumes["eTIV"] = float(parts[3])
                elif not line.startswith("#") and line.strip():
                    # 数据行
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        structure = parts[4]  # 结构名称
                        volume = float(parts[3])  # 体积
                        volumes[structure] = volume

        return volumes

    def _parse_aparc_stats(self, stats_file: Path) -> Dict[str, float]:
        """解析aparc.stats文件，提取皮层厚度"""
        thickness = {}

        with open(stats_file, "r") as f:
            for line in f:
                if not line.startswith("#") and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        region = parts[0]  # 脑区名称
                        thick = float(parts[4])  # 平均厚度
                        thickness[region] = thick

        return thickness


class LocalFSLTool(BaseTool):
    """本地FSL工具 - 结构、功能和弥散MRI分析（通过WSL2）

    DWI/DTI分析支持：
    - bet: 脑提取
    - eddy: 涡流和运动校正（自动生成acqp/index）
    - dtifit: DTI张量拟合

    eddy必需参数：
    - bvecs, bvals: 梯度方向和b值文件（来自dcm2niix转换）
    - mask: 脑掩膜（可自动生成）
    - acqp, index: 采集参数和体积索引（可自动生成）
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fsl_analysis",
            description="""使用FSL进行神经影像分析，通过WSL2执行。
            DWI分析支持eddy涡流校正和dtifit张量拟合。
            eddy所需的acqp和index文件可自动生成。""",
            category="analysis",
            supported_modalities=[Modality.ANAT, Modality.FUNC, Modality.DWI],
            executor_type=ExecutorType.CLI,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入文件路径列表（支持批量处理）"
                    },
                    "command": {
                        "type": "string",
                        "enum": FSL_SUPPORTED_COMMANDS,  # 从config_local_tools导入，确保与执行白名单一致
                        "description": "FSL命令"
                    },
                    "options": {
                        "type": "string",
                        "default": "",
                        "description": "额外命令选项"
                    },
                    # DWI/DTI参数
                    "bvecs": {
                        "type": "string",
                        "description": "梯度方向文件路径（eddy/dtifit必需，可自动查找）"
                    },
                    "bvals": {
                        "type": "string",
                        "description": "b值文件路径（eddy/dtifit必需，可自动查找）"
                    },
                    "mask": {
                        "type": "string",
                        "description": "脑掩膜文件路径（eddy/dtifit必需，可用bet自动生成）"
                    },
                    "acqp": {
                        "type": "string",
                        "description": "采集参数文件路径（eddy必需，可自动生成）"
                    },
                    "index": {
                        "type": "string",
                        "description": "体积索引文件路径（eddy必需，可自动生成）"
                    },
                    # 批处理参数（参考FreeSurfer实现）
                    "batch_size": {
                        "type": "integer",
                        "default": 6,
                        "description": "每批处理的文件数量（eddy内存占用大，建议6）"
                    },
                    "checkpoint": {
                        "type": "boolean",
                        "default": True,
                        "description": "是否跳过已处理的文件"
                    },
                    "release_interval": {
                        "type": "integer",
                        "default": 20,
                        "description": "批次间资源释放等待时间（秒）"
                    }
                },
                "required": ["input_files", "command"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "rotated_bvecs": {"type": "string", "description": "eddy输出的旋转后bvecs文件"}
                }
            },
            version="6.0",
            dependencies=["FSL", "WSL2"]
        )

    def _find_dwi_auxiliary_files(self, input_file: str, output_dir: Path = None) -> dict:
        """自动查找DWI的伴随文件（bvecs, bvals, mask, json）

        搜索位置优先级：
        1. 输入文件所在目录
        2. 输入文件的父目录（如果输入在子目录中）
        3. output_dir（如果提供）
        4. 同级任务目录（如eddy/bet生成的mask）
        """
        input_path = Path(input_file)
        base = input_path.stem.replace('.nii', '').replace('.gz', '')
        parent = input_path.parent

        files = {
            'bvecs': None,
            'bvals': None,
            'mask': None,
            'json': None
        }

        # 构建搜索目录列表
        search_dirs = [parent]

        # 如果输入在子目录中（如HC1_0001/HC1_0001.nii.gz），也搜索父目录
        if parent.name == base or parent.name.startswith(base.split('_')[0]):
            search_dirs.append(parent.parent)

        # 如果提供了output_dir，也在其中搜索
        if output_dir and output_dir.exists():
            search_dirs.append(output_dir)

        # 搜索同级任务目录（如task_03_fsl_analysis可能有eddy生成的mask）
        tools_parent = parent.parent if parent.name == base else parent.parent.parent if parent.parent.name == base else None
        if tools_parent and 'tools' in str(tools_parent):
            for sibling in tools_parent.iterdir():
                if sibling.is_dir() and sibling != parent and 'fsl' in sibling.name.lower():
                    search_dirs.append(sibling)

        # 查找bvecs
        bvec_patterns = ['.bvec', '_bvec.txt', '.bvecs', '_bvecs', '_eddy.eddy_rotated_bvecs']
        for search_dir in search_dirs:
            for pattern in bvec_patterns:
                candidate = search_dir / f"{base}{pattern}"
                if candidate.exists():
                    files['bvecs'] = str(candidate)
                    break
            if files['bvecs']:
                break

        # 查找bvals
        bval_patterns = ['.bval', '_bval.txt', '.bvals', '_bvals']
        for search_dir in search_dirs:
            for pattern in bval_patterns:
                candidate = search_dir / f"{base}{pattern}"
                if candidate.exists():
                    files['bvals'] = str(candidate)
                    break
            if files['bvals']:
                break

        # 查找mask - 扩展搜索模式包括eddy生成的mask
        mask_patterns = [
            '_mask.nii.gz', '_mask.nii',
            '_brain_mask.nii.gz', '_brain.nii.gz',
            '_b0_brain_mask.nii.gz', '_b0_brain.nii.gz',  # eddy预处理生成的
            '_eddy_brain_mask.nii.gz'  # 可能的eddy后处理
        ]
        for search_dir in search_dirs:
            for pattern in mask_patterns:
                candidate = search_dir / f"{base}{pattern}"
                if candidate.exists():
                    files['mask'] = str(candidate)
                    break
            if files['mask']:
                break

        # 查找JSON（用于获取PE方向）
        for search_dir in search_dirs:
            candidate = search_dir / f"{base}.json"
            if candidate.exists():
                files['json'] = str(candidate)
                break

        return files

    def _get_pe_direction_from_json(self, json_file: str) -> str:
        """从dcm2niix生成的JSON sidecar中获取相位编码方向"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            pe_dir = data.get('PhaseEncodingDirection', '')

            # 映射JSON值到FSL格式
            json_to_fsl = {
                'j-': 'AP',
                'j': 'PA',
                'i-': 'LR',
                'i': 'RL',
            }

            return json_to_fsl.get(pe_dir, 'AP')  # 默认AP
        except Exception:
            return 'AP'

    def _generate_acqp_file(self, output_dir: Path, pe_direction: str = "AP",
                            total_readout_time: float = 0.05) -> str:
        """生成FSL eddy所需的acqp.txt文件"""
        # 相位编码方向映射
        pe_map = {
            "AP": "0 -1 0",   # Anterior-Posterior (y负方向)
            "PA": "0 1 0",    # Posterior-Anterior (y正方向)
            "LR": "-1 0 0",   # Left-Right (x负方向)
            "RL": "1 0 0",    # Right-Left (x正方向)
            "IS": "0 0 -1",   # Inferior-Superior (z负方向)
            "SI": "0 0 1"     # Superior-Inferior (z正方向)
        }

        direction = pe_map.get(pe_direction.upper(), "0 -1 0")
        acqp_content = f"{direction} {total_readout_time}\n"

        acqp_path = output_dir / "acqp.txt"
        acqp_path.write_text(acqp_content)

        return str(acqp_path)

    def _generate_index_file(self, output_dir: Path, num_volumes: int) -> str:
        """生成FSL eddy所需的index.txt文件"""
        # 所有体积使用acqp.txt中的第1行参数
        index_content = " ".join(["1"] * num_volumes) + "\n"

        index_path = output_dir / "index.txt"
        index_path.write_text(index_content)

        return str(index_path)

    def _get_dwi_num_volumes(self, nii_file: str) -> int:
        """获取DWI文件的体积数量"""
        try:
            import nibabel as nib
            img = nib.load(nii_file)
            if len(img.shape) == 4:
                return img.shape[3]
            return 1
        except:
            return 1

    def _filter_unprocessed(self, input_files: list, output_dir: Path, command: str) -> list:
        """检查点机制：跳过已处理的文件（参考FreeSurfer实现）"""
        # 各命令对应的输出文件模式
        output_patterns = {
            "bet": "_brain.nii.gz",
            "fast": "_seg.nii.gz",
            "flirt": "_flirt.nii.gz",
            "eddy": "_eddy.nii.gz",
            "dtifit": "_dti_FA.nii.gz"
        }

        pattern = output_patterns.get(command, "_out.nii.gz")
        unprocessed = []
        skipped_count = 0

        for input_file in input_files:
            if not input_file:
                continue
            basename = Path(input_file).stem.replace(".nii", "").replace(".gz", "")
            expected_output = output_dir / f"{basename}{pattern}"

            if expected_output.exists():
                skipped_count += 1
            else:
                unprocessed.append(input_file)

        if skipped_count > 0:
            print(f"  [FSL] 检查点: 跳过 {skipped_count} 个已处理文件")

        return unprocessed

    def _release_resources(self, wait_time: int = 20):
        """批次间释放资源（参考FreeSurfer实现）"""
        import gc
        import time

        # 强制垃圾回收
        gc.collect()

        # 等待WSL资源释放
        if wait_time > 0:
            print(f"  [FSL] 释放资源，等待 {wait_time}s...")
            time.sleep(wait_time)

    def _execute_group_command(self, command: str, request: ToolCallRequest,
                               input_files: list, output_dir: Path,
                               options: str, modality: str) -> ToolCallResult:
        """
        执行群组分析命令（TBSS、bedpostx、probtrackx）
        这些命令需要处理所有被试的数据，不是逐个文件处理
        """
        import shutil

        # ========== TBSS 白质分析工具链 ==========
        if command.startswith("tbss_"):
            print(f"  [TBSS] {command} - 群组白质分析（需要所有被试的FA图像）")

            # TBSS工作目录 - 对于tbss_2/3/4，使用tbss_1_preproc创建的目录
            tbss_base_dir = request.params.get("tbss_base_dir")
            if tbss_base_dir and command in ["tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats"]:
                # 使用tbss_1_preproc创建的TBSS目录
                tbss_dir = Path(tbss_base_dir)
                print(f"  [TBSS] 使用现有TBSS目录: {tbss_dir}")
            else:
                # tbss_1_preproc: 在当前output_dir创建新的TBSS目录
                tbss_dir = Path(output_dir) / "TBSS"
            tbss_dir.mkdir(parents=True, exist_ok=True)

            if command == "tbss_1_preproc":
                # 步骤1: 预处理 - 复制所有FA图像到TBSS/FA目录
                fa_dir = tbss_dir / "FA"
                origdata_dir = tbss_dir / "origdata"
                fa_dir.mkdir(exist_ok=True)
                origdata_dir.mkdir(exist_ok=True)

                # 调试日志：显示收到的输入文件
                print(f"  [TBSS] 收到 {len(input_files)} 个输入文件")
                if input_files:
                    print(f"  [TBSS] 输入文件示例: {[Path(f).name for f in input_files[:3]]}")

                # 筛选FA图像
                fa_files = [f for f in input_files if "_FA." in str(f) or "_fa." in str(f)]
                print(f"  [TBSS] 筛选后找到 {len(fa_files)} 个FA图像")

                if not fa_files:
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error=f"tbss_1_preproc需要FA图像。在{len(input_files)}个输入文件中未找到FA图像（文件名应包含'_FA'）"
                    )

                print(f"  [TBSS] 找到 {len(fa_files)} 个FA图像")

                # 复制FA文件到TBSS/FA目录
                for fa_file in fa_files:
                    dest = fa_dir / Path(fa_file).name
                    if not dest.exists():
                        shutil.copy(fa_file, dest)
                        print(f"    复制: {Path(fa_file).name}")

                # 构建TBSS预处理命令 - 需要在FA目录中执行
                wsl_fa_dir = windows_to_wsl_path(str(fa_dir))
                fsl_cmd = f"cd {wsl_fa_dir} && tbss_1_preproc *.nii.gz"

                print(f"  [执行] {fsl_cmd}")
                result = run_fsl(fsl_cmd)

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded" if result.get("returncode") == 0 else "failed",
                    outputs={
                        "output_dir": str(tbss_dir),
                        "fa_count": len(fa_files),
                        "command": command,
                        "stdout": result.get("stdout", ""),
                        "modality": modality
                    },
                    error=result.get("stderr", "") if result.get("returncode") != 0 else None
                )

            elif command == "tbss_2_reg":
                # 步骤2: 配准到FMRIB58_FA标准空间
                wsl_tbss_dir = windows_to_wsl_path(str(tbss_dir))

                fa_dir = tbss_dir / "FA"
                if not fa_dir.exists():
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error="tbss_2_reg需要先运行tbss_1_preproc。TBSS/FA目录不存在"
                    )

                # 检查FA目录中是否有文件
                fa_files_in_dir = list(fa_dir.glob("*.nii.gz"))
                if not fa_files_in_dir:
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error=f"tbss_2_reg需要先运行tbss_1_preproc。TBSS/FA目录存在但为空（没有FA图像）。请检查tbss_1_preproc是否成功执行。"
                    )

                print(f"  [TBSS] FA目录包含 {len(fa_files_in_dir)} 个FA图像")

                target_option = options if options else "-T"
                fsl_cmd = f"cd {wsl_tbss_dir} && tbss_2_reg {target_option}"

                print(f"  [执行] {fsl_cmd}")
                result = run_fsl(fsl_cmd)

                # 验证输出：检查是否生成了配准结果文件
                # tbss_2_reg -T 使用FMRIB58_FA模板时，会生成*_FA_to_target*.nii.gz文件
                registered_files_after = list(fa_dir.glob("*_FA_to_target*.nii.gz")) + list(fa_dir.glob("*_to_target.nii.gz"))
                target_file = fa_dir / "target.nii.gz"

                # 检查是否有输出（正常执行FNIRT会有大量输出）
                has_output = bool(result.get("stdout", "").strip() or result.get("stderr", "").strip())

                if result.get("returncode") == 0:
                    # 即使returncode=0，也要验证输出
                    if not target_file.exists() and not registered_files_after:
                        if not has_output:
                            # 命令可能未实际执行
                            return ToolCallResult(
                                call_id=request.call_id,
                                tool_name=self.definition.name,
                                status="failed",
                                error="tbss_2_reg命令执行异常：返回码为0但无任何输出且未生成配准文件。请检查FSL环境配置或手动运行tbss_2_reg -T诊断问题。"
                            )
                        # 有输出但没生成文件，给出警告
                        print(f"  [警告] tbss_2_reg完成但未检测到配准文件（*_FA_to_target*.nii.gz或target.nii.gz）")
                        print(f"  [提示] 可能需要等待后台FNIRT任务完成，或检查FSL版本是否兼容")
                else:
                    # returncode != 0，失败
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        outputs={
                            "output_dir": str(tbss_dir),
                            "command": command,
                            "stdout": result.get("stdout", ""),
                            "modality": modality
                        },
                        error=result.get("stderr", "") or f"tbss_2_reg执行失败，返回码: {result.get('returncode')}"
                    )

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded",
                    outputs={
                        "output_dir": str(tbss_dir),
                        "command": command,
                        "stdout": result.get("stdout", ""),
                        "registered_files": [str(f) for f in registered_files_after],
                        "modality": modality
                    },
                    error=None
                )

            elif command == "tbss_3_postreg":
                # 步骤3: 后处理和骨架投影
                wsl_tbss_dir = windows_to_wsl_path(str(tbss_dir))

                # 检查tbss_2_reg的输出：FA目录中应该有配准后的文件或target文件
                fa_dir = tbss_dir / "FA"
                if not fa_dir.exists():
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error="tbss_3_postreg需要先运行tbss_2_reg。TBSS/FA目录不存在"
                    )

                # 检查是否有target文件或配准后的FA文件（tbss_2_reg的输出）
                target_file = fa_dir / "target.nii.gz"
                registered_files = list(fa_dir.glob("*_to_target.nii.gz"))
                if not target_file.exists() and not registered_files:
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error="tbss_3_postreg需要先运行tbss_2_reg。未找到target.nii.gz或配准后的FA文件（*_to_target.nii.gz）"
                    )

                print(f"  [TBSS] 找到配准结果: target={target_file.exists()}, 配准文件={len(registered_files)}个")

                fsl_cmd = f"cd {wsl_tbss_dir} && tbss_3_postreg -S"

                print(f"  [执行] {fsl_cmd}")
                result = run_fsl(fsl_cmd)

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded" if result.get("returncode") == 0 else "failed",
                    outputs={
                        "output_dir": str(tbss_dir),
                        "command": command,
                        "stdout": result.get("stdout", ""),
                        "modality": modality
                    },
                    error=result.get("stderr", "") if result.get("returncode") != 0 else None
                )

            elif command == "tbss_4_prestats":
                # 步骤4: 准备统计分析
                wsl_tbss_dir = windows_to_wsl_path(str(tbss_dir))

                skeleton_file = tbss_dir / "stats" / "all_FA_skeletonised.nii.gz"
                if not skeleton_file.exists():
                    return ToolCallResult(
                        call_id=request.call_id,
                        tool_name=self.definition.name,
                        status="failed",
                        error="tbss_4_prestats需要先运行tbss_3_postreg。骨架文件不存在"
                    )

                threshold = options if options else "0.2"
                fsl_cmd = f"cd {wsl_tbss_dir} && tbss_4_prestats {threshold}"

                print(f"  [执行] {fsl_cmd}")
                result = run_fsl(fsl_cmd)

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded" if result.get("returncode") == 0 else "failed",
                    outputs={
                        "output_dir": str(tbss_dir),
                        "command": command,
                        "threshold": threshold,
                        "stdout": result.get("stdout", ""),
                        "modality": modality
                    },
                    error=result.get("stderr", "") if result.get("returncode") != 0 else None
                )

        # ========== bedpostx 纤维追踪预处理 ==========
        elif command == "bedpostx":
            print(f"  [bedpostx] 贝叶斯扩散参数估计 - 为概率纤维追踪准备")

            if not input_files:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="bedpostx需要输入DWI文件"
                )

            # 解析参数
            n_fibers, weight, burnin = "2", "1", "1000"
            if options:
                for opt in options.split():
                    if opt.startswith("n="):
                        n_fibers = opt.split("=")[1]
                    elif opt.startswith("w="):
                        weight = opt.split("=")[1]
                    elif opt.startswith("b="):
                        burnin = opt.split("=")[1]

            # bedpostx需要逐个被试处理（每个被试需要独立的输入目录）
            all_results = []
            failed_subjects = []
            total_subjects = len(input_files)

            print(f"  [bedpostx] 开始处理 {total_subjects} 个被试")

            for idx, input_file in enumerate(input_files):
                if not input_file or not Path(input_file).exists():
                    print(f"  [跳过] 文件不存在: {input_file}")
                    continue

                input_basename = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                print(f"\n  [{idx+1}/{total_subjects}] 被试: {input_basename}")

                # 为每个被试创建独立的bedpostx输入目录
                subject_bedpostx_dir = Path(output_dir) / f"{input_basename}_bedpostx"
                subject_bedpostx_dir.mkdir(parents=True, exist_ok=True)

                # 查找必需的辅助文件（bvecs, bvals, mask）
                aux_files = self._find_dwi_auxiliary_files(input_file, output_dir)
                bvecs = request.inputs.get("bvecs") or aux_files.get('bvecs')
                bvals = request.inputs.get("bvals") or aux_files.get('bvals')
                mask = request.inputs.get("mask") or aux_files.get('mask')

                # **关键修复**: 从数组中匹配对应的bvecs/bvals/mask（参考eddy实现）
                if not bvecs or not bvals:
                    bvec_files = request.inputs.get("bvec_files", [])
                    bval_files = request.inputs.get("bval_files", [])

                    if bvec_files and bval_files:
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        # 查找匹配的bvec
                        for bvec_f in bvec_files:
                            bvec_stem = Path(bvec_f).stem.replace('.bvec', '').replace('_eddy', '')
                            if bvec_stem == input_stem or input_stem.startswith(bvec_stem):
                                bvecs = bvec_f
                                print(f"    [匹配] bvecs: {Path(bvec_f).name}")
                                break
                        # 查找匹配的bval
                        for bval_f in bval_files:
                            bval_stem = Path(bval_f).stem.replace('.bval', '')
                            if bval_stem == input_stem or input_stem.startswith(bval_stem):
                                bvals = bval_f
                                print(f"    [匹配] bvals: {Path(bval_f).name}")
                                break

                if not mask:
                    mask_files = request.inputs.get("mask_files", [])
                    if mask_files:
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        for mask_f in mask_files:
                            mask_stem = Path(mask_f).stem.replace('_mask', '').replace('.nii', '').replace('.gz', '')
                            if mask_stem == input_stem or input_stem.startswith(mask_stem):
                                mask = mask_f
                                print(f"    [匹配] mask: {Path(mask_f).name}")
                                break

                # 验证必需文件
                if not bvecs or not Path(bvecs).exists():
                    error_msg = f"bedpostx需要bvecs文件。被试: {input_basename}"
                    print(f"    [错误] {error_msg}")
                    failed_subjects.append(input_basename)
                    continue

                if not bvals or not Path(bvals).exists():
                    error_msg = f"bedpostx需要bvals文件。被试: {input_basename}"
                    print(f"    [错误] {error_msg}")
                    failed_subjects.append(input_basename)
                    continue

                # mask可选，如果没有则警告（bedpostx可以运行但结果可能不准确）
                if not mask or not Path(mask).exists():
                    print(f"    [警告] 未找到mask文件，将不使用mask（可能影响结果质量）")

                # 准备bedpostx输入目录（FSL bedpostx要求特定文件名）
                # 必需文件: data.nii.gz, bvecs, bvals
                # 可选文件: nodif_brain_mask.nii.gz
                print(f"    [准备] 复制文件到bedpostx输入目录...")

                data_file = subject_bedpostx_dir / "data.nii.gz"
                shutil.copy(input_file, data_file)
                print(f"      ✓ data.nii.gz")

                shutil.copy(bvecs, subject_bedpostx_dir / "bvecs")
                print(f"      ✓ bvecs")

                shutil.copy(bvals, subject_bedpostx_dir / "bvals")
                print(f"      ✓ bvals")

                if mask and Path(mask).exists():
                    shutil.copy(mask, subject_bedpostx_dir / "nodif_brain_mask.nii.gz")
                    print(f"      ✓ nodif_brain_mask.nii.gz")

                # 执行bedpostx
                wsl_input_dir = windows_to_wsl_path(str(subject_bedpostx_dir))
                fsl_cmd = f"bedpostx {wsl_input_dir} -n {n_fibers} -w {weight} -b {burnin}"

                print(f"    [执行] {fsl_cmd}")
                print(f"    [提示] bedpostx可能需要数小时，请耐心等待...")

                result = run_fsl(fsl_cmd)

                if result.get("returncode") == 0:
                    # bedpostx会创建 .bedpostX 后缀的输出目录
                    output_bedpostx_dir = str(subject_bedpostx_dir) + ".bedpostX"
                    all_results.append({
                        "subject": input_basename,
                        "status": "succeeded",
                        "output_dir": output_bedpostx_dir
                    })
                    print(f"    [成功] bedpostx完成")
                else:
                    failed_subjects.append(input_basename)
                    all_results.append({
                        "subject": input_basename,
                        "status": "failed",
                        "error": result.get("stderr", "Unknown error")
                    })
                    print(f"    [失败] bedpostx执行失败")

            # 汇总结果
            succeeded = [r for r in all_results if r.get("status") == "succeeded"]
            print(f"\n  [bedpostx汇总] 成功: {len(succeeded)}/{total_subjects}, 失败: {len(failed_subjects)}")

            if len(succeeded) == 0:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"所有被试的bedpostx都失败了。失败被试: {', '.join(failed_subjects)}",
                    outputs={
                        "modality": modality,
                        "failed_subjects": failed_subjects
                    }
                )
            else:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded" if len(failed_subjects) == 0 else "partial",
                    outputs={
                        "output_dir": str(output_dir),
                        "command": command,
                        "n_fibers": n_fibers,
                        "results": all_results,
                        "succeeded_count": len(succeeded),
                        "failed_count": len(failed_subjects),
                        "failed_subjects": failed_subjects if failed_subjects else None,
                        "modality": modality
                    }
                )

        # ========== probtrackx 概率纤维追踪 ==========
        elif command == "probtrackx":
            print(f"  [probtrackx] 概率纤维追踪")

            # **P0修复1**: 验证bedpostx_dir参数
            bedpostx_dir = request.inputs.get("bedpostx_dir")
            if not bedpostx_dir:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="probtrackx需要bedpostx_dir参数（bedpostx的输出目录）"
                )

            # **P0修复2**: 验证bedpostx输出目录存在性
            bedpostx_path = Path(bedpostx_dir)
            if not bedpostx_path.exists():
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"bedpostx输出目录不存在: {bedpostx_dir}"
                )

            # **P0修复3**: bedpostx会创建 .bedpostX 后缀的目录，自动处理
            if not bedpostx_path.name.endswith('.bedpostX'):
                # 尝试查找 .bedpostX 后缀的目录
                bedpostx_with_suffix = bedpostx_path.parent / (bedpostx_path.name + '.bedpostX')
                if bedpostx_with_suffix.exists():
                    bedpostx_path = bedpostx_with_suffix
                    print(f"  [自动修正] 使用bedpostx输出目录: {bedpostx_path.name}")
                else:
                    # 在当前目录下查找所有 .bedpostX 目录
                    parent_dir = bedpostx_path if bedpostx_path.is_dir() else bedpostx_path.parent
                    bedpostx_dirs = list(parent_dir.glob("*.bedpostX"))
                    if bedpostx_dirs:
                        bedpostx_path = bedpostx_dirs[0]
                        print(f"  [自动检测] 找到bedpostx输出目录: {bedpostx_path.name}")
                    else:
                        return ToolCallResult(
                            call_id=request.call_id,
                            tool_name=self.definition.name,
                            status="failed",
                            error=f"未找到bedpostx输出目录（预期目录名以.bedpostX结尾）。提供的路径: {bedpostx_dir}"
                        )

            # **P0修复4**: 验证bedpostx输出完整性
            required_files = {
                "merged": "贝叶斯采样结果",
                "nodif_brain_mask.nii.gz": "脑掩膜",
                "mean_fsumsamples.nii.gz": "平均纤维方向"
            }

            missing_files = []
            for req_file, description in required_files.items():
                file_path = bedpostx_path / req_file
                if not file_path.exists():
                    missing_files.append(f"{req_file} ({description})")

            if missing_files:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"bedpostx输出不完整，缺少以下文件:\n  - " + "\n  - ".join(missing_files) +
                          f"\n\nbedpostx可能尚未完成或执行失败。请确认bedpostx目录: {bedpostx_path}"
                )

            print(f"  [验证通过] bedpostx输出目录完整: {bedpostx_path.name}")

            # **P0修复5**: 验证种子点mask文件
            seed_mask = request.inputs.get("seed") or request.inputs.get("seed_mask")
            if not seed_mask:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="probtrackx需要seed参数（种子点mask文件）"
                )

            seed_path = Path(seed_mask)
            if not seed_path.exists():
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"种子点mask文件不存在: {seed_mask}"
                )

            # 验证seed mask是NIfTI格式
            if not (seed_path.suffix in ['.nii', '.gz'] or seed_path.name.endswith('.nii.gz')):
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=f"种子点mask必须是NIfTI格式(.nii或.nii.gz)。当前文件: {seed_path.name}"
                )

            print(f"  [验证通过] 种子点mask: {seed_path.name}")

            # 构建probtrackx2命令
            wsl_bedpostx = windows_to_wsl_path(str(bedpostx_path))
            wsl_seed = windows_to_wsl_path(str(seed_path))
            wsl_out = windows_to_wsl_path(str(output_dir))

            # probtrackx2命令（FSL 6.0+版本）
            fsl_cmd = (f"probtrackx2 "
                      f"--samples={wsl_bedpostx}/merged "
                      f"--mask={wsl_bedpostx}/nodif_brain_mask "
                      f"--seed={wsl_seed} "
                      f"--dir={wsl_out}")

            # 添加用户自定义选项
            if options:
                fsl_cmd += f" {options}"
            else:
                # 默认参数（如果用户未提供）
                fsl_cmd += " --nsamples=5000 --nsteps=2000"

            print(f"  [执行] {fsl_cmd}")
            print(f"  [提示] probtrackx可能需要较长时间，具体取决于采样数和种子点数量...")

            result = run_fsl(fsl_cmd)

            if result.get("returncode") == 0:
                # 检查输出文件是否生成
                expected_output = output_dir / "fdt_paths.nii.gz"
                if expected_output.exists():
                    print(f"  [成功] 概率追踪完成，输出: {expected_output.name}")
                else:
                    print(f"  [警告] 命令成功但未找到预期输出文件: fdt_paths.nii.gz")

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded",
                    outputs={
                        "output_dir": str(output_dir),
                        "command": command,
                        "bedpostx_dir": str(bedpostx_path),
                        "seed_mask": str(seed_path),
                        "output_files": [str(f) for f in output_dir.glob("fdt_*")],
                        "stdout": result.get("stdout", ""),
                        "modality": modality
                    }
                )
            else:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    outputs={
                        "output_dir": str(output_dir),
                        "command": command,
                        "modality": modality
                    },
                    error=result.get("stderr", "Unknown error")
                )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行FSL分析（支持批量处理）"""
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 先获取命令类型（用于判断是否需要输入文件）
        command = request.params.get("command", "bet")

        # 获取输入文件列表（兼容单文件和多文件）
        input_files = request.inputs.get("input_files", [])
        if not input_files:
            # 兼容旧的单文件参数
            single_file = request.inputs.get("input_file", "")
            if single_file:
                input_files = [single_file]

        # tbss_2/3/4 不需要输入文件，它们在 TBSS 工作目录中操作
        tbss_no_input_commands = ["tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats"]
        if not input_files and command not in tbss_no_input_commands:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error="未提供输入文件"
            )
        options = request.params.get("options", "")
        wsl_output = windows_to_wsl_path(str(output_dir))

        # 【关键】FSL 命令白名单验证 - 防止 LLM 发明不存在的命令
        # 使用统一定义的命令列表（从config_local_tools导入）
        if command not in FSL_SUPPORTED_COMMANDS:
            # 提供针对性的错误消息
            if "tbss" in command.lower():
                error_msg = (
                    f"TBSS命令需要按顺序执行4个步骤。支持的TBSS命令:\n"
                    f"  - tbss_1_preproc: 预处理FA图像\n"
                    f"  - tbss_2_reg: 配准到标准空间\n"
                    f"  - tbss_3_postreg: 骨架投影\n"
                    f"  - tbss_4_prestats: 准备统计分析\n"
                    f"所有支持的FSL命令: {', '.join(FSL_SUPPORTED_COMMANDS)}"
                )
            elif command in ["bedpostx", "probtrackx"]:
                error_msg = (
                    f"FSL纤维追踪命令: bedpostx（扩散参数估计）和 probtrackx（概率追踪）\n"
                    f"所有支持的FSL命令: {', '.join(FSL_SUPPORTED_COMMANDS)}"
                )
            else:
                error_msg = f"不支持的FSL命令: {command}。支持的命令: {', '.join(FSL_SUPPORTED_COMMANDS)}"

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=error_msg
            )

        # 根据命令确定数据模态
        dwi_commands = {
            "eddy", "dtifit", "eddy_correct", "bedpostx", "probtrackx", "fslmeants",
            "tbss_1_preproc", "tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats"
        }
        modality = "dwi" if command in dwi_commands else "anat"

        # 批处理参数
        # **关键修复**: eddy命令内存占用极高，特别是多壳层数据(127 volumes)
        # 强制串行处理以避免内存耗尽导致系统死机
        if command == "eddy":
            batch_size = request.params.get("batch_size", 2)  # eddy默认串行处理
            release_interval = request.params.get("release_interval", 30)  # eddy间隔60秒
            print(f"  [FSL] eddy命令检测到，使用串行处理模式 (batch_size=1) 以避免内存问题")
        else:
            batch_size = request.params.get("batch_size", 6)
            release_interval = request.params.get("release_interval", 20)
        checkpoint = request.params.get("checkpoint", True)

        # 检查点：跳过已处理的文件
        # 注意：tbss_2/3/4 不需要输入文件，它们在 TBSS 目录中操作，不应被检查点跳过
        tbss_no_input_commands = ["tbss_2_reg", "tbss_3_postreg", "tbss_4_prestats"]
        if checkpoint and command not in tbss_no_input_commands:
            input_files = self._filter_unprocessed(input_files, output_dir, command)
            if not input_files:
                print(f"  [FSL] 所有文件已处理完成")
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="succeeded",
                    started_at=start_time.isoformat(),
                    finished_at=datetime.now().isoformat(),
                    duration_seconds=0,
                    outputs={
                        "output_files": [],
                        "modality": modality,
                        "message": "所有文件已在之前处理完成（检查点跳过）"
                    }
                )

        all_output_files = []
        all_commands = []
        failed_files = []

        # 计算批次数
        total_files = len(input_files)
        num_batches = (total_files + batch_size - 1) // batch_size

        print(f"  [FSL] 开始批处理: {total_files} 个文件, {num_batches} 批, 每批最多 {batch_size} 个")

        # ========== 群组分析命令（不按单个文件循环）==========
        # TBSS、bedpostx、probtrackx需要处理所有文件，不是逐个文件处理
        if command.startswith("tbss_") or command in ["bedpostx", "probtrackx"]:
            return self._execute_group_command(command, request, input_files, output_dir, options, modality)

        # 批量处理每个输入文件
        for file_idx, input_file in enumerate(input_files):
            # 批次边界检测和资源释放
            if file_idx > 0 and file_idx % batch_size == 0:
                batch_num = file_idx // batch_size
                print(f"  [FSL] 完成批次 {batch_num}/{num_batches}, 释放资源...")
                self._release_resources(release_interval)

            if not input_file or not Path(input_file).exists():
                print(f"  [跳过] 文件不存在: {input_file}")
                continue

            wsl_input = windows_to_wsl_path(input_file)
            input_basename = Path(input_file).stem.replace(".nii", "")

            # 构建FSL命令
            if command == "bet":
                fsl_cmd = f"bet {wsl_input} {wsl_output}/{input_basename}_brain -f 0.5 -g 0 {options}"
            elif command == "fast":
                fsl_cmd = f"fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o {wsl_output}/{input_basename} {wsl_input} {options}"
            elif command == "flirt":
                # 默认使用FSL标准MNI模板（$FSLDIR会被shell展开）
                ref = request.inputs.get("reference", "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz")
                # 如果用户提供了Windows路径，转换为WSL路径
                if ref and not ref.startswith("$") and os.path.isabs(ref):
                    ref = windows_to_wsl_path(ref)
                fsl_cmd = f"flirt -in {wsl_input} -ref {ref} -out {wsl_output}/{input_basename}_flirt -omat {wsl_output}/{input_basename}.mat {options}"
            elif command == "fslstats":
                fsl_cmd = f"fslstats {wsl_input} -M -S -R {options}"
            elif command == "fslmaths":
                fsl_cmd = f"fslmaths {wsl_input} {options} {wsl_output}/{input_basename}_out"
            elif command == "eddy":
                # eddy涡流校正 - 使用eddy_openmp（CPU版本，更通用）
                # FSL没有plain "eddy"命令，只有eddy_openmp、eddy_cuda*、eddy_correct

                # 获取或自动查找参数
                aux_files = self._find_dwi_auxiliary_files(input_file)

                bvecs = request.inputs.get("bvecs") or aux_files.get('bvecs')
                bvals = request.inputs.get("bvals") or aux_files.get('bvals')
                mask = request.inputs.get("mask") or aux_files.get('mask')
                acqp = request.inputs.get("acqp")
                index_file = request.inputs.get("index")

                # **关键修复1**: 如果没有找到mask，从mask_files数组中查找
                if not mask:
                    mask_files = request.inputs.get("mask_files", [])
                    if mask_files:
                        # 根据文件名匹配对应的mask
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        for mask_f in mask_files:
                            mask_stem = Path(mask_f).stem.replace('_mask', '').replace('.nii', '').replace('.gz', '')
                            if mask_stem == input_stem or input_stem.startswith(mask_stem) or mask_stem in input_stem:
                                mask = mask_f
                                print(f"  [输入匹配] {input_basename}: 从数组中找到对应的mask文件")
                                break
                        # 如果没有精确匹配，使用第一个mask
                        if not mask and mask_files:
                            mask = mask_files[0]
                            print(f"  [输入备用] {input_basename}: 使用第一个可用的mask文件")

                # **关键修复2**: 如果没有找到bvecs/bvals，从bvec_files/bval_files数组中查找
                if not bvecs or not bvals:
                    bvec_files = request.inputs.get("bvec_files", [])
                    bval_files = request.inputs.get("bval_files", [])

                    # 根据文件名匹配对应的bvec/bval
                    if bvec_files and bval_files:
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        for bvec_f in bvec_files:
                            bvec_stem = Path(bvec_f).stem.replace('.bvec', '')
                            if bvec_stem == input_stem or input_stem.startswith(bvec_stem):
                                bvecs = bvec_f
                                break
                        for bval_f in bval_files:
                            bval_stem = Path(bval_f).stem.replace('.bval', '')
                            if bval_stem == input_stem or input_stem.startswith(bval_stem):
                                bvals = bval_f
                                break
                        if bvecs and bvals:
                            print(f"  [输入匹配] {input_basename}: 从数组中找到对应的bvec/bval文件")

                # 验证bvecs和bvals（这两个无法自动生成，必须来自数据）
                if not bvecs or not bvals:
                    print(f"  [错误] {input_basename}: eddy缺少bvecs/bvals文件")
                    failed_files.append(input_file)
                    continue

                # 自动生成acqp（如果未提供）
                if not acqp:
                    pe_dir = 'AP'
                    if aux_files.get('json'):
                        pe_dir = self._get_pe_direction_from_json(aux_files['json'])
                    acqp = self._generate_acqp_file(output_dir, pe_dir)
                    print(f"  [自动生成] acqp.txt (PE方向: {pe_dir})")

                # 自动生成index（如果未提供）
                if not index_file:
                    num_volumes = self._get_dwi_num_volumes(input_file)
                    index_file = self._generate_index_file(output_dir, num_volumes)
                    print(f"  [自动生成] index.txt ({num_volumes} 个体积)")

                # 如果没有mask，先用bet生成
                if not mask:
                    print(f"  [自动生成] 使用bet生成脑掩膜...")
                    # 提取B0作为bet输入（取第一个体积）
                    b0_file = output_dir / f"{input_basename}_b0.nii.gz"
                    b0_brain = output_dir / f"{input_basename}_b0_brain"

                    # 使用fslroi提取第一个体积
                    fslroi_cmd = f"fslroi {wsl_input} {windows_to_wsl_path(str(b0_file))} 0 1"
                    roi_result = run_fsl(fslroi_cmd)

                    if roi_result.get("status") == "succeeded":
                        # 使用bet进行脑提取
                        bet_cmd = f"bet {windows_to_wsl_path(str(b0_file))} {windows_to_wsl_path(str(b0_brain))} -m -f 0.3"
                        bet_result = run_fsl(bet_cmd)

                        if bet_result.get("status") == "succeeded":
                            mask = str(b0_brain) + "_mask.nii.gz"
                            print(f"  [OK] 脑掩膜生成成功")
                        else:
                            print(f"  [警告] bet失败，尝试继续...")
                    else:
                        print(f"  [警告] fslroi失败，尝试继续...")

                if not mask:
                    print(f"  [错误] {input_basename}: 无法获取或生成脑掩膜")
                    failed_files.append(input_file)
                    continue

                # 转换路径为WSL格式
                wsl_bvecs = windows_to_wsl_path(bvecs)
                wsl_bvals = windows_to_wsl_path(bvals)
                wsl_mask = windows_to_wsl_path(mask)
                wsl_acqp = windows_to_wsl_path(acqp)
                wsl_index = windows_to_wsl_path(index_file)

                # **检测数据规模**: 多壳层数据(>100 volumes)需要更长处理时间和更多内存
                # 读取bval文件获取volume数量
                try:
                    with open(bvals, 'r') as f:
                        bval_values = f.read().strip().split()
                        num_volumes = len(bval_values)
                        unique_bvals = len(set(bval_values))

                        if num_volumes > 100:
                            print(f"    [eddy警告] 检测到大规模数据: {num_volumes} volumes, {unique_bvals} b-shells")
                            print(f"    [eddy警告] 此数据集需要大量内存和时间 (可能数小时)")
                        else:
                            print(f"    [eddy] 数据规模: {num_volumes} volumes, {unique_bvals} b-shells")
                except Exception as e:
                    print(f"    [eddy] 无法读取bval文件: {e}")

                # 构建eddy命令
                fsl_cmd = (f"eddy_openmp --imain={wsl_input} --mask={wsl_mask} "
                          f"--acqp={wsl_acqp} --index={wsl_index} "
                          f"--bvecs={wsl_bvecs} --bvals={wsl_bvals} "
                          f"--out={wsl_output}/{input_basename}_eddy --repol {options}")
            # eddy_quad 已移除 - 该工具未安装在 FSL 6.0.5.1 中
            # 替代方案: 使用 python_stats 读取 eddy 输出的 .eddy_movement_rms 文件
            elif command == "dtifit":
                # DTI张量拟合
                # 获取或自动查找参数（传入output_dir以扩展搜索范围）
                aux_files = self._find_dwi_auxiliary_files(input_file, output_dir)

                bvecs = request.inputs.get("bvecs") or aux_files.get('bvecs')
                bvals = request.inputs.get("bvals") or aux_files.get('bvals')
                mask = request.inputs.get("mask") or aux_files.get('mask')

                # **关键修复1**: 如果没有找到mask，从mask_files数组中查找
                if not mask:
                    mask_files = request.inputs.get("mask_files", [])
                    if mask_files:
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        for mask_f in mask_files:
                            mask_stem = Path(mask_f).stem.replace('_mask', '').replace('.nii', '').replace('.gz', '')
                            if mask_stem == input_stem or input_stem.startswith(mask_stem) or mask_stem in input_stem:
                                mask = mask_f
                                print(f"  [输入匹配] {input_basename}: 从数组中找到对应的mask文件")
                                break
                        if not mask and mask_files:
                            mask = mask_files[0]
                            print(f"  [输入备用] {input_basename}: 使用第一个可用的mask文件")

                # **关键修复2**: 如果没有找到bvecs/bvals，从bvec_files/bval_files数组中查找
                if not bvecs or not bvals:
                    bvec_files = request.inputs.get("bvec_files", [])
                    bval_files = request.inputs.get("bval_files", [])

                    # 根据文件名匹配对应的bvec/bval
                    if bvec_files and bval_files:
                        input_stem = Path(input_file).stem.replace('.nii', '').replace('.gz', '')
                        for bvec_f in bvec_files:
                            bvec_stem = Path(bvec_f).stem.replace('.bvec', '')
                            if bvec_stem == input_stem or input_stem.startswith(bvec_stem):
                                bvecs = bvec_f
                                break
                        for bval_f in bval_files:
                            bval_stem = Path(bval_f).stem.replace('.bval', '')
                            if bval_stem == input_stem or input_stem.startswith(bval_stem):
                                bvals = bval_f
                                break
                        if bvecs and bvals:
                            print(f"  [输入匹配] {input_basename}: 从数组中找到对应的bvec/bval文件")

                # 验证必需参数
                if not bvecs or not bvals:
                    print(f"  [错误] {input_basename}: dtifit缺少bvecs/bvals文件")
                    failed_files.append(input_file)
                    continue

                # 如果没有mask，自动生成（提取b0并运行bet）
                if not mask:
                    print(f"  [dtifit] {input_basename}: 未找到mask，自动生成...")

                    # 提取b0体积
                    b0_file = output_dir / f"{input_basename}_b0.nii.gz"
                    b0_brain = output_dir / f"{input_basename}_b0_brain"
                    wsl_b0 = windows_to_wsl_path(str(b0_file))
                    wsl_b0_brain = windows_to_wsl_path(str(b0_brain))

                    # fslroi提取第一个体积（b0）
                    fslroi_cmd = f"fslroi {wsl_input} {wsl_b0} 0 1"
                    print(f"    [FSL] 提取b0: fslroi")
                    roi_result = run_fsl(fslroi_cmd)

                    if roi_result and roi_result.get("status") == "succeeded" and b0_file.exists():
                        # bet生成mask
                        bet_cmd = f"bet {wsl_b0} {wsl_b0_brain} -m -f 0.3"
                        print(f"    [FSL] 生成mask: bet")
                        bet_result = run_fsl(bet_cmd)

                        if bet_result and bet_result.get("status") == "succeeded":
                            mask = str(b0_brain) + "_mask.nii.gz"
                            if Path(mask).exists():
                                print(f"    [OK] mask生成成功: {Path(mask).name}")
                            else:
                                print(f"  [错误] {input_basename}: bet完成但mask文件未生成")
                                failed_files.append(input_file)
                                continue
                        else:
                            print(f"  [错误] {input_basename}: bet失败")
                            failed_files.append(input_file)
                            continue
                    else:
                        print(f"  [错误] {input_basename}: fslroi提取b0失败")
                        failed_files.append(input_file)
                        continue

                wsl_bvecs = windows_to_wsl_path(bvecs)
                wsl_bvals = windows_to_wsl_path(bvals)
                wsl_mask = windows_to_wsl_path(mask)
                # 正确的FSL dtifit参数顺序: -k input -m mask -r bvecs -b bvals -o output
                fsl_cmd = f"dtifit -k {wsl_input} -m {wsl_mask} -r {wsl_bvecs} -b {wsl_bvals} -o {wsl_output}/{input_basename}_dti {options}"
            elif command == "fnirt":
                # 非线性配准到MNI空间
                # FNIRT需要先运行FLIRT进行线性配准
                ref = request.inputs.get("reference", "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz")
                if ref and not ref.startswith("$") and os.path.isabs(ref):
                    ref = windows_to_wsl_path(ref)

                # 输出文件路径
                affine_mat = f"{wsl_output}/{input_basename}_to_MNI_affine.mat"
                warp_coef = f"{wsl_output}/{input_basename}_to_MNI_warp"
                warped_out = f"{wsl_output}/{input_basename}_MNI"

                # 步骤1: FLIRT线性配准（获取初始仿射变换）
                flirt_cmd = f"flirt -in {wsl_input} -ref {ref} -omat {affine_mat} -out {warped_out}_flirt"
                print(f"    [FSL] 步骤1: flirt线性配准")
                flirt_result = run_fsl(flirt_cmd)

                if not flirt_result or flirt_result.get("status") != "succeeded":
                    print(f"  [错误] {input_basename}: flirt线性配准失败")
                    failed_files.append(input_file)
                    continue

                # 步骤2: FNIRT非线性配准
                fnirt_cmd = f"fnirt --in={wsl_input} --ref={ref} --aff={affine_mat} --cout={warp_coef} --iout={warped_out} {options}"
                print(f"    [FSL] 步骤2: fnirt非线性配准")
                fsl_cmd = fnirt_cmd  # 这将在下面执行
            elif command == "applywarp":
                # 应用变形场
                ref = request.inputs.get("reference", "$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz")
                warp = request.inputs.get("warp")
                if not warp:
                    print(f"  [错误] {input_basename}: applywarp需要warp参数")
                    failed_files.append(input_file)
                    continue
                if ref and not ref.startswith("$") and os.path.isabs(ref):
                    ref = windows_to_wsl_path(ref)
                wsl_warp = windows_to_wsl_path(warp)
                fsl_cmd = f"applywarp --in={wsl_input} --ref={ref} --warp={wsl_warp} --out={wsl_output}/{input_basename}_warped {options}"
            elif command == "fslroi":
                # 提取ROI/子卷
                # options应包含起始索引和大小，如 "0 1" 表示提取第一个体积
                fsl_cmd = f"fslroi {wsl_input} {wsl_output}/{input_basename}_roi {options}"
            elif command == "fslstats":
                # 统计信息（不输出文件，只打印结果）
                fsl_cmd = f"fslstats {wsl_input} {options}"
            elif command == "fslmaths":
                # 数学运算
                fsl_cmd = f"fslmaths {wsl_input} {options} {wsl_output}/{input_basename}_math"
            elif command == "fslmeants":
                # 提取ROI平均值 - 用于从atlas中提取指标
                # 常用于提取FA/MD等DTI指标在JHU atlas等ROI中的平均值
                mask = request.inputs.get("mask") or request.params.get("mask")
                label_file = request.inputs.get("label") or request.params.get("label")
                atlas_type = request.params.get("atlas", "jhu")  # 默认使用JHU atlas

                # 如果没有提供mask，使用默认的JHU atlas (适用于DTI分析)
                if not mask:
                    # 检测输入文件是否是DTI指标 (FA, MD等)
                    input_lower = input_basename.lower()
                    is_dti_metric = any(m in input_lower for m in ["_fa", "_md", "_rd", "_ad", "_l1", "_l2", "_l3"])

                    if is_dti_metric or atlas_type == "jhu":
                        # 使用FSL自带的JHU白质标签图谱
                        mask = "$FSLDIR/data/atlases/JHU/JHU-ICBM-labels-2mm.nii.gz"
                        print(f"    [FSL] 使用默认JHU白质atlas: {mask}")

                # 构建输出文件路径
                output_txt = f"{wsl_output}/{input_basename}_means.txt"

                # 构建命令
                fsl_cmd = f"fslmeants -i {wsl_input}"

                if mask:
                    # 如果是环境变量路径，不转换
                    if mask.startswith("$"):
                        wsl_mask = mask
                    elif os.path.isabs(mask):
                        wsl_mask = windows_to_wsl_path(mask)
                    else:
                        wsl_mask = mask
                    fsl_cmd += f" -m {wsl_mask}"

                if label_file:
                    if label_file.startswith("$"):
                        wsl_label = label_file
                    elif os.path.isabs(label_file):
                        wsl_label = windows_to_wsl_path(label_file)
                    else:
                        wsl_label = label_file
                    fsl_cmd += f" --label={wsl_label}"

                fsl_cmd += f" -o {output_txt}"

                if options:
                    fsl_cmd += f" {options}"

                print(f"    [FSL] fslmeants: 提取ROI平均值")
            elif command == "mcflirt":
                # fMRI头动校正 (Motion Correction using FLIRT)
                # 输出: <basename>_mcf.nii.gz (校正后图像), <basename>_mcf.par (运动参数)
                # 参考: https://fsl.fmrib.ox.ac.uk/fsl/docs/registration/mcflirt.html

                # 获取可选参数
                refvol = request.params.get("refvol")  # 参考volume索引
                use_meanvol = request.params.get("meanvol", True)  # 默认配准到均值
                cost_func = request.params.get("cost", "normcorr")  # 成本函数

                # 构建mcflirt命令
                output_base = f"{wsl_output}/{input_basename}_mcf"
                fsl_cmd = f"mcflirt -in {wsl_input} -out {output_base}"

                # 添加配准目标选项
                if refvol is not None:
                    fsl_cmd += f" -refvol {refvol}"
                elif use_meanvol:
                    fsl_cmd += " -meanvol"

                # 添加成本函数
                fsl_cmd += f" -cost {cost_func}"

                # 总是生成运动参数文件和变换矩阵
                fsl_cmd += " -plots -mats"

                # 添加用户自定义选项
                if options:
                    fsl_cmd += f" {options}"

                print(f"    [FSL] mcflirt: fMRI头动校正 (配准到{'均值' if use_meanvol else f'volume {refvol}'})")

            elif command == "slicetimer":
                # fMRI层时间校正 (Slice Timing Correction)
                # 输出: <basename>_st.nii.gz (校正后图像)
                # 参考: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT

                # 获取必需参数
                tr = request.params.get("tr")
                slice_direction = request.params.get("direction", 3)  # z=3是默认
                slice_order = request.params.get("slice_order", "ascending")  # 采集顺序

                # TR是必需参数
                if not tr:
                    print(f"  [错误] {input_basename}: slicetimer需要tr参数")
                    failed_files.append(input_file)
                    continue

                # 构建slicetimer命令
                output_file = f"{wsl_output}/{input_basename}_st.nii.gz"
                fsl_cmd = f"slicetimer -i {wsl_input} -o {output_file} -r {tr} -d {slice_direction}"

                # 根据采集顺序添加选项
                if slice_order == "descending" or slice_order == "down":
                    fsl_cmd += " --down"
                elif slice_order == "interleaved" or slice_order == "odd":
                    fsl_cmd += " --odd"
                # ascending是默认，不需要额外选项

                # 添加用户自定义选项
                if options:
                    fsl_cmd += f" {options}"

                print(f"    [FSL] slicetimer: 层时间校正 (TR={tr}s, 顺序={slice_order})")

            else:
                # 不支持的命令 - 返回错误而不是直接执行
                # 使用统一定义的命令列表
                error_msg = f"不支持的FSL命令: {command}。支持的命令: {', '.join(FSL_SUPPORTED_COMMANDS)}"
                print(f"  [ERROR] {error_msg}")

                # 检查是否是eddy_quad - 该工具未安装在 FSL 6.0.5.1 中
                if command == "eddy_quad":
                    error_msg = (
                        f"eddy_quad 工具未安装在 FSL 6.0.5.1 中。\n"
                        f"请使用以下替代方法获取运动参数：\n"
                        f"1. 直接读取 eddy 输出的 .eddy_movement_rms 文件\n"
                        f"2. 使用 python_stats 工具解析 eddy 输出文件\n"
                        f"3. 参考文档: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/eddy/UsersGuide"
                    )
                # 检查是否是eddy_qc - 该命令需要单独安装，不在标准FSL中
                elif command == "eddy_qc":
                    error_msg = f"eddy_qc命令不可用。该命令需要单独安装FSL eddy_qc工具包。建议跳过此QC步骤或手动检查eddy输出。"

                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error=error_msg,
                    outputs={"unsupported_command": command}
                )

            print(f"  [FSL] 执行: {command} on {Path(input_file).name}")
            result = run_fsl(fsl_cmd)
            all_commands.append(fsl_cmd)

            if result and result.get("status") == "succeeded":
                print(f"  [OK] {Path(input_file).name}")
            else:
                failed_files.append(input_file)
                # 同时检查 error 和 stderr 字段（run_wsl_command 返回的是 stderr）
                error_msg = result.get('error') or result.get('stderr') or f"returncode={result.get('returncode')}" if result else "执行未返回结果"
                print(f"  [FAILED] {Path(input_file).name}: {error_msg[:200]}")

            # **关键修复**: eddy处理完后立即释放内存，避免累积导致系统崩溃
            if command == "eddy":
                print(f"  [FSL] eddy处理完成，释放内存...")
                import gc
                import time
                gc.collect()  # 强制垃圾回收
                time.sleep(10)  # 等待10秒让系统释放资源
                # 每个eddy之间额外等待，让内存完全释放
                if file_idx < len(input_files) - 1:  # 不是最后一个文件
                    print(f"  [FSL] 等待内存释放后继续下一个被试...")
                    time.sleep(20)  # 额外等待30秒

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 收集所有输出文件
        all_output_files = list(output_dir.glob("*.nii*"))
        # 也收集fslmeants等命令生成的文本文件
        all_output_files.extend(list(output_dir.glob("*_means.txt")))
        all_output_files.extend(list(output_dir.glob("*.csv")))

        # 判断整体成功/失败
        if len(failed_files) < len(input_files):
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "output_files": [str(f) for f in all_output_files],
                    "modality": modality,
                    "command": command,  # 用于识别具体命令类型（如dtifit）
                    "commands": all_commands,
                    "processed_count": len(input_files) - len(failed_files),
                    "failed_count": len(failed_files)
                }
            )
        else:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                outputs={"modality": modality, "command": command},
                error=f"所有 {len(input_files)} 个文件处理失败"
            )


# ============== FSL eddy QC 辅助函数 ==============

def read_eddy_motion_parameters(eddy_basename: str) -> Dict[str, Any]:
    """
    读取 FSL eddy 输出的运动参数文件

    eddy 自动生成以下 QC 文件:
    - <basename>.eddy_movement_rms: 每个体积的总运动量 (RMS)
    - <basename>.eddy_parameters: 完整的场和运动参数矩阵

    Args:
        eddy_basename: eddy 输出文件的 basename (不含 .nii.gz 扩展名)
                      例如: "outputs/HC1_0001_eddy" 或 "I:/AGENT/outputs/runs/xxx/HC1_0001_eddy"

    Returns:
        {
            "movement_rms": [0.123, 0.456, ...],  # 每个体积的运动 RMS
            "mean_movement": 0.234,               # 平均运动量
            "max_movement": 0.567,                # 最大运动量
            "volumes_with_high_motion": [5, 12],  # 运动量 > 阈值的体积索引
            "num_volumes": 100,                   # 总体积数
            "qc_files": {
                "movement_rms": "path/to/file.eddy_movement_rms",
                "parameters": "path/to/file.eddy_parameters"
            },
            "qc_summary": "平均运动: 0.234mm, 最大: 0.567mm, 高运动体积数: 3"
        }

    Raises:
        FileNotFoundError: 如果 .eddy_movement_rms 文件不存在

    Example:
        >>> qc_data = read_eddy_motion_parameters("outputs/HC1_0001_eddy")
        >>> print(qc_data["qc_summary"])
        平均运动: 0.234mm, 最大: 0.567mm, 高运动体积数: 3
    """
    import numpy as np
    from pathlib import Path

    # 读取 .eddy_movement_rms (单列文本文件)
    rms_file = f"{eddy_basename}.eddy_movement_rms"
    params_file = f"{eddy_basename}.eddy_parameters"

    if not Path(rms_file).exists():
        raise FileNotFoundError(
            f"eddy 运动参数文件不存在: {rms_file}\n"
            f"请确保 eddy 命令已成功执行。\n"
            f"eddy 输出文件应包含: {eddy_basename}.eddy_movement_rms"
        )

    try:
        # 读取运动 RMS 数据
        movement_rms = np.loadtxt(rms_file)

        # 计算统计量
        mean_movement = float(np.mean(movement_rms))
        max_movement = float(np.max(movement_rms))
        min_movement = float(np.min(movement_rms))

        # 识别高运动体积 (阈值: 1mm RMS)
        threshold = 1.0
        high_motion_volumes = np.where(movement_rms > threshold)[0].tolist()

        # 构建返回结果
        result = {
            "movement_rms": movement_rms.tolist(),
            "mean_movement": mean_movement,
            "max_movement": max_movement,
            "min_movement": min_movement,
            "volumes_with_high_motion": high_motion_volumes,
            "num_volumes": len(movement_rms),
            "qc_files": {
                "movement_rms": rms_file,
                "parameters": params_file if Path(params_file).exists() else None
            },
            "qc_summary": f"平均运动: {mean_movement:.3f}mm, 最大: {max_movement:.3f}mm, 高运动体积数: {len(high_motion_volumes)}"
        }

        # 如果 parameters 文件存在，也读取（可选）
        if Path(params_file).exists():
            try:
                parameters = np.loadtxt(params_file)
                result["parameters_shape"] = parameters.shape
                result["has_parameters"] = True
            except Exception as e:
                print(f"  [警告] 无法读取 .eddy_parameters 文件: {e}")
                result["has_parameters"] = False
        else:
            result["has_parameters"] = False

        return result

    except Exception as e:
        raise RuntimeError(
            f"读取 eddy 运动参数失败: {str(e)}\n"
            f"文件: {rms_file}"
        )


class LocalPythonStatsTool(BaseTool):
    """本地Python统计工具 - 直接执行统计分析"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_stats",
            description="使用Python进行统计分析，支持t检验、ANOVA、相关分析等",
            category="statistics",
            supported_modalities=[Modality.ALL],
            executor_type=ExecutorType.PYTHON,
            input_schema={
                "type": "object",
                "properties": {
                    "data_file": {
                        "type": "string",
                        "description": "数据文件路径(CSV/Parquet/NIfTI)"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["ttest", "anova", "correlation", "regression", "mann_whitney", "wilcoxon", "read_eddy_qc"],
                        "description": "分析类型: ttest(t检验), anova(方差分析), correlation(相关分析), regression(回归), mann_whitney(Mann-Whitney U检验), wilcoxon(Wilcoxon检验), read_eddy_qc(读取eddy QC指标)"
                    },
                    "group_var": {
                        "type": "string",
                        "description": "分组变量"
                    },
                    "dependent_var": {
                        "type": "string",
                        "description": "因变量"
                    },
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "协变量列表"
                    }
                },
                "required": ["analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "statistics": {"type": "object"},
                    "effect_size": {"type": "number"},
                    "p_value": {"type": "number"}
                }
            },
            version="1.0.0",
            dependencies=["numpy", "scipy", "pandas", "statsmodels"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行统计分析"""
        import numpy as np
        from scipy import stats as sp_stats

        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_type = request.params.get("analysis_type", "ttest")

        try:
            # ========== 新增：从影像文件中提取脑指标 ==========
            processed_files = request.inputs.get("processed_files", [])
            cohort = request.inputs.get("cohort", {})

            # 如果有processed_files和cohort，从影像中提取脑指标
            if processed_files and cohort and not request.inputs.get("group1_data"):
                print(f"  [指标提取] 从 {len(processed_files)} 个影像文件中提取脑指标...")
                group1_data, group2_data = self._extract_brain_metrics_from_images(
                    processed_files, cohort
                )
                print(f"  [指标提取] 组1: {len(group1_data)} 个被试, 组2: {len(group2_data)} 个被试")
            else:
                # 使用原有的数据获取方式
                data_file = request.inputs.get("data_file")
                group1_data = request.inputs.get("group1_data")
                group2_data = request.inputs.get("group2_data")

            results = {}

            if analysis_type == "ttest":
                if not group1_data or not group2_data:
                    raise ValueError("t检验需要提供group1_data和group2_data两组数据")

                g1 = np.array(group1_data)
                g2 = np.array(group2_data)

                t_stat, p_value = sp_stats.ttest_ind(g1, g2)

                # Cohen's d
                pooled_std = np.sqrt((np.std(g1)**2 + np.std(g2)**2) / 2)
                cohens_d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

                results = {
                    "test": "independent_samples_t_test",
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "cohens_d": float(cohens_d),
                    "group1_mean": float(np.mean(g1)),
                    "group1_std": float(np.std(g1)),
                    "group1_n": int(len(g1)),
                    "group2_mean": float(np.mean(g2)),
                    "group2_std": float(np.std(g2)),
                    "group2_n": int(len(g2)),
                    "significant": bool(p_value < 0.05)
                }

            elif analysis_type == "mann_whitney":
                if not group1_data or not group2_data:
                    raise ValueError("Mann-Whitney检验需要提供group1_data和group2_data两组数据")

                g1 = np.array(group1_data)
                g2 = np.array(group2_data)

                u_stat, p_value = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')

                # 效应量 r = Z / sqrt(N)
                n = len(g1) + len(g2)
                z = sp_stats.norm.ppf(1 - p_value/2)
                effect_r = z / np.sqrt(n)

                results = {
                    "test": "mann_whitney_u",
                    "u_statistic": float(u_stat),
                    "p_value": float(p_value),
                    "effect_r": float(effect_r),
                    "group1_median": float(np.median(g1)),
                    "group2_median": float(np.median(g2)),
                    "significant": bool(p_value < 0.05)
                }

            elif analysis_type == "correlation":
                var1 = request.inputs.get("var1_data")
                var2 = request.inputs.get("var2_data")

                if not var1 or not var2:
                    raise ValueError("相关分析需要提供var1_data和var2_data两组数据")

                var1 = np.array(var1)
                var2 = np.array(var2)

                r, p_value = sp_stats.pearsonr(var1, var2)
                rho, p_spearman = sp_stats.spearmanr(var1, var2)

                results = {
                    "test": "correlation",
                    "pearson_r": float(r),
                    "pearson_p": float(p_value),
                    "spearman_rho": float(rho),
                    "spearman_p": float(p_spearman),
                    "n": int(len(var1))
                }

            elif analysis_type == "anova":
                groups = request.inputs.get("groups_data")

                if not groups or len(groups) < 2:
                    raise ValueError("ANOVA分析需要提供groups_data（至少包含2组数据的列表）")

                f_stat, p_value = sp_stats.f_oneway(*groups)

                # 计算eta squared
                all_data = np.concatenate(groups)
                grand_mean = np.mean(all_data)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
                ss_total = np.sum((all_data - grand_mean)**2)
                eta_squared = ss_between / ss_total if ss_total > 0 else 0

                results = {
                    "test": "one_way_anova",
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "eta_squared": float(eta_squared),
                    "num_groups": int(len(groups)),
                    "significant": bool(p_value < 0.05)
                }

            elif analysis_type == "read_eddy_qc":
                # 读取 eddy QC 指标 - 调用已有的 read_eddy_motion_parameters 函数
                # 这是为了让 LLM 能够通过 python_stats 工具提取 eddy QC 数据
                input_pattern = request.params.get("input_pattern", "*_eddy.eddy_movement_rms")

                # 从之前的工具输出中查找 eddy 输出目录
                eddy_dir = None
                input_files = request.inputs.get("input_files", [])
                if input_files:
                    # 如果提供了输入文件，使用其父目录
                    eddy_dir = Path(input_files[0]).parent
                else:
                    # 否则在 run 目录中查找 fsl_analysis 输出
                    run_dir = output_dir.parent
                    for subdir in run_dir.glob("task_*_fsl_analysis_*"):
                        if subdir.is_dir():
                            eddy_dir = subdir
                            break

                if not eddy_dir or not eddy_dir.exists():
                    raise ValueError(f"未找到 eddy 输出目录。请确保 eddy 命令已成功执行。")

                print(f"  [eddy QC] 在目录中查找: {eddy_dir}")

                # 查找所有 .eddy_movement_rms 文件
                rms_files = list(eddy_dir.glob("*.eddy_movement_rms"))
                print(f"  [eddy QC] 找到 {len(rms_files)} 个 movement_rms 文件")

                all_qc_data = []
                for rms_file in rms_files:
                    # 从 .eddy_movement_rms 文件名提取 basename
                    basename = str(rms_file).replace('.eddy_movement_rms', '')
                    try:
                        qc_data = read_eddy_motion_parameters(basename)
                        # 提取被试ID
                        subject_id = rms_file.stem.replace('_eddy.eddy_movement_rms', '').replace('_eddy', '')
                        qc_data['subject_id'] = subject_id
                        all_qc_data.append(qc_data)
                    except FileNotFoundError as e:
                        print(f"    [警告] 无法读取 {rms_file.name}: {e}")
                        continue
                    except Exception as e:
                        print(f"    [警告] 处理 {rms_file.name} 时出错: {e}")
                        continue

                if not all_qc_data:
                    raise ValueError("未能读取任何 eddy QC 数据")

                # 计算汇总统计
                mean_motions = [d.get('mean_movement', 0) for d in all_qc_data]
                max_motions = [d.get('max_movement', 0) for d in all_qc_data]

                results = {
                    "test": "eddy_qc_summary",
                    "n_subjects": len(all_qc_data),
                    "mean_motion_avg": float(np.mean(mean_motions)),
                    "mean_motion_std": float(np.std(mean_motions)),
                    "max_motion_avg": float(np.mean(max_motions)),
                    "max_motion_max": float(np.max(max_motions)),
                    "subjects_with_high_motion": sum(1 for m in max_motions if m > 2.0),  # >2mm 视为高运动
                    "qc_data": all_qc_data,
                    "qc_summary": f"共 {len(all_qc_data)} 个被试，平均运动: {np.mean(mean_motions):.3f}mm, 最大运动: {np.max(max_motions):.3f}mm"
                }

                print(f"  [eddy QC] {results['qc_summary']}")

            else:
                results = {"error": f"Unknown analysis type: {analysis_type}"}

            # 保存结果
            result_path = output_dir / f"stats_result_{analysis_type}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs=results,
                artifacts=[
                    {"name": f"stats_result_{analysis_type}.json", "path": str(result_path)}
                ]
            )

        except Exception as e:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=str(e)
            )

    def _extract_brain_metrics_from_images(self, image_files: List[str], cohort: Dict) -> Tuple[List[float], List[float]]:
        """
        从脑影像文件中提取指标（支持灰质VBM和DTI指标FA/MD）

        Args:
            image_files: 处理后的影像文件列表
            cohort: 队列信息，包含分组

        Returns:
            (group1_data, group2_data): 两组的脑指标数据
        """
        import nibabel as nib
        import numpy as np
        from pathlib import Path

        # 获取分组信息
        groups = cohort.get("groups", {})
        group_names = list(groups.keys())

        if len(group_names) < 2:
            raise ValueError(f"需要至少2个组进行比较，当前只有 {len(group_names)} 个组")

        group1_name = group_names[0]
        group2_name = group_names[1]

        group1_subjects = groups[group1_name].get("subjects", [])
        group2_subjects = groups[group2_name].get("subjects", [])

        print(f"  [分组] 组1({group1_name}): {len(group1_subjects)} 个被试")
        print(f"  [分组] 组2({group2_name}): {len(group2_subjects)} 个被试")
        print(f"  [示例] 组1前3个: {group1_subjects[:3] if len(group1_subjects) >= 3 else group1_subjects}")
        print(f"  [示例] 组2前3个: {group2_subjects[:3] if len(group2_subjects) >= 3 else group2_subjects}")

        # 检测文件类型：灰质VBM还是DTI指标
        dti_metrics = ['_FA', '_MD', '_RD', '_AD', '_L1', '_L2', '_L3', '_fa', '_md', '_rd', '_ad']
        gm_prefixes = ['swc1', 'wc1', 'c1', 'smwc1', 'mwc1']

        dti_files = []
        gm_files = []

        for f in image_files:
            fname = Path(f).name
            fname_lower = fname.lower()

            # 检测DTI指标文件
            if any(metric.lower() in fname_lower for metric in dti_metrics):
                # 优先使用FA文件进行群组比较
                if '_fa' in fname_lower:
                    dti_files.insert(0, f)  # FA放在前面
                else:
                    dti_files.append(f)
            # 检测灰质VBM文件
            elif any(fname.startswith(prefix) for prefix in gm_prefixes):
                if not any(fname.startswith(x) for x in ['c2', 'c3', 'c4', 'c5', 'c6', 'wc2', 'wc3', 'swc2', 'swc3']):
                    gm_files.append(f)

        # 决定使用哪种文件类型
        if dti_files:
            # 使用DTI指标文件（优先使用FA）
            target_files = dti_files
            file_type = "DTI"
            # 只保留同一种指标类型的文件（如只用FA）
            fa_files = [f for f in dti_files if '_fa' in Path(f).name.lower()]
            if fa_files:
                target_files = fa_files
                print(f"  [文件筛选] 找到 {len(target_files)} 个FA文件用于群组比较")
            else:
                print(f"  [文件筛选] 找到 {len(target_files)} 个DTI指标文件")
        elif gm_files:
            target_files = gm_files
            file_type = "VBM"
            print(f"  [文件筛选] 找到 {len(target_files)} 个灰质概率图文件")
        else:
            raise ValueError("未找到可用于群组比较的文件（需要FA/MD或灰质概率图文件）")

        # 提取每个被试的指标
        group1_values = []
        group2_values = []

        for target_file in target_files:
            # 从文件名中提取被试ID
            fname = Path(target_file).stem
            original_fname = fname

            # 根据文件类型移除不同的前缀/后缀
            subject_id = fname
            if file_type == "VBM":
                # 移除VBM前缀
                changed = True
                while changed:
                    changed = False
                    for prefix in ['smwc1', 'mwc1', 'swc1', 'wc1', 'sc1', 'c1', 'smw', 'mw', 'sw', 's', 'w', 'm']:
                        if subject_id.startswith(prefix):
                            subject_id = subject_id[len(prefix):]
                            changed = True
                            break
            else:
                # 移除DTI后缀 (如 _dti_FA, _FA, _MD 等)
                for suffix in ['_dti_FA', '_dti_MD', '_dti_RD', '_dti_AD', '_FA', '_MD', '_RD', '_AD', '_L1', '_L2', '_L3']:
                    if subject_id.endswith(suffix):
                        subject_id = subject_id[:-len(suffix)]
                        break
                # 还要移除可能的前缀
                for prefix in ['dti_', 'DTI_']:
                    if subject_id.startswith(prefix):
                        subject_id = subject_id[len(prefix):]

            # 加载NIfTI文件并计算指标
            try:
                img = nib.load(target_file)
                data = img.get_fdata()

                if file_type == "VBM":
                    # VBM: 计算总灰质体积
                    voxel_volume = np.prod(img.header.get_zooms()[:3])  # mm^3
                    metric_value = np.sum(data) * voxel_volume / 1000.0  # mL
                    unit = "mL"
                else:
                    # DTI: 计算脑区平均FA/MD值（根据指标类型使用不同阈值）
                    fname_lower = original_fname.lower()

                    # 根据文件名判断指标类型，设置合适的阈值和单位
                    if '_fa' in fname_lower or fname_lower.endswith('fa'):
                        # FA (Fractional Anisotropy): 范围 0-1
                        valid_mask = (data > 0.05) & (data < 1.0)
                        unit = ""
                    elif any(m in fname_lower for m in ['_md', '_rd', '_ad']):
                        # MD/RD/AD (Diffusivity): 范围 0-0.005 mm²/s
                        valid_mask = (data > 0.0001) & (data < 0.005)
                        unit = "×10⁻³ mm²/s"
                    elif any(m in fname_lower for m in ['_l1', '_l2', '_l3']):
                        # L1/L2/L3 (Eigenvalues): 范围类似MD
                        valid_mask = (data > 0.0001) & (data < 0.005)
                        unit = "×10⁻³ mm²/s"
                    else:
                        # 默认：非零值
                        valid_mask = data > 0
                        unit = ""

                    if np.sum(valid_mask) > 0:
                        metric_value = float(np.mean(data[valid_mask]))
                    else:
                        # 如果掩码全为空，使用所有非零值
                        metric_value = float(np.mean(data[data > 0])) if np.any(data > 0) else 0.0

                # 尝试匹配到对应组
                matched = False

                # 精确匹配
                if subject_id in group1_subjects:
                    group1_values.append(metric_value)
                    print(f"    [{group1_name}] {original_fname} -> {subject_id}: {metric_value:.4f} {unit} (精确)")
                    matched = True
                elif subject_id in group2_subjects:
                    group2_values.append(metric_value)
                    print(f"    [{group2_name}] {original_fname} -> {subject_id}: {metric_value:.4f} {unit} (精确)")
                    matched = True

                # 模糊匹配：允许部分匹配
                if not matched:
                    for s in group1_subjects:
                        if s in subject_id or subject_id in s:
                            group1_values.append(metric_value)
                            print(f"    [{group1_name}] {original_fname} -> {subject_id}: {metric_value:.4f} {unit} (模糊: {s})")
                            matched = True
                            break

                if not matched:
                    for s in group2_subjects:
                        if s in subject_id or subject_id in s:
                            group2_values.append(metric_value)
                            print(f"    [{group2_name}] {original_fname} -> {subject_id}: {metric_value:.4f} {unit} (模糊: {s})")
                            matched = True
                            break

                if not matched:
                    print(f"    [警告] {original_fname} -> {subject_id}: 无法匹配到分组")

            except Exception as e:
                print(f"    [错误] 处理 {original_fname} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        if not group1_values or not group2_values:
            raise ValueError(f"无法提取脑指标: 组1有{len(group1_values)}个数据，组2有{len(group2_values)}个数据")

        return group1_values, group2_values


class LocalDataVisualizationTool(BaseTool):
    """本地数据可视化工具 - 使用高端LLM动态生成画图代码"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="data_visualization",
            description="使用高端LLM动态生成Python画图代码，支持各类神经影像数据可视化",
            category="visualization",
            supported_modalities=[Modality.ALL],
            executor_type=ExecutorType.PYTHON,
            input_schema={
                "type": "object",
                "properties": {
                    "data_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "数据文件路径列表"
                    },
                    "visualization_type": {
                        "type": "string",
                        "enum": ["brain_slices", "statistical_maps", "comparison_plot", "custom"],
                        "description": "可视化类型"
                    },
                    "description": {
                        "type": "string",
                        "description": "可视化需求描述（自然语言）"
                    },
                    "figure_title": {
                        "type": "string",
                        "description": "图表标题"
                    }
                },
                "required": ["data_files", "visualization_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "figure_path": {"type": "string"},
                    "code": {"type": "string"}
                }
            },
            version="1.0.0",
            dependencies=["matplotlib", "seaborn", "nibabel", "nilearn"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行数据可视化 - 使用高端LLM生成代码"""
        from src.utils.llm import get_llm_client

        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            data_files = request.inputs.get("data_files", [])
            viz_type = request.params.get("visualization_type", "custom")
            description = request.params.get("description", "")
            title = request.params.get("figure_title", "Neuroimaging Data Visualization")

            # ========== 使用高端LLM生成画图代码 ==========
            print(f"  [高端LLM] 切换到代码生成模型...")
            llm = get_llm_client()
            llm.set_task_type("data_visualization")  # 切换到高端模型

            # 构建提示词
            prompt = f"""你是神经影像分析专家。请生成Python代码来可视化以下数据。

数据文件:
{chr(10).join(f'- {Path(f).name}: {f}' for f in data_files[:5])}

可视化类型: {viz_type}
需求描述: {description if description else '默认可视化'}
图表标题: {title}

要求:
1. 使用matplotlib、seaborn、nibabel、nilearn等库
2. 生成高质量的学术级别图表
3. 保存图表到: {output_dir / "visualization.png"}
4. 代码必须完整可执行，包含所有import
5. 处理异常情况
6. 对于NIfTI影像文件，使用nilearn.plotting进行可视化
7. 对于统计结果，使用适当的图表类型（柱状图、箱线图等）

请只返回Python代码，不要有额外说明。代码应该直接可以运行。"""

            messages = [
                {"role": "system", "content": "你是Python数据可视化和神经影像分析专家。"},
                {"role": "user", "content": prompt}
            ]

            # 调用高端LLM生成代码
            response = llm.chat(messages, temperature=0.3, max_tokens=2048)
            generated_code = response["choices"][0]["message"]["content"]

            # 提取代码块
            import re
            code_match = re.search(r'```(?:python)?\s*([\s\S]*?)```', generated_code)
            if code_match:
                code = code_match.group(1).strip()
            else:
                code = generated_code.strip()

            # 重置LLM模型
            llm.reset_model()
            print(f"  [高端LLM] 代码生成完成，重置为基础模型")

            # 保存生成的代码
            code_path = output_dir / "visualization_code.py"
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)

            # 执行生成的代码
            print(f"  执行生成的可视化代码...")
            exec_globals = {"__file__": str(code_path)}
            exec(code, exec_globals)

            # 查找生成的图表文件
            figure_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=duration,
                outputs={
                    "figure_paths": [str(f) for f in figure_files],
                    "code_path": str(code_path),
                    "llm_model_used": "advanced"
                },
                artifacts=[
                    {"name": "visualization_code.py", "path": str(code_path)},
                    *[{"name": f.name, "path": str(f)} for f in figure_files]
                ]
            )

        except Exception as e:
            import traceback
            error_detail = f"{str(e)}\n{traceback.format_exc()}"

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=error_detail
            )


class LocalDICOMConverterTool(BaseTool):
    """DICOM转NIfTI工具 - 支持SPM和dcm2niix两种转换方式

    对于DWI数据，使用dcm2niix以自动生成bvec/bval文件
    对于anat/func数据，优先使用dcm2niix，失败时回退到SPM
    """

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dicom_to_nifti",
            description="""将DICOM文件夹转换为NIfTI格式。
            - DWI数据：使用dcm2niix（自动生成bvec/bval文件）
            - anat/func数据：优先dcm2niix，失败时回退到SPM
            输出文件包括：*.nii.gz, *.bvec, *.bval, *.json""",
            category="preprocessing",
            supported_modalities=[Modality.ANAT, Modality.FUNC, Modality.DWI],
            executor_type=ExecutorType.CLI,
            input_schema={
                "type": "object",
                "properties": {
                    "input_dirs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "DICOM文件夹路径列表"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["anat", "func", "dwi", "auto"],
                        "default": "auto",
                        "description": "数据模态类型。dwi模态会生成bvec/bval文件"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["nii", "nii.gz"],
                        "default": "nii.gz",
                        "description": "输出格式（推荐nii.gz节省空间）"
                    },
                    "converter": {
                        "type": "string",
                        "enum": ["dcm2niix", "spm", "auto"],
                        "default": "auto",
                        "description": "转换器选择。auto时DWI使用dcm2niix，其他优先dcm2niix"
                    }
                },
                "required": ["input_dirs"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "bvec_files": {"type": "array"},
                    "bval_files": {"type": "array"},
                    "json_files": {"type": "array"},
                    "conversion_log": {"type": "string"}
                }
            },
            version="2.0.0",
            dependencies=["dcm2niix", "MATLAB R2019b (可选)", "SPM25 (可选)"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DICOM转换 - 支持dcm2niix和SPM两种转换方式"""
        import shutil
        start_time = datetime.now()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_dirs = request.inputs.get("input_dirs", [])
        modality = request.params.get("modality", "auto")
        output_format = request.params.get("output_format", "nii.gz")
        converter = request.params.get("converter", "auto")

        output_files = []
        bvec_files = []
        bval_files = []
        json_files = []
        conversion_log = []

        # 检查dcm2niix可用性
        dcm2niix_available = DCM2NIIX_PATH and Path(DCM2NIIX_PATH).exists()

        for idx, dicom_dir in enumerate(input_dirs):
            dicom_path = Path(dicom_dir)

            if not dicom_path.exists():
                conversion_log.append(f"[跳过] 目录不存在: {dicom_dir}")
                continue

            # 检查是否是DICOM目录
            dcm_files = list(dicom_path.glob("*.dcm"))
            if not dcm_files:
                # 没有.dcm后缀，尝试检测无后缀的DICOM文件
                # 使用pydicom验证文件是否真的是DICOM格式
                all_files = [f for f in dicom_path.iterdir() if f.is_file()]
                if all_files:
                    import pydicom
                    for f in all_files:
                        try:
                            # 只读取DICOM头部，不加载像素数据
                            pydicom.dcmread(f, stop_before_pixels=True)
                            dcm_files.append(f)
                        except Exception:
                            # 不是DICOM文件，跳过
                            continue

            if not dcm_files:
                conversion_log.append(f"[跳过] 未找到DICOM文件: {dicom_dir}")
                continue

            # 提取被试ID
            subject_id = dicom_path.name
            subject_output_dir = output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)

            conversion_log.append(f"[转换] {subject_id}: {len(dcm_files)} 个DICOM文件 (模态: {modality})")

            # 确定使用哪个转换器
            use_dcm2niix = False
            if converter == "dcm2niix":
                use_dcm2niix = dcm2niix_available
            elif converter == "spm":
                use_dcm2niix = False
            else:  # auto
                # DWI必须用dcm2niix（生成bvec/bval）
                if modality == "dwi":
                    if not dcm2niix_available:
                        # DWI转换必须使用dcm2niix，否则无法生成bvec/bval文件
                        error_msg = f"DWI conversion requires dcm2niix (not available). SPM cannot generate bvec/bval files required for DTI analysis."
                        conversion_log.append(f"[错误] {subject_id}: {error_msg}")
                        return ToolResult(
                            status="failed",
                            outputs={},
                            error=error_msg,
                            logs="\n".join(conversion_log)
                        )
                    use_dcm2niix = True
                else:
                    # 其他模态优先dcm2niix
                    use_dcm2niix = dcm2niix_available

            # 执行转换
            if use_dcm2niix:
                result = self._convert_with_dcm2niix(
                    dicom_path, subject_output_dir, subject_id,
                    compress=(output_format == "nii.gz")
                )
            else:
                result = self._convert_with_spm(
                    dicom_path, subject_output_dir, subject_id
                )

            if result.get("status") == "succeeded":
                # 收集输出文件
                subject_nii = result.get("nii_files", [])
                subject_bvec = result.get("bvec_files", [])
                subject_bval = result.get("bval_files", [])
                subject_json = result.get("json_files", [])

                output_files.extend(subject_nii)
                bvec_files.extend(subject_bvec)
                bval_files.extend(subject_bval)
                json_files.extend(subject_json)

                conversion_log.append(
                    f"[成功] {subject_id}: {len(subject_nii)} NIfTI"
                    + (f", {len(subject_bvec)} bvec" if subject_bvec else "")
                    + (f", {len(subject_bval)} bval" if subject_bval else "")
                )
            else:
                error_msg = result.get("error", "Unknown error")
                conversion_log.append(f"[失败] {subject_id}: {error_msg[:200]}")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 保存转换日志
        log_path = output_dir / "conversion_log.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(conversion_log))

        status = "succeeded" if output_files else "failed"
        error = None if output_files else "未能转换任何DICOM文件"

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status=status,
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=duration,
            outputs={
                "output_files": output_files,
                "bvec_files": bvec_files,
                "bval_files": bval_files,
                "json_files": json_files,
                "conversion_count": len(output_files),
                "conversion_log": conversion_log
            },
            artifacts=[
                {"name": "conversion_log.txt", "path": str(log_path)}
            ],
            error=error
        )

    def _convert_with_dcm2niix(self, dicom_path: Path, output_dir: Path,
                                subject_id: str, compress: bool = True) -> dict:
        """使用dcm2niix转换DICOM"""
        result = run_dcm2niix(str(dicom_path), str(output_dir), compress=compress)

        if result.get("status") != "succeeded":
            return result

        # 获取输出文件
        files = result.get("files", {})
        nii_files = files.get("nii", [])
        bvec_files = files.get("bvec", [])
        bval_files = files.get("bval", [])
        json_files = files.get("json", [])

        # 重命名文件以匹配被试ID
        renamed_nii = []
        renamed_bvec = []
        renamed_bval = []
        renamed_json = []

        for i, nii_file in enumerate(nii_files):
            nii_path = Path(nii_file)
            base_name = nii_path.stem.replace('.nii', '')  # 处理.nii.gz的情况

            # 生成新的基础名称
            if len(nii_files) == 1:
                new_base = subject_id
            else:
                new_base = f"{subject_id}_{i+1:02d}"

            # 确定扩展名
            if nii_path.suffix == '.gz':
                ext = '.nii.gz'
            else:
                ext = '.nii'

            new_nii_path = output_dir / f"{new_base}{ext}"
            self._rename_with_sidecar(nii_path, new_nii_path, base_name, new_base, output_dir)
            renamed_nii.append(str(new_nii_path))

            # 同步重命名伴随文件
            for old_bvec in bvec_files:
                if base_name in Path(old_bvec).stem:
                    new_bvec = output_dir / f"{new_base}.bvec"
                    if Path(old_bvec).exists() and not new_bvec.exists():
                        Path(old_bvec).rename(new_bvec)
                        renamed_bvec.append(str(new_bvec))
                        bvec_files.remove(old_bvec)
                    break

            for old_bval in bval_files:
                if base_name in Path(old_bval).stem:
                    new_bval = output_dir / f"{new_base}.bval"
                    if Path(old_bval).exists() and not new_bval.exists():
                        Path(old_bval).rename(new_bval)
                        renamed_bval.append(str(new_bval))
                        bval_files.remove(old_bval)
                    break

            for old_json in json_files:
                if base_name in Path(old_json).stem:
                    new_json = output_dir / f"{new_base}.json"
                    if Path(old_json).exists() and not new_json.exists():
                        Path(old_json).rename(new_json)
                        renamed_json.append(str(new_json))
                        json_files.remove(old_json)
                    break

        return {
            "status": "succeeded",
            "nii_files": renamed_nii,
            "bvec_files": renamed_bvec,
            "bval_files": renamed_bval,
            "json_files": renamed_json
        }

    def _rename_with_sidecar(self, old_path: Path, new_path: Path,
                             old_base: str, new_base: str, output_dir: Path):
        """重命名NIfTI文件"""
        if old_path.exists() and old_path != new_path:
            old_path.rename(new_path)

    def _convert_with_spm(self, dicom_path: Path, output_dir: Path,
                          subject_id: str) -> dict:
        """使用SPM转换DICOM（回退方案）"""
        matlab_dicom_dir = str(dicom_path).replace("\\", "/")
        matlab_output_dir = str(output_dir).replace("\\", "/")

        script = f"""% SPM DICOM Import Script
% Generated by Research Agent

spm('defaults', 'fmri');
spm_jobman('initcfg');

% 获取DICOM文件列表
dicom_dir = '{matlab_dicom_dir}';
output_dir = '{matlab_output_dir}';

% 查找DICOM文件
dcm_files = spm_select('FPList', dicom_dir, '.*');
if isempty(dcm_files)
    dcm_files = spm_select('FPListRec', dicom_dir, '.*');
end

if isempty(dcm_files)
    error('No DICOM files found in: %s', dicom_dir);
end

% 读取DICOM头信息
fprintf('Reading %d DICOM files...\\n', size(dcm_files, 1));
hdrs = spm_dicom_headers(dcm_files);

% 转换为NIfTI
fprintf('Converting to NIfTI...\\n');
out = spm_dicom_convert(hdrs, 'all', 'flat', 'nii', output_dir);

fprintf('Conversion completed. Output directory: %s\\n', output_dir);
"""

        script_path = output_dir / "dicom_convert.m"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        result = run_matlab_script(script, str(output_dir))

        if result and result.get("status") == "succeeded":
            # 收集并重命名输出文件
            nii_files = list(output_dir.glob("*.nii")) + list(output_dir.glob("*.nii.gz"))

            renamed_files = []
            for idx, nii_file in enumerate(nii_files):
                if len(nii_files) == 1:
                    new_name = f"{subject_id}.nii"
                else:
                    new_name = f"{subject_id}_{idx+1:02d}.nii"

                new_path = output_dir / new_name
                if nii_file != new_path:
                    nii_file.rename(new_path)
                renamed_files.append(str(new_path))

            return {
                "status": "succeeded",
                "nii_files": renamed_files,
                "bvec_files": [],
                "bval_files": [],
                "json_files": []
            }
        else:
            return {
                "status": "failed",
                "error": result.get("error") or result.get("stderr", "Unknown error")
            }


def convert_dicom_to_nifti(dicom_dirs: List[str], output_dir: str) -> Dict[str, Any]:
    """
    便捷函数：批量转换DICOM到NIfTI

    Args:
        dicom_dirs: DICOM目录列表
        output_dir: 输出目录

    Returns:
        转换结果字典
    """
    tool = LocalDICOMConverterTool()
    request = ToolCallRequest(
        tool_name="dicom_to_nifti",
        call_id="batch_convert",
        inputs={"input_dirs": dicom_dirs},
        params={},
        output_dir=output_dir
    )
    result = tool.execute(request)
    return {
        "status": result.status,
        "output_files": result.outputs.get("output_files", []),
        "conversion_log": result.outputs.get("conversion_log", []),
        "error": result.error
    }


def register_local_tools(registry):
    """注册所有本地工具到注册表"""
    tools = [
        LocalDICOMConverterTool(),  # DICOM转换工具
        LocalSPMTool(),
        LocalDPABITool(),
        LocalDSIStudioTool(),
        LocalDiPyTool(),            # DiPy DTI分析工具（Python实现）
        LocalFreeSurferTool(),
        LocalFSLTool(),
        LocalPythonStatsTool(),
        LocalDataVisualizationTool()  # 数据可视化工具（使用高端LLM）
    ]

    for tool in tools:
        registry.register(tool)

    return tools
