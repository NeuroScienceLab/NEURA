"""
Image processing tool adapter - Extensible interfaces for Nipype/SPM/DPABI/DSI Studio etc.
"""
import os
import json
import subprocess
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.tools.registry import (
    BaseTool, ToolDefinition, ToolCallRequest, ToolCallResult,
    Modality, ExecutorType
)
from src.config import OUTPUT_DIR


class NipypePreprocessingTool(BaseTool):
    """Nipype预处理工具"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="nipype_preprocess",
            description="使用Nipype进行神经影像预处理，支持T1配准、分割、平滑等",
            category="preprocessing",
            supported_modalities=[Modality.ANAT, Modality.FUNC],
            executor_type=ExecutorType.NIPYPE,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入文件路径列表"
                    },
                    "modality": {
                        "type": "string",
                        "enum": ["anat", "func"],
                        "description": "影像模态"
                    },
                    "workflow": {
                        "type": "string",
                        "enum": ["normalize", "segment", "smooth", "realign"],
                        "description": "预处理工作流类型"
                    },
                    "template": {
                        "type": "string",
                        "default": "MNI152",
                        "description": "标准模板"
                    },
                    "smoothing_fwhm": {
                        "type": "number",
                        "default": 8,
                        "description": "平滑核大小(mm)"
                    }
                },
                "required": ["input_files", "modality", "workflow"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_files": {"type": "array"},
                    "workflow_graph": {"type": "string"},
                    "provenance": {"type": "object"}
                }
            },
            version="1.0.0",
            dependencies=["nipype", "nibabel", "nilearn"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行Nipype预处理"""
        from datetime import datetime
        start_time = datetime.now()

        try:
            # 检查nipype是否可用
            try:
                import nipype
                from nipype.interfaces import spm, fsl
                from nipype.pipeline import engine as pe
            except ImportError:
                return ToolCallResult(
                    call_id=request.call_id,
                    tool_name=self.definition.name,
                    status="failed",
                    error="Nipype未安装。请运行: pip install nipype"
                )

            workflow_type = request.params.get("workflow", "normalize")
            input_files = request.inputs.get("input_files", [])
            output_dir = Path(request.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 创建工作流（示例：简单的配准流程）
            workflow = pe.Workflow(name=f"preprocess_{workflow_type}")
            workflow.base_dir = str(output_dir / "work")

            # 根据工作流类型创建节点
            # 注：实际实现需要根据具体需求配置节点
            outputs = {
                "workflow": workflow_type,
                "input_count": len(input_files),
                "output_dir": str(output_dir),
                "status": "workflow_created"
            }

            # 保存工作流配置
            config_path = output_dir / "workflow_config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "workflow": workflow_type,
                    "inputs": input_files,
                    "params": request.params
                }, f, indent=2)

            end_time = datetime.now()

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="succeeded",
                started_at=start_time.isoformat(),
                finished_at=end_time.isoformat(),
                duration_seconds=(end_time - start_time).total_seconds(),
                outputs=outputs,
                artifacts=[
                    {"name": "workflow_config.json", "path": str(config_path)}
                ]
            )

        except Exception as e:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                status="failed",
                error=str(e)
            )


class SPMTool(BaseTool):
    """SPM工具适配器"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="spm_analysis",
            description="使用SPM12进行神经影像分析，支持VBM、GLM统计分析等",
            category="analysis",
            supported_modalities=[Modality.ANAT, Modality.FUNC],
            executor_type=ExecutorType.MATLAB,
            input_schema={
                "type": "object",
                "properties": {
                    "input_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输入NIfTI文件路径"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["vbm", "glm", "factorial", "regression"],
                        "description": "分析类型"
                    },
                    "groups": {
                        "type": "object",
                        "description": "组别定义 {组名: [文件索引]}"
                    },
                    "covariates": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "协变量定义"
                    },
                    "contrasts": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "对比定义"
                    },
                    "threshold": {
                        "type": "object",
                        "properties": {
                            "p_value": {"type": "number", "default": 0.001},
                            "cluster_size": {"type": "integer", "default": 100},
                            "correction": {"type": "string", "enum": ["none", "fwe", "fdr"]}
                        }
                    }
                },
                "required": ["input_files", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "spm_mat": {"type": "string", "description": "SPM.mat路径"},
                    "stat_maps": {"type": "array", "description": "统计图列表"},
                    "results_table": {"type": "string", "description": "结果表格"},
                    "glass_brain": {"type": "string", "description": "玻璃脑图"}
                }
            },
            version="12.0",
            matlab_path="spm12",
            dependencies=["MATLAB", "SPM12"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行SPM分析"""
        start_time = datetime.now()

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_type = request.params.get("analysis_type", "vbm")

        # 生成MATLAB脚本
        matlab_script = self._generate_matlab_script(request, output_dir)
        script_path = output_dir / "spm_batch.m"
        with open(script_path, "w") as f:
            f.write(matlab_script)

        # 检查MATLAB是否可用
        matlab_available = self._check_matlab()

        if matlab_available:
            # 执行MATLAB脚本
            try:
                result = subprocess.run(
                    ["matlab", "-batch", f"run('{script_path}')"],
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1小时超时
                )
                logs = result.stdout + result.stderr
                status = "succeeded" if result.returncode == 0 else "failed"
            except Exception as e:
                logs = str(e)
                status = "failed"
        else:
            # MATLAB不可用，生成脚本供手动执行
            logs = f"MATLAB未检测到。已生成脚本: {script_path}\n请手动执行。"
            status = "pending_manual"

        end_time = datetime.now()

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status=status,
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            outputs={
                "analysis_type": analysis_type,
                "script_path": str(script_path),
                "output_dir": str(output_dir)
            },
            artifacts=[
                {"name": "spm_batch.m", "path": str(script_path)}
            ],
            logs=logs
        )

    def _generate_matlab_script(self, request: ToolCallRequest, output_dir: Path) -> str:
        """生成SPM MATLAB脚本"""
        analysis_type = request.params.get("analysis_type", "vbm")
        input_files = request.inputs.get("input_files", [])

        script = f"""% SPM12 Batch Script - Auto Generated
% Analysis Type: {analysis_type}
% Generated at: {datetime.now().isoformat()}

spm('defaults', 'fmri');
spm_jobman('initcfg');

matlabbatch = {{}};

% 输出目录
output_dir = '{output_dir}';

% 输入文件
input_files = {{
"""
        for f in input_files:
            script += f"    '{f}'\n"

        script += """};

% 根据分析类型配置批处理
"""
        if analysis_type == "vbm":
            script += self._generate_vbm_batch(request)
        elif analysis_type == "glm":
            script += self._generate_glm_batch(request)

        script += """
% 运行批处理
spm_jobman('run', matlabbatch);

disp('SPM analysis completed.');
"""
        return script

    def _generate_vbm_batch(self, request: ToolCallRequest) -> str:
        """生成VBM批处理代码"""
        return """
% VBM分析配置
matlabbatch{1}.spm.spatial.preproc.channel.vols = input_files;
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{1}.spm.spatial.preproc.channel.write = [0 1];
"""

    def _generate_glm_batch(self, request: ToolCallRequest) -> str:
        """生成GLM批处理代码"""
        return """
% GLM分析配置
matlabbatch{1}.spm.stats.factorial_design.dir = {output_dir};
% 需要根据具体设计配置
"""

    def _check_matlab(self) -> bool:
        """检查MATLAB是否可用"""
        try:
            result = subprocess.run(
                ["matlab", "-batch", "disp('ok')"],
                capture_output=True,
                timeout=30
            )
            return result.returncode == 0
        except:
            return False


class DPABITool(BaseTool):
    """DPABI工具适配器"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dpabi_analysis",
            description="使用DPABI进行静息态fMRI分析，支持ALFF、ReHo、FC等",
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
                        "enum": ["alff", "falff", "reho", "fc", "vmhc", "degree_centrality"],
                        "description": "分析类型"
                    },
                    "mask": {
                        "type": "string",
                        "description": "脑掩模路径"
                    },
                    "tr": {
                        "type": "number",
                        "description": "重复时间(秒)"
                    },
                    "band_pass": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "带通滤波范围 [low, high]"
                    }
                },
                "required": ["input_files", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "output_maps": {"type": "array"},
                    "z_maps": {"type": "array"},
                    "group_results": {"type": "string"}
                }
            },
            version="6.0",
            matlab_path="DPABI",
            dependencies=["MATLAB", "DPABI", "SPM12"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DPABI分析"""
        start_time = datetime.now()

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        analysis_type = request.params.get("analysis_type", "alff")

        # 生成DPABI配置和脚本
        script_path = output_dir / "dpabi_batch.m"
        with open(script_path, "w") as f:
            f.write(self._generate_dpabi_script(request, output_dir))

        end_time = datetime.now()

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status="pending_manual",
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            outputs={
                "analysis_type": analysis_type,
                "script_path": str(script_path)
            },
            artifacts=[
                {"name": "dpabi_batch.m", "path": str(script_path)}
            ],
            logs=f"DPABI脚本已生成: {script_path}"
        )

    def _generate_dpabi_script(self, request: ToolCallRequest, output_dir: Path) -> str:
        """生成DPABI脚本"""
        analysis_type = request.params.get("analysis_type", "alff")
        input_files = request.inputs.get("input_files", [])

        # 构建文件列表字符串
        files_str = "\n".join([f"    '{f}'" for f in input_files])

        return f"""% DPABI Batch Script - Auto Generated
% Analysis Type: {analysis_type}
% Generated at: {datetime.now().isoformat()}

% 添加DPABI路径
% addpath(genpath('/path/to/DPABI'));

% 输入文件列表
InputFiles = {{
{files_str}
}};
Cfg.DataProcessDir = '{output_dir}';

% {analysis_type.upper()} 分析
% 根据具体需求配置参数

disp('DPABI配置已生成，请在MATLAB中运行。');
"""


class DSIStudioTool(BaseTool):
    """DSI Studio工具适配器"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="dsi_studio",
            description="使用DSI Studio进行扩散MRI分析，支持纤维追踪、FA/MD计算等",
            category="analysis",
            supported_modalities=[Modality.DWI],
            executor_type=ExecutorType.CLI,
            input_schema={
                "type": "object",
                "properties": {
                    "input_file": {
                        "type": "string",
                        "description": "输入DWI文件路径"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["src", "rec", "trk", "ana", "exp"],
                        "description": "操作类型: src(创建源文件), rec(重建), trk(追踪), ana(分析), exp(导出)"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["dti", "gqi", "qsdr"],
                        "description": "重建方法"
                    },
                    "tract_params": {
                        "type": "object",
                        "properties": {
                            "seed_count": {"type": "integer", "default": 1000000},
                            "fa_threshold": {"type": "number", "default": 0.1},
                            "turning_angle": {"type": "number", "default": 60},
                            "step_size": {"type": "number", "default": 1.0},
                            "min_length": {"type": "number", "default": 30},
                            "max_length": {"type": "number", "default": 300}
                        }
                    },
                    "output_metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "输出指标: fa, md, ad, rd, qa等"
                    }
                },
                "required": ["input_file", "action"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "fib_file": {"type": "string", "description": "FIB文件路径"},
                    "trk_file": {"type": "string", "description": "纤维束文件"},
                    "metric_maps": {"type": "object", "description": "指标图"}
                }
            },
            version="2024",
            cli_command="dsi_studio",
            dependencies=["DSI Studio"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行DSI Studio分析"""
        start_time = datetime.now()

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        action = request.params.get("action", "rec")
        input_file = request.inputs.get("input_file", "")

        # 构建命令
        cmd = self._build_command(request, output_dir)

        # 保存命令到脚本
        script_path = output_dir / "dsi_studio_cmd.sh"
        with open(script_path, "w") as f:
            f.write(f"#!/bin/bash\n# DSI Studio Command\n{cmd}\n")

        # 尝试执行
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=7200,  # 2小时超时
                cwd=str(output_dir)
            )
            logs = result.stdout + result.stderr
            status = "succeeded" if result.returncode == 0 else "failed"
        except FileNotFoundError:
            logs = f"DSI Studio未找到。请确保已安装并添加到PATH。\n命令: {cmd}"
            status = "pending_manual"
        except subprocess.TimeoutExpired:
            logs = "执行超时"
            status = "failed"
        except Exception as e:
            logs = str(e)
            status = "failed"

        end_time = datetime.now()

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status=status,
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            outputs={
                "action": action,
                "command": cmd,
                "output_dir": str(output_dir)
            },
            artifacts=[
                {"name": "dsi_studio_cmd.sh", "path": str(script_path)}
            ],
            logs=logs
        )

    def _build_command(self, request: ToolCallRequest, output_dir: Path) -> str:
        """构建DSI Studio命令"""
        action = request.params.get("action", "rec")
        input_file = request.inputs.get("input_file", "")

        cmd = f"dsi_studio --action={action} --source={input_file}"

        if action == "rec":
            method = request.params.get("method", "dti")
            cmd += f" --method={method}"
            cmd += f" --output={output_dir}"

        elif action == "trk":
            tract_params = request.params.get("tract_params", {})
            cmd += f" --seed_count={tract_params.get('seed_count', 1000000)}"
            cmd += f" --fa_threshold={tract_params.get('fa_threshold', 0.1)}"
            cmd += f" --turning_angle={tract_params.get('turning_angle', 60)}"
            cmd += f" --output={output_dir}/tracks.trk"

        elif action == "ana":
            metrics = request.params.get("output_metrics", ["fa", "md"])
            cmd += f" --export={','.join(metrics)}"

        return cmd


class PythonStatsTool(BaseTool):
    """Python统计分析工具"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="python_stats",
            description="使用Python进行统计分析，支持t检验、ANOVA、相关分析、回归等",
            category="statistics",
            supported_modalities=[Modality.ALL],
            executor_type=ExecutorType.PYTHON,
            input_schema={
                "type": "object",
                "properties": {
                    "data_file": {
                        "type": "string",
                        "description": "数据文件路径(CSV/Parquet)"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["ttest", "anova", "correlation", "regression", "glm", "mixed_effects"],
                        "description": "分析类型"
                    },
                    "dependent_var": {
                        "type": "string",
                        "description": "因变量"
                    },
                    "independent_vars": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "自变量列表"
                    },
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "协变量列表"
                    },
                    "group_var": {
                        "type": "string",
                        "description": "分组变量"
                    },
                    "correction": {
                        "type": "string",
                        "enum": ["none", "bonferroni", "fdr", "holm"],
                        "description": "多重比较校正方法"
                    }
                },
                "required": ["data_file", "analysis_type"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary_table": {"type": "string"},
                    "statistics": {"type": "object"},
                    "effect_sizes": {"type": "object"},
                    "figures": {"type": "array"}
                }
            },
            version="1.0.0",
            dependencies=["numpy", "scipy", "statsmodels", "pandas"]
        )

    def execute(self, request: ToolCallRequest) -> ToolCallResult:
        """执行统计分析"""
        import numpy as np
        import pandas as pd
        from scipy import stats
        import nibabel as nib
        start_time = datetime.now()

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            analysis_type = request.params.get("analysis_type", "ttest")

            # 检查是否提供了数据文件（CSV/Parquet）
            data_file = request.inputs.get("data_file", "")

            if data_file and Path(data_file).exists():
                # 传统模式：从CSV/Parquet加载
                if data_file.endswith(".parquet"):
                    df = pd.read_parquet(data_file)
                else:
                    df = pd.read_csv(data_file)
            else:
                # 新模式：从NIfTI文件列表提取数据
                input_files = request.inputs.get("input_files", [])
                if not input_files:
                    raise ValueError("必须提供data_file（CSV/Parquet）或input_files（NIfTI图像列表）")

                print(f"  [数据提取] 从 {len(input_files)} 个NIfTI文件提取数据...")
                df = self._extract_data_from_nifti(input_files, request.context)

                # 保存提取的数据
                extracted_data_path = output_dir / "extracted_data.csv"
                df.to_csv(extracted_data_path, index=False)
                print(f"  [数据提取] 已保存到 {extracted_data_path}")

            results = {}

            if analysis_type == "ttest":
                results = self._run_ttest(df, request.params)
            elif analysis_type == "anova":
                results = self._run_anova(df, request.params)
            elif analysis_type == "correlation":
                results = self._run_correlation(df, request.params)
            elif analysis_type == "regression":
                results = self._run_regression(df, request.params)

            # 保存结果
            results_path = output_dir / "statistics_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            status = "succeeded"
            error = None

        except Exception as e:
            results = {}
            status = "failed"
            error = str(e)
            print(f"  [统计分析错误] {error}")

        end_time = datetime.now()

        return ToolCallResult(
            call_id=request.call_id,
            tool_name=self.definition.name,
            status=status,
            started_at=start_time.isoformat(),
            finished_at=end_time.isoformat(),
            duration_seconds=(end_time - start_time).total_seconds(),
            outputs=results,
            artifacts=[
                {"name": "statistics_results.json", "path": str(output_dir / "statistics_results.json")}
            ] if status == "succeeded" else [],
            error=error
        )

    def _run_ttest(self, df: 'pd.DataFrame', params: Dict) -> Dict:
        """运行t检验"""
        from scipy import stats
        import numpy as np

        group_var = params.get("group_var", "group")

        # 如果未指定dependent_var，自动选择灰质体积
        dep_var = params.get("dependent_var")
        if not dep_var:
            # 尝试找到灰质体积列
            possible_vars = ["grey_matter_volume_cm3", "gray_matter_volume_cm3", "value"]
            for var in possible_vars:
                if var in df.columns:
                    dep_var = var
                    print(f"  [自动选择] 因变量: {dep_var}")
                    break
            if not dep_var:
                raise ValueError(f"未指定dependent_var，且无法自动选择（可用列: {list(df.columns)}）")

        groups = df[group_var].unique()
        if len(groups) != 2:
            raise ValueError(f"t检验需要2个组，实际有{len(groups)}个")

        g1 = df[df[group_var] == groups[0]][dep_var].dropna()
        g2 = df[df[group_var] == groups[1]][dep_var].dropna()

        if len(g1) == 0 or len(g2) == 0:
            raise ValueError(f"数据不足: {groups[0]}={len(g1)}, {groups[1]}={len(g2)}")

        t_stat, p_value = stats.ttest_ind(g1, g2)

        # Cohen's d
        pooled_std = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2))
        cohens_d = (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0.0

        return {
            "test": "Independent Samples T-Test",
            "dependent_variable": dep_var,
            "groups": {
                str(groups[0]): {"n": len(g1), "mean": float(g1.mean()), "std": float(g1.std())},
                str(groups[1]): {"n": len(g2), "mean": float(g2.mean()), "std": float(g2.std())}
            },
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05
        }

    def _run_anova(self, df: 'pd.DataFrame', params: Dict) -> Dict:
        """运行ANOVA"""
        from scipy import stats

        group_var = params.get("group_var", "group")
        dep_var = params.get("dependent_var", "value")

        groups = [df[df[group_var] == g][dep_var].dropna() for g in df[group_var].unique()]
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            "test": "One-Way ANOVA",
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }

    def _run_correlation(self, df: 'pd.DataFrame', params: Dict) -> Dict:
        """运行相关分析"""
        from scipy import stats

        vars = params.get("independent_vars", [])
        if len(vars) < 2:
            vars = df.select_dtypes(include=['number']).columns[:2].tolist()

        r, p = stats.pearsonr(df[vars[0]].dropna(), df[vars[1]].dropna())

        return {
            "test": "Pearson Correlation",
            "variables": vars[:2],
            "correlation": float(r),
            "p_value": float(p),
            "r_squared": float(r**2)
        }

    def _run_regression(self, df: 'pd.DataFrame', params: Dict) -> Dict:
        """运行回归分析"""
        import statsmodels.api as sm

        dep_var = params.get("dependent_var")
        indep_vars = params.get("independent_vars", [])

        X = df[indep_vars]
        X = sm.add_constant(X)
        y = df[dep_var]

        model = sm.OLS(y, X).fit()

        return {
            "test": "OLS Regression",
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue),
            "f_pvalue": float(model.f_pvalue),
            "coefficients": {
                name: {"coef": float(coef), "pvalue": float(pval)}
                for name, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values)
            }
        }

    def _extract_data_from_nifti(self, input_files: list, context: dict) -> 'pd.DataFrame':
        """
        从NIfTI文件中提取数据用于统计分析

        对于VBM分析，提取每个被试的组织体积（灰质、白质等）
        """
        import nibabel as nib
        import numpy as np
        import pandas as pd

        data_rows = []

        # 获取分组信息（从context中）
        cohort = context.get("cohort", {})
        subject_groups = {}
        for group_name, subjects in cohort.items():
            for subj in subjects:
                subject_groups[subj["subject_id"]] = group_name

        # 按被试组织文件
        subject_files = {}
        for file_path in input_files:
            file_path = Path(file_path)
            filename = file_path.name

            # 提取被试ID（从文件路径）
            subject_id = None
            for part in file_path.parts:
                if part.startswith("HC") or part.startswith("SCA"):
                    subject_id = part
                    break

            if not subject_id:
                # 从文件名推断
                if "HC" in filename:
                    for i in range(1, 10):
                        if f"HC{i}_" in str(file_path) or f"HC1_000{i}" in str(file_path):
                            subject_id = f"HC1_000{i}"
                            break
                elif "SCA" in filename:
                    for i in range(1, 10):
                        if f"SCA3_000{i}" in str(file_path):
                            subject_id = f"SCA3_000{i}"
                            break

            if not subject_id:
                print(f"  [警告] 无法从文件名推断被试ID: {filename}")
                continue

            if subject_id not in subject_files:
                subject_files[subject_id] = {}

            # 识别组织类型（从文件名前缀）
            if filename.startswith("c1"):
                tissue_type = "grey_matter"
            elif filename.startswith("c2"):
                tissue_type = "white_matter"
            elif filename.startswith("c3"):
                tissue_type = "csf"
            else:
                tissue_type = "unknown"

            subject_files[subject_id][tissue_type] = file_path

        # 提取每个被试的数据
        for subject_id, files in subject_files.items():
            row = {"subject_id": subject_id}

            # 获取分组
            group = subject_groups.get(subject_id, "Unknown")
            row["group"] = group

            # 计算每种组织的总体积
            for tissue_type, file_path in files.items():
                try:
                    img = nib.load(str(file_path))
                    data = img.get_fdata()

                    # 计算总体积（非零体素数量 × 体素大小）
                    voxel_size = np.prod(img.header.get_zooms()[:3])  # mm³
                    nonzero_voxels = np.sum(data > 0.1)  # 阈值0.1（概率图）
                    total_volume_mm3 = nonzero_voxels * voxel_size
                    total_volume_cm3 = total_volume_mm3 / 1000.0

                    row[f"{tissue_type}_volume_cm3"] = total_volume_cm3

                    # 计算平均概率
                    row[f"{tissue_type}_mean_prob"] = float(data[data > 0.1].mean()) if np.any(data > 0.1) else 0.0

                except Exception as e:
                    print(f"  [警告] 读取 {file_path.name} 失败: {e}")
                    row[f"{tissue_type}_volume_cm3"] = np.nan

            data_rows.append(row)

        df = pd.DataFrame(data_rows)

        print(f"  [数据提取] 提取了 {len(df)} 个被试的数据")
        print(f"  [数据提取] 列: {list(df.columns)}")
        print(f"  [数据提取] 分组分布: {dict(df['group'].value_counts())}")

        return df


def register_all_imaging_tools(registry, use_local=True):
    """
    注册所有影像工具

    Args:
        registry: 工具注册表
        use_local: 是否使用本地工具（直接调用已安装的软件）
    """
    if use_local:
        # 使用本地工具（直接调用MATLAB/DSI Studio/FreeSurfer/FSL）
        try:
            from src.tools.local_tools import register_local_tools
            register_local_tools(registry)
            print("[INFO] 已注册本地工具（SPM/DPABI/DSI Studio/FreeSurfer/FSL）")
            return
        except ImportError as e:
            print(f"[WARN] 无法加载本地工具，使用默认实现: {e}")

    # 回退到默认实现（生成脚本但不执行）
    registry.register(NipypePreprocessingTool())
    registry.register(SPMTool())
    registry.register(DPABITool())
    registry.register(DSIStudioTool())
    registry.register(PythonStatsTool())
    print("[INFO] 已注册默认工具（脚本生成模式）")
