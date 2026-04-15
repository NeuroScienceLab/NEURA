"""
Local tool configuration - directly invoke installed neuroimaging analysis tools
"""
import os
import subprocess
from pathlib import Path

# ============== MATLAB执行配置 ==============
# MATLAB脚本执行超时时间（秒），可通过环境变量覆盖
# 默认2小时，适应大批量fMRI处理（如168个4D文件的slice timing）
MATLAB_TIMEOUT = int(os.environ.get("MATLAB_TIMEOUT", "7200"))

# fMRI分批处理大小：每批处理的最大文件数
# 避免SPM slice timing等模块因一次性加载过多volume导致内存溢出
FMRI_BATCH_SIZE = int(os.environ.get("FMRI_BATCH_SIZE", "10"))

# ============== 本地工具路径配置 (通过环境变量) ==============
# 为了灵活性，请在您的操作系统中设置以下环境变量:
#
# MATLAB_ROOT:      MATLAB的根目录 (例如: C:\Program Files\MATLAB\R2019b)
# SPM_PATH:         SPM工具箱的路径 (例如: C:\spm12)
# DPABI_PATH:       DPABI工具箱的路径 (例如: C:\DPABI_V6.1)
# DSI_STUDIO_PATH:  DSI Studio的路径 (例如: C:\dsi_studio)
# FREESURFER_HOME:  WSL2中FreeSurfer的路径 (例如: /usr/local/freesurfer/7.4.1)
# FSLDIR:           WSL2中FSL的路径 (例如: /usr/local/fsl)
# ....

def get_path_from_env(env_var: str, default_path: str) -> Path:
    """从环境变量获取路径，如果未设置则返回默认值"""
    path_str = os.environ.get(env_var, default_path)
    if not path_str:
        # 如果环境变量和默认值都为空，则引发错误或返回None
        # 这里我们选择打印警告并继续，让可用性检查来处理
        print(f"警告: 环境变量 '{env_var}' 未设置，且没有提供默认路径。")
        return None
    return Path(path_str)

# MATLAB路径
MATLAB_ROOT = get_path_from_env("MATLAB_ROOT", r"H:\MatlabR2019b")
MATLAB_EXE = MATLAB_ROOT / "bin" / "MATLAB R2019b.exe" if MATLAB_ROOT else None

# SPM路径
SPM_PATH = get_path_from_env("SPM_PATH", r"H:\MatlabR2019b\tool\spm25")

# DPABI路径
DPABI_PATH = get_path_from_env("DPABI_PATH", r"H:\MatlabR2019b\tool\DPABI_V90")

# DSI Studio路径
DSI_STUDIO_PATH = get_path_from_env("DSI_STUDIO_PATH", r"H:\dsi_studio_win_2024")
DSI_STUDIO_EXE = DSI_STUDIO_PATH / "dsi_studio.exe" if DSI_STUDIO_PATH else None

# dcm2niix路径（用于DICOM转NIfTI，特别是DWI数据的bvec/bval生成）
DCM2NIIX_PATH = get_path_from_env("DCM2NIIX_PATH", r"I:\AGENT\dcm2niix.exe")

# ============== WSL2 工具路径 ==============

# FreeSurfer路径（WSL2中）
FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer/7.4.1")

# FSL路径（WSL2中）
FSLDIR = os.environ.get("FSLDIR", "/usr/local/fsl")

# FSL专用的WSL发行版（FSL安装在Ubuntu中）
FSL_WSL_DISTRO = os.environ.get("FSL_WSL_DISTRO", "Ubuntu")

# WSL分发版名称（FreeSurfer 7.4.1已安装在Ubuntu-22.04发行版中）
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu-22.04")


# ============== FSL命令白名单定义 ==============
# 统一定义所有支持的FSL命令，确保schema、执行白名单、参数补充保持一致
FSL_SUPPORTED_COMMANDS = [
    # === 基础工具 ===
    "bet",           # 脑提取 (Brain Extraction Tool)
    "fast",          # 组织分割 (FMRIB's Automated Segmentation Tool)
    "flirt",         # 线性配准 (FMRIB's Linear Image Registration Tool)
    "fnirt",         # 非线性配准 (FMRIB's Non-linear Image Registration Tool)
    "applywarp",     # 应用形变场
    "fslroi",        # 提取子卷/ROI
    "fslstats",      # 统计信息提取
    "fslmaths",      # 图像数学运算
    "fslmeants",     # 时间序列/ROI均值提取

    # === DTI/DWI工具 ===
    "eddy",          # 涡流和运动校正（DWI专用）
    "dtifit",        # DTI张量拟合

    # === TBSS白质分析工具链 ===
    "tbss_1_preproc",   # TBSS步骤1: 预处理FA图像
    "tbss_2_reg",       # TBSS步骤2: 配准到FMRIB58_FA标准空间
    "tbss_3_postreg",   # TBSS步骤3: 后处理和骨架投影
    "tbss_4_prestats",  # TBSS步骤4: 准备统计分析（创建skeleton mask）

    # === 纤维追踪预处理 ===
    "bedpostx",      # 贝叶斯估计扩散参数（概率纤维追踪的前置步骤）
    "probtrackx",    # 概率纤维追踪

    # === fMRI预处理工具 ===
    "mcflirt",       # 头动校正 (Motion Correction using FLIRT)
    "slicetimer",    # 层时间校正 (Slice Timing Correction)

    # === 功能分析 (暂不支持，需要完整GLM实现) ===
    # "feat",        # 功能分析 - 需要完整的设计矩阵和GLM实现
]

# FreeSurfer命令白名单（与FSL分开管理）
FREESURFER_SUPPORTED_COMMANDS = [
    "recon-all",
    "recon-all-clinical",
    "segmentBS",           # 脑干分割
    "segmentHA",           # 海马分割
    "segmentThalamus",     # 丘脑分割
    "mri_convert",
    "mri_vol2vol",
    "mri_surf2surf",
    "mris_anatomical_stats",
    "asegstats2table",
    "aparcstats2table",
    "mri_segstats",
    "mris_preproc"
]


# ============== 工具调用函数 ==============

def run_matlab_script(script_content: str, working_dir: str = None) -> dict:
    """
    执行MATLAB脚本

    Args:
        script_content: MATLAB脚本内容
        working_dir: 工作目录

    Returns:
        执行结果字典
    """
    import tempfile
    import time

    # 检查MATLAB是否配置
    if not MATLAB_EXE or not MATLAB_EXE.exists():
        return {
            "status": "failed",
            "error": "MATLAB路径未正确配置或'matlab.exe'不存在。请设置'MATLAB_ROOT'环境变量。",
        }


    # 创建临时脚本文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.m', delete=False, encoding='utf-8') as f:
        # 添加工具路径到MATLAB path
        setup_script = f"""
% 自动添加工具路径
addpath('{str(SPM_PATH).replace(chr(92), "/")}');
addpath(genpath('{str(DPABI_PATH).replace(chr(92), "/")}'));

% 用户脚本
{script_content}

% 退出MATLAB
exit;
"""
        f.write(setup_script)
        script_path = f.name

    # 创建日志文件路径
    log_dir = Path(working_dir) if working_dir else Path(tempfile.gettempdir())
    stdout_log = log_dir / "matlab_stdout.log"
    stderr_log = log_dir / "matlab_stderr.log"

    try:
        # 构建MATLAB命令
        cmd = [
            str(MATLAB_EXE),
            "-batch",  # 无GUI批处理模式
            f"run('{script_path.replace(chr(92), '/')}')"
        ]

        # **改进**: 使用Popen并将输出重定向到文件，避免Windows上的线程问题
        with open(stdout_log, 'w', encoding='utf-8', errors='replace') as stdout_f, \
             open(stderr_log, 'w', encoding='utf-8', errors='replace') as stderr_f:

            process = subprocess.Popen(
                cmd,
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=working_dir,
                text=False  # 使用二进制模式避免编码问题
            )

            # 等待进程完成，超时时间由MATLAB_TIMEOUT配置
            timeout_seconds = MATLAB_TIMEOUT
            start_time = time.time()

            while process.poll() is None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    process.kill()
                    return {
                        "status": "failed",
                        "error": f"MATLAB执行超时（>{timeout_seconds}秒）",
                        "script_path": script_path
                    }
                time.sleep(1)  # 每秒检查一次

            returncode = process.returncode

        # 读取日志内容
        stdout_content = ""
        stderr_content = ""
        try:
            if stdout_log.exists():
                stdout_content = stdout_log.read_text(encoding='utf-8', errors='replace')
        except Exception:
            pass
        try:
            if stderr_log.exists():
                stderr_content = stderr_log.read_text(encoding='utf-8', errors='replace')
        except Exception:
            pass

        return {
            "status": "succeeded" if returncode == 0 else "failed",
            "stdout": stdout_content,
            "stderr": stderr_content,
            "returncode": returncode,
            "script_path": script_path
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "script_path": script_path
        }


def run_dcm2niix(input_dir: str, output_dir: str, compress: bool = True) -> dict:
    """
    使用dcm2niix将DICOM转换为NIfTI

    Args:
        input_dir: DICOM文件夹路径
        output_dir: 输出目录
        compress: 是否压缩输出（.nii.gz）

    Returns:
        执行结果字典，包含输出文件列表
    """
    # 检查dcm2niix是否存在
    if not DCM2NIIX_PATH or not Path(DCM2NIIX_PATH).exists():
        return {
            "status": "failed",
            "error": f"dcm2niix未找到。请设置DCM2NIIX_PATH环境变量或确认路径: {DCM2NIIX_PATH}"
        }

    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # dcm2niix参数:
    # -z y/n: 是否gzip压缩
    # -f %p_%s: 文件名格式（协议名_序列号）
    # -o: 输出目录
    # -b y: 生成BIDS sidecar JSON
    cmd = [
        str(DCM2NIIX_PATH),
        "-z", "y" if compress else "n",  # 压缩输出
        "-f", "%p_%s",                   # 文件名格式：协议名_序列号
        "-b", "y",                       # 生成JSON sidecar
        "-o", output_dir,                # 输出目录
        input_dir                        # 输入DICOM目录
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
            encoding='utf-8',
            errors='replace'
        )

        # 解析输出，查找生成的文件
        output_path = Path(output_dir)
        output_files = {
            'nii': [],
            'bvec': [],
            'bval': [],
            'json': []
        }

        for f in output_path.iterdir():
            if f.is_file():
                if f.suffix == '.gz' and f.stem.endswith('.nii'):
                    output_files['nii'].append(str(f))
                elif f.suffix == '.nii':
                    output_files['nii'].append(str(f))
                elif f.suffix == '.bvec':
                    output_files['bvec'].append(str(f))
                elif f.suffix == '.bval':
                    output_files['bval'].append(str(f))
                elif f.suffix == '.json':
                    output_files['json'].append(str(f))

        return {
            'status': 'succeeded' if result.returncode == 0 else 'failed',
            'files': output_files,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'command': ' '.join(cmd)
        }

    except subprocess.TimeoutExpired:
        return {
            'status': 'failed',
            'error': 'dcm2niix执行超时（>10分钟）'
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }


def run_dsi_studio(args: list, working_dir: str = None) -> dict:
    """
    执行DSI Studio命令

    Args:
        args: 命令行参数列表
        working_dir: 工作目录

    Returns:
        执行结果字典
    """
    # 可用性检查
    if not DSI_STUDIO_EXE or not DSI_STUDIO_EXE.exists():
        return {
            "status": "failed",
            "error": f"DSI Studio未找到。请设置DSI_STUDIO_PATH环境变量或确认路径: {DSI_STUDIO_EXE}"
        }

    cmd = [str(DSI_STUDIO_EXE)] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            cwd=working_dir,
            timeout=7200,  # 2小时超时
            encoding='utf-8',
            errors='replace'  # 替换无法解码的字符
        )

        return {
            "status": "succeeded" if result.returncode == 0 else "failed",
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "returncode": result.returncode,
            "command": " ".join(cmd)
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "error": "DSI Studio执行超时（>2小时）"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }


def run_wsl_command(command: str, env_setup: str = None, clean_path: bool = True,
                    timeout: int = 7200, retries: int = 2, distro: str = None) -> dict:
    """
    在WSL2中执行命令

    Args:
        command: 要执行的命令
        env_setup: 环境设置脚本（如source FreeSurfer/FSL）
        clean_path: 是否清理PATH（避免Windows路径中空格导致的问题）
        timeout: 超时时间（秒），默认2小时
        retries: 重试次数，默认2次
        distro: 指定WSL发行版名称（如果不指定则使用默认的WSL_DISTRO）

    Returns:
        执行结果字典
    """
    # 清理PATH以避免Windows路径中的空格导致问题
    if clean_path:
        path_cleanup = "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    else:
        path_cleanup = ""

    # 构建完整的WSL命令
    # 重要：先cd到根目录，避免Windows当前目录导致的路径解析问题
    parts = ["cd /"]  # 切换到Linux根目录
    if path_cleanup:
        parts.append(path_cleanup)
    if env_setup:
        parts.append(env_setup)
    parts.append(command)

    full_command = " && ".join(parts)

    # 使用指定的发行版，如果没有指定则使用默认值
    target_distro = distro or WSL_DISTRO

    # 使用--cd //来强制切换到Linux根目录（双斜杠避免路径解析问题）
    cmd = [
        "wsl",
        "-d", target_distro,
        "--cd", "//",  # 使用双斜杠确保切换到Linux根目录
        "--", "bash", "-c", full_command
    ]

    last_error = None
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'
            )

            # 检查是否是连接超时错误（返回码127且stderr包含HCS_E_CONNECTION_TIMEOUT）
            if result.returncode == 127 and "CONNECTION_TIMEOUT" in result.stderr:
                last_error = "WSL连接超时"
                if attempt < retries:
                    print(f"  [WSL] 连接超时，重试 {attempt + 2}/{retries + 1}...")
                    # 尝试重启WSL
                    subprocess.run(["wsl", "--shutdown"], timeout=30, capture_output=True)
                    import time
                    time.sleep(2)
                    continue

            return {
                "status": "succeeded" if result.returncode == 0 else "failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": full_command
            }

        except subprocess.TimeoutExpired:
            last_error = f"WSL命令执行超时（>{timeout}秒）"
            if attempt < retries:
                print(f"  [WSL] 超时，重试 {attempt + 2}/{retries + 1}...")
                try:
                    subprocess.run(["wsl", "--shutdown"], timeout=30, capture_output=True)
                except Exception:
                    pass  # 忽略shutdown超时
                import time
                time.sleep(2)
                continue
        except Exception as e:
            last_error = str(e)
            if attempt < retries:
                print(f"  [WSL] 错误: {e}，重试 {attempt + 2}/{retries + 1}...")
                import time
                time.sleep(1)
                continue

    return {
        "status": "failed",
        "error": last_error or "未知错误"
    }


def run_freesurfer(command: str, subjects_dir: str = None) -> dict:
    """
    执行FreeSurfer命令

    Args:
        command: FreeSurfer命令
        subjects_dir: SUBJECTS_DIR路径

    Returns:
        执行结果字典
    """
    # FreeSurfer环境设置 - 使用完整路径避免变量展开问题
    env_setup = f"export FREESURFER_HOME={FREESURFER_HOME} && source {FREESURFER_HOME}/SetUpFreeSurfer.sh"

    if subjects_dir:
        # 将Windows路径转换为WSL路径
        wsl_subjects_dir = windows_to_wsl_path(subjects_dir)
        env_setup += f" && export SUBJECTS_DIR={wsl_subjects_dir}"

    return run_wsl_command(command, env_setup)


def run_fsl(command: str) -> dict:
    """
    执行FSL命令

    Args:
        command: FSL命令（可以使用完整路径或命令名）

    Returns:
        执行结果字典
    """
    # FSL环境设置
    # 注意：fsl.sh 不会自动添加 $FSLDIR/bin 到 PATH，需要手动添加
    # 直接设置完整PATH，避免使用$PATH变量（因为bash会在命令解析时展开，可能包含Windows路径中的空格）
    # FSL 6.x版本的某些工具(imcp, imglob, fsl_sub等)安装在conda环境中
    clean_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    fsl_conda_bin = "/usr/local/miniconda3/envs/fslpython/bin"

    # fsl_sub脚本已手动创建在/usr/local/fsl/bin/fsl_sub
    # 不再动态创建，避免heredoc在bash -c中的解析问题

    env_setup = f"export FSLDIR={FSLDIR} && export PATH={FSLDIR}/bin:{fsl_conda_bin}:{clean_path} && source {FSLDIR}/etc/fslconf/fsl.sh && export FSLOUTPUTTYPE=NIFTI_GZ"

    return run_wsl_command(command, env_setup, distro=FSL_WSL_DISTRO)


def windows_to_wsl_path(windows_path: str) -> str:
    """
    将Windows路径转换为WSL路径

    例如: H:\\data\\subject01 -> /mnt/h/data/subject01
    """
    path = str(windows_path)

    # 处理驱动器号
    if len(path) >= 2 and path[1] == ':':
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"

    return path.replace('\\', '/')


def wsl_to_windows_path(wsl_path: str) -> str:
    """
    将WSL路径转换为Windows路径

    例如: /mnt/h/data/subject01 -> H:\\data\\subject01
    """
    if wsl_path.startswith('/mnt/'):
        parts = wsl_path[5:].split('/', 1)
        if len(parts) == 2:
            drive = parts[0].upper()
            rest = parts[1].replace('/', '\\')
            return f"{drive}:\\{rest}"
        elif len(parts) == 1:
            return f"{parts[0].upper()}:\\"

    return wsl_path


# ============== 工具可用性检查 ==============

def check_tools_availability() -> dict:
    """
    检查所有工具的可用性

    Returns:
        工具可用性状态字典
    """
    status = {}

    # 检查MATLAB
    matlab_available = bool(MATLAB_EXE and MATLAB_EXE.exists())
    status["matlab"] = {
        "available": matlab_available,
        "path": str(MATLAB_EXE) if MATLAB_EXE else "未配置",
        "version": "R2019b (请确保版本兼容)"
    }

    # 检查SPM
    spm_available = bool(SPM_PATH and (SPM_PATH / "spm.m").exists())
    status["spm"] = {
        "available": spm_available,
        "path": str(SPM_PATH) if SPM_PATH else "未配置",
        "version": "SPM25 (或您指定的版本)"
    }

    # 检查DPABI
    dpabi_available = bool(DPABI_PATH and DPABI_PATH.exists())
    status["dpabi"] = {
        "available": dpabi_available,
        "path": str(DPABI_PATH) if DPABI_PATH else "未配置",
        "version": "V90 (或您指定的版本)"
    }

    # 检查DSI Studio
    dsi_available = bool(DSI_STUDIO_EXE and DSI_STUDIO_EXE.exists())
    status["dsi_studio"] = {
        "available": dsi_available,
        "path": str(DSI_STUDIO_EXE) if DSI_STUDIO_EXE else "未配置",
        "version": "2024 (或您指定的版本)"
    }

    # 检查dcm2niix
    dcm2niix_available = bool(DCM2NIIX_PATH and Path(DCM2NIIX_PATH).exists())
    status["dcm2niix"] = {
        "available": dcm2niix_available,
        "path": str(DCM2NIIX_PATH) if DCM2NIIX_PATH else "未配置"
    }

    # 检查WSL2
    try:
        result = subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "--", "echo", "OK"],
            capture_output=True,
            text=True,
            timeout=10
        )
        wsl_available = result.returncode == 0 and "OK" in result.stdout
    except:
        wsl_available = False

    status["wsl"] = {
        "available": wsl_available,
        "distro": WSL_DISTRO
    }

    # 检查FreeSurfer（通过WSL）
    if wsl_available:
        result = run_wsl_command(f"test -d {FREESURFER_HOME} && echo OK")
        status["freesurfer"] = {
            "available": "OK" in result.get("stdout", ""),
            "path": FREESURFER_HOME,
            "version": "7.4.1"
        }

        # 检查FSL
        result = run_wsl_command(f"test -d {FSLDIR} && echo OK")
        status["fsl"] = {
            "available": "OK" in result.get("stdout", ""),
            "path": FSLDIR
        }
    else:
        status["freesurfer"] = {"available": False, "error": "WSL不可用"}
        status["fsl"] = {"available": False, "error": "WSL不可用"}

    return status


if __name__ == "__main__":
    # 测试工具可用性
    print("检查本地工具可用性...\n")
    status = check_tools_availability()

    for tool, info in status.items():
        available = info.get("available", False)
        symbol = "[OK]" if available else "[X]"
        print(f"{symbol} {tool:12s}: {info}")
