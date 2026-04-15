"""
Data Tools Module - DICOM Reading and Processing, Environmental Awareness
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

from src.config import DATA_DIR, DATA_GROUPS, MODALITIES, KNOWN_MODALITY_DIRS, SCAN_SKIP_PATTERNS


def _infer_group(data_dir: Path, subject_id: str) -> Optional[str]:
    """从文件系统推断被试所属组别"""
    for d in data_dir.iterdir():
        if not d.is_dir() or d.name in SCAN_SKIP_PATTERNS or d.name.endswith(".db"):
            continue
        sub_names = {sd.name for sd in d.iterdir() if sd.is_dir()}
        if sub_names & KNOWN_MODALITY_DIRS:
            # 这是一个组目录，检查是否包含该被试
            for mod_name in sub_names & KNOWN_MODALITY_DIRS:
                if (d / mod_name / subject_id).exists():
                    return d.name
    return None


def _discover_all_groups(data_dir: Path) -> Dict[str, List[str]]:
    """从文件系统自动发现所有组别和被试"""
    groups = {}
    for d in data_dir.iterdir():
        if not d.is_dir() or d.name in SCAN_SKIP_PATTERNS or d.name.endswith(".db"):
            continue
        sub_names = {sd.name for sd in d.iterdir() if sd.is_dir()}
        if sub_names & KNOWN_MODALITY_DIRS:
            subjects = set()
            for mod_name in sub_names & KNOWN_MODALITY_DIRS:
                for subject_dir in (d / mod_name).iterdir():
                    if subject_dir.is_dir():
                        subjects.add(subject_dir.name)
            if subjects:
                groups[d.name] = sorted(subjects)
    return groups


@dataclass
class SubjectInfo:
    """受试者信息"""
    subject_id: str
    group: str  # 组别（自动发现）
    modalities: List[str]  # 可用的模态
    file_counts: Dict[str, int]  # 每个模态的文件数
    metadata: Dict = field(default_factory=dict)


@dataclass
class DatasetSummary:
    """数据集摘要"""
    total_subjects: int
    groups: Dict[str, int]  # 每组的受试者数
    modalities: Dict[str, int]  # 每个模态的受试者数
    subjects: List[SubjectInfo]


class DataTools:
    """数据工具类"""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR

    def scan_dataset(self) -> DatasetSummary:
        """
        扫描数据集（自动发现组别）

        Returns:
            数据集摘要
        """
        subjects = []
        groups_count = {}
        modalities_count = {}

        discovered = _discover_all_groups(self.data_dir)

        for group_name, group_subjects in discovered.items():
            group_path = self.data_dir / group_name
            groups_count[group_name] = 0

            # 遍历模态
            for modality in MODALITIES.keys():
                modality_path = group_path / modality
                if not modality_path.exists():
                    continue

                # 遍历受试者
                for subject_dir in modality_path.iterdir():
                    if not subject_dir.is_dir():
                        continue

                    subject_id = subject_dir.name

                    # 检查是否已记录该受试者
                    existing = next((s for s in subjects if s.subject_id == subject_id), None)

                    if existing:
                        if modality not in existing.modalities:
                            existing.modalities.append(modality)
                        existing.file_counts[modality] = len(list(subject_dir.glob("*")))
                    else:
                        file_count = len(list(subject_dir.glob("*")))
                        subjects.append(SubjectInfo(
                            subject_id=subject_id,
                            group=group_name,
                            modalities=[modality],
                            file_counts={modality: file_count}
                        ))
                        groups_count[group_name] = groups_count.get(group_name, 0) + 1

        # 统计模态数量
        for subject in subjects:
            for modality in subject.modalities:
                modalities_count[modality] = modalities_count.get(modality, 0) + 1

        return DatasetSummary(
            total_subjects=len(subjects),
            groups=groups_count,
            modalities=modalities_count,
            subjects=subjects
        )

    def get_subject_data(self, subject_id: str, modality: str) -> Dict[str, Any]:
        """
        获取受试者数据信息

        Args:
            subject_id: 受试者ID
            modality: 模态

        Returns:
            数据信息字典
        """
        # 确定组别（从文件系统推断）
        group = _infer_group(self.data_dir, subject_id)
        if not group:
            return {"error": f"无法确定被试 {subject_id} 的组别"}

        data_path = self.data_dir / group / modality / subject_id

        if not data_path.exists():
            return {"error": f"数据路径不存在: {data_path}"}

        files = list(data_path.glob("*"))

        return {
            "subject_id": subject_id,
            "group": group,
            "group_name": DATA_GROUPS.get(group, "未知"),
            "modality": modality,
            "modality_name": MODALITIES.get(modality, "未知"),
            "path": str(data_path),
            "file_count": len(files),
            "files": [f.name for f in files[:10]]  # 只返回前10个文件名
        }

    def read_dicom_header(self, subject_id: str, modality: str) -> Dict[str, Any]:
        """
        读取DICOM头信息

        Args:
            subject_id: 受试者ID
            modality: 模态

        Returns:
            DICOM头信息
        """
        try:
            import pydicom
        except ImportError:
            return {"error": "需要安装pydicom库: pip install pydicom"}

        # 确定组别（从文件系统推断）
        group = _infer_group(self.data_dir, subject_id)
        if not group:
            return {"error": f"无法确定被试 {subject_id} 的组别"}
        data_path = self.data_dir / group / modality / subject_id

        if not data_path.exists():
            return {"error": f"数据路径不存在: {data_path}"}

        # 找到第一个DICOM文件
        dcm_files = list(data_path.glob("*.dcm")) + list(data_path.glob("*"))
        if not dcm_files:
            return {"error": "未找到DICOM文件"}

        try:
            dcm = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)

            header_info = {
                "subject_id": subject_id,
                "modality": modality,
                "file": dcm_files[0].name,
                "patient_id": str(getattr(dcm, "PatientID", "N/A")),
                "study_date": str(getattr(dcm, "StudyDate", "N/A")),
                "modality_dicom": str(getattr(dcm, "Modality", "N/A")),
                "manufacturer": str(getattr(dcm, "Manufacturer", "N/A")),
                "series_description": str(getattr(dcm, "SeriesDescription", "N/A")),
                "slice_thickness": str(getattr(dcm, "SliceThickness", "N/A")),
                "rows": int(getattr(dcm, "Rows", 0)),
                "columns": int(getattr(dcm, "Columns", 0)),
            }

            return header_info

        except Exception as e:
            return {"error": f"读取DICOM失败: {str(e)}"}

    def load_volume(self, subject_id: str, modality: str) -> Dict[str, Any]:
        """
        加载3D体数据

        Args:
            subject_id: 受试者ID
            modality: 模态

        Returns:
            体数据信息
        """
        try:
            import pydicom
            import numpy as np
        except ImportError:
            return {"error": "需要安装pydicom和numpy库"}

        group = _infer_group(self.data_dir, subject_id)
        if not group:
            return {"error": f"无法确定被试 {subject_id} 的组别"}
        data_path = self.data_dir / group / modality / subject_id

        if not data_path.exists():
            return {"error": f"数据路径不存在: {data_path}"}

        dcm_files = sorted(data_path.glob("*"))
        if not dcm_files:
            return {"error": "未找到DICOM文件"}

        try:
            # 读取所有切片
            slices = []
            for dcm_file in dcm_files:
                try:
                    dcm = pydicom.dcmread(str(dcm_file))
                    if hasattr(dcm, "pixel_array"):
                        slices.append(dcm)
                except:
                    continue

            if not slices:
                return {"error": "无法读取像素数据"}

            # 按位置排序
            slices.sort(key=lambda x: float(getattr(x, "InstanceNumber", 0)))

            # 构建3D数组
            volume = np.stack([s.pixel_array for s in slices])

            return {
                "subject_id": subject_id,
                "modality": modality,
                "shape": list(volume.shape),
                "dtype": str(volume.dtype),
                "min_value": float(volume.min()),
                "max_value": float(volume.max()),
                "mean_value": float(volume.mean()),
                "std_value": float(volume.std())
            }

        except Exception as e:
            return {"error": f"加载体数据失败: {str(e)}"}

    def compare_groups(self) -> Dict[str, Any]:
        """
        比较各组数据（自动发现组别）

        Returns:
            组间比较信息
        """
        summary = self.scan_dataset()

        # 按组分类
        groups_data = {}
        for subject in summary.subjects:
            if subject.group not in groups_data:
                groups_data[subject.group] = []
            groups_data[subject.group].append(subject)

        result_groups = {}
        for group_name, group_subjects in groups_data.items():
            desc = DATA_GROUPS.get(group_name, group_name)
            result_groups[group_name] = {
                "name": desc,
                "count": len(group_subjects),
                "subjects": [s.subject_id for s in group_subjects]
            }

        result_modalities = {}
        for modality in MODALITIES.keys():
            mod_info = {"name": MODALITIES[modality]}
            for group_name, group_subjects in groups_data.items():
                mod_info[f"{group_name}_count"] = len([s for s in group_subjects if modality in s.modalities])
            result_modalities[modality] = mod_info

        return {
            "groups": result_groups,
            "modalities": result_modalities
        }


# 工具函数定义（NEURA）
def get_tool_definitions() -> List[Dict]:
    """返回数据工具的函数定义"""
    return [
        {
            "type": "function",
            "function": {
                "name": "scan_dataset",
                "description": "扫描数据集，获取所有受试者和可用模态的概览信息",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_subject_data",
                "description": "获取指定受试者的特定模态数据信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_id": {
                            "type": "string",
                            "description": "受试者ID，如 HC1_0001 或 SCA3_0001"
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["anat", "dwi", "func"],
                            "description": "影像模态: anat(解剖), dwi(扩散), func(功能)"
                        }
                    },
                    "required": ["subject_id", "modality"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_dicom_header",
                "description": "读取DICOM文件头信息，获取扫描参数",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_id": {
                            "type": "string",
                            "description": "受试者ID"
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["anat", "dwi", "func"],
                            "description": "影像模态"
                        }
                    },
                    "required": ["subject_id", "modality"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "load_volume",
                "description": "加载3D体数据并返回统计信息（形状、数值范围、均值、标准差）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject_id": {
                            "type": "string",
                            "description": "受试者ID"
                        },
                        "modality": {
                            "type": "string",
                            "enum": ["anat", "dwi", "func"],
                            "description": "影像模态"
                        }
                    },
                    "required": ["subject_id", "modality"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compare_groups",
                "description": "比较各组（自动发现）的数据情况",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    ]


# 全局实例
_data_tools = None

def get_data_tools() -> DataTools:
    """获取全局数据工具实例"""
    global _data_tools
    if _data_tools is None:
        _data_tools = DataTools()
    return _data_tools
