"""工具标准规范加载器"""
import json
from pathlib import Path

STANDARDS_DIR = Path(__file__).parent


def load_tool_standard(tool_name: str, operation: str = None) -> dict:
    """加载指定工具的标准规范"""
    spec_file = STANDARDS_DIR / f"{tool_name}.json"
    if not spec_file.exists():
        return {}
    with open(spec_file, "r", encoding="utf-8") as f:
        spec = json.load(f)
    if operation and "operations" in spec:
        return spec["operations"].get(operation, {})
    return spec


def load_review_criteria() -> dict:
    """加载通用审查标准"""
    return load_tool_standard("review_criteria")


def list_available_standards() -> dict:
    """列出所有可用的标准规范及其操作"""
    result = {}
    for f in STANDARDS_DIR.glob("*.json"):
        name = f.stem
        with open(f, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
        ops = list(spec.get("operations", {}).keys())
        if ops:
            result[name] = ops
    return result
