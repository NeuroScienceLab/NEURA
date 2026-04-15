import json
import os
from pathlib import Path

class ConfigManager:
    """
    应用配置管理类
    负责加载、保存和管理应用配置
    """
    
    def __init__(self):
        self.config_dir = Path(os.path.expanduser("~/.encore_agent"))
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            "data_path": r"I:\AGENT-think\data",
            "paper_path": r"I:\AGENT-think\paper",
            "memory_path": r"I:\AGENT-think\memory",
            "theme": "light",
            "log_level": "info",
            "font_size": 12,
            "auto_save": True
        }
        self.config = self.load_config()
    
    def load_config(self):
        """
        加载配置文件
        如果文件不存在，返回默认配置
        """
        try:
            if self.config_file.exists():
                with open(self.config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # 合并默认配置和加载的配置
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            else:
                return self.default_config.copy()
        except Exception as e:
            print(f"加载配置失败: {e}")
            return self.default_config.copy()
    
    def save_config(self):
        """
        保存配置到文件
        """
        try:
            # 确保配置目录存在
            self.config_dir.mkdir(exist_ok=True)
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def get(self, key, default=None):
        """
        获取配置值
        """
        return self.config.get(key, default)
    
    def set(self, key, value):
        """
        设置配置值
        """
        self.config[key] = value
        return self.save_config()
    
    def get_data_path(self):
        """
        获取数据路径
        """
        return self.get("data_path")
    
    def set_data_path(self, path):
        """
        设置数据路径
        """
        return self.set("data_path", path)
    
    def get_paper_path(self):
        """
        获取研究文献路径
        """
        return self.get("paper_path")
    
    def set_paper_path(self, path):
        """
        设置研究文献路径
        """
        return self.set("paper_path", path)

    def get_memory_path(self):
        """
        获取记忆库路径
        """
        return self.get("memory_path")

    def set_memory_path(self, path):
        """
        设置记忆库路径
        """
        return self.set("memory_path", path)

    def get_theme(self):
        """
        获取主题设置
        """
        return self.get("theme", "light")
    
    def set_theme(self, theme):
        """
        设置主题
        """
        return self.set("theme", theme)
    
    def get_log_level(self):
        """
        获取日志级别
        """
        return self.get("log_level", "info")
    
    def set_log_level(self, level):
        """
        设置日志级别
        """
        return self.set("log_level", level)
    
    def get_font_size(self):
        """
        获取字体大小
        """
        return self.get("font_size", 12)
    
    def set_font_size(self, size):
        """
        设置字体大小
        """
        return self.set("font_size", size)
    
    def is_auto_save(self):
        """
        获取自动保存设置
        """
        return self.get("auto_save", True)
    
    def set_auto_save(self, auto_save):
        """
        设置自动保存
        """
        return self.set("auto_save", auto_save)

# 创建全局配置实例
config_manager = ConfigManager()
