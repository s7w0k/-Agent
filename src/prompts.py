"""
System Prompt 管理
从 skills 文件动态加载系统提示词
"""

from typing import Optional
from skill_loader import SkillLoader


class SystemPromptManager:
    """系统提示词管理器"""

    _loader: Optional[SkillLoader] = None
    _system_prompt: Optional[str] = None

    @classmethod
    def get_loader(cls) -> SkillLoader:
        """获取 SkillLoader 实例（懒加载）"""
        if cls._loader is None:
            cls._loader = SkillLoader()
            cls._loader.load_all_skills()
        return cls._loader

    @classmethod
    def get_system_prompt(cls) -> str:
        """获取系统提示词（带缓存）"""
        if cls._system_prompt is None:
            cls._system_prompt = cls.get_loader().generate_system_prompt()
        return cls._system_prompt

    @classmethod
    def reload(cls) -> str:
        """重新加载系统提示词"""
        cls._system_prompt = None
        return cls.get_system_prompt()


# 便捷函数
def get_system_prompt() -> str:
    """获取系统提示词"""
    return SystemPromptManager.get_system_prompt()


SYSTEM_PROMPT = get_system_prompt()
