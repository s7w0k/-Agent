"""
类型定义模块
提供所有类型的集中定义
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillConfig:
    """技能配置"""

    name: str
    description: str
    mcp: Optional[str] = None
    mcp_servers: List[str] = field(default_factory=list)
    env: List[str] = field(default_factory=list)
    mcp_config: Optional[Dict[str, Any]] = None
    file_path: Optional[Any] = None
    full_content: str = ""
