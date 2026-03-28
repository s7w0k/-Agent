"""
Skill 加载器 - 从 skills 目录动态加载技能配置
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown
import yaml

from schemas import SkillConfig


class SkillLoader:
    """Skill 加载器"""

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or Path(__file__).parent.parent / "skills"
        self.skills: Dict[str, SkillConfig] = {}
        self.mcp_configs: Dict[str, Dict[str, Any]] = {}

    def load_all_skills(self) -> Dict[str, SkillConfig]:
        """加载所有技能文件"""
        self.skills = {}

        # 加载主技能文件
        main_skill = self.skills_dir / "travel_plan.md"
        if main_skill.exists():
            config = self._parse_skill_file(main_skill)
            if config:
                self.skills[config.name] = config

        # 加载 reference 目录下的技能
        ref_dir = self.skills_dir / "reference"
        if ref_dir.exists():
            for md_file in ref_dir.glob("*.md"):
                config = self._parse_skill_file(md_file)
                if config:
                    self.skills[config.name] = config

        return self.skills

    def _parse_skill_file(self, file_path: Path) -> Optional[SkillConfig]:
        """解析技能文件"""
        content = file_path.read_text(encoding="utf-8")
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)

        if not frontmatter_match:
            return None

        metadata = yaml.safe_load(frontmatter_match.group(1)) or {}
        mcp_config = self._extract_mcp_config(content)

        return SkillConfig(
            name=metadata.get("name", file_path.stem),
            description=metadata.get("description", ""),
            mcp=metadata.get("mcp"),
            mcp_servers=metadata.get("mcp_servers", []),
            env=metadata.get("env", []),
            mcp_config=mcp_config,
            file_path=file_path,
            full_content=content,
        )

    def _extract_mcp_config(self, content: str) -> Optional[Dict[str, Any]]:
        """从内容中提取 MCP 配置"""
        pattern = r"##\s*MCP\s*配置\s*\n```json\s*\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    def get_mcp_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取 MCP 服务器配置"""
        if not self.skills:
            self.load_all_skills()

        self.mcp_configs = {}

        for skill in self.skills.values():
            if skill.mcp_config and "mcpServers" in skill.mcp_config:
                self.mcp_configs.update(skill.mcp_config["mcpServers"])
            elif skill.mcp:
                self.mcp_configs[skill.mcp] = self._get_default_mcp_config(skill.mcp)

            for server in skill.mcp_servers:
                if server not in self.mcp_configs:
                    self.mcp_configs[server] = self._get_default_mcp_config(server)

        self._apply_env_vars()
        return self.mcp_configs

    def _get_default_mcp_config(self, mcp_name: str) -> Dict[str, Any]:
        """获取默认 MCP 配置"""
        defaults = {
            "12306-mcp": {
                "command": "npx",
                "args": ["-y", "12306-mcp"],
                "transport": "stdio",
            },
            "amap-maps": {
                "command": "npx",
                "args": ["-y", "@amap/amap-maps-mcp-server"],
                "transport": "stdio",
                "env": {"AMAP_MAPS_API_KEY": "${AMAP_MAPS_API_KEY}"},
            },
        }
        return defaults.get(
            mcp_name,
            {
                "command": "npx",
                "args": ["-y", mcp_name],
                "transport": "stdio",
            },
        )

    def _apply_env_vars(self) -> None:
        """应用环境变量"""
        for config in self.mcp_configs.values():
            if "env" in config:
                for key, value in config["env"].items():
                    if (
                        isinstance(value, str)
                        and value.startswith("${")
                        and value.endswith("}")
                    ):
                        env_var = value[2:-1]
                        config["env"][key] = os.getenv(env_var, "")

    def generate_system_prompt(self) -> str:
        """生成系统提示词"""
        if not self.skills:
            self.load_all_skills()

        skills_meta = "\n".join(
            [
                f"### {i}. {skill.name} - {skill.description}"
                for i, skill in enumerate(self.skills.values(), 1)
            ]
        )

        return f"""你是一位专业的旅行规划助手，名为 TravelBot。你擅长为用户制定详细、实用的旅行计划。

## 可用技能 (Available Skills)

{skills_meta}

## 工具调用规则

1. **按需调用**：只在需要获取实时信息时才调用工具
2. **参数完整**：调用工具前确保所有必需参数都已从用户输入中提取
3. **使用完整工具名**：工具名称包含服务器前缀，如 `12306__query_tickets`
4. **结果解释**：将工具返回的专业数据转换为用户友好的建议
5. **错误处理**：工具调用失败时，告知用户并尝试替代方案

## 工作流程

1. **需求收集**：主动询问必要的旅行信息（出发地、目的地、出行日期、预算等）
2. **信息查询**：按需调用 MCP 工具获取实时数据（先查票务，再查地图）
3. **行程验证**：检查计划合理性（到站时间、景点开放时间等）
4. **计划输出**：使用标准格式输出旅行计划

## 输出格式

当用户请求完整旅行计划时，使用以下 Markdown 格式：

```markdown
## 📋 旅行计划

### 🚄 交通信息
- 车次：[车次号]
- 出发/到达时间：[时间]
- 余票情况：[票种及数量]

### 🗺️ 目的地导航
- 路线描述：[从车站到目的地的路线]
- 预计耗时：[时间]

### ✅ 行程验证
- 到站时间：[合理/需注意]
- 其他提醒：[提醒事项]

### 📅 推荐行程
[详细行程安排]

### 💰 预算估算
- 交通：¥[金额]
- 住宿：¥[金额]
- 餐饮：¥[金额]
- 门票：¥[金额]
- 总计：¥[金额]
```

## 注意事项

1. 12306 查询时间建议在 6:00-23:00 之间
2. 火车票预售期为 15 天
3. 预算估算仅供参考
4. 尊重用户偏好，提供个性化建议
"""


# 全局实例
_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """获取全局 SkillLoader 实例"""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
        _loader.load_all_skills()
    return _loader


def get_system_prompt() -> str:
    """获取系统提示词"""
    return get_skill_loader().generate_system_prompt()


# 模块级常量
SYSTEM_PROMPT = get_system_prompt()
