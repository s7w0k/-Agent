"""
Visualization Agent - 可视化路线图 Agent
功能：
调用文生图大模型，将生成的攻略生成可视化攻略风格路线图
支持豆包 Seedream 模型
"""

import asyncio
import json
import base64
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from backend.logger import get_logger
from backend.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class RouteMap(BaseModel):
    """路线图"""
    image_url: Optional[str] = ""  # 图片 URL
    image_base64: Optional[str] = ""  # Base64 编码的图片
    description: str = ""  # 路线描述
    spots_marked: List[str] = []  # 标记的景点


class VisualizationAgent:
    """可视化 Agent - 负责生成路线图和可视化内容"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.SEEDREAM_API_KEY or settings.DEEPSEEK_API_KEY
        self.base_url = settings.SEEDREAM_BASE_URL
        self.llm = None
        self._init_components()

    def _init_components(self):
        """初始化组件"""
        from langchain_deepseek import ChatDeepSeek
        self.llm = ChatDeepSeek(
            model="deepseek-chat",
            api_key=settings.DEEPSEEK_API_KEY,
            temperature=0.7,
        )

        # 初始化豆包图像生成客户端
        self._init_image_generator()

    def _init_image_generator(self):
        """初始化豆包 Seedream 图像生成器"""
        if not self.api_key:
            logger.warning("未配置 SEEDREAM_API_KEY，图像生成功能不可用")
            self.image_client = None
            return

        try:
            # 使用 OpenAI 兼容接口连接豆包
            from openai import AsyncOpenAI
            self.image_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            self.image_model = "doubao-seedream-5-0-260128"  # 豆包 Seedream 5.0 lite
            logger.info(f"豆包图像生成器初始化成功，模型: {self.image_model}")
        except Exception as e:
            logger.warning(f"豆包图像生成器初始化失败: {e}")
            self.image_client = None

    async def process(
        self,
        travel_plan: Any,
        guide_content: Any = None,
        style: str = "travel poster",
    ) -> RouteMap:
        """生成可视化路线图

        Args:
            travel_plan: Planner Agent 生成的计划
            guide_content: Writer Agent 生成的攻略内容
            style: 图像风格

        Returns:
            RouteMap: 包含图像和描述的路线图
        """
        logger.info(f"VisualizationAgent 生成路线图，风格: {style}")

        # 1. 提取关键信息
        route_description = self._extract_route_description(travel_plan)

        # 2. 生成图像提示词
        image_prompt = self._generate_image_prompt(travel_plan, guide_content, style)

        # 3. 生成图像
        if self.image_client:
            route_map = await self._generate_image(image_prompt, route_description)
        else:
            # 无 API 时返回文字描述
            logger.warning("图像生成功能不可用，返回文字路线描述")
            route_map = RouteMap(
                description=route_description,
                spots_marked=[s.name for day in travel_plan.days for s in day.spots],
            )

        logger.info(f"VisualizationAgent 完成，标记 {len(route_map.spots_marked)} 个景点")

        return route_map

    def _extract_route_description(self, plan: Any) -> str:
        """提取路线描述"""
        spots = []
        for day in plan.days:
            for spot in day.spots:
                spots.append(f"{spot.name}({spot.address})" if spot.address else spot.name)

        return " → ".join(spots) if spots else "路线待定"

    def _generate_image_prompt(
        self,
        travel_plan: Any,
        guide_content: Any = None,
        style: str = "travel poster",
    ) -> str:
        """生成图像提示词

        Args:
            travel_plan: 旅行计划
            guide_content: 攻略内容
            style: 图像风格

        Returns:
            英文图像提示词（豆包 Seedream 支持中文，但英文效果更好）
        """
        # 提取目的地和景点
        destination = travel_plan.destination
        all_spots = []
        for day in travel_plan.days:
            for spot in day.spots:
                all_spots.append(spot.name)

        spots_str = ", ".join(all_spots[:6])

        # 风格映射
        style_map = {
            "travel poster": "beautiful travel poster style, vibrant colors, scenic landscape",
            "illustration": "colorful illustration style, cartoon, fun",
            "watercolor": "watercolor painting style, artistic, soft colors",
            "realistic": "photorealistic style, high detail",
            "chinese": "traditional Chinese painting style, ink wash, elegant",
        }

        style_desc = style_map.get(style, style_map["travel poster"])

        # 构建提示词
        prompt = f"""A beautiful travel poster for {destination}, showing a journey through {spots_str}. {style_desc}. The image should include clear route indicators, map elements, and travel icons. Bright colors, clean design, travel guide style, Chinese cultural elements if relevant."""

        return prompt

    async def _generate_image(
        self,
        prompt: str,
        route_description: str,
    ) -> RouteMap:
        """调用豆包 Seedream 生成图像

        Args:
            prompt: 图像提示词
            route_description: 路线描述

        Returns:
            RouteMap: 包含图像的路线图
        """
        try:
            # 调用豆包 Seedream 图像生成 API
            response = await self.image_client.images.generate(
                model=self.image_model,
                prompt=prompt,
                size="2048x2048",
                quality="standard",
                n=1,
            )

            # 解析响应
            if response.data and len(response.data) > 0:
                image_data = response.data[0]
                image_url = getattr(image_data, 'url', None) or getattr(image_data, 'b64_json', None)

                return RouteMap(
                    image_url=image_url if isinstance(image_url, str) else "",
                    image_base64=getattr(image_data, 'b64_json', ""),
                    description=route_description,
                    spots_marked=[],
                )
            else:
                logger.warning("图像生成返回为空")
                return RouteMap(
                    description=route_description,
                    spots_marked=[],
                )

        except Exception as e:
            logger.error(f"豆包图像生成失败: {e}")

            # 返回备选方案
            return RouteMap(
                description=route_description,
                spots_marked=[],
                image_url="",
            )

    async def generate_spot_images(
        self,
        spots: List[Any],
        style: str = "illustration",
    ) -> List[Dict[str, str]]:
        """为每个景点生成单独的图片

        Args:
            spots: 景点列表
            style: 图像风格

        Returns:
            景点图片列表 [{"name": "景点名", "image_url": "图片URL"}]
        """
        if not self.image_client:
            logger.warning("图像生成功能不可用")
            return [{"name": spot.name, "image_url": ""} for spot in spots]

        results = []

        # 风格映射
        style_map = {
            "illustration": "colorful illustration, cartoon style, fun",
            "realistic": "photorealistic, high detail",
            "watercolor": "watercolor painting, soft colors",
            "chinese": "traditional Chinese painting, ink wash",
        }

        style_desc = style_map.get(style, style_map["illustration"])

        for spot in spots:
            prompt = f"""A beautiful illustration of {spot.name}, {style_desc}. Show the landmark characteristic, attractive travel destination, clean background, travel poster style."""

            try:
                response = await self.image_client.images.generate(
                    model=self.image_model,
                    prompt=prompt,
                    size="2048x2048",
                    n=1,
                )

                if response.data and len(response.data) > 0:
                    image_url = getattr(response.data[0], 'url', None) or ""
                else:
                    image_url = ""

                results.append({
                    "name": spot.name,
                    "image_url": image_url,
                })

            except Exception as e:
                logger.warning(f"生成景点图片失败 {spot.name}: {e}")
                results.append({
                    "name": spot.name,
                    "image_url": "",
                })

        return results


# 全局实例
_visualization_agent: Optional["VisualizationAgent"] = None


def get_visualization_agent(api_key: str = None) -> "VisualizationAgent":
    """获取 VisualizationAgent 全局实例"""
    global _visualization_agent
    if _visualization_agent is None:
        from backend.config import get_settings
        settings = get_settings()
        api_key = api_key or settings.SEEDREAM_API_KEY
        _visualization_agent = VisualizationAgent(api_key)
    return _visualization_agent
