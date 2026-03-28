---
name: amap_maps
description: 高德地图服务技能，提供位置搜索、路线规划、周边查询等功能
mcp: amap-maps
env:
  - AMAP_MAPS_API_KEY
---

# AMAP Maps MCP Skill

高德地图服务技能，通过 AMAP MCP Server 接入高德地图 API。

## MCP 配置

```json
{
  "mcpServers": {
    "amap-maps": {
      "command": "npx",
      "args": ["-y", "@amap/amap-maps-mcp-server"],
      "env": {
        "AMAP_MAPS_API_KEY": "${AMAP_MAPS_API_KEY}"
      },
      "transport": "stdio"
    }
  }
}
```

## 环境要求

- Node.js 18+
- npx 可用
- `AMAP_MAPS_API_KEY` 环境变量（[高德开放平台](https://lbs.amap.com/) 申请）

## 功能

- 地点搜索：搜索 POI 信息（酒店/餐厅/景点等）
- 路线规划：驾车/步行/公交路线
- 周边查询：查询指定地点周边的设施

## 注意事项

- API Key 需要妥善保管，不要提交到代码仓库
- 高德地图 API 有调用频率限制，请注意合理使用
- 坐标格式为 `经度，纬度`（GCJ-02 坐标系）
