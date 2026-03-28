"""
知识库初始化脚本
用于添加示例知识数据（小红书、全网）
"""

import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.rag.retriever import get_retriever


# 小红书旅行攻略示例
XIAOHONGSHU_EXAMPLES = [
    {
        "title": "上海到苏州一日游攻略",
        "content": """上海到苏州超级方便！坐高铁只要25分钟就能到。

🚄 交通：
- 上海虹桥站 → 苏州站/苏州园区站
- 票价约40-50元
- 建议提前12306买票

🏯 推荐景点：
1. 拙政园（必去！门票70元）
2. 平江路历史街区（免费）
3. 苏州博物馆（免费需预约）
4. 山塘街（夜景超美）

🍜 美食推荐：
- 平江路：桂花糕、酒酿饼、蟹壳黄
- 观前街：松鹤楼、得月楼

⚠️ 小tips：
- 拙政园早上8点开门，建议早点去人少
- 苏州博物馆周一闭馆
- 带好身份证，大部分景点需要扫码""",
        "source": "小红书"
    },
    {
        "title": "北京故宫游览全攻略",
        "content": """故宫游览最强攻略！跟着这篇走不踩雷～

🎫 门票：
- 成人60元（旺季）
- 珍宝馆、钟表馆需额外购票
- 提前在故宫官网预约！

🗺️ 游览路线：
推荐路线：午门 → 太和殿 → 中和殿 → 保和殿 → 乾清宫 → 交泰殿 → 坤宁宫 → 御花园 → 神武门

⏰ 时间安排：
- 建议游玩3-4小时
- 早上8:30开门，尽量9点前进

📸 拍照点：
- 午门入口
- 太和殿广场
- 御花园
- 景山公园俯瞰故宫（需要额外门票）

⚠️ 注意：
- 带身份证入园
- 不能带打火机
- 穿舒适的鞋！""",
        "source": "小红书"
    },
]

# 全网旅行知识示例
WEB_EXAMPLES = [
    {
        "title": "火车票购买全攻略",
        "content": """12306购票完全指南

📅 预售时间：
- 开车前15天开始预售
- 每天6:00-23:00开放
- 侯补购票：开车前48小时

🎫 票种说明：
- G/D字头：高铁/动车
- C字头：城际列车
- Z/T/K字头：普速列车

💺 座位类型：
- 商务座：一等座 → 二等座
- 卧铺：软卧 → 硬卧

⚠️ 注意事项：
1. 同一个账户一天取消3次当天不能购票
2. 改签免费机会只有一次
3. 开车前48小时可改签任意有余票的车次

🔧 捡漏技巧：
- 开车前48小时是退票高峰期
- 凌晨12306会释放未支付的票
- 候补购票成功率很高""",
        "source": "全网"
    },
    {
        "title": "旅行保险购买指南",
        "content": """旅行保险选购指南

🛡️ 为什么要买旅行保险：
- 航班延误/取消保障
- 意外伤害医疗
- 行李丢失赔偿
- 紧急救援服务

📋 主要类型：
1. 境内游：短途旅行险
2. 境外游：旅游意外险
3. 高原游：高原反应险

💰 价格参考：
- 境内1-3天：10-30元
- 境内7天：30-80元
- 境外10天：100-300元

🔍 选购要点：
1. 看保障范围
2. 看免责条款
3. 看理赔速度
4. 选正规保险公司

⚠️ 特别注意：
- 攀岩、潜水等高风险活动需要专项保险
- 既往病史通常不保
- 保留好所有单据和凭证""",
        "source": "全网"
    },
]


async def init_knowledge_base():
    """初始化知识库"""
    retriever = get_retriever()

    print("=" * 50)
    print("开始初始化知识库...")
    print("=" * 50)

    # 添加小红书内容
    print("\n[小红书] 添加内容...")
    for i, item in enumerate(XIAOHONGSHU_EXAMPLES, 1):
        print(f"  [{i}/{len(XIAOHONGSHU_EXAMPLES)}] 添加: {item['title']}")
        await retriever.add_document(
            content=item["content"],
            title=item["title"],
            source=item["source"]
        )

    # 添加全网内容
    print("\n[全网] 添加内容...")
    for i, item in enumerate(WEB_EXAMPLES, 1):
        print(f"  [{i}/{len(WEB_EXAMPLES)}] 添加: {item['title']}")
        await retriever.add_document(
            content=item["content"],
            title=item["title"],
            source=item["source"]
        )

    # 打印统计
    stats = retriever.get_knowledge_stats()
    print("\n" + "=" * 50)
    print("知识库初始化完成!")
    print(f"   集合名称: {stats.get('collection_name')}")
    print(f"   文档数量: {stats.get('document_count')}")
    print(f"   存储路径: {stats.get('persist_dir')}")
    print("=" * 50)

    # 测试检索
    print("\n测试检索...")
    test_results = await retriever.retrieve("上海到苏州", top_k=2)
    print(f"检索到 {len(test_results)} 条结果")


if __name__ == "__main__":
    asyncio.run(init_knowledge_base())
