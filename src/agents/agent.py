"""
管线调研Agent - 核心逻辑
提供高质量、结构化的调研服务
"""
import os
import json
from typing import Annotated, List, Dict, Any, Optional
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from coze_coding_utils.runtime_ctx.context import default_headers, new_context
from storage.memory.memory_saver import get_memory_saver
from tools.web_search_tool import web_search
from tools.fetch_url_tool import fetch_url_content
from tools.document_generation_tool import generate_docx_report


LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def build_agent(ctx=None):
    """构建管线调研Agent"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg['config'].get('temperature', 0.7),
        streaming=True,
        timeout=cfg['config'].get('timeout', 600),
        extra_body={
            "thinking": {
                "type": cfg['config'].get('thinking', 'enabled')
            }
        },
        default_headers=default_headers(ctx) if ctx else {}
    )

    # 注册工具
    tools = [web_search, fetch_url_content, generate_docx_report]

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=tools,
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
    )


def identify_research_directions(topic: str) -> List[str]:
    """
    根据调研主题识别需要调研的方向

    返回: 方向列表，如 ["B1", "E", "B2", "D", "A", "C"]
    """
    # 调度顺序: B1 → E → B2+D → A/C
    return ["B1", "E", "B2", "D", "A", "C"]


def build_search_query(topic: str, direction: str) -> str:
    """根据调研主题和方向构建搜索查询词"""
    direction_map = {
        "A": f"{topic} 市场规模 增长趋势",
        "B1": f"{topic} 竞争格局 头部玩家 市场份额",
        "B2": f"{topic} 细分赛道 差异化机会 垂直领域",
        "C": f"{topic} 技术路线 专利布局 研发投入",
        "D": f"{topic} 政策环境 监管动态 法规标准",
        "E": f"{topic} 商业模式 盈利分析 收入结构"
    }
    return direction_map.get(direction, f"{topic} {direction}")


def format_research_report(topic: str, results: Dict[str, Any]) -> str:
    """格式化调研报告"""
    from datetime import datetime

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report = f"# {topic} 调研报告\n\n"
    report += f"**生成时间**: {now}\n\n"
    report += "---\n\n"

    direction_names = {
        "A": "市场规模与增长趋势",
        "B1": "竞争格局与头部玩家",
        "B2": "细分赛道与差异化机会",
        "C": "技术路线与专利布局",
        "D": "政策环境与监管动态",
        "E": "商业模式与盈利分析"
    }

    for direction, content in results.items():
        report += f"## {direction}. {direction_names.get(direction, direction)}\n\n"
        report += content.get("summary", "暂无数据")
        report += "\n\n"

    return report


def save_report(topic: str, report: str) -> str:
    """保存报告到文件"""
    import hashlib
    from datetime import datetime

    # 生成安全的文件名
    safe_topic = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in topic)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_topic}_{timestamp}.md"

    filepath = f"/tmp/{filename}"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)

    return filepath


if __name__ == "__main__":
    # 测试Agent构建
    agent = build_agent()
    print("管线调研Agent构建成功!")
    print(f"配置模型: {agent}")
