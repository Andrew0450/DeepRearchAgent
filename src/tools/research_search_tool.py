"""
ResearchFlow 调研搜索引擎工具
提供联网搜索能力，支持多维度调研（技术方案/行业/产品/学术/竞品/政策）
"""

import json
import logging
from langchain.tools import tool
from coze_coding_dev_sdk import SearchClient
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)


def _do_web_search(query: str, count: int = 10, time_range: str = None,
                   sites: str = None, need_content: bool = False) -> str:
    """执行网络搜索的公共逻辑（非 @tool，供多个 tool 复用）"""
    ctx = request_context.get() or new_context(method="research_search")
    client = SearchClient(ctx=ctx)

    try:
        response = client.search(
            query=query,
            search_type="web",
            count=count,
            need_content=need_content,
            need_url=True,
            need_summary=True,
            sites=sites,
            time_range=time_range,
        )

        results = []
        if response.summary:
            results.append(f"## AI 摘要\n{response.summary}\n")

        if response.web_items:
            results.append(f"## 搜索结果（共 {len(response.web_items)} 条）\n")
            for i, item in enumerate(response.web_items, 1):
                entry = {
                    "序号": i,
                    "标题": item.title or "无标题",
                    "来源": item.site_name or "未知",
                    "URL": item.url or "",
                    "摘要": item.snippet or "",
                    "发布时间": item.publish_time or "未知",
                    "权威等级": item.auth_info_level or "未知",
                    "权威描述": item.auth_info_des or "",
                }
                if item.content:
                    # 截取前 500 字符避免过长
                    entry["正文片段"] = item.content[:500]
                results.append(json.dumps(entry, ensure_ascii=False))

        if not results:
            return "未找到相关结果，建议调整关键词后重试。"

        return "\n".join(results)

    except Exception as e:
        logger.error(f"搜索执行失败: query={query}, error={str(e)}")
        return f"搜索执行失败: {str(e)}"


@tool
def research_search(query: str, direction: str = "综合", time_range: str = "") -> str:
    """通用调研搜索工具。根据调研方向和关键词进行网络搜索，返回结构化搜索结果。

    Args:
        query: 搜索关键词，尽量具体明确
        direction: 调研方向，可选值: 技术方案(A)/行业+市场(B1)/学术综述(C)/竞品对比(D)/政策分析(E)/综合，默认"综合"
        time_range: 时间范围过滤，可选: 1d(近1天)/1w(近1周)/1m(近1月)/1y(近1年)，留空不过滤
    """
    # 根据方向调整搜索策略
    direction_sites = {
        "学术综述": "scholar.google.com,cnki.net,arxiv.org,semanticscholar.org",
        "政策趋势": "gov.cn,mof.gov.cn,ndrc.gov.cn,stats.gov.cn",
    }

    sites = direction_sites.get(direction)

    return _do_web_search(
        query=query,
        count=10,
        time_range=time_range if time_range else None,
        sites=sites,
        need_content=False,
    )


@tool
def deep_search(query: str, sites: str = "", time_range: str = "") -> str:
    """深度搜索工具。获取搜索结果的正文内容，适合需要详细信息的场景。

    Args:
        query: 搜索关键词
        sites: 限定搜索站点，多个用逗号分隔，如 "arxiv.org,github.com"，留空不限
        time_range: 时间范围过滤，可选: 1d/1w/1m/1y，留空不过滤
    """
    return _do_web_search(
        query=query,
        count=5,
        time_range=time_range if time_range else None,
        sites=sites if sites else None,
        need_content=True,
    )
