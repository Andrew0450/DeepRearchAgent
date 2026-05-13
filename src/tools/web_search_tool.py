"""
管线调研Agent - 搜索工具
用于从网络获取搜索结果和URL
"""
from langchain.tools import tool
from coze_coding_dev_sdk import SearchClient
from coze_coding_utils.runtime_ctx.context import new_context


@tool
def web_search(query: str, count: int = 5) -> str:
    """
    执行网络搜索，获取相关信息源URL和摘要。

    参数:
        query: 搜索查询关键词，用于查找相关信息
        count: 返回结果数量，默认5条（建议不超过10条）

    返回:
        格式化的搜索结果，包含标题、URL、摘要等信息
    """
    ctx = new_context(method="search.web")
    client = SearchClient(ctx=ctx)

    try:
        response = client.web_search(query=query, count=count)

        if not response.web_items:
            return f"未找到关于「{query}」的相关结果。"

        results = []
        for i, item in enumerate(response.web_items, 1):
            # 限制摘要长度，避免输出过长
            snippet = item.snippet[:200] + "..." if len(item.snippet) > 200 else item.snippet
            result = f"""[{i}] {item.title}
来源: {item.site_name}
URL: {item.url}
摘要: {snippet}"""
            results.append(result)

        return "\n\n".join(results)

    except Exception as e:
        return f"搜索出错: {str(e)}"


@tool
def image_search(query: str, count: int = 5) -> str:
    """
    执行图片搜索，获取相关图片URL。

    参数:
        query: 搜索查询关键词
        count: 返回结果数量，默认5条

    返回:
        格式化的图片搜索结果
    """
    ctx = new_context(method="search.image")
    client = SearchClient(ctx=ctx)

    try:
        response = client.image_search(query=query, count=count)

        if not response.image_items:
            return f"未找到关于「{query}」的相关图片。"

        results = []
        for i, item in enumerate(response.image_items, 1):
            result = f"""[{i}] {item.title or '无标题'}
来源: {item.site_name}
图片URL: {item.image.url}
尺寸: {item.image.width}x{item.image.height}"""
            results.append(result)

        return "\n\n".join(results)

    except Exception as e:
        return f"图片搜索出错: {str(e)}"
