"""
管线调研Agent - URL内容获取工具
用于从指定URL提取文本和图片内容
"""
from langchain.tools import tool
from coze_coding_dev_sdk.fetch import FetchClient
from coze_coding_utils.runtime_ctx.context import new_context


@tool
def fetch_url_content(url: str) -> str:
    """
    从指定URL获取完整的文本内容。

    参数:
        url: 目标网页URL

    返回:
        提取的文本内容，格式化为结构化输出
    """
    ctx = new_context(method="fetch.url")
    client = FetchClient(ctx=ctx)

    try:
        response = client.fetch(url=url)

        if response.status_code != 0:
            return f"获取内容失败: {response.status_message}"

        # 提取纯文本内容
        text_parts = []
        for item in response.content:
            if item.type == "text":
                text_parts.append(item.text)

        full_text = "\n".join(text_parts)

        result = f"""## {response.title}

来源: {response.url}
抓取时间: {response.publish_time or '未知'}

### 内容:

{full_text}"""

        return result

    except Exception as e:
        return f"获取内容出错: {str(e)}"


@tool
def fetch_url_images(url: str) -> str:
    """
    从指定URL获取所有图片的URL列表。

    参数:
        url: 目标网页URL

    返回:
        图片URL列表
    """
    ctx = new_context(method="fetch.url")
    client = FetchClient(ctx=ctx)

    try:
        response = client.fetch(url=url)

        if response.status_code != 0:
            return f"获取图片失败: {response.status_message}"

        images = [
            {
                "url": item.image.display_url,
                "original_url": item.image.image_url,
                "width": item.image.width,
                "height": item.image.height
            }
            for item in response.content if item.type == "image"
        ]

        if not images:
            return f"在 {url} 中未找到图片"

        result = f"在 {response.title} 中找到 {len(images)} 张图片:\n\n"
        for i, img in enumerate(images, 1):
            result += f"[{i}] {img['url']} ({img['width']}x{img['height']})\n"

        return result

    except Exception as e:
        return f"获取图片出错: {str(e)}"
