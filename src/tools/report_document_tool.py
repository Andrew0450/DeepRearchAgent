"""报告文档生成工具 - 将调研报告转换为 DOCX 文件"""
import logging
import re
import time
from typing import Optional

from langchain.tools import tool
from coze_coding_dev_sdk import DocumentGenerationClient, DOCXConfig
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)


def _to_safe_filename(text: str) -> str:
    """将中文主题转换为安全的英文文件名"""
    import hashlib

    # 基础英文词映射
    topic_map = {
        "人形机器人": "humanoid_robot",
        "智能眼镜": "smart_glasses",
        "低空经济": "low_altitude_economy",
        "新能源汽车": "nev",
        "人工智能": "ai",
        "大模型": "llm",
        "自动驾驶": "autonomous_driving",
        "半导体": "semiconductor",
    }

    # 尝试匹配已知主题
    for cn, en in topic_map.items():
        if cn in text:
            return en

    # 未知主题：用拼音首字母 + 哈希
    slug = re.sub(r"[^\w]", "_", text)[:20]
    safe = re.sub(r"_+", "_", slug).strip("_")
    if safe.encode("utf-8").isascii():
        return safe
    # 非ASCII：用时间戳哈希
    ts = int(time.time())
    return f"report_{ts % 100000:05d}"


@tool
def generate_report_document(
    report_content: str,
    topic: str,
    doc_format: str = "docx",
) -> str:
    """将调研报告内容生成可下载的文档文件。

    适用场景：用户需要将调研报告保存为本地文件时使用。
    工作流程：先调用 run_research_flow 生成完整报告内容，再调用本工具生成文档。

    参数:
        report_content: 调研报告的完整 Markdown 内容（从 run_research_flow 返回值中提取第一部分）
        topic: 调研主题，用于生成文件名（如"人形机器人产业"）
        doc_format: 文档格式，默认为 "docx"（也支持 "pdf"）

    返回:
        文件下载链接（24小时内有效），返回格式：'📎 文档已生成：[格式]下载 | 报告标题 | 下载链接'
    """
    ctx = request_context.get() or new_context(method="generate_report_document")

    client = DocumentGenerationClient(
        docx_config=DOCXConfig(
            font_name="Noto Sans CJK SC",
            font_size=11,
            top_margin=0.75,
            bottom_margin=0.75,
            left_margin=0.75,
            right_margin=0.75,
        )
    )

    # 清理报告内容：移除工具返回标记
    cleaned = report_content.strip()
    # 移除顶部的报告标题行
    cleaned = re.sub(r"^#+\s*📋\s*ResearchFlow.*?\n+", "", cleaned, flags=re.IGNORECASE)
    # 移除 Markdown 分隔符
    cleaned = re.sub(r"^---+\s*$", "", cleaned, flags=re.MULTILINE)
    # 移除"第一部分：调研报告全文"等章节标记
    cleaned = re.sub(
        r"^#{1,3}\s*(?:第一|第二|第三|第四|第五|第六)\s*部分[：:].*?$",
        "",
        cleaned,
        flags=re.MULTILINE,
    )
    cleaned = cleaned.strip()

    # 生成安全的文件名
    filename = _to_safe_filename(topic)

    try:
        url = client.create_docx_from_markdown(cleaned, filename)
        return f"📎 文档已生成：[DOCX下载]({url}) | 主题：{topic}"
    except Exception as e:
        logger.error(f"文档生成失败: {e}")
        return f"❌ 文档生成失败：{str(e)}"
