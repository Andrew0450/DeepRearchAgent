"""
管线调研Agent - 文档生成工具
用于将Markdown内容转换为DOCX格式的报告
"""
from langchain.tools import tool
from coze_coding_dev_sdk import DocumentGenerationClient, DOCXConfig


@tool
def generate_docx_report(markdown_content: str, report_title: str) -> str:
    """
    将Markdown格式的报告内容转换为DOCX文件，并生成下载链接。

    参数:
        markdown_content: Markdown格式的报告内容
        report_title: 报告标题（用于文件名，必须为英文）

    返回:
        DOCX文件的下载链接
    """
    config = DOCXConfig(
        font_name="Noto Sans CJK SC",
        font_size=11,
        top_margin=0.75,
        bottom_margin=0.75,
        left_margin=0.75,
        right_margin=0.75
    )
    client = DocumentGenerationClient(docx_config=config)

    try:
        # 生成安全的文件名
        safe_title = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in report_title)
        url = client.create_docx_from_markdown(markdown_content, safe_title)
        return url
    except Exception as e:
        return f"生成文档失败: {str(e)}"
