"""
Chart Data Extraction Tool
从图片中提取图表数据（柱状图、折线图、饼图、表格、截图等）
集成 extract-chart-data 技能
"""

import json
import os
import subprocess
import tempfile
from typing import Optional

from langchain.tools import tool
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context


# 默认从 .env 或环境变量读取
API_TOKEN = os.getenv("COZE_API_TOKEN") or os.getenv("COZE_COZE_API_EXTRACT_CHART_DATA_PLACEHOLDER")

SCRIPT_PATH = "/skills/user/extract-chart-data/scripts/extract_chart_data.py"


@tool
def extract_chart_data(
    image_url: str,
    source_url: str,
    topic: str,
    expected_data_type: str = "other"
) -> str:
    """Extract structured data from chart images (bar charts, line charts, pie charts, tables, screenshots).

    Use this tool when:
    - A search result webpage contains chart images that need data extraction
    - You need to verify data from research report screenshots
    - You encounter bar charts, line charts, pie charts, or parameter comparison tables

    Args:
        image_url: The direct URL of the image to analyze (must be publicly accessible)
        source_url: The URL of the webpage where the image was found
        topic: Current research topic for context (e.g., "AI chip market")
        expected_data_type: Type of data expected. Options:
            - "market_size" for market size bar/line charts
            - "market_share" for market share pie/donut charts
            - "competitor_specs" for product parameter comparison tables
            - "policy_timeline" for policy timeline charts
            - "table" for data tables
            - "other" for other chart types (default)

    Returns:
        JSON string with extracted chart data including chart_type, unit, data points, confidence.
    """
    ctx = request_context.get() or new_context(method="extract_chart_data")

    # 确保环境变量已设置
    env = os.environ.copy()
    if API_TOKEN:
        env["COZE_API_TOKEN"] = API_TOKEN

    # 构建命令
    cmd = [
        "python", SCRIPT_PATH,
        "--image-url", image_url,
        "--source-url", source_url,
        "--topic", topic,
        "--expected-data-type", expected_data_type,
        "--language", "zh-CN"
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )

        if result.returncode != 0:
            return json.dumps({
                "status": "failed",
                "error": f"Script execution failed: {result.stderr}",
                "image_url": image_url
            }, ensure_ascii=False)

        # 脚本输出JSON到stdout
        output = result.stdout.strip()
        if not output:
            return json.dumps({
                "status": "failed",
                "error": "Empty output from script",
                "image_url": image_url
            }, ensure_ascii=False)

        # 尝试解析JSON
        try:
            data = json.loads(output)
            # 添加source_url用于溯源
            data["source_url"] = source_url
            data["image_url"] = image_url
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 可能输出包含其他内容，尝试提取JSON部分
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        data = json.loads(line)
                        data["source_url"] = source_url
                        data["image_url"] = image_url
                        return json.dumps(data, ensure_ascii=False, indent=2)
                    except:
                        pass
            return json.dumps({
                "status": "failed",
                "error": f"Cannot parse JSON output: {output[:500]}",
                "image_url": image_url
            }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "status": "failed",
            "error": "Chart extraction timeout (60s)",
            "image_url": image_url
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "failed",
            "error": f"Exception: {str(e)}",
            "image_url": image_url
        }, ensure_ascii=False)
