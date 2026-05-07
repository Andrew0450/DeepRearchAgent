"""
ResearchFlow 调研引擎 Agent
通用调研工作流引擎，支持6个方向（技术方案/行业驱动/产品驱动/学术综述/竞品分析/政策趋势）
输入主题关键词，输出讲稿 + PPT提示词 + 参考文献 + 可下载文档

架构：
- Agent 作为协调器，接收用户输入
- run_research_flow 工具封装完整的 LangGraph 工作流
- 工作流内部包含：主题解析 → 多维度搜索 → 筛选决策 → 方案设计 → 成果输出
- 调研完成后自动生成 DOCX 文档，返回下载链接
"""

import os
import json
import logging
from typing import Annotated

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

from coze_coding_utils.runtime_ctx.context import default_headers
from storage.memory.memory_saver import get_memory_saver
from graphs.research_flow import build_research_flow, ResearchFlowState
from tools.research_search_tool import research_search, deep_search
from tools.report_document_tool import generate_report_document

logger = logging.getLogger(__name__)

LLM_CONFIG = "config/agent_llm_config.json"

# 默认保留最近 20 轮对话 (40 条消息)
MAX_MESSAGES = 40


def _windowed_messages(old, new):
    """滑动窗口: 只保留最近 MAX_MESSAGES 条消息"""
    return add_messages(old, new)[-MAX_MESSAGES:]  # type: ignore


class AgentState(MessagesState):
    messages: Annotated[list[AnyMessage], _windowed_messages]


@wrap_tool_call
def handle_tool_errors(request, handler):
    """工具执行错误处理中间件，避免工具异常阻塞 Agent 循环"""
    try:
        return handler(request)
    except Exception as e:
        logger.error(f"工具执行出错: {str(e)}")
        return ToolMessage(
            content=f"工具执行出错: ({str(e)})，请检查输入参数后重试。",
            tool_call_id=request.tool_call["id"],
        )


def _report_type_for_direction(direction: str) -> str:
    """根据调研方向返回对应的报告类型名称"""
    mapping = {
        "A": "技术方案报告",
        "B1": "行业调研报告",
        "B2": "产品调研报告",
        "C": "学术综述报告",
        "D": "竞品分析报告",
        "E": "政策分析报告",
    }
    return mapping.get(direction.upper(), "调研报告")


def build_agent(ctx=None):
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")

    llm = ChatOpenAI(
        model=cfg["config"].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=cfg["config"].get("temperature", 0.7),
        streaming=True,
        timeout=cfg["config"].get("timeout", 600),
        extra_body={
            "thinking": {
                "type": cfg["config"].get("thinking", "disabled"),
            }
        },
        default_headers=default_headers(ctx) if ctx else {},
    )

    # ------------------------------------------------------------------------
    # 创建闭包工具：run_research_flow（绑定 LLM 实例）
    # ------------------------------------------------------------------------
    def _invoke_research_flow(topic: str, direction: str = "",
                              template: str = "", extra_requirements: str = "") -> str:
        """内部实现：调用 ResearchFlow 工作流，完成后自动生成 DOCX 文档"""
        workflow = build_research_flow(llm=llm)

        initial_state: ResearchFlowState = {
            "topic": topic,
            "direction": direction,
            "template": template,
            "extra_requirements": extra_requirements,
            "parsed": None,
            "collected_data": None,
            "screening_result": None,
            "solution_framework": None,
            "report": None,
            "ppt_prompt": None,
            "references": None,
            "error": None,
        }

        logger.info(f"启动 ResearchFlow: topic={topic}, direction={direction}")

        try:
            result = workflow.invoke(initial_state)

            # 组装完整报告内容（供文档生成使用）
            report_parts = []
            if result.get("report"):
                report_parts.append(result["report"])
            if result.get("ppt_prompt"):
                report_parts.append("\n---\n\n## 📊 PPT 提示词\n\n" + result["ppt_prompt"])
            if result.get("references"):
                report_parts.append("\n---\n\n## 📚 附录：信息来源\n\n" + result["references"])

            full_report_content = "\n".join(report_parts)
            error_msg = result.get("error", "")

            # 自动调用文档生成工具
            doc_url = ""
            try:
                doc_result = generate_report_document.invoke({
                    "report_content": full_report_content,
                    "topic": topic,
                })
                doc_url = doc_result.strip()
                logger.info(f"文档生成成功: {doc_url[:80]}...")
            except Exception as doc_err:
                logger.warning(f"文档生成失败，跳过: {doc_err}")
                doc_url = ""

            # 组装最终输出
            output_parts = [f"# 📋 ResearchFlow 调研成果：{topic}\n"]

            if error_msg:
                output_parts.append(f"\n> ⚠️ 执行过程中出现异常: {error_msg}\n")

            if full_report_content:
                output_parts.append(full_report_content)

            # 追加下载链接
            if doc_url:
                safe_name = topic.replace("/", "_").replace("\\", "_")
                output_parts.append(
                    f"\n---\n\n📎 **完整报告文件（点击下载，链接有效期24小时）：**\n"
                    f"[{safe_name}_调研报告.docx]({doc_url})"
                )

            return "\n".join(output_parts)

        except Exception as e:
            logger.error(f"ResearchFlow 执行失败: {e}")
            return f"ResearchFlow 调研工作流执行失败: {str(e)}。请检查输入主题是否明确，或稍后重试。"

    @tool
    def run_research_flow(topic: str, direction: str = "", template: str = "",
                          extra_requirements: str = "") -> str:
        """启动 ResearchFlow 调研工作流，执行从主题解析到成果输出的完整调研流程。

        支持6个调研方向：
        - A: 技术方案（课设/毕设找可复现方案）
        - B1: 行业驱动（了解行业全貌）
        - B2: 产品驱动（了解产品背后行业）
        - C: 学术综述（写综述/找研究空白）
        - D: 竞品分析（对比同类产品/方案）
        - E: 政策趋势（政策分析/趋势研究）

        当用户给出明确的调研主题后，调用此工具执行调研。
        工具返回完整的调研报告（Markdown格式）、PPT提示词、参考文献，以及可下载的 DOCX 文件链接。

        Args:
            topic: 调研主题关键词，如"人形机器人产业"
            direction: 调研方向代码（A/B1/B2/C/D/E），留空自动判断
            template: 讲稿模板代码（A1/A2/B1/B2/C/D/E），留空使用默认
            extra_requirements: 额外要求，如"重点分析上游供应链"
        """
        return _invoke_research_flow(topic, direction, template, extra_requirements)

    # 收集所有工具
    all_tools = [run_research_flow, research_search, deep_search, generate_report_document]

    return create_agent(
        model=llm,
        system_prompt=cfg.get("sp"),
        tools=all_tools,
        checkpointer=get_memory_saver(),
        state_schema=AgentState,
        middleware=[handle_tool_errors],
    )
