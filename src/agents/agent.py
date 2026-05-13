"""
管线调研Agent - 基于LangGraph的严格工作流
遵循SOP v3.1，每个方向有独立节点和模板检查
"""
import os
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from coze_coding_utils.runtime_ctx.context import default_headers
from tools.web_search_tool import web_search, image_search
from tools.fetch_url_tool import fetch_url_content, fetch_url_images
from tools.document_generation_tool import generate_docx_report


LLM_CONFIG = "config/agent_llm_config.json"


# ============ 状态定义 ============

class ResearchState(dict):
    """工作流状态"""
    topic: str = ""
    directions: List[str] = []
    current_direction: str = ""
    b1_result: str = ""
    e_result: str = ""
    b2_result: str = ""
    d_result: str = ""
    ac_result: str = ""
    completed_direction: str = ""
    next_direction: str = ""
    market_size_data: str = ""
    policy_data: str = ""
    competitor_data: str = ""
    full_report: str = ""
    source_quality_report: str = ""
    docx_url: str = ""
    summary: str = ""


# ============ LLM初始化 ============

def get_llm(ctx=None):
    """获取LLM实例"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", "/workspace/projects")
    config_path = os.path.join(workspace_path, LLM_CONFIG)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    api_key = os.getenv("COZE_WORKLOAD_IDENTITY_API_KEY")
    base_url = os.getenv("COZE_INTEGRATION_MODEL_BASE_URL")
    
    return ChatOpenAI(
        model=cfg['config'].get("model"),
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
        timeout=cfg['config'].get('timeout', 600),
        extra_body={"thinking": {"type": "enabled"}},
        default_headers=default_headers(ctx) if ctx else {}
    )


# ============ 搜索执行辅助函数 ============

URL_RE = re.compile(r"https?://[^\s\]\)>\"]+")
CRITICAL_NUMBER_RE = re.compile(
    r"(?<!\d)(?:\d+(?:\.\d+)?\s*(?:%|％|亿元|亿美元|万亿元|万台|台|人|家|倍|年|月|日)|CR[35]\s*[:：]?\s*\d+(?:\.\d+)?%?)"
)

REFERENCE_FILES = {
    "B1": ["B1-行业定位.md", "B1+B2行业调研报告.md"],
    "B2": ["B2-市场量化.md", "B1+B2行业调研报告.md"],
    "D": ["D-竞品对比.md", "D-竞品调研报告.md"],
    "E": ["E-政策环境.md"],
    "A": ["AC-技术深挖.md"],
    "C": ["AC-技术深挖.md"],
}

AUTHORITY_SOURCES = {
    "industry_cn": "国家统计局、工信部、发改委、商务部、行业协会、上市公司年报/招股书、艾瑞、赛迪、头豹、亿欧、中商产业研究院、前瞻产业研究院",
    "industry_global": "IDC、Gartner、Counterpoint、Omdia、Statista、Euromonitor、World Bank、OECD、IMF、Our World in Data",
    "policy": "gov.cn、国务院、发改委、工信部、财政部、商务部、新华社、人民日报、地方政府官网、行业监管部门",
    "academic": "Semantic Scholar、Google Scholar、arXiv、IEEE、ACM、Nature、Science、CNKI、专利数据库",
    "competitor": "产品官网、开发者文档、GitHub、上市公司公告、招股书、Gartner Magic Quadrant、IDC MarketScape、权威测评媒体、电商官方旗舰店",
}


def _extract_urls(text: str) -> List[str]:
    """从搜索结果或报告文本中提取 URL，并保持顺序去重。"""
    urls = []
    seen = set()
    for raw_url in URL_RE.findall(text or ""):
        url = raw_url.rstrip("。；;，,.)]")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def load_reference_outline(direction: str, max_lines: int = 80) -> str:
    """读取 assets 中的目标效果样例标题，作为结构和粒度参考。"""
    workspace_path = os.getenv("COZE_WORKSPACE_PATH", os.getcwd())
    assets_dir = os.path.join(workspace_path, "assets")
    outlines = []
    for filename in REFERENCE_FILES.get(direction, []):
        path = os.path.join(assets_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            headings = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        headings.append(stripped)
                    if len(headings) >= max_lines:
                        break
            if headings:
                outlines.append(f"### {filename}\n" + "\n".join(headings))
        except Exception:
            continue
    return "\n\n".join(outlines) or "暂无目标样例结构。"


def build_authoritative_queries(topic: str, direction: str) -> List[str]:
    """按方向生成更密集的权威源查询。"""
    common = [
        f"{topic} 官方 白皮书 报告 2024 2025 2026",
        f"{topic} 数据 来源 市场规模 增速 2024 2025 2026",
    ]
    direction_queries = {
        "B1": [
            f"{topic} 行业报告 市场规模 CAGR 产业链 竞争格局 IDC Gartner Statista Omdia",
            f"{topic} 市场规模 中国 全球 出货量 销量 Counterpoint IDC Omdia",
            f"{topic} 产业链 上游 中游 下游 价值分布 上市公司 年报 招股书",
            f"{topic} 头部企业 市场份额 CR3 CR5 年报 招股书",
            f"{topic} 行业痛点 发展趋势 白皮书 艾瑞 赛迪 头豹 亿欧",
            f"{topic} site:stats.gov.cn OR site:miit.gov.cn OR site:ndrc.gov.cn",
            f"{topic} annual report prospectus market share revenue",
            f"{topic} market size forecast CAGR IDC Gartner Statista Counterpoint Omdia",
        ],
        "B2": [
            f"{topic} TAM SAM SOM 市场规模 测算 2024 2025",
            f"{topic} 产品市场规模 父类市场 核心器件 市场规模",
            f"{topic} 应用场景 用户画像 消费者调研 渠道 数据",
            f"{topic} 细分市场 按产品 按地区 按用户 规模",
            f"{topic} 出货量 销量 客单价 渗透率 IDC Counterpoint Omdia",
            f"{topic} 产业链 成本结构 BOM 价值分布 年报 招股书",
            f"{topic} market size TAM SAM SOM shipment ASP penetration",
            f"{topic} consumer survey user persona application scenarios report",
        ],
        "D": [
            f"{topic} 竞品 对比 功能 参数 价格 官网",
            f"{topic} vs 竞品 评测 benchmark 性能 成本",
            f"{topic} 产品官网 参数 规格 定价 用户评价",
            f"{topic} GitHub open source alternatives benchmark",
            f"{topic} Gartner Magic Quadrant IDC MarketScape Forrester Wave",
            f"{topic} 电商 官方旗舰店 价格 参数 评价",
            f"{topic} competitors comparison pricing features benchmark review",
        ],
        "E": [
            f"{topic} 政策 site:gov.cn 2024 2025 2026",
            f"{topic} 国务院 政策 文件 通知 规划",
            f"{topic} 发改委 工信部 财政部 商务部 政策",
            f"{topic} 新华社 人民日报 领导人 讲话 会议",
            f"{topic} 地方政府 试点 示范 项目 政策",
            f"{topic} 监管 法规 标准 合规 风险",
            f"{topic} policy regulation government strategy 2024 2025",
        ],
        "A": [
            f"{topic} 技术方案 对比 原理 白皮书",
            f"{topic} survey review arxiv IEEE ACM 2023 2024 2025",
            f"{topic} benchmark dataset experiment result paper",
            f"{topic} 开源实现 GitHub 技术路线",
            f"{topic} 专利 技术路线 发展趋势",
            f"{topic} technical architecture comparison benchmark latest research",
        ],
        "C": [
            f"{topic} 综述 survey review arxiv IEEE ACM 2023 2024 2025",
            f"{topic} research gap future direction benchmark dataset",
            f"{topic} Semantic Scholar citation influential paper",
            f"{topic} Google Scholar high cited papers latest progress",
            f"{topic} CNKI 综述 研究进展 研究空白",
        ],
    }
    return _dedupe_keep_order(common + direction_queries.get(direction, []))


def _fetch_key_sources(search_text: str, max_urls: int = 8) -> str:
    """抓取搜索结果中的关键网页正文，作为搜索摘要之外的证据来源。"""
    urls = _extract_urls(search_text)[:max_urls]
    if not urls:
        return "未从搜索结果中提取到可抓取 URL。"

    fetched = []
    for url in urls:
        try:
            content = fetch_url_content.invoke({"url": url})
            # 控制单个网页片段长度，避免后续提示词被单页吞掉。
            fetched.append(f"### 抓取来源: {url}\n{str(content)[:4000]}\n")
        except Exception as e:
            fetched.append(f"### 抓取来源: {url}\n抓取失败: {str(e)}\n")
    return "\n".join(fetched)


def _collect_image_sources(search_text: str, topic: str, direction: str, max_urls: int = 5) -> str:
    """收集网页内图片和图片搜索结果；当前只列出图片，OCR需平台插件。"""
    parts = []
    for url in _extract_urls(search_text)[:max_urls]:
        try:
            images = fetch_url_images.invoke({"url": url})
            parts.append(f"### 网页图片来源: {url}\n{str(images)[:2500]}\n")
        except Exception as e:
            parts.append(f"### 网页图片来源: {url}\n图片提取失败: {str(e)}\n")

    if direction in {"B1", "B2", "D", "E"}:
        image_queries = [
            f"{topic} 市场规模 图表",
            f"{topic} 市场份额 图表",
            f"{topic} 产业链 图谱",
        ]
        if direction == "D":
            image_queries = [f"{topic} 产品参数 对比 图", f"{topic} 竞品对比 表格"]
        elif direction == "E":
            image_queries = [f"{topic} 政策 路线图", f"{topic} 监管 时间表"]

        for query in image_queries:
            try:
                result = image_search.invoke({"query": query, "count": 5})
                parts.append(f"### 图片搜索: {query}\n{str(result)[:2500]}\n")
            except Exception as e:
                parts.append(f"### 图片搜索: {query}\n图片搜索失败: {str(e)}\n")

    if not parts:
        return "未提取到图片来源。"
    return "\n".join(parts) + "\n\n提示：当前工具只能提供图片URL，不能直接识别图片中的图表数据；如需使用图片里的数值，必须通过OCR/视觉插件提取后再引用，并标注“数据来源于图片，经OCR/视觉识别提取，需人工核验”。"


def execute_searches(queries: List[str], topic: str = "", direction: str = "", fetch_pages: bool = True, collect_images: bool = True) -> str:
    """执行多轮搜索，随后抓取关键URL正文并整合结果。"""
    results = []
    for query in _dedupe_keep_order(queries):
        try:
            result = web_search.invoke({"query": query, "count": 10})
            results.append(f"### 搜索: {query}\n{result}\n")
        except Exception as e:
            results.append(f"### 搜索: {query}\n搜索失败: {str(e)}\n")
    search_text = "\n".join(results)
    if not fetch_pages:
        return search_text

    fetched_text = _fetch_key_sources(search_text)
    image_text = _collect_image_sources(search_text, topic, direction) if collect_images else ""
    return f"{search_text}\n\n## 关键网页正文抓取\n{fetched_text}\n\n## 图片/图表线索\n{image_text}"


def build_source_quality_report(report: str) -> str:
    """生成轻量质量审计，提示来源、估算、时效和缺失风险。"""
    urls = _extract_urls(report)
    lines = [
        "## 质量审计",
        "",
        f"- 来源URL数量：{len(urls)}",
    ]

    if urls:
        lines.append("- 已检测到可追溯链接，仍建议人工复核关键市场规模、份额、政策原文与官网一致性。")
    else:
        lines.append("- ⚠️ 未检测到URL。报告不满足SOP的来源可溯要求，需要补充公开来源。")

    vague_terms = ["据统计", "业内人士", "公开资料显示", "可能", "预计", "估计"]
    found_vague = [term for term in vague_terms if term in report]
    if found_vague:
        lines.append(f"- ⚠️ 检测到模糊表述：{', '.join(found_vague)}。如涉及关键数据，应替换为明确来源或标注估算。")

    if "⚠️" in report or "估算" in report:
        has_formula = any(marker in report for marker in ["公式", "=", "×", "*"])
        if has_formula:
            lines.append("- 已检测到估算相关表述和公式痕迹，请人工复核参数来源与假设是否齐全。")
        else:
            lines.append("- ⚠️ 检测到估算相关表述，但未检测到明显公式，需要补充估算公式、参数来源和假设。")

    missing_markers = ["未检索到公开数据", "需补充", "来源待验证"]
    found_missing = [marker for marker in missing_markers if marker in report]
    if found_missing:
        lines.append(f"- 待补充项：报告中包含 {', '.join(found_missing)}，正式使用前应继续检索。")

    suspicious_lines = []
    for line in report.splitlines():
        if CRITICAL_NUMBER_RE.search(line) and "http" not in line and "未检索到公开数据" not in line:
            suspicious_lines.append(line.strip())
        if len(suspicious_lines) >= 5:
            break
    if suspicious_lines:
        lines.append("- ⚠️ 以下含数字的句子未在同一行检测到URL，建议检查是否需要补来源：")
        for item in suspicious_lines:
            lines.append(f"  - {item[:160]}")

    return "\n".join(lines)


def _complete_single_direction(state: ResearchState) -> None:
    """完成当前方向后停止本轮运行，并记录下一建议方向。"""
    directions = state.get("directions", [])
    current = state.get("current_direction", "")
    if not directions or not current:
        state["current_direction"] = ""
        state["next_direction"] = ""
        return

    try:
        idx = directions.index(current)
    except ValueError:
        state["current_direction"] = ""
        state["next_direction"] = ""
        return

    state["completed_direction"] = current
    state["next_direction"] = directions[idx + 1] if idx + 1 < len(directions) else ""
    state["current_direction"] = ""
    # 从队列中移除已完成的方向，避免下次重复执行
    state["directions"] = directions[idx + 1:] if idx + 1 < len(directions) else []


def _direction_to_node(direction: str) -> str:
    """将 SOP 方向映射到实际工作流节点。A/C 当前共用技术学术节点。"""
    return "node_ac" if direction in {"A", "C"} else f"node_{direction.lower()}"


# ============ 节点1：方向识别 ============

DIRECTION_IDENTIFY_PROMPT = """你是调研方向识别专家。

根据用户输入的调研主题，判断需要执行哪些调研方向。

可用方向：
- B1: 行业驱动型调研（宏观视角：行业结构、产业链、规模、格局、痛点）
- E: 政策与趋势（制度视角：政策梳理、解读、影响方向、落地进展）
- B2: 产品驱动型调研（中观视角：市场量化、应用场景、产业链价值、痛点）
- D: 竞品/产品分析（微观视角：竞品对比、优劣势、推荐方案、迁移路径）
- A: 技术方案调研（技术视角：技术原理、方案对比、实验效果、创新点）
- C: 学术综述（学术视角：方法演进、研究空白、未来方向）

判断逻辑：
1. 如果用户明确指定方向（如“方向B2”“只做D”“继续方向E”），directions 只返回该方向
2. 如果主题是一个具体技术/方案且未指定方向 → 推荐顺序 A + B1 + B2 + D
3. 如果主题是一个行业名称且未指定方向 → 推荐顺序 B1 + E + B2 + D
4. 如果主题是一个具体产品且未指定方向 → 推荐顺序 B2 + D + B1 + E
5. 如果主题需要写综述/找研究空白且未指定方向 → 推荐顺序 C + A
6. 如果主题涉及政策驱动且未指定方向 → 推荐顺序 E + B1 + B2

重要执行策略：
- 本工作流每次只执行 directions 中的第一个方向，以保留搜索次数和报告精度，避免一次性全管线超时
- 后续方向由用户在下一轮对话中明确要求继续

输出格式（严格JSON，不要有任何其他内容）：
{
    "directions": ["B1", "E", "B2", "D"],
    "reasoning": "判断理由"
}"""

def node_identify_directions(state: ResearchState) -> ResearchState:
    """识别调研方向"""
    llm = get_llm()
    topic = state.get("topic", "")
    
    # 如果topic为空或未指定，设置默认值
    if not topic or topic == "未指定主题":
        # 尝试从params获取
        if state.get("params"):
            topic = str(state["params"])
            state["topic"] = topic
        else:
            # 设置一个默认方向
            state["directions"] = ["B1", "E", "B2", "D"]
            state["current_direction"] = "B1"
            return state

    # 优先解析 "继续调研 XXX 方向 Y" / "下一步方向 Y" / "接着执行 Y" 等显式指令
    continue_match = re.search(
        r'(?:继续|下一步|接着)(?:调研|分析|执行)?\s*[：:]?\s*(.+?)(?:\s+方向\s*([A-E0-9/]+))?$',
        topic, re.IGNORECASE
    )
    if continue_match:
        continued_topic = continue_match.group(1).strip()
        continued_direction = continue_match.group(2).strip().upper() if continue_match.group(2) else ""
        if continued_direction in {"A", "B1", "B2", "C", "D", "E"}:
            state["topic"] = continued_topic
            state["directions"] = [continued_direction]
            state["current_direction"] = continued_direction
            return state

    # 兜底：单独匹配 "方向X" / "只做X" / "继续X" 等简短指令（不修改 topic）
    explicit_match = re.search(r"(?:方向|只做|继续|执行|调研)\s*(B1|B2|A|C|D|E)", topic, re.IGNORECASE)
    if explicit_match:
        direction = explicit_match.group(1).upper()
        state["directions"] = [direction]
        state["current_direction"] = direction
        return state
    
    messages = [
        SystemMessage(content=DIRECTION_IDENTIFY_PROMPT),
        HumanMessage(content=f"调研主题：{topic}")
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    try:
        json_str = re.search(r'\{.*\}', content, re.DOTALL).group()
        result = json.loads(json_str)
        directions = result.get("directions", ["B1", "B2"])
    except:
        directions = ["B1", "E", "B2", "D"]
    
    valid_directions = {"A", "B1", "B2", "C", "D", "E"}
    directions = [d for d in directions if d in valid_directions]
    directions = list(dict.fromkeys(directions))
    if not directions:
        directions = ["B1", "E", "B2", "D"]

    priority = {"B1": 1, "E": 2, "B2": 3, "D": 4, "A": 5, "C": 5}
    directions = sorted(directions, key=lambda x: priority.get(x, 99))
    
    state["directions"] = directions
    state["current_direction"] = directions[0] if directions else ""
    
    return state


# ============ 节点2：B1 行业驱动型调研 ============

B1_PROMPT = """你是行业调研专家，执行方向B1：行业驱动型调研（宏观视角）。

调研主题：{topic}

## 严格模板要求（每个部分必须填充，禁止留空）

### 1. 行业概述
- 行业定义与范围
- 发展阶段判断

### 2. 产业链拆解
- 上游/中游/下游图谱
- 各环节价值分布（百分比）

### 3. 市场规模与增速
- 历史规模与增速（近3-5年，表格）
- 未来3-5年预测
- 数据来源标注

### 4. 行业痛点
- 核心问题（至少3个）
- 技术/市场层面瓶颈
- 未被满足的需求

### 5. 竞争格局
- 头部玩家与市场份额（表格）
- CR3/CR5集中度
- 竞争要素分析

### 6. 关键玩家（TOP 5-8企业画像）
每个企业包含：名称、成立时间、核心产品、市场地位、最新数据、官网

### 7. 驱动与抑制因素
- 技术驱动、需求驱动
- 抑制因素/风险

### 8. 趋势判断
- 短期（1年）、中期（3年）、长期（5年+）

## 数据规范（严格执行）
1. **数据来源优先级**：优先使用Wind、Statista、IDC、Gartner、艾瑞咨询、赛迪顾问、中商产业研究院、政府官网、上市公司年报、权威智库报告
2. **每条数据必须标注来源**：[来源名称](完整URL)
3. **禁止使用无URL来源的数据**：如果搜索数据中没有找到某数据，必须标注"未检索到公开数据"，禁止编造
4. **超2年数据标注**：[数据截至XX年]
5. **估算值附公式+参数来源+假设**，标注⚠️
6. **表格中的每个数字都必须有来源**，禁止在表格中使用无来源数据
7. **交叉验证**：市场规模、份额、CAGR、出货量等关键数字至少尝试用2个来源互相校验；若来源冲突，列出不同口径和原因，不强行合并
8. **图片图表**：图片线索只能作为“待OCR/视觉识别”的线索；未经过OCR/视觉插件识别的图片数字不得当作正式数据

## 输出要求
- 每个部分必须有实质内容
- 表格用Markdown格式
- 市场规模章节必须优先输出近5年年份表；缺年份写"未检索到公开数据"
- 总字数不少于3000字
"""

def node_b1_research(state: ResearchState) -> ResearchState:
    """执行B1方向调研"""
    llm = get_llm()
    topic = state["topic"]
    reference_outline = load_reference_outline("B1")
    queries = build_authoritative_queries(topic, "B1")
    search_content = execute_searches(queries, topic=topic, direction="B1")
    
    messages = [
        SystemMessage(content=B1_PROMPT.format(topic=topic)),
        HumanMessage(content=f"目标效果结构参考（只学习结构和粒度，不复制样例数据）：\n{reference_outline}\n\n搜索结果、网页正文与图片线索：\n{search_content}\n\n请根据以上数据，严格按照模板生成B1行业调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["b1_result"] = response.content
    state["market_size_data"] = response.content[:3000]
    _complete_single_direction(state)
    
    return state


# ============ 节点3：E 政策与趋势 ============

E_PROMPT = """你是政策分析专家，执行方向E：政策与趋势调研。

调研主题：{topic}

## 严格模板要求

### 1. 政策背景
- 宏观政策环境
- 政策出台的驱动因素

### 2. 关键人物讲话（必须列出，格式严格）
| 讲话人 | 时间 | 场合/会议 | 核心表述 |
|--------|------|----------|----------|

如果没有找到，明确标注"未检索到公开讲话记录"

### 3. 政策解读
- 核心政策文件梳理（表格）
- 政策要点提取
- 政策层级与关联

### 4. 政策影响方向
- 受政策鼓励的方向
- 受政策限制的方向

### 5. 落地进展
- 试点/示范项目
- 政策执行时间表

### 6. 风险提示
- 政策变动风险
- 合规风险

## 数据来源要求（严格执行）
1. **政策文件来源**：必须是政府官网（gov.cn）、国务院、发改委、财政部、工信部等部委官网
2. **领导人讲话**：必须标注具体时间、场合，如有官方报道链接必须附上
3. **禁止编造政策**：如果未检索到某政策文件，必须标注"未检索到相关政策文件"
4. **试点项目**：必须标注具体项目名称、发布单位、发布时间
5. **政策层级**：优先级为领导人讲话/中央会议精神 > 国务院/国家级规划 > 部委政策 > 地方政策 > 媒体解读
6. **图片图表**：图片线索中的政策路线图、时间表不得直接引用数值，必须标注待OCR核验

## 搜索策略
1. "{topic} 政策 site:gov.cn 2024 2025"
2. "{topic} 国务院 发改委 政策文件"
3. "{topic} 监管 法规 标准 实施"
4. "{topic} 试点 示范 项目 进展"
5. "{topic} 政策解读 智库 影响分析"

## 输出要求
- 每个部分必须有实质内容
- 政策文件必须用表格
- 每条政策必须有具体发文单位、文号（如有）
- 总字数不少于2000字
"""

def node_e_research(state: ResearchState) -> ResearchState:
    """执行E方向调研"""
    llm = get_llm()
    topic = state["topic"]
    reference_outline = load_reference_outline("E")
    queries = build_authoritative_queries(topic, "E")
    search_content = execute_searches(queries, topic=topic, direction="E")
    
    messages = [
        SystemMessage(content=E_PROMPT.format(topic=topic)),
        HumanMessage(content=f"目标效果结构参考（只学习结构和粒度，不复制样例数据）：\n{reference_outline}\n\n搜索结果、网页正文与图片线索：\n{search_content}\n\n请严格按照模板生成E方向政策调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["e_result"] = response.content
    state["policy_data"] = response.content[:2000]
    _complete_single_direction(state)
    
    return state


# ============ 节点4：B2 产品驱动型调研 ============

B2_PROMPT = """你是市场分析专家，执行方向B2：产品驱动型调研。

调研主题：{topic}

## 严格模板要求

### 1. 产品与行业定义
- 产品定位与功能边界
- 产品所属父类市场定义

### 2. 产品直接市场
- 市场规模量化（TAM/SAM/SOM，附计算公式）
- 产品市场规模与增速（表格）
- 父类/同类产品市场总和
- 核心器件/核心技术市场

### 3. 市场细分（STP）
- Segmentation：细分维度
- Targeting：目标市场选择
- Positioning：产品定位

### 4. 应用场景市场
- 上游供应端
- 下游应用端
- 消费者端
- 渠道端

### 5. 产业链全景
- 从核心器件到终端消费者的完整链路
- 各环节价值分布

### 6. 行业痛点
- 产品当前存在的问题
- 行业层面的痛点
- 未被满足的需求

### 7. 竞争格局
- 头部玩家与市场份额

### 8. 驱动与抑制因素
### 9. 趋势判断

## 数据来源要求（严格执行）
1. **市场数据**：优先使用IDC、Gartner、艾瑞咨询、易观、中商产业研究院等权威机构数据
2. **TAM/SAM/SOM计算**：必须附计算公式和参数来源，标注⚠️
3. **用户画像**：必须来自实际调研报告或平台数据，禁止虚构用户特征
4. **产业链数据**：必须标注数据来源和统计时间
5. **测算透明**：TAM/SAM/SOM必须区分官方数据、报告引用数据和估算数据；估算必须写清参数、假设和适用边界
6. **交叉验证**：直接市场、父类市场、核心器件市场至少分别尝试寻找权威来源；缺失则写"未检索到公开数据"
7. **图片图表**：图片图表中的市场规模、份额、渗透率必须经过OCR/视觉插件识别后才能引用；否则只列为待核验线索

## 搜索策略
1. "{topic} 市场规模 TAM SAM SOM IDC Gartner 艾瑞"
2. "{topic} 应用场景 用户画像 行业报告"
3. "{topic} 产业链 价值分布 分析"
4. "{topic} 细分市场 定位 规模"
5. "{topic} 消费者 需求 痛点 调研"

## 输出要求
- 必须包含TAM/SAM/SOM量化（附计算公式）
- 表格用Markdown格式（每个数字必须有来源）
- 禁止编造市场数据
- 总字数不少于3000字
"""

def node_b2_research(state: ResearchState) -> ResearchState:
    """执行B2方向调研"""
    llm = get_llm()
    topic = state["topic"]
    reference_outline = load_reference_outline("B2")
    queries = build_authoritative_queries(topic, "B2")
    search_content = execute_searches(queries, topic=topic, direction="B2")
    
    # 引用B1的市场规模数据
    b1_data = state.get("market_size_data", "")
    
    messages = [
        SystemMessage(content=B2_PROMPT.format(topic=topic)),
        HumanMessage(content=f"目标效果结构参考（只学习结构和粒度，不复制样例数据）：\n{reference_outline}\n\n方向B1已产出的市场规模数据：\n{b1_data}\n\n搜索结果、网页正文与图片线索：\n{search_content}\n\n请严格按照模板生成B2产品驱动型调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["b2_result"] = response.content
    _complete_single_direction(state)
    
    return state


# ============ 节点5：D 竞品分析 ============

D_PROMPT = """你是竞品分析专家，执行方向D：竞品/产品分析。

调研主题：{topic}

## 严格模板要求（竞品分析四段式）

### ① 需求定义
- 核心需求清单（至少5个）
- 优先级排序

### ② 候选方案筛选
- 初筛标准
- 入围方案列表（至少3-5个竞品）

### ③ 多维对比（必须有表格）

#### 功能对比矩阵（表格）
| 功能维度 | 竞品A | 竞品B | 竞品C | 竞品D | ... |
|---------|-------|-------|-------|-------|-----|
| 功能1 | ✅/❌ | ✅/❌ | ... | ... | ... |

#### 性能对比（Benchmark数据）
| 性能指标 | 竞品A | 竞品B | 竞品C | ... |
|---------|-------|-------|-------|-----|
| 指标1 | 数值 | 数值 | ... | ... |

#### 成本对比
| 成本项 | 竞品A | 竞品B | 竞品C | ... |
|-------|-------|-------|-------|-----|

#### 生态对比
| 生态维度 | 竞品A | 竞品B | ... |
|---------|-------|-------|-----|

### ④ 优劣势矩阵
- SWOT分析或优劣势清单
- 关键差异点

### ⑤ 推荐方案
- 综合评分与推荐理由
- 适用场景说明

### ⑥ 迁移路径
- 从现状到推荐方案的路径
- 风险与注意事项

## 数据来源要求（严格执行）
1. **Benchmark数据**：必须来自官方测试、第三方测评机构或权威媒体报道，禁止编造性能数据
2. **价格数据**：必须标注价格来源和时间（如官网报价、电商平台价格）
3. **功能对比**：必须基于实际产品功能，禁止虚构功能点
4. **用户评价**：如引用用户评价，必须标注来源平台
5. **竞品入围**：优先选择官网可查、参数可查、价格可查、用户评价可查的产品；信息缺失的竞品必须标注"资料不足"
6. **图片图表**：竞品参数图、价格截图、对比表截图不得直接转写为数据，必须经OCR/视觉插件识别并标注

## 搜索策略
1. "{topic} 竞品对比 功能 专业测评"
2. "{topic} vs 对比 评测报告"
3. "{topic} 性能测试 Benchmark 数据"
4. "{topic} 价格 成本 报价"
5. "{topic} 用户评价 体验 优缺点"

## 输出要求
- 必须包含功能对比矩阵表格
- 必须包含性能对比表格（表格中的每个数字必须有来源）
- 每个竞品必须有具体数据
- 禁止编造Benchmark数据
- 总字数不少于3000字
"""

def node_d_research(state: ResearchState) -> ResearchState:
    """执行D方向调研"""
    llm = get_llm()
    topic = state["topic"]
    reference_outline = load_reference_outline("D")
    queries = build_authoritative_queries(topic, "D")
    search_content = execute_searches(queries, topic=topic, direction="D")
    
    messages = [
        SystemMessage(content=D_PROMPT.format(topic=topic)),
        HumanMessage(content=f"目标效果结构参考（只学习结构和粒度，不复制样例数据）：\n{reference_outline}\n\n搜索结果、网页正文与图片线索：\n{search_content}\n\n请严格按照模板生成D方向竞品分析报告。")
    ]
    
    response = llm.invoke(messages)
    state["d_result"] = response.content
    state["competitor_data"] = response.content[:2000]
    _complete_single_direction(state)
    
    return state


# ============ 节点6：A/C 技术方案/学术综述 ============

AC_PROMPT = """你是技术分析专家，执行方向{direction}：技术方案调研或学术综述。

调研主题：{topic}

## 严格模板要求

当前执行方向是 **{direction}**：
- 如果是 A，只输出“方向A：技术方案调研”对应内容。
- 如果是 C，只输出“方向C：学术综述”对应内容。
- 如果用户明确同时需要 A/C，才同时覆盖两个模板。

### 方向A：技术方案调研

#### 1. 技术背景
- 技术领域的核心问题（2-3句话）

#### 2. 是什么
- 技术/方案的核心定义
- 关键原理简述

#### 3. 技术发展趋势
- 技术演进方向
- 学术/产业前沿

#### 4. 方案对比（必须有表格）
| 方案 | 原理 | 优势 | 劣势 | 适用场景 | 代表企业/论文 |
|------|------|------|------|----------|--------------|

#### 5. 技术痛点
- 现有方案的核心瓶颈
- 未解决的技术挑战

#### 6. 我们的方案（如适用）
#### 7. 效果
- 实验结果、关键指标对比

#### 8. 创新点
#### 9. 应用前景

### 方向C：学术综述

#### 1. 研究背景
#### 2. 主流方法演进
#### 3. 代表工作对比（表格）
| 方法 | 年份 | 核心思想 | 优点 | 缺点 | 引用数 |
|------|------|----------|------|------|--------|

#### 4. 研究空白
#### 5. 未来方向
#### 6. 我们的研究切入点

## 数据来源要求（严格执行）
1. **实验数据**：必须来自论文原文或官方报告，禁止编造实验结果
2. **论文引用**：必须标注论文标题、作者、年份、发表会议/期刊
3. **技术参数**：必须来自论文或官方文档，禁止虚构参数
4. **方案对比**：必须基于实际论文或官方资料，禁止臆造方案
5. **论文质量**：优先近3-5年高引用、高相关论文；如果只能检索到摘要，必须标注"未阅读全文"
6. **可复现性**：明确数据集、代码、实验环境、指标是否公开；缺失则写"未检索到公开信息"

## 搜索策略
1. "{topic} 技术方案 对比 原理"
2. "{topic} 技术路线 arxiv IEEE 论文"
3. "{topic} 实验结果 Benchmark 性能"
4. "{topic} 技术瓶颈 挑战 最新研究"

## 输出要求
- 必须包含方案对比表格（表格中的每个参数必须有来源）
- 每个方案必须有具体参数和来源
- 禁止编造实验数据和论文引用
- 总字数不少于3000字
"""

def node_ac_research(state: ResearchState) -> ResearchState:
    """执行A/C方向调研"""
    llm = get_llm()
    topic = state["topic"]
    direction = state.get("current_direction", "A/C") or "A/C"
    reference_direction = direction if direction in {"A", "C"} else "A"
    reference_outline = load_reference_outline(reference_direction)
    queries = build_authoritative_queries(topic, reference_direction)
    search_content = execute_searches(queries, topic=topic, direction=reference_direction)
    
    messages = [
        SystemMessage(content=AC_PROMPT.format(topic=topic, direction=direction)),
        HumanMessage(content=f"目标效果结构参考（只学习结构和粒度，不复制样例数据）：\n{reference_outline}\n\n搜索结果、网页正文与图片线索：\n{search_content}\n\n请严格按照模板生成A/C方向技术调研报告。")
    ]
    
    response = llm.invoke(messages)
    if state.get("ac_result"):
        state["ac_result"] = f"{state['ac_result']}\n\n---\n\n{response.content}"
    else:
        state["ac_result"] = response.content
    _complete_single_direction(state)
    
    return state


# ============ 节点7：报告整合 ============

INTEGRATE_PROMPT = """你是报告整合专家。将各方向调研结果整合为一份完整的调研报告。

## 整合规则
1. 每个方向的内容保持独立章节
2. 添加执行摘要（核心发现概述）
3. 统一数据来源标注格式为：[来源名称](URL)
4. 在文档末尾添加完整的参考资料列表
5. 去除重复内容
6. **严格检查**：删除所有无来源的数据，删除所有"可能"、"估计"等模糊表述（除非标注⚠️估算）
7. **来源质量检查**：优先保留来自Wind、Statista、IDC、Gartner、政府官网、上市公司年报的数据
8. 如果某条数据无具体URL来源，删除该数据或标注"未检索到公开数据"
9. **图片图表处理**：图片/图表线索只能进入"待核验线索"或"图片数据待OCR"小节；没有OCR/视觉识别结果时，不得把图片中的数字写入正文数据表
10. **冲突口径处理**：同一指标多来源不一致时，保留来源、年份、统计口径差异，说明为何采用某一口径

## 输出格式
1. 报告标题
2. 执行摘要
3. 各方向章节
4. 图片数据待OCR/视觉识别线索（如有）
5. 参考资料（所有引用的URL列表）

主题：{topic}
"""

def node_integrate_report(state: ResearchState) -> ResearchState:
    """整合报告"""
    llm = get_llm()
    topic = state["topic"]
    
    # 收集所有方向结果
    sections = []
    if state.get("b1_result"):
        sections.append(f"## 方向B1：行业驱动型调研\n\n{state['b1_result']}")
    if state.get("e_result"):
        sections.append(f"## 方向E：政策与趋势\n\n{state['e_result']}")
    if state.get("b2_result"):
        sections.append(f"## 方向B2：产品驱动型调研\n\n{state['b2_result']}")
    if state.get("d_result"):
        sections.append(f"## 方向D：竞品分析\n\n{state['d_result']}")
    if state.get("ac_result"):
        sections.append(f"## 方向A/C：技术方案/学术综述\n\n{state['ac_result']}")
    
    full_content = "\n\n---\n\n".join(sections)
    
    messages = [
        SystemMessage(content=INTEGRATE_PROMPT.format(topic=topic)),
        HumanMessage(content=f"各方向调研结果：\n\n{full_content}\n\n请整合为一份完整的调研报告。")
    ]
    
    response = llm.invoke(messages)
    
    now = datetime.now().strftime('%Y-%m-%d')
    completed = state.get("completed_direction", "")
    next_direction = state.get("next_direction", "")
    progress_note = ""
    if completed:
        progress_note = f"\n\n> 本次仅完成方向 {completed}，以保留搜索深度并降低单次运行超时风险。"
        if next_direction:
            progress_note += f" 如需继续，请在对话中说明“继续调研 {topic} 方向 {next_direction}”。"
        else:
            progress_note += " 当前方向队列已完成。"

    report = f"# {topic} 调研报告\n\n**报告日期**: {now}{progress_note}\n\n---\n\n{response.content}"
    quality_report = build_source_quality_report(report)
    state["source_quality_report"] = quality_report
    state["full_report"] = f"{report}\n\n---\n\n{quality_report}"
    
    return state


# ============ 节点8：生成摘要 ============

SUMMARY_PROMPT = """请根据以下完整调研报告，生成一个不超过200字的执行摘要。

**严格要求**：
1. 摘要中的每个数字必须来自报告原文，禁止编造
2. 必须提及数据来源（如"据IDC统计"、"据Wind数据"等）
3. 核心发现（3-5个关键点）
4. 市场规模和增长趋势（附数据来源）
5. 竞争格局要点
6. 不要使用任何格式标记（如**、###等）
7. 纯文本输出

报告内容：
{report}
"""

def node_generate_summary(state: ResearchState) -> ResearchState:
    """生成报告摘要"""
    llm = get_llm()
    
    messages = [
        SystemMessage(content="你是摘要生成专家。"),
        HumanMessage(content=SUMMARY_PROMPT.format(report=state["full_report"][:5000]))
    ]
    
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):
        content = "\n".join(str(c) for c in content)
    summary = content.strip() if isinstance(content, str) else str(content)
    completed = state.get("completed_direction", "")
    next_direction = state.get("next_direction", "")
    topic = state.get("topic", "")
    if completed:
        if next_direction:
            summary += f"\n\n本次已完成方向 {completed}。为了保证搜索精度并避免超时，本轮只执行一个方向；如果要继续方向 {next_direction} 的调研，请告诉我“继续调研 {topic} 方向 {next_direction}”。"
        else:
            summary += f"\n\n本次已完成方向 {completed}。当前推荐方向队列已完成。"
    state["summary"] = summary
    
    return state


# ============ 节点9：生成DOCX ============

def node_generate_docx(state: ResearchState) -> ResearchState:
    """生成DOCX文件并返回精简的最终输出"""
    topic = state.get("topic", "")
    report = state.get("full_report", "")

    # 防御：如果 topic 丢失，尝试从报告标题或摘要恢复
    if not topic or topic == "未指定主题":
        # 尝试从 full_report 第一行提取标题
        title_match = re.search(r'^#\s+(.+?)(?:\n|$)', report, re.MULTILINE)
        if title_match:
            topic = title_match.group(1).strip()
        else:
            topic = "调研报告"
        state["topic"] = topic

    # 生成安全的文件名
    safe_title = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in topic)
    safe_title = safe_title[:50]

    try:
        docx_url = generate_docx_report.invoke({
            "markdown_content": report,
            "report_title": safe_title
        })
        state["docx_url"] = docx_url
    except Exception as e:
        state["docx_url"] = f"生成文档失败: {str(e)}"

    summary = state.get("summary", "")
    docx_url = state.get("docx_url", "")
    completed = state.get("completed_direction", "")
    next_dir = state.get("next_direction", "")

    # 构建用户可见的最终输出（Markdown 格式）
    final_output = f"""# 📋 调研报告摘要

{summary}

---

📄 **完整报告下载**: [{docx_url}]({docx_url})
"""
    if completed:
        if next_dir:
            final_output += f"\n\n💡 本次已完成方向 **{completed}**。如需继续方向 **{next_dir}** 的调研，请回复：`继续调研 {topic} 方向 {next_dir}`"
        else:
            final_output += f"\n\n✅ 本次已完成方向 **{completed}**。当前推荐方向队列已完成。"

    # 返回精简 state：只保留用户可见内容和 checkpoint 所需的核心字段
    # 注意：不返回 b1_result / full_report 等大字段，避免 test_run/前端输出过长
    # 通过 messages 字段向用户呈现最终摘要+下载链接（AgentStreamRunner 会捕获并展示）
    from langchain_core.messages import AIMessage
    result: ResearchState = {
        "topic": topic,
        "directions": state.get("directions", []),
        "current_direction": "",
        "b1_result": "",
        "e_result": "",
        "b2_result": "",
        "d_result": "",
        "ac_result": "",
        "completed_direction": completed,
        "next_direction": next_dir,
        "market_size_data": "",
        "policy_data": "",
        "competitor_data": "",
        "full_report": "",
        "source_quality_report": "",
        "docx_url": docx_url,
        "summary": summary,
        "messages": [AIMessage(content=final_output)],
    }
    return result


# ============ 路由函数 ============

def route_next(state: ResearchState) -> str:
    """决定下一个节点"""
    current = state.get("current_direction", "")
    return _direction_to_node(current) if current else "integrate"


def get_first_node(state: ResearchState) -> str:
    """获取第一个方向节点"""
    directions = state.get("directions", [])
    if not directions:
        return "integrate"
    
    first = directions[0]
    return _direction_to_node(first)


# ============ 提取topic节点 ============

def _extract_topic_from_messages(messages) -> str:
    """从消息列表中提取用户主题，支持多种消息格式。"""
    if not messages:
        return ""
    for msg in reversed(messages):
        content = None
        if hasattr(msg, 'content') and msg.content:
            content = str(msg.content)
        elif isinstance(msg, dict):
            content = msg.get('content') or msg.get('text') or msg.get('message')
            if not content and msg.get('role') == 'user':
                content = msg.get('content', '')
        if content:
            # 清理常见的调研前缀
            cleaned = re.sub(r'^(请|麻烦|帮我|给我|需要|想要)\s*(调研|分析|研究|了解|查查|看看)\s*[一下]?\s*', '', content.strip())
            cleaned = re.sub(r'[，,\.\s]+$', '', cleaned)
            return cleaned if cleaned else content.strip()
    return ""


def node_extract_topic(state: ResearchState) -> ResearchState:
    """从输入中提取调研主题"""
    # 如果已经有有效topic，直接返回
    if state.get("topic") and state["topic"] not in ("", "未指定主题"):
        return state

    # 尝试从params字段提取（test_run传入的格式）
    if state.get("params"):
        state["topic"] = str(state["params"])
        return state

    # 尝试从messages中提取
    topic = _extract_topic_from_messages(state.get("messages", []))
    if topic:
        state["topic"] = topic
        return state

    # 尝试从其他字段提取
    for key in ["input", "query", "question", "text", "content", "agent_input", "extra"]:
        val = state.get(key)
        if val and isinstance(val, str) and val.strip():
            state["topic"] = val.strip()
            return state

    # 如果无法提取，设置默认值
    state["topic"] = "未指定主题"
    return state


# ============ 构建工作流 ============

def build_agent(ctx=None):
    """构建LangGraph工作流Agent"""
    
    # 创建状态图
    workflow = StateGraph(ResearchState)
    
    # 添加节点
    workflow.add_node("extract_topic", node_extract_topic)
    workflow.add_node("identify", node_identify_directions)
    workflow.add_node("node_b1", node_b1_research)
    workflow.add_node("node_e", node_e_research)
    workflow.add_node("node_b2", node_b2_research)
    workflow.add_node("node_d", node_d_research)
    workflow.add_node("node_ac", node_ac_research)
    workflow.add_node("integrate", node_integrate_report)
    workflow.add_node("summary", node_generate_summary)
    workflow.add_node("docx", node_generate_docx)
    
    # 添加边
    workflow.set_entry_point("extract_topic")
    workflow.add_edge("extract_topic", "identify")
    
    # 从identify到第一个方向节点
    workflow.add_conditional_edges(
        "identify",
        get_first_node,
        {
            "node_b1": "node_b1",
            "node_e": "node_e",
            "node_b2": "node_b2",
            "node_d": "node_d",
            "node_ac": "node_ac",
            "integrate": "integrate"
        }
    )
    
    # 方向节点之间的路由
    for node_name in ["node_b1", "node_e", "node_b2", "node_d", "node_ac"]:
        workflow.add_conditional_edges(
            node_name,
            route_next,
            {
                "node_b1": "node_b1",
                "node_e": "node_e",
                "node_b2": "node_b2",
                "node_d": "node_d",
                "node_ac": "node_ac",
                "integrate": "integrate"
            }
        )
    
    # 报告整合流程
    workflow.add_edge("integrate", "summary")
    workflow.add_edge("summary", "docx")
    workflow.add_edge("docx", END)
    
    # 编译工作流
    app = workflow.compile(checkpointer=MemorySaver())
    
    return app


# ============ 对外接口 ============

def run_research(topic: str, ctx=None) -> Dict[str, str]:
    """运行调研工作流"""
    app = build_agent(ctx)
    
    # 初始状态
    initial_state = ResearchState(
        topic=topic,
        directions=[],
        current_direction="",
        b1_result="",
        e_result="",
        b2_result="",
        d_result="",
        ac_result="",
        completed_direction="",
        next_direction="",
        market_size_data="",
        policy_data="",
        competitor_data="",
        full_report="",
        source_quality_report="",
        docx_url="",
        summary=""
    )
    
    # 运行工作流
    final_state = app.invoke(initial_state, {"configurable": {"thread_id": "research_" + topic}})
    
    return {
        "summary": final_state.get("summary", ""),
        "docx_url": final_state.get("docx_url", ""),
        "full_report": final_state.get("full_report", ""),
        "completed_direction": final_state.get("completed_direction", ""),
        "next_direction": final_state.get("next_direction", "")
    }


if __name__ == "__main__":
    # 测试
    result = run_research("AI芯片市场")
    print(f"摘要: {result['summary']}")
    print(f"下载链接: {result['docx_url']}")
