"""
ResearchFlow 调研工作流 (LangGraph)
6 方向通用调研引擎，输入主题 → 输出讲稿 + PPT提示词 + 参考文献

工作流节点:
1. topic_parser    - 主题解析，确定方向/关键词/范围
2. collect_data    - 按方向执行多维度搜索 + LLM提取
3. screening       - 数据筛选与质量评估
4. solution_design - 按模板设计讲稿框架
5. output_generation - 生成完整报告 + PPT提示词 + 参考文献
"""

import json
import logging
from typing import TypedDict, Optional, Annotated, List
from functools import partial

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from coze_coding_dev_sdk import SearchClient
from coze_coding_utils.log.write_log import request_context
from coze_coding_utils.runtime_ctx.context import new_context

logger = logging.getLogger(__name__)

# ============================================================================
# State
# ============================================================================

class ResearchFlowState(TypedDict):
    # 输入
    topic: str
    direction: str
    template: str
    extra_requirements: str

    # Node 1 输出
    parsed: Optional[dict]

    # 多任务检测
    multi_topic: Optional[List[str]]  # 检测到多个任务时，填入任务列表

    # Node 2 输出
    collected_data: Optional[str]

    # Node 3 输出
    screening_result: Optional[str]

    # Node 4 输出
    solution_framework: Optional[str]

    # Node 5 输出
    report: Optional[str]
    ppt_prompt: Optional[str]
    references: Optional[str]

    # 错误信息
    error: Optional[str]


# ============================================================================
# 搜索辅助函数
# ============================================================================

def _execute_search(query: str, count: int = 8, time_range: str = None,
                    sites: str = None, need_content: bool = False) -> str:
    """执行单次搜索，返回格式化结果"""
    ctx = request_context.get() or new_context(method="research_flow.search")
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

        parts = []
        if response.summary:
            parts.append(f"【AI摘要】{response.summary}\n")

        if response.web_items:
            for i, item in enumerate(response.web_items, 1):
                parts.append(
                    f"[{i}] {item.title or '无标题'}\n"
                    f"    来源: {item.site_name or '未知'} | URL: {item.url or ''}\n"
                    f"    摘要: {item.snippet or ''}\n"
                )

        return "\n".join(parts) if parts else f"搜索 '{query}' 未找到结果。"

    except Exception as e:
        logger.error(f"搜索失败: query={query}, error={e}")
        return f"搜索 '{query}' 执行失败: {str(e)}"


def _run_multi_searches(queries: List[str], sites: str = None, time_range: str = None) -> str:
    """执行多轮搜索，合并结果"""
    results = []
    for i, query in enumerate(queries, 1):
        logger.info(f"搜索 [{i}/{len(queries)}]: {query}")
        result = _execute_search(query, count=8, sites=sites, time_range=time_range)
        results.append(f"=== 搜索 {i}: {query} ===\n{result}\n")
    return "\n".join(results)


# ============================================================================
# Node 1: 主题解析
# ============================================================================

_TOPIC_PARSER_PROMPT = """你是调研主题解析器。请分析用户给出的调研主题，检测是否包含多个独立任务，并输出结构化信息。

## 输入
- 主题：{topic}
- 用户指定方向：{direction}
- 用户额外要求：{extra_requirements}

## 任务
1. 【多任务检测】检查主题是否包含多个独立调研任务
   - 触发条件：包含"和""与""+""、"、"或"等连接词，或用顿号、换行符分隔
   - 示例1："人形机器人产业和低空经济" → 两个任务
   - 示例2："竞品分析+政策研究" → 两个任务
   - 示例3："技术方案调研" → 一个任务
2. 将每个任务转化为明确的调研问题
3. 如果用户没有指定方向，判断最合适的方向（每个任务可以不同）
4. 生成搜索用的关键词列表

## 方向判断规则
- 包含"技术""方案""设计""实现""仿真""复现" → A
- 是一个行业名称（如"人形机器人""低空经济"） → B1
- 是一个具体产品（如"水果无损检测仪""便携式血糖仪"） → B2
- 包含"综述""研究进展""研究空白""方法论" → C
- 包含"对比""竞品""选型""评测""替代" → D
- 包含"政策""趋势""监管""合规""规划" → E

## 输出格式（严格JSON，不要输出其他内容）

// 如果检测到多个任务：
{{
  "is_multi_topic": true,
  "topics": [
    {{
      "topic": "任务1具体主题",
      "direction": "B1",
      "direction_reason": "判断理由",
      "template": "B1",
      "search_keywords": {{
        "primary": ["核心关键词1"],
        "secondary": ["补充关键词1"],
        "english": ["English keyword 1"]
      }},
      "scope": {{
        "time_range": "近3年",
        "depth": "standard",
        "focus_areas": ["重点关注领域"]
      }}
    }},
    {{
      "topic": "任务2具体主题",
      "direction": "E",
      "direction_reason": "判断理由",
      "template": "E",
      "search_keywords": {{
        "primary": ["核心关键词2"],
        "secondary": ["补充关键词2"],
        "english": []
      }},
      "scope": {{
        "time_range": "近3年",
        "depth": "standard",
        "focus_areas": ["重点关注领域"]
      }}
    }}
  ]
}}

// 如果只有一个任务：
{{
  "is_multi_topic": false,
  "questions": ["调研问题1", "调研问题2"],
  "direction": "B1",
  "direction_reason": "判断理由",
  "template": "B1",
  "search_keywords": {{
    "primary": ["核心关键词1", "核心关键词2"],
    "secondary": ["补充关键词1", "补充关键词2"],
    "english": ["English keyword 1", "English keyword 2"]
  }},
  "scope": {{
    "time_range": "近3年",
    "depth": "standard",
    "focus_areas": ["重点关注领域1", "重点关注领域2"]
  }}
}}"""


def topic_parser(state: ResearchFlowState, llm: ChatOpenAI) -> dict:
    """解析主题，输出结构化信息（支持多任务检测）"""
    topic = state["topic"]
    direction = state.get("direction", "")
    extra = state.get("extra_requirements", "")

    prompt = _TOPIC_PARSER_PROMPT.format(
        topic=topic,
        direction=direction or "未指定",
        extra_requirements=extra or "无",
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)

        # 提取 JSON
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(json_str)

        # 如果用户指定了方向，强制使用用户指定值
        if direction and direction.strip():
            parsed["direction"] = direction.strip()

        # 检测多任务
        is_multi = parsed.get("is_multi_topic", False)
        if is_multi:
            topics_list = parsed.get("topics", [])
            topic_names = [t.get("topic", "") for t in topics_list]
            logger.info(f"检测到多任务: {topic_names}")
            return {
                "parsed": parsed,
                "multi_topic": topic_names,
            }

        logger.info(f"主题解析完成: direction={parsed.get('direction')}, "
                    f"questions={parsed.get('questions')}")
        return {"parsed": parsed, "multi_topic": None}

    except Exception as e:
        logger.error(f"主题解析失败: {e}")
        # 降级处理
        fallback_dir = direction.strip() if direction and direction.strip() else "B1"
        return {
            "parsed": {
                "questions": [topic],
                "direction": fallback_dir,
                "direction_reason": "解析失败，使用默认方向",
                "template": fallback_dir,
                "search_keywords": {
                    "primary": [topic],
                    "secondary": [],
                    "english": [],
                },
                "scope": {
                    "time_range": "近3年",
                    "depth": "standard",
                    "focus_areas": [],
                },
            },
            "multi_topic": None,
            "error": f"主题解析异常，已降级处理: {str(e)}",
        }


# ============================================================================
# Node 2: 信息采集 (按方向分发)
# ============================================================================

# ---- 方向 A: 技术方案 ----
_PROMPT_A = """你是学术论文筛选器。请根据搜索结果，筛选出最相关的论文与方案。

## 调研问题
{questions}

## 搜索结果
{search_results}

## 筛选标准
| 维度 | 权重 |
|------|------|
| 创新性 | 30% |
| 可复现性 | 35% |
| 难度适中性 | 25% |
| 时效性 | 10% |

## 输出要求
1. 按相关度排序，选出Top 10论文/方案
2. 每篇输出：标题、作者、年份、核心方法、创新点、可复现性评估、难度评估、来源URL
3. 最终推荐Top 3，附推荐理由
4. 绝不编造论文，搜索结果中没有的不要写
5. 所有数据标注来源 [来源名称](URL)"""

_SEARCH_QUERIES_A = [
    '{primary} 方案 对比 综述',
    '{primary} 开源实现 GitHub',
    '{english} survey review',
]

# ---- 方向 B1: 行业驱动 ----
_PROMPT_B1 = """你是行业数据提取器。请从搜索结果中提取行业关键数据。

## 调研主题
{topic}

## 调研问题
{questions}

## 搜索结果
{search_results}

## 需要提取的数据（严格按以下8个维度）
1. **行业定义与范围**：这个行业的边界是什么？包含哪些细分领域？
2. **市场规模与趋势**：
   - 当前市场规模（全球+中国）
   - **历年市场规模数据**（至少近5年：2021/2022/2023/2024/2025，如无可获取年份请标注"未给出"）
   - 增速预测（CAGR）
   - 市场增长的驱动因素和阻碍因素
3. **产业链结构**（上游/中游/下游）：各环节代表性企业、核心壁垒、价值分配
4. **竞争格局**：头部玩家名单、市场份额、竞争优势对比
5. **驱动因素**（政策/技术/需求/资本）
6. **行业痛点与挑战**：
   - 行业普遍面临的核心痛点（技术瓶颈、成本压力、人才短缺、政策限制等）
   - 产业链各环节的主要困难
   - 客户/用户端的普遍抱怨
   - 这些痛点的严重程度和发展趋势
7. **风险与挑战**：宏观风险、政策风险、技术替代风险
8. **趋势判断**：未来3-5年发展方向

## 输出要求
- **年份数据必须整理成表格**：| 年份 | 全球市场规模 | 中国市场规模 | 增长率 | 来源 |
- 每条数据标注来源 [来源名称](URL)
- 没有的写"未给出"，绝不允许编造
- 超过2年数据标注年份
- 估算值写明公式+参数来源+假设，标注 ⚠️估算值
- 痛点部分必须具体，不能泛泛而谈，每条痛点附具体案例或数据支撑
- 优先级：官方统计 > 权威研报 > 行业媒体 > 社区讨论"""

_SEARCH_QUERIES_B1 = [
    '{primary} 市场规模 产业链 头部企业',
    '{primary} 行业分析 竞争格局 研报',
    '{primary} 行业痛点 挑战 困难 瓶颈',
    '{primary} 成本压力 技术瓶颈 人才短缺',
    '{english} market size industry report pain points',
]

# ---- 方向 B2: 产品驱动 ----
_PROMPT_B2 = """你是产品驱动型调研数据提取器。按SOP规定的三层框架提取数据，重点调研行业痛点。

## 调研主题
{topic}

## 调研问题
{questions}

## 三层调研框架（严格按此框架输出）

### 第一层：产品本身的市场
1. **产品定义与范围**：这个产品是什么？解决什么问题？技术原理？
2. **直接市场规模**：该产品品类的市场规模（当前+历年数据+预测）
3. **父类市场规模**：该产品所属大类（如"无损检测仪器"→"检测仪器"→"仪器仪表"）的市场规模
4. **核心器件/材料市场**：构成该产品的核心零部件市场规模
5. **产品现存问题与痛点**：
   - 当前产品普遍存在的问题（精度、成本、便携性、可靠性等）
   - 用户/客户的普遍抱怨和不满
   - 现有技术路线的瓶颈和限制
   - 这些痛点是否已被解决？解决到什么程度？

### 第二层：应用场景的市场
1. **主要应用场景**：各场景的市场规模、渗透率、增速
2. **上游供应链**：核心供应商、原材料、关键零部件
3. **下游客户画像**：B端客户类型、采购决策因素、价格敏感度
4. **消费者端**（如适用）：C端消费者需求、使用场景、购买渠道
5. **渠道与分销**：主要销售渠道、渠道成本结构

### 第三层：宏观环境
1. **政策环境**：相关政策、标准、监管要求
2. **技术趋势**：技术发展方向、新兴技术威胁
3. **国际市场**：全球市场格局、进出口情况、国际竞争
4. **替代品威胁**：是否存在替代方案？替代品的优劣势？

## 搜索结果
{search_results}

## 输出要求
- **年份数据必须整理成表格**：| 年份 | 全球市场规模 | 中国市场规模 | 增长率 | 来源 |
- **痛点必须单独成章节**，每条痛点附：
  - 痛点描述（具体、可量化）
  - 影响范围（涉及多少企业/用户）
  - 严重程度（高/中/低，附依据）
  - 现有解决方案及效果
  - 未解决的原因分析
- 每条数据标注来源 [来源名称](URL)
- 没有的写"未给出"，绝不允许编造
- 估算值标注 ⚠️ 并附公式+参数+假设"""

_SEARCH_QUERIES_B2 = [
    '{primary} 市场规模 行业分析 痛点',
    '{primary} 产品问题 用户抱怨 技术瓶颈',
    '{primary} 同类产品 替代品 应用场景',
    '{primary} 应用场景 消费者 渠道',
    '{english} market size pain points user complaints',
]

# ---- 方向 C: 学术综述 ----
_PROMPT_C = """你是学术论文结构化分析器。请梳理研究脉络。

## 调研主题
{topic}

## 搜索结果
{search_results}

## 输出要求
1. 研究背景：问题定义与研究意义
2. 方法演进：按时间线梳理主流方法
3. 代表工作对比：核心论文（标题/年份/方法/创新/引用/URL）
4. 研究空白：未解决的问题
5. 未来方向：技术趋势和突破口

注意：绝不编造论文，搜索结果中没有的不要写。"""

_SEARCH_QUERIES_C = [
    '{primary} 综述 survey review',
    '{primary} 研究进展 最新',
    '{english} survey review state-of-the-art',
]

# ---- 方向 D: 竞品分析 ----
_PROMPT_D = """你是竞品数据提取器。请提取竞品信息。

## 调研主题
{topic}

## 搜索结果
{search_results}

## 分层搜索策略
第一层：头部产品（通用搜索+电商畅销榜）
第二层：细分领域（开源/众筹/垂直平台）
第三层：冷门/长尾（多语言/技术反查）

## 输出要求
对每个竞品提取：产品名、厂商、核心功能、关键参数、定价、来源URL
注意：绝不编造价格和参数，没有的写"未给出"。"""

_SEARCH_QUERIES_D = [
    '{primary} 竞品对比 review comparison',
    '{primary} 开源 GitHub 替代方案',
    '{english} alternatives competitors comparison',
]

# ---- 方向 E: 政策趋势 ----
_PROMPT_E = """你是政策数据提取器。请提取政策与趋势信息。

## 调研主题
{topic}

## 搜索结果
{search_results}

## 需要提取的数据
1. 政策背景与宏观环境
2. 核心政策文件（名称/机构/时间/要点/URL）
3. 高层讲话与名人名言（必须重点提取！）
   - **国家领导人讲话**：习近平/李强等关于本主题的重要讲话，提取：人名、场合（会议名称/地点）、时间（年月日）、核心原话或要点概括、来源URL
   - **领域名人言论**：吴恩达、李飞飞、周志华、周鸿祎等行业领袖/专家的相关言论，提取：人名、场合、时间、核心表述、来源URL
   - 如果未找到领导人讲话，搜索"习近平 + 主题"、"李强 + 主题"、"部长 + 主题"补充
4. 产业映射：受益/受限环节
5. 落地进展：试点/示范
6. 国际对比：其他国家政策

## 输出要求
- 政策找原文，附来源URL
- 标注发布时间和生效时间
- 关注"试点""示范"类政策
- **领导人讲话必须附时间和场合**，如"2024年3月，习近平在XX会议上指出：..."
- **名人名言必须标注出处和场合**
- 绝不编造政策内容或讲话内容"""

_SEARCH_QUERIES_E = [
    '{primary} 政策 规划 管理办法',
    '{primary} 试点 示范 落地进展',
    '{english} policy regulation framework',
    '习近平 {primary} 重要讲话',
    '{primary} 吴恩达 OR 李飞飞 OR 专家 观点',
    '{primary} 官方 发布会 吹风会',
]


# ============================================================================
# 全局结构化提取 Prompt（所有方向通用）
# 强制从采集文本中提取：市场数据、竞品、政策、技术、学术等所有方向的关键数据
# ============================================================================

_EXTRACT_GLOBAL = """你是结构化数据提取器。请从以下采集文本中，提取所有能找到的结构化数据。

## 重要原则
- **宁可多提取，不可漏提**：如果搜索结果中出现了具体数字、表格、排名、政策名称、产品参数，务必提取
- **严格基于文本**：只提取文本中实际存在的数据，不编造、不推测
- **未找到标注"未找到"**：某个字段在文本中完全没有提及时，写"未找到"
- **保留来源**：每个数据项必须附上来源 URL
- **领导人/名人讲话必须提取**：若文本中出现习近平、李强等领导人，或吴恩达、李飞飞、周志华等行业名人对主题的相关言论，务必提取（人名、场合、时间、核心表述、来源URL）

## 要提取的字段（全部方向适用，按实际内容提取，不全有）

```json
{{
  "market_data": {{
    "yearly_data": [
      {{
        "year": "年份",
        "global_size": "全球市场规模（含单位）",
        "china_size": "中国市场规模（含单位）",
        "global_shipment": "全球出货量（含单位）",
        "china_shipment": "中国出货量（含单位）",
        "growth_rate": "同比增长率",
        "source": "[来源名称](URL)"
      }}
    ]
  }},
  "competitive_products": [
    {{
      "name": "产品/竞品名称",
      "price": "价格（含单位）",
      "key_specs": "关键参数（可多行）",
      "advantages": "主要优势",
      "disadvantages": "主要劣势",
      "rating": "综合评级",
      "target_users": "目标用户",
      "source": "[来源名称](URL)"
    }}
  ],
  "pain_points": [
    {{
      "type": "痛点类型（技术/成本/政策/用户适配/市场竞争等）",
      "problem": "具体问题描述",
      "evidence": "证据（具体数据或用户反馈）",
      "severity": "严重程度（高/中/低）",
      "current_solution": "现有解决方案及效果",
      "unresolved_reason": "未解决的根本原因",
      "source": "[来源名称](URL)"
    }}
  ],
  "policies": [
    {{
      "name": "政策名称",
      "issued_by": "发布机构",
      "date": "发布时间",
      "key_content": "核心内容",
      "impact": "对行业的影响",
      "source": "[来源名称](URL)"
    }}
  ],
  "leadership_statements": [
    {{
      "speaker": "讲话人姓名（如习近平、李强，或领域名人如吴恩达）",
      "title": "职务/身份",
      "occasion": "场合（如'全国两会'、'XX论坛'、'XX考察'）",
      "date": "时间（YYYY-MM-DD）",
      "quote": "核心原话或要点概括",
      "source": "[来源名称](URL)"
    }}
  ],
  "technical_specs": [
    {{
      "category": "技术类别（硬件/算法/材料等）",
      "spec": "具体技术指标或方案",
      "advantage": "优势",
      "limitation": "局限性",
      "maturity": "成熟度（实验/小规模/成熟）",
      "source": "[来源名称](URL)"
    }}
  ],
  "academic_papers": [
    {{
      "title": "论文标题",
      "authors": "作者",
      "year": "年份",
      "journal": "期刊/会议",
      "core_method": "核心方法",
      "innovation": "创新点",
      "reproducibility": "可复现性评估",
      "url": "URL"
    }}
  ],
  "upstream_components": ["核心上游供应商/技术1", "核心上游供应商/技术2"],
  "downstream_applications": ["下游应用场景1", "下游应用场景2"],
  "industry_overview": "一句话行业概述",
  "notable_findings": ["其他值得关注的发现1", "其他值得关注的发现2"]
}}
```

## 待分析文本
{raw_text}
"""


def _build_queries(direction: str, topic: str, keywords: dict) -> List[str]:
    """根据方向构建搜索查询列表"""
    primary = ", ".join(keywords.get("primary", [topic]))
    english = ", ".join(keywords.get("english", [topic]))

    template_map = {
        "A": _SEARCH_QUERIES_A,
        "B1": _SEARCH_QUERIES_B1,
        "B2": _SEARCH_QUERIES_B2,
        "C": _SEARCH_QUERIES_C,
        "D": _SEARCH_QUERIES_D,
        "E": _SEARCH_QUERIES_E,
    }

    templates = template_map.get(direction, _SEARCH_QUERIES_B1)
    queries = []
    for tpl in templates:
        q = tpl.format(primary=primary, english=english, topic=topic)
        queries.append(q)
    return queries


def _build_collection_prompt(direction: str, topic: str, questions: List[str],
                             search_results: str) -> str:
    """根据方向构建采集节点Prompt"""
    prompt_map = {
        "A": _PROMPT_A,
        "B1": _PROMPT_B1,
        "B2": _PROMPT_B2,
        "C": _PROMPT_C,
        "D": _PROMPT_D,
        "E": _PROMPT_E,
    }
    template = prompt_map.get(direction, _PROMPT_B1)
    return template.format(
        topic=topic,
        questions="\n".join(f"- {q}" for q in questions),
        search_results=search_results,
    )


def collect_data(state: ResearchFlowState, llm: ChatOpenAI) -> dict:
    """按方向执行多维度搜索 + LLM 提取"""
    parsed = state["parsed"]
    direction = parsed.get("direction", "B1")
    topic = state["topic"]
    keywords = parsed.get("search_keywords", {"primary": [topic]})
    questions = parsed.get("questions", [topic])

    logger.info(f"开始信息采集: direction={direction}, topic={topic}")

    # 1. 构建搜索查询
    queries = _build_queries(direction, topic, keywords)

    # 2. 执行搜索
    search_results = _run_multi_searches(queries)

    # 3. LLM 提取结构化数据
    prompt = _build_collection_prompt(direction, topic, questions, search_results)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"信息采集完成: direction={direction}, content_length={len(content)}")

        # ---- 步骤3.5: 强制结构化提取（B1/B2 专用）----
        # 确保关键数据（竞品表格、年份数据）被明确提取，不被最终输出阶段遗漏
        if direction in ("B1", "B2") and len(content) > 200:
            structured = _do_structured_extraction(llm, direction, content, search_results)
            if structured:
                content = content + "\n\n## 【结构化提取数据 - 关键字段】\n" + structured
                logger.info(f"结构化提取完成: structured_length={len(structured)}")

        return {"collected_data": content}
    except Exception as e:
        logger.error(f"信息采集LLM处理失败: {e}")
        return {
            "collected_data": f"数据采集完成，但结构化处理失败。原始搜索结果:\n{search_results}",
            "error": f"信息采集处理异常: {str(e)}",
        }


def _do_structured_extraction(llm: ChatOpenAI, direction: str,
                               prose_data: str, raw_search: str) -> str:
    """强制从采集数据中提取关键结构化字段（全局通用，所有方向适用）

    提取：市场数据、竞品、政策、技术、学术、痛点等所有方向的关键数据。
    """
    # 优先用 prose_data（已有LLM提取），raw_search兜底
    raw_text = prose_data if len(prose_data) > 200 else raw_search
    raw_text = raw_text[:8000]  # 控制token

    prompt = _EXTRACT_GLOBAL.format(raw_text=raw_text)
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else str(response.content)
        # 去掉markdown代码块包装
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:]) if len(lines) > 2 else raw
            raw = raw.replace("```json", "").replace("```", "").strip()
        logger.info("结构化提取完成（全局通用）")
        return raw
    except Exception as e:
        logger.warning(f"结构化提取失败（不影响主流程）: {e}")
        return ""


# ============================================================================
# Node 3: 筛选决策
# ============================================================================

_SCREENING_PROMPT = """你是调研数据筛选与决策器。请根据采集到的信息，按筛选标准评估。

## 调研方向
{direction}

## 调研问题
{questions}

## 采集到的信息
{collected_data}

## 筛选标准（按方向）
- A 技术方案：创新性30% + 可复现性35% + 难度适中25% + 时效性10%
- B1 行业驱动：数据可靠性30% + 时效性30% + 增速潜力25% + 竞争格局15%
- B2 产品驱动：市场相关性30% + 数据可靠性30% + 时效性25% + 覆盖完整性15%
- C 学术综述：研究空白35% + 引用影响力25% + 时效性25% + 方法代表性15%
- D 竞品分析：功能覆盖30% + 社区活跃度25% + 上手成本25% + 生态成熟度20%
- E 政策趋势：政策力度30% + 落地进展35% + 覆盖范围20% + 时效性15%

## 输出要求
1. 按标准评估，输出Top 3-5推荐项
2. 每项附评分（百分制）和推荐理由
3. 标注关键数据来源
4. 列出缺失数据（标注"需补充"）
5. 明确最终核心结论

铁律：绝不编造数据，来源必须附URL，估算值必须透明标注。"""


def screening_decision(state: ResearchFlowState, llm: ChatOpenAI) -> dict:
    """数据筛选与质量评估"""
    parsed = state["parsed"]
    direction = parsed.get("direction", "B1")
    questions = parsed.get("questions", [])
    collected = state.get("collected_data", "")

    prompt = _SCREENING_PROMPT.format(
        direction=direction,
        questions="\n".join(f"- {q}" for q in questions),
        collected_data=collected[:15000],  # 截断避免超长
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"筛选决策完成: content_length={len(content)}")
        return {"screening_result": content}
    except Exception as e:
        logger.error(f"筛选决策失败: {e}")
        return {
            "screening_result": "筛选评估跳过（处理异常），直接使用采集数据。",
            "error": f"筛选决策异常: {str(e)}",
        }


# ============================================================================
# Node 4: 方案设计
# ============================================================================

_SOLUTION_DESIGN_PROMPT = """你是调研方案设计器。请根据筛选结果设计完整调研方案框架。

## 调研方向
{direction}

## 使用的模板
{template}

## 筛选结论
{screening_result}

## 采集到的数据
{collected_data}

## 讲稿模板

### 模板A1（技术深度型）
行业背景 → 是什么 → 发展趋势 → 方案对比 → 行业痛点 → 我们的方案 → 效果 → 创新点 → 市场价值

### 模板A2（应用展示型）
行业背景 → 政策背景 → 行业痛点 → 我们的方案 → 创新点 → 效果对比 → 项目成果 → 应用价值

### 模板B1（行业驱动型）
行业概述 → 市场规模与趋势（必须含历年数据表格） → 产业链拆解 → 竞争格局 → 关键玩家 → 驱动因素 → 行业痛点与挑战（重点章节） → 风险与挑战 → 趋势判断

### 模板B2（产品驱动型）
产品与行业定义 → 产品直接市场（含历年数据表格） → 应用场景市场 → 产业链全景 → 竞争格局 → 驱动因素 → 行业痛点与挑战（重点章节：产品现存问题+用户抱怨+技术瓶颈） → 风险与挑战 → 趋势判断

### 模板C（学术综述型）
研究背景 → 主流方法演进 → 代表工作对比 → 研究空白 → 未来方向 → 我们的研究切入点

### 模板D（竞品分析型）
需求定义 → 候选方案筛选 → 多维对比 → 优劣势矩阵 → 推荐方案 → 迁移路径

### 模板E（政策趋势型）
政策背景 → 政策解读 → 产业映射 → 受益环节 → 标的/方向 → 风险提示

## 输出要求
1. 严格按照选定模板结构展开
2. 每个章节写清楚核心要点（3-5条）
3. 紧密结合数据和筛选结论
4. 每个论点附数据或引用支撑
5. 遵循"发现问题→解决问题"逻辑
6. 来源标注格式：[来源名称](URL)
7. 缺失数据标注"需补充"

输出 Markdown 格式的完整方案框架大纲。"""


def solution_design(state: ResearchFlowState, llm: ChatOpenAI) -> dict:
    """按模板设计讲稿框架"""
    parsed = state["parsed"]
    direction = parsed.get("direction", "B1")
    template = state.get("template") or parsed.get("template") or direction
    screening = state.get("screening_result", "")
    collected = state.get("collected_data", "")

    prompt = _SOLUTION_DESIGN_PROMPT.format(
        direction=direction,
        template=template,
        screening_result=screening[:8000],
        collected_data=collected[:8000],
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"方案设计完成: content_length={len(content)}")
        return {"solution_framework": content}
    except Exception as e:
        logger.error(f"方案设计失败: {e}")
        return {
            "solution_framework": "方案设计跳过（处理异常），直接进入成果输出。",
            "error": f"方案设计异常: {str(e)}",
        }


# ============================================================================
# Node 5: 成果输出
# ============================================================================

_OUTPUT_GENERATION_PROMPT = """你是调研成果生成器。请根据方案框架，生成完整调研报告和PPT提示词。

## 方案框架
{solution_framework}

## 采集数据
{collected_data}

## 【强制要求】数据使用规则
你必须严格按以下规则使用采集数据：
1. **市场规模章节**：从采集数据的 "market_data" 或 "yearly_data" 字段中提取历年数据，逐一填入表格，不能留空或编造
2. **竞品分析章节**：从采集数据的 "competitive_products" 或 "competing_products" 字段中提取每个竞品的名称、价格、核心参数，逐一填入表格
3. **痛点章节**：从采集数据的 "pain_points" 或 "product_pain_points" 字段中提取每条痛点，逐一列出
4. **如果上述字段存在JSON数据，优先使用JSON中的数据**，不要用自己训练知识重新生成
5. **如果采集数据中没有某字段，才用自己的知识，但必须标注"需补充"**

## 铁律提醒
1. 绝不编造数据，没有的写"未给出"
2. 每条关键数据附来源：[来源名称](URL)
3. 估算值标注⚠️并附公式+参数来源+假设
4. 超过2年数据标注年份

## 输出内容

### 第一部分：调研报告全文（Markdown）

按方案框架展开完整报告，要求：
- 每个章节内容充实，不是一句话概括
- 数据全部附来源标注
- 逻辑连贯，章节间有过渡
- 结论明确，不含糊
- **行业痛点与挑战章节必须深入**：不能泛泛而谈，每条痛点要有具体案例、数据或用户反馈支撑
- **市场规模章节必须包含历年数据表格**（近5年），格式如下：

```markdown
| 年份 | 全球市场规模(亿元) | 中国市场规模(亿元) | 同比增长率 | 数据来源 |
|------|-------------------|-------------------|-----------|---------|
| 2021 | xx | xx | xx% | [来源](URL) |
| 2022 | xx | xx | xx% | [来源](URL) |
| 2023 | xx | xx | xx% | [来源](URL) |
| 2024 | xx | xx | xx% | [来源](URL) |
| 2025 | xx | xx | xx% | [来源](URL) |
| 2026E | xx | xx | xx% | [来源](URL) |
| 2030E | xx | xx | xx% | [来源](URL) |
```

> 注：E表示预测值。如某些年份数据无法获取，标注"未给出"，绝不允许编造。

### 第二部分：PPT提示词

为每一页PPT生成：
```
第X页：[页面标题]
- 要点1：xxx
- 要点2：xxx
- 要点3：xxx
- 布局建议：[左右分栏/上下结构/表格/图表/全图]
- 视觉建议：[配什么类型的图/用什么颜色调性]
```

**必含页面**：市场规模趋势页（用历年数据做折线图/柱状图）、行业痛点页（痛点清单+严重程度可视化）

### 第三部分：附录 - 信息来源表

| 序号 | 内容 | 来源 | URL | 可靠性评级 |
|------|------|------|-----|-----------|
| 1 | xxx | xxx | [链接](URL) | ★★★★★ |

可靠性评级：★★★★★官方原文 / ★★★★权威机构 / ★★★第三方平台 / ★★转载需核实 / ★未验证
"""


def output_generation(state: ResearchFlowState, llm: ChatOpenAI) -> dict:
    """生成最终报告 + PPT提示词 + 参考文献"""
    framework = state.get("solution_framework", "")
    collected = state.get("collected_data", "")

    prompt = _OUTPUT_GENERATION_PROMPT.format(
        solution_framework=framework[:10000],
        collected_data=collected[:10000],
    )

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content if isinstance(response.content, str) else str(response.content)
        logger.info(f"成果输出完成: content_length={len(content)}")

        # 尝试拆分三个部分
        report = content
        ppt_prompt = ""
        references = ""

        if "### 第二部分" in content:
            parts = content.split("### 第二部分")
            report = parts[0].strip()
            rest = parts[1]
            if "### 第三部分" in rest:
                ppt_prompt = rest.split("### 第三部分")[0].strip()
                references = rest.split("### 第三部分")[1].strip()
            else:
                ppt_prompt = rest.strip()
        elif "## 第二部分" in content:
            parts = content.split("## 第二部分")
            report = parts[0].strip()
            rest = parts[1]
            if "## 第三部分" in rest:
                ppt_prompt = rest.split("## 第三部分")[0].strip()
                references = rest.split("## 第三部分")[1].strip()
            else:
                ppt_prompt = rest.strip()

        return {
            "report": report,
            "ppt_prompt": ppt_prompt,
            "references": references,
        }

    except Exception as e:
        logger.error(f"成果输出失败: {e}")
        return {
            "report": f"成果生成异常，以下是原始框架:\n{framework}",
            "ppt_prompt": "",
            "references": "",
            "error": f"成果输出异常: {str(e)}",
        }


# ============================================================================
# 编译工作流
# ============================================================================

def build_research_flow(llm: ChatOpenAI):
    """编译并返回 ResearchFlow 工作流图"""
    workflow = StateGraph(ResearchFlowState)

    # 添加节点
    workflow.add_node("topic_parser", partial(topic_parser, llm=llm))
    workflow.add_node("collect_data", partial(collect_data, llm=llm))
    workflow.add_node("screening", partial(screening_decision, llm=llm))
    workflow.add_node("solution_design", partial(solution_design, llm=llm))
    workflow.add_node("output_generation", partial(output_generation, llm=llm))

    # 设置边
    workflow.add_edge(START, "topic_parser")
    workflow.add_edge("topic_parser", "collect_data")
    workflow.add_edge("collect_data", "screening")
    workflow.add_edge("screening", "solution_design")
    workflow.add_edge("solution_design", "output_generation")
    workflow.add_edge("output_generation", END)

    return workflow.compile()
