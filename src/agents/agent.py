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
from tools.web_search_tool import web_search
from tools.fetch_url_tool import fetch_url_content
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
    market_size_data: str = ""
    policy_data: str = ""
    competitor_data: str = ""
    full_report: str = ""
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

def execute_searches(queries: List[str]) -> str:
    """执行多轮搜索并整合结果"""
    results = []
    for query in queries:
        try:
            result = web_search.invoke({"query": query, "count": 8})
            results.append(f"### 搜索: {query}\n{result}\n")
        except Exception as e:
            results.append(f"### 搜索: {query}\n搜索失败: {str(e)}\n")
    return "\n".join(results)


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
1. 如果主题是一个具体技术/方案 → A + B1 + B2 + D
2. 如果主题是一个行业名称 → B1 + E + B2 + D
3. 如果主题是一个具体产品 → B2 + D + B1 + E
4. 如果主题需要写综述/找研究空白 → C + A
5. 如果主题涉及政策驱动 → E + B1 + B2

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

## 输出要求
- 每个部分必须有实质内容
- 表格用Markdown格式
- 总字数不少于3000字
"""

def node_b1_research(state: ResearchState) -> ResearchState:
    """执行B1方向调研"""
    llm = get_llm()
    topic = state["topic"]
    
    queries = [
        f"{topic} 行业报告 市场规模 2024 2025 Wind Statista IDC Gartner",
        f"{topic} 产业链 上下游 头部企业 营收 年报",
        f"{topic} 行业痛点 发展趋势 预测 艾瑞 赛迪"
    ]
    
    search_content = execute_searches(queries)
    
    messages = [
        SystemMessage(content=B1_PROMPT.format(topic=topic)),
        HumanMessage(content=f"搜索数据：\n{search_content}\n\n请根据以上数据，严格按照模板生成B1行业调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["b1_result"] = response.content
    state["market_size_data"] = response.content[:3000]
    
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
    
    queries = [
        f"{topic} 政策 site:gov.cn OR site:ndrc.gov.cn OR site:mof.gov.cn 2024 2025",
        f"{topic} 政策解读 国务院 发改委 智库报告",
        f"{topic} 监管 法规 标准 实施 影响分析"
    ]
    
    search_content = execute_searches(queries)
    
    messages = [
        SystemMessage(content=E_PROMPT.format(topic=topic)),
        HumanMessage(content=f"搜索数据：\n{search_content}\n\n请严格按照模板生成E方向政策调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["e_result"] = response.content
    state["policy_data"] = response.content[:2000]
    
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
    
    queries = [
        f"{topic} 市场规模 TAM SAM SOM IDC Gartner 艾瑞 易观",
        f"{topic} 应用场景 用户画像 产业链 价值分布",
        f"{topic} 细分市场 定位 消费者 需求 痛点"
    ]
    
    search_content = execute_searches(queries)
    
    # 引用B1的市场规模数据
    b1_data = state.get("market_size_data", "")
    
    messages = [
        SystemMessage(content=B2_PROMPT.format(topic=topic)),
        HumanMessage(content=f"方向B1已产出的市场规模数据：\n{b1_data}\n\n搜索数据：\n{search_content}\n\n请严格按照模板生成B2产品驱动型调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["b2_result"] = response.content
    
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
    
    queries = [
        f"{topic} 竞品对比 功能评测 专业测评",
        f"{topic} vs 对比 性能测试 Benchmark 报告",
        f"{topic} 价格 成本 生态 用户评价"
    ]
    
    search_content = execute_searches(queries)
    
    messages = [
        SystemMessage(content=D_PROMPT.format(topic=topic)),
        HumanMessage(content=f"搜索数据：\n{search_content}\n\n请严格按照模板生成D方向竞品分析报告。")
    ]
    
    response = llm.invoke(messages)
    state["d_result"] = response.content
    state["competitor_data"] = response.content[:2000]
    
    return state


# ============ 节点6：A/C 技术方案/学术综述 ============

AC_PROMPT = """你是技术分析专家，执行方向A/C：技术方案调研或学术综述。

调研主题：{topic}

## 严格模板要求

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
    
    queries = [
        f"{topic} 技术方案 对比 原理 技术路线",
        f"{topic} 最新研究 论文 arxiv IEEE",
        f"{topic} 实验结果 Benchmark 性能"
    ]
    
    search_content = execute_searches(queries)
    
    messages = [
        SystemMessage(content=AC_PROMPT.format(topic=topic)),
        HumanMessage(content=f"搜索数据：\n{search_content}\n\n请严格按照模板生成A/C方向技术调研报告。")
    ]
    
    response = llm.invoke(messages)
    state["ac_result"] = response.content
    
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

## 输出格式
1. 报告标题
2. 执行摘要
3. 各方向章节
4. 参考资料（所有引用的URL列表）

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
    state["full_report"] = f"# {topic} 调研报告\n\n**报告日期**: {now}\n\n---\n\n{response.content}"
    
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
    state["summary"] = content.strip() if isinstance(content, str) else str(content)
    
    return state


# ============ 节点9：生成DOCX ============

def node_generate_docx(state: ResearchState) -> ResearchState:
    """生成DOCX文件"""
    topic = state["topic"]
    report = state["full_report"]
    
    # 生成安全的文件名
    safe_title = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in topic)
    safe_title = safe_title[:50]  # 限制长度
    
    try:
        docx_url = generate_docx_report.invoke({
            "markdown_content": report,
            "report_title": safe_title
        })
        state["docx_url"] = docx_url
    except Exception as e:
        state["docx_url"] = f"生成文档失败: {str(e)}"
    
    # 只返回摘要和下载链接
    return ResearchState(
        topic="",
        directions=[],
        current_direction="",
        b1_result="",
        e_result="",
        b2_result="",
        d_result="",
        ac_result="",
        market_size_data="",
        policy_data="",
        competitor_data="",
        full_report="",
        docx_url=state.get("docx_url", ""),
        summary=state.get("summary", "")
    )


# ============ 路由函数 ============

def route_next(state: ResearchState) -> str:
    """决定下一个节点"""
    directions = state.get("directions", [])
    current = state.get("current_direction", "")
    
    if not directions:
        return "integrate"
    
    # 找到当前方向在列表中的位置
    try:
        idx = directions.index(current)
    except ValueError:
        return "integrate"
    
    # 如果还有下一个方向
    if idx + 1 < len(directions):
        next_direction = directions[idx + 1]
        state["current_direction"] = next_direction
        return f"node_{next_direction.lower()}"
    
    # 所有方向执行完毕
    return "integrate"


def get_first_node(state: ResearchState) -> str:
    """获取第一个方向节点"""
    directions = state.get("directions", [])
    if not directions:
        return "integrate"
    
    first = directions[0]
    state["current_direction"] = first
    return f"node_{first.lower()}"


# ============ 提取topic节点 ============

def node_extract_topic(state: ResearchState) -> ResearchState:
    """从输入中提取调研主题"""
    # 如果已经有topic，直接返回
    if state.get("topic") and state["topic"] != "未指定主题":
        return state
    
    # 尝试从params字段提取（test_run传入的格式）
    if state.get("params"):
        state["topic"] = str(state["params"])
        return state
    
    # 尝试从messages中提取
    messages = state.get("messages", [])
    if messages:
        # 获取最后一条用户消息
        for msg in reversed(messages):
            if hasattr(msg, 'content') and msg.content:
                state["topic"] = str(msg.content)
                return state
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                state["topic"] = str(msg.get('content', ''))
                return state
    
    # 尝试从其他字段提取
    for key in ["input", "query", "question", "text", "content"]:
        if state.get(key):
            state["topic"] = str(state[key])
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
        market_size_data="",
        policy_data="",
        competitor_data="",
        full_report="",
        docx_url="",
        summary=""
    )
    
    # 运行工作流
    final_state = app.invoke(initial_state, {"configurable": {"thread_id": "research_" + topic}})
    
    return {
        "summary": final_state.get("summary", ""),
        "docx_url": final_state.get("docx_url", ""),
        "full_report": final_state.get("full_report", "")
    }


if __name__ == "__main__":
    # 测试
    result = run_research("AI芯片市场")
    print(f"摘要: {result['summary']}")
    print(f"下载链接: {result['docx_url']}")
