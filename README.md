# 管线调研 Agent

基于 LangGraph 工作流构建的智能调研 Agent，对齐 `assets/README.md` 中的通用调研 SOP v3.1，目标是输出结构化、可追溯、可复核的调研报告。

## 核心能力

### 1. 方向识别

根据用户输入自动识别 SOP 中的 6 种调研方向：

| 方向 | 类型 | 适用场景 |
|------|------|----------|
| **A** | 技术方案调研 | 技术原理、方案对比、实验效果、创新点 |
| **B1** | 行业驱动型调研 | 行业结构、产业链、规模、竞争格局、痛点 |
| **B2** | 产品驱动型调研 | 产品市场量化、应用场景、STP、产业链价值 |
| **C** | 学术综述 | 方法演进、代表论文、研究空白、未来方向 |
| **D** | 竞品/产品分析 | 功能对比、性能/成本/生态对比、推荐方案、迁移路径 |
| **E** | 政策与趋势 | 政策梳理、领导人/关键人物讲话、影响方向、落地进展、风险 |

### 2. 工作流执行

当前主 Agent 采用 LangGraph StateGraph：

```text
用户输入
→ 提取主题
→ 识别方向
→ 只执行本轮队列中的第一个方向
→ 整合该方向报告
→ 生成摘要
→ 生成 DOCX
```

说明：
- A 和 C 当前在同一个技术/学术节点中处理，输出会按实际方向拆分内容。
- `assets/*方向*.md` 是目标效果参考样例，主流程会读取这些样例的标题结构作为报告粒度参考，但不会复制样例数据。
- Semantic Scholar、秘塔、微信公众号/搜狗微信等需要专门 API 或平台能力的信源暂未接入，当前以通用网页搜索、网页正文抓取和平台 OCR 技能为主。
- 为保留搜索次数和精度，同时避免单次运行超时，当前主流程每次只执行一个方向；报告结尾会提示下一建议方向，用户确认后再继续。

### 3. 数据来源规范

Agent 会遵循以下约束：

1. 按方向生成更密集的权威源查询，再抓取关键网页正文用于整合。
2. 每条关键数据尽量使用 `[来源名称](URL)` 格式标注。
3. 超过 2 年的数据需要标注数据年份或“数据截至 XX 年”。
4. 估算值必须给出公式、参数来源、假设，并标注 `⚠️估算值`。
5. 未检索到的数据必须写“未检索到公开数据”或“需补充”，禁止编造。
6. 报告末尾包含参考资料列表、图片 OCR 数据/待核验线索和质量审计提示。

### 图片/图表数据

当前主流程可以：

- 从网页正文中提取图片 URL。
- 使用图片搜索寻找市场规模图、份额图、产业链图、竞品对比图等线索。
- 在 Coze 平台调用已配置的 `extract-chart-data` OCR/视觉识别技能，提取图表标题、单位、数据点、OCR 文本和置信度。
- 将 `success` 且置信度 >= 0.75 的图片识别数据写入正文，并标注“数据来源于图片，经 OCR/视觉识别提取，需人工核验”。
- 将 `partial`、`failed` 或低置信度结果写入“图片数据待核验线索”。

注意：本地 LangGraph 代码目前只收集图片 URL；OCR 识别由 Coze 平台已配置的 `OCR识别` 技能负责调度。若平台未自动触发该技能，再考虑在主程序里显式调用 `OCR识别/scripts/extract_chart_data.py`。

## 快速开始

### 本地运行完整工作流

```bash
bash scripts/local_run.sh -m flow -i "请调研一下人工智能芯片市场"
```

### 运行指定节点

```bash
bash scripts/local_run.sh -m node -n extract_topic -i "请调研一下人工智能芯片市场"
```

当前主 Agent 节点：

- `extract_topic` - 提取调研主题
- `identify` - 识别调研方向
- `node_b1` - 行业驱动型调研
- `node_e` - 政策与趋势调研
- `node_b2` - 产品驱动型调研
- `node_d` - 竞品/产品分析
- `node_ac` - 技术方案/学术综述
- `integrate` - 整合报告
- `summary` - 生成摘要
- `docx` - 生成 DOCX 文件

### 连续调研方式

如果一次主题识别出多个方向，Agent 会先执行优先级最高的一个方向，并在摘要中提示下一方向。例如：

```text
本次已完成方向 B1。如果要继续方向 E 的调研，请告诉我“继续调研 [主题] 方向 E”。
```

这样不会减少单方向搜索次数，报告深度更稳定，也更不容易触发平台超时。

### 启动 HTTP 服务

```bash
bash scripts/http_run.sh -p 5000
```

主要接口：

- `POST /run`
- `POST /stream_run`
- `POST /node_run/{node_id}`
- `POST /cancel/{run_id}`
- `GET /health`
- `POST /v1/chat/completions`

## 输出格式

最终响应包含：

```markdown
### 报告摘要
[200字以内核心发现]

### 完整报告
[保留在返回状态 full_report 中，便于调试与复核]

### DOCX下载链接
[报告标题.docx](下载链接)
```

DOCX 中包含完整报告、来源列表和质量审计提示。PPT 提示词能力在 `src/graphs/research_flow.py` 中已有实验性实现，但当前主 Agent 默认不输出 PPT。

## 项目结构

```text
.
├── config/
│   └── agent_llm_config.json        # 模型与系统提示词配置
├── src/
│   ├── agents/
│   │   └── agent.py                 # 当前主 Agent 工作流
│   ├── graphs/
│   │   └── research_flow.py         # 实验性 SOP 工作流：报告 + PPT提示词 + 参考文献
│   ├── tools/
│   │   ├── web_search_tool.py       # 网络搜索
│   │   ├── fetch_url_tool.py        # URL 正文抓取
│   │   └── document_generation_tool.py
│   ├── storage/
│   └── main.py                      # HTTP/本地入口
├── assets/
│   ├── README.md                    # 通用调研 SOP v3.1
│   ├── B1-行业定位.md               # B1 目标效果参考
│   └── B2-市场量化.md               # B2 目标效果参考
├── OCR识别/
│   ├── SKILL.md                     # Coze 图片图表 OCR/视觉识别技能说明
│   └── scripts/extract_chart_data.py
├── scripts/
└── pyproject.toml
```

## 待扩展项

- 接入 Semantic Scholar/OpenAlex 等论文结构化检索。
- 接入秘塔搜索 API。
- 接入微信公众号/搜狗微信搜索。
- 如 Coze 平台不会自动触发 OCR 技能，可在 `src/agents/agent.py` 中增加显式脚本调用包装。
- 将 `research_flow.py` 中的 PPT 提示词输出稳定并入主 Agent。
- 增强自动化质量校验，对无 URL 数字、过期数据、估算缺公式等情况给出更严格的结构化告警。
