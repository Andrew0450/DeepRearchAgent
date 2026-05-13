# Coze 图片图表 OCR/视觉识别插件开发提示词

请为管线调研 Agent 开发一个图片图表数据提取工具，供调研报告生成流程调用。

## 工具名称

`extract_chart_data_from_image`

## 使用场景

网页、PDF、公众号文章、研报截图中经常把关键数据放在图片里，例如市场规模柱状图、年份折线图、份额饼图、竞品参数截图、政策时间线。当前 Agent 只能拿到图片 URL，无法可靠读取图片中的文字和数据。该工具需要对图片进行 OCR 和视觉图表理解，输出可复核的结构化数据。

## 输入参数

```json
{
  "image_url": "https://example.com/chart.png",
  "source_url": "https://example.com/report-page",
  "topic": "调研主题",
  "expected_data_type": "market_size | market_share | competitor_specs | policy_timeline | table | other",
  "language": "zh-CN"
}
```

字段说明：
- `image_url`：图片地址，必填。
- `source_url`：图片所在网页或报告页面 URL，必填，用于来源标注。
- `topic`：当前调研主题，辅助判断图表上下文。
- `expected_data_type`：期望提取的数据类型。
- `language`：默认 `zh-CN`。

## 输出格式

必须返回严格 JSON：

```json
{
  "status": "success | partial | failed",
  "image_url": "https://example.com/chart.png",
  "source_url": "https://example.com/report-page",
  "chart_type": "bar | line | pie | table | timeline | screenshot | unknown",
  "title": "识别出的图表标题",
  "unit": "亿元 / 亿美元 / 万台 / % / 未识别",
  "data": [
    {
      "label": "2024",
      "series": "中国市场规模",
      "value": "123.4",
      "unit": "亿元",
      "confidence": 0.86
    }
  ],
  "ocr_text": "图片中识别出的全部文字，保留换行",
  "extraction_notes": [
    "说明识别依据、可能误差、被遮挡或低清晰度问题"
  ],
  "citation": "[图片来源](https://example.com/report-page)",
  "must_verify": true
}
```

## 质量要求

1. 如果图片清晰度不足、文字过小、图例不完整，返回 `status=partial` 或 `failed`，不要猜数。
2. 图表数据必须带 `confidence`，低于 0.75 的数据在报告中只能进入“待核验线索”。
3. 对柱状图、折线图、饼图，尽量提取标题、单位、横轴、纵轴、图例、年份、数值。
4. 对表格截图，按行列输出结构化数据。
5. 对竞品参数图，输出产品名、参数项、参数值、价格、来源。
6. 输出中必须保留 `source_url` 和 `image_url`，方便人工核验。
7. 不允许根据常识补全图片里看不清的数据。

## Agent 集成方式

在主调研流程中：
1. `fetch_url_images` 或 `image_search` 获取图片 URL。
2. 对疑似图表图片调用 `extract_chart_data_from_image`。
3. 成功提取的数据进入正文，但必须标注：“数据来源于图片，经 OCR/视觉识别提取，需人工核验”。
4. `partial` 或低置信度结果进入“图片数据待 OCR/视觉识别线索”或“待人工核验”小节，不进入核心结论。
