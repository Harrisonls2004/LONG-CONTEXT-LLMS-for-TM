#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件 - NYT数据集主题层次化处理
参考topic_evaluator.py的配置方式
"""

import os

# 模型配置（参考topic_evaluator.py）
RECOMMENDED_MODELS = [
    "qwen/qwen3-14b:free",           # 默认推荐，性能好（与topic_evaluator.py一致）
    "qwen/qwen3-coder:free",         # 代码理解能力强
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1-0528:free",
    "deepseek/deepseek-chat-v3.1:free"
]

# 默认模型（与topic_evaluator.py保持一致）
DEFAULT_MODEL = "qwen/qwen3-14b:free"

# 基本配置
RATE_LIMIT_INTERVAL = 2.0  # 秒
MAX_RETRIES = 3
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TEMPERATURE = 0.3
DEFAULT_ENCODING_OPTIONS = ['utf-8', 'gbk', 'latin-1', 'cp1252']

# 数据文件路径
DEFAULT_INPUT_FILE = r"../../data/NYT_Dataset.csv"

# 提示词模板配置（英文版本）
# 可选的提示词模板 - 根据需要选择使用

# 模板1: 适度控制版本（推荐）
PROMPT_TEMPLATE_MODERATE = """You are a professional news topic analysis expert. Please generate a focused topic hierarchy for the following news article.

**Article Information:**
Title: {title}
Primary Topic: {primary_topic}
Keywords: {keywords}
Content: {text_preview}...

**Requirements:**
1. Generate 2-5 secondary topics that represent the main aspects of the article
2. Generate 3-8 tertiary topics that provide more specific details
3. Topics should be specific, accurate, and directly relevant to the article content
4. Secondary topics should be subdivisions of the primary topic
5. Tertiary topics should be further subdivisions of secondary topics
6. Focus on quality over quantity - only include truly relevant topics

**Output Format (strictly follow JSON format):**
```json
{{
    "secondary_topics": [
        "Specific secondary topic 1",
        "Specific secondary topic 2",
        "Specific secondary topic 3"
    ],
    "tertiary_topics": [
        "Specific tertiary topic 1",
        "Specific tertiary topic 2",
        "Specific tertiary topic 3",
        "Specific tertiary topic 4"
    ]
}}
```

Please ensure:
- Generate only the most important and relevant topics
- Topic descriptions are concise and clear, typically 2-6 words
- Avoid redundant or overly similar topics
- Output must be valid JSON format"""

# 模板2: 严格控制版本
PROMPT_TEMPLATE_STRICT = """You are a professional news topic analysis expert. Please generate a concise topic hierarchy for the following news article.

**Article Information:**
Title: {title}
Primary Topic: {primary_topic}
Keywords: {keywords}
Content: {text_preview}...

**Requirements:**
1. Generate exactly 2-3 secondary topics that capture the main themes
2. Generate exactly 3-5 tertiary topics that provide specific details
3. Topics must be highly relevant and non-redundant
4. Secondary topics should be clear subdivisions of the primary topic
5. Tertiary topics should be specific aspects of secondary topics

**Output Format (strictly follow JSON format):**
```json
{{
    "secondary_topics": [
        "Secondary topic 1",
        "Secondary topic 2"
    ],
    "tertiary_topics": [
        "Tertiary topic 1",
        "Tertiary topic 2",
        "Tertiary topic 3"
    ]
}}
```

Please ensure:
- Generate only the most essential topics
- Avoid redundancy and overlap
- Topic descriptions are concise (2-4 words)
- Output must be valid JSON format"""

# 模板3: 原始版本（生成最多主题）
PROMPT_TEMPLATE_MAXIMUM = """You are a professional news topic analysis expert. Please generate a detailed topic hierarchy for the following news article.

**Article Information:**
Title: {title}
Primary Topic: {primary_topic}
Keywords: {keywords}
Content: {text_preview}...

**Requirements:**
1. Based on the article content, generate as many secondary and tertiary topics as possible
2. Do not limit the number of topics - generate all relevant topics
3. Topics should be specific, accurate, and reflect different aspects of the article
4. Secondary topics should be subdivisions of the primary topic
5. Tertiary topics should be further subdivisions of secondary topics

**Output Format (strictly follow JSON format):**
```json
{{
    "secondary_topics": [
        "Specific secondary topic 1",
        "Specific secondary topic 2",
        "Specific secondary topic 3"
    ],
    "tertiary_topics": [
        "Specific tertiary topic 1",
        "Specific tertiary topic 2",
        "Specific tertiary topic 3",
        "Specific tertiary topic 4"
    ]
}}
```

Please ensure:
- The number of generated topics is not fixed, determined by the richness of article content
- Topic descriptions are concise and clear, typically 2-6 words
- All topics are highly relevant to the article content
- Output must be valid JSON format"""

# 模板4: 聚焦核心版本（限制数量）
PROMPT_TEMPLATE_FOCUSED = """You are a professional news topic analysis expert. Please generate a focused topic hierarchy for the following news article.

**Article Information:**
Title: {title}
Primary Topic: {primary_topic}
Keywords: {keywords}
Content: {text_preview}...

**Requirements:**
1. Generate exactly 2-4 secondary topics that represent the core themes and key aspects
2. Generate exactly 3-6 tertiary topics that provide important specific details
3. Focus on relevance and importance - include only topics that truly matter
4. Avoid redundant, overly similar, or tangential topics
5. Secondary topics should be clear subdivisions of the primary topic
6. Tertiary topics should be meaningful aspects of secondary topics
7. Quality over quantity - each topic should add genuine value

**Output Format (strictly follow JSON format):**
```json
{{
    "secondary_topics": [
        "Core secondary topic 1",
        "Core secondary topic 2",
        "Core secondary topic 3"
    ],
    "tertiary_topics": [
        "Important tertiary topic 1",
        "Important tertiary topic 2",
        "Important tertiary topic 3",
        "Important tertiary topic 4"
    ]
}}
```

Please ensure:
- Generate 2-4 secondary topics and 3-6 tertiary topics
- Only include topics that are genuinely important and relevant
- Avoid redundancy and overlap between topics
- Topic descriptions are concise and clear, typically 2-6 words
- Each topic should contribute to understanding the article
- Output must be valid JSON format"""

# 默认使用聚焦核心版本
PROMPT_TEMPLATE = PROMPT_TEMPLATE_FOCUSED

# 其他配置
TEXT_PREVIEW_LENGTH = 1000
TOPIC_SEPARATOR = '; '
ERROR_MARKER = 'ERROR'
