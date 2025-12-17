# NYT数据集主题层次化处理工具

使用LLM为NYT数据集的每一行生成二级和三级主题，并添加到CSV文件的新列中。

## 🚀 快速使用

### 1. 设置API密钥

打开 `process.py` 文件，在配置区域修改API密钥：

```python
# API密钥（参考topic_evaluator.py的方式）
API_KEY = "your_actual_api_key_here"  # 请替换为您的实际API密钥
```

**如何获取API密钥：**
1. 访问 [OpenRouter官网](https://openrouter.ai/)
2. 注册并登录账号
3. 在控制台获取您的API密钥
4. 将密钥替换到上面的 `API_KEY` 变量中

### 2. 配置处理参数

打开 `process.py` 文件，在顶部的配置区域修改：

```python
# ==================== 配置区域 ====================
# 1. 数据文件路径（修改为您的文件路径）
input_file = "../../data/NYT_Dataset.csv"

# 2. 使用的模型（取消注释选择一个）
selected_model = DEFAULT_MODEL  # 默认: qwen/qwen3-14b:free
# selected_model = "qwen/qwen3-coder:free"                  # 代码理解能力强
# selected_model = "meta-llama/llama-3.3-70b-instruct:free" # 大模型

# 3. 处理设置
process_all_data = True    # True=处理全部数据, False=只处理部分数据
test_rows = 10            # 如果process_all_data=False，处理多少行
save_every = 50           # 每处理多少行保存一次
# ==================== 配置区域结束 ====================
```

### 3. 运行处理

```bash
# 直接运行
python process.py

# 或者使用命令行参数
python process.py --input "your_file.csv" --model "qwen/qwen3-coder:free" --max-rows 100
```

## ✨ 功能特点

- 🤖 使用OpenRouter API调用免费LLM模型
- 🌍 **英文提示词**，提高模型理解准确性
- 📊 逐行处理CSV数据，生成层次化主题
- ⚙️  **脚本内配置**，方便修改模型和数据文件
- 📝 **直接在原文件上修改**，添加新列
- 💾 **自动备份**原文件，支持断点续传
- 🔄 包含重试机制和速率限制
- 📈 实时显示处理进度

## 📊 输出格式

**直接在原CSV文件上添加两个新列：**
- `secondary_topics`: 二级主题（用分号分隔）
- `tertiary_topics`: 三级主题（用分号分隔）

**自动创建备份文件：**
- `原文件名_backup.csv`: 处理前的原始数据备份

## 🤖 推荐模型

- `qwen/qwen3-14b:free` (默认，性能好)
- `qwen/qwen3-coder:free` (代码理解能力强)
- `meta-llama/llama-3.3-70b-instruct:free` (大模型)
- `google/gemini-2.0-flash-exp:free` (Google模型)
- `deepseek/deepseek-r1-0528:free` (DeepSeek模型)

## ⚙️ 配置说明

### 修改数据文件
```python
input_file = "你的文件路径.csv"
```

### 选择模型
```python
# 取消注释选择模型
selected_model = "qwen/qwen3-coder:free"
```

### 处理模式
```python
process_all_data = False  # 改为False进入测试模式
test_rows = 20           # 测试模式处理行数
```

## 💡 使用建议

1. **首次使用**：设置 `process_all_data = False` 和 `test_rows = 5` 进行测试
2. **选择模型**：推荐使用 `qwen/qwen3-14b:free` 或 `qwen/qwen3-coder:free`
3. **大数据集**：设置较大的 `save_every` 值（如100）提高效率
4. **网络不稳定**：设置较小的 `save_every` 值（如20）避免数据丢失

## 🔧 命令行参数（可选）

- `--input, -i`: 输入CSV文件路径
- `--output, -o`: 输出CSV文件路径
- `--model`: LLM模型名称
- `--max-rows`: 最大处理行数
- `--start-row`: 开始处理的行号
- `--save-interval`: 保存间隔
- `--api-key`: API密钥
