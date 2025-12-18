# Long-Context LLMs for Topic Modeling

A comprehensive research project comparing Large Language Model (LLM) based topic modeling with traditional topic modeling methods, leveraging long-context capabilities of modern LLMs for enhanced topic discovery and analysis.

## 📋 Project Overview

This project investigates the effectiveness of **Long-Context Large Language Models** for topic modeling tasks, comparing them against traditional methods like LDA, NMF, and neural topic models. Our research focuses on how extended context windows enable better topic coherence and coverage across large document collections.

### Key Research Questions
- How do long-context LLMs perform compared to traditional topic modeling methods?
- What are the optimal strategies for leveraging extended context in topic discovery?
- How can we evaluate and measure the quality of LLM-generated topics?

## 🏗️ Project Structure

```
LONG-CONTEXT-LLMS-for-TM/
├── data/                        # NYT Dataset (primary evaluation data)
├── llm_analysis/               # 🤖 LLM-based Topic Analysis
│   ├── topic_analyzer.py      # Core analyzer with 20+ model support
│   └── results/                # Analysis results (JSON format)
├── evaluation/                 # 📊 Evaluation Framework  
├── metrics/                    # 📈 6 Custom Metrics Suite
└── traditional_models/         # 🔬 TopMost Integration
    ├── models/basic/           # ETM, DecTM, TSCTM, CombinedTM, NSTM, ECRTM
    └── ...                     # Training & evaluation tools
```

## 🚀 Key Features

### 🤖 Long-Context LLM Analysis
- **Extended Context Processing**: Leverage models with 128K-1M+ token context windows
- **Multi-Model Support**: Integration with OpenRouter API supporting 20+ state-of-the-art models
- **Intelligent Sampling**: Context-aware document sampling and processing strategies
- **Hybrid Approaches**: Combine LLM keyword generation with full dataset matching
- **Cost-Effective Processing**: Optimized token usage and batch processing capabilities

### 🔬 Traditional Topic Modeling (TopMost Integration)
Built on the **TopMost** framework ([bobxwu/TopMost](https://github.com/bobxwu/TopMost)) - *A Topic Modeling System Toolkit (ACL 2024 Demo)*:

- **Core Neural Models**: ETM, DecTM, TSCTM, CombinedTM, NSTM, ECRTM
- **Baseline Models**: LDA, NMF for comparison
- **Evaluation Tools**: NPMI, coherence, perplexity metrics

### 📊 Comprehensive Evaluation Framework
- **Statistical Validation**: NPMI, coherence, perplexity, and custom metrics
- **Comparative Analysis**: Direct head-to-head performance evaluation

## 🎯 Supported Models

### 🤖 Long-Context LLM Models (via OpenRouter API)

| Provider | Model | Context Length | Key Features |
|----------|-------|----------------|--------------|
| **OpenAI** | GPT-5, GPT-4o-2024-11-20 | 128K-400K | Industry-leading reasoning |
| **Anthropic** | Claude Sonnet 4, Claude Opus 4.1 | 200K | Superior text analysis |
| **Google** | Gemini 2.5 Pro, Gemini 2.0 Flash | 1M+ | Massive context windows |
| **Meta** | Llama 4 Scout, Llama 4 Maverick | 1M+ | Open-source excellence |
| **DeepSeek** | DeepSeek Chat v3.1, DeepSeek R1 | 128K+ | Cost-effective reasoning |
| **Qwen** | Qwen 3 Coder, Qwen 3-235B | 262K+ | Multilingual support |
| **Others** | Kimi, Mistral Medium 3.1, Grok-4 | 128K-256K | Specialized capabilities |

### 🔬 Traditional Models (TopMost Framework)

| Category | Models Used | Description |
|----------|-------------|-------------|
| **Neural Topic Models** | ETM, DecTM, TSCTM, CombinedTM, NSTM, ECRTM | Advanced neural approaches |
| **Baseline Models** | LDA, NMF | Classical probabilistic methods |

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- OpenRouter API key (for LLM models)
- CUDA-compatible GPU (optional, for traditional models)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/your-username/LONG-CONTEXT-LLMS-for-TM.git
cd LONG-CONTEXT-LLMS-for-TM

# Install dependencies
pip install -r requirements.txt

# Set up OpenRouter API key
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

### Dependencies
```bash
# Core dependencies
pip install openai pandas numpy scipy scikit-learn
pip install torch transformers sentence-transformers
pip install gensim bertopic octis

# For traditional models (TopMost)
pip install -e traditional_models/
```

## 🚀 Quick Start Guide

### 1. LLM-Based Topic Analysis

```python
from llm_analysis.topic_analyzer import TopicAnalyzer, OpenRouterClient

# Initialize with long-context model
client = OpenRouterClient(
    model="google/gemini-2.5-pro-preview",  # 1M+ context
    api_key="your-api-key"
)
analyzer = TopicAnalyzer(client)

# Analyze large document collection
results = analyzer.analyze_text("data/NYT_Dataset.csv")
print(f"Generated {results['total_topics']} topics with {results['cost_analysis']['total_cost']:.4f}$ cost")

# Use hybrid approach for maximum coverage
hybrid_results = analyzer.analyze_text_hybrid("data/NYT_Dataset.csv")
```

### 2. Traditional Topic Modeling (TopMost)

```python
import sys
sys.path.append('traditional_models')
import topmost

# Load and preprocess dataset
dataset = topmost.BasicDataset("data/NYT_Dataset.csv")
preprocessor = topmost.Preprocess()
processed_dataset = preprocessor.preprocess(dataset)

# Train neural topic models for comparison
models = {
    'ETM': topmost.ETMTrainer(),
    'DecTM': topmost.DecTMTrainer(),
    'TSCTM': topmost.TSCTMTrainer(),
    'CombinedTM': topmost.CombinedTMTrainer()
}

results = {}
for name, trainer in models.items():
    model = trainer.train(processed_dataset)
    results[name] = model
```

### 3. Run Evaluation

```bash
# Compare LLM vs Traditional models
python evaluation/topic_eval_combine.py \
    --llm_results llm_analysis/results/ \
    --traditional_results traditional_models/results/
```


## 🔬 Key Research Findings

We conducted a comprehensive comparison between Long-Context LLMs and Neural Topic Models (NTMs) using a multifaceted evaluation framework that includes traditional statistical metrics (NPMI, Topic Diversity) and LLM-based subjective evaluation (coherence, conciseness, informativeness), along with assignment accuracy assessment.

**Main Results:** LLMs substantially outperform NTMs on diversity and subjective evaluation metrics, indicating markedly higher topic quality. While some NTM models achieve higher assignment accuracy, manual inspection reveals prevalent issues such as topic mixing and redundancy in NTM outputs. Among LLMs, Claude Sonnet4 leads on nearly all metrics, demonstrating that larger context windows and stronger model capabilities improve performance. Our findings empirically support the claim that zero-shot LLMs can match or surpass strong NTMs in readability and interpretability, while offering advantages in ease of use, flexible topic representations, and support for multimodal inputs.

## 📚 Dataset

**Primary Evaluation Dataset:**
- **New York Times (NYT)**: News articles dataset used for comprehensive evaluation
  - `NYT_Dataset.csv`: Full dataset  
  - `NYT_sampled.csv`: Sampled subset for testing


## 🙏 Acknowledgments

- **[TopMost](https://github.com/bobxwu/TopMost)** by bobxwu - The comprehensive topic modeling toolkit that powers our traditional model implementations (ACL 2024 Demo)
- **OpenRouter** - For providing unified access to multiple state-of-the-art LLM APIs
- **Research Community** - For open datasets and evaluation methodologies that enable reproducible research
- **Model Providers** - OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen, and others for advancing long-context capabilities
