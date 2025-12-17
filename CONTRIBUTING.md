# Contributing to Long-Context LLMs for Topic Modeling

We welcome contributions to this research project! Here's how you can help advance the state-of-the-art in topic modeling.

## 🎯 Areas of Interest

### High Priority
- **New Evaluation Metrics**: Design metrics specifically for LLM topic modeling
- **Long-Context Model Integration**: Add support for new models with extended context
- **Cross-lingual Evaluation**: Extend evaluation to non-English datasets
- **Computational Efficiency**: Optimize processing for large-scale datasets

### Medium Priority
- **Visualization Tools**: Interactive topic exploration and comparison
- **Benchmark Datasets**: Curate new datasets for evaluation
- **Documentation**: Improve tutorials and examples
- **Bug Fixes**: Address issues in existing code

## 🛠️ Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/LONG-CONTEXT-LLMS-for-TM.git
   cd LONG-CONTEXT-LLMS-for-TM
   ```

2. **Create Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up API Keys**
   ```bash
   export OPENROUTER_API_KEY="your-key-here"
   ```

## 📝 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all public functions
- Include type hints where appropriate

### Testing
- Add tests for new functionality
- Ensure existing tests pass
- Test with multiple datasets and models

### Documentation
- Update README.md for new features
- Add inline comments for complex logic
- Include usage examples

## 🔄 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Implement your feature
   - Add tests
   - Update documentation

3. **Test Thoroughly**
   ```bash
   python -m pytest tests/
   python metrics/run_all_metrics.py  # If applicable
   ```

4. **Submit Pull Request**
   - Provide clear description
   - Reference related issues
   - Include performance benchmarks if applicable

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version and OS
- Model and dataset used
- Complete error traceback
- Steps to reproduce
- Expected vs actual behavior

## 💡 Feature Requests

For new features, please provide:
- Clear use case description
- Proposed implementation approach
- Expected impact on performance
- Compatibility considerations

## 📊 Research Contributions

We especially welcome:
- **Novel Evaluation Metrics**: New ways to measure topic quality
- **Comparative Studies**: Systematic evaluation across models/datasets
- **Theoretical Analysis**: Mathematical foundations for LLM topic modeling
- **Empirical Findings**: Insights from large-scale experiments

## 🤝 Community Guidelines

- Be respectful and constructive
- Focus on technical merit
- Provide evidence for claims
- Help others learn and contribute

## 📞 Contact

- **Issues**: Use GitHub Issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Email**: [maintainer-email] for sensitive matters

Thank you for contributing to advancing topic modeling research! 🚀