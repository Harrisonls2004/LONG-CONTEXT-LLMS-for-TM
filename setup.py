#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Long-Context LLMs for Topic Modeling
"""

from setuptools import setup, find_packages
import os

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="long-context-llm-topic-modeling",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive research project comparing LLM-based and traditional topic modeling methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/LONG-CONTEXT-LLMS-for-TM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch[cuda]>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-topic-analyze=llm_analysis.topic_analyzer:main",
            "run-all-metrics=metrics.run_all_metrics:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.csv"],
    },
)