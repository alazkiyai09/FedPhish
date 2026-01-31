"""
Setup script for Human-Aligned Phishing Explanation System.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="human-aligned-explanation",
    version="1.0.0",
    description="Human-aligned explanations for phishing detection following cognitive processing patterns",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PhD Portfolio Project",
    author_email="your-email@example.com",
    url="https://github.com/yourusername/human-aligned-explanation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "streamlit>=1.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "jupyter>=1.0.0",
        ],
        "xai": [
            "shap>=0.42.0",
            "captum>=0.7.0",
        ],
        "privacy": [
            "opacus>=1.3.0",
            "tenseal>=0.3.0",
        ],
        "nlp": [
            "transformers>=4.30.0",
            "tokenizers>=0.13.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="phishing detection explainable-ai xai cybersecurity human-computer-interaction",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/human-aligned-explanation/issues",
        "Source": "https://github.com/yourusername/human-aligned-explanation",
        "Documentation": "https://github.com/yourusername/human-aligned-explanation/docs",
    },
)
