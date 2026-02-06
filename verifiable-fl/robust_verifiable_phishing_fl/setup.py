"""
Setup script for Robust Verifiable Federated Learning for Phishing Detection
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_file(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

setup(
    name="robust-verifiable-phishing-fl",
    version="1.0.0",
    description="Robust and Verifiable Federated Learning for Phishing Detection",
    long_description=read_file("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Ahmad Whafa Azka Al Azkiyai",
    author_email="azka.alazkiyai@outlook.com",
    url="https://github.com/your-repo/robust-verifiable-phishing-fl",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "flwr>=1.5.0",
        "cryptography>=41.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "adversarial": [
            "torchattacks>=3.5.0",
            "advertorch>=0.2.3",
        ],
        "viz": [
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rvpfl-demo=examples.robust_verifiable_fl_demo:main",
            "rvpfl-experiment=experiments.run_combined_defenses:main",
            "rvpfl-test=tests.run_all_tests:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="federated-learning phishing-detection zero-knowledge-proofs adversarial-robustness byzantine-robust",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/robust-verifiable-phishing-fl/issues",
        "Source": "https://github.com/your-repo/robust-verifiable-phishing-fl",
        "Documentation": "https://robust-verifiable-phishing-fl.readthedocs.io/",
    },
)
