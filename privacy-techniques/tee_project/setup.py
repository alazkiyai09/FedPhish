"""
TEE ML: Trusted Execution Environment for Machine Learning
===========================================================

A simulation framework for privacy-preserving ML using Trusted Execution
Environments (Intel SGX, ARM TrustZone) as part of the HT2ML hybrid system.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="tee-ml",
    version="0.1.0",
    author="Privacy-Preserving ML Researcher",
    author_email="researcher@example.com",
    description="TEE simulation for privacy-preserving ML operations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tee-ml",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "psutil>=5.8.0",
        "cryptography>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
        ],
    },
)
