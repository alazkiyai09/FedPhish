from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="verifiable-fl",
    version="0.1.0",
    author="Research Team",
    author_email="research@example.com",
    description="Verifiable Federated Learning with Zero-Knowledge Proofs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/verifiable-fl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "mypy"],
    },
    entry_points={
        "console_scripts": [
            "verifiable-fl=experiments.run_verifiable_fl:main",
        ],
    },
)
