# FedPhish Research Paper Materials

This directory contains all materials for the FedPhish research paper submission.

## 📁 Directory Structure

```
fedphish-paper/
├── experiments/           # Experiment configurations
│   └── configs/          # YAML experiment configs
├── figures/              # Figure generation
│   ├── output/           # Generated figures (PDF + PNG)
│   └── src/              # Figure generation scripts
├── tables/               # Table generation
│   ├── output/           # Generated tables (LaTeX + CSV)
│   └── src/              # Table generation scripts
├── paper/                # LaTeX paper source
│   ├── fedphish_template.tex
│   └── references.bib
├── supplementary/        # Supplementary materials
│   └── appendix_a.tex
└── scripts/              # Utility scripts
    ├── run_single_exp.py
    ├── analyze_results.py
    └── plot_results.py
```

## 🚀 Quick Start

### Generate All Figures and Tables

```bash
# Quick test (uses placeholder results, no experiments)
python run_full_evaluation.py --quick-test

# Full evaluation (runs all experiments)
python run_full_evaluation.py --runs 5
```

### Generate Individual Components

```bash
# Tables only
python generate_all_tables.py

# Figures only
python generate_all_figures.py
```

## 📊 Deliverables

### Figures (6)

1. **fig1_architecture.pdf** - FedPhish system architecture diagram
2. **fig2_convergence.pdf** - Training convergence comparison
3. **fig3_non_iid.pdf** - Non-IID data impact analysis
4. **fig4_privacy_pareto.pdf** - Privacy-accuracy Pareto frontier
5. **fig5_attacks.pdf** - Attack success rate over rounds
6. **fig6_fairness.pdf** - Per-bank fairness analysis

### Tables (4)

1. **table1_detection.tex** - Detection performance comparison
2. **table2_privacy.tex** - Privacy-utility tradeoff
3. **table3_robustness.tex** - Robustness against attacks
4. **table4_overhead.tex** - Computational and communication overhead

### Paper Sections

- **Introduction** - Motivation and contributions
- **Background** - Phishing detection, FL challenges
- **Threat Model** - Adversaries and privacy requirements
- **System Design** - Architecture, privacy mechanisms, ZK proofs, Byzantine defenses
- **Implementation** - System details
- **Evaluation** - Detection, privacy, robustness, fairness, overhead
- **Related Work** - FL for security, privacy-preserving ML
- **Conclusion** - Summary and future work

### Supplementary Materials

- **Appendix A** - Additional experimental results, ablation studies, algorithm pseudocode, security proofs

## 📈 Experimental Design

### Datasets

- **Combined Phishing Dataset**: 100K samples from 5 banks
- **Non-IID Analysis**: Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0, 3.0, 10.0}

### Methods

- Local (Per-Bank)
- Centralized
- FedAvg
- FedProx
- FedPhish (DP only)
- FedPhish (DP + HE)
- **FedPhish (DP + HE + TEE)** ← Our method

### Metrics

- Accuracy, AUPRC, F1 Score, FPR @ 95% TPR
- Privacy budget (ε)
- Attack success rate
- Fairness index (accuracy variance)
- Communication/computation overhead

### Statistical Significance

- 5 independent runs per experiment
- 95% confidence intervals
- Mean ± std reported

## 🎯 Target Venues

Top-tier security and ML conferences:
- **ACM CCS** - ACM Conference on Computer and Communications Security
- **USENIX Security** - USENIX Security Symposium
- **IEEE S&P** - IEEE Symposium on Security and Privacy
- **NDSS** - Network and Distributed System Security Symposium
- **NeurIPS** - Neural Information Processing Systems
- **ICML** - International Conference on Machine Learning

## 🔧 Requirements

```bash
# Python dependencies
pip install matplotlib numpy pandas scipy seaborn pyyaml

# LaTeX (for compiling paper)
# - texlive-full (Ubuntu/Debian)
# - mactex (macOS)
# - MiKTeX (Windows)
```

## 📝 Paper Template

The paper uses the ACM conference format:

```bash
# Compile paper
cd paper/
pdflatex fedphish_template.tex
bibtex fedphish_template
pdflatex fedphish_template.tex
pdflatex fedphish_template.tex
```

## 🔬 Reproducibility

All experiments are configured with YAML files in `experiments/configs/`:

- `detection_comparison.yaml` - Compare FedPhish vs baselines
- `privacy_utility.yaml` - Privacy-accuracy tradeoff
- `robustness.yaml` - Byzantine attack defense
- `overhead.yaml` - Computational overhead measurement
- `non_iid.yaml` - Non-IID data impact

## ✅ Status

- [x] Figure generation scripts
- [x] Table generation scripts
- [x] Experiment configuration files
- [x] Paper template
- [x] Supplementary materials
- [x] Master evaluation pipeline
- [ ] Full experimental runs (requires GPU access)

## 📧 Contact

For questions about FedPhish research materials, contact [authors].

## 📄 License

FedPhish research materials © 2025
