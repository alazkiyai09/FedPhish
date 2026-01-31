#!/bin/bash
# Setup script for Day 3 Transformer Phishing Detection

set -e

echo "🔧 Setting up environment for Day 3..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install torch transformers datasets peft accelerate scikit-learn numpy
pip install wandb onnx onnxruntime pytest pyyaml matplotlib seaborn tqdm

# Download HuggingFace dataset
echo "📥 Downloading HuggingFace phishing email dataset..."
python data/download_hf_dataset.py

# Generate synthetic data
echo "📧 Generating synthetic test data..."
python data/generate_synthetic_data.py

echo "✅ Setup complete!"
echo "   To activate the environment: source venv/bin/activate"
