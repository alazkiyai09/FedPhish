#!/bin/bash
# Train all 4 transformer models with recommended settings
# Usage: bash train_all_models.sh [dataset] [epochs] [batch_size]

# Default arguments
DATASET=${1:-"data/processed/phishing_emails_2k.csv"}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-8}

echo "=========================================="
echo "Training All Transformer Models"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Create logs directory
mkdir -p logs

# Function to train a single model
train_model() {
    local MODEL=$1
    local LOG_FILE="logs/train_${MODEL}.log"

    echo "Starting training for $MODEL..."
    echo "Log file: $LOG_FILE"

    python3 train.py \
        --model "$MODEL" \
        --data "$DATASET" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --no-wandb \
        > "$LOG_FILE" 2>&1 &

    local PID=$!
    echo "Started $MODEL training (PID: $PID)"
    echo ""

    # Store PID for monitoring
    echo "$PID" > "logs/${MODEL}.pid"
}

# Train all 4 models
train_model "distilbert"
sleep 2
train_model "bert"
sleep 2
train_model "roberta"
sleep 2
train_model "lora-bert"

echo "=========================================="
echo "All 4 models training in background!"
echo "=========================================="
echo ""
echo "Monitor training with:"
echo "  tail -f logs/train_*.log"
echo ""
echo "Check progress with:"
echo "  bash check_training_progress.sh"
echo ""
echo "Stop all training with:"
echo "  bash stop_all_training.sh"
echo ""
