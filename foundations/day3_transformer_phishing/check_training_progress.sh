#!/bin/bash
# Check training progress for all models

echo "=========================================="
echo "Training Progress Check"
echo "=========================================="
echo ""

# Check if training processes are running
PROCESSES=$(ps aux | grep "python3 train.py" | grep -v grep | wc -l)
echo "Active training processes: $PROCESSES"
echo ""

# Check each model's log
for MODEL in distilbert bert roberta lora-bert; do
    LOG_FILE="logs/train_${MODEL}.log"

    if [ -f "$LOG_FILE" ]; then
        echo "=== $MODEL ==="

        # Check if process is running
        if [ -f "logs/${MODEL}.pid" ]; then
            PID=$(cat "logs/${MODEL}.pid")
            if ps -p $PID > /dev/null 2>&1; then
                echo "Status: RUNNING (PID: $PID)"
            else
                echo "Status: STOPPED or FINISHED"
            fi
        else
            echo "Status: Unknown (no PID file)"
        fi

        # Get latest progress
        echo ""
        echo "Latest progress:"
        tail -10 "$LOG_FILE" | grep -E "(Epoch|batch|loss|accuracy|auprc|Validation|saved|Killed)" || echo "No progress markers found"

        # Check for completion
        if grep -q "Training completed" "$LOG_FILE"; then
            echo ""
            echo "✅ Training completed!"
        fi

        # Check for errors
        if grep -q "Killed\|Error\|Exception" "$LOG_FILE"; then
            echo ""
            echo "⚠️  Errors detected in log!"
        fi

        echo ""
    else
        echo "=== $MODEL ==="
        echo "No log file found"
        echo ""
    fi
done

echo "=========================================="
echo "Check full logs with: tail -f logs/train_*.log"
echo "=========================================="
