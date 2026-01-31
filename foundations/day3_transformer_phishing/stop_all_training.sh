#!/bin/bash
# Stop all training processes gracefully

echo "=========================================="
echo "Stopping All Training Processes"
echo "=========================================="
echo ""

# Kill all python3 train.py processes
pkill -f "python3 train.py"

# Wait a moment
sleep 2

# Check if any processes remain
REMAINING=$(ps aux | grep "python3 train.py" | grep -v grep | wc -l)

if [ $REMAINING -eq 0 ]; then
    echo "✅ All training processes stopped successfully"
else
    echo "⚠️  Some processes may still be running. Trying force kill..."
    pkill -9 -f "python3 train.py"
    sleep 1
    echo "✅ Force kill completed"
fi

echo ""
echo "Active training processes: $(ps aux | grep 'python3 train.py' | grep -v grep | wc -l)"
echo ""
