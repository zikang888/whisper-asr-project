#!/bin/bash

echo "========================================"
echo "Training Status Monitor"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

while true; do
    clear
    echo "========================================"
    echo "Training Status Monitor - $(date)"
    echo "========================================"
    echo ""
    
    # Check if training process is running
    if ps aux | grep -v grep | grep "python train.py" > /dev/null; then
        echo "✓ Training is running"
        echo ""
        
        # Get process info
        ps aux | grep -v grep | grep "python train.py" | head -1
        echo ""
        
        # Get last 30 lines of output
        if [ -f train_output.log ]; then
            echo "Last 30 lines of training output:"
            echo "-----------------------------------"
            tail -30 train_output.log 2>/dev/null || echo "No log file yet"
            echo ""
        fi
        
        # Check GPU usage
        if command -v nvidia-smi &> /dev/null; then
            echo "GPU Status:"
            echo "-----------------------------------"
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo "Cannot query GPU"
            echo ""
        fi
        
        # System load
        echo "System Load:"
        echo "-----------------------------------"
        uptime
        echo ""
        
    else
        echo "✗ Training process not found"
        echo ""
        if [ -f train_output.log ]; then
            echo "Last 50 lines of training output:"
            echo "-----------------------------------"
            tail -50 train_output.log 2>/dev/null
            echo ""
        fi
        
        if [ -f output/checkpoint-* ]; then
            echo "✓ Checkpoints found in output directory"
        fi
    fi
    
    echo "Press Ctrl+C to exit, waiting 60 seconds before next check..."
    sleep 60
done
