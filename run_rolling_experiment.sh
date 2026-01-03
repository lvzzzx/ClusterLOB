#!/bin/bash
# Rolling walk-forward experiment for available data range
# Data available: 2024-05-01 to 2024-08-31 (123 days)

cd /Users/zjx/Documents/clusterlob
source venv/bin/activate

mkdir -p outputs/rolling_may_aug_2024

./venv/bin/python experiments/02_rolling_walk_forward.py \
    --start-date 2024-05-01 \
    --end-date 2024-08-31 \
    --train-window 7 \
    --features-dir data/daily_features \
    --out-dir outputs/rolling_may_aug_2024 \
    > outputs/rolling_may_aug_2024/run.log 2>&1 &

echo "Process started with PID: $!"
echo "Monitor with: tail -f outputs/rolling_may_aug_2024/run.log"
echo ""
echo "Note: This will process 123 days of data."
echo "Stage 1 (feature extraction) may take 30-60 minutes."
echo "Stage 2 (rolling walk-forward) will then run on ~116 test days."
