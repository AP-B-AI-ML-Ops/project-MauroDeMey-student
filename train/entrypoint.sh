#!/bin/sh

sleep 10
# Create Docker work pool if it doesn't exist
prefect work-pool create --type process train || true

# Start worker in background
prefect worker start --pool train &

# Serve all flows
python main.py

# Keep container running
tail -f /dev/null
