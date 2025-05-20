#!/bin/bash
sleep 20
prefect work-pool create --type process train --overwrite
prefect worker start -p train &

python /main.py
