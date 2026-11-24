#!/usr/bin/env bash
cd "$(dirname "$0")/.."
.venv/Scripts/python -m src.run.transport >> results/transport_run.log 2>&1
echo TRANSPORT_CHAIN_END >> results/transport_run.log

