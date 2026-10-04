#!/bin/bash
# Starts the bobgpt API on lab. Run by the bobgpt systemd unit, which loads
# /etc/bobgpt/host.env (BOBGPT_VENV, BOBGPT_HOST, BOBGPT_PORT, ...).
set -euo pipefail

# api.py loads checkpoints/... relative to the working directory.
cd "$(dirname "$0")"

# exec: uvicorn becomes the unit's main process, so systemd's signals reach it.
# One worker: each worker would load its own copy of the model.
exec "$BOBGPT_VENV/bin/uvicorn" api:app \
     --host "$BOBGPT_HOST" --port "$BOBGPT_PORT" --workers 1
