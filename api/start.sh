#!/usr/bin/env bash
set -e
echo "Starting Textropy AI API..."
cd "$(dirname "$0")/.."
uvicorn api.main:app --reload --port 8000
