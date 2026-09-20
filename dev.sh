#!/usr/bin/env bash
set -e

case "${1:-start}" in
  install)
    echo "Installing dependencies..."
    pip install -r requirements.txt
    ;;
  start)
    echo "Starting Textropy AI Dashboard..."
    streamlit run app.py
    ;;
  api)
    echo "Starting Textropy AI API..."
    uvicorn api.main:app --reload --port 8000
    ;;
  all)
    echo "Starting API + Dashboard..."
    uvicorn api.main:app --reload --port 8000 &
    sleep 2
    streamlit run app.py
    ;;
  docker-build)
    echo "Building Docker images..."
    docker compose build
    ;;
  docker-up)
    echo "Starting Docker containers..."
    docker compose up -d
    ;;
  docker-down)
    echo "Stopping Docker containers..."
    docker compose down
    ;;
  *)
    echo "Usage: ./dev.sh {install|start|api|all|docker-build|docker-up|docker-down}"
    exit 1
    ;;
esac
