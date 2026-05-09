#!/bin/bash
cd "$(dirname "$0")" || exit

if command -v python3 >/dev/null 2>&1; then
    python3 run_app.py
elif command -v python >/dev/null 2>&1; then
    python run_app.py
else
    echo "Python is not installed or not available on PATH."
    exit 1
fi
