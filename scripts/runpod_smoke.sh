#!/usr/bin/env bash
set -e

echo '=== GPU Info ==='
nvidia-smi || true

echo '=== Python ==='
python --version

echo '=== Running smoke test ==='
python src/train.py
