#!/bin/bash
# Script to run tests for intent-classifier

cd src/nautilus-server

echo "=== Running Intent Classifier Tests ==="
echo ""

echo "1. Running unit tests (no Python service needed)..."
cargo test --features=intent-classifier --lib -- --nocapture

echo ""
echo "2. Running ALL tests including ignored ones (requires Python service on localhost:8000)..."
echo "   Make sure Python service is running: cd ai-main && python3 app.py"
read -p "   Press Enter to continue or Ctrl+C to skip..."
cargo test --features=intent-classifier --lib -- --ignored --nocapture

echo ""
echo "=== Tests Complete ==="

