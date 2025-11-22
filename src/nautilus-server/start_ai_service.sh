#!/bin/sh
# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

# Script to start the Python AI service for intent-classifier
# This script is called from run.sh when ENCLAVE_APP is intent-classifier

set -e

echo "Starting AI service for intent-classifier..."

# Change to the ai-main directory
cd /ai-main || {
    echo "Warning: /ai-main directory not found, AI service will not start"
    exit 0
}

# Check if model files exist
if [ ! -f "intent_classifier_model.keras" ] || [ ! -f "model_artifacts.pkl" ]; then
    echo "Warning: Model files not found, AI service will not start"
    exit 0
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "Warning: app.py not found, AI service will not start"
    exit 0
fi

# Try to install Python dependencies if pip is available
# Note: TensorFlow and other large packages may not install in minimal enclave environment
# Consider pre-bundling dependencies or using a lighter ML framework
if command -v pip3 >/dev/null 2>&1; then
    echo "Installing Python dependencies..."
    pip3 install --user -r requirements.txt 2>&1 || {
        echo "Warning: Failed to install some Python dependencies"
        echo "The AI service may not work correctly without all dependencies"
    }
fi

# Start the Python AI service in the background
echo "Starting Python AI service on localhost:8000..."
python3 app.py > /tmp/ai_service.log 2>&1 &

# Wait a moment for the service to start
sleep 2

# Check if the service is running
if ! pgrep -f "python3 app.py" > /dev/null; then
    echo "Warning: Python AI service failed to start. Check /tmp/ai_service.log for details"
    exit 0
fi

echo "AI service started successfully"
exit 0

