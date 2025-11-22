#!/bin/sh
# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

# Script to start the Python AI service for intent-classifier
# This script is called from run.sh when ENCLAVE_APP is intent-classifier

echo "========================================="
echo "Starting AI service for intent-classifier..."
echo "========================================="

# Change to the ai-main directory
if [ ! -d "ai-main" ]; then
    echo "ERROR: ai-main directory not found!"
    echo "Current directory: $(pwd)"
    echo "Contents of /:"
    ls -la / | head -20
    exit 1
fi

cd ai-main || {
    echo "ERROR: Failed to change to ai-main directory"
    exit 1
}

echo "Current directory: $(pwd)"
echo "Contents of ai-main:"
ls -la

# Check if model files exist
if [ ! -f "intent_classifier_model.keras" ]; then
    echo "ERROR: intent_classifier_model.keras not found!"
    exit 1
fi

if [ ! -f "model_artifacts.pkl" ]; then
    echo "ERROR: model_artifacts.pkl not found!"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found!"
    exit 1
fi

echo "All required files found. Starting Python service..."

# Set up Python path to include user site-packages
export PYTHONPATH=/lib/python3.11:/usr/local/lib/python3.11/lib-dynload:/usr/local/lib/python3.11/site-packages:/root/.local/lib/python3.11/site-packages:$PYTHONPATH
export PATH=/root/.local/bin:$PATH

echo "PYTHONPATH set to: $PYTHONPATH"

# Try to install Python dependencies if pip is available
# Note: TensorFlow and other large packages may not install in minimal enclave environment
# Consider pre-bundling dependencies or using a lighter ML framework
if command -v pip3 >/dev/null 2>&1; then
    echo "Installing Python dependencies..."
    echo "This may take several minutes, especially for TensorFlow..."
    if pip3 install --user -r requirements.txt > /tmp/pip_install.log 2>&1; then
        echo "✅ Python dependencies installed successfully"
    else
        echo "❌ WARNING: Failed to install some Python dependencies"
        echo "Last 30 lines of pip install log:"
        tail -30 /tmp/pip_install.log 2>/dev/null || echo "Log file not found"
        echo "The AI service may not work correctly without all dependencies"
        echo "Attempting to continue anyway..."
    fi
else
    echo "WARNING: pip3 not found, skipping dependency installation"
fi

# Check if critical dependencies are available
echo "Checking if critical Python modules are available..."
python3 -c "import fastapi" 2>/dev/null && echo "✅ fastapi available" || echo "❌ fastapi NOT available"
python3 -c "import uvicorn" 2>/dev/null && echo "✅ uvicorn available" || echo "❌ uvicorn NOT available"
python3 -c "import tensorflow" 2>/dev/null && echo "✅ tensorflow available" || echo "❌ tensorflow NOT available"

# Check if we have at least fastapi and uvicorn (minimum required)
if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ ERROR: Required modules (fastapi, uvicorn) are not available!"
    echo "Cannot start AI service without these dependencies."
    echo ""
    echo "Possible solutions:"
    echo "1. Pre-install dependencies in the enclave image"
    echo "2. Use a Python environment with dependencies pre-bundled"
    echo "3. Check /tmp/pip_install.log for installation errors"
    exit 1
fi

# Start the Python AI service in the background
echo "Starting Python AI service on localhost:8000..."
echo "Logs will be written to /tmp/ai_service.log"
python3 app.py > /tmp/ai_service.log 2>&1 &
PYTHON_PID=$!

echo "Python process started with PID: $PYTHON_PID"

# Wait and check if the service is actually running
echo "Waiting for service to start (checking process and logs)..."
for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    # Check if process is still running
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "ERROR: Python process died!"
        echo "Last 50 lines of /tmp/ai_service.log:"
        tail -50 /tmp/ai_service.log 2>/dev/null || echo "Log file not found"
        exit 1
    fi
    # Check log for "Application startup complete" or "Uvicorn running"
    if grep -q "Application startup complete\|Uvicorn running\|started server process\|Uvicorn.*started" /tmp/ai_service.log 2>/dev/null; then
        echo "AI service started successfully (found startup message in logs)"
        echo "Service should be available on localhost:8000"
        echo "Last 10 lines of log:"
        tail -10 /tmp/ai_service.log 2>/dev/null
        exit 0
    fi
    echo "  Attempt $i/10: Waiting for startup..."
done

# Final check - if process is still running, assume it's working
if kill -0 $PYTHON_PID 2>/dev/null; then
    echo "Python process is running (assuming service started)"
    echo "Last 20 lines of /tmp/ai_service.log:"
    tail -20 /tmp/ai_service.log 2>/dev/null || echo "Log file not found"
    exit 0
else
    echo "ERROR: AI service failed to start"
    echo "Python process status:"
    ps aux | grep python3 || echo "No python3 process found"
    echo "Last 50 lines of /tmp/ai_service.log:"
    tail -50 /tmp/ai_service.log 2>/dev/null || echo "Log file not found"
    exit 1
fi

