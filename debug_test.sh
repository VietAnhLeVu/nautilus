#!/bin/bash
# Debug script to test the endpoint and see what's happening

echo "=== Debugging Intent Classifier Endpoint ==="
echo ""

# Check Python service
echo "1. Checking Python AI service..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "   ✓ Python service is running on port 8000"
    PYTHON_STATUS=$(curl -s http://localhost:8000/)
    echo "   Response: $PYTHON_STATUS"
else
    echo "   ❌ Python service is NOT running on port 8000"
    echo "   Please start it: cd ai-main && python3 app.py"
    exit 1
fi

# Test Python service directly
echo ""
echo "2. Testing Python service directly..."
PYTHON_TEST=$(curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "solver_window_ms": 5000,
    "user_decision_timeout_ms": 30000,
    "time_to_deadline_ms": 60000,
    "max_slippage_bps": 50,
    "max_gas_cost_usd": 10.0,
    "max_hops": 3,
    "surplus_weight": 5,
    "gas_cost_weight": 3,
    "execution_speed_weight": 4,
    "reputation_weight": 2,
    "input_count": 1,
    "output_count": 1,
    "input_value_usd": 1000.0,
    "expected_output_value_usd": 1005.0,
    "benchmark_confidence": 0.95,
    "expected_gas_usd": 5.0,
    "expected_slippage_bps": 10,
    "nlp_confidence": 0.8,
    "tag_count": 2,
    "time_in_force": "immediate",
    "optimization_goal": "balanced",
    "benchmark_source": "coingecko",
    "client_platform": "web",
    "input_asset_types": ["native"],
    "output_asset_types": ["stable"],
    "has_whitelist": false,
    "has_blacklist": false,
    "has_limit_price": false,
    "require_simulation": true,
    "has_nlp_input": false
  }' 2>&1)

if echo "$PYTHON_TEST" | grep -q "primary_category"; then
    echo "   ✓ Python service is working"
    echo "   Response preview: $(echo "$PYTHON_TEST" | head -c 100)..."
else
    echo "   ❌ Python service returned an error:"
    echo "$PYTHON_TEST"
    exit 1
fi

# Check Rust server
echo ""
echo "3. Checking Rust server..."
if curl -s http://localhost:3000/ > /dev/null 2>&1; then
    echo "   ✓ Rust server is running on port 3000"
    RUST_STATUS=$(curl -s http://localhost:3000/)
    echo "   Response: $RUST_STATUS"
else
    echo "   ❌ Rust server is NOT running on port 3000"
    echo "   Please start it: cd src/nautilus-server && RUST_LOG=info cargo run --features=intent-classifier --bin nautilus-server"
    exit 1
fi

# Test Rust endpoint with verbose output
echo ""
echo "4. Testing Rust /process_data endpoint with verbose output..."
echo "   Sending request..."

RESPONSE=$(curl -v -X POST http://localhost:3000/process_data \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "solver_window_ms": 5000,
      "user_decision_timeout_ms": 30000,
      "time_to_deadline_ms": 60000,
      "max_slippage_bps": 50,
      "max_gas_cost_usd": 10.0,
      "max_hops": 3,
      "surplus_weight": 5,
      "gas_cost_weight": 3,
      "execution_speed_weight": 4,
      "reputation_weight": 2,
      "input_count": 1,
      "output_count": 1,
      "input_value_usd": 1000.0,
      "expected_output_value_usd": 1005.0,
      "benchmark_confidence": 0.95,
      "expected_gas_usd": 5.0,
      "expected_slippage_bps": 10,
      "nlp_confidence": 0.8,
      "tag_count": 2,
      "time_in_force": "immediate",
      "optimization_goal": "balanced",
      "benchmark_source": "coingecko",
      "client_platform": "web",
      "input_asset_types": ["native"],
      "output_asset_types": ["stable"],
      "has_whitelist": false,
      "has_blacklist": false,
      "has_limit_price": false,
      "require_simulation": true,
      "has_nlp_input": false
    }
  }' 2>&1)

echo ""
echo "   Full response:"
echo "$RESPONSE" | tail -n +1

if echo "$RESPONSE" | grep -q "signature"; then
    echo ""
    echo "   ✅ SUCCESS! Endpoint is working"
elif echo "$RESPONSE" | grep -q "error"; then
    echo ""
    echo "   ❌ Error in response"
elif [ -z "$RESPONSE" ]; then
    echo ""
    echo "   ❌ No response received (empty response)"
else
    echo ""
    echo "   ⚠️  Unexpected response format"
fi

echo ""
echo "=== Debug Complete ==="

