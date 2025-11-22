#!/bin/bash
# Simple script to test the intent-classifier endpoint locally

echo "=== Testing Intent Classifier Endpoint Locally ==="
echo ""

# Check if Python service is running
if ! curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "❌ Python AI service is not running on port 8000"
    echo ""
    echo "Please start it in a separate terminal:"
    echo "  cd ai-main"
    echo "  python3 app.py"
    echo ""
    exit 1
fi

echo "✓ Python AI service is running"

# Check if Rust server is running
if ! curl -s http://localhost:3000/ > /dev/null 2>&1; then
    echo "❌ Rust server is not running on port 3000"
    echo ""
    echo "Please start it in a separate terminal:"
    echo "  cd src/nautilus-server"
    echo "  RUST_LOG=info cargo run --features=intent-classifier --bin nautilus-server"
    echo ""
    exit 1
fi

echo "✓ Rust server is running"
echo ""

# Test the endpoint
echo "Testing /process_data endpoint..."
echo ""

RESPONSE=$(curl -s -X POST http://localhost:3000/process_data \
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
  }')

# Check if response contains signature (success indicator)
if echo "$RESPONSE" | grep -q "signature"; then
    echo "✅ SUCCESS! Endpoint is working"
    echo ""
    echo "Response summary:"
    echo "$RESPONSE" | jq -r '.response.data | "Primary Category: \(.primary_category) (\(.primary_category_confidence))"' 2>/dev/null || echo "Response received with signature"
    echo ""
    echo "Full response saved to /tmp/test_response.json"
    echo "$RESPONSE" | jq '.' > /tmp/test_response.json 2>/dev/null || echo "$RESPONSE" > /tmp/test_response.json
else
    echo "❌ FAILED! Endpoint returned an error"
    echo ""
    echo "Response:"
    echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
    exit 1
fi

echo ""
echo "=== Testing other endpoints ==="
echo ""

# Test health check
echo "Testing /health_check..."
HEALTH=$(curl -s http://localhost:3000/health_check)
if echo "$HEALTH" | grep -q "pk"; then
    echo "✅ Health check passed"
    echo "$HEALTH" | jq -r '.pk' | head -c 20 | xargs -I {} echo "Public key: {}..."
else
    echo "⚠️  Health check returned: $HEALTH"
fi

echo ""
echo "Testing / (ping)..."
PING=$(curl -s http://localhost:3000/)
if [ "$PING" = "Pong!" ]; then
    echo "✅ Ping successful"
else
    echo "⚠️  Ping returned: $PING"
fi

echo ""
echo "=== All tests complete! ==="

