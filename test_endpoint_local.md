# Testing Intent Classifier Endpoint Locally

## Quick Test (3 Steps)

### Step 1: Start Python AI Service

Open Terminal 1:
```bash
cd ai-main
python3 app.py
```

You should see:
```
✓ Model loaded successfully
✓ Artifacts loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Start Rust Server

Open Terminal 2:
```bash
cd src/nautilus-server
RUST_LOG=info cargo run --features=intent-classifier --bin nautilus-server
```

You should see:
```
listening on 0.0.0.0:3000
```

### Step 3: Test the Endpoint

Open Terminal 3 (or use curl/Postman):
```bash
curl -X POST http://localhost:3000/process_data \
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
  }'
```

## Expected Response

You should get a JSON response with:
- `response`: Contains the AI prediction with all fields
- `signature`: Ed25519 signature of the response

Example:
```json
{
  "response": {
    "intent": 0,
    "timestamp_ms": 1234567890,
    "data": {
      "primary_category": "...",
      "primary_category_confidence": 0.95,
      "detected_priority": "...",
      ...
    }
  },
  "signature": "abc123..."
}
```

## Other Endpoints to Test

### Health Check
```bash
curl http://localhost:3000/health_check
```

### Ping
```bash
curl http://localhost:3000/
```

### Get Attestation (Won't work locally)
```bash
curl http://localhost:3000/get_attestation
# This will fail locally - it requires NSM driver (only in enclave)
```

## Troubleshooting

1. **Python service not starting**: 
   - Check if model files exist: `ls ai-main/intent_classifier_model.keras`
   - Install dependencies: `pip install -r ai-main/requirements.txt`

2. **Rust server can't connect to Python**:
   - Verify Python service is running: `curl http://localhost:8000/`
   - Check Rust logs for connection errors

3. **Port already in use**:
   - Kill existing processes: `lsof -ti:8000 | xargs kill` or `lsof -ti:3000 | xargs kill`

