# Intent Classifier Enclave App

This app integrates the AI intent classifier model from `ai-main` into a Nautilus enclave.

## Overview

The intent-classifier app runs a Python FastAPI service inside the enclave that provides AI-powered intent classification. The Rust server acts as a proxy, handling Nautilus-specific operations (attestation, signing) while delegating AI predictions to the Python service.

## Architecture

1. **Rust Server** (`mod.rs`): Handles enclave operations, attestation, and request signing
2. **Python AI Service** (`ai-main/app.py`): Runs the TensorFlow model for intent classification
3. **Communication**: Rust server calls Python service via HTTP on `localhost:8000`

## Building

To build the enclave with intent-classifier:

```bash
make ENCLAVE_APP=intent-classifier
```

This will:
- Build the Rust server with the `intent-classifier` feature
- Copy AI model files (`intent_classifier_model.keras`, `model_artifacts.pkl`) into the enclave
- Copy the Python application (`app.py`) into the enclave
- Include the startup script to launch the Python service

## Python Dependencies

**Important**: The minimal enclave environment may not have all Python packages pre-installed. TensorFlow and other dependencies from `requirements.txt` may need to be:

1. **Pre-bundled**: Include pre-compiled Python packages in the enclave image
2. **Installed at runtime**: The `start_ai_service.sh` script attempts to install dependencies, but this may fail for large packages like TensorFlow
3. **Alternative**: Consider using a lighter ML framework or pre-compiling the model to a more portable format

## Running

When the enclave starts:
1. The `run.sh` script detects the AI service files
2. `start_ai_service.sh` launches the Python service on `localhost:8000`
3. The Rust server starts on `localhost:3000` and proxies AI requests

## API

The Rust server exposes the standard Nautilus endpoints:
- `POST /process_data`: Accepts `IntentRequest` and returns signed `IntentPredictionResponse`
- `GET /get_attestation`: Returns enclave attestation
- `GET /health_check`: Health check endpoint

### Request Format

```json
{
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
}
```

### Response Format

The response is a signed `ProcessedDataResponse` containing:
- `response`: The AI prediction results
- `signature`: Ed25519 signature of the response

## Troubleshooting

1. **Python service not starting**: Check `/tmp/ai_service.log` in the enclave
2. **Missing dependencies**: Verify Python packages are available or pre-installed
3. **Model files not found**: Ensure `intent_classifier_model.keras` and `model_artifacts.pkl` are in `ai-main/` directory

