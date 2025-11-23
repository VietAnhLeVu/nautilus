from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from enum import Enum
import tensorflow as tf
import pickle
import numpy as np
import json

app = FastAPI(title="Intent Classifier API", version="1.0.0")

# Global variables for model and artifacts
model = None
label_encoders = None
gt_encoders = None
strategy_encoder = None
scaler = None

# Enums
class TimeInForce(str, Enum):
    immediate = "immediate"
    good_til_cancel = "good_til_cancel"
    fill_or_kill = "fill_or_kill"

class OptimizationGoal(str, Enum):
    maximize_output = "maximize_output"
    minimize_gas = "minimize_gas"
    fastest_execution = "fastest_execution"
    balanced = "balanced"

class AssetType(str, Enum):
    native = "native"
    stable = "stable"
    volatile = "volatile"

class Intent(BaseModel):
    # Numerical features
    solver_window_ms: int
    user_decision_timeout_ms: int
    time_to_deadline_ms: int
    max_slippage_bps: int
    max_gas_cost_usd: float
    max_hops: int
    surplus_weight: int
    gas_cost_weight: int
    execution_speed_weight: int
    reputation_weight: int
    input_count: int
    output_count: int
    input_value_usd: float
    expected_output_value_usd: float
    benchmark_confidence: float
    expected_gas_usd: float
    expected_slippage_bps: int
    nlp_confidence: float
    tag_count: int
    
    # Categorical features (enums)
    time_in_force: TimeInForce
    optimization_goal: OptimizationGoal
    benchmark_source: str  # coingecko, 1inch, paraswap, internal, etc.
    client_platform: str  # web, mobile, api, bot
    
    # Array features
    input_asset_types: List[AssetType]
    output_asset_types: List[AssetType]
    
    # Boolean features
    has_whitelist: bool
    has_blacklist: bool
    has_limit_price: bool
    require_simulation: bool
    has_nlp_input: bool

class PredictionResponse(BaseModel):
    primary_category: str
    primary_category_confidence: float
    detected_priority: str
    detected_priority_confidence: float
    complexity_level: str
    complexity_level_confidence: float
    risk_level: str
    risk_level_confidence: float
    strategy: str
    strategy_confidence: float
    weights: dict  # 9 weights: surplus_usd, surplus_percentage, gas_cost, protocol_fees, total_hops, protocols_count, execution_time, solver_reputation, solver_success_rate

@app.on_event("startup")
async def load_model_and_artifacts():
    """Load the trained model and preprocessing artifacts on startup"""
    global model, label_encoders, gt_encoders, strategy_encoder, scaler
    
    try:
        # Load the trained model
        model = tf.keras.models.load_model('model/intent_classifier_model.keras')
        print("✓ Model loaded successfully")
        
        # Load preprocessing artifacts
        with open('model/model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
            label_encoders = artifacts['label_encoders']
            gt_encoders = artifacts['gt_encoders']
            strategy_encoder = artifacts['strategy_encoder']
            scaler = artifacts['scaler']
        print("✓ Artifacts loaded successfully")
        
    except Exception as e:
        print(f"✗ Error loading model or artifacts: {e}")
        raise

def prepare_features(intent: Intent) -> np.ndarray:
    """Prepare and scale features from intent input"""
    
    # Parse asset types to get counts (matching train_model.py logic)
    input_asset_count = len(intent.input_asset_types)
    output_asset_count = len(intent.output_asset_types)
    
    # Encode categorical features (matching train_model.py)
    time_in_force_enc = label_encoders['time_in_force'].transform([intent.time_in_force.value])[0]
    optimization_goal_enc = label_encoders['optimization_goal'].transform([intent.optimization_goal.value])[0]
    benchmark_source_enc = label_encoders['benchmark_source'].transform([intent.benchmark_source])[0]
    client_platform_enc = label_encoders['client_platform'].transform([intent.client_platform])[0]
    
    # Construct feature vector matching train_model.py order
    # intent_features = [numerical (21) + encoded categorical (4) + boolean (5)] = 30 features
    features = [
        # Numerical features (21)
        intent.solver_window_ms,
        intent.user_decision_timeout_ms,
        intent.time_to_deadline_ms,
        intent.max_slippage_bps,
        intent.max_gas_cost_usd,
        intent.max_hops,
        intent.surplus_weight,
        intent.gas_cost_weight,
        intent.execution_speed_weight,
        intent.reputation_weight,
        intent.input_count,
        intent.output_count,
        intent.input_value_usd,
        intent.expected_output_value_usd,
        intent.benchmark_confidence,
        intent.expected_gas_usd,
        intent.expected_slippage_bps,
        intent.nlp_confidence,
        intent.tag_count,
        input_asset_count,
        output_asset_count,
        # Encoded categorical features (4)
        time_in_force_enc,
        optimization_goal_enc,
        benchmark_source_enc,
        client_platform_enc,
        # Boolean features as integers (5)
        int(intent.has_whitelist),
        int(intent.has_blacklist),
        int(intent.has_limit_price),
        int(intent.require_simulation),
        int(intent.has_nlp_input)
    ]
    
    # Scale features
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Reshape for CNN1D input (batch_size, sequence_length, features)
    X_input = X_scaled.reshape(1, -1, 1)
    
    return X_input

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Intent Classifier API is running",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(intent: Intent):
    """
    Predict ground truth labels, strategy, and weights for a given intent
    
    Args:
        intent: Intent object containing all required features
        
    Returns:
        Predictions for all tasks with confidence scores
    """
    try:
        # Prepare features
        X_input = prepare_features(intent)
        
        # Make predictions
        predictions = model.predict(X_input, verbose=0)
        
        # Extract predictions for each output (6 outputs total)
        # Model outputs: [primary_category, detected_priority, complexity_level, risk_level, strategy, weights]
        primary_cat_pred = predictions[0][0]
        priority_pred = predictions[1][0]
        complexity_pred = predictions[2][0]
        risk_pred = predictions[3][0]
        strategy_pred = predictions[4][0]
        weights_pred = predictions[5][0]
        
        # Decode predictions
        primary_category = gt_encoders['primary_category'].inverse_transform([np.argmax(primary_cat_pred)])[0]
        primary_category_conf = float(np.max(primary_cat_pred))
        
        detected_priority = gt_encoders['detected_priority'].inverse_transform([np.argmax(priority_pred)])[0]
        detected_priority_conf = float(np.max(priority_pred))
        
        complexity_level = gt_encoders['complexity_level'].inverse_transform([np.argmax(complexity_pred)])[0]
        complexity_level_conf = float(np.max(complexity_pred))
        
        risk_level = gt_encoders['risk_level'].inverse_transform([np.argmax(risk_pred)])[0]
        risk_level_conf = float(np.max(risk_pred))
        
        strategy = strategy_encoder.inverse_transform([np.argmax(strategy_pred)])[0]
        strategy_conf = float(np.max(strategy_pred))
        
        # Weights are already normalized by softmax in model (sum to 1.0)
        # 9 shared weights across all strategies
        weights_dict = {
            "gt_weight_surplus_usd": float(weights_pred[0]),
            "gt_weight_surplus_percentage": float(weights_pred[1]),
            "gt_weight_gas_cost": float(weights_pred[2]),
            "gt_weight_protocol_fees": float(weights_pred[3]),
            "gt_weight_total_hops": float(weights_pred[4]),
            "gt_weight_protocols_count": float(weights_pred[5]),
            "gt_weight_estimated_execution_time": float(weights_pred[6]),
            "gt_weight_solver_reputation_score": float(weights_pred[7]),
            "gt_weight_solver_success_rate": float(weights_pred[8])
        }
        
        return PredictionResponse(
            primary_category=primary_category,
            primary_category_confidence=primary_category_conf,
            detected_priority=detected_priority,
            detected_priority_confidence=detected_priority_conf,
            complexity_level=complexity_level,
            complexity_level_confidence=complexity_level_conf,
            risk_level=risk_level,
            risk_level_confidence=risk_level_conf,
            strategy=strategy,
            strategy_confidence=strategy_conf,
            weights=weights_dict
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "artifacts_loaded": all([
            label_encoders is not None,
            gt_encoders is not None,
            strategy_encoder is not None,
            scaler is not None
        ])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
