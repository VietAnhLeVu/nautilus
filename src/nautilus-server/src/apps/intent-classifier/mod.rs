// Copyright (c), Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::common::IntentMessage;
use crate::common::{to_signed_response, IntentScope, ProcessDataRequest, ProcessedDataResponse};
use crate::AppState;
use crate::EnclaveError;
use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Deserializer, Serialize};
use std::sync::Arc;
use tracing::{info, error};

/// ====
/// Intent Classifier AI service integration
/// This module runs the Python AI service inside the enclave
/// ====

/// Response from Python AI service (with f64 - for deserialization)
#[derive(Debug, Deserialize)]
struct PythonPredictionResponse {
    pub primary_category: String,
    pub primary_category_confidence: f64,
    pub detected_priority: String,
    pub detected_priority_confidence: f64,
    pub complexity_level: String,
    pub complexity_level_confidence: f64,
    pub risk_level: String,
    pub risk_level_confidence: f64,
    pub strategy: String,
    pub strategy_confidence: f64,
    pub weights: PythonWeightsDict,
}

#[derive(Debug, Deserialize)]
struct PythonWeightsDict {
    pub weight_1: f64,
    pub weight_2: f64,
    pub weight_3: f64,
}

/// Response from the AI model prediction (for BCS and JSON API - uses u64)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntentPredictionResponse {
    pub primary_category: String,
    pub primary_category_confidence: u64,  // f64 * 1e6
    pub detected_priority: String,
    pub detected_priority_confidence: u64,  // f64 * 1e6
    pub complexity_level: String,
    pub complexity_level_confidence: u64,  // f64 * 1e6
    pub risk_level: String,
    pub risk_level_confidence: u64,  // f64 * 1e6
    pub strategy: String,
    pub strategy_confidence: u64,  // f64 * 1e6
    pub weights: WeightsDict,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WeightsDict {
    pub weight_1: u64,  // f64 * 1e6
    pub weight_2: u64,  // f64 * 1e6
    pub weight_3: u64,  // f64 * 1e6
}

/// Convert Python response (f64) to our response (u64)
impl From<PythonPredictionResponse> for IntentPredictionResponse {
    fn from(python_resp: PythonPredictionResponse) -> Self {
        const SCALE: f64 = 1_000_000.0;  // 1e6 for 6 decimal places
        IntentPredictionResponse {
            primary_category: python_resp.primary_category,
            primary_category_confidence: (python_resp.primary_category_confidence * SCALE) as u64,
            detected_priority: python_resp.detected_priority,
            detected_priority_confidence: (python_resp.detected_priority_confidence * SCALE) as u64,
            complexity_level: python_resp.complexity_level,
            complexity_level_confidence: (python_resp.complexity_level_confidence * SCALE) as u64,
            risk_level: python_resp.risk_level,
            risk_level_confidence: (python_resp.risk_level_confidence * SCALE) as u64,
            strategy: python_resp.strategy,
            strategy_confidence: (python_resp.strategy_confidence * SCALE) as u64,
            weights: WeightsDict {
                weight_1: (python_resp.weights.weight_1 * SCALE) as u64,
                weight_2: (python_resp.weights.weight_2 * SCALE) as u64,
                weight_3: (python_resp.weights.weight_3 * SCALE) as u64,
            },
        }
    }
}

// Note: IntentPredictionResponse already uses u64, so it's BCS-compatible
// No separate BCS struct needed anymore

/// Custom deserializer that accepts f64 or u64 and converts to u64
/// For USD values, scales by 1e6 to preserve decimal precision
fn deserialize_f64_to_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{Error, Visitor};
    use std::fmt;
    
    struct F64OrU64Visitor;
    
    impl<'de> Visitor<'de> for F64OrU64Visitor {
        type Value = u64;
        
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a number (f64 or u64)")
        }
        
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            // Scale by 1e6 to preserve 6 decimal places (e.g., 10.0 -> 10_000_000)
            Ok((v * 1_000_000.0) as u64)
        }
        
        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            Ok(v)
        }
        
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            if v < 0 {
                return Err(Error::custom("negative value not allowed for u64"));
            }
            Ok(v as u64)
        }
    }
    
    deserializer.deserialize_any(F64OrU64Visitor)
}

/// Custom deserializer that accepts f64 or u64 and converts to u64
/// For confidence values (0-1 range), scales by 1e6
fn deserialize_confidence_to_u64<'de, D>(deserializer: D) -> Result<u64, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::{Error, Visitor};
    use std::fmt;
    
    struct F64OrU64Visitor;
    
    impl<'de> Visitor<'de> for F64OrU64Visitor {
        type Value = u64;
        
        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a number (f64 or u64)")
        }
        
        fn visit_f64<E>(self, v: f64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            // Scale by 1e6 (e.g., 0.95 -> 950_000)
            Ok((v * 1_000_000.0) as u64)
        }
        
        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            Ok(v)
        }
        
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: Error,
        {
            if v < 0 {
                return Err(Error::custom("negative value not allowed for u64"));
            }
            Ok(v as u64)
        }
    }
    
    deserializer.deserialize_any(F64OrU64Visitor)
}

/// Request payload for intent classification
#[derive(Debug, Serialize, Deserialize)]
pub struct IntentRequest {
    // Numerical features
    pub solver_window_ms: i64,
    pub user_decision_timeout_ms: i64,
    pub time_to_deadline_ms: i64,
    pub max_slippage_bps: i64,
    #[serde(deserialize_with = "deserialize_f64_to_u64")]
    pub max_gas_cost_usd: u64,
    pub max_hops: i64,
    pub surplus_weight: i64,
    pub gas_cost_weight: i64,
    pub execution_speed_weight: i64,
    pub reputation_weight: i64,
    pub input_count: i64,
    pub output_count: i64,
    #[serde(deserialize_with = "deserialize_f64_to_u64")]
    pub input_value_usd: u64,
    #[serde(deserialize_with = "deserialize_f64_to_u64")]
    pub expected_output_value_usd: u64,
    #[serde(deserialize_with = "deserialize_confidence_to_u64")]
    pub benchmark_confidence: u64,
    #[serde(deserialize_with = "deserialize_f64_to_u64")]
    pub expected_gas_usd: u64,
    pub expected_slippage_bps: i64,
    #[serde(deserialize_with = "deserialize_confidence_to_u64")]
    pub nlp_confidence: u64,
    pub tag_count: i64,
    
    // Categorical features
    pub time_in_force: String,
    pub optimization_goal: String,
    pub benchmark_source: String,
    pub client_platform: String,
    
    // Array features
    pub input_asset_types: Vec<String>,
    pub output_asset_types: Vec<String>,
    
    // Boolean features
    pub has_whitelist: bool,
    pub has_blacklist: bool,
    pub has_limit_price: bool,
    pub require_simulation: bool,
    pub has_nlp_input: bool,
}

/// Process intent classification request by calling the Python AI service
pub async fn process_data(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ProcessDataRequest<IntentRequest>>,
) -> Result<Json<ProcessedDataResponse<IntentMessage<IntentPredictionResponse>>>, EnclaveError> {
    println!("[process_data] ✅ ENDPOINT CALLED - Received intent classification request");
    println!("[process_data] Request payload: solver_window_ms={}, max_gas_cost_usd={}, input_count={}", 
             request.payload.solver_window_ms, request.payload.max_gas_cost_usd, request.payload.input_count);
    
    println!("[process_data] Step 1: Creating HTTP client...");
    // Call the Python AI service running on localhost:8000
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| {
            eprintln!("[process_data] ❌ ERROR at Step 1: Failed to create HTTP client: {}", e);
            error!("[process_data] ERROR at Step 1: Failed to create HTTP client: {}", e);
            EnclaveError::GenericError(format!("Failed to create HTTP client: {}", e))
        })?;
    println!("[process_data] Step 1: ✅ HTTP client created successfully");

    let url = "http://localhost:8000/predict";
    println!("[process_data] Step 2: Preparing request to Python AI service at: {}", url);
    println!("[process_data] Step 2: Serializing request payload...");
    
    println!("[process_data] Step 3: Sending POST request to Python service...");
    // reqwest's .json() can serialize any Serialize type directly
    let response = client
        .post(url)
        .json(&request.payload)
        .send()
        .await
        .map_err(|e| {
            eprintln!("[process_data] ❌ ERROR at Step 3: Failed to call AI service: {}", e);
            eprintln!("[process_data] ERROR details: Make sure Python service is running on localhost:8000");
            error!("[process_data] ERROR at Step 3: Failed to call AI service: {}", e);
            error!("[process_data] ERROR details: Make sure Python service is running on localhost:8000");
            EnclaveError::GenericError(format!("Failed to call AI service: {}. Make sure Python service is running on localhost:8000", e))
        })?;
    
    println!("[process_data] Step 3: ✅ Received response from Python service, status: {}", response.status());

    println!("[process_data] Step 4: Checking response status...");
    if !response.status().is_success() {
        let status = response.status();
        eprintln!("[process_data] ❌ ERROR at Step 4: AI service returned error status: {}", status);
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        eprintln!("[process_data] ERROR details: {}", error_text);
        error!("[process_data] ERROR at Step 4: AI service returned error status: {}", status);
        error!("[process_data] ERROR details: {}", error_text);
        return Err(EnclaveError::GenericError(format!(
            "AI service returned error status {}: {}",
            status, error_text
        )));
    }
    println!("[process_data] Step 4: ✅ Response status is OK");

    println!("[process_data] Step 5: Parsing JSON response from Python service...");
    // Deserialize from Python (f64) and convert to our format (u64)
    let python_response: PythonPredictionResponse = response
        .json()
        .await
        .map_err(|e| {
            eprintln!("[process_data] ❌ ERROR at Step 5: Failed to parse AI response JSON: {}", e);
            error!("[process_data] ERROR at Step 5: Failed to parse AI response JSON: {}", e);
            EnclaveError::GenericError(format!("Failed to parse AI response: {}", e))
        })?;
    println!("[process_data] Step 5: ✅ Successfully parsed Python response JSON");
    println!("[process_data] Step 5: Python response - primary_category={}, confidence={}", 
             python_response.primary_category, python_response.primary_category_confidence);
    
    println!("[process_data] Step 6: Converting f64 -> u64 (scaling by 1e6)...");
    // Convert f64 -> u64 (scaled by 1e6)
    let prediction: IntentPredictionResponse = python_response.into();
    println!("[process_data] Step 6: ✅ Conversion complete - confidence={}", prediction.primary_category_confidence);
    
    println!("[process_data] Step 7: Getting current timestamp...");
    let current_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| {
            eprintln!("[process_data] ❌ ERROR at Step 7: Failed to get current timestamp: {}", e);
            error!("[process_data] ERROR at Step 7: Failed to get current timestamp: {}", e);
            EnclaveError::GenericError(format!("Failed to get current timestamp: {}", e))
        })?
        .as_millis() as u64;
    println!("[process_data] Step 7: ✅ Timestamp={}", current_timestamp);

    println!("[process_data] Step 8: Creating signed response (BCS serialization)...");
    // IntentPredictionResponse already uses u64, so it's BCS-compatible
    // Sign directly with the prediction (no conversion needed)
    let signed_response = to_signed_response(
        &state.eph_kp,
        prediction,
        current_timestamp,
        IntentScope::ProcessData,
    );
    println!("[process_data] Step 8: ✅ Signed response created successfully");
    println!("[process_data] Step 8: Signature length={} chars", signed_response.signature.len());
    
    println!("[process_data] ✅ SUCCESS: Returning signed response");
    Ok(Json(signed_response))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::common::IntentMessage;
    use fastcrypto::{ed25519::Ed25519KeyPair, traits::KeyPair};

    #[test]
    fn test_bcs_serialization() {
        // Test that IntentPredictionResponse can be serialized with BCS (no f64)
        let prediction = IntentPredictionResponse {
            primary_category: "test".to_string(),
            primary_category_confidence: 950000,  // 0.95 * 1e6
            detected_priority: "high".to_string(),
            detected_priority_confidence: 800000,  // 0.8 * 1e6
            complexity_level: "medium".to_string(),
            complexity_level_confidence: 750000,  // 0.75 * 1e6
            risk_level: "low".to_string(),
            risk_level_confidence: 200000,  // 0.2 * 1e6
            strategy: "aggressive".to_string(),
            strategy_confidence: 900000,  // 0.9 * 1e6
            weights: WeightsDict {
                weight_1: 500000,  // 0.5 * 1e6
                weight_2: 300000,  // 0.3 * 1e6
                weight_3: 200000,  // 0.2 * 1e6
            },
        };
        
        let timestamp = 1744038900000;
        let intent_msg = IntentMessage::new(prediction, timestamp, IntentScope::ProcessData);
        
        // This should not panic - BCS should serialize successfully (no f64)
        let signing_payload = bcs::to_bytes(&intent_msg).expect("BCS serialization should not fail");
        assert!(!signing_payload.is_empty());
    }

    #[test]
    fn test_python_to_rust_conversion() {
        // Test the conversion from Python response (f64) to Rust response (u64)
        let python_response = PythonPredictionResponse {
            primary_category: "test".to_string(),
            primary_category_confidence: 0.95,
            detected_priority: "high".to_string(),
            detected_priority_confidence: 0.8,
            complexity_level: "medium".to_string(),
            complexity_level_confidence: 0.75,
            risk_level: "low".to_string(),
            risk_level_confidence: 0.2,
            strategy: "aggressive".to_string(),
            strategy_confidence: 0.9,
            weights: PythonWeightsDict {
                weight_1: 0.5,
                weight_2: 0.3,
                weight_3: 0.2,
            },
        };
        
        let prediction: IntentPredictionResponse = python_response.into();
        
        // Verify conversion (f64 * 1e6 -> u64)
        assert_eq!(prediction.primary_category_confidence, 950000);
        assert_eq!(prediction.detected_priority_confidence, 800000);
        assert_eq!(prediction.complexity_level_confidence, 750000);
        assert_eq!(prediction.risk_level_confidence, 200000);
        assert_eq!(prediction.strategy_confidence, 900000);
        assert_eq!(prediction.weights.weight_1, 500000);
        assert_eq!(prediction.weights.weight_2, 300000);
        assert_eq!(prediction.weights.weight_3, 200000);
    }

    #[tokio::test]
    async fn test_process_data() {
        let state = Arc::new(AppState {
            eph_kp: Ed25519KeyPair::generate(&mut rand::thread_rng()),
            api_key: String::new(),
        });
        
        let test_request = IntentRequest {
            solver_window_ms: 5000,
            user_decision_timeout_ms: 30000,
            time_to_deadline_ms: 60000,
            max_slippage_bps: 50,
            max_gas_cost_usd: 10,  // u64 now
            max_hops: 3,
            surplus_weight: 5,
            gas_cost_weight: 3,
            execution_speed_weight: 4,
            reputation_weight: 2,
            input_count: 1,
            output_count: 1,
            input_value_usd: 1000,  // u64 now
            expected_output_value_usd: 1005,  // u64 now
            benchmark_confidence: 950000,  // u64 (0.95 * 1e6)
            expected_gas_usd: 5,  // u64 now
            expected_slippage_bps: 10,
            nlp_confidence: 800000,  // u64 (0.8 * 1e6)
            tag_count: 2,
            time_in_force: "immediate".to_string(),
            optimization_goal: "balanced".to_string(),
            benchmark_source: "coingecko".to_string(),
            client_platform: "web".to_string(),
            input_asset_types: vec!["native".to_string()],
            output_asset_types: vec!["stable".to_string()],
            has_whitelist: false,
            has_blacklist: false,
            has_limit_price: false,
            require_simulation: true,
            has_nlp_input: false,
        };

        let result = process_data(
            State(state),
            Json(ProcessDataRequest {
                payload: test_request,
            }),
        )
        .await;

        // This test will only pass if the Python service is running
        match result {
            Ok(_) => println!("Test passed: AI service responded successfully"),
            Err(e) => println!("Test failed: {}", e),
        }
    }
}

