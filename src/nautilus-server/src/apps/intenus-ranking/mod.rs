// Copyright (c), Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::common::IntentMessage;
use crate::common::{to_signed_response, IntentScope, ProcessDataRequest, ProcessedDataResponse};
use crate::AppState;
use crate::EnclaveError;
use axum::extract::State;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// ====
/// Intenus Ranking Engine - Processes PreRanking results and produces ranked solutions
/// ====

// ===== Input Types (PreRanking Result) =====

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SolutionSubmission {
    pub solution_id: String,
    pub intent_id: String,
    pub solver_address: String,
    pub submitted_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IntentClassification {
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
    pub weights: ClassificationWeights,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ClassificationWeights {
    pub gt_weight_surplus_usd: f64,
    pub gt_weight_surplus_percentage: f64,
    pub gt_weight_gas_cost: f64,
    pub gt_weight_protocol_fees: f64,
    pub gt_weight_total_hops: f64,
    pub gt_weight_protocols_count: f64,
    pub gt_weight_estimated_execution_time: f64,
    pub gt_weight_solver_reputation_score: f64,
    pub gt_weight_solver_success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FailedSolution {
    pub solution_id: String,
    pub failure_reason: String,
    pub errors: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SolutionFeatures {
    pub surplus_usd: f64,
    pub surplus_percentage: f64,
    pub gas_cost: f64,
    pub protocol_fees: f64,
    pub total_cost: f64,
    pub total_hops: u32,
    pub protocols_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_execution_time: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_reputation_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_success_rate: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FeatureVector {
    pub solution_id: String,
    pub features: SolutionFeatures,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DryRunResult {
    pub solution_id: String,
    pub result: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreRankingStats {
    pub total_submitted: u32,
    pub passed: u32,
    pub failed: u32,
    pub dry_run_executed: u32,
    pub dry_run_successful: u32,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PreRankingResult {
    pub intent_id: String,
    pub intent_classification: IntentClassification,
    pub passed_solution_ids: Vec<String>,
    pub failed_solution_ids: Vec<FailedSolution>,
    pub feature_vectors: Vec<FeatureVector>,
    pub dry_run_results: Vec<DryRunResult>,
    pub stats: PreRankingStats,
    pub processed_at: u64,
}

// ===== Output Types (Ranking Result) =====
// Note: All scores are scaled by 100 (e.g., 85.5% = 8550) to avoid f64 in BCS

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ScoreBreakdown {
    pub surplus_score: u64,      // Scaled by 100
    pub cost_score: u64,          // Scaled by 100
    pub speed_score: u64,         // Scaled by 100
    pub reputation_score: u64,    // Scaled by 100
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SolutionReasoning {
    pub primary_reason: String,
    pub secondary_reasons: Vec<String>,
    pub risk_assessment: String,
    pub confidence_level: u64,    // Scaled by 10000 (0-10000 for 0.0-1.0)
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RankedSolution {
    pub rank: u32,
    pub score: u64,               // Scaled by 100 (0-10000 for 0-100.0)
    pub solution_id: String,
    pub solver_address: String,
    pub transaction_bytes: String,
    pub score_breakdown: ScoreBreakdown,
    pub reasoning: SolutionReasoning,
    pub personalization_applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_fit_score: Option<u64>,  // Scaled by 100
    pub warnings: Vec<String>,
    pub expires_at: u64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RankingMetadata {
    pub total_solutions: u32,
    pub average_score: u64,       // Scaled by 100
    pub strategy: String,
    pub intent_category: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RankingResult {
    pub intent_id: String,
    pub ranked_solutions: Vec<RankedSolution>,
    pub best_solution: Option<RankedSolution>,
    pub metadata: RankingMetadata,
    pub ranked_at: u64,
    pub expires_at: u64,
}

// ===== Ranking Logic =====

fn calculate_score(
    features: &SolutionFeatures,
    weights: &ClassificationWeights,
    intent_classification: &IntentClassification,
) -> (u64, ScoreBreakdown) {
    // Normalize features to 0-1 range (assuming reasonable bounds)
    let surplus_norm = (features.surplus_usd / 1000.0).min(1.0).max(0.0);
    let surplus_pct_norm = (features.surplus_percentage / 10.0).min(1.0).max(0.0);
    
    // Cost features (lower is better, so invert)
    let gas_norm = 1.0 - (features.gas_cost / 100.0).min(1.0).max(0.0);
    let protocol_fees_norm = 1.0 - (features.protocol_fees / 100.0).min(1.0).max(0.0);
    let total_cost_norm = 1.0 - (features.total_cost / 200.0).min(1.0).max(0.0);
    
    // Complexity features (lower is better)
    let hops_norm = 1.0 - (features.total_hops as f64 / 10.0).min(1.0).max(0.0);
    let protocols_norm = 1.0 - (features.protocols_count as f64 / 5.0).min(1.0).max(0.0);
    
    // Time and reputation
    let time_norm = features
        .estimated_execution_time
        .map(|t| 1.0 - (t / 10000.0).min(1.0).max(0.0))
        .unwrap_or(0.5);
    let reputation_norm = features.solver_reputation_score.unwrap_or(0.5);
    let success_rate_norm = features.solver_success_rate.unwrap_or(0.5);

    // Calculate component scores based on detected priority (0-100 range, as f64)
    let surplus_score_f64 = match intent_classification.detected_priority.as_str() {
        "output" => (surplus_norm * 0.6 + surplus_pct_norm * 0.4) * 100.0,
        "balanced" => (surplus_norm * 0.5 + surplus_pct_norm * 0.5) * 100.0,
        _ => (surplus_norm * 0.4 + surplus_pct_norm * 0.6) * 100.0,
    };

    let cost_score_f64 = match intent_classification.detected_priority.as_str() {
        "cost" => (gas_norm * 0.5 + protocol_fees_norm * 0.3 + total_cost_norm * 0.2) * 100.0,
        "balanced" => (gas_norm * 0.4 + protocol_fees_norm * 0.3 + total_cost_norm * 0.3) * 100.0,
        _ => (gas_norm * 0.3 + protocol_fees_norm * 0.3 + total_cost_norm * 0.4) * 100.0,
    };

    let speed_score_f64 = match intent_classification.detected_priority.as_str() {
        "speed" => (time_norm * 0.6 + hops_norm * 0.3 + protocols_norm * 0.1) * 100.0,
        "balanced" => (time_norm * 0.4 + hops_norm * 0.3 + protocols_norm * 0.3) * 100.0,
        _ => (time_norm * 0.3 + hops_norm * 0.4 + protocols_norm * 0.3) * 100.0,
    };

    let reputation_score_f64 = (reputation_norm * 0.6 + success_rate_norm * 0.4) * 100.0;

    // Apply ML-derived weights
    let weighted_score_f64 = match intent_classification.strategy.as_str() {
        "Surplus-First"=> weights.gt_weight_surplus_usd*features.surplus_usd + weights.gt_weight_gas_cost*(1.0/features.gas_cost) + weights.gt_weight_solver_reputation_score*features.solver_reputation_score.unwrap_or(0.5),
        "Cost-Minimization"=> weights.gt_weight_gas_cost*(features.gas_cost) + weights.gt_weight_surplus_percentage*features.surplus_percentage + weights.gt_weight_solver_success_rate*features.solver_success_rate.unwrap_or(0.5),
        "Surplus-Maximization"=> features.surplus_usd*weights.gt_weight_surplus_usd + features.surplus_percentage*weights.gt_weight_surplus_percentage + weights.gt_weight_total_hops*1.0/features.total_hops as f64 + weights.gt_weight_solver_reputation_score*features.solver_reputation_score.unwrap_or(0.5),
        "Speed-Priority"=> weights.gt_weight_estimated_execution_time*1.0/features.estimated_execution_time as f64 +weights.gt_weight_solver_success_rate + weights.gt_weight_surplus_usd*features.surplus_usd + weights.gt_weight_gas_cost*1.0/features.gas_cost,
        "Reliability-Focus"=> weights.gt_weight_solver_success_rate*features.solver_success_rate.unwrap_or(0.5)+weights.gt_weight_total_hops*1.0/features.total_hops as f64 +weights.gt_weight_surplus_usd*features.surplus_usd + weights.gt_weight_solver_reputation_score*features.solver_reputation_score.unwrap_or(0.5),
    };

    // Convert to u64 scaled by 100 (e.g., 85.5 -> 8550)
    let surplus_score = (surplus_score_f64 * 100.0).round() as u64;
    let cost_score = (cost_score_f64 * 100.0).round() as u64;
    let speed_score = (speed_score_f64 * 100.0).round() as u64;
    let reputation_score = (reputation_score_f64 * 100.0).round() as u64;
    let weighted_score = (weighted_score_f64 * 100.0).round().min(10000.0).max(0.0) as u64;

    let breakdown = ScoreBreakdown {
        surplus_score,
        cost_score,
        speed_score,
        reputation_score,
    };

    (weighted_score, breakdown)
}

fn generate_reasoning(
    features: &SolutionFeatures,
    score_breakdown: &ScoreBreakdown,
    intent_classification: &IntentClassification,
) -> SolutionReasoning {
    let mut secondary_reasons = Vec::new();

    // Note: scores are scaled by 100, so 8000 = 80.00
    let primary_reason = match intent_classification.detected_priority.as_str() {
        "output" => {
            if score_breakdown.surplus_score > 8000 {
                "Excellent surplus generation aligning with output-focused priority"
            } else {
                "Moderate surplus generation for output priority"
            }
        }
        "cost" => {
            if score_breakdown.cost_score > 8000 {
                "Highly cost-efficient solution minimizing fees"
            } else {
                "Acceptable cost efficiency"
            }
        }
        "speed" => {
            if score_breakdown.speed_score > 8000 {
                "Fast execution with minimal complexity"
            } else {
                "Moderate execution speed"
            }
        }
        _ => "Balanced solution across all metrics",
    }
    .to_string();

    if features.total_hops <= 2 {
        secondary_reasons.push("Simple routing path".to_string());
    }
    if features.protocols_count == 1 {
        secondary_reasons.push("Single protocol usage reduces risk".to_string());
    }
    if features.surplus_percentage > 1.0 {
        secondary_reasons.push("Above 1% surplus generation".to_string());
    }
    if let Some(success_rate) = features.solver_success_rate {
        if success_rate > 0.95 {
            secondary_reasons.push("Highly reliable solver".to_string());
        }
    }

    let risk_assessment = if features.total_hops > 5 || features.protocols_count > 3 {
        "high"
    } else if features.total_hops > 3 || features.protocols_count > 2 {
        "medium"
    } else {
        "low"
    }
    .to_string();

    // Convert confidence to u64 scaled by 10000 (0.0-1.0 -> 0-10000)
    let confidence_f64 = (intent_classification.strategy_confidence * 0.5
        + intent_classification.detected_priority_confidence * 0.5)
        .min(1.0)
        .max(0.0);
    let confidence_level = (confidence_f64 * 10000.0).round() as u64;

    SolutionReasoning {
        primary_reason,
        secondary_reasons,
        risk_assessment,
        confidence_level,
    }
}

fn rank_solutions(pre_ranking: &PreRankingResult) -> Result<RankingResult, EnclaveError> {
    let current_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| EnclaveError::GenericError(format!("Failed to get timestamp: {}", e)))?
        .as_millis() as u64;

    let expiration_time = current_timestamp + 300_000; // 5 minutes

    let mut ranked_solutions: Vec<RankedSolution> = Vec::new();
    let mut total_score: u64 = 0;

    // Score each solution
    for feature_vec in &pre_ranking.feature_vectors {
        let (score, breakdown) = calculate_score(
            &feature_vec.features,
            &pre_ranking.intent_classification.weights,
            &pre_ranking.intent_classification,
        );

        let reasoning = generate_reasoning(
            &feature_vec.features,
            &breakdown,
            &pre_ranking.intent_classification,
        );

        let mut warnings = Vec::new();
        if feature_vec.features.total_hops > 5 {
            warnings.push("Complex routing path may increase failure risk".to_string());
        }
        if feature_vec.features.gas_cost > 50.0 {
            warnings.push("High gas cost detected".to_string());
        }
        if pre_ranking.intent_classification.risk_level == "high" {
            warnings.push("Intent classified as high risk".to_string());
        }

        ranked_solutions.push(RankedSolution {
            rank: 0, // Will be assigned after sorting
            score,
            solution_id: feature_vec.solution_id.clone(),
            solver_address: "unknown".to_string(), // Would need to join with submission data
            transaction_bytes: "0x00".to_string(),  // Would come from dry_run_results
            score_breakdown: breakdown,
            reasoning,
            personalization_applied: false,
            user_fit_score: None,
            warnings,
            expires_at: expiration_time,
        });

        total_score = total_score.saturating_add(score);
    }

    // Sort by score descending and assign ranks
    ranked_solutions.sort_by(|a, b| b.score.cmp(&a.score));
    for (idx, solution) in ranked_solutions.iter_mut().enumerate() {
        solution.rank = (idx + 1) as u32;
    }

    let average_score = if !ranked_solutions.is_empty() {
        total_score / ranked_solutions.len() as u64
    } else {
        0
    };

    let best_solution = ranked_solutions.first().cloned();

    let metadata = RankingMetadata {
        total_solutions: ranked_solutions.len() as u32,
        average_score,
        strategy: pre_ranking.intent_classification.strategy.clone(),
        intent_category: pre_ranking.intent_classification.primary_category.clone(),
    };

    Ok(RankingResult {
        intent_id: pre_ranking.intent_id.clone(),
        ranked_solutions,
        best_solution,
        metadata,
        ranked_at: current_timestamp,
        expires_at: expiration_time,
    })
}

// ===== Endpoint =====

pub async fn process_data(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ProcessDataRequest<PreRankingResult>>,
) -> Result<Json<ProcessedDataResponse<IntentMessage<RankingResult>>>, EnclaveError> {
    let pre_ranking = request.payload;

    // Validate input
    if pre_ranking.passed_solution_ids.is_empty() {
        return Err(EnclaveError::GenericError(
            "No solutions passed validation".to_string(),
        ));
    }

    if pre_ranking.feature_vectors.is_empty() {
        return Err(EnclaveError::GenericError(
            "No feature vectors provided".to_string(),
        ));
    }

    // Perform ranking
    let ranking_result = rank_solutions(&pre_ranking)?;

    let current_timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| EnclaveError::GenericError(format!("Failed to get timestamp: {}", e)))?
        .as_millis() as u64;

    Ok(Json(to_signed_response(
        &state.eph_kp,
        ranking_result,
        current_timestamp,
        IntentScope::ProcessData,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_calculation() {
        let features = SolutionFeatures {
            surplus_usd: 100.0,
            surplus_percentage: 2.0,
            gas_cost: 10.0,
            protocol_fees: 5.0,
            total_cost: 15.0,
            total_hops: 2,
            protocols_count: 1,
            estimated_execution_time: Some(1000.0),
            solver_reputation_score: Some(0.95),
            solver_success_rate: Some(0.98),
        };

        let weights = ClassificationWeights {
            gt_weight_surplus_usd: 0.3,
            gt_weight_surplus_percentage: 0.0,
            gt_weight_gas_cost: 0.25,
            gt_weight_protocol_fees: 0.0,
            gt_weight_total_hops: 0.0,
            gt_weight_protocols_count: 0.0,
            gt_weight_estimated_execution_time: 0.25,
            gt_weight_solver_reputation_score: 0.2,
            gt_weight_solver_success_rate: 0.0,
        };

        let classification = IntentClassification {
            primary_category: "swap".to_string(),
            primary_category_confidence: 0.95,
            detected_priority: "balanced".to_string(),
            detected_priority_confidence: 0.9,
            complexity_level: "simple".to_string(),
            complexity_level_confidence: 0.92,
            risk_level: "low".to_string(),
            risk_level_confidence: 0.88,
            strategy: "direct_swap".to_string(),
            strategy_confidence: 0.93,
            weights: weights.clone(),
        };

        let (score, breakdown) = calculate_score(&features, &weights, &classification);

        // Scores are scaled by 100, so max is 10000 (representing 100.00)
        assert!(score <= 10000);
        assert!(breakdown.surplus_score <= 10000);
        assert!(breakdown.cost_score <= 10000);
        assert!(breakdown.speed_score <= 10000);
        assert!(breakdown.reputation_score <= 10000);
    }
}

