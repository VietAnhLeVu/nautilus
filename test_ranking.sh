#!/bin/bash
# Copyright (c), Mysten Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

# Test script for Intenus Ranking endpoint

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <enclave_url>"
    echo "Example: $0 http://54.123.45.67:3000"
    exit 1
fi

ENCLAVE_URL=$1

echo "Testing Intenus Ranking Endpoint at $ENCLAVE_URL"
echo "================================================"

# Sample PreRankingResult payload
PAYLOAD='{
  "payload": {
    "intent_id": "intent_test_123",
    "intent_classification": {
      "primary_category": "swap",
      "primary_category_confidence": 0.95,
      "detected_priority": "cost",
      "detected_priority_confidence": 0.9,
      "complexity_level": "simple",
      "complexity_level_confidence": 0.92,
      "risk_level": "low",
      "risk_level_confidence": 0.88,
      "strategy": "Speed-Priority",
      "strategy_confidence": 0.93,
      "weights": {
        "gt_weight_surplus_usd": 0.3,
        "gt_weight_surplus_percentage": 0.0,
        "gt_weight_gas_cost": 0.25,
        "gt_weight_protocol_fees": 0.0,
        "gt_weight_total_hops": 0.0,
        "gt_weight_protocols_count": 0.0,
        "gt_weight_estimated_execution_time": 0.25,
        "gt_weight_solver_reputation_score": 0.2,
        "gt_weight_solver_success_rate": 0.0
      }
    },
    "passed_solution_ids": ["sol_1", "sol_2", "sol_3"],
    "failed_solution_ids": [],
    "feature_vectors": [
      {
        "solution_id": "sol_1",
        "features": {
          "surplus_usd": 100.0,
          "surplus_percentage": 2.0,
          "gas_cost": 10.0,
          "protocol_fees": 5.0,
          "total_cost": 15.0,
          "total_hops": 2,
          "protocols_count": 1,
          "estimated_execution_time": 1000.0,
          "solver_reputation_score": 0.95,
          "solver_success_rate": 0.98
        }
      },
      {
        "solution_id": "sol_2",
        "features": {
          "surplus_usd": 95.0,
          "surplus_percentage": 1.8,
          "gas_cost": 8.0,
          "protocol_fees": 4.0,
          "total_cost": 12.0,
          "total_hops": 3,
          "protocols_count": 2,
          "estimated_execution_time": 1500.0,
          "solver_reputation_score": 0.88,
          "solver_success_rate": 0.92
        }
      },
      {
        "solution_id": "sol_3",
        "features": {
          "surplus_usd": 105.0,
          "surplus_percentage": 2.2,
          "gas_cost": 12.0,
          "protocol_fees": 6.0,
          "total_cost": 18.0,
          "total_hops": 1,
          "protocols_count": 1,
          "estimated_execution_time": 800.0,
          "solver_reputation_score": 0.92,
          "solver_success_rate": 0.96
        }
      }
    ],
    "dry_run_results": [],
    "stats": {
      "total_submitted": 3,
      "passed": 3,
      "failed": 0,
      "dry_run_executed": 3,
      "dry_run_successful": 3
    },
    "processed_at": 1744684007462
  }
}'

echo ""
echo "Sending request..."
echo ""

RESPONSE=$(curl -s -X POST "$ENCLAVE_URL/process_data" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

if [ $? -ne 0 ]; then
    echo "Error: Failed to connect to enclave"
    exit 1
fi

echo "Response:"
echo "$RESPONSE" | jq '.'

# Extract and display key information
echo ""
echo "Summary:"
echo "--------"
BEST_SOLUTION=$(echo "$RESPONSE" | jq -r '.response.data.best_solution.solution_id')
BEST_SCORE=$(echo "$RESPONSE" | jq -r '.response.data.best_solution.score')
TOTAL_SOLUTIONS=$(echo "$RESPONSE" | jq -r '.response.data.metadata.total_solutions')
AVERAGE_SCORE=$(echo "$RESPONSE" | jq -r '.response.data.metadata.average_score')

echo "Best Solution: $BEST_SOLUTION"
echo "Best Score: $BEST_SCORE"
echo "Total Solutions Ranked: $TOTAL_SOLUTIONS"
echo "Average Score: $AVERAGE_SCORE"

echo ""
echo "All Rankings:"
echo "-------------"
echo "$RESPONSE" | jq -r '.response.data.ranked_solutions[] | "Rank \(.rank): \(.solution_id) - Score: \(.score)"'

echo ""
echo "Signature (for on-chain verification):"
echo "---------------------------------------"
echo "$RESPONSE" | jq -r '.signature'

