// Copyright (c), Mysten Labs, Inc.
// SPDX-License-Identifier: Apache-2.0

module intenus_ranking::ranking;

use enclave::enclave::{Self, Enclave};
use std::string::String;

/// ====
/// Intenus Ranking - On-chain verification of ranked solutions
/// ====

const RANKING_INTENT: u8 = 0;
const EInvalidSignature: u64 = 1;
const EExpired: u64 = 2;
const ENoSolutions: u64 = 3;

/// Represents a verified ranking result stored on-chain
public struct RankingRecord has key, store {
    id: UID,
    intent_id: String,
    best_solution_id: String,
    best_solver_address: String,
    best_score: u64, // Scaled by 100 (e.g., 85.5 = 8550)
    total_solutions: u64,
    strategy: String,
    intent_category: String,
    ranked_at: u64,
    expires_at: u64,
    verified_at: u64,
}

/// OTW for capability
public struct RANKING has drop {}

/// Initialize the module
fun init(otw: RANKING, ctx: &mut TxContext) {
    let cap = enclave::new_cap(otw, ctx);

    cap.create_enclave_config(
        b"intenus-ranking enclave".to_string(),
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", // pcr0
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", // pcr1
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000", // pcr2
        ctx,
    );

    transfer::public_transfer(cap, ctx.sender())
}

/// Simplified ranking result for Move (matching Rust struct)
/// Note: Scores are scaled by 100 (e.g., 85.5% = 8550)
public struct RankingResultPayload has copy, drop {
    intent_id: String,
    best_solution_id: String,
    best_solver_address: String,
    best_score: u64,        // Scaled by 100 (0-10000 for 0-100.0)
    total_solutions: u64,
    strategy: String,
    intent_category: String,
    ranked_at: u64,
    expires_at: u64,
}

/// Verify and store ranking result on-chain
public fun verify_ranking<T>(
    intent_id: String,
    best_solution_id: String,
    best_solver_address: String,
    best_score: u64,
    total_solutions: u64,
    strategy: String,
    intent_category: String,
    ranked_at: u64,
    expires_at: u64,
    timestamp_ms: u64,
    sig: &vector<u8>,
    enclave: &Enclave<T>,
    clock: &sui::clock::Clock,
    ctx: &mut TxContext,
): RankingRecord {
    let current_time = clock.timestamp_ms();
    
    // Verify not expired
    assert!(expires_at > current_time, EExpired);
    
    // Verify at least one solution
    assert!(total_solutions > 0, ENoSolutions);

    let payload = RankingResultPayload {
        intent_id,
        best_solution_id,
        best_solver_address,
        best_score,
        total_solutions,
        strategy,
        intent_category,
        ranked_at,
        expires_at,
    };

    // Verify signature from enclave
    let res = enclave.verify_signature(
        RANKING_INTENT,
        timestamp_ms,
        payload,
        sig,
    );
    assert!(res, EInvalidSignature);

    // Create and return ranking record
    RankingRecord {
        id: object::new(ctx),
        intent_id,
        best_solution_id,
        best_solver_address,
        best_score,
        total_solutions,
        strategy,
        intent_category,
        ranked_at,
        expires_at,
        verified_at: current_time,
    }
}

/// Execute the best solution (placeholder - would integrate with actual execution)
public entry fun execute_best_solution<T>(
    _ranking_record: &RankingRecord,
    _enclave: &Enclave<T>,
    _ctx: &mut TxContext,
) {
    // In production, this would:
    // 1. Verify the ranking record is still valid
    // 2. Execute the transaction_bytes from the best solution
    // 3. Record the execution result
    // 4. Update solver reputation
}

/// Getters
public fun intent_id(record: &RankingRecord): &String {
    &record.intent_id
}

public fun best_solution_id(record: &RankingRecord): &String {
    &record.best_solution_id
}

public fun best_score(record: &RankingRecord): u64 {
    record.best_score
}

public fun total_solutions(record: &RankingRecord): u64 {
    record.total_solutions
}

#[test_only]
public fun destroy_for_testing(record: RankingRecord) {
    let RankingRecord { id, .. } = record;
    id.delete();
}

#[test]
fun test_ranking_verification() {
    use sui::test_scenario;
    use sui::clock;
    use enclave::enclave::{Cap, EnclaveConfig};
    
    let mut scenario = test_scenario::begin(@0x1);
    let mut clock = clock::create_for_testing(scenario.ctx());
    clock.set_for_testing(1744684007462);

    // Create capability and config
    let cap = enclave::new_cap(RANKING {}, scenario.ctx());
    cap.create_enclave_config(
        b"intenus-ranking enclave".to_string(),
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        x"000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        scenario.ctx(),
    );

    scenario.next_tx(@0x1);
    let mut config = scenario.take_shared<EnclaveConfig<RANKING>>();
    
    // Update with real PCRs (would come from actual build)
    config.update_pcrs(
        &cap,
        x"cbe1afb6ed0ff89f10295af0b802247ec5670da8f886e71a4226373b032c322f4e42c9c98288e7211682b258684505a2",
        x"cbe1afb6ed0ff89f10295af0b802247ec5670da8f886e71a4226373b032c322f4e42c9c98288e7211682b258684505a2",
        x"21b9efbc184807662e966d34f390821309eeac6802309798826296bf3e8bec7c10edb30948c90ba67310f7b964fc500a",
    );

    test_scenario::return_shared(config);
    clock.destroy_for_testing();
    sui::test_utils::destroy(cap);
    scenario.end();
}

