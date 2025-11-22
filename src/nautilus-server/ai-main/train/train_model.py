import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("🚀 Loading training data...")
df = pd.read_csv('/home/nguyen-viet-an/intenus/ai_model/training_data.csv')
print(f"✅ Loaded {len(df)} samples")

print("\n📊 Processing features...")

# Parse JSON arrays
def parse_asset_types(x):
    try:
        parsed = json.loads(x)
        return len(parsed)
    except:
        return 0

df['input_asset_count'] = df['input_asset_types'].apply(parse_asset_types)
df['output_asset_count'] = df['output_asset_types'].apply(parse_asset_types)

label_encoders = {}
categorical_features = ['time_in_force', 'optimization_goal', 'benchmark_source', 'client_platform']

for col in categorical_features:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

intent_features = [
    'solver_window_ms', 'user_decision_timeout_ms', 'time_to_deadline_ms',
    'max_slippage_bps', 'max_gas_cost_usd', 'max_hops',
    'surplus_weight', 'gas_cost_weight', 'execution_speed_weight', 'reputation_weight',
    'input_count', 'output_count', 'input_value_usd', 'expected_output_value_usd',
    'benchmark_confidence', 'expected_gas_usd', 'expected_slippage_bps',
    'nlp_confidence', 'tag_count', 'input_asset_count', 'output_asset_count'
]

intent_features.extend([col + '_encoded' for col in categorical_features])

boolean_features = ['has_whitelist', 'has_blacklist', 'has_limit_price', 
                   'require_simulation', 'has_nlp_input']
for col in boolean_features:
    df[col + '_int'] = df[col].astype(int)
    intent_features.append(col + '_int')

print(f"✅ Total intent features: {len(intent_features)}")

gt_encoders = {}
gt_labels = ['primary_category', 'detected_priority', 'complexity_level', 'risk_level']

for label in gt_labels:
    le = LabelEncoder()
    df[label + '_encoded'] = le.fit_transform(df[label])
    gt_encoders[label] = le
    print(f"   {label}: {len(le.classes_)} classes - {list(le.classes_)}")

# Encode Strategy
strategy_encoder = LabelEncoder()
df['strategy_encoded'] = strategy_encoder.fit_transform(df['strategy'])
print(f"   strategy: {len(strategy_encoder.classes_)} classes - {list(strategy_encoder.classes_)}")

# ============================
# 3. PREPARE WEIGHTS (REGRESSION TARGETS)
# ============================
print("\n⚖️  Processing weights...")

# For each sample, collect the non-empty weights based on strategy
def get_weights_for_strategy(row):
    strategy = row['strategy']
    if strategy == 'Surplus-First':
        return [row['surplus_usd_weight'], row['surplus_pct_weight'], row['price_impact_weight']]
    elif strategy == 'Gas-Optimized':
        return [row['gas_cost_weight'], row['gas_efficiency_weight'], row['total_cost_weight']]
    elif strategy == 'Reputation-Trust':
        return [row['success_rate_weight'], row['reliability_weight'], row['guarantee_weight']]
    return [0, 0, 0]

# Create weight columns
weight_cols = ['weight_1', 'weight_2', 'weight_3']
df[weight_cols] = df.apply(get_weights_for_strategy, axis=1, result_type='expand')

# Fill NaN values in weight columns
for col in ['surplus_usd_weight', 'surplus_pct_weight', 'price_impact_weight',
            'gas_cost_weight', 'gas_efficiency_weight', 'total_cost_weight',
            'success_rate_weight', 'reliability_weight', 'guarantee_weight']:
    df[col] = df[col].fillna(0)

print("✅ Weights processed")

X = df[intent_features].values
y_primary_category = df['primary_category_encoded'].values
y_detected_priority = df['detected_priority_encoded'].values
y_complexity_level = df['complexity_level_encoded'].values
y_risk_level = df['risk_level_encoded'].values
y_strategy = df['strategy_encoded'].values
y_weights = df[weight_cols].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN1D (samples, timesteps, features)
# We'll treat each feature as a time step
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# Split data
X_train, X_test, \
y_train_pc, y_test_pc, \
y_train_dp, y_test_dp, \
y_train_cl, y_test_cl, \
y_train_rl, y_test_rl, \
y_train_st, y_test_st, \
y_train_w, y_test_w = train_test_split(
    X_cnn, y_primary_category, y_detected_priority, y_complexity_level,
    y_risk_level, y_strategy, y_weights,
    test_size=0.2, random_state=42, stratify=y_strategy
)

print(f"✅ Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ============================
# 5. BUILD CNN1D MULTI-TASK MODEL
# ============================
print("\n🏗️  Building CNN1D Multi-Task Model...")

input_shape = (X_cnn.shape[1], 1)
inputs = layers.Input(shape=input_shape)

# CNN1D Feature Extraction
x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.4)(x)

# Shared dense layer
shared = layers.Dense(256, activation='relu')(x)
shared = layers.BatchNormalization()(shared)
shared = layers.Dropout(0.3)(shared)

# ===== Classification Heads =====

# Primary Category Head
pc_hidden = layers.Dense(128, activation='relu', name='pc_hidden')(shared)
pc_output = layers.Dense(len(gt_encoders['primary_category'].classes_), 
                         activation='softmax', name='primary_category')(pc_hidden)

# Detected Priority Head
dp_hidden = layers.Dense(128, activation='relu', name='dp_hidden')(shared)
dp_output = layers.Dense(len(gt_encoders['detected_priority'].classes_), 
                         activation='softmax', name='detected_priority')(dp_hidden)

# Complexity Level Head
cl_hidden = layers.Dense(128, activation='relu', name='cl_hidden')(shared)
cl_output = layers.Dense(len(gt_encoders['complexity_level'].classes_), 
                         activation='softmax', name='complexity_level')(cl_hidden)

# Risk Level Head
rl_hidden = layers.Dense(128, activation='relu', name='rl_hidden')(shared)
rl_output = layers.Dense(len(gt_encoders['risk_level'].classes_), 
                         activation='softmax', name='risk_level')(rl_hidden)

# Strategy Head
st_hidden = layers.Dense(128, activation='relu', name='st_hidden')(shared)
st_output = layers.Dense(len(strategy_encoder.classes_), 
                         activation='softmax', name='strategy')(st_hidden)

# ===== Regression Heads for Weights =====

# Weights Head (3 weights that sum to 1)
weights_hidden = layers.Dense(128, activation='relu', name='weights_hidden')(shared)
weights_hidden = layers.Dropout(0.2)(weights_hidden)
weights_output = layers.Dense(3, activation='softmax', name='weights')(weights_hidden)  # softmax ensures sum=1

# Build model
model = Model(
    inputs=inputs,
    outputs=[pc_output, dp_output, cl_output, rl_output, st_output, weights_output]
)

# Compile with different losses and weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'primary_category': 'sparse_categorical_crossentropy',
        'detected_priority': 'sparse_categorical_crossentropy',
        'complexity_level': 'sparse_categorical_crossentropy',
        'risk_level': 'sparse_categorical_crossentropy',
        'strategy': 'sparse_categorical_crossentropy',
        'weights': 'mse'
    },
    loss_weights={
        'primary_category': 1.0,
        'detected_priority': 1.0,
        'complexity_level': 1.0,
        'risk_level': 1.0,
        'strategy': 1.5,  # Strategy is important
        'weights': 2.0    # Weights are critical
    },
    metrics={
        'primary_category': 'accuracy',
        'detected_priority': 'accuracy',
        'complexity_level': 'accuracy',
        'risk_level': 'accuracy',
        'strategy': 'accuracy',
        'weights': 'mae'
    }
)

print("✅ Model built successfully!")
print(f"\n📋 Model Summary:")
model.summary()

print("\n🎓 Training model...")

# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        '/home/nguyen-viet-an/intenus/ai_model/best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

history = model.fit(
    X_train,
    {
        'primary_category': y_train_pc,
        'detected_priority': y_train_dp,
        'complexity_level': y_train_cl,
        'risk_level': y_train_rl,
        'strategy': y_train_st,
        'weights': y_train_w
    },
    validation_data=(
        X_test,
        {
            'primary_category': y_test_pc,
            'detected_priority': y_test_dp,
            'complexity_level': y_test_cl,
            'risk_level': y_test_rl,
            'strategy': y_test_st,
            'weights': y_test_w
        }
    ),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================
# 7. EVALUATE MODEL
# ============================
print("\n📊 Evaluating model...")

test_results = model.evaluate(
    X_test,
    {
        'primary_category': y_test_pc,
        'detected_priority': y_test_dp,
        'complexity_level': y_test_cl,
        'risk_level': y_test_rl,
        'strategy': y_test_st,
        'weights': y_test_w
    },
    verbose=0
)

print("\n🎯 Test Results:")
metric_names = model.metrics_names
for i, (name, value) in enumerate(zip(metric_names, test_results)):
    if 'accuracy' in name or 'mae' in name:
        print(f"   {name}: {value:.4f}")

# ============================
# 8. SAVE ARTIFACTS
# ============================
print("\n💾 Saving model and artifacts...")

# Save model
model.save('/home/nguyen-viet-an/intenus/ai_model/intent_classifier_model.keras')

# Save encoders and scaler
artifacts = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'gt_encoders': gt_encoders,
    'strategy_encoder': strategy_encoder,
    'intent_features': intent_features,
    'feature_columns': intent_features
}

with open('/home/nguyen-viet-an/intenus/ai_model/model_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("✅ Model saved to: intent_classifier_model.keras")
print("✅ Artifacts saved to: model_artifacts.pkl")

# ============================
# 9. TEST INFERENCE
# ============================
print("\n🔮 Testing inference on random samples...")

# Select 3 random test samples
test_indices = np.random.choice(len(X_test), 3, replace=False)

for idx in test_indices:
    sample = X_test[idx:idx+1]
    
    # Predict
    predictions = model.predict(sample, verbose=0)
    pred_pc, pred_dp, pred_cl, pred_rl, pred_st, pred_weights = predictions
    
    # Decode predictions
    pc_class = gt_encoders['primary_category'].inverse_transform([np.argmax(pred_pc[0])])[0]
    dp_class = gt_encoders['detected_priority'].inverse_transform([np.argmax(pred_dp[0])])[0]
    cl_class = gt_encoders['complexity_level'].inverse_transform([np.argmax(pred_cl[0])])[0]
    rl_class = gt_encoders['risk_level'].inverse_transform([np.argmax(pred_rl[0])])[0]
    st_class = strategy_encoder.inverse_transform([np.argmax(pred_st[0])])[0]
    weights = pred_weights[0]
    
    print(f"\n--- Sample {idx+1} ---")
    print(f"Ground Truth:")
    print(f"  Primary Category: {gt_encoders['primary_category'].inverse_transform([y_test_pc[idx]])[0]}")
    print(f"  Detected Priority: {gt_encoders['detected_priority'].inverse_transform([y_test_dp[idx]])[0]}")
    print(f"  Complexity: {gt_encoders['complexity_level'].inverse_transform([y_test_cl[idx]])[0]}")
    print(f"  Risk: {gt_encoders['risk_level'].inverse_transform([y_test_rl[idx]])[0]}")
    print(f"  Strategy: {strategy_encoder.inverse_transform([y_test_st[idx]])[0]}")
    print(f"  Weights: {y_test_w[idx]}")
    
    print(f"\nPredictions:")
    print(f"  Primary Category: {pc_class} (conf: {np.max(pred_pc[0]):.3f})")
    print(f"  Detected Priority: {dp_class} (conf: {np.max(pred_dp[0]):.3f})")
    print(f"  Complexity: {cl_class} (conf: {np.max(pred_cl[0]):.3f})")
    print(f"  Risk: {rl_class} (conf: {np.max(pred_rl[0]):.3f})")
    print(f"  Strategy: {st_class} (conf: {np.max(pred_st[0]):.3f})")
    print(f"  Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}] (sum: {weights.sum():.3f})")

print("\n" + "="*60)
print("🎉 Training completed successfully!")
print("="*60)
