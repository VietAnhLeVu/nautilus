"""
Intent Classifier Model Training Module

This module provides a comprehensive training pipeline for the intent classification model.
It includes data loading, preprocessing, model building, training, and evaluation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import requests
from io import StringIO

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class IntentClassifierTrainer:
    """
    A trainer class for the intent classification model.
    
    This class handles the complete training pipeline including:
    - Data loading and preprocessing
    - Feature engineering
    - Model architecture definition
    - Training with callbacks
    - Model evaluation and saving
    """
    
    def __init__(
        self,
        data_source: str,
        model_save_path: str = 'model/intent_classifier_model.keras',
        artifacts_save_path: str = 'model/model_artifacts.pkl',
        random_seed: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            data_source: URL or local path to the training data CSV file
            model_save_path: Path to save the trained model
            artifacts_save_path: Path to save preprocessing artifacts
            random_seed: Random seed for reproducibility
        """
        self.data_source = data_source
        self.model_save_path = Path(model_save_path)
        self.artifacts_save_path = Path(artifacts_save_path)
        self.random_seed = random_seed
        
        # Ensure output directories exist
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        
        # Initialize attributes
        self.df: pd.DataFrame = None
        self.model: keras.Model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.gt_encoders: Dict[str, LabelEncoder] = {}
        self.strategy_encoder: LabelEncoder = None
        self.scaler: StandardScaler = None
        self.history: keras.callbacks.History = None
        
        # Feature columns
        self.intent_features: List[str] = None
        self.weight_columns: List[str] = [
            'gt_weight_surplus_usd',
            'gt_weight_surplus_percentage',
            'gt_weight_gas_cost',
            'gt_weight_protocol_fees',
            'gt_weight_total_hops',
            'gt_weight_protocols_count',
            'gt_weight_estimated_execution_time',
            'gt_weight_solver_reputation_score',
            'gt_weight_solver_success_rate'
        ]
        
        logger.info("IntentClassifierTrainer initialized")
        logger.info(f"Data source: {self.data_source}")
        logger.info(f"Model save path: {self.model_save_path}")
        logger.info(f"Artifacts save path: {self.artifacts_save_path}")
    
    def load_data(self) -> None:
        """
        Load training data from URL or local file.
        
        Supports:
        - HTTP/HTTPS URLs (fetches CSV data)
        - Local file paths
        """
        logger.info("Loading training data...")
        
        # Check if data source is URL or local path
        if self.data_source.startswith(('http://', 'https://')):
            logger.info(f"Fetching data from URL: {self.data_source}")
            try:
                response = requests.get(self.data_source, timeout=30)
                response.raise_for_status()
                
                # Parse CSV from response
                csv_data = StringIO(response.text)
                self.df = pd.read_csv(csv_data)
                logger.info("Data fetched successfully from URL")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch data from URL: {e}")
                raise RuntimeError(f"Failed to fetch data from URL: {e}")
        else:
            # Local file path
            data_path = Path(self.data_source)
            if not data_path.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.df = pd.read_csv(data_path)
            logger.info("Data loaded from local file")
        
        logger.info(f"Loaded {len(self.df)} samples")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        # Log strategy distribution
        if 'strategy' in self.df.columns:
            strategy_counts = self.df['strategy'].value_counts()
            logger.info("Strategy distribution:")
            for strategy, count in strategy_counts.items():
                logger.info(f"  {strategy}: {count} ({count/len(self.df)*100:.1f}%)")
    
    @staticmethod
    def parse_asset_types(x: Any) -> int:
        """
        Parse asset types from string or list format.
        
        Args:
            x: Asset types in various formats (JSON, comma-separated, etc.)
            
        Returns:
            Number of asset types
        """
        try:
            if isinstance(x, str):
                if x.startswith('['):
                    parsed = json.loads(x)
                    return len(parsed)
                else:
                    return len(x.split(','))
            return 0
        except Exception:
            return 0
    
    def preprocess_data(self) -> None:
        """Preprocess and encode features."""
        logger.info("Preprocessing data...")
        
        # Parse asset types
        logger.info("Parsing asset types...")
        self.df['input_asset_count'] = self.df['input_asset_types'].apply(self.parse_asset_types)
        self.df['output_asset_count'] = self.df['output_asset_types'].apply(self.parse_asset_types)
        
        # Encode categorical features
        logger.info("Encoding categorical features...")
        categorical_features = ['time_in_force', 'optimization_goal', 'benchmark_source', 'client_platform']
        
        for col in categorical_features:
            le = LabelEncoder()
            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"  {col}: {len(le.classes_)} classes - {list(le.classes_)}")
        
        # Define intent features (30 total)
        self.intent_features = [
            # Numerical (21)
            'solver_window_ms', 'user_decision_timeout_ms', 'time_to_deadline_ms',
            'max_slippage_bps', 'max_gas_cost_usd', 'max_hops',
            'surplus_weight', 'gas_cost_weight', 'execution_speed_weight', 'reputation_weight',
            'input_count', 'output_count', 'input_value_usd', 'expected_output_value_usd',
            'benchmark_confidence', 'expected_gas_usd', 'expected_slippage_bps',
            'nlp_confidence', 'tag_count', 'input_asset_count', 'output_asset_count',
            # Encoded categorical (4)
            'time_in_force_encoded', 'optimization_goal_encoded',
            'benchmark_source_encoded', 'client_platform_encoded',
            # Boolean (5)
            'has_whitelist', 'has_blacklist', 'has_limit_price',
            'require_simulation', 'has_nlp_input'
        ]
        
        logger.info(f"Total intent features: {len(self.intent_features)}")
        
        # Encode ground truth labels
        logger.info("Encoding ground truth labels...")
        gt_labels = ['primary_category', 'detected_priority', 'complexity_level', 'risk_level']
        
        for label in gt_labels:
            le = LabelEncoder()
            self.df[label + '_encoded'] = le.fit_transform(self.df[label].astype(str))
            self.gt_encoders[label] = le
            logger.info(f"  {label}: {len(le.classes_)} classes - {list(le.classes_)}")
        
        # Encode strategy
        logger.info("Encoding strategy...")
        self.strategy_encoder = LabelEncoder()
        self.df['strategy_encoded'] = self.strategy_encoder.fit_transform(self.df['strategy'].astype(str))
        logger.info(f"  strategy: {len(self.strategy_encoder.classes_)} classes - {list(self.strategy_encoder.classes_)}")
        
        logger.info("Preprocessing complete")
    
    def prepare_train_test_split(self, test_size: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Prepare train/test split with feature scaling.
        
        Args:
            test_size: Fraction of data to use for testing
            
        Returns:
            Tuple of train and test arrays
        """
        logger.info("Preparing train/test split...")
        
        # Extract features
        X = self.df[self.intent_features].values
        
        # Extract labels
        y_pc = self.df['primary_category_encoded'].values
        y_dp = self.df['detected_priority_encoded'].values
        y_cl = self.df['complexity_level_encoded'].values
        y_rl = self.df['risk_level_encoded'].values
        y_st = self.df['strategy_encoded'].values
        y_weights = self.df[self.weight_columns].values
        
        # Convert labels to categorical
        num_pc_classes = len(self.gt_encoders['primary_category'].classes_)
        num_dp_classes = len(self.gt_encoders['detected_priority'].classes_)
        num_cl_classes = len(self.gt_encoders['complexity_level'].classes_)
        num_rl_classes = len(self.gt_encoders['risk_level'].classes_)
        num_st_classes = len(self.strategy_encoder.classes_)
        
        y_pc_cat = keras.utils.to_categorical(y_pc, num_pc_classes)
        y_dp_cat = keras.utils.to_categorical(y_dp, num_dp_classes)
        y_cl_cat = keras.utils.to_categorical(y_cl, num_cl_classes)
        y_rl_cat = keras.utils.to_categorical(y_rl, num_rl_classes)
        y_st_cat = keras.utils.to_categorical(y_st, num_st_classes)
        
        # Train/test split
        logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
        split_result = train_test_split(
            X, y_pc_cat, y_dp_cat, y_cl_cat, y_rl_cat, y_st_cat, y_weights,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y_st
        )
        
        (X_train, X_test, y_train_pc, y_test_pc, y_train_dp, y_test_dp,
         y_train_cl, y_test_cl, y_train_rl, y_test_rl, y_train_st, y_test_st,
         y_train_w, y_test_w) = split_result
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # Scale features
        logger.info("Scaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Reshape for CNN1D (batch_size, sequence_length, features)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], -1, 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], -1, 1)
        
        logger.info(f"Input shape: {X_train_reshaped.shape}")
        
        return (X_train_reshaped, X_test_reshaped,
                y_train_pc, y_test_pc, y_train_dp, y_test_dp,
                y_train_cl, y_test_cl, y_train_rl, y_test_rl,
                y_train_st, y_test_st, y_train_w, y_test_w)
    
    def build_model(self) -> None:
        """Build the CNN-based multi-task model."""
        logger.info("Building model architecture...")
        
        num_features = len(self.intent_features)
        num_pc_classes = len(self.gt_encoders['primary_category'].classes_)
        num_dp_classes = len(self.gt_encoders['detected_priority'].classes_)
        num_cl_classes = len(self.gt_encoders['complexity_level'].classes_)
        num_rl_classes = len(self.gt_encoders['risk_level'].classes_)
        num_st_classes = len(self.strategy_encoder.classes_)
        num_weights = len(self.weight_columns)
        
        # Input layer
        inputs = layers.Input(shape=(num_features, 1), name='intent_input')
        
        # CNN layers
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.GlobalMaxPooling1D()(x)
        
        # Shared dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output heads
        primary_category_output = layers.Dense(
            num_pc_classes, activation='softmax', name='primary_category'
        )(x)
        
        detected_priority_output = layers.Dense(
            num_dp_classes, activation='softmax', name='detected_priority'
        )(x)
        
        complexity_level_output = layers.Dense(
            num_cl_classes, activation='softmax', name='complexity_level'
        )(x)
        
        risk_level_output = layers.Dense(
            num_rl_classes, activation='softmax', name='risk_level'
        )(x)
        
        strategy_output = layers.Dense(
            num_st_classes, activation='softmax', name='strategy'
        )(x)
        
        weights_output = layers.Dense(
            num_weights, activation='softmax', name='weights'
        )(x)
        
        # Create model
        self.model = Model(
            inputs=inputs,
            outputs=[
                primary_category_output,
                detected_priority_output,
                complexity_level_output,
                risk_level_output,
                strategy_output,
                weights_output
            ]
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'primary_category': 'categorical_crossentropy',
                'detected_priority': 'categorical_crossentropy',
                'complexity_level': 'categorical_crossentropy',
                'risk_level': 'categorical_crossentropy',
                'strategy': 'categorical_crossentropy',
                'weights': 'mse'
            },
            loss_weights={
                'primary_category': 1.0,
                'detected_priority': 1.0,
                'complexity_level': 1.0,
                'risk_level': 1.0,
                'strategy': 2.0,
                'weights': 3.0
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
        
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {self.model.count_params():,}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val_dict: Dict[str, np.ndarray],
        epochs: int = 150,
        batch_size: int = 32
    ) -> None:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train_dict: Training labels dictionary
            X_val: Validation features
            y_val_dict: Validation labels dictionary
            epochs: Maximum number of epochs
            batch_size: Batch size for training
        """
        logger.info("Starting training...")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                str(self.model_save_path),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed")
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test_dict: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test_dict: Test labels dictionary
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set...")
        
        test_loss = self.model.evaluate(X_test, y_test_dict, verbose=0)
        
        metrics = {
            'total_loss': test_loss[0],
            'primary_category_accuracy': test_loss[6],
            'detected_priority_accuracy': test_loss[7],
            'complexity_level_accuracy': test_loss[8],
            'risk_level_accuracy': test_loss[9],
            'strategy_accuracy': test_loss[10],
            'weights_mae': test_loss[11]
        }
        
        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Loss: {metrics['total_loss']:.4f}")
        logger.info("")
        logger.info("Accuracy Metrics:")
        logger.info(f"  Primary Category: {metrics['primary_category_accuracy']:.2%}")
        logger.info(f"  Detected Priority: {metrics['detected_priority_accuracy']:.2%}")
        logger.info(f"  Complexity Level: {metrics['complexity_level_accuracy']:.2%}")
        logger.info(f"  Risk Level: {metrics['risk_level_accuracy']:.2%}")
        logger.info(f"  Strategy: {metrics['strategy_accuracy']:.2%}")
        logger.info(f"  Weights MAE: {metrics['weights_mae']:.4f}")
        logger.info("=" * 60)
        
        return metrics
    
    def save_artifacts(self) -> None:
        """Save preprocessing artifacts."""
        logger.info("Saving preprocessing artifacts...")
        
        artifacts = {
            'label_encoders': self.label_encoders,
            'gt_encoders': self.gt_encoders,
            'strategy_encoder': self.strategy_encoder,
            'scaler': self.scaler,
            'intent_features': self.intent_features,
            'weight_columns': self.weight_columns
        }
        
        with open(self.artifacts_save_path, 'wb') as f:
            pickle.dump(artifacts, f)
        
        logger.info(f"Artifacts saved to {self.artifacts_save_path}")
    
    def run_training_pipeline(
        self,
        test_size: float = 0.2,
        epochs: int = 150,
        batch_size: int = 32
    ) -> Dict[str, float]:
        """
        Run the complete training pipeline.
        
        Args:
            test_size: Fraction of data for testing
            epochs: Maximum training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Prepare data
        (X_train, X_test,
         y_train_pc, y_test_pc, y_train_dp, y_test_dp,
         y_train_cl, y_test_cl, y_train_rl, y_test_rl,
         y_train_st, y_test_st, y_train_w, y_test_w) = self.prepare_train_test_split(test_size)
        
        # Build model
        self.build_model()
        
        # Train
        y_train_dict = {
            'primary_category': y_train_pc,
            'detected_priority': y_train_dp,
            'complexity_level': y_train_cl,
            'risk_level': y_train_rl,
            'strategy': y_train_st,
            'weights': y_train_w
        }
        
        y_test_dict = {
            'primary_category': y_test_pc,
            'detected_priority': y_test_dp,
            'complexity_level': y_test_cl,
            'risk_level': y_test_rl,
            'strategy': y_test_st,
            'weights': y_test_w
        }
        
        self.train(X_train, y_train_dict, X_test, y_test_dict, epochs, batch_size)
        
        # Evaluate
        metrics = self.evaluate(X_test, y_test_dict)
        
        # Save artifacts
        self.save_artifacts()
        
        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {self.model_save_path}")
        logger.info(f"Artifacts saved to: {self.artifacts_save_path}")
        
        return metrics


def main():
    """Main training function."""
    # Configuration
    DATA_SOURCE = 'https://wal-aggregator-testnet.staketab.org/v1/blobs/by-quilt-patch-id/yoSJPzv1ykudTccWun7qElklLEmvY3J9fK3mPZzUeroBAQDoAQ'
    MODEL_SAVE_PATH = '/home/nguyen-viet-an/intenus/ai_model/model/intent_classifier_model.keras'
    ARTIFACTS_SAVE_PATH = '/home/nguyen-viet-an/intenus/ai_model/model/model_artifacts.pkl'
    
    # Create trainer
    trainer = IntentClassifierTrainer(
        data_source=DATA_SOURCE,
        model_save_path=MODEL_SAVE_PATH,
        artifacts_save_path=ARTIFACTS_SAVE_PATH
    )
    
    # Run training pipeline
    metrics = trainer.run_training_pipeline(
        test_size=0.2,
        epochs=150,
        batch_size=32
    )
    
    return metrics


if __name__ == "__main__":
    main()
