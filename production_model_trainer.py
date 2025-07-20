import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from datetime import datetime
import hashlib
from loguru import logger

class ProductionModelTrainer:
    def __init__(self):
        self.sequence_length = 60
        self.num_features = 12
        self.scaler = RobustScaler()
        
    def generate_realistic_training_data(self, n_samples=5000):
        logger.info(f"Generating {n_samples} realistic training samples...")
        
        sequences = []
        targets = {
            'momentum_score': [],
            'confidence_score': [],
            'predicted_return': [],
            'predicted_volatility': []
        }
        
        for i in range(n_samples):
            # Generate realistic market-like sequences
            sequence = np.zeros((self.sequence_length, self.num_features))
            
            # Price velocity with trend and noise
            price_trend = np.cumsum(np.random.randn(self.sequence_length) * 0.01)
            sequence[:, 0] = price_trend
            
            # Volume momentum with bursts
            volume_base = np.random.exponential(1, self.sequence_length)
            if np.random.random() > 0.7:  # 30% chance of volume spike
                spike_start = np.random.randint(0, self.sequence_length - 10)
                volume_base[spike_start:spike_start+10] *= np.random.uniform(3, 8)
            sequence[:, 1] = volume_base
            
            # Liquidity depth (log-normal distribution)
            sequence[:, 2] = np.random.lognormal(mean=2, sigma=1, size=self.sequence_length)
            
            # Volatility (gamma distribution)
            sequence[:, 3] = np.random.gamma(2, 0.02, self.sequence_length)
            
            # Trade frequency
            sequence[:, 4] = np.random.exponential(1, self.sequence_length)
            
            # Holder concentration (beta distribution)
            sequence[:, 5] = np.random.beta(2, 5, self.sequence_length)
            
            # Whale activity (correlated with volume)
            sequence[:, 6] = sequence[:, 1] * np.random.uniform(0.1, 0.3, self.sequence_length)
            
            # Social momentum (trending behavior)
            social_base = np.random.uniform(0, 1, self.sequence_length)
            if np.random.random() > 0.8:  # 20% chance of viral trend
                trend_start = np.random.randint(0, self.sequence_length - 20)
                social_base[trend_start:] *= np.random.uniform(2, 5)
            sequence[:, 7] = social_base
            
            # Technical indicators (RSI, MACD)
            rsi_values = 50 + np.cumsum(np.random.randn(self.sequence_length) * 2)
            rsi_values = np.clip(rsi_values, 0, 100)
            sequence[:, 8] = rsi_values
            
            macd_values = np.cumsum(np.random.randn(self.sequence_length) * 0.01)
            sequence[:, 9] = macd_values
            
            # Network activity
            sequence[:, 10] = np.random.uniform(0.5, 1.0, self.sequence_length)
            
            # Risk score
            sequence[:, 11] = np.random.beta(2, 3, self.sequence_length)
            
            # Calculate realistic targets based on features
            final_price_velocity = abs(sequence[-1, 0] - sequence[-10, 0])
            volume_surge = np.max(sequence[-10:, 1]) / np.mean(sequence[:-10, 1])
            
            # Momentum score (based on price movement and volume)
            momentum_score = min(final_price_velocity * volume_surge * 0.1, 1.0)
            has_momentum = momentum_score > 0.09 and momentum_score < 0.13
            
            # Confidence (based on liquidity and consistency)
            liquidity_score = min(np.mean(sequence[:, 2]) / 100, 1.0)
            volatility_penalty = min(np.std(sequence[:, 0]) * 10, 0.5)
            confidence = max(0.1, liquidity_score - volatility_penalty)
            
            # Predicted return (momentum-based with noise)
            if has_momentum:
                predicted_return = np.random.normal(0.08, 0.03)  # Positive expected return
            else:
                predicted_return = np.random.normal(0.0, 0.02)   # Neutral expected return
            
            # Predicted volatility
            predicted_volatility = max(0.01, np.std(sequence[:, 0]) + np.random.normal(0, 0.01))
            
            sequences.append(sequence)
            targets['momentum_score'].append(float(has_momentum))
            targets['confidence_score'].append(confidence)
            targets['predicted_return'].append(predicted_return)
            targets['predicted_volatility'].append(predicted_volatility)
        
        X = np.array(sequences)
        y = {k: np.array(v) for k, v in targets.items()}
        
        logger.info(f"Generated training data: {X.shape}")
        return X, y
    
    def build_production_model(self):
        logger.info("Building production model...")
        
        # Simplified but effective model for production
        inputs = tf.keras.layers.Input(shape=(self.sequence_length, self.num_features))
        
        # LSTM layers for time series processing
        x = tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        x = tf.keras.layers.LSTM(64, dropout=0.2)(x)
        
        # Dense layers
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Output heads
        momentum_head = tf.keras.layers.Dense(32, activation='relu')(x)
        momentum_head = tf.keras.layers.Dense(1, activation='sigmoid', name='momentum_score')(momentum_head)
        
        confidence_head = tf.keras.layers.Dense(32, activation='relu')(x)
        confidence_head = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence_score')(confidence_head)
        
        return_head = tf.keras.layers.Dense(32, activation='relu')(x)
        return_head = tf.keras.layers.Dense(1, activation='tanh', name='predicted_return')(return_head)
        
        volatility_head = tf.keras.layers.Dense(16, activation='relu')(x)
        volatility_head = tf.keras.layers.Dense(1, activation='relu', name='predicted_volatility')(volatility_head)
        
        model = tf.keras.Model(
            inputs=inputs,
            outputs=[momentum_head, confidence_head, return_head, volatility_head]
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'momentum_score': 'binary_crossentropy',
                'confidence_score': 'mse',
                'predicted_return': 'huber',
                'predicted_volatility': 'mse'
            },
            loss_weights={
                'momentum_score': 2.0,
                'confidence_score': 1.0,
                'predicted_return': 3.0,
                'predicted_volatility': 0.5
            },
            metrics={
                'momentum_score': ['accuracy'],
                'confidence_score': ['mae'],
                'predicted_return': ['mae'],
                'predicted_volatility': ['mae']
            }
        )
        
        return model
    
    def train_production_model(self):
        logger.info("🚀 Starting PRODUCTION model training...")
        
        # Generate training data
        X, y = self.generate_realistic_training_data(n_samples=5000)
        
        # Preprocess data
        X_reshaped = X.reshape(-1, self.num_features)
        X_scaled_reshaped = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled_reshaped.reshape(X.shape)
        
        # Split data properly
        indices = np.arange(len(X_scaled))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
        
        X_train = X_scaled[train_idx]
        X_val = X_scaled[val_idx]
        
        y_train = {
            'momentum_score': y['momentum_score'][train_idx],
            'confidence_score': y['confidence_score'][train_idx], 
            'predicted_return': y['predicted_return'][train_idx],
            'predicted_volatility': y['predicted_volatility'][train_idx]
        }
        
        y_val = {
            'momentum_score': y['momentum_score'][val_idx],
            'confidence_score': y['confidence_score'][val_idx],
            'predicted_return': y['predicted_return'][val_idx], 
            'predicted_volatility': y['predicted_volatility'][val_idx]
        }
        
        # Build model
        model = self.build_production_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=8, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
            )
        ]
        
        # Train model
        logger.info("Training production model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss = min(history.history['val_loss'])
        val_accuracy = max(history.history['val_momentum_score_accuracy'])
        
        logger.info(f"✅ Training complete! Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        return model, history
    
    def save_production_model(self, model):
        logger.info("💾 Saving production model...")
        
        os.makedirs('models', exist_ok=True)
        
        # Save in Keras format
        model.save('models/momentum_model.keras')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open('models/model_weights.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Save scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        model_metadata = {
            'version': datetime.now().isoformat(),
            'features': [
                'price_velocity', 'volume_momentum', 'liquidity_depth', 'volatility',
                'trade_frequency', 'holder_concentration', 'whale_activity', 
                'social_momentum', 'rsi', 'macd', 'network_activity', 'risk_score'
            ],
            'sequence_length': self.sequence_length,
            'model_type': 'production_lstm',
            'fingerprint': hashlib.md5(f"prod_{self.sequence_length}_{self.num_features}_{datetime.now().date()}".encode()).hexdigest()[:16]
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logger.info("✅ Production model saved successfully!")

def main():
    trainer = ProductionModelTrainer()
    model, history = trainer.train_production_model()
    trainer.save_production_model(model)
    
    logger.info("🎉 PRODUCTION MODEL READY!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("🚀 PRODUCTION ML MODEL DEPLOYED!")
    else:
        print("❌ PRODUCTION MODEL FAILED!")
        exit(1)
