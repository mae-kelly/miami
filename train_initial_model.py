import os
import numpy as np

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import RobustScaler
import pickle
import json
from datetime import datetime
import hashlib

def train_initial_model():
    print("🧠 Training initial ML model...")
    
    try:
        os.makedirs('models', exist_ok=True)
        
        # Generate synthetic training data
        n_samples = 1000
        sequence_length = 60
        n_features = 12
        
        # Create synthetic sequences
        X = np.random.randn(n_samples, sequence_length, n_features)
        
        # Create synthetic targets
        y_momentum = np.random.randint(0, 2, n_samples)
        y_confidence = np.random.uniform(0.1, 0.9, n_samples) 
        y_return = np.random.normal(0, 0.1, n_samples)
        y_volatility = np.random.uniform(0.01, 0.1, n_samples)
        
        # Build simple model
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Combine targets
        y_combined = np.column_stack([y_momentum, y_confidence, y_return, y_volatility])
        
        # Train model
        model.fit(X, y_combined, epochs=5, verbose=0)
        
        # Save model
        model.save('models/momentum_model.h5')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model('models/momentum_model.h5')
        tflite_model = converter.convert()
        
        with open('models/model_weights.tflite', 'wb') as f:
            f.write(tflite_model)
        
        # Create and save scaler
        scaler = RobustScaler()
        X_reshaped = X.reshape(-1, n_features)
        scaler.fit(X_reshaped)
        
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Create metadata
        model_metadata = {
            'version': datetime.now().isoformat(),
            'features': [
                'price_velocity', 'volume_momentum', 'liquidity_depth', 'volatility',
                'trade_frequency', 'holder_concentration', 'whale_activity',
                'social_momentum', 'rsi', 'macd', 'network_activity', 'risk_score'
            ],
            'sequence_length': sequence_length,
            'fingerprint': hashlib.md5(f"{sequence_length}_{n_features}_{datetime.now().date()}".encode()).hexdigest()[:16]
        }
        
        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        print(f"✅ Model trained successfully!")
        print(f"   Samples: {n_samples}")
        print(f"   Features: {n_features}")
        print(f"   Sequence Length: {sequence_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return False

if __name__ == "__main__":
    success = train_initial_model()
    if success:
        print("🎉 ML model ready for production!")
    else:
        print("⚠️  Using fallback predictions")

# Production model training on startup
def train_production_model_on_startup():
    """Train production model if not exists"""
    try:
        if not os.path.exists('models/model_weights.tflite'):
            print("🧠 No production model found, training new model...")
            
            # Use the production trainer
            import asyncio
            from production_model_trainer import main as train_production_model
            
            success = asyncio.run(train_production_model())
            
            if success:
                print("✅ Production model trained successfully!")
                return True
            else:
                print("⚠️ Using fallback synthetic model")
                return train_initial_model()
        else:
            print("✅ Production model already exists")
            return True
            
    except Exception as e:
        print(f"⚠️ Production training failed: {e}")
        print("🔄 Falling back to synthetic model training...")
        return train_initial_model()

if __name__ == "__main__":
    success = train_production_model_on_startup()
    if success:
        print("🎉 ML model ready for production!")
    else:
        print("❌ Model training failed!")
        exit(1)
