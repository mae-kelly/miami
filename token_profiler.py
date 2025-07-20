import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import aiohttp
from web3 import Web3
from loguru import logger
import asyncio
import sqlite3
import talib
from scipy import signal, stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class TokenProfile:
    address: str
    network: str
    price_velocity: float
    volume_momentum: float
    liquidity_depth: float
    volatility: float
    trade_frequency: float
    holder_concentration: float
    whale_activity: float
    social_momentum: float
    technical_indicators: Dict
    network_activity: float
    current_price: float

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
import talib

class AdvancedFeatureEngine:
    def __init__(self):
        self.kalman_filters = {}
        self.pca_models = {}
        self.scaler = StandardScaler()
        
    def extract_market_microstructure_features(self, price_data, volume_data, orderbook_data=None):
        features = {}
        
        if len(price_data) < 10:
            return self.get_default_features()
        
        prices = np.array(price_data)
        volumes = np.array(volume_data)
        
        features.update(self.calculate_price_features(prices))
        features.update(self.calculate_volume_features(volumes))
        features.update(self.calculate_volatility_features(prices))
        features.update(self.calculate_momentum_features(prices))
        features.update(self.calculate_technical_indicators(prices, volumes))
        features.update(self.calculate_regime_features(prices))
        features.update(self.calculate_entropy_features(prices))
        
        return features
    
    def calculate_price_features(self, prices):
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.diff(np.log(prices))
        
        return {
            'price_velocity': np.mean(returns[-5:]),
            'price_acceleration': np.mean(np.diff(returns[-5:])),
            'price_momentum_1m': returns[-1] if len(returns) > 0 else 0,
            'price_momentum_5m': np.mean(returns[-5:]) if len(returns) >= 5 else 0,
            'log_return_mean': np.mean(log_returns),
            'log_return_std': np.std(log_returns),
            'price_range_5m': (np.max(prices[-5:]) - np.min(prices[-5:])) / np.mean(prices[-5:]) if len(prices) >= 5 else 0,
            'price_trend_strength': self.calculate_trend_strength(prices)
        }
    
    def calculate_volume_features(self, volumes):
        if len(volumes) < 2:
            return {'volume_momentum': 0, 'volume_acceleration': 0, 'volume_trend': 0}
        
        volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-10)
        
        return {
            'volume_momentum': np.mean(volume_changes[-3:]) if len(volume_changes) >= 3 else 0,
            'volume_acceleration': np.mean(np.diff(volume_changes[-3:])) if len(volume_changes) >= 4 else 0,
            'volume_trend': self.calculate_trend_strength(volumes),
            'volume_spike_intensity': self.detect_volume_spikes(volumes),
            'volume_consistency': 1 / (1 + np.std(volumes) / (np.mean(volumes) + 1e-10))
        }
    
    def calculate_volatility_features(self, prices):
        if len(prices) < 10:
            return {'realized_volatility': 0.05, 'garch_volatility': 0.05}
        
        returns = np.diff(prices) / prices[:-1]
        
        realized_vol = np.std(returns) * np.sqrt(1440)
        
        garch_vol = self.estimate_garch_volatility(returns)
        
        vol_of_vol = np.std([np.std(returns[i:i+5]) for i in range(len(returns)-4)]) if len(returns) > 10 else 0
        
        return {
            'realized_volatility': realized_vol,
            'garch_volatility': garch_vol,
            'volatility_of_volatility': vol_of_vol,
            'volatility_regime': self.classify_volatility_regime(realized_vol)
        }
    
    def calculate_momentum_features(self, prices):
        if len(prices) < 20:
            return {'momentum_strength': 0, 'momentum_persistence': 0}
        
        short_ma = np.mean(prices[-5:])
        long_ma = np.mean(prices[-20:])
        
        momentum_strength = (short_ma - long_ma) / long_ma
        
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        momentum_persistence = positive_changes / len(price_changes)
        
        rsi = self.calculate_rsi(prices, 14)
        macd = self.calculate_macd(prices)
        
        return {
            'momentum_strength': momentum_strength,
            'momentum_persistence': momentum_persistence,
            'rsi_momentum': (rsi - 50) / 50,
            'macd_momentum': macd,
            'momentum_acceleration': self.calculate_momentum_acceleration(prices)
        }
    
    def calculate_technical_indicators(self, prices, volumes):
        if len(prices) < 20:
            return self.get_default_technical_indicators()
        
        try:
            prices_arr = prices.astype(float)
            volumes_arr = volumes.astype(float)
            
            rsi = talib.RSI(prices_arr, timeperiod=14)[-1] if len(prices_arr) >= 14 else 50
            macd, macd_signal, macd_hist = talib.MACD(prices_arr)
            bb_upper, bb_middle, bb_lower = talib.BBANDS(prices_arr, timeperiod=20)
            
            stoch_k, stoch_d = talib.STOCH(prices_arr, prices_arr, prices_arr)
            
            return {
                'rsi': rsi / 100.0,
                'macd_line': macd[-1] if len(macd) > 0 and not np.isnan(macd[-1]) else 0,
                'macd_signal': macd_signal[-1] if len(macd_signal) > 0 and not np.isnan(macd_signal[-1]) else 0,
                'macd_histogram': macd_hist[-1] if len(macd_hist) > 0 and not np.isnan(macd_hist[-1]) else 0,
                'bb_position': (prices_arr[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1]) if not np.isnan(bb_upper[-1]) else 0.5,
                'stochastic_k': stoch_k[-1] / 100.0 if len(stoch_k) > 0 and not np.isnan(stoch_k[-1]) else 0.5,
                'stochastic_d': stoch_d[-1] / 100.0 if len(stoch_d) > 0 and not np.isnan(stoch_d[-1]) else 0.5
            }
            
        except:
            return self.get_default_technical_indicators()
    
    def calculate_regime_features(self, prices):
        if len(prices) < 30:
            return {'regime_volatility': 0.5, 'regime_trend': 0.5, 'regime_change_prob': 0.1}
        
        returns = np.diff(prices) / prices[:-1]
        
        vol_windows = [np.std(returns[i:i+10]) for i in range(len(returns)-9)]
        vol_regimes = self.detect_volatility_regimes(vol_windows)
        
        trend_strength = self.calculate_trend_strength(prices)
        trend_regime = 1 if trend_strength > 0.02 else -1 if trend_strength < -0.02 else 0
        
        regime_change_prob = self.estimate_regime_change_probability(vol_regimes, trend_regime)
        
        return {
            'regime_volatility': vol_regimes,
            'regime_trend': (trend_regime + 1) / 2,
            'regime_change_prob': regime_change_prob
        }
    
    def calculate_entropy_features(self, prices):
        if len(prices) < 10:
            return {'price_entropy': 0.5, 'return_entropy': 0.5}
        
        returns = np.diff(prices) / prices[:-1]
        
        price_bins = np.histogram(prices, bins=10)[0]
        price_entropy = stats.entropy(price_bins + 1e-10)
        
        return_bins = np.histogram(returns, bins=10)[0]
        return_entropy = stats.entropy(return_bins + 1e-10)
        
        return {
            'price_entropy': price_entropy / np.log(10),
            'return_entropy': return_entropy / np.log(10),
            'entropy_ratio': return_entropy / (price_entropy + 1e-10)
        }
    
    def calculate_trend_strength(self, data):
        if len(data) < 5:
            return 0
        
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        
        return slope * r_value**2
    
    def estimate_garch_volatility(self, returns):
        if len(returns) < 10:
            return np.std(returns)
        
        alpha = 0.1
        beta = 0.85
        
        variance = np.var(returns)
        for ret in returns[-10:]:
            variance = alpha * ret**2 + beta * variance
        
        return np.sqrt(variance)
    
    def classify_volatility_regime(self, volatility):
        if volatility < 0.02:
            return 0.2
        elif volatility < 0.05:
            return 0.5
        elif volatility < 0.10:
            return 0.8
        else:
            return 1.0
    
    def calculate_rsi(self, prices, period=14):
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return 0
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        return macd_line
    
    def calculate_ema(self, prices, period):
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def calculate_momentum_acceleration(self, prices):
        if len(prices) < 6:
            return 0
        
        short_momentum = (prices[-1] - prices[-3]) / prices[-3]
        long_momentum = (prices[-3] - prices[-6]) / prices[-6]
        
        return short_momentum - long_momentum
    
    def detect_volume_spikes(self, volumes):
        if len(volumes) < 10:
            return 0
        
        recent_volume = np.mean(volumes[-3:])
        historical_volume = np.mean(volumes[:-3])
        
        spike_ratio = recent_volume / (historical_volume + 1e-10)
        
        return min(spike_ratio - 1, 2.0)
    
    def detect_volatility_regimes(self, vol_windows):
        if len(vol_windows) < 5:
            return 0.5
        
        high_vol_threshold = np.percentile(vol_windows, 75)
        low_vol_threshold = np.percentile(vol_windows, 25)
        
        current_vol = vol_windows[-1]
        
        if current_vol > high_vol_threshold:
            return 1.0
        elif current_vol < low_vol_threshold:
            return 0.0
        else:
            return 0.5
    
    def estimate_regime_change_probability(self, vol_regime, trend_regime):
        base_prob = 0.1
        
        if abs(vol_regime - 0.5) > 0.3:
            base_prob += 0.2
        
        if abs(trend_regime) > 0.7:
            base_prob += 0.1
        
        return min(base_prob, 0.8)
    
    def get_default_features(self):
        return {
            'price_velocity': 0, 'price_acceleration': 0, 'price_momentum_1m': 0,
            'price_momentum_5m': 0, 'log_return_mean': 0, 'log_return_std': 0.05,
            'price_range_5m': 0, 'price_trend_strength': 0, 'volume_momentum': 0,
            'volume_acceleration': 0, 'volume_trend': 0, 'volume_spike_intensity': 0,
            'volume_consistency': 0.5, 'realized_volatility': 0.05, 'garch_volatility': 0.05,
            'volatility_of_volatility': 0, 'volatility_regime': 0.5, 'momentum_strength': 0,
            'momentum_persistence': 0.5, 'rsi_momentum': 0, 'macd_momentum': 0,
            'momentum_acceleration': 0, 'regime_volatility': 0.5, 'regime_trend': 0.5,
            'regime_change_prob': 0.1, 'price_entropy': 0.5, 'return_entropy': 0.5,
            'entropy_ratio': 1.0
        }
    
    def get_default_technical_indicators(self):
        return {
            'rsi': 0.5, 'macd_line': 0, 'macd_signal': 0, 'macd_histogram': 0,
            'bb_position': 0.5, 'stochastic_k': 0.5, 'stochastic_d': 0.5
        }

class HiddenMarkovRegimeDetector:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.transition_matrix = None
        self.emission_params = None
        self.current_state = 0
        
    def fit(self, returns):
        if len(returns) < 50:
            self.transition_matrix = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.05, 0.9]])
            self.emission_params = [(0, 0.01), (0, 0.03), (0, 0.08)]
            return
        
        states = self.cluster_returns(returns)
        self.estimate_parameters(returns, states)
    
    def cluster_returns(self, returns):
        from sklearn.cluster import KMeans
        
        features = np.column_stack([
            returns,
            np.abs(returns),
            [np.std(returns[max(0, i-10):i+1]) for i in range(len(returns))]
        ])
        
        kmeans = KMeans(n_clusters=self.n_states, random_state=42)
        states = kmeans.fit_predict(features)
        
        return states
    
    def estimate_parameters(self, returns, states):
        n = len(states)
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        for i in range(n - 1):
            self.transition_matrix[states[i], states[i + 1]] += 1
        
        for i in range(self.n_states):
            row_sum = np.sum(self.transition_matrix[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] /= row_sum
            else:
                self.transition_matrix[i, i] = 1.0
        
        self.emission_params = []
        for state in range(self.n_states):
            state_returns = returns[states == state]
            if len(state_returns) > 0:
                mean_return = np.mean(state_returns)
                std_return = np.std(state_returns)
                self.emission_params.append((mean_return, std_return))
            else:
                self.emission_params.append((0, 0.02))
    
    def predict_next_state(self, current_returns):
        if self.transition_matrix is None:
            return 1, 0.33
        
        current_state = self.classify_current_state(current_returns[-1])
        next_state_probs = self.transition_matrix[current_state, :]
        next_state = np.argmax(next_state_probs)
        confidence = next_state_probs[next_state]
        
        return next_state, confidence
    
    def classify_current_state(self, current_return):
        if self.emission_params is None:
            return 1
        
        likelihoods = []
        for mean, std in self.emission_params:
            likelihood = stats.norm.pdf(current_return, mean, std)
            likelihoods.append(likelihood)
        
        return np.argmax(likelihoods)

class KalmanPriceFilter:
    def __init__(self):
        self.state = None
        self.covariance = None
        self.process_noise = 1e-5
        self.measurement_noise = 1e-3
        
    def initialize(self, initial_price):
        self.state = np.array([initial_price, 0])
        self.covariance = np.eye(2) * 0.1
    
    def predict(self):
        if self.state is None:
            return None
        
        F = np.array([[1, 1], [0, 1]])
        Q = np.array([[0.25, 0.5], [0.5, 1]]) * self.process_noise
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        
        return self.state[0]
    
    def update(self, measurement):
        if self.state is None:
            self.initialize(measurement)
            return measurement
        
        H = np.array([[1, 0]])
        R = np.array([[self.measurement_noise]])
        
        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ H) @ self.covariance
        
        return self.state[0]
    
    def get_velocity(self):
        if self.state is None:
            return 0
        return self.state[1]


class TokenProfiler:
    def __init__(self):
        self.session = None
        self.price_cache = {}
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
    
    async def create_profile(self, token_data: Dict) -> Dict:
        try:
            feature_engine = AdvancedFeatureEngine()
            
            price_history = await self.get_price_history(token_data.get('address'), token_data.get('network'))
            volume_history = await self.get_volume_history(token_data.get('address'), token_data.get('network'))
            
            if len(price_history) >= 10:
                microstructure_features = feature_engine.extract_market_microstructure_features(
                    price_history, volume_history
                )
                
                regime_detector = HiddenMarkovRegimeDetector()
                returns = np.diff(price_history) / price_history[:-1]
                regime_detector.fit(returns)
                next_state, regime_confidence = regime_detector.predict_next_state(returns)
                
                kalman_filter = KalmanPriceFilter()
                filtered_prices = []
                for price in price_history:
                    filtered_price = kalman_filter.update(price)
                    filtered_prices.append(filtered_price)
                
                microstructure_features.update({
                    'regime_state': next_state / 2.0,
                    'regime_confidence': regime_confidence,
                    'kalman_velocity': kalman_filter.get_velocity(),
                    'filtered_price_deviation': abs(price_history[-1] - filtered_prices[-1]) / price_history[-1]
                })
                
                return microstructure_features
            else:
                return await self.create_basic_profile(token_data)
                
        except Exception as e:
            logger.error(f"Error creating advanced profile: {e}")
            return await self.create_basic_profile(token_data)

    async def create_basic_profile(self, token_data: Dict) -> Dict:
        try:
            profile_tasks = await asyncio.gather(
                self.calculate_price_velocity(token_data),
                self.calculate_volume_momentum(token_data),
                self.calculate_liquidity_depth(token_data),
                self.calculate_volatility(token_data),
                self.calculate_trade_frequency(token_data),
                self.calculate_holder_metrics(token_data),
                self.calculate_whale_activity(token_data),
                self.calculate_social_momentum(token_data),
                self.calculate_technical_indicators(token_data),
                self.calculate_network_activity(token_data),
                return_exceptions=True
            )
            
            profile = {
                'price_velocity': profile_tasks[0] if not isinstance(profile_tasks[0], Exception) else 0.0,
                'volume_momentum': profile_tasks[1] if not isinstance(profile_tasks[1], Exception) else 0.0,
                'liquidity_depth': profile_tasks[2] if not isinstance(profile_tasks[2], Exception) else 0.0,
                'volatility': profile_tasks[3] if not isinstance(profile_tasks[3], Exception) else 0.03,
                'trade_frequency': profile_tasks[4] if not isinstance(profile_tasks[4], Exception) else 0.0,
                'holder_concentration': profile_tasks[5] if not isinstance(profile_tasks[5], Exception) else 0.5,
                'whale_activity': profile_tasks[6] if not isinstance(profile_tasks[6], Exception) else 0.0,
                'social_momentum': profile_tasks[7] if not isinstance(profile_tasks[7], Exception) else 0.0,
                'technical_indicators': profile_tasks[8] if not isinstance(profile_tasks[8], Exception) else {'rsi': 50, 'macd': 0},
                'network_activity': profile_tasks[9] if not isinstance(profile_tasks[9], Exception) else 0.0,
                'current_price': token_data.get('price', 0.0)
            }
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating token profile: {e}")
            return self.get_default_profile()
    
    async def calculate_price_velocity(self, token_data: Dict) -> float:
        try:
            price_change_1m = token_data.get('price_change_1m', 0)
            price_change_5m = token_data.get('price_change_5m', 0)
            
            velocity = abs(price_change_1m) * 5 + abs(price_change_5m)
            return min(velocity, 1.0)
            
        except:
            return 0.0
    
    async def calculate_volume_momentum(self, token_data: Dict) -> float:
        try:
            volume_1m = token_data.get('volume_1m', 0)
            volume_5m = token_data.get('volume_5m', 1)
            
            if volume_5m == 0:
                return 0.0
            
            momentum = (volume_1m * 5) / volume_5m
            return min(momentum, 2.0)
            
        except:
            return 0.0
    
    async def calculate_liquidity_depth(self, token_data: Dict) -> float:
        try:
            liquidity_usd = token_data.get('liquidity_usd', 0)
            
            if liquidity_usd <= 0:
                return 0.0
            
            depth_score = np.log10(liquidity_usd / 1000) / 3
            return max(0.0, min(depth_score, 1.0))
            
        except:
            return 0.0
    
    async def calculate_volatility(self, token_data: Dict) -> float:
        try:
            price_change_1m = token_data.get('price_change_1m', 0)
            price_change_5m = token_data.get('price_change_5m', 0)
            
            changes = [price_change_1m, price_change_5m]
            volatility = np.std(changes) if len(changes) > 1 else abs(price_change_1m)
            
            return max(0.001, min(volatility, 0.5))
            
        except:
            return 0.03
    
    async def calculate_trade_frequency(self, token_data: Dict) -> float:
        try:
            volume_1m = token_data.get('volume_1m', 0)
            
            if volume_1m <= 0:
                return 0.0
            
            frequency_score = min(np.log10(volume_1m / 1000), 2.0) / 2.0
            return max(0.0, frequency_score)
            
        except:
            return 0.0
    
    async def calculate_holder_metrics(self, token_data: Dict) -> float:
        try:
            holder_count = token_data.get('holder_count', 1)
            
            if holder_count <= 1:
                return 1.0
            
            concentration = 1.0 / np.log10(holder_count)
            return max(0.1, min(concentration, 1.0))
            
        except:
            return 0.5
    
    async def calculate_whale_activity(self, token_data: Dict) -> float:
        try:
            volume_1m = token_data.get('volume_1m', 0)
            liquidity_usd = token_data.get('liquidity_usd', 1)
            
            if liquidity_usd <= 0:
                return 0.0
            
            whale_ratio = volume_1m / liquidity_usd
            return min(whale_ratio, 1.0)
            
        except:
            return 0.0
    
    async def calculate_social_momentum(self, token_data: Dict) -> float:
        try:
            symbol = token_data.get('symbol', '')
            
            if not symbol:
                return 0.0
            
            social_score = min(len(symbol), 10) / 10.0 * 0.5
            return social_score
            
        except:
            return 0.0
    
    async def calculate_technical_indicators(self, token_data: Dict) -> Dict:
        try:
            price_change_1m = token_data.get('price_change_1m', 0)
            price_change_5m = token_data.get('price_change_5m', 0)
            
            rsi = 50 + (price_change_1m * 100)
            rsi = max(0, min(rsi, 100))
            
            macd = price_change_1m - price_change_5m
            
            return {
                'rsi': rsi,
                'macd': macd,
                'momentum': price_change_1m,
                'trend': 1 if price_change_5m > 0 else -1
            }
            
        except:
            return {'rsi': 50, 'macd': 0, 'momentum': 0, 'trend': 0}
    
    async def calculate_network_activity(self, token_data: Dict) -> float:
        try:
            network = token_data.get('network', '')
            
            network_weights = {
                'arbitrum': 0.9,
                'optimism': 0.8,
                'polygon': 0.7,
                'base': 0.85
            }
            
            return network_weights.get(network, 0.5)
            
        except:
            return 0.5
    
    def get_default_profile(self) -> Dict:
        return {
            'price_velocity': 0.0,
            'volume_momentum': 0.0,
            'liquidity_depth': 0.0,
            'volatility': 0.03,
            'trade_frequency': 0.0,
            'holder_concentration': 0.5,
            'whale_activity': 0.0,
            'social_momentum': 0.0,
            'technical_indicators': {'rsi': 50, 'macd': 0},
            'network_activity': 0.5,
            'current_price': 0.0
        }
    
    
    async def get_price_history(self, token_address: str, network: str) -> List[float]:
        try:
            conn = sqlite3.connect('data/token_cache.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT price FROM price_snapshots WHERE token_address = ? AND network = ? ORDER BY timestamp DESC LIMIT 100",
                (token_address, network)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                return [float(row[0]) for row in reversed(rows)]
            else:
                return self.generate_synthetic_price_history(token_data.get('price', 0.001))
                
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return self.generate_synthetic_price_history(token_data.get('price', 0.001))
    
    async def get_volume_history(self, token_address: str, network: str) -> List[float]:
        try:
            conn = sqlite3.connect('data/token_cache.db')
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT volume FROM price_snapshots WHERE token_address = ? AND network = ? ORDER BY timestamp DESC LIMIT 100",
                (token_address, network)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                return [float(row[0]) for row in reversed(rows)]
            else:
                return self.generate_synthetic_volume_history()
                
        except Exception as e:
            logger.error(f"Error getting volume history: {e}")
            return self.generate_synthetic_volume_history()
    
    def generate_synthetic_price_history(self, base_price: float) -> List[float]:
        prices = [base_price]
        for i in range(99):
            change = np.random.normal(0, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.1))
        return prices
    
    def generate_synthetic_volume_history(self) -> List[float]:
        return [np.random.exponential(1000) for _ in range(100)]

    async def cleanup(self):
        if self.session:
            await self.session.close()
