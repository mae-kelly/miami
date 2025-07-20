import sys
import os
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import yaml
from loguru import logger
import asyncio
from scipy.optimize import minimize, {"success": False, "x": [0.09, 0.03, 0.7]})()
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process.kernels import RBF, Matern


class RealtimePerformanceOptimizer:
    def __init__(self, config):
        self.config = config
        self.performance_buffer = deque(maxlen=100)
        self.optimization_frequency = 50
        self.last_optimization = 0
        self.parameter_sensitivity = {
            'momentum_threshold': 0.001,
            'confidence_threshold': 0.01,
            'slippage_tolerance': 0.001,
            'position_size': 0.01
        }
        
    def optimize_hyperparameters_realtime(self, current_performance):
        try:
            if len(self.performance_buffer) < 20:
                self.performance_buffer.append(current_performance)
                return self.config['trading']
            
            recent_performance = list(self.performance_buffer)[-20:]
            
            # Calculate performance gradients
            win_rate_trend = self.calculate_trend([p['win_rate'] for p in recent_performance])
            return_trend = self.calculate_trend([p['avg_return'] for p in recent_performance])
            execution_trend = self.calculate_trend([p['avg_execution_time'] for p in recent_performance])
            
            optimization_decisions = {}
            
            # Momentum threshold optimization
            if win_rate_trend < -0.1:  # Win rate declining
                optimization_decisions['increase_momentum_selectivity'] = True
                new_momentum_threshold = min(
                    self.config['trading']['min_momentum_threshold'] * 1.02,
                    0.15
                )
                self.config['trading']['min_momentum_threshold'] = new_momentum_threshold
            
            elif win_rate_trend > 0.1 and recent_performance[-1]['win_rate'] > 0.7:
                optimization_decisions['decrease_momentum_selectivity'] = True
                new_momentum_threshold = max(
                    self.config['trading']['min_momentum_threshold'] * 0.98,
                    0.05
                )
                self.config['trading']['min_momentum_threshold'] = new_momentum_threshold
            
            # Slippage optimization based on execution performance
            if execution_trend > 500:  # Execution getting slower
                optimization_decisions['increase_slippage_tolerance'] = True
                new_slippage = min(
                    self.config['trading']['slippage_tolerance'] * 1.05,
                    0.08
                )
                self.config['trading']['slippage_tolerance'] = new_slippage
            
            # Position sizing optimization
            if return_trend > 0.02:  # Returns improving
                optimization_decisions['increase_position_sizing'] = True
                new_position_size = min(
                    self.config['trading']['max_position_size'] * 1.02,
                    0.4
                )
                self.config['trading']['max_position_size'] = new_position_size
            
            elif return_trend < -0.02:  # Returns declining
                optimization_decisions['decrease_position_sizing'] = True
                new_position_size = max(
                    self.config['trading']['max_position_size'] * 0.98,
                    0.05
                )
                self.config['trading']['max_position_size'] = new_position_size
            
            if optimization_decisions:
                logger.info(f"🎯 REALTIME OPTIMIZATION: {optimization_decisions}")
            
            return self.config['trading']
            
        except Exception as e:
            logger.error(f"Realtime optimization error: {e}")
            return self.config['trading']
    
    def calculate_trend(self, values):
        if len(values) < 5:
            return 0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def optimize_scanning_parameters(self, scanning_performance):
        try:
            tokens_per_minute = scanning_performance.get('tokens_per_minute', 100)
            detection_rate = scanning_performance.get('momentum_detection_rate', 0.1)
            
            optimization_decisions = {}
            
            if tokens_per_minute < 150:  # Below target
                optimization_decisions['increase_batch_size'] = True
                optimization_decisions['reduce_analysis_depth'] = True
                
                self.config['scanning']['batch_size'] = min(
                    self.config['scanning']['batch_size'] * 1.2, 200
                )
                
                self.config['scanning']['scan_interval_ms'] = max(
                    self.config['scanning']['scan_interval_ms'] * 0.8, 50
                )
            
            if detection_rate < 0.05:  # Too few signals
                optimization_decisions['lower_detection_thresholds'] = True
                
            return optimization_decisions
            
        except Exception as e:
            logger.error(f"Scanning optimization error: {e}")
            return {}
    
    def adaptive_model_optimization(self, model_performance):
        try:
            accuracy = model_performance.get('accuracy', 0.5)
            latency = model_performance.get('avg_latency_ms', 100)
            
            optimizations = {}
            
            if accuracy < 0.6:
                optimizations['schedule_retraining'] = True
                optimizations['increase_feature_engineering'] = True
            
            if latency > 200:
                optimizations['reduce_model_complexity'] = True
                optimizations['increase_batch_inference'] = True
            
            if accuracy > 0.8 and latency < 50:
                optimizations['increase_model_complexity'] = True
                optimizations['add_ensemble_methods'] = True
            
            return optimizations
            
        except Exception as e:
            logger.error(f"Model optimization error: {e}")
            return {}


class DynamicOptimizer:
    def __init__(self, config: Dict):
        self.config = config
        self.trading_params = config['trading'].copy()
        self.optimization_params = config['optimization']
        self.performance_history = []
        self.parameter_history = []
        self.gp_regressor = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=1.5),
            alpha=1e-6,
            normalize_y=True
        )
        
    async def update_parameters(self, active_positions: Dict, portfolio_value: float):
        try:
            # Original dynamic optimization
            recent_performance = await self.calculate_recent_performance()
            
            if len(self.performance_history) >= 10:
                optimized_params = await self.optimize_parameters_bayesian()
                
                if optimized_params:
                    await self.apply_parameter_updates(optimized_params)
                    logger.info(f"Bayesian optimization: {optimized_params}")
            
            # Enhanced real-time optimization
            if hasattr(self, 'realtime_optimizer'):
                realtime_params = self.realtime_optimizer.optimize_hyperparameters_realtime(recent_performance)
                
                scanning_optimization = self.realtime_optimizer.optimize_scanning_parameters({
                    'tokens_per_minute': len(active_positions) * 10,
                    'momentum_detection_rate': 0.08
                })
                
                if scanning_optimization:
                    logger.info(f"Scanning optimization: {scanning_optimization}")
            
            await self.adjust_confidence_thresholds(recent_performance)
            await self.update_risk_parameters(active_positions, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error in enhanced parameter update: {e}")
    
    def initialize_realtime_optimizer(self):
        try:
            self.realtime_optimizer = RealtimePerformanceOptimizer(self.config)
            logger.info("✅ Real-time optimizer initialized")
        except Exception as e:
            logger.error(f"Real-time optimizer initialization error: {e}")

    async def update_parameters_original(self, active_positions: Dict, portfolio_value: float):
        try:
            recent_performance = await self.calculate_recent_performance()
            
            if len(self.performance_history) >= 10:
                optimized_params = await self.optimize_parameters_bayesian()
                
                if optimized_params:
                    await self.apply_parameter_updates(optimized_params)
                    logger.info(f"Parameters updated: {optimized_params}")
            
            await self.adjust_confidence_thresholds(recent_performance)
            await self.update_risk_parameters(active_positions, portfolio_value)
            
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
    
    async def calculate_recent_performance(self) -> Dict:
        conn = sqlite3.connect('data/token_cache.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT profit_loss, execution_time, confidence_score, momentum_at_entry, timestamp
            FROM trades
            WHERE timestamp >= datetime('now', '-24 hours')
            ORDER BY timestamp DESC
        """)
        
        trades = cursor.fetchall()
        conn.close()
        
        if not trades:
            return {'sharpe_ratio': 0, 'win_rate': 0, 'avg_return': 0, 'volatility': 0.05}
        
        returns = [trade[0] for trade in trades]
        execution_times = [trade[1] for trade in trades]
        
        win_rate = len([r for r in returns if r > 0]) / len(returns)
        avg_return = np.mean(returns)
        volatility = np.std(returns) if len(returns) > 1 else 0.05
        sharpe_ratio = (avg_return / volatility) if volatility > 0 else 0
        
        performance = {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'volatility': volatility,
            'avg_execution_time': np.mean(execution_times),
            'total_trades': len(trades)
        }
        
        self.performance_history.append(performance)
        return performance
    
    async def optimize_parameters_bayesian(self) -> Optional[Dict]:
        try:
            if len(self.performance_history) < 10:
                return None
            
            X = np.array([[
                p['avg_return'],
                p['volatility'],
                p['win_rate'],
                p['avg_execution_time']
            ] for p in self.performance_history[-20:]])
            
            y = np.array([p['sharpe_ratio'] for p in self.performance_history[-20:]])
            
            self.gp_regressor.fit(X, y)
            
            def objective(params):
                momentum_threshold, slippage_tolerance, confidence_threshold = params
                
                predicted_performance = self.gp_regressor.predict([[
                    0.02, 0.05, 0.6, 5.0
                ]])[0]
                
                return -predicted_performance
            
            bounds = [
                (0.05, 0.15),
                (0.01, 0.05),
                (0.4, 0.9)
            ]
            
            result = minimize(
                objective,
                x0=[0.09, 0.03, 0.7],
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                return {
                    'min_momentum_threshold': result.x[0],
                    'slippage_tolerance': result.x[1],
                    'confidence_threshold': result.x[2]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return None
    
    async def apply_parameter_updates(self, new_params: Dict):
        for param, value in new_params.items():
            if param in self.trading_params:
                old_value = self.trading_params[param]
                self.trading_params[param] = value
                logger.info(f"Updated {param}: {old_value:.4f} -> {value:.4f}")
        
        self.parameter_history.append({
            'timestamp': datetime.now(),
            'parameters': new_params.copy()
        })
    
    async def adjust_confidence_thresholds(self, performance: Dict):
        current_win_rate = performance['win_rate']
        target_win_rate = 0.65
        
        if current_win_rate < target_win_rate - 0.05:
            adjustment = 0.02
        elif current_win_rate > target_win_rate + 0.05:
            adjustment = -0.01
        else:
            adjustment = 0
        
        current_threshold = self.trading_params.get('confidence_threshold', 0.7)
        new_threshold = np.clip(current_threshold + adjustment, 0.4, 0.9)
        
        if abs(new_threshold - current_threshold) > 0.001:
            self.trading_params['confidence_threshold'] = new_threshold
            logger.info(f"Confidence threshold adjusted: {current_threshold:.3f} -> {new_threshold:.3f}")
    
    async def update_risk_parameters(self, active_positions: Dict, portfolio_value: float):
        position_count = len(active_positions)
        max_positions = self.config['risk_management']['max_concurrent_positions']
        
        current_exposure = sum(pos['initial_value'] for pos in active_positions.values())
        exposure_ratio = current_exposure / portfolio_value if portfolio_value > 0 else 0
        
        if exposure_ratio > 0.8:
            self.trading_params['max_position_size'] *= 0.9
        elif exposure_ratio < 0.3 and position_count < max_positions:
            self.trading_params['max_position_size'] *= 1.05
        
        self.trading_params['max_position_size'] = np.clip(
            self.trading_params['max_position_size'], 0.1, 0.5
        )
    
    def calculate_entropy_score(self, price_data: List[float]) -> float:
        if len(price_data) < 2:
            return 0.5
        
        returns = np.diff(price_data) / price_data[:-1]
        
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        
        entropy = -np.sum(hist * np.log(hist))
        normalized_entropy = entropy / np.log(len(hist)) if len(hist) > 1 else 0.5
        
        return np.clip(normalized_entropy, 0, 1)
    
    def get_current_parameters(self) -> Dict:
        return self.trading_params.copy()
    
    
    async def optimize_gas_strategy(self, network_conditions):
        try:
            gas_optimizations = {}
            
            for network, conditions in network_conditions.items():
                congestion = conditions.get('congestion_level', 0.5)
                avg_gas_price = conditions.get('avg_gas_price', 20e9)
                
                if congestion > 0.8:
                    gas_optimizations[network] = {
                        'increase_gas_price': True,
                        'reduce_transaction_frequency': True,
                        'batch_transactions': True
                    }
                elif congestion < 0.3:
                    gas_optimizations[network] = {
                        'reduce_gas_price': True,
                        'increase_transaction_frequency': True
                    }
            
            return gas_optimizations
            
        except Exception as e:
            logger.error(f"Gas strategy optimization error: {e}")
            return {}
    
    async def optimize_liquidity_timing(self, market_conditions):
        try:
            timing_optimizations = {}
            
            volatility = market_conditions.get('market_volatility', 0.05)
            volume = market_conditions.get('total_volume', 1000000)
            
            if volatility > 0.1:
                timing_optimizations['delay_large_trades'] = True
                timing_optimizations['increase_slippage_tolerance'] = True
                timing_optimizations['reduce_position_sizes'] = True
            
            elif volatility < 0.02:
                timing_optimizations['increase_trade_frequency'] = True
                timing_optimizations['tighten_slippage_tolerance'] = True
            
            if volume < 500000:
                timing_optimizations['avoid_large_trades'] = True
                timing_optimizations['wait_for_volume_increase'] = True
            
            return timing_optimizations
            
        except Exception as e:
            logger.error(f"Liquidity timing optimization error: {e}")
            return {}
    
    def calculate_optimization_effectiveness(self):
        try:
            if len(self.performance_history) < 10:
                return 0.5
            
            recent_performance = self.performance_history[-10:]
            baseline_performance = self.performance_history[-20:-10] if len(self.performance_history) >= 20 else self.performance_history[:10]
            
            recent_avg_return = np.mean([p['avg_return'] for p in recent_performance])
            baseline_avg_return = np.mean([p['avg_return'] for p in baseline_performance])
            
            recent_win_rate = np.mean([p['win_rate'] for p in recent_performance])
            baseline_win_rate = np.mean([p['win_rate'] for p in baseline_performance])
            
            return_improvement = (recent_avg_return - baseline_avg_return) / abs(baseline_avg_return) if baseline_avg_return != 0 else 0
            win_rate_improvement = recent_win_rate - baseline_win_rate
            
            overall_effectiveness = (return_improvement + win_rate_improvement) / 2
            return max(-1.0, min(1.0, overall_effectiveness))
            
        except Exception as e:
            logger.error(f"Effectiveness calculation error: {e}")
            return 0.0

    def get_optimization_stats(self) -> Dict:
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        return {
            'avg_sharpe_ratio': np.mean([p['sharpe_ratio'] for p in recent_performance]),
            'avg_win_rate': np.mean([p['win_rate'] for p in recent_performance]),
            'parameter_updates': len(self.parameter_history),
            'last_optimization': self.parameter_history[-1]['timestamp'].isoformat() if self.parameter_history else None
        }
