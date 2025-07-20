import sys
import os
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from loguru import logger
import pickle
import asyncio
from model_trainer import MomentumModelTrainer

class FeedbackLoop:
    def __init__(self):
        self.trade_results = []
        self.model_performance = {}
        self.parameter_adjustments = []
        
    async def record_trade_result(self, signal: Dict, profit_loss: float, duration_seconds: int):
        try:
            trade_result = {
                'token_address': signal.get('token_address'),
                'network': signal.get('network'),
                'entry_confidence': signal.get('confidence'),
                'momentum_score': signal.get('momentum_score'),
                'predicted_return': signal.get('predicted_return'),
                'actual_return': profit_loss,
                'duration': duration_seconds,
                'timestamp': datetime.now(),
                'success': profit_loss > 0
            }
            
            self.trade_results.append(trade_result)
            
            await self.store_trade_feedback(trade_result)
            
            if len(self.trade_results) % 50 == 0:
                await self.analyze_model_performance()
                
        except Exception as e:
            logger.error(f"Error recording trade result: {e}")
    
    async def store_trade_feedback(self, trade_result: Dict):
        try:
            conn = sqlite3.connect('data/token_cache.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT,
                    network TEXT,
                    entry_confidence REAL,
                    momentum_score REAL,
                    predicted_return REAL,
                    actual_return REAL,
                    duration INTEGER,
                    success BOOLEAN,
                    timestamp TIMESTAMP
                )
            """)
            
            cursor.execute("""
                INSERT INTO trade_feedback 
                (token_address, network, entry_confidence, momentum_score, predicted_return, 
                 actual_return, duration, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_result['token_address'],
                trade_result['network'],
                trade_result['entry_confidence'],
                trade_result['momentum_score'],
                trade_result['predicted_return'],
                trade_result['actual_return'],
                trade_result['duration'],
                trade_result['success'],
                trade_result['timestamp'].isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trade feedback: {e}")
    
    async def analyze_model_performance(self):
        try:
            recent_trades = self.trade_results[-100:] if len(self.trade_results) >= 100 else self.trade_results
            
            if len(recent_trades) < 10:
                return
            
            accuracy = len([t for t in recent_trades if t['success']]) / len(recent_trades)
            
            predicted_returns = [t['predicted_return'] for t in recent_trades]
            actual_returns = [t['actual_return'] for t in recent_trades]
            
            correlation = np.corrcoef(predicted_returns, actual_returns)[0, 1] if len(predicted_returns) > 1 else 0
            
            mean_error = np.mean([abs(p - a) for p, a in zip(predicted_returns, actual_returns)])
            
            confidence_accuracy = {}
            for trade in recent_trades:
                conf_bucket = round(trade['entry_confidence'], 1)
                if conf_bucket not in confidence_accuracy:
                    confidence_accuracy[conf_bucket] = []
                confidence_accuracy[conf_bucket].append(trade['success'])
            
            for conf, results in confidence_accuracy.items():
                accuracy_at_conf = len([r for r in results if r]) / len(results)
                logger.info(f"Accuracy at confidence {conf}: {accuracy_at_conf:.2%}")
            
            self.model_performance = {
                'accuracy': accuracy,
                'correlation': correlation,
                'mean_error': mean_error,
                'sample_size': len(recent_trades),
                'timestamp': datetime.now()
            }
            
            logger.info(f"Model performance - Accuracy: {accuracy:.2%}, Correlation: {correlation:.3f}")
            
            if accuracy < 0.4 or correlation < 0.2:
                logger.warning("Poor model performance detected, scheduling retraining")
                await self.schedule_model_retraining()
                
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
    
    async def schedule_model_retraining(self):
        try:
            logger.info("Starting model retraining based on feedback")
            
            trainer = MomentumModelTrainer()
            
            feedback_data = await self.prepare_feedback_data()
            
            if len(feedback_data) < 50:
                logger.warning("Insufficient feedback data for retraining")
                return False
            
            success = trainer.retrain_incremental(feedback_data)
            
            if success:
                logger.info("Model retraining completed successfully")
                await self.update_model_version()
            else:
                logger.error("Model retraining failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return False
    
    async def prepare_feedback_data(self) -> pd.DataFrame:
        try:
            conn = sqlite3.connect('data/token_cache.db')
            
            query = """
            SELECT tf.*, st.current_price, st.volume_1m, st.liquidity_usd, st.volatility
            FROM trade_feedback tf
            JOIN scanned_tokens st ON tf.token_address = st.address AND tf.network = st.network
            WHERE tf.timestamp >= datetime('now', '-7 days')
            ORDER BY tf.timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing feedback data: {e}")
            return pd.DataFrame()
    
    async def update_model_version(self):
        try:
            conn = sqlite3.connect('data/token_cache.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_performance 
                (model_version, accuracy, precision, recall, trades_analyzed, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"retrained_{datetime.now().strftime('%Y%m%d_%H%M')}",
                self.model_performance.get('accuracy', 0),
                self.model_performance.get('accuracy', 0),
                self.model_performance.get('accuracy', 0),
                self.model_performance.get('sample_size', 0),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating model version: {e}")
    
    async def adjust_trading_parameters(self, current_params: Dict) -> Dict:
        try:
            if len(self.trade_results) < 20:
                return current_params
            
            recent_trades = self.trade_results[-20:]
            win_rate = len([t for t in recent_trades if t['success']]) / len(recent_trades)
            
            adjusted_params = current_params.copy()
            
            if win_rate < 0.4:
                adjusted_params['min_momentum_threshold'] *= 1.1
                adjusted_params['confidence_threshold'] *= 1.05
                logger.info("Increased thresholds due to low win rate")
            elif win_rate > 0.7:
                adjusted_params['min_momentum_threshold'] *= 0.95
                adjusted_params['confidence_threshold'] *= 0.98
                logger.info("Decreased thresholds due to high win rate")
            
            avg_duration = np.mean([t['duration'] for t in recent_trades])
            if avg_duration > 300:
                adjusted_params['momentum_decay_threshold'] *= 1.1
                logger.info("Increased decay threshold due to long hold times")
            
            return adjusted_params
            
        except Exception as e:
            logger.error(f"Error adjusting trading parameters: {e}")
            return current_params
    
    def get_performance_summary(self) -> Dict:
        if not self.trade_results:
            return {}
        
        recent_trades = self.trade_results[-50:] if len(self.trade_results) >= 50 else self.trade_results
        
        total_return = sum([t['actual_return'] for t in recent_trades])
        win_rate = len([t for t in recent_trades if t['success']]) / len(recent_trades)
        avg_return_per_trade = total_return / len(recent_trades)
        
        return {
            'total_trades': len(recent_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_return_per_trade': avg_return_per_trade,
            'model_performance': self.model_performance
        }

async def retrain_model():
    feedback_loop = FeedbackLoop()
    success = await feedback_loop.schedule_model_retraining()
    return success
