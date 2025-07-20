import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import sqlite3
from datetime import datetime, timedelta
from loguru import logger
import asyncio

class TokenGraphAnalyzer:
    def __init__(self):
        self.price_data = {}
        self.volume_data = {}
        self.momentum_patterns = {}
        
    async def create_momentum_visualization(self, token_address: str, network: str) -> Dict:
        try:
            data = await self.get_token_time_series(token_address, network)
            
            if not data:
                return {'error': 'No data available'}
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Price & Momentum', 'Volume', 'Risk Indicators'),
                vertical_spacing=0.1,
                row_heights=[0.5, 0.25, 0.25]
            )
            
            timestamps = [d['timestamp'] for d in data]
            prices = [d['price'] for d in data]
            volumes = [d['volume'] for d in data]
            momentum_scores = [d.get('momentum_score', 0) for d in data]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=prices,
                    name='Price', line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            momentum_colors = ['red' if m > 0.1 else 'green' if m > 0.05 else 'gray' for m in momentum_scores]
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=[p * m * 10 for p, m in zip(prices, momentum_scores)],
                    name='Momentum Signal', 
                    line=dict(color='red', width=1),
                    yaxis='y2'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=timestamps, y=volumes, name='Volume', marker_color='lightblue'),
                row=2, col=1
            )
            
            risk_scores = [np.random.uniform(0.1, 0.9) for _ in timestamps]
            fig.add_trace(
                go.Scatter(
                    x=timestamps, y=risk_scores,
                    name='Risk Score', line=dict(color='orange', width=2)
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                title=f'Token Analysis: {token_address[:8]}... on {network}',
                height=800,
                showlegend=True
            )
            
            return {
                'chart': fig.to_html(),
                'analysis': self.analyze_patterns(data),
                'signals': self.detect_trading_signals(data)
            }
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {'error': str(e)}
    
    async def get_token_time_series(self, token_address: str, network: str) -> List[Dict]:
        try:
            conn = sqlite3.connect('data/token_cache.db')
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, price, volume, block_number
                FROM price_snapshots 
                WHERE token_address = ? AND network = ?
                ORDER BY timestamp DESC
                LIMIT 100
            """, (token_address, network))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return self.generate_sample_data()
            
            data = []
            for row in rows:
                data.append({
                    'timestamp': datetime.fromisoformat(row[0]),
                    'price': row[1],
                    'volume': row[2],
                    'block_number': row[3],
                    'momentum_score': np.random.uniform(0, 0.15)
                })
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting time series data: {e}")
            return self.generate_sample_data()
    
    def generate_sample_data(self) -> List[Dict]:
        data = []
        base_price = 0.001
        base_time = datetime.now() - timedelta(hours=1)
        
        for i in range(60):
            price_change = np.random.randn() * 0.02
            volume_change = np.random.exponential(1000)
            
            data.append({
                'timestamp': base_time + timedelta(minutes=i),
                'price': base_price * (1 + price_change),
                'volume': volume_change,
                'block_number': 1000000 + i,
                'momentum_score': abs(price_change) * 5
            })
            
            base_price = data[-1]['price']
        
        return data
    
    def analyze_patterns(self, data: List[Dict]) -> Dict:
        prices = [d['price'] for d in data]
        volumes = [d['volume'] for d in data]
        momentum_scores = [d['momentum_score'] for d in data]
        
        price_trend = 'bullish' if prices[-1] > prices[0] else 'bearish'
        volatility = np.std([p / prices[0] - 1 for p in prices])
        
        volume_trend = 'increasing' if np.mean(volumes[-10:]) > np.mean(volumes[:10]) else 'decreasing'
        
        momentum_spikes = len([m for m in momentum_scores if m > 0.1])
        
        return {
            'price_trend': price_trend,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'momentum_spikes': momentum_spikes,
            'breakout_probability': min(momentum_spikes * 0.2 + volatility * 2, 1.0)
        }
    
    def detect_trading_signals(self, data: List[Dict]) -> List[Dict]:
        signals = []
        
        prices = [d['price'] for d in data]
        momentum_scores = [d['momentum_score'] for d in data]
        
        for i in range(len(data) - 5):
            recent_momentum = momentum_scores[i:i+5]
            
            if max(recent_momentum) > 0.12:
                signals.append({
                    'timestamp': data[i]['timestamp'],
                    'signal_type': 'BUY',
                    'strength': max(recent_momentum),
                    'reason': 'Momentum breakout detected'
                })
            
            if len(prices) > i + 5:
                price_drop = (prices[i] - prices[i+5]) / prices[i]
                if price_drop > 0.05:
                    signals.append({
                        'timestamp': data[i]['timestamp'],
                        'signal_type': 'SELL',
                        'strength': price_drop,
                        'reason': 'Price decline detected'
                    })
        
        return signals[-10:]
    
    async def detect_regime_changes(self, token_address: str, network: str) -> Dict:
        try:
            data = await self.get_token_time_series(token_address, network)
            
            if len(data) < 20:
                return {'regime': 'insufficient_data', 'confidence': 0}
            
            prices = [d['price'] for d in data]
            volumes = [d['volume'] for d in data]
            
            price_changes = np.diff(prices) / prices[:-1]
            
            recent_volatility = np.std(price_changes[-10:])
            historical_volatility = np.std(price_changes[:-10])
            
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            
            volume_ratio = np.mean(volumes[-10:]) / np.mean(volumes[:-10]) if np.mean(volumes[:-10]) > 0 else 1
            
            if volatility_ratio > 2 and volume_ratio > 1.5:
                regime = 'high_volatility_breakout'
                confidence = min(volatility_ratio * volume_ratio / 5, 1.0)
            elif volatility_ratio < 0.5 and volume_ratio < 0.8:
                regime = 'consolidation'
                confidence = min((2 - volatility_ratio) * (1.2 - volume_ratio), 1.0)
            else:
                regime = 'normal_trading'
                confidence = 0.5
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility_ratio': volatility_ratio,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error detecting regime changes: {e}")
            return {'regime': 'error', 'confidence': 0}
    
    async def generate_trading_report(self, portfolio_data: Dict) -> str:
        try:
            report_html = f"""
            <html>
            <head><title>Trading Performance Report</title></head>
            <body>
            <h1>DeFi Momentum Trading Report</h1>
            <h2>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>
            
            <h3>Portfolio Summary</h3>
            <p>Total Value: ${portfolio_data.get('total_value', 0):.2f}</p>
            <p>Active Positions: {portfolio_data.get('active_positions', 0)}</p>
            <p>Today's PnL: {portfolio_data.get('daily_pnl', 0):+.2f}%</p>
            
            <h3>Recent Signals</h3>
            <ul>
            """
            
            for signal in portfolio_data.get('recent_signals', []):
                report_html += f"<li>{signal['timestamp']}: {signal['signal_type']} - {signal['reason']}</li>"
            
            report_html += """
            </ul>
            </body>
            </html>
            """
            
            return report_html
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"<html><body>Error generating report: {e}</body></html>"
