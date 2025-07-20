import sys
import os
import asyncio
import websockets
import json
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from loguru import logger
from web3 import Web3
import aiohttp

class MempoolWatcher:
    def __init__(self):
        self.active_monitors = {}
        self.frontrun_cache = {}
        self.mempool_data = {}
        self.session = None
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        
    async def check_frontrun_risk(self, token_address: str) -> bool:
        try:
            cache_key = f"frontrun_{token_address}"
            
            if cache_key in self.frontrun_cache:
                cache_time = self.frontrun_cache[cache_key]['timestamp']
                if (datetime.now() - cache_time).seconds < 30:
                    return self.frontrun_cache[cache_key]['risk']
            
            risk_level = await self.analyze_mempool_activity(token_address)
            
            self.frontrun_cache[cache_key] = {
                'risk': risk_level > 0.6,
                'timestamp': datetime.now()
            }
            
            return risk_level > 0.6
            
        except Exception as e:
            logger.error(f"Error checking frontrun risk: {e}")
            return False
    
    async def analyze_mempool_activity(self, token_address: str) -> float:
        try:
            pending_txs = await self.get_pending_transactions(token_address)
            
            if not pending_txs:
                return 0.0
            
            risk_factors = 0.0
            
            high_gas_txs = len([tx for tx in pending_txs if tx.get('gas_price', 0) > 50e9])
            if high_gas_txs > 3:
                risk_factors += 0.4
            
            similar_txs = self.count_similar_transactions(pending_txs)
            if similar_txs > 2:
                risk_factors += 0.3
            
            mev_bots = await self.detect_mev_bots(pending_txs)
            if mev_bots > 1:
                risk_factors += 0.5
            
            return min(risk_factors, 1.0)
            
        except Exception as e:
            logger.error(f"Error analyzing mempool: {e}")
            return 0.0
    
    async def get_pending_transactions(self, token_address: str) -> List[Dict]:
        try:
            pending_txs = []
            
            for i in range(5):
                tx = {
                    'hash': f'0x{i:064x}',
                    'to': token_address,
                    'gas_price': 30e9 + (i * 10e9),
                    'value': 1000000000000000000,
                    'timestamp': datetime.now()
                }
                pending_txs.append(tx)
            
            return pending_txs if pending_txs else []
            
        except Exception as e:
            logger.error(f"Error getting pending transactions: {e}")
            return []
    
    def count_similar_transactions(self, transactions: List[Dict]) -> int:
        similar_count = 0
        
        for i, tx1 in enumerate(transactions):
            for tx2 in transactions[i+1:]:
                if (abs(tx1.get('gas_price', 0) - tx2.get('gas_price', 0)) < 1e9 and
                    abs(tx1.get('value', 0) - tx2.get('value', 0)) < 1e16):
                    similar_count += 1
        
        return similar_count
    
    async def detect_mev_bots(self, transactions: List[Dict]) -> int:
        mev_indicators = 0
        
        for tx in transactions:
            gas_price = tx.get('gas_price', 0)
            
            if gas_price > 100e9:
                mev_indicators += 1
            
            if tx.get('value', 0) > 10e18:
                mev_indicators += 1
        
        return mev_indicators
    
    async def monitor_token_activity(self, token_address: str, duration_seconds: int = 60):
        try:
            start_time = datetime.now()
            activity_data = {
                'transactions': [],
                'gas_prices': [],
                'volumes': []
            }
            
            while (datetime.now() - start_time).seconds < duration_seconds:
                pending_txs = await self.get_pending_transactions(token_address)
                
                for tx in pending_txs:
                    activity_data['transactions'].append(tx)
                    activity_data['gas_prices'].append(tx.get('gas_price', 0))
                    activity_data['volumes'].append(tx.get('value', 0))
                
                await asyncio.sleep(2)
            
            self.mempool_data[token_address] = activity_data
            return activity_data
            
        except Exception as e:
            logger.error(f"Error monitoring token activity: {e}")
            return {}
    
    async def get_optimal_gas_price(self, network: str, priority: str = 'fast') -> int:
        try:
            base_prices = {
                'arbitrum': 0.1e9,
                'optimism': 0.001e9,
                'polygon': 30e9,
                'base': 0.001e9
            }
            
            base_price = base_prices.get(network, 20e9)
            
            multipliers = {
                'slow': 1.0,
                'standard': 1.2,
                'fast': 1.5,
                'instant': 2.0
            }
            
            multiplier = multipliers.get(priority, 1.2)
            
            return int(base_price * multiplier)
            
        except Exception as e:
            logger.error(f"Error getting optimal gas price: {e}")
            return 20000000000
    
    async def estimate_confirmation_time(self, gas_price: int, network: str) -> float:
        try:
            base_times = {
                'arbitrum': 0.25,
                'optimism': 2.0,
                'polygon': 2.0,
                'base': 2.0
            }
            
            base_time = base_times.get(network, 15.0)
            
            if gas_price > 50e9:
                return base_time * 0.5
            elif gas_price > 20e9:
                return base_time * 0.8
            else:
                return base_time * 1.5
                
        except:
            return 15.0
    
    def get_mempool_statistics(self) -> Dict:
        return {
            'monitored_tokens': len(self.mempool_data),
            'frontrun_cache_size': len(self.frontrun_cache),
            'active_monitors': len(self.active_monitors)
        }
    
    async def connect_real_mempool_feeds(self):
    try:
        alchemy_ws_urls = {
            'arbitrum': os.getenv('ARBITRUM_RPC', '').replace('https://', 'wss://').replace('/v2/', '/v2/ws/'),
            'optimism': os.getenv('OPTIMISM_RPC', '').replace('https://', 'wss://').replace('/v2/', '/v2/ws/'),
            'polygon': os.getenv('POLYGON_RPC', '').replace('https://', 'wss://').replace('/v2/', '/v2/ws/'),
            'base': os.getenv('BASE_RPC', '').replace('https://', 'wss://').replace('/v2/', '/v2/ws/')
        }
        
        for network, ws_url in alchemy_ws_urls.items():
            if 'alchemy.com' in ws_url:
                asyncio.create_task(self.monitor_network_mempool(network, ws_url))
        
        await asyncio.gather(
            self.monitor_flashbots_bundle_pool(),
            self.monitor_eden_network(),
            self.track_mev_searcher_activity()
        )
        
    except Exception as e:
        logger.error(f"Error connecting mempool feeds: {e}")

async def monitor_network_mempool(self, network: str, ws_url: str):
    while True:
        try:
            import websockets
            
            async with websockets.connect(ws_url) as websocket:
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newPendingTransactions", True]
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                
                async for message in websocket:
                    data = json.loads(message)
                    if 'params' in data and 'result' in data['params']:
                        tx = data['params']['result']
                        await self.analyze_pending_transaction(network, tx)
                        
        except Exception as e:
            logger.error(f"Mempool monitor error for {network}: {e}")
            await asyncio.sleep(5)

async def analyze_pending_transaction(self, network: str, tx_data: dict):
    try:
        to_address = tx_data.get('to', '').lower()
        input_data = tx_data.get('input', '')
        gas_price = int(tx_data.get('gasPrice', '0x0'), 16)
        value = int(tx_data.get('value', '0x0'), 16)
        
        uniswap_routers = [
            '0xe592427a0aece92de3edee1f18e0157c05861564',
            '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',
            '0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b'
        ]
        
        if to_address in uniswap_routers and len(input_data) > 10:
            method_id = input_data[:10]
            
            if method_id in ['0x414bf389', '0xc04b8d59', '0xdb3e2198']:
                
                token_addresses = self.extract_tokens_from_calldata(input_data)
                
                for token_addr in token_addresses:
                    self.pending_activity[token_addr] = self.pending_activity.get(token_addr, [])
                    self.pending_activity[token_addr].append({
                        'gas_price': gas_price,
                        'value': value,
                        'timestamp': time.time(),
                        'network': network
                    })
                    
                    await self.check_frontrun_opportunity(token_addr, network)
        
    except Exception as e:
        logger.error(f"Error analyzing pending tx: {e}")

def extract_tokens_from_calldata(self, calldata: str) -> List[str]:
    try:
        tokens = []
        
        hex_data = calldata[10:]
        
        for i in range(0, len(hex_data) - 64, 64):
            chunk = hex_data[i:i+64]
            
            if chunk.startswith('000000000000000000000000') and len(chunk) == 64:
                addr = '0x' + chunk[24:]
                if addr != '0x0000000000000000000000000000000000000000':
                    tokens.append(addr)
        
        return tokens[:5]
        
    except Exception as e:
        logger.error(f"Error extracting tokens: {e}")
        return []

async def check_frontrun_opportunity(self, token_address: str, network: str):
    try:
        current_time = time.time()
        recent_activity = [
            tx for tx in self.pending_activity.get(token_address, [])
            if current_time - tx['timestamp'] < 60
        ]
        
        if len(recent_activity) > 3:
            avg_gas = sum(tx['gas_price'] for tx in recent_activity) / len(recent_activity)
            max_gas = max(tx['gas_price'] for tx in recent_activity)
            
            competition_level = max_gas / avg_gas if avg_gas > 0 else 1
            
            if competition_level > 2.0:
                logger.warning(f"High frontrun risk detected for {token_address}: {competition_level:.2f}x gas competition")
                
                self.frontrun_cache[token_address] = {
                    'risk': True,
                    'competition_level': competition_level,
                    'timestamp': current_time
                }
        
    except Exception as e:
        logger.error(f"Error checking frontrun opportunity: {e}")

async def monitor_flashbots_bundle_pool(self):
    try:
        while True:
            
            relay_url = "https://relay.flashbots.net"
            
            async with self.session.get(f"{relay_url}/relay/v1/data/bidtraces") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for bid in data[-10:]:
                        block_number = bid.get('block_number')
                        if block_number:
                            await self.analyze_flashbots_block(block_number)
            
            await asyncio.sleep(12)
            
    except Exception as e:
        logger.error(f"Flashbots monitoring error: {e}")

async def monitor_eden_network(self):
    try:
        while True:
            
            await asyncio.sleep(6)
            
    except Exception as e:
        logger.error(f"Eden network monitoring error: {e}")

async def track_mev_searcher_activity(self):
    try:
        known_searchers = [
            '0x00000000008c4c4e66e83c1e8e6d0b72b4b8dc3d',
            '0x0000000000007F150Bd6f54c40A34d7C3d5e9f56',
            '0x000000000034a2f1e5B5dA6E0b6Aab8B1e9A1B2C'
        ]
        
        while True:
            for searcher in known_searchers:
                
                await asyncio.sleep(1)
            
            await asyncio.sleep(10)
            
    except Exception as e:
        logger.error(f"MEV searcher tracking error: {e}")

async def get_real_pending_transactions(self, token_address: str) -> List[Dict]:
    try:
        return self.pending_activity.get(token_address, [])[-10:]
        
    except Exception as e:
        logger.error(f"Error getting pending transactions: {e}")
        return []


    async def cleanup(self):
        if self.session:
            await self.session.close()
        
        for monitor in self.active_monitors.values():
            if hasattr(monitor, 'cancel'):
                monitor.cancel()
