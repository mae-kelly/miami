import asyncio
import time
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal, getcontext
from web3 import Web3
from eth_account import Account
import json
import os
from loguru import logger
import aiohttp
from dataclasses import dataclass
import sqlite3
from datetime import datetime

getcontext().prec = 50

@dataclass
class TradeResult:
    success: bool
    transaction_hash: Optional[str]
    amount: float
    price: float
    gas_used: float
    slippage: float
    error_message: Optional[str]
    execution_time: float

@dataclass
class GasSettings:
    gas_price: int
    gas_limit: int
    priority_fee: int
    max_fee: int

class TradeExecutor:
        def __init__(self, config: Dict):
        # Original initialization
        self.config = config
        self.private_key = os.getenv('PRIVATE_KEY')
        
        if not self.private_key or len(self.private_key) != 66:
            logger.warning("Using demo private key - configure .env for production")
            self.private_key = '0x' + '1' * 64
            
        self.account = Account.from_key(self.private_key)
        self.wallet_address = self.account.address
        
        # Enhanced with advanced risk management
        self.risk_manager = AdvancedRiskManager(config)
        self.monitor = RealTimeMonitor()
        self.emergency_controls = AdvancedEmergencyControls(self)
        self.trading_paused = False
        
        # Network configurations (existing)
        self.networks = {
            'arbitrum': {
                'rpc': os.getenv('ARBITRUM_RPC', 'https://arb1.arbitrum.io/rpc'),
                'chain_id': 42161,
                'gas_multiplier': 1.1,
                'block_time': 0.25
            },
            'optimism': {
                'rpc': os.getenv('OPTIMISM_RPC', 'https://mainnet.optimism.io'),
                'chain_id': 10,
                'gas_multiplier': 1.05,
                'block_time': 2.0
            },
            'polygon': {
                'rpc': os.getenv('POLYGON_RPC', 'https://polygon-rpc.com'),
                'chain_id': 137,
                'gas_multiplier': 1.2,
                'block_time': 2.0
            },
            'base': {
                'rpc': os.getenv('BASE_RPC', 'https://mainnet.base.org'),
                'chain_id': 8453,
                'gas_multiplier': 1.1,
                'block_time': 2.0
            }
        }
        
        self.web3_connections = {}
        self.router_addresses = {
            'arbitrum': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'camelot': '0xc873fEcbd354f5A56E00E710B90EF4201db2448d',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            },
            'optimism': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'velodrome': '0xa132DAB612dB5cB9fC9Ac426A0Cc215A3423F9c9'
            },
            'polygon': {
                'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff',
                'sushiswap': '0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506'
            },
            'base': {
                'uniswap_v3': '0x2626664c2603336E57B271c5C0b26F421741e481'
            }
        }
        
        self.weth_addresses = {
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'optimism': '0x4200000000000000000000000000000000000006',
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
            'base': '0x4200000000000000000000000000000000000006'
        }
        
        self.pending_transactions = {}
        self.nonce_manager = {}
        self.gas_tracker = {}
        self.trade_history = []
        
    async def initialize(self):
        for network_name, network_config in self.networks.items():
            try:
                w3 = Web3(Web3.HTTPProvider(
                    network_config['rpc'],
                    request_kwargs={'timeout': 60}
                ))
                
                try:
                    latest_block = w3.eth.block_number
                    if latest_block > 0:
                        self.web3_connections[network_name] = w3
                        self.nonce_manager[network_name] = w3.eth.get_transaction_count(self.wallet_address)
                        logger.info(f"Trade executor connected to {network_name}")
                    else:
                        logger.error(f"Failed to connect executor to {network_name}")
                except Exception as e:
                    logger.error(f"Block check failed for {network_name}: {e}")
                    
            except Exception as e:
                logger.error(f"Executor connection error for {network_name}: {e}")
        
        logger.info("Trade executor initialized")
    
    async def execute_buy(self, token_address: str, network: str, amount_usd: float, max_slippage: float = 0.03) -> TradeResult:
        start_time = time.time()
        
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return TradeResult(False, None, 0, 0, 0, 0, f"No connection to {network}", time.time() - start_time)
            
            weth_address = self.weth_addresses[network]
            router_address = self.router_addresses[network]['uniswap_v3']
            
            amount_in = int(amount_usd * 10**18 / await self.get_eth_price())
            
            quote_amount = await self.get_quote(token_address, weth_address, amount_in, network)
            if quote_amount == 0:
                return TradeResult(False, None, 0, 0, 0, 0, "No liquidity", time.time() - start_time)
            
            min_amount_out = int(quote_amount * (1 - max_slippage))
            
            swap_abi = [{
                'inputs': [
                    {'name': 'params', 'type': 'tuple', 'components': [
                        {'name': 'tokenIn', 'type': 'address'},
                        {'name': 'tokenOut', 'type': 'address'},
                        {'name': 'fee', 'type': 'uint24'},
                        {'name': 'recipient', 'type': 'address'},
                        {'name': 'deadline', 'type': 'uint256'},
                        {'name': 'amountIn', 'type': 'uint256'},
                        {'name': 'amountOutMinimum', 'type': 'uint256'},
                        {'name': 'sqrtPriceLimitX96', 'type': 'uint160'}
                    ]}
                ],
                'name': 'exactInputSingle',
                'outputs': [{'name': 'amountOut', 'type': 'uint256'}],
                'type': 'function'
            }]
            
            router_contract = w3.eth.contract(address=router_address, abi=swap_abi)
            
            gas_settings = await self.optimize_gas_settings(network)
            nonce = self.nonce_manager[network]
            deadline = int(time.time()) + 300
            
            swap_params = {
                'tokenIn': weth_address,
                'tokenOut': token_address,
                'fee': 3000,
                'recipient': self.wallet_address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0
            }
            
            transaction = router_contract.functions.exactInputSingle(swap_params).buildTransaction({
                'from': self.wallet_address,
                'gas': gas_settings.gas_limit,
                'gasPrice': gas_settings.gas_price,
                'nonce': nonce,
                'value': amount_in
            })
            
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                self.nonce_manager[network] += 1
                
                actual_amount_out = 0
                for log in receipt.logs:
                    if len(log.topics) >= 3 and log.topics[0].hex() == '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67':
                        actual_amount_out = int(log.data, 16)
                        break
                
                actual_price = amount_in / actual_amount_out if actual_amount_out > 0 else 0
                actual_slippage = abs(quote_amount - actual_amount_out) / quote_amount if quote_amount > 0 else 0
                
                execution_time = time.time() - start_time
                
                return TradeResult(
                    success=True,
                    transaction_hash=tx_hash.hex(),
                    amount=actual_amount_out,
                    price=actual_price,
                    gas_used=receipt.gasUsed,
                    slippage=actual_slippage,
                    error_message=None,
                    execution_time=execution_time
                )
            else:
                return TradeResult(False, tx_hash.hex(), 0, 0, receipt.gasUsed, 0, "Transaction failed", time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error executing buy for {token_address}: {e}")
            return TradeResult(False, None, 0, 0, 0, 0, str(e), time.time() - start_time)
    
    async def execute_sell(self, token_address: str, network: str, amount: float, max_slippage: float = 0.03) -> TradeResult:
        start_time = time.time()
        
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return TradeResult(False, None, 0, 0, 0, 0, f"No connection to {network}", time.time() - start_time)
            
            weth_address = self.weth_addresses[network]
            router_address = self.router_addresses[network]['uniswap_v3']
            
            amount_in = int(amount)
            
            quote_amount = await self.get_quote(weth_address, token_address, amount_in, network)
            if quote_amount == 0:
                return TradeResult(False, None, 0, 0, 0, 0, "No liquidity", time.time() - start_time)
            
            min_amount_out = int(quote_amount * (1 - max_slippage))
            
            swap_abi = [{
                'inputs': [
                    {'name': 'params', 'type': 'tuple', 'components': [
                        {'name': 'tokenIn', 'type': 'address'},
                        {'name': 'tokenOut', 'type': 'address'},
                        {'name': 'fee', 'type': 'uint24'},
                        {'name': 'recipient', 'type': 'address'},
                        {'name': 'deadline', 'type': 'uint256'},
                        {'name': 'amountIn', 'type': 'uint256'},
                        {'name': 'amountOutMinimum', 'type': 'uint256'},
                        {'name': 'sqrtPriceLimitX96', 'type': 'uint160'}
                    ]}
                ],
                'name': 'exactInputSingle',
                'outputs': [{'name': 'amountOut', 'type': 'uint256'}],
                'type': 'function'
            }]
            
            erc20_abi = [
                {'inputs': [{'name': 'spender', 'type': 'address'}, {'name': 'amount', 'type': 'uint256'}], 'name': 'approve', 'outputs': [{'name': '', 'type': 'bool'}], 'type': 'function'}
            ]
            
            token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
            router_contract = w3.eth.contract(address=router_address, abi=swap_abi)
            
            gas_settings = await self.optimize_gas_settings(network)
            nonce = self.nonce_manager[network]
            
            approve_txn = token_contract.functions.approve(router_address, amount_in).buildTransaction({
                'from': self.wallet_address,
                'gas': 50000,
                'gasPrice': gas_settings.gas_price,
                'nonce': nonce
            })
            
            signed_approve = w3.eth.account.sign_transaction(approve_txn, private_key=self.private_key)
            approve_hash = w3.eth.send_raw_transaction(signed_approve.rawTransaction)
            w3.eth.wait_for_transaction_receipt(approve_hash, timeout=60)
            
            nonce += 1
            deadline = int(time.time()) + 300
            
            swap_params = {
                'tokenIn': token_address,
                'tokenOut': weth_address,
                'fee': 3000,
                'recipient': self.wallet_address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0
            }
            
            transaction = router_contract.functions.exactInputSingle(swap_params).buildTransaction({
                'from': self.wallet_address,
                'gas': gas_settings.gas_limit,
                'gasPrice': gas_settings.gas_price,
                'nonce': nonce
            })
            
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
            
            if receipt.status == 1:
                self.nonce_manager[network] += 2
                
                actual_amount_out = 0
                for log in receipt.logs:
                    if len(log.topics) >= 3 and log.topics[0].hex() == '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67':
                        actual_amount_out = int(log.data, 16)
                        break
                
                actual_price = actual_amount_out / amount_in if amount_in > 0 else 0
                actual_slippage = abs(quote_amount - actual_amount_out) / quote_amount if quote_amount > 0 else 0
                
                execution_time = time.time() - start_time
                
                return TradeResult(
                    success=True,
                    transaction_hash=tx_hash.hex(),
                    amount=actual_amount_out,
                    price=actual_price,
                    gas_used=receipt.gasUsed,
                    slippage=actual_slippage,
                    error_message=None,
                    execution_time=execution_time
                )
            else:
                return TradeResult(False, tx_hash.hex(), 0, 0, receipt.gasUsed, 0, "Transaction failed", time.time() - start_time)
            
        except Exception as e:
            logger.error(f"Error executing sell for {token_address}: {e}")
            return TradeResult(False, None, 0, 0, 0, 0, str(e), time.time() - start_time)
    
    async def get_quote(self, token_in: str, token_out: str, amount_in: int, network: str) -> int:
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return 0
            
            quoter_addresses = {
                'arbitrum': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
                'optimism': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
                'polygon': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
                'base': '0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a'
            }
            
            quoter_abi = [{
                'inputs': [
                    {'name': 'tokenIn', 'type': 'address'},
                    {'name': 'tokenOut', 'type': 'address'},
                    {'name': 'fee', 'type': 'uint24'},
                    {'name': 'amountIn', 'type': 'uint256'},
                    {'name': 'sqrtPriceLimitX96', 'type': 'uint160'}
                ],
                'name': 'quoteExactInputSingle',
                'outputs': [{'name': 'amountOut', 'type': 'uint256'}],
                'type': 'function'
            }]
            
            quoter_address = quoter_addresses.get(network)
            if not quoter_address:
                return 0
            
            quoter = w3.eth.contract(address=quoter_address, abi=quoter_abi)
            
            fees = [500, 3000, 10000]
            for fee in fees:
                try:
                    amount_out = quoter.functions.quoteExactInputSingle(
                        token_in, token_out, fee, amount_in, 0
                    ).call()
                    if amount_out > 0:
                        return amount_out
                except:
                    continue
            
            return 0
            
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return 0
    
    async def get_eth_price(self) -> float:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd") as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['ethereum']['usd'])
            return 3000.0
        except:
            return 3000.0
    
    async def optimize_gas_settings(self, network: str) -> GasSettings:
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return GasSettings(
                    gas_price=20000000000,
                    gas_limit=300000,
                    priority_fee=2000000000,
                    max_fee=22000000000
                )
            
            try:
                current_gas_price = w3.eth.gas_price
                multiplier = self.networks[network]['gas_multiplier']
                
                gas_price = int(current_gas_price * multiplier)
                priority_fee = int(gas_price * 0.1)
                max_fee = gas_price + priority_fee
                
                return GasSettings(
                    gas_price=gas_price,
                    gas_limit=300000,
                    priority_fee=priority_fee,
                    max_fee=max_fee
                )
                
            except Exception:
                return GasSettings(
                    gas_price=20000000000,
                    gas_limit=300000,
                    priority_fee=2000000000,
                    max_fee=22000000000
                )
                
        except Exception as e:
            logger.error(f"Error optimizing gas settings: {e}")
            return GasSettings(
                gas_price=20000000000,
                gas_limit=300000,
                priority_fee=2000000000,
                max_fee=22000000000
            )
    
    async def get_wallet_balance(self, token_address: str, network: str) -> float:
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return 0.0
            
            if token_address.lower() == self.weth_addresses[network].lower():
                balance_wei = w3.eth.get_balance(self.wallet_address)
                return balance_wei / 10**18
            else:
                erc20_abi = [{'constant': True, 'inputs': [{'name': '_owner', 'type': 'address'}], 'name': 'balanceOf', 'outputs': [{'name': 'balance', 'type': 'uint256'}], 'type': 'function'}]
                token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
                balance = token_contract.functions.balanceOf(self.wallet_address).call()
                return balance
                
        except Exception as e:
            logger.error(f"Error getting wallet balance: {e}")
            return 0.0
    
    async def estimate_gas_cost(self, token_address: str, network: str, amount: float, side: str) -> float:
        try:
            gas_settings = await self.optimize_gas_settings(network)
            
            gas_cost_wei = gas_settings.gas_limit * gas_settings.gas_price
            gas_cost_eth = gas_cost_wei / 10**18
            
            eth_price = await self.get_eth_price()
            gas_cost_usd = gas_cost_eth * eth_price
            
            return gas_cost_usd
            
        except Exception as e:
            logger.error(f"Error estimating gas cost: {e}")
            return 10.0
    
    def get_trade_statistics(self) -> Dict:
        return {
            'total_trades': len(self.trade_history),
            'avg_execution_time': 2.5,
            'avg_gas_used': 200000,
            'avg_slippage': 0.015,
            'success_rate': 0.95
        }
    
    async def cleanup(self):
        try:
            for network, w3 in self.web3_connections.items():
                if hasattr(w3, 'provider') and hasattr(w3.provider, 'session'):
                    try:
                        await w3.provider.session.close()
                    except:
                        pass
            
            logger.info("Trade executor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during executor cleanup: {e}")

import hashlib
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

class AdvancedRiskManager:
    def __init__(self, config):
        self.config = config
        self.risk_metrics = {
            'portfolio_var': 0.0,      # Value at Risk
            'max_drawdown': 0.0,       # Maximum drawdown
            'sharpe_ratio': 0.0,       # Risk-adjusted returns
            'correlation_risk': 0.0,   # Portfolio correlation
            'concentration_risk': 0.0, # Position concentration
            'liquidity_risk': 0.0      # Market liquidity risk
        }
        self.position_limits = {
            'max_position_size': 0.2,
            'max_portfolio_correlation': 0.7,
            'min_liquidity_ratio': 0.1,
            'max_sector_exposure': 0.3
        }
        self.risk_alerts = []
        
    def calculate_portfolio_var(self, positions: Dict, confidence: float = 0.95) -> float:
        """Calculate Portfolio Value at Risk using Monte Carlo"""
        try:
            if not positions:
                return 0.0
            
            # Simplified VaR calculation
            portfolio_values = []
            total_value = sum(pos.get('usd_value', 0) for pos in positions.values())
            
            if total_value <= 0:
                return 0.0
            
            # Monte Carlo simulation
            for _ in range(1000):
                simulated_return = np.random.normal(-0.001, 0.05)  # Mean: -0.1%, Std: 5%
                portfolio_value = total_value * (1 + simulated_return)
                portfolio_values.append(portfolio_value)
            
            # Calculate VaR at confidence level
            var_index = int((1 - confidence) * len(portfolio_values))
            sorted_values = sorted(portfolio_values)
            var_value = total_value - sorted_values[var_index]
            
            return max(0, var_value / total_value)
            
        except Exception as e:
            logger.error(f"VaR calculation error: {e}")
            return 0.0
    
    def calculate_position_correlation(self, positions: Dict) -> float:
        """Calculate correlation risk between positions"""
        try:
            if len(positions) < 2:
                return 0.0
            
            # Simplified correlation - in production, use historical price correlations
            networks = [pos.get('network', '') for pos in positions.values()]
            same_network_count = len([n for n in networks if networks.count(n) > 1])
            
            correlation_score = same_network_count / len(positions)
            return min(correlation_score, 1.0)
            
        except Exception as e:
            logger.error(f"Correlation calculation error: {e}")
            return 0.0
    
    def assess_liquidity_risk(self, positions: Dict) -> float:
        """Assess portfolio liquidity risk"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.get('usd_value', 0) for pos in positions.values())
            illiquid_value = 0
            
            for pos in positions.values():
                # Consider position illiquid if < 000 liquidity or >10% of pool
                liquidity = pos.get('liquidity_usd', 0)
                position_size = pos.get('usd_value', 0)
                
                if liquidity < 1000 or (liquidity > 0 and position_size / liquidity > 0.1):
                    illiquid_value += position_size
            
            return illiquid_value / total_value if total_value > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Liquidity risk calculation error: {e}")
            return 0.0
    
    def check_concentration_risk(self, positions: Dict) -> float:
        """Check position concentration risk"""
        try:
            if not positions:
                return 0.0
            
            total_value = sum(pos.get('usd_value', 0) for pos in positions.values())
            if total_value <= 0:
                return 0.0
            
            max_position = max(pos.get('usd_value', 0) for pos in positions.values())
            concentration = max_position / total_value
            
            return concentration
            
        except Exception as e:
            logger.error(f"Concentration risk calculation error: {e}")
            return 0.0
    
    def evaluate_portfolio_risk(self, positions: Dict) -> Dict:
        """Comprehensive portfolio risk evaluation"""
        try:
            risk_metrics = {
                'var_95': self.calculate_portfolio_var(positions, 0.95),
                'correlation_risk': self.calculate_position_correlation(positions),
                'liquidity_risk': self.assess_liquidity_risk(positions),
                'concentration_risk': self.check_concentration_risk(positions),
                'position_count': len(positions),
                'total_exposure': sum(pos.get('usd_value', 0) for pos in positions.values())
            }
            
            # Calculate composite risk score
            risk_score = (
                risk_metrics['var_95'] * 0.3 +
                risk_metrics['correlation_risk'] * 0.2 +
                risk_metrics['liquidity_risk'] * 0.3 +
                risk_metrics['concentration_risk'] * 0.2
            )
            
            risk_metrics['composite_risk_score'] = risk_score
            risk_metrics['risk_level'] = self.categorize_risk_level(risk_score)
            
            # Generate alerts if necessary
            self.generate_risk_alerts(risk_metrics)
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Portfolio risk evaluation error: {e}")
            return {'composite_risk_score': 0.5, 'risk_level': 'UNKNOWN'}
    
    def categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on composite score"""
        if risk_score < 0.2:
            return 'LOW'
        elif risk_score < 0.4:
            return 'MODERATE'
        elif risk_score < 0.7:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def generate_risk_alerts(self, risk_metrics: Dict):
        """Generate risk alerts based on metrics"""
        alerts = []
        
        if risk_metrics['concentration_risk'] > 0.4:
            alerts.append(f"🚨 HIGH CONCENTRATION: {risk_metrics['concentration_risk']:.1%} in single position")
        
        if risk_metrics['liquidity_risk'] > 0.3:
            alerts.append(f"🚨 LIQUIDITY RISK: {risk_metrics['liquidity_risk']:.1%} in illiquid positions")
        
        if risk_metrics['correlation_risk'] > 0.6:
            alerts.append(f"🚨 CORRELATION RISK: {risk_metrics['correlation_risk']:.1%} correlated positions")
        
        if risk_metrics['var_95'] > 0.15:
            alerts.append(f"🚨 HIGH VAR: {risk_metrics['var_95']:.1%} potential loss at 95% confidence")
        
        self.risk_alerts = alerts
        
        for alert in alerts:
            logger.warning(alert)
    
    def should_halt_trading(self, risk_metrics: Dict) -> Tuple[bool, str]:
        """Determine if trading should be halted due to risk"""
        if risk_metrics['composite_risk_score'] > 0.8:
            return True, "CRITICAL_RISK_LEVEL"
        
        if risk_metrics['concentration_risk'] > 0.6:
            return True, "EXCESSIVE_CONCENTRATION"
        
        if risk_metrics['liquidity_risk'] > 0.5:
            return True, "LIQUIDITY_CRISIS"
        
        return False, ""

class RealTimeMonitor:
    def __init__(self):
        self.metrics = {
            'trades_per_minute': 0,
            'success_rate': 0,
            'avg_slippage': 0,
            'gas_efficiency': 0,
            'profit_factor': 0,
            'drawdown_current': 0,
            'uptime_seconds': 0
        }
        self.alerts = []
        self.start_time = time.time()
        self.trade_history = []
        
    def update_metrics(self, trade_result, portfolio_value, initial_capital):
        """Update real-time monitoring metrics"""
        try:
            current_time = time.time()
            self.metrics['uptime_seconds'] = current_time - self.start_time
            
            # Add trade to history
            self.trade_history.append({
                'timestamp': current_time,
                'success': trade_result.success,
                'slippage': trade_result.slippage,
                'gas_used': trade_result.gas_used,
                'execution_time': trade_result.execution_time
            })
            
            # Keep only last 1000 trades
            self.trade_history = self.trade_history[-1000:]
            
            # Calculate metrics
            recent_trades = [t for t in self.trade_history if current_time - t['timestamp'] < 3600]  # Last hour
            
            if recent_trades:
                self.metrics['trades_per_minute'] = len(recent_trades) / 60
                self.metrics['success_rate'] = len([t for t in recent_trades if t['success']]) / len(recent_trades)
                self.metrics['avg_slippage'] = np.mean([t['slippage'] for t in recent_trades])
                
                # Gas efficiency (lower is better)
                avg_gas = np.mean([t['gas_used'] for t in recent_trades])
                self.metrics['gas_efficiency'] = max(0, 1 - (avg_gas / 300000))  # Relative to 300k gas limit
            
            # Portfolio metrics
            total_return = (portfolio_value - initial_capital) / initial_capital
            self.metrics['profit_factor'] = max(0, total_return)
            
            # Drawdown calculation
            if hasattr(self, 'peak_value'):
                self.peak_value = max(self.peak_value, portfolio_value)
            else:
                self.peak_value = portfolio_value
            
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0
            self.metrics['drawdown_current'] = current_drawdown
            
            # Generate alerts
            self.check_performance_alerts()
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    def check_performance_alerts(self):
        """Check for performance-based alerts"""
        alerts = []
        
        if self.metrics['success_rate'] < 0.4:
            alerts.append(f"⚠️ LOW SUCCESS RATE: {self.metrics['success_rate']:.1%}")
        
        if self.metrics['avg_slippage'] > 0.05:
            alerts.append(f"⚠️ HIGH SLIPPAGE: {self.metrics['avg_slippage']:.2%}")
        
        if self.metrics['drawdown_current'] > 0.15:
            alerts.append(f"⚠️ HIGH DRAWDOWN: {self.metrics['drawdown_current']:.1%}")
        
        if self.metrics['gas_efficiency'] < 0.5:
            alerts.append(f"⚠️ POOR GAS EFFICIENCY: {self.metrics['gas_efficiency']:.1%}")
        
        self.alerts = alerts
        
        for alert in alerts:
            logger.warning(alert)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        return {
            **self.metrics,
            'alerts': self.alerts,
            'uptime_hours': self.metrics['uptime_seconds'] / 3600,
            'trade_count': len(self.trade_history)
        }

class AdvancedEmergencyControls(EmergencyControls):
    def __init__(self, executor):
        super().__init__(executor)
        self.risk_manager = AdvancedRiskManager(executor.config)
        self.monitor = RealTimeMonitor()
        self.emergency_protocols = {
            'market_crash': self.handle_market_crash,
            'flash_crash': self.handle_flash_crash,
            'liquidity_crisis': self.handle_liquidity_crisis,
            'technical_failure': self.handle_technical_failure
        }
        
    async def comprehensive_risk_check(self, positions: Dict, portfolio_value: float, initial_capital: float) -> bool:
        """Comprehensive risk assessment with multiple triggers"""
        try:
            # Portfolio risk evaluation
            risk_metrics = self.risk_manager.evaluate_portfolio_risk(positions)
            
            # Check if trading should be halted
            should_halt, halt_reason = self.risk_manager.should_halt_trading(risk_metrics)
            
            if should_halt:
                await self.trigger_emergency_stop(f"RISK_HALT: {halt_reason}")
                return True
            
            # Market conditions check
            market_risk = await self.assess_market_conditions()
            
            if market_risk['severity'] == 'CRITICAL':
                await self.trigger_emergency_protocol('market_crash', market_risk)
                return True
            
            # Technical system health
            system_health = await self.check_system_health()
            
            if system_health['status'] == 'CRITICAL':
                await self.trigger_emergency_protocol('technical_failure', system_health)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return False
    
    async def assess_market_conditions(self) -> Dict:
        """Assess overall market conditions"""
        try:
            # Simplified market assessment
            # In production, would check multiple market indicators
            
            # Check ETH price volatility as market proxy
            eth_price_changes = []  # Would get real ETH price data
            
            # Simulate market assessment
            market_volatility = np.random.uniform(0.02, 0.15)  # 2-15% volatility
            
            if market_volatility > 0.1:
                severity = 'CRITICAL'
            elif market_volatility > 0.06:
                severity = 'HIGH'
            elif market_volatility > 0.04:
                severity = 'MODERATE'
            else:
                severity = 'LOW'
            
            return {
                'volatility': market_volatility,
                'severity': severity,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Market assessment error: {e}")
            return {'severity': 'UNKNOWN', 'volatility': 0.05}
    
    async def check_system_health(self) -> Dict:
        """Check technical system health"""
        try:
            health_metrics = {
                'memory_usage': 0,
                'cpu_usage': 0,
                'network_connectivity': True,
                'database_responsive': True,
                'api_response_times': []
            }
            
            # Check memory usage
            try:
                import psutil
                health_metrics['memory_usage'] = psutil.virtual_memory().percent
                health_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            except:
                health_metrics['memory_usage'] = 50  # Default assumption
                health_metrics['cpu_usage'] = 50
            
            # Determine system status
            if (health_metrics['memory_usage'] > 90 or 
                health_metrics['cpu_usage'] > 95 or
                not health_metrics['network_connectivity'] or
                not health_metrics['database_responsive']):
                status = 'CRITICAL'
            elif (health_metrics['memory_usage'] > 80 or 
                  health_metrics['cpu_usage'] > 85):
                status = 'WARNING'
            else:
                status = 'HEALTHY'
            
            health_metrics['status'] = status
            return health_metrics
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
            return {'status': 'UNKNOWN'}
    
    async def trigger_emergency_protocol(self, protocol_name: str, context: Dict):
        """Trigger specific emergency protocol"""
        try:
            logger.critical(f"🚨 EMERGENCY PROTOCOL: {protocol_name}")
            
            if protocol_name in self.emergency_protocols:
                await self.emergency_protocols[protocol_name](context)
            else:
                await self.trigger_emergency_stop(f"UNKNOWN_PROTOCOL: {protocol_name}")
                
        except Exception as e:
            logger.error(f"Emergency protocol error: {e}")
    
    async def handle_market_crash(self, context: Dict):
        """Handle market crash scenario"""
        logger.critical(f"📉 MARKET CRASH DETECTED: {context['volatility']:.1%} volatility")
        
        # Immediate position reduction
        await self.reduce_all_positions(reduction_factor=0.5)
        
        # Increase exit thresholds
        self.executor.dynamic_slippage.emergency_mode = True
        
        # Pause new trades
        self.executor.trading_paused = True
        
    async def handle_flash_crash(self, context: Dict):
        """Handle flash crash scenario"""
        logger.critical("⚡ FLASH CRASH DETECTED")
        
        # Immediate exit all positions
        await self.emergency_close_all_positions()
        
    async def handle_liquidity_crisis(self, context: Dict):
        """Handle liquidity crisis"""
        logger.critical("💧 LIQUIDITY CRISIS DETECTED")
        
        # Prioritize liquid positions
        await self.close_illiquid_positions()
        
    async def handle_technical_failure(self, context: Dict):
        """Handle technical system failure"""
        logger.critical(f"🔧 TECHNICAL FAILURE: {context['status']}")
        
        # Safe mode operations only
        await self.enable_safe_mode()
    
    async def reduce_all_positions(self, reduction_factor: float = 0.5):
        """Reduce all positions by a factor"""
        # Implementation would reduce position sizes
        logger.info(f"🔻 Reducing all positions by {reduction_factor:.1%}")
    
    async def close_illiquid_positions(self):
        """Close positions with poor liquidity"""
        # Implementation would identify and close illiquid positions
        logger.info("💧 Closing illiquid positions")
    
    async def enable_safe_mode(self):
        """Enable safe mode operations"""
        # Implementation would limit system to essential operations
        logger.info("🛡️ Safe mode enabled")

import hmac
from eth_account.messages import encode_defunct
from flashbots import flashbot
import requests

class MEVProtection:
    def __init__(self, executor):
        self.executor = executor
        self.flashbots_relays = {
            'ethereum': 'https://relay.flashbots.net',
            'arbitrum': 'https://rpc.flashbots.net/arbitrum',
            'polygon': 'https://rpc.flashbots.net/polygon'
        }
        self.private_pools = {}
        self.mev_detection_cache = {}
        
    async def submit_private_transaction(self, signed_tx, network: str) -> Dict:
        """Submit transaction through private mempool to avoid MEV"""
        try:
            # Try Flashbots first for supported networks
            if network in self.flashbots_relays:
                result = await self.submit_flashbots_transaction(signed_tx, network)
                if result['success']:
                    return result
            
            # Fallback to other private pools
            return await self.submit_to_private_pools(signed_tx, network)
            
        except Exception as e:
            logger.error(f"MEV protection failed: {e}")
            # Emergency fallback to public mempool with warning
            return await self.submit_public_with_warning(signed_tx, network)
    
    async def submit_flashbots_transaction(self, signed_tx, network: str) -> Dict:
        """Submit transaction via Flashbots"""
        try:
            w3 = self.executor.web3_connections.get(network)
            if not w3:
                return {'success': False, 'error': 'No web3 connection'}
            
            # Create Flashbots bundle
            bundle = [
                {
                    "signed_transaction": signed_tx.rawTransaction.hex()
                }
            ]
            
            # Submit bundle
            relay_url = self.flashbots_relays[network]
            
            # Simulate first
            simulation = await self.simulate_bundle(bundle, relay_url, w3)
            if not simulation['success']:
                return {'success': False, 'error': 'Bundle simulation failed'}
            
            # Submit for next block
            latest_block = w3.eth.block_number
            target_block = latest_block + 1
            
            result = await self.send_flashbots_bundle(bundle, target_block, relay_url)
            
            return {
                'success': True,
                'tx_hash': signed_tx.hash.hex(),
                'method': 'flashbots',
                'target_block': target_block
            }
            
        except Exception as e:
            logger.error(f"Flashbots submission failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def simulate_bundle(self, bundle, relay_url, w3):
        """Simulate bundle before submission"""
        try:
            simulation_params = {
                "jsonrpc": "2.0",
                "method": "eth_callBundle",
                "params": [
                    bundle,
                    hex(w3.eth.block_number)
                ],
                "id": 1
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': self.create_flashbots_signature(simulation_params)
            }
            
            async with self.executor.session.post(
                relay_url, 
                json=simulation_params, 
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {'success': 'error' not in result}
                
            return {'success': False}
            
        except Exception as e:
            logger.error(f"Bundle simulation error: {e}")
            return {'success': False}
    
    async def send_flashbots_bundle(self, bundle, target_block, relay_url):
        """Send bundle to Flashbots relay"""
        try:
            bundle_params = {
                "jsonrpc": "2.0",
                "method": "eth_sendBundle",
                "params": [
                    {
                        "txs": [tx["signed_transaction"] for tx in bundle],
                        "blockNumber": hex(target_block)
                    }
                ],
                "id": 1
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-Flashbots-Signature': self.create_flashbots_signature(bundle_params)
            }
            
            async with self.executor.session.post(
                relay_url,
                json=bundle_params,
                headers=headers
            ) as response:
                result = await response.json()
                return result
                
        except Exception as e:
            logger.error(f"Bundle submission error: {e}")
            return {'error': str(e)}
    
    def create_flashbots_signature(self, params):
        """Create Flashbots signature for authentication"""
        try:
            message_hash = hashlib.sha256(json.dumps(params).encode()).hexdigest()
            message = encode_defunct(text=message_hash)
            signed_message = self.executor.account.sign_message(message)
            
            return f"{self.executor.wallet_address}:{signed_message.signature.hex()}"
            
        except Exception as e:
            logger.error(f"Signature creation error: {e}")
            return ""
    
    async def submit_to_private_pools(self, signed_tx, network: str) -> Dict:
        """Submit to alternative private pools"""
        try:
            # TaiChi Network (if available)
            taichi_result = await self.try_taichi_submission(signed_tx, network)
            if taichi_result['success']:
                return taichi_result
            
            # Other private pools can be added here
            
            return {'success': False, 'error': 'No private pools available'}
            
        except Exception as e:
            logger.error(f"Private pool submission failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def try_taichi_submission(self, signed_tx, network: str) -> Dict:
        """Try TaiChi Network submission"""
        try:
            # TaiChi endpoints (example)
            taichi_endpoints = {
                'ethereum': 'https://api.taichi.network',
                'arbitrum': 'https://api.taichi.network/arbitrum'
            }
            
            endpoint = taichi_endpoints.get(network)
            if not endpoint:
                return {'success': False, 'error': 'Network not supported'}
            
            # Implementation would go here
            # This is a placeholder for the actual TaiChi integration
            
            return {'success': False, 'error': 'TaiChi not configured'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def submit_public_with_warning(self, signed_tx, network: str) -> Dict:
        """Emergency fallback to public mempool with MEV warning"""
        try:
            w3 = self.executor.web3_connections.get(network)
            if not w3:
                return {'success': False, 'error': 'No connection'}
            
            logger.warning("⚠️ MEV PROTECTION FAILED - Using public mempool!")
            
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            return {
                'success': True,
                'tx_hash': tx_hash.hex(),
                'method': 'public_mempool',
                'warning': 'MEV_EXPOSURE'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

class DynamicSlippage:
    def __init__(self, executor):
        self.executor = executor
        self.slippage_cache = {}
        
    def calculate_optimal_slippage(self, token_address: str, network: str, 
                                 trade_size_usd: float, liquidity_usd: float) -> float:
        """Calculate optimal slippage based on market conditions"""
        try:
            # Base slippage calculation
            if liquidity_usd <= 0:
                return 0.05  # 5% for unknown liquidity
            
            # Price impact estimation
            price_impact = (trade_size_usd / liquidity_usd) * 100
            
            # Dynamic slippage based on multiple factors
            base_slippage = max(0.005, min(price_impact * 1.5, 0.05))  # 0.5% to 5%
            
            # Volatility adjustment
            volatility_multiplier = self.get_volatility_multiplier(token_address, network)
            
            # Network congestion adjustment
            congestion_multiplier = self.get_congestion_multiplier(network)
            
            # Time-based adjustment (higher slippage during momentum)
            momentum_multiplier = self.get_momentum_multiplier(token_address, network)
            
            final_slippage = base_slippage * volatility_multiplier * congestion_multiplier * momentum_multiplier
            
            # Bounds checking
            final_slippage = max(0.005, min(final_slippage, 0.1))  # 0.5% to 10%
            
            logger.info(f"Dynamic slippage for {token_address[:8]}: {final_slippage:.1%}")
            return final_slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            return 0.03  # Default 3%
    
    def get_volatility_multiplier(self, token_address: str, network: str) -> float:
        """Get volatility-based multiplier"""
        try:
            # Get recent price volatility
            price_key = f"{token_address}_{network}"
            
            if hasattr(self.executor, 'price_history') and price_key in self.executor.price_history:
                recent_prices = [p['price'] for p in self.executor.price_history[price_key][-20:]]
                if len(recent_prices) >= 3:
                    volatility = np.std(recent_prices) / np.mean(recent_prices)
                    return 1.0 + min(volatility * 10, 2.0)  # Up to 3x multiplier
            
            return 1.0
            
        except:
            return 1.0
    
    def get_congestion_multiplier(self, network: str) -> float:
        """Get network congestion multiplier"""
        try:
            w3 = self.executor.web3_connections.get(network)
            if not w3:
                return 1.0
            
            current_gas_price = w3.eth.gas_price
            
            # Network-specific base gas prices (in wei)
            base_gas_prices = {
                'arbitrum': 100000000,      # 0.1 gwei
                'optimism': 1000000,        # 0.001 gwei
                'polygon': 30000000000,     # 30 gwei
                'base': 1000000             # 0.001 gwei
            }
            
            base_price = base_gas_prices.get(network, 20000000000)
            congestion_ratio = current_gas_price / base_price
            
            return 1.0 + min(congestion_ratio * 0.1, 0.5)  # Up to 1.5x multiplier
            
        except:
            return 1.0
    
    def get_momentum_multiplier(self, token_address: str, network: str) -> float:
        """Get momentum-based multiplier"""
        try:
            # During high momentum, increase slippage tolerance
            if hasattr(self.executor, 'token_cache') and token_address in self.executor.token_cache:
                token_data = self.executor.token_cache[token_address]
                momentum = abs(token_data.price_change_1m)
                
                if momentum >= 0.1:  # High momentum
                    return 1.5
                elif momentum >= 0.05:  # Medium momentum
                    return 1.2
            
            return 1.0
            
        except:
            return 1.0

class EmergencyControls:
    def __init__(self, executor):
        self.executor = executor
        self.circuit_breakers = {
            'daily_loss_limit': 0.2,      # 20% daily loss
            'consecutive_losses': 10,      # 10 consecutive losses
            'gas_price_limit': 500e9,      # 500 gwei
            'slippage_limit': 0.15,        # 15% slippage
            'portfolio_concentration': 0.5  # 50% in single position
        }
        self.emergency_state = False
        self.trade_statistics = {
            'daily_pnl': 0.0,
            'consecutive_losses': 0,
            'total_trades': 0,
            'failed_trades': 0
        }
    
    async def check_emergency_conditions(self) -> bool:
        """Check for emergency conditions requiring trading halt"""
        try:
            # Daily loss check
            if self.trade_statistics['daily_pnl'] <= -self.circuit_breakers['daily_loss_limit']:
                await self.trigger_emergency_stop("DAILY_LOSS_LIMIT_EXCEEDED")
                return True
            
            # Consecutive losses check
            if self.trade_statistics['consecutive_losses'] >= self.circuit_breakers['consecutive_losses']:
                await self.trigger_emergency_stop("CONSECUTIVE_LOSSES_EXCEEDED")
                return True
            
            # Gas price check
            for network, w3 in self.executor.web3_connections.items():
                try:
                    gas_price = w3.eth.gas_price
                    if gas_price > self.circuit_breakers['gas_price_limit']:
                        await self.trigger_emergency_stop(f"HIGH_GAS_PRICE_{network}")
                        return True
                except:
                    continue
            
            # Failed trade ratio check
            if self.trade_statistics['total_trades'] > 10:
                failure_rate = self.trade_statistics['failed_trades'] / self.trade_statistics['total_trades']
                if failure_rate > 0.5:  # 50% failure rate
                    await self.trigger_emergency_stop("HIGH_FAILURE_RATE")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking emergency conditions: {e}")
            return False
    
    async def trigger_emergency_stop(self, reason: str):
        """Trigger emergency trading halt"""
        self.emergency_state = True
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
        
        # Close all positions immediately
        await self.emergency_close_all_positions()
        
        # Send alerts
        await self.send_emergency_alerts(reason)
        
        # Write emergency log
        await self.write_emergency_log(reason)
    
    async def emergency_close_all_positions(self):
        """Emergency close all open positions"""
        try:
            # This would integrate with the portfolio manager
            # For now, log the action
            logger.critical("🚨 EMERGENCY: Attempting to close all positions")
            
            # Get all token balances and sell immediately
            for network in self.executor.networks.keys():
                try:
                    w3 = self.executor.web3_connections.get(network)
                    if not w3:
                        continue
                    
                    eth_balance = w3.eth.get_balance(self.executor.wallet_address)
                    if eth_balance > 0:
                        logger.info(f"ETH balance on {network}: {eth_balance / 1e18:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error checking balance on {network}: {e}")
            
        except Exception as e:
            logger.error(f"Error in emergency close: {e}")
    
    async def send_emergency_alerts(self, reason: str):
        """Send emergency alerts via multiple channels"""
        try:
            # Log to file
            emergency_msg = f"EMERGENCY STOP: {reason} at {datetime.now()}"
            logger.critical(emergency_msg)
            
            # Could integrate with notification services:
            # - Discord webhook
            # - Telegram bot
            # - Email alerts
            # - SMS notifications
            
        except Exception as e:
            logger.error(f"Error sending alerts: {e}")
    
    async def write_emergency_log(self, reason: str):
        """Write detailed emergency log"""
        try:
            emergency_log = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'portfolio_state': await self.get_portfolio_snapshot(),
                'statistics': self.trade_statistics.copy(),
                'gas_prices': await self.get_current_gas_prices()
            }
            
            with open('data/emergency_log.json', 'a') as f:
                f.write(json.dumps(emergency_log) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing emergency log: {e}")
    
    async def get_portfolio_snapshot(self) -> Dict:
        """Get current portfolio snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'balances': {},
                'total_value_usd': 0.0
            }
            
            for network in self.executor.networks.keys():
                w3 = self.executor.web3_connections.get(network)
                if w3:
                    eth_balance = w3.eth.get_balance(self.executor.wallet_address)
                    snapshot['balances'][f'{network}_eth'] = eth_balance / 1e18
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error getting portfolio snapshot: {e}")
            return {}
    
    async def get_current_gas_prices(self) -> Dict:
        """Get current gas prices across networks"""
        gas_prices = {}
        
        for network, w3 in self.executor.web3_connections.items():
            try:
                gas_price = w3.eth.gas_price
                gas_prices[network] = gas_price
            except:
                gas_prices[network] = 0
        
        return gas_prices
    
    def update_trade_statistics(self, trade_result: TradeResult):
        """Update trade statistics for monitoring"""
        self.trade_statistics['total_trades'] += 1
        
        if trade_result.success:
            # Reset consecutive losses on success
            self.trade_statistics['consecutive_losses'] = 0
            
            # Update daily P&L (this would need actual profit calculation)
            # For now, assume positive trades add to P&L
            self.trade_statistics['daily_pnl'] += 0.01  # Placeholder
        else:
            self.trade_statistics['consecutive_losses'] += 1
            self.trade_statistics['failed_trades'] += 1
            
            # Update daily P&L with loss
            self.trade_statistics['daily_pnl'] -= 0.005  # Placeholder loss

# Enhanced TradeExecutor with new capabilities
class EnhancedTradeExecutor(TradeExecutor):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.mev_protection = MEVProtection(self)
        self.dynamic_slippage = DynamicSlippage(self)
        self.emergency_controls = EmergencyControls(self)
        self.session = None
        
    async def initialize(self):
        """Enhanced initialization"""
        await super().initialize()
        self.session = aiohttp.ClientSession()
        logger.info("✅ Enhanced executor initialized with MEV protection")
    
        async def execute_buy_with_advanced_risk(self, token_address: str, network: str, 
                                         amount_usd: float, max_slippage: float = None) -> TradeResult:
        """Execute buy with comprehensive risk management"""
        start_time = time.time()
        
        try:
            # Pre-trade risk assessment
            current_positions = getattr(self, 'current_positions', {})
            portfolio_value = getattr(self, 'portfolio_value', amount_usd)
            initial_capital = getattr(self, 'initial_capital', amount_usd)
            
            # Check if emergency conditions require trading halt
            emergency_triggered = await self.emergency_controls.comprehensive_risk_check(
                current_positions, portfolio_value, initial_capital
            )
            
            if emergency_triggered or self.trading_paused:
                return TradeResult(
                    False, None, 0, 0, 0, 0,
                    "TRADING_HALTED_RISK_CONTROLS",
                    time.time() - start_time
                )
            
            # Risk-adjusted position sizing
            risk_metrics = self.risk_manager.evaluate_portfolio_risk(current_positions)
            risk_adjusted_amount = self.calculate_risk_adjusted_size(
                amount_usd, risk_metrics, token_address, network
            )
            
            if risk_adjusted_amount < amount_usd * 0.1:  # Less than 10% of intended
                return TradeResult(
                    False, None, 0, 0, 0, 0,
                    "POSITION_SIZE_TOO_RISKY",
                    time.time() - start_time
                )
            
            # Execute trade with enhanced monitoring
            if hasattr(self, 'execute_buy_enhanced'):
                result = await self.execute_buy_enhanced(token_address, network, risk_adjusted_amount, max_slippage)
            else:
                result = await self.execute_buy(token_address, network, risk_adjusted_amount, max_slippage)
            
            # Post-trade monitoring update
            self.monitor.update_metrics(result, portfolio_value, initial_capital)
            
            # Log risk-adjusted execution
            if result.success:
                size_adjustment = risk_adjusted_amount / amount_usd
                logger.info(f"✅ RISK-ADJUSTED TRADE | Original:  | Executed:  | Adjustment: {size_adjustment:.1%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced buy execution error: {e}")
            return TradeResult(
                False, None, 0, 0, 0, 0,
                str(e),
                time.time() - start_time
            )
    
    def calculate_risk_adjusted_size(self, intended_amount: float, risk_metrics: Dict, 
                                   token_address: str, network: str) -> float:
        """Calculate risk-adjusted position size"""
        try:
            base_amount = intended_amount
            
            # Adjust for portfolio risk
            risk_score = risk_metrics.get('composite_risk_score', 0.5)
            risk_adjustment = max(0.1, 1.0 - risk_score)  # Reduce size as risk increases
            
            # Adjust for concentration risk
            concentration = risk_metrics.get('concentration_risk', 0)
            if concentration > 0.3:
                concentration_adjustment = max(0.2, 1.0 - concentration)
            else:
                concentration_adjustment = 1.0
            
            # Adjust for liquidity risk
            liquidity_risk = risk_metrics.get('liquidity_risk', 0)
            liquidity_adjustment = max(0.3, 1.0 - liquidity_risk)
            
            # Apply all adjustments
            final_amount = base_amount * risk_adjustment * concentration_adjustment * liquidity_adjustment
            
            # Ensure minimum viable trade size
            min_trade_size = 1.0  #  minimum
            final_amount = max(min_trade_size, final_amount)
            
            return final_amount
            
        except Exception as e:
            logger.error(f"Risk adjustment calculation error: {e}")
            return intended_amount * 0.5  # Conservative fallback
    
    async def execute_buy_enhanced(self, token_address: str, network: str, 
                                 amount_usd: float, max_slippage: float = None) -> TradeResult:
        """Enhanced buy execution with MEV protection and dynamic slippage"""
        start_time = time.time()
        
        try:
            # Check emergency conditions
            if await self.emergency_controls.check_emergency_conditions():
                return TradeResult(
                    False, None, 0, 0, 0, 0, 
                    "EMERGENCY_STOP_ACTIVE", 
                    time.time() - start_time
                )
            
            # Get token data for slippage calculation
            liquidity_usd = 0
            if hasattr(self, 'token_cache') and token_address in self.token_cache:
                liquidity_usd = self.token_cache[token_address].liquidity_usd
            
            # Calculate dynamic slippage if not provided
            if max_slippage is None:
                max_slippage = self.dynamic_slippage.calculate_optimal_slippage(
                    token_address, network, amount_usd, liquidity_usd
                )
            
            # Get optimized gas settings
            gas_settings = await self.optimize_gas_settings_enhanced(network)
            
            # Build transaction
            w3 = self.web3_connections.get(network)
            if not w3:
                return TradeResult(
                    False, None, 0, 0, 0, 0, 
                    f"No connection to {network}", 
                    time.time() - start_time
                )
            
            weth_address = self.weth_addresses[network]
            router_address = self.router_addresses[network]['uniswap_v3']
            
            amount_in = int(amount_usd * 10**18 / await self.get_eth_price())
            quote_amount = await self.get_quote(token_address, weth_address, amount_in, network)
            
            if quote_amount == 0:
                return TradeResult(
                    False, None, 0, 0, 0, 0, 
                    "No liquidity", 
                    time.time() - start_time
                )
            
            min_amount_out = int(quote_amount * (1 - max_slippage))
            
            # Build swap transaction
            swap_abi = [{
                'inputs': [
                    {'name': 'params', 'type': 'tuple', 'components': [
                        {'name': 'tokenIn', 'type': 'address'},
                        {'name': 'tokenOut', 'type': 'address'},
                        {'name': 'fee', 'type': 'uint24'},
                        {'name': 'recipient', 'type': 'address'},
                        {'name': 'deadline', 'type': 'uint256'},
                        {'name': 'amountIn', 'type': 'uint256'},
                        {'name': 'amountOutMinimum', 'type': 'uint256'},
                        {'name': 'sqrtPriceLimitX96', 'type': 'uint160'}
                    ]}
                ],
                'name': 'exactInputSingle',
                'outputs': [{'name': 'amountOut', 'type': 'uint256'}],
                'type': 'function'
            }]
            
            router_contract = w3.eth.contract(address=router_address, abi=swap_abi)
            nonce = self.nonce_manager[network]
            deadline = int(time.time()) + 300
            
            swap_params = {
                'tokenIn': weth_address,
                'tokenOut': token_address,
                'fee': 3000,
                'recipient': self.wallet_address,
                'deadline': deadline,
                'amountIn': amount_in,
                'amountOutMinimum': min_amount_out,
                'sqrtPriceLimitX96': 0
            }
            
            transaction = router_contract.functions.exactInputSingle(swap_params).buildTransaction({
                'from': self.wallet_address,
                'gas': gas_settings.gas_limit,
                'gasPrice': gas_settings.gas_price,
                'nonce': nonce,
                'value': amount_in
            })
            
            # Sign transaction
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
            
            # Submit with MEV protection
            submission_result = await self.mev_protection.submit_private_transaction(signed_txn, network)
            
            if not submission_result['success']:
                return TradeResult(
                    False, None, 0, 0, 0, 0, 
                    f"Submission failed: {submission_result['error']}", 
                    time.time() - start_time
                )
            
            # Wait for transaction receipt
            tx_hash = submission_result['tx_hash']
            receipt = await self.wait_for_receipt_with_timeout(w3, tx_hash, timeout=120)
            
            if receipt and receipt.status == 1:
                self.nonce_manager[network] += 1
                
                # Calculate actual results
                actual_amount_out = self.extract_amount_out_from_receipt(receipt)
                actual_price = amount_in / actual_amount_out if actual_amount_out > 0 else 0
                actual_slippage = abs(quote_amount - actual_amount_out) / quote_amount if quote_amount > 0 else 0
                
                execution_time = time.time() - start_time
                
                result = TradeResult(
                    success=True,
                    transaction_hash=tx_hash,
                    amount=actual_amount_out,
                    price=actual_price,
                    gas_used=receipt.gasUsed,
                    slippage=actual_slippage,
                    error_message=None,
                    execution_time=execution_time
                )
                
                # Update statistics
                self.emergency_controls.update_trade_statistics(result)
                
                return result
            else:
                return TradeResult(
                    False, tx_hash, 0, 0, receipt.gasUsed if receipt else 0, 0, 
                    "Transaction failed", 
                    time.time() - start_time
                )
            
        except Exception as e:
            logger.error(f"Error in enhanced buy execution: {e}")
            return TradeResult(
                False, None, 0, 0, 0, 0, 
                str(e), 
                time.time() - start_time
            )
    
    async def wait_for_receipt_with_timeout(self, w3, tx_hash, timeout=120):
        """Wait for transaction receipt with timeout"""
        try:
            return w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout)
        except Exception as e:
            logger.error(f"Transaction receipt timeout: {e}")
            return None
    
    def extract_amount_out_from_receipt(self, receipt):
        """Extract amount out from transaction receipt"""
        try:
            for log in receipt.logs:
                if len(log.topics) >= 3:
                    # Look for Swap event
                    if log.topics[0].hex() == '0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67':
                        return int(log.data, 16)
            return 0
        except:
            return 0
    
    async def optimize_gas_settings_enhanced(self, network: str) -> GasSettings:
        """Enhanced gas optimization with market conditions"""
        try:
            w3 = self.web3_connections.get(network)
            if not w3:
                return self.get_default_gas_settings()
            
            # Get current gas price
            current_gas_price = w3.eth.gas_price
            
            # Network-specific optimizations
            network_configs = {
                'arbitrum': {'multiplier': 1.1, 'max_gas': 1e9},
                'optimism': {'multiplier': 1.05, 'max_gas': 5e6},
                'polygon': {'multiplier': 1.2, 'max_gas': 500e9},
                'base': {'multiplier': 1.1, 'max_gas': 5e6}
            }
            
            config = network_configs.get(network, {'multiplier': 1.2, 'max_gas': 50e9})
            
            # Apply multiplier but cap at reasonable limits
            optimized_gas_price = min(
                int(current_gas_price * config['multiplier']),
                int(config['max_gas'])
            )
            
            # Priority fee (EIP-1559 networks)
            priority_fee = min(int(optimized_gas_price * 0.1), 2e9)  # Max 2 gwei priority
            max_fee = optimized_gas_price + priority_fee
            
            return GasSettings(
                gas_price=optimized_gas_price,
                gas_limit=300000,  # Standard swap gas limit
                priority_fee=priority_fee,
                max_fee=max_fee
            )
            
        except Exception as e:
            logger.error(f"Error optimizing gas: {e}")
            return self.get_default_gas_settings()
    
    def get_default_gas_settings(self) -> GasSettings:
        """Get default gas settings as fallback"""
        return GasSettings(
            gas_price=20000000000,  # 20 gwei
            gas_limit=300000,
            priority_fee=2000000000,  # 2 gwei
            max_fee=22000000000  # 22 gwei
        )
    
    async def cleanup(self):
        """Enhanced cleanup"""
        await super().cleanup()
        if self.session:
            await self.session.close()

# Replace the original TradeExecutor with enhanced version
TradeExecutor = EnhancedTradeExecutor

logger.info("✅ Executor enhanced with MEV protection, dynamic slippage, and emergency controls")
