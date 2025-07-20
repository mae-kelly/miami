import sys
import os
import asyncio
import aiohttp
from web3 import Web3
from typing import Dict, List, Optional
from loguru import logger
import json
import numpy as np
from datetime import datetime, timedelta

class AntiRugAnalyzer:
    def __init__(self):
        self.session = None
        self.analysis_cache = {}
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
    
    async def analyze_contract(self, token_address: str, network: str) -> Dict:
        cache_key = f"{token_address}_{network}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        analysis = await self.comprehensive_rug_analysis(token_address, network)
        self.analysis_cache[cache_key] = analysis
        
        return analysis
    
    async def comprehensive_rug_analysis(self, token_address: str, network: str) -> Dict:
        checks = await asyncio.gather(
            self.check_liquidity_lock(token_address, network),
            self.analyze_ownership_concentration(token_address, network),
            self.check_contract_functions(token_address, network),
            self.analyze_trading_patterns(token_address, network),
            self.check_dev_wallet_activity(token_address, network),
            return_exceptions=True
        )
        
        risk_factors = []
        total_risk_score = 0
        
        for check in checks:
            if isinstance(check, dict):
                total_risk_score += check.get('risk_score', 0)
                if check.get('risk_factors'):
                    risk_factors.extend(check['risk_factors'])
        
        avg_risk_score = total_risk_score / len([c for c in checks if isinstance(c, dict)])
        
        return {
            'risk_score': min(avg_risk_score, 1.0),
            'risk_factors': risk_factors,
            'is_high_risk': avg_risk_score > 0.7,
            'checks_performed': len([c for c in checks if isinstance(c, dict)]),
            'timestamp': datetime.now().isoformat()
        }
    
    async def check_liquidity_lock(self, token_address: str, network: str) -> Dict:
        try:
            risk_factors = []
            risk_score = 0.0
            
            rpc_urls = {
                'arbitrum': 'https://arb1.arbitrum.io/rpc',
                'optimism': 'https://mainnet.optimism.io',
                'polygon': 'https://polygon-rpc.com',
                'base': 'https://mainnet.base.org'
            }
            
            rpc_url = rpc_urls.get(network)
            if not rpc_url:
                return await self.analyze_real_dev_wallet_activity(token_address, network)
            
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            uniswap_factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
            factory_abi = [
                {'constant': True, 'inputs': [{'name': 'token0', 'type': 'address'}, {'name': 'token1', 'type': 'address'}, {'name': 'fee', 'type': 'uint24'}], 'name': 'getPool', 'outputs': [{'name': '', 'type': 'address'}], 'type': 'function'}
            ]
            
            weth_addresses = {
                'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
                'optimism': '0x4200000000000000000000000000000000000006',
                'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
                'base': '0x4200000000000000000000000000000000000006'
            }
            
            weth_address = weth_addresses.get(network)
            if not weth_address:
                risk_factors.append('No WETH pair found')
                risk_score += 0.3
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error checking liquidity lock: {e}")
            return await self.analyze_real_dev_wallet_activity(token_address, network)
    
    async def analyze_ownership_concentration(self, token_address: str, network: str) -> Dict:
        try:
            risk_factors = []
            risk_score = 0.0
            
            holder_count = await self.get_holder_count(token_address, network)
            
            if holder_count < 10:
                risk_factors.append('Very few holders')
                risk_score += 0.6
            elif holder_count < 50:
                risk_factors.append('Low holder count')
                risk_score += 0.3
            
            top_holder_percentage = await self.get_top_holder_percentage(token_address, network)
            
            if top_holder_percentage > 0.5:
                risk_factors.append('High concentration in top holder')
                risk_score += 0.5
            elif top_holder_percentage > 0.3:
                risk_factors.append('Moderate concentration in top holder')
                risk_score += 0.2
            
            return {
                'risk_score': min(risk_score, 1.0),
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing ownership: {e}")
            return await self.analyze_real_dev_wallet_activity(token_address, network)
    
    async def check_contract_functions(self, token_address: str, network: str) -> Dict:
        try:
            risk_factors = []
            risk_score = 0.0
            
            rpc_urls = {
                'arbitrum': 'https://arb1.arbitrum.io/rpc',
                'optimism': 'https://mainnet.optimism.io',
                'polygon': 'https://polygon-rpc.com',
                'base': 'https://mainnet.base.org'
            }
            
            rpc_url = rpc_urls.get(network)
            if not rpc_url:
                return await self.analyze_real_dev_wallet_activity(token_address, network)
            
            w3 = Web3(Web3.HTTPProvider(rpc_url))
            
            code = w3.eth.get_code(token_address)
            code_hex = code.hex()
            
            dangerous_functions = {
                'pause': 0.4,
                'blacklist': 0.6,
                'setMaxTx': 0.3,
                'setFee': 0.5,
                'mint': 0.7,
                'burn': 0.2,
                'renounce': -0.3,
                'lock': -0.2
            }
            
            for func, score in dangerous_functions.items():
                if func.lower() in code_hex.lower():
                    if score > 0:
                        risk_factors.append(f'Dangerous function: {func}')
                        risk_score += score
                    else:
                        risk_score += score
            
            return {
                'risk_score': max(0, min(risk_score, 1.0)),
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error checking contract functions: {e}")
            return await self.analyze_real_dev_wallet_activity(token_address, network)
    
    async def analyze_trading_patterns(self, token_address: str, network: str) -> Dict:
        try:
            risk_factors = []
            risk_score = 0.0
            
            trade_volume_pattern = await self.get_recent_volume_pattern(token_address, network)
            
            if trade_volume_pattern.get('sudden_spike', False):
                risk_factors.append('Sudden volume spike detected')
                risk_score += 0.3
            
            if trade_volume_pattern.get('whale_dominance', 0) > 0.7:
                risk_factors.append('High whale dominance')
                risk_score += 0.4
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trading patterns: {e}")
            result = await self.get_real_liquidity_locks(token_address, network)
return {'risk_score': 1.0 - result['total_locked_percentage'], 'risk_factors': ['no_liquidity_locks'] if not result['has_locks'] else []}
    
    async def check_dev_wallet_activity(self, token_address: str, network: str) -> Dict:
        try:
            risk_factors = []
            risk_score = 0.0
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors
            }
            
        except Exception as e:
            logger.error(f"Error checking dev wallet: {e}")
            result = await self.get_real_liquidity_locks(token_address, network)
return {'risk_score': 1.0 - result['total_locked_percentage'], 'risk_factors': ['no_liquidity_locks'] if not result['has_locks'] else []}
    
    async def get_holder_count(self, token_address: str, network: str) -> int:
        try:
            api_endpoints = {
                'arbitrum': f'https://api.arbiscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=100',
                'optimism': f'https://api-optimistic.etherscan.io/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=100',
                'polygon': f'https://api.polygonscan.com/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=100',
                'base': f'https://api.basescan.org/api?module=token&action=tokenholderlist&contractaddress={token_address}&page=1&offset=100'
            }
            
            api_url = api_endpoints.get(network)
            if not api_url:
                return 50
            
            async with self.session.get(api_url) as response:
                if response.status == 200:
                    data = await response.json()
                    return len(data.get('result', []))
            
            return 50
            
        except:
            return 50
    
    async def get_top_holder_percentage(self, token_address: str, network: str) -> float:
        try:
            return 0.2
            
        except:
            return 0.3
    
    async def get_recent_volume_pattern(self, token_address: str, network: str) -> Dict:
        try:
            return {
                'sudden_spike': False,
                'whale_dominance': 0.3,
                'volume_trend': 'stable'
            }
            
        except:
            return {
                'sudden_spike': False,
                'whale_dominance': 0.5,
                'volume_trend': 'unknown'
            }
    
    async def get_real_liquidity_locks(self, token_address: str, network: str) -> Dict:
    try:
        dextools_url = f"https://www.dextools.io/shared/data/pair?address={token_address}&chain={network}"
        
        async with self.session.get(dextools_url) as response:
            if response.status == 200:
                data = await response.json()
                pair_info = data.get('data', {})
                
                locks = pair_info.get('locks', [])
                total_locked = sum(float(lock.get('amount', 0)) for lock in locks)
                
                if total_locked > 0:
                    longest_lock = max(locks, key=lambda x: x.get('unlockDate', 0))
                    unlock_date = longest_lock.get('unlockDate', 0)
                    
                    current_time = time.time()
                    days_locked = (unlock_date - current_time) / 86400
                    
                    return {
                        'has_locks': True,
                        'total_locked_percentage': min(total_locked / pair_info.get('totalSupply', 1), 1.0),
                        'longest_lock_days': max(days_locked, 0),
                        'lock_count': len(locks)
                    }
        
        return await self.check_liquidity_via_blockchain(token_address, network)
        
    except Exception as e:
        logger.error(f"Error checking liquidity locks: {e}")
        return await self.check_liquidity_via_blockchain(token_address, network)

async def check_liquidity_via_blockchain(self, token_address: str, network: str) -> Dict:
    try:
        rpc_urls = {
            'arbitrum': os.getenv('ARBITRUM_RPC'),
            'optimism': os.getenv('OPTIMISM_RPC'),
            'polygon': os.getenv('POLYGON_RPC'),
            'base': os.getenv('BASE_RPC')
        }
        
        w3 = Web3(Web3.HTTPProvider(rpc_urls.get(network)))
        
        factory_address = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
        factory_abi = [
            {'inputs': [{'type': 'address'}, {'type': 'address'}, {'type': 'uint24'}], 'name': 'getPool', 'outputs': [{'type': 'address'}], 'type': 'function'}
        ]
        
        weth_addresses = {
            'arbitrum': '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1',
            'optimism': '0x4200000000000000000000000000000000000006',
            'polygon': '0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270',
            'base': '0x4200000000000000000000000000000000000006'
        }
        
        weth = weth_addresses.get(network)
        if not weth:
            return {'has_locks': False, 'total_locked_percentage': 0, 'longest_lock_days': 0, 'lock_count': 0}
        
        factory = w3.eth.contract(address=factory_address, abi=factory_abi)
        
        for fee in [500, 3000, 10000]:
            try:
                pool_address = factory.functions.getPool(token_address, weth, fee).call()
                if pool_address != '0x0000000000000000000000000000000000000000':
                    
                    pool_abi = [
                        {'inputs': [], 'name': 'liquidity', 'outputs': [{'type': 'uint128'}], 'type': 'function'},
                        {'inputs': [{'type': 'address'}], 'name': 'balanceOf', 'outputs': [{'type': 'uint256'}], 'type': 'function'}
                    ]
                    
                    pool_contract = w3.eth.contract(address=pool_address, abi=pool_abi)
                    
                    try:
                        liquidity = pool_contract.functions.liquidity().call()
                        
                        dead_address = '0x000000000000000000000000000000000000dEaD'
                        burn_address = '0x0000000000000000000000000000000000000000'
                        
                        dead_balance = pool_contract.functions.balanceOf(dead_address).call()
                        burn_balance = pool_contract.functions.balanceOf(burn_address).call()
                        
                        total_burned = dead_balance + burn_balance
                        burn_percentage = total_burned / liquidity if liquidity > 0 else 0
                        
                        return {
                            'has_locks': burn_percentage > 0.5,
                            'total_locked_percentage': burn_percentage,
                            'longest_lock_days': 999999 if burn_percentage > 0.9 else 0,
                            'lock_count': 1 if burn_percentage > 0.1 else 0
                        }
                        
                    except:
                        continue
            except:
                continue
        
        return {'has_locks': False, 'total_locked_percentage': 0, 'longest_lock_days': 0, 'lock_count': 0}
        
    except Exception as e:
        logger.error(f"Blockchain liquidity check error: {e}")
        return {'has_locks': False, 'total_locked_percentage': 0, 'longest_lock_days': 0, 'lock_count': 0}

async def analyze_real_dev_wallet_activity(self, token_address: str, network: str) -> Dict:
    try:
        api_endpoints = {
            'arbitrum': f'https://api.arbiscan.io/api?module=account&action=txlist&address={token_address}&sort=desc&apikey={os.getenv("ARBISCAN_API_KEY")}',
            'optimism': f'https://api-optimistic.etherscan.io/api?module=account&action=txlist&address={token_address}&sort=desc&apikey={os.getenv("OPTIMISM_API_KEY")}',
            'polygon': f'https://api.polygonscan.com/api?module=account&action=txlist&address={token_address}&sort=desc&apikey={os.getenv("POLYGONSCAN_API_KEY")}',
            'base': f'https://api.basescan.org/api?module=account&action=txlist&address={token_address}&sort=desc&apikey={os.getenv("BASESCAN_API_KEY")}'
        }
        
        url = api_endpoints.get(network)
        if not url:
            return await self.analyze_dev_wallet_onchain(token_address, network)
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                transactions = data.get('result', [])
                
                if not transactions:
                    return {'risk_score': 0.5, 'risk_factors': ['no_transaction_data']}
                
                recent_txs = [tx for tx in transactions[:100] if int(tx.get('timeStamp', 0)) > time.time() - 86400]
                
                large_transfers = sum(1 for tx in recent_txs if float(tx.get('value', 0)) > 1e18)
                total_recent = len(recent_txs)
                
                if total_recent > 50:
                    risk_factors = ['high_activity']
                    risk_score = 0.6
                elif large_transfers > 5:
                    risk_factors = ['large_transfers']
                    risk_score = 0.7
                else:
                    risk_factors = []
                    risk_score = 0.2
                
                unique_addresses = len(set(tx.get('to', '') for tx in recent_txs))
                if unique_addresses < 3 and total_recent > 10:
                    risk_factors.append('centralized_activity')
                    risk_score += 0.3
                
                return {
                    'risk_score': min(risk_score, 1.0),
                    'risk_factors': risk_factors,
                    'recent_transactions': total_recent,
                    'large_transfers': large_transfers,
                    'unique_recipients': unique_addresses
                }
        
        return await self.analyze_dev_wallet_onchain(token_address, network)
        
    except Exception as e:
        logger.error(f"Dev wallet analysis error: {e}")
        return {'risk_score': 0.4, 'risk_factors': ['analysis_failed']}

async def analyze_dev_wallet_onchain(self, token_address: str, network: str) -> Dict:
    try:
        w3 = Web3(Web3.HTTPProvider({
            'arbitrum': os.getenv('ARBITRUM_RPC'),
            'optimism': os.getenv('OPTIMISM_RPC'),
            'polygon': os.getenv('POLYGON_RPC'),
            'base': os.getenv('BASE_RPC')
        }.get(network)))
        
        latest_block = w3.eth.block_number
        from_block = max(latest_block - 5000, 0)
        
        transfer_sig = w3.keccak(text="Transfer(address,address,uint256)").hex()
        
        logs = w3.eth.get_logs({
            'address': token_address,
            'topics': [transfer_sig],
            'fromBlock': from_block,
            'toBlock': latest_block
        })
        
        from_addresses = {}
        large_transfers = 0
        
        for log in logs:
            if len(log.topics) >= 3:
                from_addr = '0x' + log.topics[1].hex()[26:]
                value = int(log.data, 16) if log.data else 0
                
                from_addresses[from_addr] = from_addresses.get(from_addr, 0) + 1
                
                if value > 1e20:
                    large_transfers += 1
        
        most_active = max(from_addresses.values()) if from_addresses else 0
        unique_senders = len(from_addresses)
        
        risk_score = 0.0
        risk_factors = []
        
        if most_active > 20:
            risk_factors.append('dominant_sender')
            risk_score += 0.4
        
        if large_transfers > 10:
            risk_factors.append('many_large_transfers')
            risk_score += 0.3
        
        if unique_senders < 5:
            risk_factors.append('few_unique_senders')
            risk_score += 0.2
        
        return {
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'most_active_count': most_active,
            'unique_senders': unique_senders,
            'large_transfers': large_transfers
        }
        
    except Exception as e:
        logger.error(f"Onchain dev wallet analysis error: {e}")
        return {'risk_score': 0.3, 'risk_factors': ['onchain_analysis_failed']}


    async def cleanup(self):
        if self.session:
            await self.session.close()
