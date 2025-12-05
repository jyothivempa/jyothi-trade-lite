import asyncio
import logging
import sqlite3
import aiohttp
import io
import sys
import os
import pytz
import argparse
import pandas as pd
import pandas_ta
import threading
import copy
import yfinance as yf
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, time as dtime
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from logging.handlers import RotatingFileHandler
from tabulate import tabulate
from colorama import Fore, Style, init
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

# Try importing KiteConnect
try:
    from kiteconnect import KiteConnect, KiteTicker
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False

# ==========================================
# 1. CONFIGURATION & GLOBAL STATE
# ==========================================

init(autoreset=True)
IST = pytz.timezone('Asia/Kolkata')
app = FastAPI()

# Global Shared State for UI
UI_STATE = {
    "bot_status": "OFFLINE",
    "pnl": 0.0,
    "risk": 0.0,
    "trades": 0,
    "active_positions": [],
    "top_picks": [], # Recent valid signals
    "market_data": {} # Nifty 200 Heatmap/Sectors
}

# --- SECURE IMPORT ---
try:
    import mykeys
    print(f"{Fore.GREEN}🔑 Credentials loaded from mykeys.py{Style.RESET_ALL}")
except ImportError:
    print(f"{Fore.RED}⚠️ mykeys.py not found! Using defaults.{Style.RESET_ALL}")
    class mykeys:
        TELEGRAM_BOT_TOKEN = ""
        TELEGRAM_CHAT_ID = ""
        KITE_API_KEY = ""
        KITE_ACCESS_TOKEN = ""

@dataclass
class SystemConfig:
    # --- CREDENTIALS ---
    TELEGRAM_TOKEN: str = mykeys.TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID: str = mykeys.TELEGRAM_CHAT_ID
    API_KEY: str = mykeys.KITE_API_KEY
    ACCESS_TOKEN: str = mykeys.KITE_ACCESS_TOKEN

    # --- MODE ---
    DATA_SOURCE: str = "YFINANCE"       
    EXECUTION_MODE: str = "PAPER"       
    
    # --- SYSTEM ---
    STRATEGY_NAME: str = "TREND"
    DB_PATH: str = "titanium_v27.db"
    LOG_FILE: str = "titanium_v27.log"
    UNIVERSE_URL: str = "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"
    
    FORCE_RUN: bool = False             
    SCAN_INTERVAL: int = 15             
    MONITOR_INTERVAL: int = 2           
    NETWORK_TIMEOUT: int = 5            
    CONCURRENT_FETCHES: int = 5         
    CHUNK_SIZE: int = 10                
    MAX_UNIVERSE_SIZE: int = 100        

    # --- RISK ---
    CAPITAL: float = 1_000_000
    RISK_PER_TRADE: float = 5_000
    MAX_DAILY_RISK: float = 25_000
    MAX_POSITIONS: int = 10
    MIN_TURNOVER: float = 20_000_000    
    
    # --- SAFETY ---
    TRANSACTION_COST_PCT: float = 0.001   
    SLIPPAGE_PCT: float = 0.0005          
    GLOBAL_KILL_SWITCH_PCT: float = 0.03  
    MAX_DATA_DELAY_MIN: int = 3           
    
    ENTRY_CUTOFF: dtime = dtime(15, 0)      
    SQUARE_OFF_TIME: dtime = dtime(15, 20)  

@dataclass
class TradeSignal:
    ticker: str
    signal: str
    price: float
    qty: int
    sl: float
    target: float
    score: float
    strategy_ref: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(IST))

CONFIG = SystemConfig()

# ==========================================
# 2. TELEGRAM ENGINE
# ==========================================

class TelegramNotifier:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        self.enabled = len(self.token) > 10 and len(self.chat_id) > 5
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        if self.enabled: self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session: await self.session.close()

    async def send(self, message: str):
        if not self.enabled or not self.session: return
        try:
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            async with self.session.post(self.base_url, json=payload) as response:
                if response.status != 200: pass
        except Exception: pass

    async def send_entry(self, s: TradeSignal, filled_price: float):
        mode_icon = "🔴 LIVE" if self.config.EXECUTION_MODE == "LIVE" else "⚪ PAPER"
        msg = f"⚡ *OPENING* - {mode_icon}\nSymbol: `{s.ticker}`\nSide: *{s.signal}*\nQty: `{s.qty}` @ `{filled_price:.2f}`"
        await self.send(msg)

    async def send_exit(self, ticker, side, price, pnl, reason):
        icon = "✅" if pnl > 0 else "🛑"
        msg = f"{icon} *CLOSED*\nSymbol: `{ticker}`\nExit: `{price:.2f}`\nP&L: *₹{pnl:.2f}*\nReason: _{reason}_"
        await self.send(msg)

    async def send_summary(self, count, pnl, risk):
        msg = f"📊 *EOD SUMMARY*\nTrades: `{count}`\nNet P&L: *₹{pnl:.2f}*\nRisk Used: ₹{risk:.2f}"
        await self.send(msg)

# ==========================================
# 3. LOGGING
# ==========================================

class ISTFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, IST).timetuple()

def setup_logger(config: SystemConfig):
    logger = logging.getLogger("TITANIUM")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = ISTFormatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = RotatingFileHandler(config.LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

LOGGER = setup_logger(CONFIG)

# ==========================================
# 4. DATA PROVIDER INTERFACE & IMPLEMENTATIONS
# ==========================================

class IDataProvider(ABC):
    @abstractmethod
    async def initialize_universe(self): pass
    @abstractmethod
    async def fetch_candles(self, tickers: List[str], timeframe: str) -> Optional[pd.DataFrame]: pass
    @abstractmethod
    def is_data_fresh(self, df: pd.DataFrame, ticker: Optional[str] = None) -> bool: pass
    @abstractmethod
    async def start(self): pass
    @abstractmethod
    async def stop(self): pass

class YFinanceProvider(IDataProvider):
    def __init__(self, config: SystemConfig):
        self.config = config
        self.universe = []
        self.semaphore = asyncio.Semaphore(config.CONCURRENT_FETCHES)
        self.session: Optional[aiohttp.ClientSession] = None
        self.priority_tickers = {"RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS"}

    async def start(self): self.session = aiohttp.ClientSession()
    async def stop(self): 
        if self.session: await self.session.close()

    async def initialize_universe(self):
        if not self.session: await self.start()
        LOGGER.info("⌛ [YF] Initializing Universe...")
        try:
            async with self.session.get(self.config.UNIVERSE_URL, timeout=10) as response:
                if response.status == 200:
                    csv_text = await response.text()
                    df = pd.read_csv(io.StringIO(csv_text))
                    raw_universe = [f"{x}.NS" for x in df['Symbol'].tolist()]
                    self.universe = sorted(raw_universe, key=lambda x: 0 if x in self.priority_tickers else 1)
                    LOGGER.info(f"✅ Universe loaded: {len(self.universe)} tickers")
                else: raise Exception(f"HTTP {response.status}")
        except Exception:
            self.universe = list(self.priority_tickers)

    def is_data_fresh(self, df: pd.DataFrame, ticker: Optional[str] = None) -> bool:
        if df is None or df.empty: return False
        if self.config.FORCE_RUN: return True
        last_ts = df.index[-1]
        if last_ts.tzinfo is None: last_ts = IST.localize(last_ts)
        delta = datetime.now(IST) - last_ts
        return delta <= timedelta(minutes=self.config.MAX_DATA_DELAY_MIN)

    async def fetch_candles(self, tickers: List[str], timeframe: str) -> Optional[pd.DataFrame]:
        period = "2d" if timeframe == "5m" else "1d"
        async with self.semaphore:
            loop = asyncio.get_running_loop()
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(
                        None, 
                        lambda: yf.download(tickers, period=period, interval=timeframe, group_by='ticker', progress=False, threads=True)
                    ),
                    timeout=self.config.NETWORK_TIMEOUT
                )
            except Exception: return None

class KiteWebSocketProvider(IDataProvider):
    def __init__(self, config: SystemConfig):
        if not KITE_AVAILABLE: raise ImportError("KiteConnect required")
        self.config = config
        self.kws = KiteTicker(config.API_KEY, config.ACCESS_TOKEN)
        self.cache_lock = threading.Lock()
        self.candle_cache = {} 
        self.token_map = {} 
        self.symbol_map = {} 
        self.universe = []
        self.connected = False
        self.last_tick_time: Dict[str, datetime] = {} 

    async def start(self):
        LOGGER.info("🔌 Connecting to Kite Ticker...")
        self.kws.on_ticks = self.on_ticks
        self.kws.on_connect = self.on_connect
        self.kws.on_error = self.on_error
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, lambda: self.kws.connect(threaded=True))
        for _ in range(10):
            if self.connected: break
            await asyncio.sleep(1)

    async def stop(self):
        if self.kws.is_connected(): self.kws.close()

    def on_connect(self, ws, response):
        self.connected = True
        LOGGER.info(f"⚡ Connected to Kite Stream!")
        if self.token_map:
            ws.subscribe(list(self.token_map.keys()))
            ws.set_mode(ws.MODE_FULL, list(self.token_map.keys()))

    def on_error(self, ws, code, reason):
        LOGGER.error(f"WebSocket Error: {code} - {reason}")

    def on_ticks(self, ws, ticks):
        timestamp = datetime.now(IST)
        with self.cache_lock:
            for tick in ticks:
                token = tick['instrument_token']
                ltp = tick['last_price']
                symbol = self.token_map.get(token)
                if symbol: 
                    self._process_tick(symbol, ltp, timestamp)
                    self.last_tick_time[symbol] = timestamp

    def _process_tick(self, symbol, ltp, timestamp):
        ts_1m = timestamp.replace(second=0, microsecond=0)
        ts_5m = timestamp.replace(minute=(timestamp.minute // 5) * 5, second=0, microsecond=0)
        self._update_candle(symbol, '1m', ts_1m, ltp)
        self._update_candle(symbol, '5m', ts_5m, ltp)

    def _update_candle(self, symbol, timeframe, bucket_ts, price):
        if symbol not in self.candle_cache:
            self.candle_cache[symbol] = {'1m': pd.DataFrame(), '5m': pd.DataFrame()}
        df = self.candle_cache[symbol][timeframe]
        if df.empty or df.index[-1] != bucket_ts:
            new_row = pd.DataFrame({'Open': [price], 'High': [price], 'Low': [price], 'Close': [price], 'Volume': [0]}, index=[bucket_ts])
            self.candle_cache[symbol][timeframe] = pd.concat([df, new_row]).tail(200)
        else:
            idx = df.index[-1]
            df.at[idx, 'High'] = max(df.at[idx, 'High'], price)
            df.at[idx, 'Low'] = min(df.at[idx, 'Low'], price)
            df.at[idx, 'Close'] = price

    async def initialize_universe(self):
        LOGGER.info("📜 Fetching Official Nifty 500 & Mapping to Kite...")
        kite = KiteConnect(api_key=self.config.API_KEY)
        kite.set_access_token(self.config.ACCESS_TOKEN)
        async with aiohttp.ClientSession() as session:
            async with session.get(self.config.UNIVERSE_URL) as response:
                if response.status != 200: raise Exception("CSV Fetch Failed")
                csv_text = await response.text()
                df = pd.read_csv(io.StringIO(csv_text))
                official_symbols = set(df['Symbol'].tolist())

        instruments = kite.instruments("NSE")
        self.symbol_map = {}
        for i in instruments:
            sym = i['tradingsymbol']
            if sym in official_symbols:
                self.symbol_map[f"{sym}.NS"] = i['instrument_token']
        
        self.symbol_map = dict(list(self.symbol_map.items())[:self.config.MAX_UNIVERSE_SIZE])
        self.token_map = {v: k for k, v in self.symbol_map.items()}
        self.universe = list(self.symbol_map.keys())
        
        if self.connected:
            self.kws.subscribe(list(self.token_map.keys()))
            self.kws.set_mode(self.kws.MODE_FULL, list(self.token_map.keys()))

    async def fetch_candles(self, tickers: List[str], timeframe: str) -> Optional[pd.DataFrame]:
        with self.cache_lock:
            safe_cache = copy.deepcopy(self.candle_cache)
        combined = {}
        for t in tickers:
            if t in safe_cache and not safe_cache[t][timeframe].empty:
                combined[t] = safe_cache[t][timeframe]
        if not combined: return None
        return pd.concat(combined, axis=1)

    def is_data_fresh(self, df: pd.DataFrame, ticker: Optional[str] = None) -> bool:
        if not self.kws.is_connected(): return False
        if self.config.FORCE_RUN: return True
        if ticker:
            last_upd = self.last_tick_time.get(ticker)
            if not last_upd: return False
            delta = datetime.now(IST) - last_upd
            return delta <= timedelta(minutes=self.config.MAX_DATA_DELAY_MIN)
        return True

# ==========================================
# 5. BROKER IMPLEMENTATIONS
# ==========================================

class IBroker(ABC):
    @abstractmethod
    async def place_order(self, signal: TradeSignal) -> float: pass
    @abstractmethod
    async def close_position(self, ticker: str, qty: int, side: str, reason: str) -> float: pass

class SimulatedBroker(IBroker):
    def __init__(self, config: SystemConfig): self.config = config
    async def place_order(self, signal: TradeSignal) -> float:
        impact = signal.price * self.config.SLIPPAGE_PCT
        return signal.price + impact if signal.signal == "BUY" else signal.price - impact
    async def close_position(self, ticker: str, qty: int, side: str, reason: str) -> float:
        return 0.0

class ZerodhaKiteBroker(IBroker):
    def __init__(self, config: SystemConfig):
        self.kite = KiteConnect(api_key=config.API_KEY)
        self.kite.set_access_token(config.ACCESS_TOKEN)

    async def place_order(self, signal: TradeSignal) -> float:
        try:
            symbol = signal.ticker.replace('.NS', '')
            type = self.kite.TRANSACTION_TYPE_BUY if signal.signal == "BUY" else self.kite.TRANSACTION_TYPE_SELL
            order_id = self.kite.place_order(self.kite.VARIETY_REGULAR, self.kite.EXCHANGE_NSE, symbol, type, signal.qty, self.kite.PRODUCT_MIS, self.kite.ORDER_TYPE_MARKET, tag="TITANIUM_V27")
            return await self._poll_order_status(order_id)
        except Exception: return 0.0

    async def close_position(self, ticker: str, qty: int, side: str, reason: str) -> float:
        try:
            symbol = ticker.replace('.NS', '')
            type = self.kite.TRANSACTION_TYPE_SELL if side == "BUY" else self.kite.TRANSACTION_TYPE_BUY
            order_id = self.kite.place_order(self.kite.VARIETY_REGULAR, self.kite.EXCHANGE_NSE, symbol, type, qty, self.kite.PRODUCT_MIS, self.kite.ORDER_TYPE_MARKET, tag=f"EXIT_{reason}")
            return await self._poll_order_status(order_id)
        except Exception: return 0.0

    async def _poll_order_status(self, order_id):
        for _ in range(10):
            hist = self.kite.order_history(order_id)
            if hist[-1]['status'] == 'COMPLETE': return float(hist[-1]['average_price'])
            await asyncio.sleep(0.5)
        raise Exception("Timeout")

# ==========================================
# 6. STATE & ORCHESTRATOR
# ==========================================

class StateManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, ticker TEXT, side TEXT, qty INTEGER, entry REAL, sl REAL, target REAL, status TEXT, timestamp DATETIME, exit_price REAL, exit_time DATETIME, net_pnl REAL, costs REAL, exit_reason TEXT, strategy TEXT, is_trailing INTEGER DEFAULT 0)")
        cur.execute("CREATE TABLE IF NOT EXISTS daily_stats (date TEXT PRIMARY KEY, risk_used REAL)")
        self.conn.commit()

    def has_open_position(self, ticker: str) -> bool:
        return self.conn.execute("SELECT 1 FROM trades WHERE ticker = ? AND status = 'OPEN'", (ticker,)).fetchone() is not None

    def log_trade(self, signal: TradeSignal, fill_price: float):
        self.conn.execute("INSERT INTO trades (ticker, side, qty, entry, sl, target, status, timestamp, strategy) VALUES (?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)", (signal.ticker, signal.signal, signal.qty, fill_price, signal.sl, signal.target, signal.timestamp, signal.strategy_ref))
        self.conn.commit()

    def get_open_trades(self) -> List[Dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM trades WHERE status = 'OPEN'").fetchall()]

    def close_trade(self, trade_id, exit_price, reason):
        t = self.conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
        if not t: return 0.0, "", "", 0.0
        gross = (exit_price - t['entry']) * t['qty'] if t['side'] == "BUY" else (t['entry'] - exit_price) * t['qty']
        costs = (t['entry']*t['qty'] + exit_price*t['qty']) * CONFIG.TRANSACTION_COST_PCT
        self.conn.execute("UPDATE trades SET status = 'CLOSED', exit_price = ?, exit_time = ?, net_pnl = ?, costs = ?, exit_reason = ? WHERE id = ?", (exit_price, datetime.now(IST), gross-costs, costs, reason, trade_id))
        self.conn.commit()
        return gross-costs, t['ticker'], t['side'], costs

    def update_sl(self, tid, new_sl):
        self.conn.execute("UPDATE trades SET sl = ?, is_trailing = 1 WHERE id = ?", (new_sl, tid))
        self.conn.commit()

    def get_risk_pnl(self):
        today = str(datetime.now(IST).date())
        risk = self.conn.execute("SELECT risk_used FROM daily_stats WHERE date = ?", (today,)).fetchone()
        pnl = self.conn.execute("SELECT sum(net_pnl) as p FROM trades WHERE date(exit_time) = ?", (today,)).fetchone()
        cnt = self.conn.execute("SELECT count(*) as c FROM trades WHERE date(timestamp) = ?", (today,)).fetchone()
        return cnt['c'], (pnl['p'] or 0.0), (risk['risk_used'] if risk else 0.0)

    def add_risk(self, amt):
        today = str(datetime.now(IST).date())
        curr = self.conn.execute("SELECT risk_used FROM daily_stats WHERE date = ?", (today,)).fetchone()
        val = (curr['risk_used'] if curr else 0.0) + amt
        self.conn.execute("INSERT INTO daily_stats (date, risk_used) VALUES (?, ?) ON CONFLICT(date) DO UPDATE SET risk_used = ?", (today, val, val))
        self.conn.commit()

class TrendFollowing(ABC):
    def analyze(self, ticker: str, df: pd.DataFrame, config: SystemConfig) -> Optional[TradeSignal]:
        if df is None or len(df) < 50: return None
        try:
            price = float(df.iloc[-1]['Close'])
            if 'Volume' in df.columns:
                if (price * df['Volume'].rolling(20).mean().iloc[-1]) < config.MIN_TURNOVER: return None
            df.ta.vwap(append=True); df.ta.rsi(length=14, append=True); df.ta.ema(length=200, append=True); df.ta.atr(length=14, append=True)
            curr = df.iloc[-1]
            vwap, rsi, ema, atr = curr.get('VWAP_D', price), curr.get('RSI_14', 50), curr.get('EMA_200', price), curr.get('ATR_14', price*0.01)
            
            signal = None
            if price > ema and price > vwap and 55 < rsi < 70: signal = "BUY"
            elif price < ema and price < vwap and 30 < rsi < 45: signal = "SELL"
            if not signal: return None
            
            sl = price - (atr*2) if signal == "BUY" else price + (atr*2)
            tgt = price + (atr*4) if signal == "BUY" else price - (atr*4)
            qty = int(config.RISK_PER_TRADE / abs(price-sl))
            if qty < 1: return None
            
            return TradeSignal(ticker.replace('.NS',''), signal, price, qty, round(sl,2), round(tgt,2), 8.0, "TREND")
        except: return None

class TradingSystem:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.db = StateManager(self.config.DB_PATH)
        self.telegram = TelegramNotifier(self.config) 
        self.data = KiteWebSocketProvider(self.config) if self.config.DATA_SOURCE == "KITE" else YFinanceProvider(self.config)
        self.broker = ZerodhaKiteBroker(self.config) if self.config.EXECUTION_MODE == "LIVE" else SimulatedBroker(self.config)
        self.strategy = TrendFollowing()
        self.running = True
        self.db_lock = asyncio.Lock() 

    async def execution_gatekeeper(self, signal: TradeSignal):
        async with self.db_lock:
            if self.db.has_open_position(signal.ticker): return
            if len(self.db.get_open_trades()) >= self.config.MAX_POSITIONS: return
            _, _, r = self.db.get_risk_pnl()
            if (r + self.config.RISK_PER_TRADE) > self.config.MAX_DAILY_RISK: return
            
            fill = await self.broker.place_order(signal)
            self.db.log_trade(signal, fill)
            self.db.add_risk(self.config.RISK_PER_TRADE)
            
            # UI Update
            UI_STATE["top_picks"].insert(0, {"ticker": signal.ticker, "signal": signal.signal, "price": fill, "score": signal.score, "time": datetime.now(IST).strftime("%H:%M")})
            UI_STATE["top_picks"] = UI_STATE["top_picks"][:10]

        await self.telegram.send_entry(signal, fill)
        print(f"{Fore.GREEN}⚡ FILLED {signal.signal} {signal.ticker} @ {fill:.2f}{Style.RESET_ALL}")

    async def position_monitor(self):
        await self.telegram.send("🚀 *Titanium Online*")
        UI_STATE["bot_status"] = "RUNNING"
        
        while self.running:
            try:
                # 1. Update UI State
                async with self.db_lock:
                    cnt, pnl, risk = self.db.get_risk_pnl()
                    open_trades = self.db.get_open_trades()
                    
                UI_STATE["pnl"] = round(pnl, 2)
                UI_STATE["risk"] = round(risk, 2)
                UI_STATE["trades"] = cnt
                UI_STATE["active_positions"] = open_trades

                # 2. Logic (Simplified from V26 for brevity in merged code)
                if not open_trades: await asyncio.sleep(2); continue
                
                tickers = [f"{t['ticker']}.NS" for t in open_trades]
                bulk = await self.data.fetch_candles(tickers, "1m")
                if bulk is None: continue
                
                # Check Exits
                is_multi = isinstance(bulk.columns, pd.MultiIndex)
                for t in open_trades:
                    tk = t['ticker']
                    try:
                        df = bulk.xs(f"{tk}.NS", axis=1, level=0, drop_level=True) if is_multi else bulk
                        price = float(df.iloc[-1]['Close'])
                        
                        reason = None
                        if t['side'] == "BUY":
                            if price <= t['sl']: reason = "SL"
                            elif price >= t['target']: reason = "TGT"
                        else:
                            if price >= t['sl']: reason = "SL"
                            elif price <= t['target']: reason = "TGT"
                            
                        if reason:
                            fill = await self.broker.close_position(tk, t['qty'], t['side'], reason)
                            fill = fill if fill > 0 else price
                            async with self.db_lock: pnl, _, _, _ = self.db.close_trade(t['id'], fill, reason)
                            await self.telegram.send_exit(tk, t['side'], fill, pnl, reason)
                    except: pass
                    
            except Exception: pass
            await asyncio.sleep(self.config.MONITOR_INTERVAL)

    async def scanner_loop(self):
        while self.running:
            if not self.config.FORCE_RUN:
                now = datetime.now(IST)
                if now.weekday() > 4 or not (dtime(9,15) <= now.time() <= dtime(15,30)):
                    await asyncio.sleep(60); continue
            
            active = self.data.universe[:self.config.MAX_UNIVERSE_SIZE]
            chunks = [active[i:i+10] for i in range(0, len(active), 10)]
            for chunk in chunks:
                bulk = await self.data.fetch_candles(chunk, "5m")
                if bulk is None: continue
                is_multi = isinstance(bulk.columns, pd.MultiIndex)
                for t in chunk:
                    try:
                        df = bulk.xs(t, axis=1, level=0, drop_level=True) if is_multi else bulk
                        if not self.data.is_data_fresh(df, t): continue
                        sig = self.strategy.analyze(t, df, self.config)
                        if sig and sig.score >= 7.0: await self.execution_gatekeeper(sig)
                    except: pass
            await asyncio.sleep(self.config.SCAN_INTERVAL)

    async def main(self):
        await self.data.initialize_universe()
        await self.data.start()
        await asyncio.gather(self.scanner_loop(), self.position_monitor())

# ==========================================
# 7. DASHBOARD API & UI
# ==========================================

# Background worker to fetch generic market data for UI (non-critical)
async def dashboard_worker():
    while True:
        try:
            # Run blocking yfinance in executor
            loop = asyncio.get_running_loop()
            nifty200 = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ITC.NS", "SBIN.NS", "TATAMOTORS.NS"] # Shortened for demo
            
            data = await loop.run_in_executor(None, lambda: yf.download(nifty200, period="2d", interval="1d", group_by='ticker', progress=False))
            
            movers = []
            for t in nifty200:
                try:
                    df = data[t]
                    chg = round(((df['Close'].iloc[-1] - df['Close'].iloc[-2])/df['Close'].iloc[-2])*100, 2)
                    movers.append({"symbol": t.replace('.NS',''), "change": chg})
                except: pass
            
            movers.sort(key=lambda x: x['change'], reverse=True)
            UI_STATE["market_data"] = {
                "gainers": movers[:3], 
                "losers": movers[-3:],
                "indices": {"NIFTY": 24000, "BANKNIFTY": 52000} # Mock for speed
            }
        except: pass
        await asyncio.sleep(30)

@app.get("/api/bot")
def get_bot_state(): return UI_STATE

@app.get("/")
def serve_ui():
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Titanium V27 Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>body { background: #0f172a; color: white; font-family: sans-serif; }</style>
</head>
<body>
    <div id="root"></div>
    <script type="text/babel">
        const { useState, useEffect } = React;

        function App() {
            const [view, setView] = useState('BOT');
            const [data, setData] = useState(null);

            useEffect(() => {
                const timer = setInterval(() => {
                    fetch('/api/bot').then(r => r.json()).then(setData);
                }, 2000);
                return () => clearInterval(timer);
            }, []);

            if (!data) return <div className="p-10">Loading Titanium Core...</div>;

            return (
                <div className="flex h-screen">
                    {/* SIDEBAR */}
                    <div className="w-64 bg-slate-900 border-r border-slate-700 p-4">
                        <h1 className="text-2xl font-bold text-yellow-400 mb-8">Titanium<span className="text-white">V27</span></h1>
                        <div className={`p-3 rounded cursor-pointer mb-2 ${view==='BOT'?'bg-blue-600':'hover:bg-slate-800'}`} onClick={()=>setView('BOT')}>🤖 AI BOT</div>
                        <div className={`p-3 rounded cursor-pointer mb-2 ${view==='MARKET'?'bg-blue-600':'hover:bg-slate-800'}`} onClick={()=>setView('MARKET')}>📈 MARKET</div>
                    </div>

                    {/* CONTENT */}
                    <div className="flex-1 p-8 overflow-auto">
                        {view === 'BOT' && (
                            <div className="grid grid-cols-12 gap-6">
                                {/* STATS */}
                                <div className="col-span-12 grid grid-cols-4 gap-4 mb-4">
                                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                        <div className="text-gray-400 text-xs">STATUS</div>
                                        <div className="text-xl font-bold text-green-400">● {data.bot_status}</div>
                                    </div>
                                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                        <div className="text-gray-400 text-xs">NET P&L</div>
                                        <div className={`text-2xl font-bold ${data.pnl>=0?'text-green-400':'text-red-400'}`}>₹{data.pnl}</div>
                                    </div>
                                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                        <div className="text-gray-400 text-xs">RISK USED</div>
                                        <div className="text-xl font-bold text-yellow-400">₹{data.risk}</div>
                                    </div>
                                    <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
                                        <div className="text-gray-400 text-xs">TRADES</div>
                                        <div className="text-xl font-bold">{data.trades}</div>
                                    </div>
                                </div>

                                {/* ACTIVE POSITIONS */}
                                <div className="col-span-8 bg-slate-800 rounded-xl p-6 border border-slate-700">
                                    <h3 className="font-bold text-lg mb-4 text-blue-400">Active Positions</h3>
                                    {data.active_positions.length === 0 ? (
                                        <div className="text-gray-500 italic">No open trades...</div>
                                    ) : (
                                        <table className="w-full text-sm text-left">
                                            <thead><tr className="text-gray-500 border-b border-slate-600"><th>Ticker</th><th>Side</th><th>Entry</th><th>SL</th><th>TGT</th></tr></thead>
                                            <tbody>
                                                {data.active_positions.map(t => (
                                                    <tr key={t.id} className="border-b border-slate-700">
                                                        <td className="py-3 font-bold">{t.ticker}</td>
                                                        <td className={t.side==='BUY'?'text-green-400':'text-red-400'}>{t.side}</td>
                                                        <td>{t.entry}</td>
                                                        <td className="text-red-400">{t.sl}</td>
                                                        <td className="text-green-400">{t.target}</td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    )}
                                </div>

                                {/* TOP PICKS (SIGNALS) */}
                                <div className="col-span-4 bg-slate-800 rounded-xl p-6 border border-slate-700">
                                    <h3 className="font-bold text-lg mb-4 text-purple-400">Recent Signals</h3>
                                    <div className="space-y-3">
                                        {data.top_picks.map((s, i) => (
                                            <div key={i} className="flex justify-between items-center bg-slate-900 p-3 rounded border-l-4 border-purple-500">
                                                <div>
                                                    <div className="font-bold">{s.ticker}</div>
                                                    <div className="text-xs text-gray-400">{s.time} • Score: {s.score}</div>
                                                </div>
                                                <div className={`font-bold ${s.signal==='BUY'?'text-green-400':'text-red-400'}`}>{s.signal}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        )}

                        {view === 'MARKET' && (
                            <div className="text-center py-20 text-gray-500">
                                <h2 className="text-2xl">Market Overview</h2>
                                <p>Nifty: {data.market_data?.indices?.NIFTY}</p>
                                <div className="mt-4">Top Gainer: {data.market_data?.gainers?.[0]?.symbol}</div>
                            </div>
                        )}
                    </div>
                </div>
            );
        }
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<App />);
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

# --- ENTRY POINT ---
@app.on_event("startup")
async def startup_event():
    # 1. Start Titanium Bot (Background)
    sys = TradingSystem(CONFIG)
    asyncio.create_task(sys.main())
    
    # 2. Start Dashboard Data Worker (Background)
    asyncio.create_task(dashboard_worker())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--kite", action="store_true")
    args = parser.parse_args()
    
    CONFIG.FORCE_RUN = args.force
    if args.live:
        CONFIG.EXECUTION_MODE = "LIVE"
        CONFIG.DATA_SOURCE = "KITE"
    else:
        CONFIG.DATA_SOURCE = "KITE" if args.kite else "YFINANCE"

    print(f"{Fore.CYAN}🌐 Starting Dashboard at http://localhost:8000{Style.RESET_ALL}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
