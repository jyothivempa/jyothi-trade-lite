import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from threading import Thread
import time as time_module
import random

app = FastAPI()

# --- CACHE (Pre-seeded to guarantee NO blank screens) ---
cache = {
    "top_picks": [],
    "nifty200_buys": [],
    "movers": {"gainers": [], "losers": []},
    "indices": {"NIFTY 50": 24300.00, "BANK NIFTY": 52100.00, "SENSEX": 79800.00},
    "market_mood": {"score": 50, "label": "Neutral", "color": "text-yellow-600"},
    "sectors": [],
    "research_calls": [
        {"symbol": "HDFCBANK", "type": "BUY", "entry": 1520, "target": 1580, "stoploss": 1490, "timeframe": "Intraday"},
        {"symbol": "RELIANCE", "type": "SELL", "entry": 2950, "target": 2900, "stoploss": 2980, "timeframe": "Intraday"}
    ],
    "screeners": {
        "volume_shockers": [{"symbol": "Loading...", "price": 0, "change": 0, "volume": "0k"}], 
        "near_52w_high": [{"symbol": "Loading...", "price": 0, "change": 0, "high52": 0}]
    },
    "last_updated": "Initializing..."
}

# --- NIFTY 200 LIST ---
NIFTY_200 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS",
    "LT.NS", "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS", "TATAMOTORS.NS", "SUNPHARMA.NS", "ADANIENT.NS", "NTPC.NS", "TITAN.NS", "KOTAKBANK.NS",
    "ONGC.NS", "AXISBANK.NS", "ADANIPORTS.NS", "M&M.NS", "ULTRACEMCO.NS", "WIPRO.NS", "POWERGRID.NS", "COALINDIA.NS", "BAJAJFINSV.NS", "ASIANPAINT.NS",
    "JSWSTEEL.NS", "NESTLEIND.NS", "TATASTEEL.NS", "GRASIM.NS", "TECHM.NS", "HINDALCO.NS", "CIPLA.NS", "EICHERMOT.NS", "SBILIFE.NS", "DRREDDY.NS",
    "BRITANNIA.NS", "BPCL.NS", "TATACONSUM.NS", "HEROMOTOCO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS", "UPL.NS",
    "ZOMATO.NS", "DLF.NS", "HAL.NS", "BEL.NS", "VBL.NS", "TRENT.NS", "JIOFIN.NS", "PIDILITIND.NS", "SIEMENS.NS", "IOC.NS", "RECLTD.NS", "PFC.NS", 
    "ADANIENSOL.NS", "GAIL.NS", "BANKBARODA.NS", "PNB.NS", "GODREJCP.NS", "HAVELLS.NS", "VEDL.NS", "INDIGO.NS", "TVSMOTOR.NS", "LODHA.NS", "CANBK.NS", 
    "UNIONBANK.NS", "IDFCFIRSTB.NS", "JINDALSTEL.NS", "ATGL.NS", "BHEL.NS", "IRFC.NS", "RVNL.NS", "MAZDOCK.NS", "COCHINSHIP.NS", "BHARATFORG.NS"
]

SECTOR_MAP = {
    "Nifty Bank": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "Nifty IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS"],
    "Nifty Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "Nifty FMCG": ["ITC.NS", "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Nifty Metal": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS"],
    "Nifty Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Nifty Energy": ["RELIANCE.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS"],
    "Nifty Realty": ["DLF.NS", "LODHA.NS", "GODREJPROP.NS"]
}

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>JyothiTradeLite Pro</title>
    
    <script crossorigin src="https://unpkg.com/react@18.2.0/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18.2.0/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>

    <style>
        :root { --bg-body: #f3f4f6; --bg-card: #ffffff; --text-main: #111827; --text-sub: #6b7280; --border: #e5e7eb; }
        body { background-color: var(--bg-body); color: var(--text-main); font-family: 'Inter', sans-serif; overflow: hidden; }
        
        .sidebar { width: 260px; background: var(--bg-card); border-right: 1px solid var(--border); display: flex; flex-direction: column; height: 100vh; box-shadow: 4px 0 24px rgba(0,0,0,0.02); }
        .main-content { flex: 1; display: flex; flex-direction: column; height: 100vh; }
        
        .nav-item { display: flex; align-items: center; gap: 12px; padding: 12px 20px; cursor: pointer; color: var(--text-sub); font-weight: 600; margin: 4px 8px; border-radius: 8px; transition: all 0.2s; }
        .nav-item:hover { background: #f1f5f9; color: #0f172a; }
        
        .nav-item.active-TERMINAL { background: #eff6ff; color: #2563eb; border-left: 4px solid #2563eb; }
        .nav-item.active-RESEARCH { background: #ecfdf5; color: #059669; border-left: 4px solid #059669; }
        .nav-item.active-SCREENERS { background: #f5f3ff; color: #7c3aed; border-left: 4px solid #7c3aed; }
        .nav-item.active-SECTORS { background: #ecfeff; color: #0891b2; border-left: 4px solid #0891b2; }
        .nav-item.active-HEATMAP { background: #fffbeb; color: #d97706; border-left: 4px solid #d97706; }
        .nav-item.active-SCANNER { background: #fdf2f8; color: #db2777; border-left: 4px solid #db2777; }

        .card { background: var(--bg-card); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
        .scroll-area { overflow-y: auto; scrollbar-width: thin; }
        .header { height: 64px; background: white; border-bottom: 1px solid var(--border); display: flex; align-items: center; justify-content: space-between; padding: 0 24px; }
        .ticker-wrap { background: #0f172a; color: white; font-family: monospace; padding: 8px; white-space: nowrap; overflow: hidden; font-size: 13px; }
        
        @keyframes pulse { 0% { opacity: 1; transform: scale(1); } 50% { opacity: 0.5; transform: scale(1.2); } 100% { opacity: 1; transform: scale(1); } }
        .live-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: pulse 2s infinite; }
        
        .tech-meter { height: 6px; background: #e2e8f0; border-radius: 3px; overflow: hidden; margin-top: 6px; }
    </style>
</head>
<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useRef } = React;
        
        // Icons
        const Icon = ({name}) => <span className="text-lg opacity-70">
            {name==='Graph'?'📈':name==='Map'?'🗺️':name==='Target'?'🎯':name==='Zap'?'⚡':name==='Pie'?'🥧':name==='Filter'?'🔍':''}
        </span>;

        function App() {
            const [view, setView] = useState('TERMINAL');
            const [stockList, setStockList] = useState([]);
            const [selectedStock, setSelectedStock] = useState('RELIANCE');
            
            const [data, setData] = useState({
                indices: {}, 
                movers: {gainers:[], losers:[]}, 
                market_mood: {score:50, label:'Neutral'},
                nifty200_buys: [],
                research_calls: [],
                screeners: {volume_shockers:[], near_52w_high:[]}
            });
            
            const [priceData, setPriceData] = useState(null);
            const chartRef = useRef(null);
            const chartInstance = useRef(null);

            // 1. Initial Data Load
            useEffect(() => {
                fetch('/api/list').then(r=>r.json()).then(setStockList);
                
                const refreshData = () => {
                    fetch('/api/dashboard').then(r=>r.json()).then(setData);
                };
                refreshData();
                setInterval(refreshData, 5000); // 5s Global Refresh
            }, []);

            // 2. Live Price Feed
            useEffect(() => {
                if (view !== 'TERMINAL') return;
                const fetchLive = () => {
                    fetch(`/api/live/${selectedStock}`)
                        .then(r=>r.json())
                        .then(d => { if(d && d.current) setPriceData(d); })
                        .catch(e => console.log("Live feed waiting..."));
                };
                fetchLive();
                const timer = setInterval(fetchLive, 3000);
                return () => clearInterval(timer);
            }, [selectedStock, view]);

            // 3. Charting
            useEffect(() => {
                if (view !== 'TERMINAL' || !priceData || !priceData.history || !chartRef.current) return;
                const ctx = chartRef.current.getContext('2d');
                if (chartInstance.current) chartInstance.current.destroy();
                
                const isUp = priceData.change >= 0;
                
                chartInstance.current = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: priceData.history.map(d => d.time),
                        datasets: [{
                            data: priceData.history.map(d => d.price),
                            borderColor: isUp ? '#16a34a' : '#dc2626',
                            backgroundColor: (ctx) => {
                                const grad = ctx.chart.ctx.createLinearGradient(0,0,0,300);
                                grad.addColorStop(0, isUp ? 'rgba(22, 163, 74, 0.1)' : 'rgba(220, 38, 38, 0.1)');
                                grad.addColorStop(1, 'rgba(255,255,255,0)');
                                return grad;
                            },
                            borderWidth: 2, fill: true, pointRadius: 0, tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        plugins: { legend: { display: false } },
                        scales: { x: { display: false }, y: { position: 'right' } },
                        animation: false
                    }
                });
            }, [priceData, view]);

            const getTechColor = (score) => {
                if (score >= 70) return 'bg-green-500';
                if (score >= 40) return 'bg-yellow-400';
                return 'bg-red-500';
            };

            return (
                <div className="flex h-screen w-full text-gray-800">
                    <div className="sidebar">
                        <div className="h-[64px] flex items-center px-6 border-b border-gray-200">
                            <h1 className="text-xl font-extrabold text-gray-900 tracking-tight">JyothiTrade<span className="text-red-600">Lite</span></h1>
                        </div>
                        <div className="flex-1 py-4">
                            <div className={`nav-item ${view==='TERMINAL'?'active-TERMINAL':''}`} onClick={()=>setView('TERMINAL')}><Icon name="Graph"/><span>Terminal</span></div>
                            <div className={`nav-item ${view==='RESEARCH'?'active-RESEARCH':''}`} onClick={()=>setView('RESEARCH')}><Icon name="Target"/><span>Smart Research</span></div>
                            <div className={`nav-item ${view==='SCREENERS'?'active-SCREENERS':''}`} onClick={()=>setView('SCREENERS')}><Icon name="Filter"/><span>Pro Screeners</span></div>
                            <div className={`nav-item ${view==='SECTORS'?'active-SECTORS':''}`} onClick={()=>setView('SECTORS')}><Icon name="Pie"/><span>Sector Watch</span></div>
                            <div className={`nav-item ${view==='HEATMAP'?'active-HEATMAP':''}`} onClick={()=>setView('HEATMAP')}><Icon name="Map"/><span>Heatmap</span></div>
                            <div className={`nav-item ${view==='SCANNER'?'active-SCANNER':''}`} onClick={()=>setView('SCANNER')}><Icon name="Zap"/><span>Nifty 200 Gems</span></div>
                        </div>
                        
                        <div className="p-6 border-t border-gray-200 bg-gray-50">
                            <div className="text-xs font-bold text-gray-400 uppercase mb-1">Market Breadth</div>
                            <div className="flex justify-between text-xs font-bold">
                                <span className="text-green-600">{data.market_breadth?.advances || 0} Adv</span>
                                <span className="text-red-600">{data.market_breadth?.declines || 0} Dec</span>
                            </div>
                            <div className="h-1.5 bg-gray-200 rounded-full mt-2 overflow-hidden">
                                <div className="h-full bg-green-500" style={{width: `${(data.market_breadth?.advances/200)*100}%`}}></div>
                            </div>
                        </div>
                    </div>

                    <div className="main-content">
                        <div className="ticker-wrap">
                            ⚡ NIFTY 50: {data.indices?.["NIFTY 50"]} | BANK NIFTY: {data.indices?.["BANK NIFTY"]} | STATUS: {data.last_updated}
                        </div>
                        <div className="header">
                            <div className="text-sm font-bold text-gray-500">VIEW: <span className="text-red-600 uppercase">{view}</span></div>
                            <select className="bg-gray-50 border border-gray-200 text-gray-800 font-bold rounded px-3 py-2 text-sm outline-none" value={selectedStock} onChange={(e)=>setSelectedStock(e.target.value)}>
                                {stockList.map(s=><option key={s} value={s}>{s}</option>)}
                            </select>
                        </div>

                        <div className="flex-1 p-6 overflow-hidden bg-gray-50">
                            
                            {view === 'TERMINAL' && (
                                <div className="grid grid-cols-12 gap-6 h-full">
                                    <div className="col-span-9 flex flex-col gap-6 h-full">
                                        <div className="card p-6 flex-1 flex flex-col relative bg-white">
                                            {priceData ? (
                                                <>
                                                    <div className="flex justify-between items-start mb-4">
                                                        <div>
                                                            <h2 className="text-3xl font-bold text-gray-900">{selectedStock}</h2>
                                                            <div className="text-sm text-gray-500 mt-1 flex items-center gap-2">
                                                                <div className="live-dot"></div> LIVE • VOL: {priceData.volume}
                                                            </div>
                                                        </div>
                                                        <div className="text-right">
                                                            <div className="text-4xl font-bold text-gray-900">₹{priceData.current}</div>
                                                            <div className={`text-lg font-bold ${priceData.change>=0?'text-green-600':'text-red-600'}`}>{priceData.change}%</div>
                                                        </div>
                                                    </div>
                                                    <div className="flex-1 relative w-full"><canvas ref={chartRef}></canvas></div>
                                                </>
                                            ) : <div className="flex-1 flex items-center justify-center text-gray-400">Connecting to NSE...</div>}
                                        </div>
                                        
                                        <div className="card p-4 border-l-4 border-blue-500 bg-white">
                                            <div className="flex justify-between text-xs font-bold text-gray-500 uppercase mb-1"><span>Technical Strength</span><span>{priceData?.tech_score || 0}%</span></div>
                                            <div className="tech-meter"><div className={`tech-fill ${getTechColor(priceData?.tech_score||0)}`} style={{width: `${priceData?.tech_score||0}%`}}></div></div>
                                            <div className="text-xs font-bold mt-1 text-right text-gray-700">{priceData?.tech_rating || 'NEUTRAL'}</div>
                                        </div>
                                    </div>
                                    
                                    <div className="col-span-3 card p-4 h-full flex flex-col bg-white">
                                        <div className="text-xs font-bold text-gray-400 uppercase mb-4">Market Movers</div>
                                        <div className="flex-1 scroll-area">
                                            <div className="mb-6">
                                                <div className="text-xs font-bold text-green-600 bg-green-50 px-2 py-1 rounded mb-2 inline-block">TOP GAINERS</div>
                                                {data.movers?.gainers.map(s=><div key={s.symbol} onClick={()=>setSelectedStock(s.symbol)} className="flex justify-between p-2 hover:bg-gray-50 cursor-pointer rounded text-sm border-b border-gray-50"><span className="font-semibold text-gray-700">{s.symbol}</span><span className="font-bold text-green-600">+{s.change}%</span></div>)}
                                            </div>
                                            <div>
                                                <div className="text-xs font-bold text-red-600 bg-red-50 px-2 py-1 rounded mb-2 inline-block">TOP LOSERS</div>
                                                {data.movers?.losers.map(s=><div key={s.symbol} onClick={()=>setSelectedStock(s.symbol)} className="flex justify-between p-2 hover:bg-gray-50 cursor-pointer rounded text-sm border-b border-gray-50"><span className="font-semibold text-gray-700">{s.symbol}</span><span className="font-bold text-red-600">{s.change}%</span></div>)}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {view === 'RESEARCH' && (
                                <div className="h-full scroll-area">
                                    <div className="grid grid-cols-3 gap-6">
                                        {data.research_calls?.length > 0 ? data.research_calls.map((c,i) => (
                                            <div key={i} onClick={()=>{setSelectedStock(c.symbol); setView('TERMINAL')}} className="card p-6 cursor-pointer hover:shadow-lg border-t-4 border-teal-500 bg-white">
                                                <div className="flex justify-between mb-4"><h3 className="font-bold text-lg">{c.symbol}</h3><span className={`px-2 py-1 text-xs rounded text-white ${c.type==='BUY'?'bg-green-600':'bg-red-600'}`}>{c.type}</span></div>
                                                <div className="grid grid-cols-3 text-center text-sm">
                                                    <div><div className="text-gray-400 text-xs">ENTRY</div><div className="font-bold">₹{c.entry}</div></div>
                                                    <div><div className="text-gray-400 text-xs">TARGET</div><div className="font-bold text-green-600">₹{c.target}</div></div>
                                                    <div><div className="text-gray-400 text-xs">STOP</div><div className="font-bold text-red-600">₹{c.stoploss}</div></div>
                                                </div>
                                            </div>
                                        )) : <div className="col-span-3 text-center py-20 text-gray-400">Generating Calls...</div>}
                                    </div>
                                </div>
                            )}

                            {view === 'SCREENERS' && (
                                <div className="grid grid-cols-2 gap-6 h-full scroll-area">
                                    <div className="card bg-white">
                                        <div className="p-4 border-b font-bold text-purple-600">⚡ Volume Shockers (High Activity)</div>
                                        <div>
                                            {data.screeners?.volume_shockers.map(s=><div key={s.symbol} onClick={()=>{setSelectedStock(s.symbol); setView('TERMINAL')}} className="flex justify-between p-3 border-b hover:bg-gray-50 cursor-pointer"><div><div className="font-bold text-gray-700">{s.symbol}</div><div className="text-xs text-gray-400">Vol: {s.volume}</div></div><div className="text-right font-bold">₹{s.price}</div></div>)}
                                        </div>
                                    </div>
                                    <div className="card bg-white">
                                        <div className="p-4 border-b font-bold text-blue-600">📈 Near 52W High (Breakout)</div>
                                        <div>
                                            {data.screeners?.near_52w_high.map(s=><div key={s.symbol} onClick={()=>{setSelectedStock(s.symbol); setView('TERMINAL')}} className="flex justify-between p-3 border-b hover:bg-gray-50 cursor-pointer"><div><div className="font-bold text-gray-700">{s.symbol}</div><div className="text-xs text-gray-400">High: {s.high52}</div></div><div className="text-right font-bold">₹{s.price}</div></div>)}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {view === 'SCANNER' && (
                                <div className="h-full scroll-area">
                                    <div className="p-6 bg-gradient-to-r from-pink-600 to-rose-600 rounded-xl text-white mb-6 shadow-lg"><h2 className="text-2xl font-bold">💎 Nifty 200 Gems</h2><p className="text-sm opacity-90">Stocks in Uptrend (Above 200 SMA) + Value Dip (RSI &lt; 40)</p></div>
                                    <div className="grid grid-cols-3 gap-6">
                                        {data.nifty200_buys?.length > 0 ? data.nifty200_buys.map((s, i) => (
                                            <div key={s.symbol} onClick={()=>{setSelectedStock(s.symbol); setView('TERMINAL')}} className="card p-6 cursor-pointer hover:shadow-md bg-white border-l-4 border-pink-500">
                                                <div className="flex justify-between items-center mb-4"><span className="bg-gray-100 text-gray-600 px-2 py-1 rounded text-xs font-bold">Gem #{i+1}</span><span className="text-green-600 font-bold">RSI {s.rsi}</span></div>
                                                <h3 className="text-xl font-bold text-gray-800 mb-2">{s.symbol}</h3>
                                                <div className="flex justify-between text-sm pt-2 border-t"><div className="text-gray-500">Price<br/><span className="text-gray-900 font-bold">₹{s.price}</span></div><div className="text-right text-gray-500">200 SMA<br/><span className="text-gray-900 font-bold">₹{s.sma200}</span></div></div>
                                            </div>
                                        )) : <div className="col-span-3 text-center py-20 text-gray-400">Scanning for Gems...</div>}
                                    </div>
                                </div>
                            )}

                            {view === 'HEATMAP' && (
                                <div className="h-full scroll-area grid grid-cols-5 gap-3">
                                    {data.movers?.gainers.concat(data.movers?.losers).map(s => (
                                        <div key={s.symbol} onClick={()=>{setSelectedStock(s.symbol); setView('TERMINAL')}} className={`p-4 rounded-xl cursor-pointer h-24 flex flex-col justify-between text-white shadow-sm ${s.change>=0?'bg-green-500':'bg-red-500'}`}>
                                            <div className="font-bold text-sm truncate">{s.symbol}</div>
                                            <div className="text-xl font-bold">{s.change}%</div>
                                        </div>
                                    ))}
                                </div>
                            )}
                            
                            {view === 'SECTORS' && (
                                <div className="h-full scroll-area grid grid-cols-3 gap-6">
                                    {data.sectors?.map(s => (
                                        <div key={s.symbol} className={`card p-6 border-t-4 bg-white ${s.change>=0 ? 'border-green-500' : 'border-red-500'}`}>
                                            <h3 className="text-lg font-bold text-gray-800">{s.name}</h3>
                                            <div className={`text-2xl font-bold ${s.change>=0?'text-green-600':'text-red-600'}`}>{s.change>0?'+':''}{s.change}%</div>
                                        </div>
                                    ))}
                                </div>
                            )}

                        </div>
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

# --- BACKEND LOGIC (BATCH PROCESSING) ---
def get_techs(series):
    try:
        delta = series.diff()
        gain = (delta.where(delta>0, 0)).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.where(delta<0, 0)).ewm(alpha=1/14, adjust=False).mean()
        rsi = 100 - (100/(1+(gain/loss)))
        return rsi
    except: return pd.Series([50]*len(series))

def get_status():
    tz = pytz.timezone('Asia/Kolkata'); now = datetime.now(tz)
    if now.weekday()>=5: return "CLOSED"
    return "OPEN" if time(9,15) <= now.time() <= time(15,30) else "CLOSED"

@app.get("/")
def home(): return HTMLResponse(content=html_content)

@app.get("/api/list")
def list_api(): return sorted([s.replace('.NS','') for s in NIFTY_200])

@app.get("/api/dashboard")
def dashboard(): return cache

@app.get("/api/live/{s}")
def live(s:str):
    if not s.endswith('.NS'): s+=".NS"
    try:
        df = yf.download(s, period="5d", interval="1m", progress=False)
        if df.empty: return {}
        if isinstance(df.columns, pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        
        df_c = df.tail(100)
        curr = float(df_c['Close'].iloc[-1])
        op = float(df_c['Open'].iloc[0])
        chg = round(((curr-op)/op)*100, 2)
        
        rsi_series = get_techs(df['Close'])
        rsi = round(rsi_series.iloc[-1], 2)
        sma200 = df['Close'].rolling(200).mean().iloc[-1] if len(df)>200 else curr
        
        score = 50
        if rsi < 30: score += 20
        elif rsi > 70: score -= 20
        if curr > sma200: score += 10
        
        rating = "BUY" if score > 60 else ("SELL" if score < 40 else "HOLD")

        hist = [{"time":str(r.name)[11:16], "price":float(r['Close'])} for i,r in df_c.iterrows()]
        
        return {
            "current": round(curr,2), "change": chg, "history": hist, "open": round(op,2), 
            "volume": f"{int(df_c['Volume'].sum()/1000)}k", 
            "rsi": rsi, "macd_signal": "BULL" if chg>0 else "BEAR",
            "sma200": round(sma200, 2), "tech_score": score, "tech_rating": rating
        }
    except: return {}

# --- BACKGROUND WORKER (THE FIX) ---
def worker():
    while True:
        try:
            print("🔄 Updating Batch Data...")
            
            # 1. BATCH DOWNLOAD (1y Daily Data for all 200 stocks)
            # This ensures we have data for screeners, sectors, and research
            data = yf.download(NIFTY_200, period="1y", interval="1d", group_by='ticker', progress=False)
            
            movers_list = []
            vol_shockers = []
            near_52h = []
            research = []
            gems = []
            sectors_data = {k: [] for k in SECTOR_MAP.keys()}
            
            # 2. PROCESS EACH STOCK
            for t in NIFTY_200:
                try:
                    df = data[t].dropna()
                    if df.empty: continue
                    
                    curr = float(df['Close'].iloc[-1])
                    prev = float(df['Close'].iloc[-2])
                    change = round(((curr-prev)/prev)*100, 2)
                    vol = float(df['Volume'].iloc[-1])
                    avg_vol = float(df['Volume'].tail(20).mean())
                    high52 = float(df['High'].max())
                    sma200 = float(df['Close'].rolling(200).mean().iloc[-1])
                    
                    # RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta>0, 0)).ewm(alpha=1/14).mean()
                    loss = (-delta.where(delta<0, 0)).ewm(alpha=1/14).mean()
                    rsi = 100 - (100/(1+(gain/loss))).iloc[-1]

                    # Movers
                    movers_list.append({"symbol":t.replace('.NS',''), "change":change})
                    
                    # Screeners
                    if vol > 1.5 * avg_vol: 
                        vol_shockers.append({"symbol":t.replace('.NS',''), "price":round(curr,2), "change":change, "volume":f"{int(vol/1000)}k"})
                    if curr > 0.95 * high52:
                        near_52h.append({"symbol":t.replace('.NS',''), "price":round(curr,2), "change":change, "high52":round(high52,2)})
                        
                    # Research (Algo)
                    if rsi < 30: 
                        research.append({"symbol":t.replace('.NS',''), "type":"BUY", "timeframe":"Intraday", "entry":round(curr,2), "target":round(curr*1.02,2), "stoploss":round(curr*0.98,2)})
                    elif rsi > 70:
                        research.append({"symbol":t.replace('.NS',''), "type":"SELL", "timeframe":"Intraday", "entry":round(curr,2), "target":round(curr*0.98,2), "stoploss":round(curr*1.02,2)})
                        
                    # Gems (Uptrend + Dip)
                    if curr > sma200 and rsi < 45:
                        gems.append({"symbol":t.replace('.NS',''), "price":round(curr,2), "rsi":round(rsi,1), "sma200":round(sma200,2), "upside": (45-rsi)*2})

                    # Sectors
                    for sec, stocks in SECTOR_MAP.items():
                        if t in stocks: sectors_data[sec].append(change)
                        
                except: pass

            # 3. AGGREGATE & UPDATE CACHE
            movers_list.sort(key=lambda x: x['change'], reverse=True)
            gems.sort(key=lambda x: x['upside'], reverse=True)
            
            # Calculate Sector Performance
            final_sectors = []
            for sec, chgs in sectors_data.items():
                if chgs: final_sectors.append({"symbol":sec, "name":sec, "change": round(sum(chgs)/len(chgs), 2)})
            
            # Update Global Cache
            cache["movers"] = {"gainers": movers_list[:5], "losers": movers_list[-5:][::-1]}
            cache["screeners"]["volume_shockers"] = vol_shockers[:5]
            cache["screeners"]["near_52w_high"] = near_52h[:5]
            cache["research_calls"] = research[:6]
            cache["nifty200_buys"] = gems[:6]
            cache["sectors"] = final_sectors
            cache["last_updated"] = datetime.now().strftime("%H:%M:%S")
            
            print("✅ Cache Updated Successfully")
            
        except Exception as e: print(f"Error in worker: {e}")
        
        time_module.sleep(15) # Refresh every 15 seconds

Thread(target=worker, daemon=True).start()

if __name__ == "__main__":
    print("Running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)