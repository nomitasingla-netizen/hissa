# 📈 Sector Breakout Scanner

A local Streamlit dashboard that scans **US and Indian sector ETFs** to find
sectors that are **consolidating and ready for a bull breakout**, and warns you
when a sector has become **over-extended** so you can start exiting.

Scores are computed on **two timeframes — 4h and 1D** — and blended
(1D = 60%, 4h = 40%).

## What it measures

**Breakout Readiness (0–100, higher = better setup)** blends:
| Component | What it looks for |
|-----------|-------------------|
| MACD near zero | MACD line hovering near 0 and turning up (momentum reset) |
| Consolidation | Tight Bollinger bandwidth vs history (a "squeeze") |
| RSI | Neutral-bullish 45–60 zone, rising, room to run |
| Relative Strength | ETF outperforming its benchmark (SPY / NIFTY 50) |
| ADX | Low-but-rising ADX with +DI above −DI |
| Volume | Dry-up followed by a mild pickup |

**Exit / Over-extended (0–100, higher = trim/exit)** blends:
- RSI overbought (>70)
- Price stretched far above the 20-EMA (in ATR units)
- MACD stretched well above zero
- Close above the upper Bollinger band
- Very high ADX with momentum fading

### Signals
- 🟢 **STRONG BREAKOUT SETUP** / **BUILDING** — coiling, ready to run
- 🟡 **CONSOLIDATING** — not ready yet
- 🟠 **TRIM** — getting extended
- 🔴 **EXIT / TAKE PROFITS** — over-extended

## Run it

The app is pure Python + Streamlit, so it runs on **Windows, macOS, and Linux**.
You need **Python 3.9+**.

**Windows (PowerShell):**
```powershell
cd path\to\sector-breakout-scanner
pip install -r requirements.txt      # first time only
python -m streamlit run app.py       # or double-click run.bat
```

**macOS / Linux (Terminal):**
```bash
cd path/to/sector-breakout-scanner
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip3 install -r requirements.txt     # first time only
python3 -m streamlit run app.py      # or: ./run.sh
```

First time on macOS/Linux, make the launchers executable:
```bash
chmod +x run.sh start_scanner.sh stop_scanner.sh
./start_scanner.sh   # starts on port 8501 and opens your browser
./stop_scanner.sh    # stops it
```

Then open http://localhost:8501, pick a market (US / India / Both), optionally
edit the ETF lists in the sidebar, and click **Scan / Refresh**.

> The "load a CSV from your Downloads folder" picker in the SIP & Exit Plan tab
> automatically uses **your own** `~/Downloads` (or `%USERPROFILE%\Downloads` on
> Windows). Set the `SCANNER_DOWNLOADS_DIR` environment variable to point it
> elsewhere.

## Layout
```
core/
  etfs.py         default US + Indian ETF universes and benchmarks
  data.py         yfinance OHLCV fetch (1h resampled to 4h, plus 1d)
  indicators.py   MACD, RSI, ADX, ATR, Bollinger, relative strength
  scoring.py      breakout + exit scoring engine
config/
  us.json         editable US ETF/stock universe (auto-created on first run)
  india.json      editable Indian ETF/stock universe (auto-created on first run)
app.py            Streamlit dashboard
run.bat / run.sh                 double-click / terminal launcher
start_scanner.* / stop_scanner.* start & stop helpers (Windows + macOS/Linux)
```

## Configuring the ETF/stock universe

The tickers scanned for each market live in editable JSON config files that are
auto-created on first run:

```jsonc
// config/us.json
{
  "benchmark": "SPY",
  "etfs": {
    "XLK": "Technology",
    "AAPL": "Apple",        // add stocks the same way
    "QQQ": "Nasdaq 100"
  }
}
```

- Add or remove entries as `"TICKER": "Display name"`.
- Add **stocks** as well as ETFs (e.g. `"AAPL": "Apple"`, `"RELIANCE.NS": "Reliance"`).
- Indian symbols need the `.NS` (NSE) suffix; `benchmark` is used for relative strength.
- **Restart the app** after editing the files to load the changes.
- Delete a config file to regenerate it from the built-in defaults.
- You can also add tickers live from the sidebar without editing files.

## Notes
- Data comes from **yfinance** (free) and is cached for 15 minutes.
- Indian tickers use the `.NS` (NSE) suffix.
- **Not investment advice** — a decision-support tool. Always confirm with your
  own analysis and risk management.
