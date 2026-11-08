# 📈 HengSeng & KRX Sentiment Price Prediction Agent

An end-to-end quantitative AI agent and sentiment arbitrage pipeline that predicts asset prices and directional movements at $n+1$. Built for the **CrowdWisdomTrading AI Agents Developer Assessment**, this system scrapes multi-source retail social sentiment via **Apify**, ingests institutional news via **Exa.com**, aligns features to UTC market candles, computes technical support/resistance levels, and trains out-of-sample **LightGBM** models presented through both a CLI report and an interactive **Streamlit** dashboard.

---


| Requirement | Implementation Details | Status |
| :--- | :--- | :--- |
| **5 Assets per Exchange (HKEX & KRX)** | 5 liquid HKEX equities (`0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`, `1211.HK`) and 5 KRX equities (`005930.KS`, `000660.KS`, `035420.KS`, `035720.KS`, `373220.KS`). | ✅ Complete |
| **Min. 2 Social Sources per Exchange (Apify)** | **HKEX:** Telegram HK Stocks (`t.me/s/hkstocks`) + AAStocks Community.<br>**KRX:** Naver Finance Stock Discussion Boards (`종목토론실`) + Telegram Korea Stocks. | ✅ Complete |
| **Recent News Ingestion (Exa.com)** | Queries Exa.com API for recent financial news, earnings, and market catalysts over the trailing 10-day window. | ✅ Complete |
| **Engagement & Follower Extraction (UTC)** | Extracts post text, strict UTC timestamps, `likes`, `reposts`, `views`, and `author_followers` and weights sentiment logarithmically. | ✅ Complete |
| **Per-Ticker Agent Extraction** | Extracts Ticker, Asset Name, Current Price, Weighted Sentiment, $n+1$ Direction, Tactical Position Recommendation, and Support/Resistance ($S_1, S_2, R_1, R_2$). | ✅ Complete |
| **Sorted Visual Report** | Outputs both a `Rich` CLI table and an interactive **Streamlit + Plotly** web dashboard sorted by **Mentions** and **Sentiment Score** per exchange. | ✅ Complete |
| **$n+1$ Price Prediction & Validation** | Dual **LightGBM** architecture (`LGBMClassifier` for direction + `LGBMRegressor` for $n+1$ price) evaluated using walk-forward `TimeSeriesSplit`. | ✅ Complete |

---

## 🏗️ System Architecture

```text
                      ┌──────────────────────────┐
                      │   5 HKEX + 5 KRX Assets  │
                      │      (src/config.py)     │
                      └────────────┬─────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Apify Scraper   │      │   Exa.com API    │      │  Yahoo Finance   │
│ 2 Social Sources │      │ Institutional    │      │  Hourly OHLCV    │
│  per Exchange    │      │   News Stream    │      │  Price Candles   │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   ▼
                     ┌───────────────────────────┐
                     │  UTC Temporal Alignment   │
                     │   & Feature Engineering   │
                     │ (Engagement + Followers)  │
                     └─────────────┬─────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌──────────────────────────┐                       ┌──────────────────────────┐
│    AI Decision Agent     │                       │   ML Forecasting Node    │
│ - Pivot Points (S1/R1)   │                       │ - TimeSeriesSplit CV     │
│ - Sentiment Momentum     │                       │ - LightGBM Classifier    │
│ - Position Sizing Logic  │                       │ - LightGBM Regressor     │
└────────────┬─────────────┘                       └────────────┬─────────────┘
             │                                                  │
             └─────────────────────────┬────────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   Visual Reporting Layer  │
                         │ - Rich CLI Arbitrage Table│
                         │ - Streamlit + Plotly UI   │
                         └───────────────────────────┘
```

---

## 📊 Asset Universe & Data Sources

### 1. Hong Kong Exchange (`HKEX`)
* **Assets Analyzed:** Tencent (`0700.HK`), Alibaba (`9988.HK`), Meituan (`3690.HK`), Xiaomi (`1810.HK`), BYD (`1211.HK`).
* **Social Source 1:** Telegram HK Equities Channel (`https://t.me/s/hkstocks`).
* **Social Source 2:** AAStocks HK Retail Quote & Community Stream (`http://www.aastocks.com`).
* **News Source:** Exa.com Search API.

### 2. Korea Exchange (`KRX`)
* **Assets Analyzed:** Samsung Electronics (`005930.KS`), SK Hynix (`000660.KS`), NAVER (`035420.KS`), Kakao (`035720.KS`), LG Energy Solution (`373220.KS`).
* **Social Source 1:** Naver Finance Stock Discussion Boards / 종목토론실 (`https://finance.naver.com/item/board.naver`).
* **Social Source 2:** Telegram South Korea Equities Channel (`https://t.me/s/Stock_Korea`).
* **News Source:** Exa.com Search API.

---

## 🧠 Methodology & Quantitative Design

### 1. Engagement & Follower-Weighted Sentiment Scoring
Not all social posts carry equal market impact. In `src/processing/sentiment_analyzer.py`, each post's raw polarity $S_i \in [-1.0, 1.0]$ (computed via a hybrid of `TextBlob` and a bilingual English/Korean/Chinese financial lexicon) is weighted logarithmically by its engagement metrics and author reach:

$$W_i = \ln\left(1 + \text{Likes}_i + 2 \cdot \text{Reposts}_i + 0.05 \cdot \text{Views}_i + 0.01 \cdot \text{Followers}_i\right) + 1$$

$$\text{EWSS}_t = \frac{\sum_{i \in t} S_i \cdot W_i}{\sum_{i \in t} W_i}$$

Institutional news articles ingested via Exa.com receive a high baseline weight ($W_{\text{news}} = 5.0$) to reflect institutional significance.

### 2. Asynchronous UTC Candle Alignment (`pd.merge_asof`)
Asian exchanges such as `HKEX` open on the half-hour (`01:30 UTC`), whereas standard hourly resampling buckets land on the exact hour (`01:00 UTC`). To prevent dropped records and lookahead bias, `src/processing/technical_features.py` standardizes all timestamps to UTC and uses `pd.merge_asof` to map resampled sentiment blocks onto exact market trading candles, generating technical features:
* `hourly_sentiment_score`: Blended engagement-weighted sentiment and immediate price reaction.
* `rolling_sentiment_3h`: 3-bar moving average of sentiment momentum.
* `return_1h`: Hourly logarithmic/percentage price return.
* `vol_spike`: Current bar volume relative to its 10-bar rolling average.

### 3. Agent Support/Resistance & Position Sizing
In `src/agent/technical_levels.py`, the agent computes Standard Floor Pivot Points over a trailing 20-bar window:

$$P = \frac{H + L + C}{3}, \quad S_1 = (2 \cdot P) - H, \quad R_1 = (2 \cdot P) - L$$

$$S_2 = P - (H - L), \quad R_2 = P + (H - L)$$

The decision agent combines $\text{EWSS}_t$ with proximity to $S_1$ and $R_1$ to output actionable trade recommendations:
* **Strong Buy / Long (Value Bounce):** Bullish sentiment ($> 0.15$) while trading near or below $S_1$.
* **Tactical Long (Momentum Breakout):** Bullish sentiment ($> 0.15$) breaking above $R_1$.
* **Long (Accumulate):** Bullish sentiment within the normal trading range.
* **Strong Short (Resistance Rejection):** Bearish sentiment ($< -0.15$) near or above $R_1$.
* **Underweight / Reduce Exposure:** Bearish sentiment within the normal range.
* **Hold / Neutral:** Low sentiment conviction ($[-0.15, 0.15]$).

### 4. Out-of-Sample $n+1$ Price & Direction Prediction
In `src/models/predictor.py`, target labels are shifted backward by one bar (`df['Close'].shift(-1)`) to represent the unknown future state $n+1$:
* **Validation:** Uses `sklearn.model_selection.TimeSeriesSplit(n_splits=3)` to enforce chronological walk-forward training and testing (zero lookahead leakage).
* **Classification (`LGBMClassifier`):** Predicts binary $n+1$ price direction (`UP 🟢` vs. `DOWN 🔴`), reporting out-of-sample directional accuracy.
* **Regression (`LGBMRegressor`):** Predicts expected $n+1$ percentage return $\hat{r}_{t+1}$ to compute the projected price:

$$\hat{P}_{t+1} = P_t \cdot \left(1 + \hat{r}_{t+1}\right)$$

---

## 📂 Project Directory Structure

```text
heng-seng-sentiment-agent/
├── data/
│   ├── raw/                        # Raw JSON payloads from Apify & Exa.com per ticker
│   └── processed/                  # Aligned hourly OHLCV + Sentiment feature CSVs
├── src/
│   ├── config.py                   # Asset definitions, tickers, and multilingual keywords
│   ├── ingestion/
│   │   ├── apify_collector.py      # Multi-source Apify scraper (Telegram, Naver, AAStocks)
│   │   ├── exa_collector.py        # Exa.com news ingestion client
│   │   └── market_data.py          # Yahoo Finance UTC hourly bar fetcher
│   ├── processing/
│   │   ├── sentiment_analyzer.py   # Engagement + follower weighting & lexicon NLP
│   │   └── technical_features.py   # UTC merge_asof alignment & rolling indicators
│   ├── agent/
│   │   └── technical_levels.py     # Pivot S/R calculator & position recommendation engine
│   ├── models/
│   │   └── predictor.py            # LightGBM Classifier/Regressor & TimeSeriesSplit CV
│   └── reporting/
│       └── generate_dashboard.py   # Terminal report utility
├── app.py                          # Interactive Streamlit + Plotly web application
├── run_pipeline.py                 # End-to-end master pipeline orchestrator
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── README.md                       # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository & Create a Virtual Environment
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd heng-seng-sentiment-agent

# Create and activate virtual environment
python -m venv venv
# Windows PowerShell:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit plotly textblob
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory (use `.env.example` as a reference):
```env
APIFY_API_TOKEN=your_apify_api_token_here
EXA_API_KEY=your_exa_api_key_here
```

> **Note on Apify Actor Permissions:** This pipeline uses the official `apify/web-scraper` actor (`moJRLRc85AitArpNN`). Ensure full account permissions are approved for this actor in your Apify Console prior to running.

---

## 🚀 Running the Pipeline & Interactive UI

### Step 1: Run the Master Data & ML Pipeline
Executes multi-source scraping, news ingestion, feature alignment, walk-forward validation, and prints the sorted arbitrage tables directly in the terminal:
```bash
python run_pipeline.py
```

### Step 2: Launch the Interactive Web Dashboard
Starts the local **Streamlit** server with live exchange matrices sorted by mention volume and sentiment score, plus interactive dual-axis **Plotly** price-vs-sentiment charts:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🔑 Reproducibility & Actor Reference
* **Apify Actor ID Used:** `apify/web-scraper` (`moJRLRc85AitArpNN`) configured with custom JavaScript `pageFunction` selectors for Telegram Web (`t.me/s/*`), Naver Finance Discussion Boards (`finance.naver.com/item/board.naver`), and AAStocks (`aastocks.com`).
* **Exa SDK Endpoint:** `exa.search_and_contents()` using keyword/neural search with text extraction.