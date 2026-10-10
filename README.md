# AAPL Financial News Sentiment and GRU–CNN GAN Forecasting

**An end-to-end research prototype combining financial news, FinBERT sentiment, technical indicators, and time-series forecasting.**

This project studies whether sentiment extracted from Apple (AAPL) news headlines can complement market-derived features when forecasting the next trading day's adjusted closing price. It links four Jupyter notebooks into a reproducible research workflow, with chronological holdout evaluation and a no-change price baseline.

> **Scope:** This is an experimental single-asset forecasting pipeline, **not** a validated financial risk reporting system, trading strategy, or production LLM agent. FinBERT is a pretrained financial-language classifier used for inference; it is not fine-tuned here.

## Research question

Can news-derived sentiment and technical signals improve next-day AAPL price forecasts beyond the simple assumption that tomorrow's adjusted close equals today's?

The project is relevant to AI-assisted financial research because it connects unstructured news to quantitative signals, records assumptions about information availability, and compares model outputs with a transparent baseline. It does **not** yet implement hallucination checking, retrieval-augmented generation, MySQL integration, volatility-risk reporting, or crisis-period stress tests.

## Pipeline

```text
Google News RSS / Selenium
        |
        v
Timestamped AAPL headlines (CSV)
        |
        v
Pretrained ProsusAI/finbert
  positive / negative / neutral probabilities
  sentiment score = P(positive) - P(negative)
        |
        v
Daily news aggregation + AAPL OHLCV prices
        |
        v
Technical indicators + news features
        |
        v
20-trading-day feature windows (14 features)
        |
        v
GRU Generator -> predicted standardized price adjustment
        |               + most recent adjusted close
        v
Predicted next-day adjusted close
        |
        v
CNN Discriminator conditioned on three historical closes
        |
        v
Chronological train/test evaluation + no-change baseline
```

## Notebooks (run in order)

| Notebook | Purpose | Main output |
| --- | --- | --- |
| `apple_news_web_scraping_2025_2026.ipynb` | Collect dated Apple/AAPL news via Google News RSS in a Selenium-driven workflow; deduplicate and check coverage. | `data/raw/aapl_news_2025_2026.csv` |
| `apple_finbert_sentiment_2025_2026.ipynb` | Score headlines using pretrained `ProsusAI/finbert`; save class probabilities, labels, and a signed sentiment score. | `data/raw/aapl_news_finbert_2025_2026.csv` and `data/raw/aapl_daily_sentiment_2025_2026.csv` |
| `AAPL_Data_Preprocessing.ipynb` | Download AAPL daily prices from Yahoo Finance's chart endpoint (or use a local fallback), calculate indicators, aggregate news by date, create chronological windows, and fit training-only scalers. | `notebooks/gan_ready/` arrays, scalers, and metadata |
| `AAPL_GAN.ipynb` | Train and evaluate a GRU Generator / CNN Discriminator model; save checkpoints, prediction metrics, charts, and baseline comparison. | `notebooks/gan_ready/gan_models_20day_previous_close_only/` |

**Repository naming:** Rename the uploaded `AAPL_Data_Preprocessing(1).ipynb` to `AAPL_Data_Preprocessing.ipynb`, `AAPL_GAN(2).ipynb` to `AAPL_GAN.ipynb`, and `apple_finbert_sentiment_2025_2026(4).ipynb` to `apple_finbert_sentiment_2025_2026.ipynb` when committing them.

## Model and feature design

- **Period configured in notebooks:** October 9, 2025 – October 9, 2026. Actual observations depend on source availability and valid trading dates.
- **Forecast target:** next trading day's **adjusted closing price**.
- **Input:** the previous **20 trading days** of 14 features:
  - Market and technical: `Adj_Close`, `Log_Return`, `MA7_Ratio`, `MA21_Ratio`, `MACD_Ratio`, `MACD_Signal`, `Volatility20`, `RSI14`, `Volume_Log`.
  - News: `Sentiment_Score`, `News_Count`, `Has_News`, `Sentiment_MA3`, `News_Count_3D`.
- **Fusion:** early feature fusion: news features and technical features are jointly supplied to the GRU.
- **Generator:** three GRU layers; the **uploaded notebook currently defaults to `(1024, 512, 256)`**. It produces an adjustment on the target-scaler scale, which is added to the latest observed adjusted close.
- **Discriminator:** CNN operating on a four-value sequence: candidate next-day price plus the preceding three adjusted closes.
- **Training:** 165 epochs, batch size 128, Adam learning rate 0.00016 in the uploaded notebook. Its `SUPERVISED_WEIGHT = 0.0`, so the optimized Generator loss is **adversarial only**; prediction MSE is monitored, not currently optimized. A supervised component is implemented but disabled by default.
- **Evaluation:** chronological 80/20 split of modeling rows, with training-only feature/target scaler fitting; RMSE, MAE, R², prediction plots, and a **previous-close (no-change) baseline**. Window construction uses earlier rows as context for later targets.

The repository does not assert that the GAN beats the baseline. Review `baseline_comparison.csv` and `metrics.csv` after training before drawing that conclusion.

## Suggested repository layout

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── apple_news_web_scraping_2025_2026.ipynb
│   ├── apple_finbert_sentiment_2025_2026.ipynb
│   ├── AAPL_Data_Preprocessing.ipynb
│   └── AAPL_GAN.ipynb
└── data/
    └── raw/                 # locally generated CSVs (not necessarily committed)
```

The preprocessing and GAN notebooks assume they are run **from `notebooks/`**. The news collection and FinBERT notebooks support running from either the project root or `notebooks/`.

## Setup and execution

Use a fresh environment with **Python 3.10** (recommended for the dependency ranges below). The original development environment may require platform-specific TensorFlow installation, particularly on Apple Silicon.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter lab
```

Open Jupyter in the repository, navigate to `notebooks/`, and run the four notebooks in the order shown above. The news collection step requires a working Chrome/Chromedriver setup, network access, and permission to access the source. The FinBERT model downloads from Hugging Face on first use. Price retrieval also requires network access unless a compatible offline file is available.

If the scored news CSV already exists, the first two notebooks can be skipped. If `notebooks/gan_ready/` already contains the expected arrays and scalers, the GAN notebook can be rerun without repeating preprocessing.

### Expected generated artifacts

- `data/raw/aapl_news_2025_2026.csv`: timestamped, deduplicated headlines.
- `data/raw/aapl_news_finbert_2025_2026.csv`: headline-level FinBERT probabilities and sentiment scores.
- `notebooks/gan_ready/metadata.json`: feature list, target, and split metadata.
- `notebooks/gan_ready/{X_train,X_test,y_train,y_test,yc_train,yc_test}.npy`: model-ready data.
- `notebooks/gan_ready/gan_models_20day_previous_close_only/metrics.csv`: forecast metrics.
- `notebooks/gan_ready/gan_models_20day_previous_close_only/baseline_comparison.csv`: GAN versus previous-close RMSE.
- `notebooks/gan_ready/gan_models_20day_previous_close_only/predictions.csv`: dated predictions and baseline values.

## Reliability, evaluation, and limitations

1. **Chronological testing:** train/test separation follows time order, rather than random splitting. This is a holdout test, **not** walk-forward cross-validation.
2. **Transparent baseline:** a no-change forecast (`next close = latest observed close`) is calculated alongside GAN results. Good-looking fitted curves alone do not establish forecasting skill.
3. **News timing:** news is assigned by New York publication calendar date; non-trading-day news is mapped to the next trading day. The implementation does **not** enforce a strict market-close cutoff for every headline. Therefore, it is not yet a leakage-audited, point-in-time backtest. Before making tradable claims, filter each article by actual information availability at forecast time.
4. **Exploratory Fourier analysis:** a full-series FFT is calculated for exploration, but its components are **not** among the 14 GAN input features. Full-series FFT should not be used as a historical predictive feature without causal reconstruction.
5. **Small sample and distribution shift:** roughly one year of daily prices offers limited independent training observations. Regime shifts and high-volatility periods may produce persistent forecast bias.
6. **Evaluation gaps:** no repeated-seed uncertainty analysis, rolling-origin backtest, crisis-era tests, formal anomaly/regime detection, or statistical significance test is provided by the current notebooks.
7. **Reproducibility:** random seeds are set for the GAN, but exact results may vary across devices, package versions, and external news/price feeds. This repository remains notebook-based; it is not presented as production-tested Python software.

## Next steps

- Add strict as-of timestamp filtering and a documented forecast decision time.
- Introduce walk-forward validation, multiple random seeds, and confidence intervals for model-versus-baseline improvements.
- Evaluate price-change/return forecasts and compare with simpler statistical models.
- Add unit tests for date alignment, feature construction, leakage prevention, and prediction reconstruction.
- For risk-reporting applications, extend beyond AAPL price forecasts to volatility, VaR/expected shortfall, anomaly signals, and automatically verified narrative claims.

## Responsible use

Research and educational use only; not investment advice. Respect news-source access terms, and do not commit credentials, private data, or large downloaded model artifacts.
