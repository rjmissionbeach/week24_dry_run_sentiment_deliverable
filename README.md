# News Sentiment vs. Market Reaction

Streamlit app for Class 24 weekly deliverable.

## What it does

- Pulls recent company news from Finnhub's company-news endpoint.
- Scores each article's sentiment using OpenRouter.
- Pulls daily adjusted close prices with yfinance.
- Aligns news dates to trading dates.
- Computes a directional hit rate.
- Shows required charts and optional graduate word cloud.

## Streamlit secrets

In Streamlit Community Cloud, add these secrets:

```toml
FINNHUB_API_KEY = "your_finnhub_key_here"
OPENROUTER_API_KEY = "your_openrouter_key_here"
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```
