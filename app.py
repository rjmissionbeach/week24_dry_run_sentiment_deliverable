import json
import os
import re
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# Optional graduate extension imports
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False


st.set_page_config(page_title="News Sentiment vs. Market Reaction", layout="wide")

# Consistent sentiment colors across labels, counts, and timeline bars.
SENTIMENT_COLORS = {
    "positive": "#2ca02c",  # green
    "neutral": "#f1c40f",   # yellow
    "negative": "#d62728",  # red
}


# -----------------------------
# Helpers for secrets / keys
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, default)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()


# -----------------------------
# Finnhub company news
# -----------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_company_news(ticker: str, start_date: str, end_date: str, finnhub_key: str) -> list[dict]:
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": start_date, "to": end_date}
    headers = {"X-Finnhub-Token": finnhub_key}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError(f"Finnhub returned an unexpected response: {data}")
    return data


def normalize_news(raw_news: list[dict], max_articles: int, latest_usable_date=None) -> pd.DataFrame:
    rows = []
    for item in raw_news:
        unix_time = item.get("datetime")
        if not unix_time:
            continue
        published_dt = datetime.fromtimestamp(unix_time)
        rows.append(
            {
                "article_id": str(item.get("id", "")),
                "published_datetime": published_dt,
                "published_date": published_dt.date(),
                "headline": item.get("headline", "") or "",
                "summary": item.get("summary", "") or "",
                "source": item.get("source", "") or "",
                "url": item.get("url", "") or "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Newest first, but optionally exclude articles that are too recent to have
    # an observable future return. This matters because the assignment asks for
    # "what happened afterward"; news from today/yesterday often cannot be tested yet.
    if latest_usable_date is not None:
        latest_usable_date = pd.to_datetime(latest_usable_date).date()
        df = df[df["published_date"] <= latest_usable_date]

    df = df.sort_values("published_datetime", ascending=False).head(max_articles).reset_index(drop=True)
    df["article_number"] = range(1, len(df) + 1)
    return df


# -----------------------------
# OpenRouter sentiment scoring
# -----------------------------
def extract_json_array(text: str) -> list[dict]:
    """Try to parse JSON even if the model wraps it in text or markdown."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "articles" in parsed:
            return parsed["articles"]
    except Exception:
        pass

    # Remove markdown fences if present.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "articles" in parsed:
            return parsed["articles"]
    except Exception:
        pass

    # Last attempt: locate the first JSON array.
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if match:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    raise ValueError("Could not parse model response as a JSON list.")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def score_sentiment_batch(articles_payload: str, openrouter_key: str, model: str) -> list[dict]:
    articles = json.loads(articles_payload)

    system_prompt = (
        "You are a careful financial news sentiment scorer. "
        "Score sentiment from the perspective of the named company's future stock return, "
        "not general emotional tone. Return only valid JSON."
    )
    user_prompt = {
        "instructions": (
            "For each article, return article_number, score, label, and rationale. "
            "score must be a number from -1.0 to 1.0, where -1 is very negative, "
            "0 is neutral/mixed/unclear, and +1 is very positive. "
            "label must be exactly one of: positive, neutral, negative. "
            "rationale must be brief, max 15 words. Return a JSON array only."
        ),
        "articles": articles,
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io/",
            "X-Title": "News Sentiment vs Market Reaction Class Project",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt)},
            ],
            "temperature": 0,
        },
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return extract_json_array(text)


def add_sentiment(news_df: pd.DataFrame, sentiment_rows: list[dict]) -> pd.DataFrame:
    sent_df = pd.DataFrame(sentiment_rows)
    if sent_df.empty:
        raise ValueError("The model returned no sentiment scores.")

    sent_df["article_number"] = pd.to_numeric(sent_df["article_number"], errors="coerce").astype("Int64")
    sent_df["score"] = pd.to_numeric(sent_df["score"], errors="coerce").clip(-1, 1)
    sent_df["label"] = sent_df["label"].astype(str).str.lower().str.strip()
    sent_df.loc[~sent_df["label"].isin(["positive", "neutral", "negative"]), "label"] = "neutral"

    merged = news_df.merge(sent_df, on="article_number", how="left")
    merged["score"] = merged["score"].fillna(0.0)
    merged["label"] = merged["label"].fillna("neutral")
    merged["rationale"] = merged["rationale"].fillna("No rationale returned.")
    return merged


# -----------------------------
# Price data and alignment
# -----------------------------
@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_prices(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Download extra days after the news window so next-1/3-day returns can be computed.
    px_end = (pd.to_datetime(end_date) + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    prices = yf.download(ticker, start=start_date, end=px_end, progress=False, auto_adjust=True)
    if prices.empty:
        return pd.DataFrame()

    # yfinance sometimes returns MultiIndex columns.
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    out = prices.reset_index()[["Date", "Close"]].copy()
    out.columns = ["trading_date", "close"]
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.date
    out = out.dropna().reset_index(drop=True)
    return out


def compute_daily_sentiment(scored_news: pd.DataFrame) -> pd.DataFrame:
    daily = (
        scored_news.groupby("published_date")
        .agg(avg_sentiment=("score", "mean"), article_count=("score", "size"))
        .reset_index()
        .sort_values("published_date")
    )
    return daily


def align_sentiment_to_prices(daily_sentiment: pd.DataFrame, prices: pd.DataFrame, horizon_days: int, neutral_cutoff: float) -> pd.DataFrame:
    rows = []
    trading_dates = list(prices["trading_date"])

    for _, row in daily_sentiment.iterrows():
        news_date = row["published_date"]
        possible_anchor_dates = [d for d in trading_dates if d >= news_date]
        if not possible_anchor_dates:
            continue

        anchor_date = possible_anchor_dates[0]
        anchor_idx = prices.index[prices["trading_date"] == anchor_date][0]
        future_idx = anchor_idx + horizon_days
        if future_idx >= len(prices):
            continue

        anchor_close = float(prices.loc[anchor_idx, "close"])
        future_date = prices.loc[future_idx, "trading_date"]
        future_close = float(prices.loc[future_idx, "close"])
        future_return = future_close / anchor_close - 1
        avg_sentiment = float(row["avg_sentiment"])

        if avg_sentiment > neutral_cutoff:
            predicted_direction = "up"
            hit = future_return > 0
        elif avg_sentiment < -neutral_cutoff:
            predicted_direction = "down"
            hit = future_return < 0
        else:
            predicted_direction = "neutral"
            hit = None

        rows.append(
            {
                "news_date": news_date,
                "trading_date_used": anchor_date,
                "future_date": future_date,
                "article_count": int(row["article_count"]),
                "avg_sentiment": avg_sentiment,
                "anchor_close": anchor_close,
                "future_close": future_close,
                "future_return": future_return,
                "predicted_direction": predicted_direction,
                "hit": hit,
                "rolled_forward": anchor_date != news_date,
            }
        )

    return pd.DataFrame(rows)


def make_combined_chart(aligned: pd.DataFrame, prices: pd.DataFrame):
    fig = go.Figure()

    bar_colors = aligned["predicted_direction"].map(
        {
            "up": SENTIMENT_COLORS["positive"],
            "neutral": SENTIMENT_COLORS["neutral"],
            "down": SENTIMENT_COLORS["negative"],
        }
    ).fillna(SENTIMENT_COLORS["neutral"])

    fig.add_trace(
        go.Scatter(
            x=prices["trading_date"],
            y=prices["close"],
            name="Adjusted close",
            mode="lines",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Bar(
            x=aligned["trading_date_used"],
            y=aligned["avg_sentiment"],
            name="Average daily sentiment",
            yaxis="y2",
            opacity=0.65,
            marker_color=bar_colors,
            hovertemplate=(
                "Date: %{x}<br>"
                "Avg sentiment: %{y:.3f}<br>"
                "Green = positive, yellow = neutral, red = negative"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Price and News Sentiment Timeline",
        xaxis_title="Date",
        yaxis=dict(title="Adjusted close"),
        yaxis2=dict(title="Sentiment", overlaying="y", side="right", range=[-1, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=500,
    )
    return fig


def show_wordcloud(scored_news: pd.DataFrame):
    if not WORDCLOUD_AVAILABLE:
        st.warning("Word cloud package is not installed. Add wordcloud to requirements.txt.")
        return

    text = " ".join((scored_news["headline"].fillna("") + " " + scored_news["summary"].fillna("")).tolist()).strip()
    if not text:
        st.info("No text available for a word cloud.")
        return

    wc = WordCloud(width=1000, height=450, background_color="white", collocations=False).generate(text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("News Sentiment vs. Market Reaction")
st.write(
    "Enter a ticker, pull recent Finnhub company news, score sentiment with OpenRouter, "
    "and compare the signal to the stock's next trading-day price movement."
)

with st.sidebar:
    st.header("Settings")
    ticker = clean_ticker(st.text_input("Ticker", value="AAPL"))
    days_back = st.slider("News window: days back", min_value=7, max_value=90, value=30, step=1)
    max_articles = st.slider("Articles to score", min_value=3, max_value=30, value=12, step=1)
    horizon_days = st.slider("Price reaction horizon: trading days", min_value=1, max_value=3, value=1, step=1)
    neutral_cutoff = st.slider("Neutral cutoff for daily sentiment", min_value=0.00, max_value=0.25, value=0.05, step=0.01)
    model = st.text_input("OpenRouter model", value="openai/gpt-4o-mini")
    show_grad_wordcloud = st.checkbox("Show graduate word cloud extension", value=True)

    st.divider()
    st.caption("Keys are read from Streamlit secrets or environment variables.")
    finnhub_key = get_secret("FINNHUB_API_KEY")
    openrouter_key = get_secret("OPENROUTER_API_KEY")
    st.write("Finnhub key:", "✅ found" if finnhub_key else "❌ missing")
    st.write("OpenRouter key:", "✅ found" if openrouter_key else "❌ missing")

run = st.button("Run analysis", type="primary", disabled=not ticker)

if not finnhub_key or not openrouter_key:
    st.warning(
        "Add FINNHUB_API_KEY and OPENROUTER_API_KEY in Streamlit secrets before running. "
        "For local testing, you can also set them as environment variables."
    )

if run:
    if not finnhub_key or not openrouter_key:
        st.stop()

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days_back)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # To compute a future return, we need to avoid scoring articles that are too
    # close to today. The buffer is deliberately calendar-day based and simple
    # enough for students to explain. Users can still pull news through today;
    # this only affects which articles are scored for the hit-rate exercise.
    future_return_buffer_days = horizon_days + 3
    latest_usable_news_date = end_dt - timedelta(days=future_return_buffer_days)

    try:
        with st.spinner("Pulling company news from Finnhub..."):
            raw_news = fetch_company_news(ticker, start_str, end_str, finnhub_key)
            all_news_df = normalize_news(raw_news, max_articles=10_000)
            news_df = normalize_news(raw_news, max_articles, latest_usable_date=latest_usable_news_date)

        st.subheader("1. Finnhub company news checkpoint")
        c1, c2, c3 = st.columns(3)
        c1.metric("Articles requested by slider", max_articles)
        c2.metric("Articles returned by Finnhub", len(raw_news))
        c3.metric("Articles scored in this run", len(news_df))

        if not all_news_df.empty:
            too_recent_count = int((all_news_df["published_date"] > latest_usable_news_date).sum())
            st.caption(
                f"For the hit-rate test, the app scores articles published on or before "
                f"{latest_usable_news_date}. It skipped {too_recent_count} newer article(s) because "
                f"there may not be enough future trading data yet."
            )

        if news_df.empty:
            st.error(
                "Finnhub returned news, but none of the articles are old enough to test against future price data. "
                "Try a longer news window, a lower reaction horizon, or a higher-coverage ticker."
            )
            st.stop()

        min_news_date = news_df["published_date"].min()
        max_news_date = news_df["published_date"].max()
        st.write(
            f"For **{ticker}**, Finnhub returned **{len(raw_news)}** raw articles from **{start_str}** to **{end_str}**. "
            f"This app scored the newest **{len(news_df)}** articles. The scored articles range from **{min_news_date}** to **{max_news_date}**."
        )
        if len(raw_news) < max_articles:
            st.info("Finnhub returned fewer articles than requested by the slider. This is common for low-coverage tickers or short windows.")
        elif len(news_df) < max_articles:
            st.info(
                "The app intentionally scored fewer articles than requested because some of the newest articles "
                "were too recent to evaluate with a future price move."
            )

        articles_for_model = []
        for _, row in news_df.iterrows():
            articles_for_model.append(
                {
                    "article_number": int(row["article_number"]),
                    "published_date": str(row["published_date"]),
                    "headline": row["headline"][:500],
                    "summary": row["summary"][:1000],
                }
            )
        articles_payload = json.dumps(articles_for_model, sort_keys=True)

        with st.spinner("Scoring sentiment with OpenRouter..."):
            sentiment_rows = score_sentiment_batch(articles_payload, openrouter_key, model)
            scored_news = add_sentiment(news_df, sentiment_rows)

        with st.spinner("Pulling price data with yfinance and aligning dates..."):
            prices = fetch_prices(ticker, start_str, end_str)
            if prices.empty:
                st.error("No price data returned. Check the ticker symbol.")
                st.stop()
            daily_sentiment = compute_daily_sentiment(scored_news)
            aligned = align_sentiment_to_prices(daily_sentiment, prices, horizon_days, neutral_cutoff)

        st.subheader("2. Sentiment-scored articles")
        display_cols = ["published_datetime", "source", "headline", "score", "label", "rationale", "url"]

        def color_sentiment_label(value):
            color = SENTIMENT_COLORS.get(str(value).lower(), SENTIMENT_COLORS["neutral"])
            return f"background-color: {color}; color: black; font-weight: 700;"

        def color_sentiment_score(value):
            try:
                value = float(value)
            except Exception:
                value = 0.0
            if value > neutral_cutoff:
                color = SENTIMENT_COLORS["positive"]
            elif value < -neutral_cutoff:
                color = SENTIMENT_COLORS["negative"]
            else:
                color = SENTIMENT_COLORS["neutral"]
            return f"background-color: {color}; color: black;"

        styled_news = (
            scored_news[display_cols]
            .style
            .applymap(color_sentiment_label, subset=["label"])
            .applymap(color_sentiment_score, subset=["score"])
            .format({"score": "{:.3f}"})
        )
        st.dataframe(styled_news, use_container_width=True, hide_index=True)

        st.subheader("3. Sentiment distribution")
        label_order = ["positive", "neutral", "negative"]
        dist = scored_news["label"].value_counts().reindex(label_order, fill_value=0).reset_index()
        dist.columns = ["label", "count"]
        fig_dist = px.bar(
            dist,
            x="label",
            y="count",
            title="Article Sentiment Counts",
            color="label",
            color_discrete_map=SENTIMENT_COLORS,
            category_orders={"label": label_order},
        )
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.subheader("4. Price alignment and hit rate")
        if aligned.empty:
            st.error(
                "Could not align the scored news sentiment to enough future price data. "
                "This usually means the scored articles are still too close to the end of the price series. "
                "Try a longer news window, fewer articles, or a shorter horizon."
            )
            st.write("Latest available price date:", prices["trading_date"].max())
            st.write("Latest scored news date:", scored_news["published_date"].max())
            st.stop()

        directional = aligned[aligned["predicted_direction"].isin(["up", "down"])].copy()
        if directional.empty:
            st.warning("All daily sentiment averages were neutral under the current cutoff, so no directional hit rate was computed.")
            hit_rate = None
        else:
            hit_rate = directional["hit"].mean()
            st.metric(
                "Directional hit rate",
                f"{hit_rate:.1%}",
                help="Positive sentiment counts as a hit if the future return is positive; negative sentiment counts as a hit if the future return is negative. Neutral days are excluded.",
            )

        skipped_days = max(0, len(daily_sentiment) - len(aligned))
        st.write(
            f"News on weekends or market holidays is rolled forward to the next available trading close. "
            f"The future price move is measured over the next **{horizon_days}** trading day(s). "
            f"The alignment step skipped **{skipped_days}** news day(s) without enough future price data."
        )
        st.dataframe(
            aligned[[
                "news_date", "trading_date_used", "future_date", "article_count", "avg_sentiment",
                "predicted_direction", "future_return", "hit", "rolled_forward"
            ]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("5. Price and sentiment timeline")
        st.plotly_chart(make_combined_chart(aligned, prices), use_container_width=True)

        if show_grad_wordcloud:
            st.subheader("Graduate extension: word cloud")
            show_wordcloud(scored_news)

        st.subheader("Submission notes you can paste/edit")
        st.markdown(
            f"""
**Finnhub checkpoint:** For `{ticker}`, I requested up to {max_articles} articles over the last {days_back} days. Finnhub returned {len(raw_news)} raw articles, and the app scored {len(news_df)} usable articles published on or before {latest_usable_news_date}. The scored article dates ranged from {min_news_date} to {max_news_date}. This shows that the app cannot assume the requested number of articles will actually be available or usable for a future-return test, especially for low-coverage tickers, short date windows, or very recent news.

**Non-trading-day handling:** I grouped articles by calendar publication date. If a news date was not a trading day, I rolled it forward to the next available trading close rather than dropping it. I then compared that close to the close {horizon_days} trading day(s) later. I also avoided scoring articles too close to today because there may not be enough future trading data yet. This avoids crashes on weekends/holidays and keeps the rule consistent across tickers.
            """
        )

    except requests.HTTPError as e:
        st.error(f"API request failed: {e}")
        try:
            st.code(e.response.text)
        except Exception:
            pass
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.exception(e)
