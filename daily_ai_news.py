"""AI news digest.

Two modes:
  * Daily mode  (default): AI news from RSS feeds from the last 24h, plus an optional
    WATCHLIST of names/topics (comma-separated env var) searched on Google News.
  * Search mode (QUERY set): searches Google News for that name/topic and sends the results to Telegram.
"""
import html as htmllib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import feedparser
import requests
from bs4 import BeautifulSoup

FEEDS = {
    "TechCrunch AI": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "The Verge AI": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    "MIT Tech Review AI": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    "VentureBeat AI": "https://venturebeat.com/category/ai/feed/",
}

MAX_ITEMS = 15

# Free LLM via Groq (OpenAI-compatible API). Model is configurable; Qwen by default.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL") or "qwen3-32b"
GROQ_FALLBACK_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SUMMARY_PROMPT = ("You summarize AI news. Reply with a 2 sentence plain-text summary only. "
                  "No preamble, no thinking.")
OVERVIEW_PROMPT = ("You are given news headlines about one person or topic. Write a 3 sentence "
                   "plain-text overview of what is being reported, using ONLY the headlines. "
                   "Do not invent facts. No preamble, no thinking.")


# ---------- LLM helpers ----------
def _chat(model: str, system: str, text: str) -> str:
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        },
        timeout=60,
    )
    r.raise_for_status()
    out = r.json()["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()


def llm(system: str, text: str, fallback: str) -> str:
    if not GROQ_API_KEY:
        return fallback
    for model in (GROQ_MODEL, GROQ_FALLBACK_MODEL):
        try:
            result = _chat(model, system, text)
            if result:
                return result
        except Exception as e:
            print(f"LLM call with {model} failed: {e}")
    return fallback


def summarize(text: str, fallback: str) -> str:
    if len(text) < 200:
        return fallback
    return llm(SUMMARY_PROMPT, text[:3000], fallback)


# ---------- Fetching ----------
def clean(markup: str) -> str:
    return BeautifulSoup(markup or "", "html.parser").get_text(" ", strip=True)


def excerpt(text: str, limit: int = 300) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def fetch_articles():
    """AI news from the fixed RSS feeds, last 24 hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    seen, items = set(), []
    for source, url in FEEDS.items():
        try:
            feed = feedparser.parse(url)
        except Exception as e:
            print(f"Feed failed {source}: {e}")
            continue
        for e in feed.entries:
            link = e.get("link")
            ts = e.get("published_parsed") or e.get("updated_parsed")
            if not link or not ts or link in seen:
                continue
            published = datetime(*ts[:6], tzinfo=timezone.utc)
            if published < cutoff:
                continue
            seen.add(link)
            full = clean(e.get("summary", "") or e.get("content", [{}])[0].get("value", ""))
            items.append({"source": source, "title": e.get("title", ""), "link": link,
                          "published": published, "full": full, "summary": excerpt(full)})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:MAX_ITEMS]


def search_news(query: str, days: int, limit: int):
    """Search Google News (RSS) for a name/topic."""
    q = quote_plus(f"{query} when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"Search failed for {query!r}: {e}")
        return []
    items, seen = [], set()
    for e in feed.entries:
        link, title = e.get("link"), e.get("title", "")
        if not link or title in seen:
            continue
        seen.add(title)
        ts = e.get("published_parsed")
        published = datetime(*ts[:6], tzinfo=timezone.utc) if ts else datetime.now(timezone.utc)
        source = (e.get("source") or {}).get("title", "Google News")
        items.append({"source": source, "title": title, "link": link,
                      "published": published, "summary": ""})
    items.sort(key=lambda x: x["published"], reverse=True)
    return items[:limit]


def overview_for(query: str, items) -> str:
    if not items:
        return ""
    headlines = "\n".join(f"- {i['title']}" for i in items)
    return llm(OVERVIEW_PROMPT, f"Topic: {query}\n{headlines}", "")


# ---------- Telegram ----------
def esc(s: str) -> str:
    return htmllib.escape(s or "", quote=False)


def build_messages(heading: str, groups):
    """Return a list of Telegram HTML messages, each under the 4096-char limit."""
    blocks = [f"<b>{esc(heading)}</b>\n"]
    for title, overview, items in groups:
        if title:
            blocks.append(f"\n<b>{esc(title)}</b>\n")
        if overview:
            blocks.append(f"<i>{esc(overview)}</i>\n")
        if not items:
            blocks.append("No articles found.\n")
        for n, i in enumerate(items, 1):
            link = htmllib.escape(i["link"], quote=True)
            line = f'{n}. <a href="{link}">{esc(i["title"])}</a>\n'
            line += f'<i>{esc(i["source"])} - {i["published"].strftime("%d %b")}</i>\n'
            if i.get("summary"):
                line += f'{esc(i["summary"])}\n'
            blocks.append(line)
    messages, current = [], ""
    for b in blocks:
        if len(current) + len(b) + 1 > 3800 and current:
            messages.append(current)
            current = ""
        current += b + "\n"
    if current.strip():
        messages.append(current)
    return messages


def send_telegram(messages):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]
    for text in messages:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML",
                  "disable_web_page_preview": True},
            timeout=30,
        )
        if not r.ok:
            # Never print the URL (it contains the bot token)
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text[:300]}")
        time.sleep(1)


# ---------- Main ----------
def run_search(query: str):
    items = search_news(query, days=7, limit=10)
    overview = overview_for(query, items)
    send_telegram(build_messages(f"News search: {query} (last 7 days)", [(None, overview, items)]))
    print(f"Sent search results for {query!r} ({len(items)} articles)")


def run_daily():
    date = datetime.now().strftime("%d %b %Y")
    articles = fetch_articles()
    for a in articles:
        a["summary"] = summarize(a["full"], a["summary"])
        time.sleep(2)  # stay under free-tier rate limits

    groups = [("Top AI news", "", articles)]

    watchlist = [w.strip() for w in (os.environ.get("WATCHLIST") or "").split(",") if w.strip()]
    for name in watchlist:
        found = search_news(name, days=1, limit=5)
        if found:
            groups.append((f"Watchlist: {name}", overview_for(name, found), found))
            time.sleep(2)

    send_telegram(build_messages(f"AI News Digest - {date}", groups))
    print(f"Sent digest with {len(articles)} articles, {len(watchlist)} watchlist names")


if __name__ == "__main__":
    q = (os.environ.get("QUERY") or "").strip()
    if q:
        run_search(q[:100])
    else:
        run_daily()
