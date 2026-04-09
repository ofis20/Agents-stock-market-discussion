"""Contexto macro adicional para enriquecer el debate de agentes.

Tres fuentes nuevas (todas gratuitas, sin API key):
1. Datos economicos reales via FRED (Federal Reserve Economic Data) - API publica
2. Noticias macro recientes via RSS feeds de fuentes financieras
3. Earnings recientes y proximos via yfinance

Cada funcion devuelve un string formateado listo para inyectar en el briefing.
"""

from __future__ import annotations

import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import yfinance as yf

# Cargar .env si existe (sin depender de python-dotenv)
_ENV_FILE = Path(__file__).resolve().parent / ".env"
if _ENV_FILE.exists():
    for line in _ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

try:
    import feedparser
except ImportError:
    feedparser = None  # type: ignore[assignment]


# ═══════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_TTL = int(os.environ.get("MACRO_CONTEXT_CACHE_TTL", 3600))


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    return _CACHE_DIR / f"ctx_{safe}.pkl"


def _cache_get(key: str) -> Any | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = datetime.now().timestamp() - p.stat().st_mtime
    if age > _CACHE_TTL:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATOS ECONOMICOS REALES (FRED API - gratuita, sin key requerida)
# ═══════════════════════════════════════════════════════════════════════════

# Series FRED relevantes para macro (ID oficial → nombre legible)
FRED_SERIES: dict[str, str] = {
    # Inflacion
    "CPIAUCSL": "IPC USA (CPI, mensual)",
    "CPILFESL": "IPC Core USA (sin alimentos/energia)",
    "PCEPILFE": "PCE Core (preferido Fed)",
    "T5YIE": "Expectativas inflacion 5Y (breakeven)",
    "T10YIE": "Expectativas inflacion 10Y (breakeven)",
    # Tipos de interes / Fed
    "FEDFUNDS": "Tasa Fed Funds efectiva",
    "DFF": "Fed Funds diaria",
    "DGS2": "Yield Treasury 2Y",
    "DGS10": "Yield Treasury 10Y",
    "DGS30": "Yield Treasury 30Y",
    "T10Y2Y": "Spread 10Y-2Y (curva tipos)",
    "T10Y3M": "Spread 10Y-3M (curva tipos)",
    # Empleo
    "UNRATE": "Tasa desempleo USA",
    "PAYEMS": "Nominas no agricolas (NFP)",
    "ICSA": "Peticiones desempleo semanales",
    "JTSJOL": "Ofertas empleo (JOLTS)",
    # Actividad economica
    "GDP": "PIB real USA (trimestral)",
    "GDPC1": "PIB real USA encadenado",
    "INDPRO": "Produccion industrial",
    "RSXFS": "Ventas minoristas (ex alimentos)",
    "UMCSENT": "Confianza consumidor Michigan",
    # Vivienda
    "HOUST": "Inicios vivienda",
    "CSUSHPINSA": "Indice Case-Shiller vivienda",
    # Monetarios / Liquidez
    "M2SL": "Oferta monetaria M2",
    "WALCL": "Balance Fed (activos totales)",
    # Credito
    "BAMLH0A0HYM2": "Spread High Yield (OAS)",
    "BAMLC0A4CBBB": "Spread BBB Corporate",
    # Internacional
    "DTWEXBGS": "Indice Dolar ponderado comercio",
}

# API publica FRED (no requiere key para series basicas via observaciones)
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_API_KEY = os.environ.get("FRED_API_KEY", "")


def _fetch_fred_series(series_id: str, limit: int = 5) -> list[dict[str, str]]:
    """Descarga las ultimas observaciones de una serie FRED.

    Si FRED_API_KEY esta configurada, usa la API oficial.
    Si no, intenta el endpoint publico sin key (limitado).
    Timeout corto (8s) para no bloquear el debate.
    """
    if _FRED_API_KEY:
        params = {
            "series_id": series_id,
            "api_key": _FRED_API_KEY,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        try:
            resp = requests.get(_FRED_BASE, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            return data.get("observations", [])
        except Exception:
            return []

    # Fallback sin API key: endpoint GeoFRED (publico, limitado)
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "MacroDebateApp/1.0"})
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        if len(lines) < 2:
            return []
        # CSV: DATE,VALUE
        results = []
        for line in lines[-limit:]:
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip() != ".":
                results.append({"date": parts[0].strip(), "value": parts[1].strip()})
        return results
    except Exception:
        return []


def fetch_economic_data() -> str:
    """Obtiene datos economicos clave de FRED y devuelve briefing formateado."""
    cached = _cache_get("fred_economic")
    if cached is not None:
        return cached

    print("  [FRED] Descargando datos economicos...", flush=True)
    results: dict[str, list[dict[str, str]]] = {}

    # Descargar en paralelo con timeout global de 30s
    series_list = list(FRED_SERIES.keys())
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(_fetch_fred_series, sid, 3): sid for sid in series_list}
            deadline = time.monotonic() + 30  # timeout global
            for future in as_completed(futures, timeout=30):
                if time.monotonic() > deadline:
                    break
                sid = futures[future]
                try:
                    obs = future.result(timeout=5)
                    if obs:
                        results[sid] = obs
                except Exception:
                    pass
    except TimeoutError:
        print("  [FRED] Timeout global alcanzado, usando datos parciales.", flush=True)

    if not results:
        text = "(Datos economicos FRED no disponibles)"
        _cache_set("fred_economic", text)
        return text

    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== DATOS ECONOMICOS REALES (FRED, {today}) ===\n")

    # Agrupar por categoria
    categories = {
        "INFLACION": ["CPIAUCSL", "CPILFESL", "PCEPILFE", "T5YIE", "T10YIE"],
        "TIPOS DE INTERES / FED": ["FEDFUNDS", "DFF", "DGS2", "DGS10", "DGS30", "T10Y2Y", "T10Y3M"],
        "EMPLEO": ["UNRATE", "PAYEMS", "ICSA", "JTSJOL"],
        "ACTIVIDAD ECONOMICA": ["GDP", "GDPC1", "INDPRO", "RSXFS", "UMCSENT"],
        "VIVIENDA": ["HOUST", "CSUSHPINSA"],
        "LIQUIDEZ / MONETARIO": ["M2SL", "WALCL"],
        "CREDITO": ["BAMLH0A0HYM2", "BAMLC0A4CBBB"],
        "DOLAR": ["DTWEXBGS"],
    }

    for cat_name, series_ids in categories.items():
        cat_lines = []
        for sid in series_ids:
            if sid not in results:
                continue
            obs = results[sid]
            nombre = FRED_SERIES.get(sid, sid)
            if obs:
                latest = obs[-1] if isinstance(obs, list) else obs
                val = latest.get("value", "N/D")
                date = latest.get("date", "")
                # Mostrar cambio si hay >1 observacion
                prev_val = ""
                if len(obs) >= 2:
                    try:
                        curr = float(val)
                        prev = float(obs[-2]["value"])
                        diff = curr - prev
                        prev_val = f" (anterior: {prev:.2f}, cambio: {diff:+.2f})"
                    except (ValueError, KeyError):
                        pass
                cat_lines.append(f"  {nombre}: {val} ({date}){prev_val}")
        if cat_lines:
            lines.append(f"-- {cat_name} --")
            lines.extend(cat_lines)
            lines.append("")

    text = "\n".join(lines)
    loaded = sum(1 for v in results.values() if v)
    print(f"  [FRED] {loaded}/{len(series_list)} series cargadas.", flush=True)
    _cache_set("fred_economic", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 2. NOTICIAS MACRO RECIENTES (RSS feeds gratuitos)
# ═══════════════════════════════════════════════════════════════════════════

# Feeds RSS de fuentes financieras de alta calidad
_RSS_FEEDS: list[dict[str, str]] = [
    {"url": "https://feeds.reuters.com/reuters/businessNews", "name": "Reuters Business"},
    {"url": "https://feeds.reuters.com/reuters/topNews", "name": "Reuters Top"},
    {"url": "https://feeds.bbci.co.uk/news/business/rss.xml", "name": "BBC Business"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "name": "NYT Business"},
    {"url": "https://feeds.bloomberg.com/markets/news.rss", "name": "Bloomberg Markets"},
    {"url": "https://www.cnbc.com/id/10001147/device/rss/rss.html", "name": "CNBC Economy"},
    {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html", "name": "CNBC Markets"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US", "name": "Yahoo Finance S&P"},
    {"url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069", "name": "CNBC Fed"},
    {"url": "https://www.investing.com/rss/news.rss", "name": "Investing.com"},
]

# Palabras clave para filtrar noticias relevantes al debate macro
_MACRO_KEYWORDS = [
    "fed", "federal reserve", "interest rate", "inflation", "cpi", "gdp",
    "employment", "jobs", "unemployment", "tariff", "trade war", "recession",
    "earnings", "stock market", "bond", "treasury", "yield", "dollar",
    "oil", "gold", "bitcoin", "crypto", "china", "europe", "ecb",
    "bank", "crisis", "growth", "rate cut", "rate hike", "monetary",
    "fiscal", "stimulus", "debt", "deficit", "commodity", "s&p",
    "nasdaq", "dow", "market", "economy", "economic", "war", "geopolitical",
    "sanction", "ai", "artificial intelligence", "semiconductor", "chip",
    "energy", "climate", "green", "renewable", "nuclear",
    "housing", "real estate", "consumer", "spending", "retail",
    "aranceles", "tipos de interes", "inflacion", "empleo", "pib",
]


def _is_macro_relevant(title: str, summary: str = "") -> bool:
    """Comprueba si una noticia es relevante para el debate macro."""
    text = (title + " " + summary).lower()
    return any(kw in text for kw in _MACRO_KEYWORDS)


def _fetch_single_feed(feed: dict[str, str], max_articles: int = 10) -> list[dict[str, str]]:
    """Descarga y parsea un feed RSS individual."""
    if feedparser is None:
        return []
    try:
        parsed = feedparser.parse(
            feed["url"],
            request_headers={"User-Agent": "MacroDebateApp/1.0"},
        )
        articles = []
        for entry in parsed.entries[:max_articles * 2]:  # Descargamos más para filtrar
            title = getattr(entry, "title", "").strip()
            summary = getattr(entry, "summary", "").strip()
            published = getattr(entry, "published", "")
            if not title:
                continue
            # Limpiar HTML del summary
            summary_clean = re.sub(r"<[^>]+>", "", summary)[:200]
            if _is_macro_relevant(title, summary_clean):
                articles.append({
                    "source": feed["name"],
                    "title": title,
                    "summary": summary_clean,
                    "date": published,
                })
            if len(articles) >= max_articles:
                break
        return articles
    except Exception:
        return []


def fetch_macro_news(max_total: int = 30) -> str:
    """Descarga noticias macro recientes de RSS y devuelve resumen formateado."""
    if feedparser is None:
        return "(feedparser no instalado - ejecutar: pip install feedparser)"

    cached = _cache_get("macro_news")
    if cached is not None:
        return cached

    print("  [RSS] Descargando noticias macro...", flush=True)
    all_articles: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fetch_single_feed, f, 8): f for f in _RSS_FEEDS}
        for future in as_completed(futures):
            try:
                articles = future.result()
                all_articles.extend(articles)
            except Exception:
                pass

    if not all_articles:
        text = "(No se pudieron obtener noticias macro recientes)"
        _cache_set("macro_news", text)
        return text

    # Deduplicar por titulo similar
    seen_titles: set[str] = set()
    unique: list[dict[str, str]] = []
    for art in all_articles:
        # Normalizar titulo para dedup
        norm = re.sub(r"[^a-z0-9 ]", "", art["title"].lower())[:60]
        if norm not in seen_titles:
            seen_titles.add(norm)
            unique.append(art)

    unique = unique[:max_total]

    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== NOTICIAS MACRO RECIENTES ({today}) - {len(unique)} articulos ===\n")

    for art in unique:
        source = art["source"]
        title = art["title"]
        summary = art["summary"][:150] if art["summary"] else ""
        date_str = f" [{art['date'][:16]}]" if art.get("date") else ""
        line = f"  [{source}]{date_str} {title}"
        if summary:
            line += f" — {summary}"
        lines.append(line)

    text = "\n".join(lines)
    print(f"  [RSS] {len(unique)} noticias macro relevantes obtenidas.", flush=True)
    _cache_set("macro_news", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# 3. EARNINGS RECIENTES Y PROXIMOS (yfinance)
# ═══════════════════════════════════════════════════════════════════════════

# Tickers para monitorear earnings (mega caps + activos populares del debate)
EARNINGS_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK-B",
    "JPM", "V", "MA", "JNJ", "UNH", "LLY", "ABBV", "MRK",
    "XOM", "CVX", "AVGO", "ORCL", "CRM", "ADBE", "AMD",
    "BAC", "GS", "NFLX", "DIS", "PLTR", "COIN", "ARM", "SMCI",
    "BABA", "TSM", "NVO", "SAP", "ASML",
    "CAT", "BA", "COST", "WMT", "HD",
    "PG", "KO", "NKE", "CRSP", "MELI",
]


def _fetch_single_earnings(ticker: str) -> dict[str, Any] | None:
    """Obtiene datos de earnings recientes para un ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        result: dict[str, Any] = {"ticker": ticker, "name": info.get("shortName", ticker)}

        # Earnings mas recientes
        try:
            cal = t.calendar
            if cal is not None and not (hasattr(cal, "empty") and cal.empty):
                if isinstance(cal, dict):
                    result["next_earnings"] = str(cal.get("Earnings Date", ["N/D"])[0]) if cal.get("Earnings Date") else None
                    result["earnings_avg_estimate"] = cal.get("Earnings Average", None)
                    result["revenue_avg_estimate"] = cal.get("Revenue Average", None)
        except Exception:
            pass

        # Revenue/earnings growth de info
        result["earnings_growth"] = info.get("earningsGrowth")
        result["revenue_growth"] = info.get("revenueGrowth")
        result["earnings_surprise"] = info.get("earningsQuarterlyGrowth")

        # Target de analistas
        result["target_mean"] = info.get("targetMeanPrice")
        result["target_low"] = info.get("targetLowPrice")
        result["target_high"] = info.get("targetHighPrice")
        result["recommendation"] = info.get("recommendationKey")
        result["num_analysts"] = info.get("numberOfAnalystOpinions")
        result["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")

        # Calcular upside/downside vs target
        if result.get("target_mean") and result.get("current_price"):
            try:
                result["upside_pct"] = ((result["target_mean"] / result["current_price"]) - 1) * 100
            except (ZeroDivisionError, TypeError):
                pass

        return result
    except Exception:
        return None


def fetch_earnings_data() -> str:
    """Obtiene earnings recientes y estimaciones de analistas. Devuelve briefing formateado."""
    cached = _cache_get("earnings_data")
    if cached is not None:
        return cached

    print(f"  [Earnings] Descargando datos de {len(EARNINGS_TICKERS)} empresas...", flush=True)
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_single_earnings, t): t for t in EARNINGS_TICKERS}
        done = 0
        for future in as_completed(futures):
            done += 1
            try:
                data = future.result()
                if data:
                    results.append(data)
            except Exception:
                pass
            if done % 15 == 0:
                print(f"    ... {done}/{len(EARNINGS_TICKERS)} procesados", flush=True)

    if not results:
        text = "(Datos de earnings no disponibles)"
        _cache_set("earnings_data", text)
        return text

    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== EARNINGS Y ESTIMACIONES DE ANALISTAS ({today}) ===\n")

    # Proximos earnings
    upcoming: list[dict[str, Any]] = []
    for r in results:
        if r.get("next_earnings"):
            try:
                # Intentar parsear fecha
                date_str = str(r["next_earnings"])
                upcoming.append(r)
            except Exception:
                pass

    if upcoming:
        upcoming.sort(key=lambda x: str(x.get("next_earnings", "9999")))
        lines.append("-- PROXIMOS EARNINGS --")
        for r in upcoming[:15]:
            est = ""
            if r.get("earnings_avg_estimate"):
                est = f" | Est. EPS: ${r['earnings_avg_estimate']:.2f}"
            if r.get("revenue_avg_estimate"):
                rev_b = r["revenue_avg_estimate"] / 1e9
                est += f" | Est. Rev: ${rev_b:.1f}B"
            lines.append(f"  {r['name']} ({r['ticker']}): {r['next_earnings']}{est}")
        lines.append("")

    # Crecimiento de earnings y recomendaciones
    lines.append("-- CRECIMIENTO EARNINGS Y RECOMENDACIONES ANALISTAS --")
    for r in sorted(results, key=lambda x: x.get("upside_pct") or -999, reverse=True):
        parts = [f"  {r['name']} ({r['ticker']})"]

        eg = r.get("earnings_growth")
        if eg is not None:
            parts.append(f"Crec. EPS: {eg*100:+.1f}%")

        rg = r.get("revenue_growth")
        if rg is not None:
            parts.append(f"Crec. Rev: {rg*100:+.1f}%")

        rec = r.get("recommendation")
        if rec:
            parts.append(f"Rec: {rec}")

        na = r.get("num_analysts")
        if na:
            parts.append(f"({na} analistas)")

        up = r.get("upside_pct")
        if up is not None:
            parts.append(f"Upside: {up:+.1f}%")

        target = r.get("target_mean")
        if target:
            lo = r.get("target_low")
            hi = r.get("target_high")
            target_str = f"Target: ${target:.0f}"
            if lo and hi:
                target_str += f" (${lo:.0f}-${hi:.0f})"
            parts.append(target_str)

        lines.append(" | ".join(parts))

    text = "\n".join(lines)
    print(f"  [Earnings] {len(results)} empresas procesadas.", flush=True)
    _cache_set("earnings_data", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# FUNCION PRINCIPAL: cargar todo el contexto adicional
# ═══════════════════════════════════════════════════════════════════════════

def load_macro_context() -> dict[str, str]:
    """Carga los tres tipos de contexto adicional en paralelo.

    Returns:
        Dict con keys: 'economic', 'news', 'earnings'.
        Cada valor es un string formateado listo para inyectar en el briefing.
    """
    cached = _cache_get("macro_context_all")
    if cached is not None:
        print("  [Contexto macro] Cargado desde cache.", flush=True)
        return cached

    print("\n" + "=" * 60, flush=True)
    print("CARGANDO CONTEXTO MACRO ADICIONAL", flush=True)
    print("=" * 60, flush=True)

    result: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_econ = pool.submit(fetch_economic_data)
        fut_news = pool.submit(fetch_macro_news)
        fut_earn = pool.submit(fetch_earnings_data)

        # Cada fuente tiene su propio timeout para no bloquear las demas
        for name, fut, fallback in [
            ("economic", fut_econ, "(Datos FRED no disponibles)"),
            ("news", fut_news, "(Noticias no disponibles)"),
            ("earnings", fut_earn, "(Earnings no disponibles)"),
        ]:
            try:
                result[name] = fut.result(timeout=90)
            except Exception as e:
                print(f"  [Contexto macro] {name} fallo: {e}", flush=True)
                result[name] = fallback

    total_chars = sum(len(v) for v in result.values())
    print(f"\n  Contexto macro total: {total_chars:,} chars", flush=True)
    for k, v in result.items():
        print(f"    {k}: {len(v):,} chars ({v.count(chr(10))} lineas)", flush=True)
    print("=" * 60 + "\n", flush=True)

    _cache_set("macro_context_all", result)
    return result


if __name__ == "__main__":
    ctx = load_macro_context()
    for key, text in ctx.items():
        print(f"\n{'='*40} {key.upper()} {'='*40}")
        preview = "\n".join(text.split("\n")[:25])
        print(preview)
        print(f"... ({text.count(chr(10))} lineas totales)\n")
