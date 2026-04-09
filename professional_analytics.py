"""Analitica profesional para validar, medir y gobernar las previsiones.

Incluye:
- Deteccion de regimen de mercado.
- Resumen de riesgo y pesos ajustados de cartera.
- Extraccion de senales por agente.
- Metamodelo de confianza y capital a desplegar.
- Persistencia y lectura de metadatos de ejecucion.
- Reporte historico con benchmark, drawdown, Sharpe y accuracy por agente/evaluador.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from market_data import TICKERS


RUN_METADATA_BEGIN = "[RUN_METADATA_JSON_BEGIN]"
RUN_METADATA_END = "[RUN_METADATA_JSON_END]"
DB_PATH = Path(__file__).resolve().parent / "history.db"

EVALUATOR_COLUMNS = [
    "Tecnico",
    "Fundamental",
    "Riesgo",
    "Sentimiento",
    "Elliott",
    "MACD",
    "Institucional",
    "Wyckoff",
    "Relativo",
]

SECTOR_FALLBACKS = {
    "ETF": "ETF",
    "Commodity": "Commodity",
    "Cripto": "Cripto",
}


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: round((max(v, 0.0) / total) * 100, 1) for k, v in weights.items()}


def _extract_price_field(raw_prices: dict[str, dict[str, Any]], ticker: str, field: str) -> float | None:
    row = raw_prices.get(ticker, {})
    return _to_float(row.get(field))


def detect_market_regime(raw_prices: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Clasifica un regimen de mercado usable para modular riesgo y despliegue."""
    spy_3m = _extract_price_field(raw_prices, "SPY", "ret_3m") or 0.0
    qqq_3m = _extract_price_field(raw_prices, "QQQ", "ret_3m") or 0.0
    tlt_3m = _extract_price_field(raw_prices, "TLT", "ret_3m") or 0.0
    gld_3m = _extract_price_field(raw_prices, "GLD", "ret_3m") or 0.0
    btc_3m = _extract_price_field(raw_prices, "BTC-USD", "ret_3m") or 0.0
    vix = _extract_price_field(raw_prices, "^VIX", "precio") or 18.0
    dxy = _extract_price_field(raw_prices, "DX-Y.NYB", "ret_3m") or 0.0

    qqq_lead = qqq_3m - spy_3m
    risk_level = "medio"
    regime = "mixto"
    stance = "neutral"
    deploy_base = 50
    notes: list[str] = []

    if vix >= 28 or (spy_3m <= -8 and tlt_3m >= 2):
        regime = "risk_off"
        stance = "defensivo"
        risk_level = "alto"
        deploy_base = 25
        notes.append("Volatilidad elevada y sesgo de refugio en bonos.")
    elif spy_3m >= 6 and qqq_lead >= 2 and vix <= 18:
        regime = "growth_risk_on"
        stance = "pro-crecimiento"
        risk_level = "bajo"
        deploy_base = 75
        notes.append("Liderazgo growth con volatilidad contenida.")
    elif gld_3m >= max(spy_3m, 0) and vix >= 20:
        regime = "inflacion_defensiva"
        stance = "barbell"
        risk_level = "medio-alto"
        deploy_base = 40
        notes.append("Oro fuerte y volatilidad creciente sugieren sesgo defensivo con inflacion.")
    elif spy_3m > 0 and tlt_3m < 0 and dxy > 1:
        regime = "higher_for_longer"
        stance = "selectivo"
        risk_level = "medio"
        deploy_base = 45
        notes.append("Renta variable aguanta, pero bonos flojos y dolar fuerte penalizan duracion.")
    elif btc_3m > 10 and qqq_3m > 5 and vix < 22:
        regime = "liquidez_expansiva"
        stance = "agresivo controlado"
        risk_level = "medio"
        deploy_base = 65
        notes.append("Beta alta liderando, compatible con expansion de liquidez o apetito especulativo.")
    else:
        notes.append("Señales cruzadas entre equity, bonos y refugios.")

    return {
        "name": regime,
        "stance": stance,
        "risk_level": risk_level,
        "deploy_base": deploy_base,
        "signals": {
            "spy_3m": round(spy_3m, 1),
            "qqq_3m": round(qqq_3m, 1),
            "qqq_vs_spy": round(qqq_lead, 1),
            "tlt_3m": round(tlt_3m, 1),
            "gld_3m": round(gld_3m, 1),
            "btc_3m": round(btc_3m, 1),
            "vix": round(vix, 1),
            "dxy_3m": round(dxy, 1),
        },
        "summary": " ".join(notes),
    }


def summarize_portfolio_risk(
    top_assets: list[dict[str, Any]],
    raw_prices: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resume concentracion, volatilidad y propone pesos ajustados por riesgo."""
    if not top_assets:
        return {
            "warnings": ["Sin activos en cartera."],
            "type_weights": {},
            "sector_weights": {},
            "adjusted_weights": {},
        }

    type_weights: dict[str, float] = {}
    sector_weights: dict[str, float] = {}
    original_weights: dict[str, float] = {}
    weighted_vol = 0.0
    vol_weight = 0.0
    warnings: list[str] = []
    inv_vol_weights: dict[str, float] = {}

    for asset in top_assets:
        ticker = asset["ticker"]
        weight = float(asset.get("peso", 0.0))
        asset_type = asset.get("tipo", "Accion")
        sector = fundamentals.get(ticker, {}).get("sector") or SECTOR_FALLBACKS.get(asset_type, asset_type)
        vol = _extract_price_field(raw_prices, ticker, "vol_20d") or 35.0

        original_weights[ticker] = weight
        type_weights[asset_type] = type_weights.get(asset_type, 0.0) + weight
        sector_weights[sector] = sector_weights.get(sector, 0.0) + weight
        weighted_vol += weight * vol
        vol_weight += weight
        inv_vol_weights[ticker] = 1.0 / max(vol, 5.0)

    top3_concentration = sum(sorted(original_weights.values(), reverse=True)[:3])
    max_position = max(original_weights.values()) if original_weights else 0.0
    avg_vol = weighted_vol / vol_weight if vol_weight else 0.0
    hhi = sum((w / 100.0) ** 2 for w in original_weights.values())

    max_sector = max(sector_weights.values()) if sector_weights else 0.0
    max_type = max(type_weights.values()) if type_weights else 0.0

    if top3_concentration > 34:
        warnings.append(f"Top 3 demasiado concentrado ({top3_concentration:.1f}%).")
    if max_position > 9:
        warnings.append(f"Hay posiciones individuales demasiado grandes (max {max_position:.1f}%).")
    if max_sector > 35:
        warnings.append(f"Concentracion sectorial elevada (max {max_sector:.1f}%).")
    if max_type > 70:
        warnings.append(f"Dependencia excesiva de un tipo de activo (max {max_type:.1f}%).")
    if avg_vol > 42:
        warnings.append(f"Volatilidad implícita alta en cartera ({avg_vol:.1f}% anualizada).")
    if not warnings:
        warnings.append("Estructura de cartera razonablemente equilibrada para el perfil actual.")

    adjusted_raw = _normalize_weights(inv_vol_weights)
    adjusted_capped = {ticker: min(max(weight, 2.0), 9.5) for ticker, weight in adjusted_raw.items()}
    adjusted_weights = _normalize_weights(adjusted_capped)

    return {
        "top3_concentration": round(top3_concentration, 1),
        "max_position": round(max_position, 1),
        "weighted_volatility": round(avg_vol, 1),
        "hhi": round(hhi, 3),
        "type_weights": _normalize_weights(type_weights),
        "sector_weights": _normalize_weights(sector_weights),
        "warnings": warnings,
        "adjusted_weights": adjusted_weights,
        "vol_target_hint": "Reducir exposure si VIX > 25 o si la volatilidad ponderada supera 40%.",
    }


def extract_agent_signals(transcript: list[str], valid_tickers: set[str] | None = None) -> dict[str, Any]:
    """Extrae menciones de tickers por agente a partir del transcript interno."""
    ticker_universe = valid_tickers or set(TICKERS.keys())
    token_re = re.compile(r"\b[A-Z0-9]{1,6}(?:-[A-Z0-9]{1,6})?(?:\.[A-Z]{1,4})?\b")
    signals: dict[str, Any] = {}

    for line in transcript:
        if ":" not in line:
            continue
        agent_name, _, message = line.partition(":")
        mentions = sorted({tok for tok in token_re.findall(message.upper()) if tok in ticker_universe})
        if not mentions:
            continue
        entry = signals.setdefault(agent_name.strip(), {"mentions": [], "count": 0})
        entry["mentions"] = sorted(set(entry["mentions"]) | set(mentions))
        entry["count"] = len(entry["mentions"])

    return signals


def parse_confidence_from_text(text: str) -> dict[str, Any]:
    score_match = re.search(r"score de confianza\s*[:\-]\s*(\d{1,3})/100", text, flags=re.IGNORECASE)
    deploy_match = re.search(r"capital a desplegar\s*[:\-]\s*(\d{1,3})%", text, flags=re.IGNORECASE)
    verdict_match = re.search(r"dictamen final\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE)
    issues = [
        line.strip("-• ")
        for line in text.splitlines()
        if line.strip().startswith(("-", "•"))
    ]
    return {
        "llm_score": int(score_match.group(1)) if score_match else None,
        "llm_deploy": int(deploy_match.group(1)) if deploy_match else None,
        "llm_verdict": verdict_match.group(1).strip() if verdict_match else "",
        "issues": issues[:8],
        "raw_text": text.strip(),
    }


def parse_verdict_table(verdict_output: str) -> pd.DataFrame:
    lines = [line.strip() for line in verdict_output.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return pd.DataFrame()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows: list[list[str]] = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(cells[:len(header)])
    return pd.DataFrame(rows, columns=header)


def build_confidence_model(
    verdict_output: str,
    devil_advocate: dict[str, Any],
    regime: dict[str, Any],
    risk_summary: dict[str, Any],
) -> dict[str, Any]:
    """Combina calidad de señal, régimen y riesgo en una recomendación de capital."""
    verdict_df = parse_verdict_table(verdict_output)
    buy = watch = avoid = 0
    if not verdict_df.empty and "DECISION FINAL" in verdict_df.columns:
        decisions = verdict_df["DECISION FINAL"].astype(str).str.upper()
        buy = int(decisions.str.contains("COMPRAR", regex=False).sum())
        watch = int(decisions.str.contains("VIGILAR", regex=False).sum())
        avoid = int(decisions.str.contains("EVITAR", regex=False).sum())

    score = 45
    score += buy * 3
    score += watch * 1
    score -= avoid * 4
    score += int(regime.get("deploy_base", 50) / 8)
    score -= max(0, len(risk_summary.get("warnings", [])) - 1) * 4

    llm_score = devil_advocate.get("llm_score")
    if llm_score is not None:
        score = round(score * 0.55 + llm_score * 0.45)
    if regime.get("risk_level") == "alto":
        score -= 10
    if risk_summary.get("top3_concentration", 0) > 38:
        score -= 8

    score = max(0, min(100, int(score)))
    if score >= 80:
        deploy = 100
    elif score >= 68:
        deploy = 75
    elif score >= 55:
        deploy = 50
    elif score >= 40:
        deploy = 25
    else:
        deploy = 0

    rationale = [
        f"{buy} activos en COMPRAR, {watch} en VIGILAR y {avoid} en EVITAR.",
        f"Regimen actual: {regime.get('name', 'mixto')} ({regime.get('stance', 'neutral')}).",
        f"Riesgo cartera: top3={risk_summary.get('top3_concentration', 0):.1f}% y vol={risk_summary.get('weighted_volatility', 0):.1f}%.",
    ]
    if devil_advocate.get("llm_verdict"):
        rationale.append(devil_advocate["llm_verdict"])

    return {
        "score": score,
        "deploy_pct": deploy,
        "rationale": rationale,
        "buy_count": buy,
        "watch_count": watch,
        "avoid_count": avoid,
    }


def emit_run_metadata(metadata: dict[str, Any]) -> None:
    payload = json.dumps(_to_jsonable(metadata), ensure_ascii=True, sort_keys=True)
    print(RUN_METADATA_BEGIN, flush=True)
    print(payload, flush=True)
    print(RUN_METADATA_END, flush=True)


def extract_run_metadata_json(full_output: str) -> dict[str, Any]:
    pattern = re.compile(
        rf"{re.escape(RUN_METADATA_BEGIN)}\s*(.*?)\s*{re.escape(RUN_METADATA_END)}",
        flags=re.DOTALL,
    )
    match = pattern.search(full_output)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def ensure_runs_metadata_column(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA table_info(runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "metadata_json" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN metadata_json TEXT")
    conn.commit()
    conn.close()


def _load_run_records(db_path: Path = DB_PATH, limit: int = 40) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    ensure_runs_metadata_column(db_path)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, timestamp, model, scoreboard_json, prices_json, metadata_json FROM runs "
        "WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        _, timestamp, model, scoreboard_json, prices_json, metadata_json = row
        try:
            scoreboard = pd.read_json(scoreboard_json, orient="records") if scoreboard_json else pd.DataFrame()
        except Exception:
            scoreboard = pd.DataFrame()
        try:
            prices = json.loads(prices_json) if prices_json else {}
        except json.JSONDecodeError:
            prices = {}
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except json.JSONDecodeError:
            metadata = {}
        result.append(
            {
                "timestamp": timestamp,
                "model": model,
                "scoreboard": scoreboard,
                "prices": prices,
                "metadata": metadata,
            }
        )
    return result


def _download_close_frame(tickers: list[str], start: datetime) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(
            tickers,
            start=(start - timedelta(days=7)).strftime("%Y-%m-%d"),
            end=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    close = data.get("Close")
    if close is None:
        return pd.DataFrame()
    if isinstance(close, pd.Series):
        ticker = tickers[0]
        return close.to_frame(name=ticker).dropna(how="all")
    return close.dropna(how="all")


def _series_return(series: pd.Series, horizon_days: int | None = None) -> float | None:
    series = series.dropna()
    if len(series) < 2:
        return None
    if horizon_days is None or horizon_days >= len(series):
        ref = series.iloc[-1]
    else:
        ref = series.iloc[horizon_days]
    if series.iloc[0] == 0:
        return None
    return ((ref / series.iloc[0]) - 1.0) * 100.0


def _compute_series_metrics(series: pd.Series) -> dict[str, float]:
    series = series.dropna()
    if len(series) < 2:
        return {"return_pct": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0}
    returns = series.pct_change().dropna()
    cummax = series.cummax()
    drawdown = ((series / cummax) - 1.0) * 100.0
    downside = returns[returns < 0]
    sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() not in (0, None) else 0.0
    sortino = (returns.mean() / downside.std() * (252 ** 0.5)) if not downside.empty and downside.std() not in (0, None) else 0.0
    return {
        "return_pct": round(((series.iloc[-1] / series.iloc[0]) - 1.0) * 100.0, 1),
        "max_drawdown": round(float(drawdown.min()), 1),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
    }


def _build_portfolio_series(close: pd.DataFrame, weights: dict[str, float], start_date: datetime) -> pd.Series:
    if close.empty or not weights:
        return pd.Series(dtype=float)
    sub = close[sorted(set(weights) & set(close.columns))].copy()
    sub = sub[sub.index >= pd.Timestamp(start_date.date())].dropna(how="all")
    if sub.empty:
        return pd.Series(dtype=float)
    portfolio = pd.Series(0.0, index=sub.index)
    norm_weights = _normalize_weights(weights)
    for ticker, weight in norm_weights.items():
        if ticker not in sub.columns:
            continue
        series = sub[ticker].dropna()
        if series.empty:
            continue
        rebased = series / series.iloc[0] * 100.0
        portfolio = portfolio.add(rebased.reindex(sub.index).ffill() * (weight / 100.0), fill_value=0.0)
    return portfolio.dropna()


def _build_benchmark_series(close: pd.DataFrame, start_date: datetime, ticker: str = "SPY") -> pd.Series:
    if ticker not in close.columns:
        return pd.Series(dtype=float)
    series = close[ticker]
    series = series[series.index >= pd.Timestamp(start_date.date())].dropna()
    if series.empty:
        return pd.Series(dtype=float)
    return series / series.iloc[0] * 100.0


def compute_professional_report(db_path: Path = DB_PATH, limit: int = 30) -> dict[str, Any]:
    """Calcula reporte historico de calidad, alpha y precision por factor."""
    records = _load_run_records(db_path=db_path, limit=limit)
    if not records:
        return {}

    parsed_dates = [datetime.fromisoformat(record["timestamp"]) for record in records if record.get("timestamp")]
    if not parsed_dates:
        return {}
    earliest = min(parsed_dates)

    tickers: set[str] = {"SPY"}
    for record in records:
        metadata = record.get("metadata") or {}
        top20 = metadata.get("top20") or []
        if top20:
            tickers.update(asset.get("ticker", "") for asset in top20 if asset.get("ticker"))
        scoreboard = record.get("scoreboard")
        if isinstance(scoreboard, pd.DataFrame) and not scoreboard.empty and "Ticker" in scoreboard.columns:
            tickers.update(scoreboard["Ticker"].astype(str).str.upper().tolist())

    close = _download_close_frame(sorted(tickers), start=earliest)
    if close.empty:
        return {}

    portfolio_rows: list[dict[str, Any]] = []
    evaluator_stats = {col: {"hits": 0, "total": 0} for col in EVALUATOR_COLUMNS}
    agent_stats: dict[str, dict[str, int]] = {}
    regime_stats: dict[str, list[float]] = {}

    for record in records:
        timestamp = datetime.fromisoformat(record["timestamp"])
        metadata = record.get("metadata") or {}
        top20 = metadata.get("top20") or []
        weights = {asset.get("ticker"): float(asset.get("peso", 0.0)) for asset in top20 if asset.get("ticker")}
        if not weights and isinstance(record.get("scoreboard"), pd.DataFrame):
            scoreboard = record["scoreboard"]
            if not scoreboard.empty and "%Cartera" in scoreboard.columns and "Ticker" in scoreboard.columns:
                weights = {
                    str(row["Ticker"]).upper(): _to_float(str(row["%Cartera"]).replace("%", "")) or 0.0
                    for _, row in scoreboard.iterrows()
                }

        portfolio_series = _build_portfolio_series(close, weights, timestamp)
        bench_series = _build_benchmark_series(close, timestamp, ticker="SPY")
        if portfolio_series.empty or bench_series.empty:
            continue

        port_metrics = _compute_series_metrics(portfolio_series)
        bench_metrics = _compute_series_metrics(bench_series)
        alpha = round(port_metrics["return_pct"] - bench_metrics["return_pct"], 1)
        horizon_1m = _series_return(portfolio_series, 21)
        horizon_3m = _series_return(portfolio_series, 63)
        horizon_6m = _series_return(portfolio_series, 126)
        regime_name = metadata.get("market_regime", {}).get("name", "sin_regimen")

        portfolio_rows.append(
            {
                "timestamp": record["timestamp"][:10],
                "run_type": metadata.get("run_type", "debate"),
                "return_pct": port_metrics["return_pct"],
                "benchmark_pct": bench_metrics["return_pct"],
                "alpha": alpha,
                "max_drawdown": port_metrics["max_drawdown"],
                "sharpe": port_metrics["sharpe"],
                "sortino": port_metrics["sortino"],
                "horizon_1m": round(horizon_1m, 1) if horizon_1m is not None else None,
                "horizon_3m": round(horizon_3m, 1) if horizon_3m is not None else None,
                "horizon_6m": round(horizon_6m, 1) if horizon_6m is not None else None,
                "regime": regime_name,
                "confidence": metadata.get("confidence_model", {}).get("score"),
            }
        )
        regime_stats.setdefault(regime_name, []).append(alpha)

        scoreboard = record.get("scoreboard")
        if isinstance(scoreboard, pd.DataFrame) and not scoreboard.empty and "Ticker" in scoreboard.columns:
            for _, row in scoreboard.iterrows():
                ticker = str(row["Ticker"]).upper()
                if ticker not in close.columns:
                    continue
                future_series = close[ticker]
                future_series = future_series[future_series.index >= pd.Timestamp(timestamp.date())].dropna()
                future_ret = _series_return(future_series, 63)
                if future_ret is None:
                    future_ret = _series_return(future_series, None)
                if future_ret is None:
                    continue
                for col in EVALUATOR_COLUMNS:
                    if col not in scoreboard.columns:
                        continue
                    verdict = str(row.get(col, "")).upper()
                    if verdict not in {"OK", "NOK"}:
                        continue
                    evaluator_stats[col]["total"] += 1
                    hit = (verdict == "OK" and future_ret > 0) or (verdict == "NOK" and future_ret <= 0)
                    if hit:
                        evaluator_stats[col]["hits"] += 1

        for agent, payload in (metadata.get("agent_signals") or {}).items():
            stats = agent_stats.setdefault(agent, {"hits": 0, "total": 0})
            for ticker in payload.get("mentions", []):
                if ticker not in close.columns:
                    continue
                future_series = close[ticker]
                future_series = future_series[future_series.index >= pd.Timestamp(timestamp.date())].dropna()
                future_ret = _series_return(future_series, 63)
                if future_ret is None:
                    future_ret = _series_return(future_series, None)
                if future_ret is None:
                    continue
                stats["total"] += 1
                if future_ret > 0:
                    stats["hits"] += 1

    if not portfolio_rows:
        return {}

    portfolio_df = pd.DataFrame(portfolio_rows)
    avg_alpha = round(portfolio_df["alpha"].mean(), 1)
    beat_rate = round((portfolio_df["alpha"] > 0).mean() * 100, 1)
    avg_return = round(portfolio_df["return_pct"].mean(), 1)
    avg_dd = round(portfolio_df["max_drawdown"].mean(), 1)
    avg_sharpe = round(portfolio_df["sharpe"].mean(), 2)
    annualized = round(((1 + (avg_return / 100)) ** max(1, 12 / 6) - 1) * 100, 1)

    evaluator_report = [
        {
            "Evaluador": name,
            "Hit rate %": round((values["hits"] / values["total"] * 100), 1) if values["total"] else 0.0,
            "Aciertos": values["hits"],
            "Total": values["total"],
        }
        for name, values in evaluator_stats.items()
        if values["total"]
    ]
    evaluator_report.sort(key=lambda row: row["Hit rate %"], reverse=True)

    agent_report = [
        {
            "Agente": name,
            "Hit rate %": round((values["hits"] / values["total"] * 100), 1) if values["total"] else 0.0,
            "Aciertos": values["hits"],
            "Total": values["total"],
        }
        for name, values in agent_stats.items()
        if values["total"]
    ]
    agent_report.sort(key=lambda row: row["Hit rate %"], reverse=True)

    regime_report = [
        {
            "Regimen": regime,
            "Alpha medio %": round(sum(values) / len(values), 1),
            "Runs": len(values),
        }
        for regime, values in regime_stats.items()
    ]
    regime_report.sort(key=lambda row: row["Alpha medio %"], reverse=True)

    return {
        "summary": {
            "runs": int(len(portfolio_df)),
            "avg_return": avg_return,
            "annualized_return": annualized,
            "avg_alpha": avg_alpha,
            "beat_rate": beat_rate,
            "avg_drawdown": avg_dd,
            "avg_sharpe": avg_sharpe,
        },
        "portfolio_runs": portfolio_df.sort_values("timestamp", ascending=False).to_dict(orient="records"),
        "evaluators": evaluator_report,
        "agents": agent_report,
        "regimes": regime_report,
    }
