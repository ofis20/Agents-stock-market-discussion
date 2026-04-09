"""Gestor de cartera personal — posiciones reales, señales de salida, performance tracking y niveles de entrada."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

DB_PATH = Path(__file__).parent / "history.db"

# ---------------------------------------------------------------------------
# Base de datos — tabla de posiciones
# ---------------------------------------------------------------------------

def _ensure_portfolio_tables() -> None:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            shares REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT NOT NULL,
            notes TEXT DEFAULT '',
            closed INTEGER DEFAULT 0,
            close_price REAL,
            close_date TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# CRUD posiciones
# ---------------------------------------------------------------------------

def add_position(ticker: str, shares: float, entry_price: float,
                 entry_date: str | None = None, notes: str = "") -> int:
    """Registra una posición nueva. Devuelve el id."""
    _ensure_portfolio_tables()
    if entry_date is None:
        entry_date = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute(
        "INSERT INTO portfolio (ticker, shares, entry_price, entry_date, notes) VALUES (?, ?, ?, ?, ?)",
        (ticker.upper().strip(), shares, entry_price, entry_date, notes),
    )
    row_id = cur.lastrowid
    conn.commit()
    conn.close()
    return row_id


def close_position(position_id: int, close_price: float,
                   close_date: str | None = None) -> None:
    """Cierra (vende) una posición existente."""
    _ensure_portfolio_tables()
    if close_date is None:
        close_date = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "UPDATE portfolio SET closed = 1, close_price = ?, close_date = ? WHERE id = ?",
        (close_price, close_date, position_id),
    )
    conn.commit()
    conn.close()


def delete_position(position_id: int) -> None:
    """Elimina una posición (borrado permanente)."""
    _ensure_portfolio_tables()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("DELETE FROM portfolio WHERE id = ?", (position_id,))
    conn.commit()
    conn.close()


def get_open_positions() -> pd.DataFrame:
    """Devuelve todas las posiciones abiertas."""
    _ensure_portfolio_tables()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT id, ticker, shares, entry_price, entry_date, notes FROM portfolio WHERE closed = 0 ORDER BY entry_date DESC",
        conn,
    )
    conn.close()
    return df


def get_closed_positions() -> pd.DataFrame:
    """Devuelve posiciones cerradas (historial)."""
    _ensure_portfolio_tables()
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT id, ticker, shares, entry_price, entry_date, close_price, close_date, notes "
        "FROM portfolio WHERE closed = 1 ORDER BY close_date DESC",
        conn,
    )
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Snapshot de cartera (precios actuales, P&L, exposición)
# ---------------------------------------------------------------------------

def portfolio_snapshot() -> pd.DataFrame:
    """Calcula valor actual, P&L y peso de cada posición abierta."""
    positions = get_open_positions()
    if positions.empty:
        return pd.DataFrame()

    tickers = list(positions["ticker"].unique())
    try:
        data = yf.download(tickers, period="5d", progress=False)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"]
    except Exception:
        return pd.DataFrame()

    rows = []
    for _, pos in positions.iterrows():
        t = pos["ticker"]
        try:
            if len(tickers) > 1:
                price_now = float(close[t].dropna().iloc[-1])
            else:
                price_now = float(close.dropna().iloc[-1])
        except Exception:
            price_now = 0.0

        cost = pos["shares"] * pos["entry_price"]
        value = pos["shares"] * price_now
        pnl = value - cost
        pnl_pct = ((price_now - pos["entry_price"]) / pos["entry_price"]) * 100 if pos["entry_price"] > 0 else 0.0

        rows.append({
            "id": pos["id"],
            "Ticker": t,
            "Acciones": pos["shares"],
            "Precio entrada": pos["entry_price"],
            "Fecha entrada": pos["entry_date"],
            "Precio actual": round(price_now, 2),
            "Coste": round(cost, 2),
            "Valor": round(value, 2),
            "P&L": round(pnl, 2),
            "P&L %": round(pnl_pct, 1),
            "Notas": pos["notes"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    total_value = df["Valor"].sum()
    df["Peso %"] = (df["Valor"] / total_value * 100).round(1) if total_value > 0 else 0.0
    return df


def portfolio_risk_summary(snapshot: pd.DataFrame) -> dict[str, Any]:
    """Resumen de riesgo agregado de la cartera."""
    if snapshot.empty:
        return {}

    total_cost = snapshot["Coste"].sum()
    total_value = snapshot["Valor"].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0.0

    # Concentración: peso del top-3
    top3_weight = snapshot.nlargest(3, "Peso %")["Peso %"].sum() if len(snapshot) >= 3 else 100.0

    # Sector exposure (simplificado)
    tickers = snapshot["Ticker"].tolist()
    sector_map: dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            sector_map[t] = info.get("sector", "Desconocido")
        except Exception:
            sector_map[t] = "Desconocido"

    snapshot_with_sector = snapshot.copy()
    snapshot_with_sector["Sector"] = snapshot_with_sector["Ticker"].map(sector_map)
    sector_exposure = snapshot_with_sector.groupby("Sector")["Peso %"].sum().sort_values(ascending=False)

    # Posiciones en pérdida
    losing = snapshot[snapshot["P&L %"] < 0]
    winning = snapshot[snapshot["P&L %"] >= 0]

    return {
        "total_cost": round(total_cost, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 1),
        "num_positions": len(snapshot),
        "num_winning": len(winning),
        "num_losing": len(losing),
        "top3_concentration": round(top3_weight, 1),
        "sector_exposure": sector_exposure.to_dict(),
        "max_loss_ticker": losing.nsmallest(1, "P&L %")["Ticker"].iloc[0] if not losing.empty else "-",
        "max_loss_pct": round(losing["P&L %"].min(), 1) if not losing.empty else 0.0,
        "max_gain_ticker": winning.nlargest(1, "P&L %")["Ticker"].iloc[0] if not winning.empty else "-",
        "max_gain_pct": round(winning["P&L %"].max(), 1) if not winning.empty else 0.0,
    }


# ---------------------------------------------------------------------------
# Señales de SALIDA — basadas en deterioro de tesis
# ---------------------------------------------------------------------------

def compute_exit_signals(score_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Genera señales de salida comparando posiciones abiertas con scoreboard actual.

    Señales:
    - VENDER: Score bajó a EVITAR y tienes posición abierta
    - REDUCIR: Score bajó de COMPRAR a VIGILAR
    - SALIÓ TOP20: Tenías posición pero ya no aparece en el Top 20
    - STOP-LOSS: P&L < -15%
    - TOMAR BENEFICIOS: P&L > +25%
    """
    snapshot = portfolio_snapshot()
    if snapshot.empty:
        return []

    signals: list[dict[str, Any]] = []
    score_lookup: dict[str, dict] = {}
    if not score_df.empty and "Ticker" in score_df.columns:
        for _, row in score_df.iterrows():
            t = str(row["Ticker"]).upper()
            score_lookup[t] = {
                "puntaje": int(row.get("Puntaje", 0)),
                "decision": str(row.get("Decision sugerida", "")),
            }

    # Cargar historial para detectar caída de score
    history = _load_score_history(5)

    for _, pos in snapshot.iterrows():
        ticker = pos["Ticker"]
        pnl_pct = pos["P&L %"]

        current = score_lookup.get(ticker)
        prev_scores = [h.get(ticker, {}).get("puntaje") for h in history if h.get(ticker)]

        # Señal por P&L
        if pnl_pct <= -15:
            signals.append({
                "ticker": ticker, "señal": "STOP-LOSS", "urgencia": "ALTA",
                "motivo": f"Pérdida del {pnl_pct:.1f}%. Proteger capital.",
                "pnl_pct": pnl_pct,
            })
        elif pnl_pct >= 25:
            signals.append({
                "ticker": ticker, "señal": "TOMAR BENEFICIOS", "urgencia": "MEDIA",
                "motivo": f"Ganancia del {pnl_pct:+.1f}%. Considerar venta parcial.",
                "pnl_pct": pnl_pct,
            })

        # Señal por tesis
        if current is None:
            signals.append({
                "ticker": ticker, "señal": "FUERA DEL TOP 20", "urgencia": "MEDIA",
                "motivo": "Ya no aparece en las recomendaciones del sistema.",
                "pnl_pct": pnl_pct,
            })
        elif current["decision"] == "EVITAR":
            signals.append({
                "ticker": ticker, "señal": "VENDER", "urgencia": "ALTA",
                "motivo": f"Decisión actual: EVITAR (score {current['puntaje']}/7). La tesis se ha deteriorado.",
                "pnl_pct": pnl_pct,
            })
        elif current["decision"] == "VIGILAR":
            # Solo es señal de REDUCIR si antes era COMPRAR
            was_buy = any(h.get(ticker, {}).get("decision") == "COMPRAR" for h in history)
            if was_buy:
                signals.append({
                    "ticker": ticker, "señal": "REDUCIR", "urgencia": "MEDIA",
                    "motivo": f"Bajó de COMPRAR a VIGILAR (score {current['puntaje']}/7). Considerar reducir posición.",
                    "pnl_pct": pnl_pct,
                })

        # Señal por caída de score persistente
        if prev_scores and current:
            recent_max = max(s for s in prev_scores if s is not None) if prev_scores else 0
            if recent_max - current["puntaje"] >= 3:
                # Evitar duplicar si ya alertamos VENDER
                already_sell = any(s["ticker"] == ticker and s["señal"] == "VENDER" for s in signals)
                if not already_sell:
                    signals.append({
                        "ticker": ticker, "señal": "DETERIORO", "urgencia": "ALTA",
                        "motivo": f"Score cayó de {recent_max}/7 a {current['puntaje']}/7 en ejecuciones recientes.",
                        "pnl_pct": pnl_pct,
                    })

    # Ordenar por urgencia
    urgency_order = {"ALTA": 0, "MEDIA": 1, "BAJA": 2}
    signals.sort(key=lambda s: urgency_order.get(s["urgencia"], 9))
    return signals


def _load_score_history(limit: int = 5) -> list[dict[str, dict]]:
    """Carga el historial de scores como lista de dicts ticker -> {puntaje, decision}."""
    _ensure_portfolio_tables()
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT scoreboard_json FROM runs WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()

    result = []
    for (sb_json,) in rows:
        try:
            df = pd.read_json(sb_json, orient="records")
            entry: dict[str, dict] = {}
            if not df.empty and "Ticker" in df.columns:
                for _, row in df.iterrows():
                    t = str(row["Ticker"]).upper()
                    entry[t] = {
                        "puntaje": int(row.get("Puntaje", 0)),
                        "decision": str(row.get("Decision sugerida", "")),
                    }
            result.append(entry)
        except Exception:
            continue
    return result


# ---------------------------------------------------------------------------
# Performance tracking — hit rate de recomendaciones
# ---------------------------------------------------------------------------

def compute_performance_report() -> dict[str, Any]:
    """Analiza el rendimiento real de las recomendaciones pasadas.

    Para cada recomendación COMPRAR guardada en el histórico, mide:
    - Rentabilidad desde la fecha de recomendación
    - Aciertos (>0%) vs fallos (<0%)
    - Rentabilidad media, mediana, mejor, peor
    """
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT timestamp, scoreboard_json, prices_json FROM runs "
        "WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' AND prices_json IS NOT NULL "
        "ORDER BY id ASC",
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    # Recopilar todas las recomendaciones COMPRAR con precios
    recs: list[dict] = []
    for ts, sb_json, prices_json in rows:
        try:
            sb = pd.read_json(sb_json, orient="records")
            prices = json.loads(prices_json) if prices_json else {}
            if sb.empty or not prices:
                continue
            rec_date = ts[:10]
            for _, row in sb.iterrows():
                ticker = str(row.get("Ticker", "")).upper()
                decision = str(row.get("Decision sugerida", "")).upper()
                puntaje = int(row.get("Puntaje", 0))
                if ticker in prices and prices[ticker] > 0:
                    recs.append({
                        "ticker": ticker,
                        "decision": decision,
                        "puntaje": puntaje,
                        "price_at_rec": prices[ticker],
                        "date": rec_date,
                    })
        except Exception:
            continue

    if not recs:
        return {}

    # Obtener precios actuales de todos los tickers
    all_tickers = list({r["ticker"] for r in recs})
    try:
        data = yf.download(all_tickers, period="5d", progress=False)
        if data.empty:
            return {}
        close = data["Close"]
        current_prices: dict[str, float] = {}
        for t in all_tickers:
            try:
                if len(all_tickers) > 1:
                    current_prices[t] = float(close[t].dropna().iloc[-1])
                else:
                    current_prices[t] = float(close.dropna().iloc[-1])
            except Exception:
                pass
    except Exception:
        return {}

    # Calcular performance por recomendación
    comprar_results = []
    vigilar_results = []
    evitar_results = []

    seen: set[tuple[str, str]] = set()  # (ticker, date) para deduplicar

    for rec in recs:
        key = (rec["ticker"], rec["date"])
        if key in seen:
            continue
        seen.add(key)

        t = rec["ticker"]
        if t not in current_prices:
            continue

        ret = ((current_prices[t] - rec["price_at_rec"]) / rec["price_at_rec"]) * 100

        entry = {
            "ticker": t,
            "date": rec["date"],
            "price_rec": rec["price_at_rec"],
            "price_now": current_prices[t],
            "return_pct": round(ret, 1),
            "puntaje": rec["puntaje"],
        }

        if rec["decision"] == "COMPRAR":
            comprar_results.append(entry)
        elif rec["decision"] == "VIGILAR":
            vigilar_results.append(entry)
        else:
            evitar_results.append(entry)

    def _stats(results: list[dict]) -> dict[str, Any]:
        if not results:
            return {"count": 0}
        returns = [r["return_pct"] for r in results]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        return {
            "count": len(results),
            "hit_rate": round(len(wins) / len(results) * 100, 1) if results else 0,
            "avg_return": round(sum(returns) / len(returns), 1),
            "median_return": round(sorted(returns)[len(returns) // 2], 1),
            "best": round(max(returns), 1),
            "worst": round(min(returns), 1),
            "wins": len(wins),
            "losses": len(losses),
            "details": sorted(results, key=lambda x: x["return_pct"], reverse=True),
        }

    return {
        "COMPRAR": _stats(comprar_results),
        "VIGILAR": _stats(vigilar_results),
        "EVITAR": _stats(evitar_results),
        "total_recommendations": len(seen),
    }


# ---------------------------------------------------------------------------
# Niveles de entrada — soportes técnicos y zonas óptimas
# ---------------------------------------------------------------------------

def compute_entry_levels(tickers: list[str]) -> list[dict[str, Any]]:
    """Calcula niveles de entrada óptimos basados en soportes técnicos.

    Para cada ticker:
    - Soporte 1: SMA200 (soporte de largo plazo)
    - Soporte 2: Mínimo de 20 sesiones
    - Soporte 3: Banda inferior de Bollinger
    - Zona óptima: Media de los 3 soportes
    - Señal: AHORA si precio cerca de zona, ESPERAR si sobrecomprado
    """
    if not tickers:
        return []

    end = datetime.now()
    start = end - timedelta(days=400)
    try:
        data = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                           end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return []
    except Exception:
        return []

    results = []
    for t in tickers:
        try:
            if len(tickers) > 1:
                c = data["Close"][t].dropna()
                low = data["Low"][t].dropna()
            else:
                c = data["Close"].dropna()
                low = data["Low"].dropna()

            if len(c) < 50:
                continue

            price = float(c.iloc[-1])

            # SMA200
            sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None
            # SMA50
            sma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else None
            # Mínimo 20 sesiones
            low_20 = float(low.tail(20).min())
            # Bollinger inferior
            sma20 = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            bb_lower = float((sma20 - 2 * std20).iloc[-1])
            # RSI
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14).mean()
            loss_s = (-delta.clip(upper=0)).ewm(alpha=1/14, min_periods=14).mean()
            rs = gain / loss_s
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            # Niveles de soporte
            supports = [s for s in [sma200, low_20, bb_lower] if s is not None and s > 0]
            zona_optima = sum(supports) / len(supports) if supports else price * 0.95

            # Distancia al soporte más cercano
            dist_to_support = ((price - zona_optima) / price) * 100

            # Señal de entrada
            if rsi < 30 and dist_to_support < 5:
                timing = "AHORA (sobreventa + cerca de soporte)"
            elif dist_to_support < 3:
                timing = "AHORA (en zona de soporte)"
            elif rsi > 70:
                timing = "ESPERAR (sobrecompra, RSI > 70)"
            elif dist_to_support > 10:
                timing = f"ESPERAR (a {dist_to_support:.0f}% del soporte)"
            else:
                timing = "CERCANO (monitorizar para entrada)"

            # Retrocesos Fibonacci desde último máximo
            high_52w = float(c.tail(252).max()) if len(c) >= 252 else float(c.max())
            low_52w = float(c.tail(252).min()) if len(c) >= 252 else float(c.min())
            fib_range = high_52w - low_52w

            results.append({
                "Ticker": t,
                "Precio": round(price, 2),
                "SMA50": round(sma50, 2) if sma50 else None,
                "SMA200": round(sma200, 2) if sma200 else None,
                "Min 20d": round(low_20, 2),
                "BB inf.": round(bb_lower, 2),
                "Zona óptima": round(zona_optima, 2),
                "Dist. soporte": round(dist_to_support, 1),
                "RSI": round(rsi, 1),
                "Fib 38.2%": round(high_52w - fib_range * 0.382, 2),
                "Fib 50%": round(high_52w - fib_range * 0.500, 2),
                "Fib 61.8%": round(high_52w - fib_range * 0.618, 2),
                "Timing": timing,
            })
        except Exception:
            continue

    return results
