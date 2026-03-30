#!/usr/bin/env python3
"""Interfaz Streamlit para ejecutar y visualizar el debate macro en vivo."""

from __future__ import annotations

import html
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


APP_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = APP_DIR / "ollama_macro_debate.py"
HISTORY_DB = APP_DIR / "history.db"

SECTION_HINTS = {
    "ONDAS DE ELLIOTT": "Ondas de Elliott",
    "REVISION DE ANALISIS TECNICO": "Analisis tecnico",
    "REVISION DE ANALISIS FUNDAMENTAL": "Analisis fundamental",
    "REVISION DE GESTION DE RIESGOS": "Gestion de riesgos",
    "REVISION DE SENTIMIENTO DE MERCADO": "Sentimiento de mercado",
    "REVISION DE MACD": "MACD",
    "CARTERAS REALES DE GURUS": "Carteras de Gurus",
    "VEREDICTO FINAL CONSOLIDADO": "Veredicto final",
}

SCORE_TABLE_KEYS = {
    "Analisis tecnico": "Tecnico",
    "Analisis fundamental": "Fundamental",
    "Gestion de riesgos": "Riesgo",
    "Sentimiento de mercado": "Sentimiento",
    "Ondas de Elliott": "Elliott",
    "MACD": "MACD",
}


def _extract_percent_value(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(value))
    return float(match.group(1)) if match else 0.0


def build_command(model: str, host: str, seconds: int, max_turns: int, context_lines: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(SCRIPT_PATH),
        "--model",
        model,
        "--host",
        host,
        "--seconds",
        str(seconds),
        "--max-turns",
        str(max_turns),
        "--context-lines",
        str(context_lines),
    ]


def inject_app_css() -> None:
    st.markdown(
        """
        <style>
        .wrapped-log {
            background: #0b1020;
            color: #e5e7eb;
            border: 1px solid #1f2937;
            border-radius: 10px;
            padding: 0.8rem;
            max-height: 68vh;
            overflow-y: auto;
            overflow-x: hidden;
        }

        .wrapped-log pre {
            margin: 0;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-wrap: anywhere;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 0.80rem;
            line-height: 1.35;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_wrapped_log(target, text: str) -> None:
    safe = html.escape(text)
    target.markdown(
        f"<div class='wrapped-log'><pre>{safe}</pre></div>",
        unsafe_allow_html=True,
    )


def _detect_phase(line: str) -> str | None:
    upper = line.upper()
    if "CARGANDO DATOS REALES" in upper:
        return "Cargando datos de mercado..."
    if "DATOS DE MERCADO CARGADOS DESDE CACHE" in upper:
        return "Datos cargados desde cache..."
    if "DEBATE EN VIVO" in upper or line.startswith("[") and "] (modelo:" in line:
        return "Debate en curso..."
    if "SEGUNDA PASADA" in upper:
        return "Segunda pasada (sin EVITAR)..."
    if "TERCERA PASADA" in upper:
        return "Tercera pasada (filtrado final)..."
    if "TOP 20 INVERSIONES" in upper:
        return "Generando Top 20..."
    # Progreso de evaluaciones: [Evaluando X/N: TICKER]
    m = re.match(r"\[Evaluando (\d+)/(\d+): (\S+)\]", line)
    if m:
        return f"Evaluando activo {m.group(1)}/{m.group(2)}: {m.group(3)}"
    if "ANALISIS TECNICO" in upper:
        return "Analisis tecnico..."
    if "ANALISIS FUNDAMENTAL" in upper:
        return "Analisis fundamental..."
    if "GESTION DE RIESGOS" in upper:
        return "Gestion de riesgos..."
    if "SENTIMIENTO DE MERCADO" in upper:
        return "Sentimiento de mercado..."
    if "VEREDICTO FINAL" in upper:
        return "Veredicto final..."
    return None


def stream_process(cmd: list[str]) -> tuple[int, str]:
    full_output = ""
    phase_label = st.empty()
    current_phase = "Iniciando..."

    with st.status("Ejecutando analisis...", expanded=True) as status:
        phase_label.info(current_phase)

        process = subprocess.Popen(
            cmd,
            cwd=str(APP_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        line_buffer = ""
        while True:
            char = process.stdout.read(1)
            if char == "" and process.poll() is not None:
                break
            if not char:
                continue

            full_output += char
            line_buffer += char

            if "\n" in line_buffer:
                for ln in line_buffer.splitlines():
                    detected = _detect_phase(ln)
                    if detected and detected != current_phase:
                        current_phase = detected
                        phase_label.info(f"Fase actual: {current_phase}")
                line_buffer = ""

        return_code = process.wait()

        if return_code == 0:
            phase_label.empty()
            status.update(label="Analisis finalizado correctamente", state="complete", expanded=False)
        else:
            phase_label.error("Error durante la ejecucion.")
            status.update(label=f"Finalizado con error (codigo {return_code})", state="error", expanded=True)

    return return_code, full_output


def _split_pipe_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts


def _is_separator_row(cells: list[str]) -> bool:
    if not cells:
        return False
    allowed = set("-: ")
    return all(cell and set(cell) <= allowed for cell in cells)


def parse_markdown_table(table_lines: list[str]) -> pd.DataFrame:
    header = _split_pipe_row(table_lines[0])
    body_lines = table_lines[1:]

    if body_lines:
        first_row_cells = _split_pipe_row(body_lines[0])
        if _is_separator_row(first_row_cells):
            body_lines = body_lines[1:]

    rows: list[list[str]] = []
    for raw in body_lines:
        cells = _split_pipe_row(raw)
        if not cells:
            continue
        rows.append(cells)

    width = len(header)
    normalized_rows = []
    for row in rows:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        elif len(row) > width:
            row = row[:width]
        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=header)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {col.strip().upper(): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().upper()
        if key in norm:
            return norm[key]
    return None


def extract_named_tables(text: str) -> list[tuple[str, pd.DataFrame]]:
    lines = text.splitlines()
    tables: list[tuple[str, pd.DataFrame]] = []
    current_hint = "Tabla"
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        upper_line = line.upper()

        for hint, label in SECTION_HINTS.items():
            if hint in upper_line:
                current_hint = label
                break

        if line.startswith("|") and line.endswith("|"):
            block = []
            j = i
            while j < len(lines):
                candidate = lines[j].strip()
                if candidate.startswith("|") and candidate.endswith("|"):
                    block.append(candidate)
                    j += 1
                else:
                    break

            if len(block) >= 3:
                df = parse_markdown_table(block)
                if not df.empty:
                    tables.append((current_hint, df))
            i = j
            continue
        i += 1

    return tables


def extract_top10_table(text: str) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    line_re = re.compile(r"^\s*(\d+)\.\s+(.+)$")

    # Solo parsear lineas DESPUES del marker de validacion
    marker = "[Top 20 validado]"
    marker_pos = text.find(marker)
    search_text = text[marker_pos:] if marker_pos != -1 else text

    for raw_line in search_text.splitlines():
        line = raw_line.strip().replace("**", "")
        m = line_re.match(line)
        if not m:
            continue

        rank = m.group(1)
        payload = m.group(2)
        parts = [p.strip() for p in payload.split(" - ")]
        if len(parts) < 5:
            continue

        ticker = parts[0]
        nombre = parts[1]
        asset_type = parts[2]
        weight = parts[3]
        thesis = " - ".join(parts[4:])

        if not re.search(r"\d+%", weight):
            continue

        rows.append(
            {
                "#": rank,
                "Ticker": ticker,
                "Nombre": nombre,
                "Tipo": asset_type,
                "%Cartera": weight,
                "Justificacion": thesis,
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["Ticker"], keep="first")
    return df.head(20)


def extract_consensus(text: str) -> str:
    """Extrae el resumen de consenso macro (entre el marker de resumen y TOP 20)."""
    marker = "[Resumen validado - 10 lineas]"
    pos = text.find(marker)
    if pos == -1:
        return ""
    after = text[pos + len(marker):]
    # Cortar en el siguiente bloque (TOP 20 o marcador de seccion)
    end_markers = ["TOP 20 INVERSIONES", "====="]
    end_pos = len(after)
    for em in end_markers:
        idx = after.find(em)
        if idx != -1 and idx < end_pos:
            end_pos = idx
    chunk = after[:end_pos].strip()
    lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
    return "\n".join(lines[:5])


def extract_recommendations(text: str) -> list[str]:
    lines = text.splitlines()
    start = None
    for idx, raw in enumerate(lines):
        if "RECOMENDACIONES FINALES:" in raw.upper():
            start = idx + 1
            break

    if start is None:
        return []

    recos: list[str] = []
    for raw in lines[start:]:
        line = raw.strip()
        if not line or line.startswith("|"):
            continue
        cleaned = re.sub(r"^[-*\d\.)\s]+", "", line).strip()
        if cleaned:
            recos.append(cleaned)
        if len(recos) >= 3:
            break

    return recos


def render_metrics(df: pd.DataFrame) -> None:
    columns_upper = {col.upper(): col for col in df.columns}

    if "VEREDICTO" in columns_upper:
        verdict_col = columns_upper["VEREDICTO"]
        values = df[verdict_col].astype(str).str.upper()
        ok_count = int((values == "OK").sum())
        nok_count = int((values == "NOK").sum())
        c1, c2 = st.columns(2)
        c1.metric("OK", ok_count)
        c2.metric("NOK", nok_count)

    if "DECISION FINAL" in columns_upper:
        decision_col = columns_upper["DECISION FINAL"]
        values = df[decision_col].astype(str).str.upper()
        buy_count = int(values.str.contains("COMPRAR", regex=False).sum())
        watch_count = int(values.str.contains("VIGILAR", regex=False).sum())
        avoid_count = int(values.str.contains("EVITAR", regex=False).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("COMPRAR", buy_count)
        c2.metric("VIGILAR", watch_count)
        c3.metric("EVITAR", avoid_count)


def style_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    columns_upper = {col.upper(): col for col in df.columns}

    def color_cell(value: str) -> str:
        text = str(value).upper()
        if "COMPRAR" in text or text == "OK":
            return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
        if "VIGILAR" in text:
            return "background-color: #fff7e6; color: #9a6700; font-weight: 600;"
        if "EVITAR" in text or text == "NOK":
            return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
        return ""

    styled = df.style
    if "VEREDICTO" in columns_upper:
        styled = styled.map(color_cell, subset=[columns_upper["VEREDICTO"]])
    if "DECISION FINAL" in columns_upper:
        styled = styled.map(color_cell, subset=[columns_upper["DECISION FINAL"]])
    return styled


# Mapa traduccion de sectores a espanol
_SECTOR_ES = {
    "Technology": "Tecnologia",
    "Communication Services": "Comunicaciones",
    "Consumer Cyclical": "Consumo Ciclico",
    "Consumer Defensive": "Consumo Defensivo",
    "Financial Services": "Finanzas",
    "Healthcare": "Salud",
    "Industrials": "Industria",
    "Energy": "Energia",
    "Basic Materials": "Materiales",
    "Real Estate": "Inmobiliario",
    "Utilities": "Servicios Publicos",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sectors(tickers: tuple[str, ...]) -> dict[str, str]:
    """Obtiene el sector de cada ticker via yfinance."""
    result: dict[str, str] = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            raw = info.get("sector", "Otro")
            result[t] = _SECTOR_ES.get(raw, raw)
        except Exception:
            result[t] = "Desconocido"
    return result


def _make_pie(data: pd.DataFrame, theta_col: str, color_col: str, title: str) -> alt.Chart:
    """Crea un grafico de tarta (donut) con Altair y etiquetas de porcentaje."""
    total = data[theta_col].sum()
    df = data.copy()
    df["_pct"] = df[theta_col] / total * 100 if total else 0
    df["_label"] = df["_pct"].apply(lambda v: f"{v:.1f}%")

    base = alt.Chart(df).encode(
        theta=alt.Theta(f"{theta_col}:Q", stack=True),
        color=alt.Color(f"{color_col}:N", legend=alt.Legend(title=None)),
        tooltip=[f"{color_col}:N", alt.Tooltip("_pct:Q", title="%", format=".1f")],
    )

    arcs = base.mark_arc(innerRadius=50, outerRadius=120)
    labels = base.mark_text(radius=140, size=12, fontWeight="bold").encode(
        text="_label:N",
    )

    return (arcs + labels).properties(title=title, height=350)


def render_top10_charts(top10_df: pd.DataFrame) -> None:
    if top10_df.empty:
        return

    chart_df = top10_df.copy()
    chart_df["Peso"] = chart_df["%Cartera"].apply(_extract_percent_value)

    # Obtener sectores reales
    sectors = _fetch_sectors(tuple(chart_df["Ticker"].tolist()))
    chart_df["Sector"] = chart_df["Ticker"].map(sectors).fillna("Desconocido")

    c1, c2 = st.columns(2)
    with c1:
        pie1 = _make_pie(chart_df, "Peso", "Ticker", "Distribucion de cartera")
        st.altair_chart(pie1, use_container_width=True)

    with c2:
        by_sector = chart_df.groupby("Sector", as_index=False)["Peso"].sum()
        total_peso = by_sector["Peso"].sum()
        if total_peso > 0:
            by_sector["_pct_total"] = by_sector["Peso"] / total_peso * 100
            big = by_sector[by_sector["_pct_total"] >= 7].copy()
            small = by_sector[by_sector["_pct_total"] < 7]
            if not small.empty:
                otros_row = pd.DataFrame([{"Sector": "Otro", "Peso": small["Peso"].sum()}])
                by_sector = pd.concat([big[["Sector", "Peso"]], otros_row], ignore_index=True)
            else:
                by_sector = big[["Sector", "Peso"]].copy()
        pie2 = _make_pie(by_sector, "Peso", "Sector", "Peso por sector")
        st.altair_chart(pie2, use_container_width=True)


def render_decision_chart(df: pd.DataFrame) -> None:
    columns_upper = {col.upper(): col for col in df.columns}
    if "DECISION FINAL" not in columns_upper:
        return

    decision_col = columns_upper["DECISION FINAL"]
    values = df[decision_col].astype(str).str.upper()
    decisions = pd.DataFrame(
        {
            "COMPRAR": [int(values.str.contains("COMPRAR", regex=False).sum())],
            "VIGILAR": [int(values.str.contains("VIGILAR", regex=False).sum())],
            "EVITAR": [int(values.str.contains("EVITAR", regex=False).sum())],
        }
    )
    st.markdown("#### Distribucion de decisiones finales")
    st.bar_chart(decisions.T.rename(columns={0: "Activos"}), use_container_width=True)


def build_scoreboard(top10_df: pd.DataFrame, tables: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if top10_df.empty:
        return pd.DataFrame()

    score_df = top10_df[["Ticker", "Nombre", "Tipo", "%Cartera"]].copy() if "Nombre" in top10_df.columns else top10_df[["Ticker", "Tipo", "%Cartera"]].copy()

    for source_name in SCORE_TABLE_KEYS.values():
        score_df[source_name] = pd.NA

    for title, df in tables:
        if title not in SCORE_TABLE_KEYS:
            continue

        ticker_col = _find_column(df, ["Ticker"])
        verdict_col = _find_column(df, ["Veredicto", "Decision final"])
        if not ticker_col or not verdict_col:
            continue

        role_col = SCORE_TABLE_KEYS[title]
        lookup = {
            str(ticker).strip().upper(): str(verdict).strip().upper()
            for ticker, verdict in zip(df[ticker_col], df[verdict_col])
        }

        score_df[role_col] = score_df["Ticker"].astype(str).str.upper().map(lookup)

    evaluator_cols = ["Tecnico", "Fundamental", "Riesgo", "Sentimiento", "Elliott", "MACD"]
    for col in evaluator_cols:
        score_df[col] = score_df[col].fillna("N/D")

    score_df["Puntaje"] = (
        (score_df["Tecnico"] == "OK").astype(int)
        + (score_df["Fundamental"] == "OK").astype(int)
        + (score_df["Riesgo"] == "OK").astype(int)
        + (score_df["Sentimiento"] == "OK").astype(int)
        + (score_df["Elliott"] == "OK").astype(int)
        + (score_df["MACD"] == "OK").astype(int)
    )

    def suggested_action(score: int) -> str:
        if score >= 5:
            return "COMPRAR"
        if score >= 3:
            return "VIGILAR"
        return "EVITAR"

    score_df["Decision sugerida"] = score_df["Puntaje"].apply(suggested_action)
    score_df = score_df.sort_values(["Puntaje", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return score_df


def render_scoreboard(score_df: pd.DataFrame) -> None:
    if score_df.empty:
        st.info("No se pudo construir el ranking por falta de datos del Top 20 o evaluaciones.")
        return

    st.subheader("Ranking por score consolidado")

    best = score_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Mejor activo", str(best["Ticker"]))
    c2.metric("Puntaje maximo", f"{int(best['Puntaje'])}/6")
    c3.metric("Decision sugerida", str(best["Decision sugerida"]))

    def _color_score(val):
        try:
            v = int(val)
        except (ValueError, TypeError):
            return ""
        if v >= 5:
            return "background-color: #c6efce; color: #006100"
        if v >= 3:
            return "background-color: #ffeb9c; color: #9c5700"
        return "background-color: #ffc7ce; color: #9c0006"

    styled = style_table(score_df)
    styled = styled.map(_color_score, subset=["Puntaje"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_pass_summary(output: str) -> None:
    """Renderiza el resumen de una pasada (top 20, charts, recos, scoreboard, heatmap, stop-loss)."""
    tables = extract_named_tables(output)
    top10_df = extract_top10_table(output)
    recos = extract_recommendations(output)
    score_df = build_scoreboard(top10_df, tables)

    st.subheader("Top 20 consensuado")
    if top10_df.empty:
        st.info("No se pudo detectar el Top 20 en el output.")
    else:
        st.dataframe(top10_df, use_container_width=True, hide_index=True)
        render_top10_charts(top10_df)

    st.subheader("Recomendaciones finales")
    if not recos:
        st.info("No se detectaron recomendaciones finales en el output.")
    else:
        for reco in recos:
            st.markdown(f"- {reco}")

    render_scoreboard(score_df)
    render_heatmap(score_df)
    render_stop_loss(top10_df, score_df)
    render_backtest(top10_df)
    render_correlation(top10_df)

    # Alertas de datos incompletos
    warnings = _extract_data_warnings(output)
    if warnings:
        with st.expander(f"Alertas de datos incompletos ({len(warnings)} activos)"):
            for w in warnings:
                st.caption(f"⚠ {w}")


def _render_pass_tables(output: str) -> None:
    """Renderiza las tablas de evaluacion de una pasada."""
    tables = extract_named_tables(output)
    if not tables:
        st.info("No se detectaron tablas markdown en la salida.")
    else:
        for idx, (title, df) in enumerate(tables, start=1):
            st.markdown(f"### {idx}. {title}")
            render_metrics(df)
            st.dataframe(style_table(df), use_container_width=True, hide_index=True)
            render_decision_chart(df)
            st.divider()


_PASS2_MARKER = "[=== SEGUNDA PASADA (sin EVITAR) ===]"
_PASS3_MARKER = "[=== TERCERA PASADA (sin EVITAR P1+P2) ===]"


# ═══════════════════════════════════════════════════════════════════════════
# 3. HISTORICO SQLite
# ═══════════════════════════════════════════════════════════════════════════

def _init_db() -> None:
    conn = sqlite3.connect(str(HISTORY_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model TEXT,
            tickers_json TEXT,
            scoreboard_json TEXT,
            full_output TEXT
        )
    """)
    conn.commit()
    conn.close()


def _save_run(model: str, score_df: pd.DataFrame, full_output: str) -> None:
    _init_db()
    tickers = score_df["Ticker"].tolist() if not score_df.empty else []
    sb_json = score_df.to_json(orient="records") if not score_df.empty else "[]"
    conn = sqlite3.connect(str(HISTORY_DB))
    conn.execute(
        "INSERT INTO runs (timestamp, model, tickers_json, scoreboard_json, full_output) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), model, json.dumps(tickers), sb_json, full_output),
    )
    conn.commit()
    conn.close()


def _load_history() -> pd.DataFrame:
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    df = pd.read_sql_query("SELECT id, timestamp, model, tickers_json, scoreboard_json FROM runs ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df


def render_history_tab() -> None:
    st.subheader("Historico de ejecuciones")
    hist = _load_history()
    if hist.empty:
        st.info("No hay ejecuciones previas guardadas.")
        return
    for _, row in hist.iterrows():
        ts = row["timestamp"][:19].replace("T", " ")
        tickers = json.loads(row["tickers_json"]) if row["tickers_json"] else []
        with st.expander(f"{ts} — {row['model']} ({len(tickers)} activos)"):
            if row["scoreboard_json"] and row["scoreboard_json"] != "[]":
                sb = pd.read_json(row["scoreboard_json"], orient="records")
                if not sb.empty:
                    st.dataframe(style_table(sb), use_container_width=True, hide_index=True)
            st.caption(f"Tickers: {', '.join(tickers[:20])}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. BACKTESTING SIMPLE
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Calculando backtesting...")
def _fetch_backtest_data(tickers: tuple[str, ...], weights: tuple[float, ...], months: int = 6) -> pd.DataFrame | None:
    """Descarga precios historicos y calcula rendimiento ponderado vs SPY."""
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    all_tickers = list(tickers) + ["SPY"]
    try:
        data = yf.download(all_tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return None
        close = data["Close"].dropna(how="all")
    except Exception:
        return None

    # Normalizar a base 100
    result = pd.DataFrame(index=close.index)
    if "SPY" in close.columns:
        spy = close["SPY"].dropna()
        if not spy.empty:
            result["S&P 500"] = spy / spy.iloc[0] * 100

    # Cartera ponderada
    portfolio_vals = pd.Series(0.0, index=close.index)
    total_w = sum(weights)
    for t, w in zip(tickers, weights):
        if t in close.columns:
            s = close[t].dropna()
            if not s.empty:
                norm = s / s.iloc[0] * 100
                norm = norm.reindex(close.index).ffill()
                portfolio_vals += norm * (w / total_w)

    result["Cartera Top 20"] = portfolio_vals
    result = result.dropna(how="all")
    return result if not result.empty else None


def render_backtest(top10_df: pd.DataFrame) -> None:
    if top10_df.empty:
        return
    tickers = tuple(top10_df["Ticker"].tolist())
    weights = tuple(top10_df["%Cartera"].apply(_extract_percent_value).tolist())
    bt = _fetch_backtest_data(tickers, weights)
    if bt is None or bt.empty:
        st.info("No se pudieron obtener datos para backtesting.")
        return
    st.subheader("Backtesting: Cartera vs S&P 500 (6 meses)")
    st.line_chart(bt, use_container_width=True)
    if "S&P 500" in bt.columns and "Cartera Top 20" in bt.columns:
        port_ret = (bt["Cartera Top 20"].iloc[-1] / 100 - 1) * 100
        spy_ret = (bt["S&P 500"].iloc[-1] / 100 - 1) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Cartera", f"{port_ret:+.1f}%")
        c2.metric("S&P 500", f"{spy_ret:+.1f}%")
        c3.metric("Diferencia", f"{port_ret - spy_ret:+.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 5. CORRELACION ENTRE ACTIVOS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Calculando correlaciones...")
def _fetch_correlation_matrix(tickers: tuple[str, ...]) -> pd.DataFrame | None:
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=180)
    try:
        data = yf.download(list(tickers), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return None
        close = data["Close"].dropna(how="all")
        returns = close.pct_change().dropna()
        if returns.shape[1] < 2:
            return None
        return returns.corr()
    except Exception:
        return None


def render_correlation(top10_df: pd.DataFrame) -> None:
    if top10_df.empty or len(top10_df) < 2:
        return
    tickers = tuple(top10_df["Ticker"].tolist())
    corr = _fetch_correlation_matrix(tickers)
    if corr is None or corr.empty:
        st.info("No se pudieron calcular correlaciones.")
        return
    st.subheader("Matriz de correlacion (6 meses)")

    # Heatmap con Altair
    corr_reset = corr.reset_index()
    idx_col = corr_reset.columns[0]
    corr_long = corr_reset.melt(id_vars=idx_col, var_name="Ticker_B", value_name="Correlacion")
    corr_long.rename(columns={idx_col: "Ticker_A"}, inplace=True)
    chart = alt.Chart(corr_long).mark_rect().encode(
        x=alt.X("Ticker_A:N", title=None),
        y=alt.Y("Ticker_B:N", title=None),
        color=alt.Color("Correlacion:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
        tooltip=["Ticker_A:N", "Ticker_B:N", alt.Tooltip("Correlacion:Q", format=".2f")],
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

    # Alertar pares altamente correlacionados
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            val = corr.iloc[i, j]
            if abs(val) > 0.8:
                high_corr.append((corr.index[i], corr.columns[j], val))
    if high_corr:
        st.warning(f"**Alerta de correlacion alta (>0.8):** {len(high_corr)} pares detectados")
        for a, b, v in high_corr[:10]:
            st.caption(f"  {a} - {b}: {v:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. STOP-LOSS SUGERIDO
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Calculando stop-loss...")
def _compute_stop_loss(tickers: tuple[str, ...]) -> pd.DataFrame:
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=90)
    rows = []
    try:
        data = yf.download(list(tickers), start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
        if data.empty:
            return pd.DataFrame()
        close = data["Close"]
        high = data["High"]
        low = data["Low"]
    except Exception:
        return pd.DataFrame()

    for t in tickers:
        try:
            if len(tickers) > 1:
                c = close[t].dropna()
                h = high[t].dropna()
                l = low[t].dropna()
            else:
                c = close.dropna()
                h = high.dropna()
                l = low.dropna()
            if len(c) < 15:
                continue
            # ATR 14
            tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
            atr14 = tr.rolling(14).mean().iloc[-1]
            precio = c.iloc[-1]
            # Stop-loss = precio - 2*ATR
            stop = precio - 2 * atr14
            # Soporte = minimo de los ultimos 20 dias
            soporte_20 = l.tail(20).min()
            stop_sugerido = max(stop, soporte_20)
            pct_riesgo = ((stop_sugerido - precio) / precio) * 100
            rows.append({
                "Ticker": t,
                "Precio": f"{precio:.2f}",
                "ATR(14)": f"{atr14:.2f}",
                "Stop ATR": f"{stop:.2f}",
                "Soporte 20d": f"{soporte_20:.2f}",
                "Stop sugerido": f"{stop_sugerido:.2f}",
                "Riesgo %": f"{pct_riesgo:.1f}%",
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def render_stop_loss(top10_df: pd.DataFrame, score_df: pd.DataFrame) -> None:
    if score_df.empty:
        return
    # Solo activos COMPRAR
    buy_tickers = score_df[score_df.get("Decision sugerida", pd.Series(dtype=str)).str.upper() == "COMPRAR"]["Ticker"].tolist()
    if not buy_tickers:
        st.info("No hay activos con decision COMPRAR para calcular stop-loss.")
        return
    st.subheader("Stop-loss sugerido (activos COMPRAR)")
    st.caption("Basado en ATR(14) x 2 y soporte de 20 dias. Se usa el mayor de ambos.")
    sl_df = _compute_stop_loss(tuple(buy_tickers))
    if sl_df.empty:
        st.info("No se pudieron calcular stop-loss.")
        return
    st.dataframe(sl_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 7. HEATMAP SCOREBOARD
# ═══════════════════════════════════════════════════════════════════════════

def render_heatmap(score_df: pd.DataFrame) -> None:
    if score_df.empty:
        return
    eval_cols = ["Tecnico", "Fundamental", "Riesgo", "Sentimiento", "Elliott"]
    present = [c for c in eval_cols if c in score_df.columns]
    if not present:
        return
    st.subheader("Mapa de calor: evaluadores x activos")
    heat_data = score_df[["Ticker"] + present].melt(id_vars="Ticker", var_name="Evaluador", value_name="Resultado")
    heat_data["Valor"] = heat_data["Resultado"].map({"OK": 1, "NOK": 0}).fillna(0.5)

    chart = alt.Chart(heat_data).mark_rect(cornerRadius=3).encode(
        x=alt.X("Evaluador:N", title=None),
        y=alt.Y("Ticker:N", title=None, sort=score_df["Ticker"].tolist()),
        color=alt.Color("Valor:Q", scale=alt.Scale(domain=[0, 0.5, 1], range=["#f87171", "#fbbf24", "#4ade80"]), legend=None),
        tooltip=["Ticker:N", "Evaluador:N", "Resultado:N"],
    ).properties(height=max(300, len(score_df) * 22))

    text = alt.Chart(heat_data).mark_text(fontSize=11, fontWeight="bold").encode(
        x="Evaluador:N",
        y=alt.Y("Ticker:N", sort=score_df["Ticker"].tolist()),
        text="Resultado:N",
        color=alt.condition(
            alt.datum.Valor > 0.7,
            alt.value("#064e3b"),
            alt.value("#7f1d1d"),
        ),
    )

    st.altair_chart(chart + text, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# 8. FUNNEL ENTRE PASADAS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_tickers_from_pass(output: str) -> set[str]:
    """Extrae los tickers del Top 20 validado de una pasada."""
    tickers: set[str] = set()
    marker = "[Top 20 validado]"
    pos = output.find(marker)
    if pos == -1:
        return tickers
    for line in output[pos:].splitlines():
        m = re.match(r"^\s*\d+\.\s+(\S+)\s+-", line.strip())
        if m:
            tickers.add(m.group(1).upper())
    return tickers


def _extract_decisions_from_pass(output: str) -> dict[str, str]:
    """Extrae ticker -> decision del veredicto final de una pasada."""
    decisions: dict[str, str] = {}
    marker = "[Veredicto final validado]"
    pos = output.find(marker)
    if pos == -1:
        return decisions
    for line in output[pos:].splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 10:
            ticker = cells[1].upper()
            decision = cells[-1].upper()
            if ticker and any(d in decision for d in ["COMPRAR", "VIGILAR", "EVITAR"]):
                decisions[ticker] = decision
    return decisions


def render_funnel(full_output: str) -> None:
    has_pass2 = _PASS2_MARKER in full_output
    has_pass3 = _PASS3_MARKER in full_output
    if not has_pass2:
        return

    idx2 = full_output.index(_PASS2_MARKER)
    pass1 = full_output[:idx2]
    dec1 = _extract_decisions_from_pass(pass1)

    if has_pass3:
        idx3 = full_output.index(_PASS3_MARKER)
        pass2 = full_output[idx2:idx3]
        pass3 = full_output[idx3:]
        dec2 = _extract_decisions_from_pass(pass2)
        dec3 = _extract_decisions_from_pass(pass3)
    else:
        pass2 = full_output[idx2:]
        dec2 = _extract_decisions_from_pass(pass2)
        dec3 = {}

    st.subheader("Flujo entre pasadas")

    # Build funnel data
    rows = []
    all_tickers = set(dec1.keys()) | set(dec2.keys()) | set(dec3.keys())
    for t in sorted(all_tickers):
        rows.append({
            "Ticker": t,
            "Pasada 1": dec1.get(t, "-"),
            "Pasada 2": dec2.get(t, "-"),
            "Pasada 3": dec3.get(t, "-") if dec3 else "-",
        })
    if not rows:
        return

    funnel_df = pd.DataFrame(rows)

    def _color_decision(val: str) -> str:
        v = str(val).upper()
        if "COMPRAR" in v:
            return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
        if "VIGILAR" in v:
            return "background-color: #fff7e6; color: #9a6700; font-weight: 600;"
        if "EVITAR" in v:
            return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
        return "color: #888;"

    cols = ["Pasada 1", "Pasada 2"]
    if dec3:
        cols.append("Pasada 3")
    styled = funnel_df.style.map(_color_decision, subset=cols)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Resumen funnel
    for label, decisions in [("Pasada 1", dec1), ("Pasada 2", dec2), ("Pasada 3", dec3)]:
        if not decisions:
            continue
        buy = sum(1 for v in decisions.values() if "COMPRAR" in v.upper())
        watch = sum(1 for v in decisions.values() if "VIGILAR" in v.upper())
        avoid = sum(1 for v in decisions.values() if "EVITAR" in v.upper())
        st.caption(f"{label}: {buy} COMPRAR / {watch} VIGILAR / {avoid} EVITAR")


# ═══════════════════════════════════════════════════════════════════════════
# 10. VALIDACION DE DATOS INCOMPLETOS
# ═══════════════════════════════════════════════════════════════════════════

def _extract_data_warnings(output: str) -> list[str]:
    """Detecta activos con datos incompletos mencionados en el output."""
    warnings: list[str] = []
    # Buscar N/D en las tablas del veredicto
    for line in output.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        nd_count = sum(1 for c in cells if c.upper() == "N/D")
        if nd_count > 0 and len(cells) >= 3:
            ticker = cells[1] if len(cells) > 1 else ""
            if ticker and re.match(r"^[A-Z0-9\.\-]+$", ticker):
                warnings.append(f"{ticker}: {nd_count} campo(s) sin datos disponibles")
    return warnings[:20]


def render_results(full_output: str) -> None:
    has_pass2 = _PASS2_MARKER in full_output
    has_pass3 = _PASS3_MARKER in full_output

    if has_pass2:
        idx2 = full_output.index(_PASS2_MARKER)
        pass1_output = full_output[:idx2]

        if has_pass3:
            idx3 = full_output.index(_PASS3_MARKER)
            pass2_output = full_output[idx2:idx3]
            pass3_output = full_output[idx3:]

            t1, t2, t3, t4, t5, t6, t7, t8 = st.tabs([
                "Resumen Pasada 1",
                "Tablas Pasada 1",
                "Resumen Pasada 2 (sin EVITAR)",
                "Tablas Pasada 2",
                "Resumen Pasada 3 (filtrado final)",
                "Tablas Pasada 3",
                "Flujo entre pasadas",
                "Historico",
            ])

            with t1:
                _render_pass_summary(pass1_output)
            with t2:
                _render_pass_tables(pass1_output)
            with t3:
                _render_pass_summary(pass2_output)
            with t4:
                _render_pass_tables(pass2_output)
            with t5:
                _render_pass_summary(pass3_output)
            with t6:
                _render_pass_tables(pass3_output)
            with t7:
                render_funnel(full_output)
            with t8:
                render_history_tab()
        else:
            pass2_output = full_output[idx2:]

            t1, t2, t3, t4, t5, t6 = st.tabs([
                "Resumen Pasada 1",
                "Tablas Pasada 1",
                "Resumen Pasada 2 (sin EVITAR)",
                "Tablas Pasada 2",
                "Flujo entre pasadas",
                "Historico",
            ])

            with t1:
                _render_pass_summary(pass1_output)
            with t2:
                _render_pass_tables(pass1_output)
            with t3:
                _render_pass_summary(pass2_output)
            with t4:
                _render_pass_tables(pass2_output)
            with t5:
                render_funnel(full_output)
            with t6:
                render_history_tab()
    else:
        t1, t2, t3 = st.tabs(["Resumen", "Tablas de evaluacion", "Historico"])
        with t1:
            _render_pass_summary(full_output)
        with t2:
            _render_pass_tables(full_output)
        with t3:
            render_history_tab()


def main() -> None:
    st.set_page_config(page_title="Debate Macro en Vivo", layout="wide")
    inject_app_css()

    st.title("Analisis Macro & Cartera")
    st.caption("Genera un Top 10 de acciones con evaluacion tecnica, fundamental, de riesgo y sentimiento.")

    with st.sidebar:
        st.header("Parametros")
        model = st.text_input("Modelo Ollama", value="llama3.1")
        host = st.text_input("Host Ollama", value="http://127.0.0.1:11434")
        seconds = st.number_input("Duracion (segundos)", min_value=5, max_value=3600, value=180, step=5)
        max_turns = st.number_input("Maximo de turnos", min_value=7, max_value=500, value=56, step=7)
        context_lines = st.number_input("Lineas de contexto", min_value=5, max_value=100, value=18, step=1)

    if not SCRIPT_PATH.exists():
        st.error(f"No se encontro el script: {SCRIPT_PATH}")
        return

    run_clicked = st.button("Iniciar debate", type="primary", use_container_width=True)

    if run_clicked:
        cmd = build_command(
            model=model.strip(),
            host=host.strip(),
            seconds=int(seconds),
            max_turns=int(max_turns),
            context_lines=int(context_lines),
        )
        code, full_output = stream_process(cmd)

        st.divider()
        st.subheader("Resultado")
        if code == 0:
            st.success("Ejecucion completada.")
        else:
            st.error("La ejecucion termino con errores. Revisa la salida completa.")

        consensus = extract_consensus(full_output)
        if consensus:
            st.info(f"**Conclusion macroeconomica de los agentes:**\n\n{consensus}")

        # Guardar en historico (siempre la ultima pasada disponible)
        if code == 0:
            if _PASS3_MARKER in full_output:
                last_pass = full_output[full_output.index(_PASS3_MARKER):]
            elif _PASS2_MARKER in full_output:
                last_pass = full_output[full_output.index(_PASS2_MARKER):]
            else:
                last_pass = full_output
            tables = extract_named_tables(last_pass)
            top10_df = extract_top10_table(last_pass)
            score_df = build_scoreboard(top10_df, tables)
            _save_run(model.strip(), score_df, full_output)

        render_results(full_output)

        st.download_button(
            label="Descargar log completo",
            data=full_output.encode("utf-8"),
            file_name="debate_macro_output.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
