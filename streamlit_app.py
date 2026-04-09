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
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from debate_portfolio import classify_elliott_wave, find_zigzag_pivots, label_elliott_wave_pivots
from market_data import TICKERS as _TICKER_DB
from portfolio_tracker import (
    add_position, close_position, delete_position,
    get_open_positions, get_closed_positions,
    portfolio_snapshot, portfolio_risk_summary,
    compute_exit_signals, compute_performance_report, compute_entry_levels,
)
from professional_analytics import (
    compute_professional_report,
    ensure_runs_metadata_column,
    extract_run_metadata_json,
)


def _ticker_label(ticker: str) -> str:
    """Devuelve 'AAPL — Apple' o solo el ticker si no hay nombre."""
    name = _TICKER_DB.get(ticker, {}).get("nombre", "")
    return f"{ticker} — {name}" if name else ticker


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
    "REVISION INSTITUCIONAL": "Institucional",
    "REVISION WYCKOFF": "Wyckoff",
    "REVISION ANALISIS RELATIVO": "Analisis relativo",
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
    "Institucional": "Institucional",
}


def _extract_percent_value(value: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", str(value))
    return float(match.group(1)) if match else 0.0


def extract_timer_block(text: str) -> list[dict[str, str]] | None:
    """Extrae las etapas del cronometro de ejecucion del output."""
    marker = "CRONOMETRO DE EJECUCION"
    pos = text.find(marker)
    if pos == -1:
        return None
    block = text[pos:]
    end = block.find("\n=" * 1)
    # Find closing === line after the content
    lines = block.splitlines()
    stages: list[dict[str, str]] = []
    for ln in lines[2:]:  # skip header and === line
        ln = ln.strip()
        if ln.startswith("="):
            break
        if not ln or ln.startswith("─"):
            continue
        # Match: "  Nombre etapa          123.4s  (12.3%) ████"
        m = re.match(r"(.+?)\s{2,}([\d.]+)s\s+\(([\d.]+)%\)", ln)
        if m:
            stages.append({"etapa": m.group(1).strip(), "segundos": m.group(2), "pct": m.group(3)})
        # Match TOTAL line
        mt = re.match(r"TOTAL\s+([\d.]+)s\s+\((\d+m\s*\d+s)\)", ln)
        if mt:
            stages.append({"etapa": "TOTAL", "segundos": mt.group(1), "pct": mt.group(2)})
    return stages if stages else None


def render_timer(full_output: str) -> None:
    """Renderiza el cronometro de ejecucion como tabla en Streamlit."""
    stages = extract_timer_block(full_output)
    if not stages:
        return
    st.subheader("Cronometro de ejecucion")
    rows = []
    for s in stages:
        if s["etapa"] == "TOTAL":
            rows.append({"Etapa": "**TOTAL**", "Tiempo": f"{s['segundos']}s", "%": s["pct"]})
        else:
            rows.append({"Etapa": s["etapa"], "Tiempo": f"{s['segundos']}s", "%": f"{s['pct']}%"})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def build_command(model: str, host: str, seconds: int, max_turns: int, context_lines: int, portfolio: str = "") -> list[str]:
    cmd = [
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
    if portfolio:
        cmd.extend(["--portfolio", portfolio])
    return cmd


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


_RE_AGENT_TURN = re.compile(r"^\[(.+?)\]\s*\(modelo:\s*(.+?)\)")
_RE_MODEL_MAP = re.compile(r"^\s{2}(.+?)\s+->\s+(.+)$")


def _detect_phase(line: str) -> str | None:
    upper = line.upper()
    if "CARGANDO DATOS REALES" in upper:
        return "Cargando datos de mercado..."
    if "DATOS DE MERCADO CARGADOS DESDE CACHE" in upper:
        return "Datos cargados desde cache..."
    if "DEBATE EN VIVO" in upper:
        return "Debate en curso..."
    # Turno de agente: [Nombre] (modelo: xxx) — se gestiona aparte
    if _RE_AGENT_TURN.match(line):
        return "__agent_turn__"
    if "SEGUNDA PASADA" in upper:
        return "Segunda pasada (sin EVITAR)..."
    if "TERCERA PASADA" in upper:
        return "Tercera pasada (filtrado final)..."
    if "TOP 20 INVERSIONES" in upper:
        return "Generando Top 20..."
    if "VALIDACION ADVERSARIAL DEL TOP 20" in upper:
        return "Validacion adversarial..."
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
    if "REVISION INSTITUCIONAL" in upper:
        return "Analisis institucional..."
    if "VEREDICTO FINAL" in upper:
        return "Veredicto final..."
    return None


def stream_process(cmd: list[str]) -> tuple[int, str]:
    full_output = ""
    phase_label = st.empty()
    model_label = st.empty()
    current_phase = "Iniciando..."
    model_map: dict[str, str] = {}

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
                    # Capturar mapa de modelos del inicio
                    mm = _RE_MODEL_MAP.match(ln)
                    if mm:
                        model_map[mm.group(1).strip()] = mm.group(2).strip()
                        continue

                    detected = _detect_phase(ln)
                    if not detected or detected == current_phase:
                        continue

                    # Turno de agente → mostrar nombre + modelo
                    if detected == "__agent_turn__":
                        am = _RE_AGENT_TURN.match(ln)
                        if am:
                            agent_name, agent_model = am.group(1), am.group(2)
                            current_phase = f"Debate: {agent_name}"
                            phase_label.info(f"Fase actual: Debate en curso...")
                            model_label.caption(
                                f"🎙️ **{agent_name}** — `{agent_model}`"
                            )
                    else:
                        current_phase = detected
                        phase_label.info(f"Fase actual: {current_phase}")
                        # Mostrar modelo asignado si la fase tiene uno
                        _phase_role_map = {
                            "Segunda pasada (sin EVITAR)...": "Moderador Consenso",
                            "Tercera pasada (filtrado final)...": "Moderador Consenso",
                            "Generando Top 20...": "Moderador Top 20",
                            "Validacion adversarial...": "Devils Advocate",
                            "Veredicto final...": "Veredicto Final",
                        }
                        role = _phase_role_map.get(current_phase, "")
                        if role and role in model_map:
                            model_label.caption(
                                f"🤖 **{role}** — `{model_map[role]}`"
                            )
                        elif "Evaluando" in current_phase or "Analisis" in current_phase or "Gestion" in current_phase or "Sentimiento" in current_phase or "institucional" in current_phase:
                            model_label.caption("📊 Evaluador determinista (sin LLM)")
                        else:
                            model_label.empty()
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
    marker_pos = text.rfind(marker)
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

    evaluator_cols = ["Tecnico", "Fundamental", "Riesgo", "Sentimiento", "Elliott", "MACD", "Institucional"]
    for col in evaluator_cols:
        score_df[col] = score_df[col].fillna("N/D")

    score_df["Puntaje"] = (
        (score_df["Tecnico"] == "OK").astype(int)
        + (score_df["Fundamental"] == "OK").astype(int)
        + (score_df["Riesgo"] == "OK").astype(int)
        + (score_df["Sentimiento"] == "OK").astype(int)
        + (score_df["Elliott"] == "OK").astype(int)
        + (score_df["MACD"] == "OK").astype(int)
        + (score_df["Institucional"] == "OK").astype(int)
    )

    def suggested_action(score: int) -> str:
        if score >= 6:
            return "COMPRAR"
        if score >= 4:
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
    c2.metric("Puntaje maximo", f"{int(best['Puntaje'])}/7")
    c3.metric("Decision sugerida", str(best["Decision sugerida"]))

    def _color_score(val):
        try:
            v = int(val)
        except (ValueError, TypeError):
            return ""
        if v >= 6:
            return "background-color: #c6efce; color: #006100"
        if v >= 4:
            return "background-color: #ffeb9c; color: #9c5700"
        return "background-color: #ffc7ce; color: #9c0006"

    styled = style_table(score_df)
    styled = styled.map(_color_score, subset=["Puntaje"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_pass_summary(output: str, pass_id: str = "p1") -> None:
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
    render_run_deltas(score_df)
    render_heatmap(score_df)
    render_candlestick(top10_df, score_df, pass_id=pass_id)
    render_stop_loss(top10_df, score_df)
    render_backtest(top10_df)
    render_correlation(top10_df)

    # Export CSV del scoreboard con deltas
    if not score_df.empty:
        previous_df = _load_previous_scoreboard()
        export_df = compute_run_deltas(score_df, previous_df) if not previous_df.empty else score_df.copy()
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Descargar scoreboard (CSV)",
            data=csv_data,
            file_name=f"scoreboard_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key=f"csv_export_{pass_id}",
        )

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


def _extract_latest_pass_output(full_output: str) -> str:
    if _PASS3_MARKER in full_output:
        return full_output[full_output.index(_PASS3_MARKER):]
    if _PASS2_MARKER in full_output:
        return full_output[full_output.index(_PASS2_MARKER):]
    return full_output


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
    # Migrar: anadir columna prices_json si no existe
    cursor = conn.execute("PRAGMA table_info(runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "prices_json" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN prices_json TEXT")
    conn.commit()
    conn.close()
    ensure_runs_metadata_column(HISTORY_DB)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_current_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Obtiene el precio actual de una lista de tickers."""
    prices: dict[str, float] = {}
    if not tickers:
        return prices
    try:
        data = yf.download(list(tickers), period="5d", progress=False)
        if data.empty:
            return prices
        close = data["Close"]
        if len(tickers) == 1:
            last = close.dropna().iloc[-1] if not close.dropna().empty else None
            if last is not None:
                prices[tickers[0]] = float(last)
        else:
            for t in tickers:
                if t in close.columns:
                    s = close[t].dropna()
                    if not s.empty:
                        prices[t] = float(s.iloc[-1])
    except Exception:
        pass
    return prices


def _save_run(model: str, score_df: pd.DataFrame, full_output: str) -> None:
    _init_db()
    tickers = score_df["Ticker"].tolist() if not score_df.empty else []
    sb_json = score_df.to_json(orient="records") if not score_df.empty else "[]"
    # Capturar precios al momento de la recomendacion
    prices_at_rec = _fetch_current_prices(tuple(tickers)) if tickers else {}
    metadata_json = json.dumps(extract_run_metadata_json(full_output)) if full_output else "{}"
    conn = sqlite3.connect(str(HISTORY_DB))
    conn.execute(
        "INSERT INTO runs (timestamp, model, tickers_json, scoreboard_json, full_output, prices_json, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), model, json.dumps(tickers), sb_json, full_output, json.dumps(prices_at_rec), metadata_json),
    )
    conn.commit()
    conn.close()


def _load_history() -> pd.DataFrame:
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    df = pd.read_sql_query("SELECT id, timestamp, model, tickers_json, scoreboard_json, metadata_json FROM runs ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df


def _load_latest_metadata() -> dict[str, Any]:
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    row = conn.execute(
        "SELECT metadata_json FROM runs WHERE metadata_json IS NOT NULL AND metadata_json != '{}' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


def _load_previous_scoreboard() -> pd.DataFrame:
    """Carga el scoreboard de la ejecucion anterior (la mas reciente guardada)."""
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    row = conn.execute(
        "SELECT scoreboard_json FROM runs WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    if not row or not row[0]:
        return pd.DataFrame()
    try:
        return pd.read_json(row[0], orient="records")
    except Exception:
        return pd.DataFrame()


def _load_scoreboard_history(limit: int = 10) -> list[pd.DataFrame]:
    """Carga los scoreboards de las ultimas N ejecuciones (mas reciente primero)."""
    history = _load_scoreboard_history_with_dates(limit)
    return [df for _, df in history]


def _load_scoreboard_history_with_dates(limit: int = 10) -> list[tuple[str, pd.DataFrame]]:
    """Carga (timestamp, scoreboard) de las ultimas N ejecuciones (mas reciente primero)."""
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    rows = conn.execute(
        "SELECT timestamp, scoreboard_json FROM runs WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    result: list[tuple[str, pd.DataFrame]] = []
    for ts, sb_json in rows:
        try:
            df = pd.read_json(sb_json, orient="records")
            if not df.empty:
                result.append((ts, df))
        except Exception:
            continue
    return result


_DECISION_RANK = {"COMPRAR": 3, "VIGILAR": 2, "EVITAR": 1}


def compute_run_deltas(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> pd.DataFrame:
    """Compara el scoreboard actual con el anterior y anade columnas de cambio."""
    if current_df.empty:
        return current_df

    result = current_df.copy()

    if previous_df.empty:
        result["Estado"] = "NUEVA"
        result["Anterior"] = "-"
        return result

    prev_decisions: dict[str, str] = {}
    if "Ticker" in previous_df.columns and "Decision sugerida" in previous_df.columns:
        prev_decisions = dict(zip(
            previous_df["Ticker"].astype(str).str.upper(),
            previous_df["Decision sugerida"].astype(str).str.upper(),
        ))

    estados = []
    anteriores = []
    for _, row in result.iterrows():
        ticker = str(row["Ticker"]).upper()
        current_decision = str(row.get("Decision sugerida", "")).upper()
        prev_decision = prev_decisions.get(ticker)

        if prev_decision is None:
            estados.append("NUEVA")
            anteriores.append("-")
        else:
            anteriores.append(prev_decision)
            curr_rank = _DECISION_RANK.get(current_decision, 0)
            prev_rank = _DECISION_RANK.get(prev_decision, 0)
            if curr_rank > prev_rank:
                estados.append("SUBE")
            elif curr_rank < prev_rank:
                estados.append("BAJA")
            else:
                estados.append("MANTIENE")

    result["Estado"] = estados
    result["Anterior"] = anteriores
    return result


def compute_conviction_days(current_df: pd.DataFrame, history: list[tuple[str, pd.DataFrame]]) -> dict[str, int]:
    """Calcula cuantos dias lleva un ticker apareciendo consecutivamente como COMPRAR."""
    streaks: dict[str, int] = {}
    if current_df.empty:
        return streaks

    today = datetime.now()

    for _, row in current_df.iterrows():
        ticker = str(row["Ticker"]).upper()
        decision = str(row.get("Decision sugerida", "")).upper()
        if decision != "COMPRAR":
            streaks[ticker] = 0
            continue
        # Buscar la ejecucion mas antigua consecutiva con COMPRAR
        oldest_ts = today
        for ts_str, past_df in history:
            if past_df.empty or "Ticker" not in past_df.columns:
                break
            past_decisions = dict(zip(
                past_df["Ticker"].astype(str).str.upper(),
                past_df["Decision sugerida"].astype(str).str.upper() if "Decision sugerida" in past_df.columns else pd.Series(dtype=str),
            ))
            if past_decisions.get(ticker) == "COMPRAR":
                try:
                    oldest_ts = datetime.fromisoformat(ts_str)
                except (ValueError, TypeError):
                    pass
            else:
                break
        days = max(1, (today - oldest_ts).days)
        streaks[ticker] = days

    return streaks


def _find_exited_tickers(current_df: pd.DataFrame, previous_df: pd.DataFrame) -> list[dict[str, str]]:
    """Encuentra tickers que estaban en la ejecucion anterior pero ya no estan."""
    if previous_df.empty or current_df.empty:
        return []
    current_tickers = set(current_df["Ticker"].astype(str).str.upper())
    exited = []
    for _, row in previous_df.iterrows():
        ticker = str(row["Ticker"]).upper()
        if ticker not in current_tickers:
            decision = str(row.get("Decision sugerida", "?"))
            exited.append({"Ticker": ticker, "Decision anterior": decision})
    return exited


def render_run_deltas(score_df: pd.DataFrame) -> None:
    """Renderiza la seccion de cambios respecto a la ejecucion anterior."""
    previous_df = _load_previous_scoreboard()
    if previous_df.empty:
        st.info("No hay ejecucion anterior para comparar. Los cambios se mostraran a partir de la proxima ejecucion.")
        return

    delta_df = compute_run_deltas(score_df, previous_df)
    if "Estado" not in delta_df.columns:
        st.info("No se pudo calcular el delta (scoreboard vacío).")
        return
    history_with_dates = _load_scoreboard_history_with_dates(10)
    streaks = compute_conviction_days(score_df, history_with_dates)

    # Resumen rapido de movimientos
    st.subheader("Cambios vs ejecucion anterior")

    nuevas = int((delta_df["Estado"] == "NUEVA").sum())
    suben = int((delta_df["Estado"] == "SUBE").sum())
    bajan = int((delta_df["Estado"] == "BAJA").sum())
    mantienen = int((delta_df["Estado"] == "MANTIENE").sum())
    exited = _find_exited_tickers(score_df, previous_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Nuevas", nuevas)
    c2.metric("Suben", suben)
    c3.metric("Bajan", bajan)
    c4.metric("Mantienen", mantienen)
    c5.metric("Salen", len(exited))

    # Tabla de cambios con colores
    display_cols = ["Ticker"]
    if "Nombre" in delta_df.columns:
        display_cols.append("Nombre")
    display_cols += ["Decision sugerida", "Anterior", "Estado", "Puntaje"]
    display_cols = [c for c in display_cols if c in delta_df.columns]
    change_df = delta_df[display_cols].copy()

    # Anadir columna de conviccion (dias consecutivos como COMPRAR)
    if streaks:
        change_df["Conviccion"] = change_df["Ticker"].map(
            lambda t: f"{streaks.get(t.upper(), 0)}d" if streaks.get(t.upper(), 0) >= 2 else "-"
        )

    def _color_estado(val: str) -> str:
        v = str(val).upper()
        if v == "NUEVA":
            return "background-color: #dbeafe; color: #1e40af; font-weight: 600;"
        if v == "SUBE":
            return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
        if v == "BAJA":
            return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
        if v == "MANTIENE":
            return "background-color: #f3f4f6; color: #374151;"
        return ""

    styled = change_df.style.map(_color_estado, subset=["Estado"])
    if "Decision sugerida" in change_df.columns:
        def _color_decision(val: str) -> str:
            v = str(val).upper()
            if "COMPRAR" in v:
                return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
            if "VIGILAR" in v:
                return "background-color: #fff7e6; color: #9a6700; font-weight: 600;"
            if "EVITAR" in v:
                return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
            return ""
        styled = styled.map(_color_decision, subset=["Decision sugerida"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Detalle de movimientos relevantes
    if suben > 0:
        sube_df = delta_df[delta_df["Estado"] == "SUBE"]
        st.success("**Suben de nivel:** " + ", ".join(
            f"{row['Ticker']} ({row['Anterior']} -> {row['Decision sugerida']})"
            for _, row in sube_df.iterrows()
        ))

    if bajan > 0:
        baja_df = delta_df[delta_df["Estado"] == "BAJA"]
        st.warning("**Bajan de nivel:** " + ", ".join(
            f"{row['Ticker']} ({row['Anterior']} -> {row['Decision sugerida']})"
            for _, row in baja_df.iterrows()
        ))

    if exited:
        st.error("**Salen del Top 20:** " + ", ".join(
            f"{e['Ticker']} (era {e['Decision anterior']})"
            for e in exited
        ))

    # Conviccion alta
    high_conviction = {t: s for t, s in streaks.items() if s >= 3}
    if high_conviction:
        st.info("**Conviccion alta** (COMPRAR durante 3+ dias consecutivos): " + ", ".join(
            f"{t} ({s}d)" for t, s in sorted(high_conviction.items(), key=lambda x: -x[1])
        ))


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
            metadata = {}
            if "metadata_json" in row.index and isinstance(row["metadata_json"], str) and row["metadata_json"]:
                try:
                    metadata = json.loads(row["metadata_json"])
                except json.JSONDecodeError:
                    metadata = {}
            if metadata:
                regime = metadata.get("market_regime", {}).get("name")
                deploy = metadata.get("confidence_model", {}).get("deploy_pct")
                if regime or deploy is not None:
                    st.caption(f"Regimen: {regime or 'N/D'} | Capital sugerido: {deploy if deploy is not None else 'N/D'}%")
            if row["scoreboard_json"] and row["scoreboard_json"] != "[]":
                sb = pd.read_json(row["scoreboard_json"], orient="records")
                if not sb.empty:
                    st.dataframe(style_table(sb), use_container_width=True, hide_index=True)
            st.caption(f"Tickers: {', '.join(tickers[:20])}")


# ═══════════════════════════════════════════════════════════════════════════
# 3b. ALERTAS DE PRECIO
# ═══════════════════════════════════════════════════════════════════════════

def _load_runs_with_prices(limit: int = 10) -> list[dict]:
    """Carga las ultimas ejecuciones con precios y scoreboards."""
    _init_db()
    conn = sqlite3.connect(str(HISTORY_DB))
    rows = conn.execute(
        "SELECT timestamp, scoreboard_json, prices_json FROM runs "
        "WHERE scoreboard_json IS NOT NULL AND scoreboard_json != '[]' "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    result = []
    for ts, sb_json, prices_json in rows:
        try:
            sb = pd.read_json(sb_json, orient="records")
            prices = json.loads(prices_json) if prices_json else {}
            if not sb.empty:
                result.append({"timestamp": ts, "scoreboard": sb, "prices": prices})
        except Exception:
            continue
    return result


def render_price_alerts() -> None:
    """Muestra alertas comparando precios de recomendacion vs precios actuales."""
    st.subheader("Alertas de precio")
    st.caption("Compara el precio al momento de la recomendacion COMPRAR con el precio actual.")

    runs = _load_runs_with_prices(5)
    if not runs:
        st.info("No hay ejecuciones con datos de precio guardados. Las alertas apareceran despues de la proxima ejecucion.")
        return

    # Recopilar las recomendaciones COMPRAR mas recientes por ticker
    buy_recs: dict[str, dict] = {}  # ticker -> {price, timestamp, decision}
    for run_data in runs:
        sb = run_data["scoreboard"]
        prices = run_data["prices"]
        ts = run_data["timestamp"][:19].replace("T", " ")
        if "Decision sugerida" not in sb.columns:
            continue
        for _, row in sb.iterrows():
            ticker = str(row["Ticker"]).upper()
            decision = str(row.get("Decision sugerida", "")).upper()
            if decision == "COMPRAR" and ticker not in buy_recs and ticker in prices:
                buy_recs[ticker] = {
                    "precio_rec": prices[ticker],
                    "timestamp": ts,
                    "puntaje": int(row.get("Puntaje", 0)),
                }

    if not buy_recs:
        st.info("No hay recomendaciones COMPRAR con precios de referencia guardados.")
        return

    # Obtener precios actuales
    current_prices = _fetch_current_prices(tuple(buy_recs.keys()))
    if not current_prices:
        st.warning("No se pudieron obtener precios actuales.")
        return

    alert_rows = []
    for ticker, rec in sorted(buy_recs.items()):
        if ticker not in current_prices:
            continue
        price_rec = rec["precio_rec"]
        price_now = current_prices[ticker]
        if price_rec <= 0:
            continue
        change_pct = ((price_now - price_rec) / price_rec) * 100

        # Clasificar la alerta
        if change_pct >= 20:
            alerta = "TOMAR BENEFICIOS"
        elif change_pct >= 10:
            alerta = "EN GANANCIA"
        elif change_pct >= -5:
            alerta = "ESTABLE"
        elif change_pct >= -15:
            alerta = "EN PERDIDA"
        else:
            alerta = "STOP-LOSS"

        alert_rows.append({
            "Ticker": ticker,
            "Recomendado": rec["timestamp"],
            "Precio rec.": f"${price_rec:.2f}",
            "Precio actual": f"${price_now:.2f}",
            "Cambio %": f"{change_pct:+.1f}%",
            "Alerta": alerta,
            "Puntaje": f"{rec['puntaje']}/7",
            "_change": change_pct,
        })

    if not alert_rows:
        st.info("No se pudieron calcular alertas de precio.")
        return

    alert_df = pd.DataFrame(alert_rows).sort_values("_change", ascending=False)
    display_df = alert_df.drop(columns=["_change"])

    # Metricas resumen
    gains = sum(1 for r in alert_rows if r["_change"] > 0)
    losses = sum(1 for r in alert_rows if r["_change"] < 0)
    avg_change = sum(r["_change"] for r in alert_rows) / len(alert_rows)
    stop_alerts = sum(1 for r in alert_rows if r["Alerta"] == "STOP-LOSS")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("En ganancia", gains)
    c2.metric("En perdida", losses)
    c3.metric("Cambio medio", f"{avg_change:+.1f}%")
    c4.metric("Alertas stop", stop_alerts)

    def _color_alerta(val: str) -> str:
        v = str(val).upper()
        if v == "TOMAR BENEFICIOS":
            return "background-color: #dbeafe; color: #1e40af; font-weight: 600;"
        if v == "EN GANANCIA":
            return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
        if v == "ESTABLE":
            return "background-color: #f3f4f6; color: #374151;"
        if v == "EN PERDIDA":
            return "background-color: #fff7e6; color: #9a6700; font-weight: 600;"
        if v == "STOP-LOSS":
            return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
        return ""

    styled = display_df.style.map(_color_alerta, subset=["Alerta"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Alertas criticas
    stops = [r for r in alert_rows if r["Alerta"] == "STOP-LOSS"]
    if stops:
        st.error("**ALERTA STOP-LOSS:** " + ", ".join(
            f"{r['Ticker']} ({r['Cambio %']})"
            for r in stops
        ))

    takes = [r for r in alert_rows if r["Alerta"] == "TOMAR BENEFICIOS"]
    if takes:
        st.success("**Considerar tomar beneficios:** " + ", ".join(
            f"{r['Ticker']} ({r['Cambio %']})"
            for r in takes
        ))


# ═══════════════════════════════════════════════════════════════════════════
# 3b-2. SEÑALES DE SALIDA
# ═══════════════════════════════════════════════════════════════════════════

def render_exit_signals() -> None:
    """Muestra señales de salida basadas en deterioro de tesis vs posiciones abiertas."""
    st.subheader("Señales de salida")
    st.caption("Compara tus posiciones abiertas con las recomendaciones actuales del sistema.")

    positions = get_open_positions()
    if positions.empty:
        st.info("No tienes posiciones registradas. Ve a **Mi Cartera** para añadir tus posiciones reales.")
        return

    # Cargar scoreboard más reciente
    prev_df = _load_previous_scoreboard()
    signals = compute_exit_signals(prev_df)

    if not signals:
        st.success("Sin señales de salida. Todas tus posiciones están alineadas con las recomendaciones.")
        return

    # Métricas resumen
    alta = sum(1 for s in signals if s["urgencia"] == "ALTA")
    media = sum(1 for s in signals if s["urgencia"] == "MEDIA")
    c1, c2, c3 = st.columns(3)
    c1.metric("Señales urgentes", alta)
    c2.metric("Señales moderadas", media)
    c3.metric("Total posiciones", len(positions))

    # Tabla de señales
    for s in signals:
        urgencia = s["urgencia"]
        if urgencia == "ALTA":
            color = "#ef4444"
            icon = "🔴"
        else:
            color = "#f59e0b"
            icon = "🟡"

        st.markdown(
            f"{icon} **{s['ticker']}** — "
            f"<span style='color:{color};font-weight:700'>{s['señal']}</span> "
            f"| P&L: {s['pnl_pct']:+.1f}% | {s['motivo']}",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3b-3. PERFORMANCE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def render_performance() -> None:
    """Muestra el rendimiento real de las recomendaciones pasadas."""
    st.subheader("Performance de recomendaciones")
    st.caption("¿Cuánto habrías ganado/perdido siguiendo las señales del sistema?")

    report = compute_performance_report()
    if not report:
        st.info("Necesitas al menos 2 ejecuciones con datos de precio para ver el performance.")
        return

    total = report.get("total_recommendations", 0)
    st.markdown(f"**{total}** recomendaciones analizadas en el histórico.")

    for decision in ["COMPRAR", "VIGILAR", "EVITAR"]:
        stats = report.get(decision, {})
        if not stats or stats.get("count", 0) == 0:
            continue

        if decision == "COMPRAR":
            color = "#22c55e"
        elif decision == "VIGILAR":
            color = "#eab308"
        else:
            color = "#ef4444"

        st.markdown(f"### <span style='color:{color}'>{decision}</span>", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Aciertos", f"{stats.get('hit_rate', 0):.0f}%",
                  delta=f"{stats.get('wins', 0)}/{stats.get('count', 0)}")
        c2.metric("Rent. media", f"{stats.get('avg_return', 0):+.1f}%")
        c3.metric("Mejor", f"{stats.get('best', 0):+.1f}%")
        c4.metric("Peor", f"{stats.get('worst', 0):+.1f}%")

        details = stats.get("details", [])
        if details:
            df = pd.DataFrame(details)
            df = df.rename(columns={
                "ticker": "Ticker", "date": "Fecha rec.",
                "price_rec": "Precio rec.", "price_now": "Precio actual",
                "return_pct": "Rent. %", "puntaje": "Puntaje",
            })

            def _color_return(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "color: #22c55e"
                    elif val < 0:
                        return "color: #ef4444"
                return ""

            styled = df.style.applymap(_color_return, subset=["Rent. %"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        st.divider()


# ═══════════════════════════════════════════════════════════════════════════
# 3b-4. NIVELES DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════

def render_entry_levels() -> None:
    """Muestra niveles de entrada óptimos para los activos del Top 20."""
    st.subheader("Niveles de entrada")
    st.caption("Zonas de soporte técnico y señales de timing para cada activo recomendado.")

    prev_df = _load_previous_scoreboard()
    if prev_df.empty:
        st.info("Ejecuta un debate primero para ver niveles de entrada.")
        return

    # Filtrar solo COMPRAR y VIGILAR
    buy_watch = prev_df[
        prev_df["Decision sugerida"].str.upper().isin(["COMPRAR", "VIGILAR"])
    ] if "Decision sugerida" in prev_df.columns else prev_df

    tickers = buy_watch["Ticker"].tolist() if not buy_watch.empty else []
    if not tickers:
        st.info("No hay activos con decisión COMPRAR o VIGILAR.")
        return

    with st.spinner("Calculando niveles de entrada..."):
        levels = compute_entry_levels(tickers)

    if not levels:
        st.warning("No se pudieron calcular niveles de entrada.")
        return

    df = pd.DataFrame(levels)

    # Color por timing
    def _color_timing(val):
        s = str(val)
        if "AHORA" in s:
            return "background-color: rgba(34,197,94,0.2)"
        elif "ESPERAR" in s:
            return "background-color: rgba(239,68,68,0.15)"
        elif "CERCANO" in s:
            return "background-color: rgba(234,179,8,0.15)"
        return ""

    styled = df.style.applymap(_color_timing, subset=["Timing"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Resumen rápido
    ahora = [l for l in levels if "AHORA" in l["Timing"]]
    cercano = [l for l in levels if "CERCANO" in l["Timing"]]
    esperar = [l for l in levels if "ESPERAR" in l["Timing"]]

    if ahora:
        st.success(f"**Entrada óptima ahora:** {', '.join(l['Ticker'] for l in ahora)}")
    if cercano:
        st.warning(f"**Cerca de zona de entrada:** {', '.join(l['Ticker'] for l in cercano)}")
    if esperar:
        st.info(f"**Esperar pullback:** {', '.join(l['Ticker'] for l in esperar)}")


# ═══════════════════════════════════════════════════════════════════════════
# 3b-5. ANALITICA PROFESIONAL
# ═══════════════════════════════════════════════════════════════════════════

def render_professional_analytics() -> None:
    """Muestra métricas profesionales: alpha, drawdown, régimen y precisión por factor."""
    st.subheader("Analítica profesional")
    st.caption("Convierte el histórico en métricas de calidad de señal, alpha y control de riesgo.")

    latest = _load_latest_metadata()
    report = compute_professional_report(HISTORY_DB)

    if not latest and not report:
        st.info("Necesitas al menos una ejecución guardada con metadatos para ver esta analítica.")
        return

    if latest:
        confidence = latest.get("confidence_model", {})
        regime = latest.get("market_regime", {})
        risk = latest.get("risk_summary", {})
        devil = latest.get("devil_advocate", {})

        st.markdown("#### Última ejecución")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score confianza", f"{confidence.get('score', 0)}/100")
        c2.metric("Capital sugerido", f"{confidence.get('deploy_pct', 0)}%")
        c3.metric("Régimen", str(regime.get("name", "N/D")))
        c4.metric("Sesgo", str(regime.get("stance", "N/D")))

        if regime:
            st.caption(
                f"VIX {regime.get('signals', {}).get('vix', 'N/D')} | "
                f"SPY 3m {regime.get('signals', {}).get('spy_3m', 'N/D')}% | "
                f"QQQ vs SPY {regime.get('signals', {}).get('qqq_vs_spy', 'N/D')}%"
            )
            st.info(regime.get("summary", ""))

        if confidence.get("rationale"):
            st.markdown("#### Meta-modelo de confianza")
            for line in confidence["rationale"]:
                st.markdown(f"- {line}")

        if risk:
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Top 3 concentración", f"{risk.get('top3_concentration', 0):.1f}%")
            rc2.metric("Posición máxima", f"{risk.get('max_position', 0):.1f}%")
            rc3.metric("Volatilidad ponderada", f"{risk.get('weighted_volatility', 0):.1f}%")

            risk_warnings = risk.get("warnings", [])
            if risk_warnings:
                st.markdown("#### Riesgos estructurales")
                for warning in risk_warnings:
                    st.markdown(f"- {warning}")

            adjusted = risk.get("adjusted_weights", {})
            top20 = latest.get("top20", [])
            if adjusted and top20:
                original = {asset.get("ticker"): asset.get("peso") for asset in top20}
                rebalance_df = pd.DataFrame(
                    [
                        {
                            "Ticker": ticker,
                            "Peso actual": original.get(ticker, 0),
                            "Peso ajustado": adjusted.get(ticker, 0),
                        }
                        for ticker in adjusted
                    ]
                ).sort_values("Peso ajustado", ascending=False)
                st.markdown("#### Pesos ajustados por riesgo")
                st.dataframe(rebalance_df, use_container_width=True, hide_index=True)

        if devil:
            st.markdown("#### Validación adversarial")
            if devil.get("llm_verdict"):
                st.warning(devil["llm_verdict"])
            issues = devil.get("issues", [])
            if issues:
                for issue in issues:
                    st.markdown(f"- {issue}")

    if report:
        summary = report.get("summary", {})
        st.markdown("#### Histórico agregado")
        s1, s2, s3, s4, s5, s6 = st.columns(6)
        s1.metric("Runs", summary.get("runs", 0))
        s2.metric("Rent. media", f"{summary.get('avg_return', 0):+.1f}%")
        s3.metric("Rent. anualizada", f"{summary.get('annualized_return', 0):+.1f}%")
        s4.metric("Alpha medio", f"{summary.get('avg_alpha', 0):+.1f}%")
        s5.metric("Beat rate", f"{summary.get('beat_rate', 0):.0f}%")
        s6.metric("Sharpe medio", f"{summary.get('avg_sharpe', 0):.2f}")

        runs = report.get("portfolio_runs", [])
        if runs:
            st.markdown("#### Evolución por ejecución")
            runs_df = pd.DataFrame(runs)
            st.dataframe(runs_df, use_container_width=True, hide_index=True)

        evaluators = report.get("evaluators", [])
        if evaluators:
            st.markdown("#### Precisión por evaluador")
            st.dataframe(pd.DataFrame(evaluators), use_container_width=True, hide_index=True)

        agents = report.get("agents", [])
        if agents:
            st.markdown("#### Precisión por agente")
            st.dataframe(pd.DataFrame(agents), use_container_width=True, hide_index=True)

        regimes = report.get("regimes", [])
        if regimes:
            st.markdown("#### Alpha por régimen")
            st.dataframe(pd.DataFrame(regimes), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3c. DASHBOARD MULTI-EJECUCION
# ═══════════════════════════════════════════════════════════════════════════

def render_dashboard() -> None:
    """Dashboard con evolucion de scores y decisiones a lo largo de ejecuciones."""
    st.subheader("Dashboard multi-ejecucion")
    st.caption("Evolucion de puntajes y decisiones de los activos a lo largo de las ejecuciones.")

    history_wd = _load_scoreboard_history_with_dates(20)
    if len(history_wd) < 2:
        st.info("Se necesitan al menos 2 ejecuciones para mostrar el dashboard. Ejecuta el analisis varias veces.")
        return

    # Revertir para orden cronologico (mas antigua primero)
    history_wd = list(reversed(history_wd))
    history = [df for _, df in history_wd]

    # Construir datos de evolucion
    all_tickers: set[str] = set()
    for df in history:
        if "Ticker" in df.columns:
            all_tickers.update(df["Ticker"].astype(str).str.upper().tolist())

    # Contar en cuantas ejecuciones aparece cada ticker
    ticker_freq = {}
    for t in all_tickers:
        count = sum(1 for df in history if t in df["Ticker"].astype(str).str.upper().values)
        ticker_freq[t] = count

    # Solo mostrar tickers que aparecen en al menos 2 ejecuciones
    relevant_tickers = sorted([t for t, c in ticker_freq.items() if c >= 2], key=lambda t: -ticker_freq[t])

    if not relevant_tickers:
        st.info("No hay activos que aparezcan en multiples ejecuciones todavia.")
        return

    # Selector de tickers
    default_show = relevant_tickers[:8]
    selected_tickers = st.multiselect(
        "Selecciona activos para ver su evolucion:",
        options=relevant_tickers,
        default=default_show,
        key="dashboard_ticker_select",
    )

    if not selected_tickers:
        st.info("Selecciona al menos un activo.")
        return

    # 1. Grafico de evolucion de puntaje
    score_rows = []
    for run_idx, df in enumerate(history, start=1):
        if "Puntaje" not in df.columns or "Ticker" not in df.columns:
            continue
        for _, row in df.iterrows():
            ticker = str(row["Ticker"]).upper()
            if ticker in selected_tickers:
                score_rows.append({
                    "Ejecucion": f"#{run_idx}",
                    "Run": run_idx,
                    "Ticker": ticker,
                    "Puntaje": int(row.get("Puntaje", 0)),
                })

    if score_rows:
        score_evo_df = pd.DataFrame(score_rows)

        st.markdown("#### Evolucion del puntaje por activo")
        chart = alt.Chart(score_evo_df).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("Run:O", title="Ejecucion", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Puntaje:Q", title="Puntaje", scale=alt.Scale(domain=[0, 7])),
            color=alt.Color("Ticker:N", legend=alt.Legend(title="Activo")),
            tooltip=["Ticker:N", "Ejecucion:N", "Puntaje:Q"],
        ).properties(height=400)

        # Bandas de referencia
        buy_band = alt.Chart(pd.DataFrame({"y": [6]})).mark_rule(
            strokeDash=[4, 4], color="#22c55e", strokeWidth=1
        ).encode(y="y:Q")
        watch_band = alt.Chart(pd.DataFrame({"y": [4]})).mark_rule(
            strokeDash=[4, 4], color="#eab308", strokeWidth=1
        ).encode(y="y:Q")

        st.altair_chart(chart + buy_band + watch_band, use_container_width=True)
        st.caption("Linea verde: umbral COMPRAR (6+) | Linea amarilla: umbral VIGILAR (4+)")

    # 2. Mapa de calor: decision por ejecucion
    decision_rows = []
    for run_idx, df in enumerate(history, start=1):
        if "Decision sugerida" not in df.columns or "Ticker" not in df.columns:
            continue
        for _, row in df.iterrows():
            ticker = str(row["Ticker"]).upper()
            if ticker in selected_tickers:
                decision = str(row.get("Decision sugerida", "")).upper()
                decision_rows.append({
                    "Ejecucion": f"#{run_idx}",
                    "Ticker": ticker,
                    "Decision": decision,
                    "Valor": _DECISION_RANK.get(decision, 0),
                })

    if decision_rows:
        dec_evo_df = pd.DataFrame(decision_rows)

        st.markdown("#### Mapa de decisiones por ejecucion")
        heat = alt.Chart(dec_evo_df).mark_rect(cornerRadius=3).encode(
            x=alt.X("Ejecucion:N", title="Ejecucion"),
            y=alt.Y("Ticker:N", title=None, sort=selected_tickers),
            color=alt.Color("Valor:Q",
                scale=alt.Scale(domain=[0, 1, 2, 3], range=["#e5e7eb", "#f87171", "#fbbf24", "#4ade80"]),
                legend=None,
            ),
            tooltip=["Ticker:N", "Ejecucion:N", "Decision:N"],
        ).properties(height=max(250, len(selected_tickers) * 28))

        text = alt.Chart(dec_evo_df).mark_text(fontSize=10, fontWeight="bold").encode(
            x="Ejecucion:N",
            y=alt.Y("Ticker:N", sort=selected_tickers),
            text="Decision:N",
            color=alt.Color("Valor:Q",
                scale=alt.Scale(domain=[0, 1, 2, 3], range=["#7f1d1d", "#7f1d1d", "#78350f", "#064e3b"]),
                legend=None,
            ),
        )

        st.altair_chart(heat + text, use_container_width=True)

    # 3. Tabla resumen: frecuencia y estabilidad
    st.markdown("#### Resumen de activos recurrentes")
    today = datetime.now()
    summary_rows = []
    for ticker in selected_tickers:
        first_seen: datetime | None = None
        buy_count = 0
        watch_count = 0
        avoid_count = 0
        scores = []
        for ts_str, df in history_wd:
            if "Ticker" not in df.columns:
                continue
            mask = df["Ticker"].astype(str).str.upper() == ticker
            if mask.any():
                if first_seen is None:
                    try:
                        first_seen = datetime.fromisoformat(ts_str)
                    except (ValueError, TypeError):
                        pass
                row = df[mask].iloc[0]
                decision = str(row.get("Decision sugerida", "")).upper()
                if decision == "COMPRAR":
                    buy_count += 1
                elif decision == "VIGILAR":
                    watch_count += 1
                elif decision == "EVITAR":
                    avoid_count += 1
                if "Puntaje" in df.columns:
                    scores.append(int(row.get("Puntaje", 0)))

        dias = max(1, (today - first_seen).days) if first_seen else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        trend = ""
        if len(scores) >= 2:
            if scores[-1] > scores[-2]:
                trend = "Mejorando"
            elif scores[-1] < scores[-2]:
                trend = "Empeorando"
            else:
                trend = "Estable"

        summary_rows.append({
            "Ticker": ticker,
            "Dias presente": f"{dias}d",
            "COMPRAR": buy_count,
            "VIGILAR": watch_count,
            "EVITAR": avoid_count,
            "Score medio": f"{avg_score:.1f}",
            "Tendencia": trend,
        })

    summary_df = pd.DataFrame(summary_rows)

    def _color_trend(val: str) -> str:
        v = str(val)
        if v == "Mejorando":
            return "background-color: #e8f7ea; color: #166534; font-weight: 600;"
        if v == "Empeorando":
            return "background-color: #fdeaea; color: #991b1b; font-weight: 600;"
        if v == "Estable":
            return "background-color: #f3f4f6; color: #374151;"
        return ""

    styled = summary_df.style.map(_color_trend, subset=["Tendencia"])
    st.dataframe(styled, use_container_width=True, hide_index=True)


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
# 5b. GRAFICOS DE VELAS (Candlestick)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Descargando datos OHLCV...")
def _fetch_ohlcv(ticker: str, months: int = 120) -> pd.DataFrame | None:
    """Descarga datos OHLCV semanales de un ticker (10 anos por defecto)."""
    from datetime import timedelta
    end = datetime.now()
    start = end - timedelta(days=months * 30)
    try:
        data = yf.download(
            ticker, start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"), interval="1wk", progress=False,
        )
        if data.empty or len(data) < 5:
            return None
        df = data[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df
    except Exception:
        return None


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calcula el RSI (Relative Strength Index)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_stoch_rsi(series: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """Calcula el Stochastic RSI (%K y %D)."""
    rsi = _compute_rsi(series, rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min) * 100
    k = stoch_rsi.rolling(k_smooth).mean()
    d = k.rolling(d_smooth).mean()
    return k, d


def _add_elliott_overlay(fig: go.Figure, df: pd.DataFrame) -> dict:
    """Dibuja ondas de Elliott etiquetadas (1-5 impulso, A-B-C correccion) sobre el grafico."""
    closes = df["Close"].tolist()
    if len(closes) < 60:
        return {}

    ew = classify_elliott_wave(closes)
    # Reusar los pivots adaptativos del clasificador (misma estructura que el grafico)
    pivots = ew.get("_pivots") or find_zigzag_pivots(closes, pct_threshold=5.0)
    if not pivots:
        return ew

    labeled = label_elliott_wave_pivots(pivots)

    # Mapear a fechas del DataFrame — solo pivots con etiqueta de onda
    dated: list[tuple] = []
    for idx, price, ptype, label in labeled:
        if label and idx < len(df):
            dated.append((df.index[idx], price, ptype, label))

    if len(dated) < 2:
        return ew

    # --- Segmentos coloreados conectando las ondas etiquetadas ---
    _SEG_COLORS = {
        "1": "#22c55e", "3": "#22c55e", "5": "#22c55e",  # impulso
        "2": "#eab308", "4": "#eab308",                    # correccion interna
        "A": "#ef4444", "B": "#ef4444", "C": "#ef4444",    # corrección ABC
        "0": "#94a3b8",
    }
    for i in range(len(dated) - 1):
        d1, p1, _, _ = dated[i]
        d2, p2, _, l2 = dated[i + 1]
        seg_color = _SEG_COLORS.get(l2, "#94a3b8")
        fig.add_trace(go.Scatter(
            x=[d1, d2], y=[p1, p2],
            mode="lines",
            line=dict(color=seg_color, width=2.5),
            showlegend=False,
            hoverinfo="skip",
        ))

    # --- Etiquetas de onda en cada pivot ---
    _LABEL_STYLE = {
        "0": ("#94a3b8", 7, 11),
        "1": ("#4ade80", 10, 15), "3": ("#34d399", 10, 15), "5": ("#4ade80", 10, 15),
        "2": ("#facc15", 10, 15), "4": ("#facc15", 10, 15),
        "A": ("#f87171", 10, 15), "C": ("#f87171", 10, 15),
        "B": ("#fb923c", 10, 15),
    }
    for date, price, ptype, label in dated:
        color, msize, fsize = _LABEL_STYLE.get(label, ("#94a3b8", 8, 11))
        position = "top center" if ptype == "H" else "bottom center"
        fig.add_trace(go.Scatter(
            x=[date], y=[price],
            mode="markers+text",
            marker=dict(size=msize, color=color, symbol="diamond"),
            text=[label],
            textposition=position,
            textfont=dict(size=fsize, color=color, family="Arial Black"),
            showlegend=False,
            hovertext=f"Onda {label}: ${price:.2f}",
            hoverinfo="text",
        ))

    # --- Lineas horizontales de targets ---
    if ew.get("target_1y", 0) > 0:
        for tgt_label, key, color, dash in [
            ("Target 1y", "target_1y", "#42a5f5", "dash"),
            ("Target 3y", "target_3y", "#ff9800", "dash"),
            ("Target 6y", "target_6y", "#ab47bc", "dashdot"),
        ]:
            val = ew[key]
            fig.add_hline(
                y=val, line_dash=dash, line_color=color, line_width=0.8,
                annotation_text=f"{tgt_label}: ${val:.2f}",
                annotation_position="top right",
                annotation_font_color=color,
                annotation_font_size=10,
            )

    return ew


def _build_candlestick_figure(df: pd.DataFrame, ticker: str) -> tuple[go.Figure, dict]:
    """Construye un grafico de velas con medias moviles y overlay de Elliott Wave."""
    fig = go.Figure()

    # Velas
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # SMA 20
    if len(df) >= 20:
        sma20 = df["Close"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma20, mode="lines",
            name="SMA 20", line=dict(color="#ff9800", width=1),
        ))

    # SMA 50
    if len(df) >= 50:
        sma50 = df["Close"].rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma50, mode="lines",
            name="SMA 50", line=dict(color="#2196f3", width=1),
        ))

    # SMA 200
    if len(df) >= 200:
        sma200 = df["Close"].rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma200, mode="lines",
            name="SMA 200", line=dict(color="#ab47bc", width=1, dash="dot"),
        ))

    # Overlay Elliott Wave
    ew = _add_elliott_overlay(fig, df)

    title_ew = ""
    if ew.get("onda"):
        title_ew = f" | Onda {ew['onda']} — {ew.get('fase', '')}"

    fig.update_layout(
        title=f"{_ticker_label(ticker)} — Semanal (10 anos){title_ew}",
        yaxis_title="Precio",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(l=40, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, ew


def _build_volume_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Construye un grafico de barras de volumen."""
    colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig = go.Figure(go.Bar(
        x=df.index, y=df["Volume"], marker_color=colors, name="Volumen",
    ))
    fig.update_layout(
        yaxis_title="Volumen",
        template="plotly_dark",
        height=150,
        margin=dict(l=40, r=20, t=10, b=20),
        showlegend=False,
    )
    return fig


def _build_rsi_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Construye un grafico de RSI(14) con zonas de sobrecompra/sobreventa."""
    rsi = _compute_rsi(df["Close"], 14)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, mode="lines",
        name="RSI(14)", line=dict(color="#42a5f5", width=1.5),
    ))

    # Zonas de referencia
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=0.8,
                  annotation_text="Sobrecompra (70)", annotation_position="top left")
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=0.8,
                  annotation_text="Sobreventa (30)", annotation_position="bottom left")
    fig.add_hrect(y0=70, y1=100, fillcolor="#ef5350", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="#26a69a", opacity=0.07, line_width=0)

    fig.update_layout(
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark",
        height=200,
        margin=dict(l=40, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_stoch_rsi_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Construye un grafico de Stochastic RSI (%K y %D)."""
    k, d = _compute_stoch_rsi(df["Close"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=k, mode="lines",
        name="%K", line=dict(color="#42a5f5", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=d, mode="lines",
        name="%D", line=dict(color="#ff9800", width=1.2, dash="dot"),
    ))

    # Zonas de referencia
    fig.add_hline(y=80, line_dash="dash", line_color="#ef5350", line_width=0.8,
                  annotation_text="80", annotation_position="top left")
    fig.add_hline(y=20, line_dash="dash", line_color="#26a69a", line_width=0.8,
                  annotation_text="20", annotation_position="bottom left")
    fig.add_hrect(y0=80, y1=100, fillcolor="#ef5350", opacity=0.07, line_width=0)
    fig.add_hrect(y0=0, y1=20, fillcolor="#26a69a", opacity=0.07, line_width=0)

    fig.update_layout(
        yaxis_title="Stoch RSI",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark",
        height=200,
        margin=dict(l=40, r=20, t=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def render_candlestick(top10_df: pd.DataFrame, score_df: pd.DataFrame, pass_id: str = "p1") -> None:
    """Renderiza graficos de velas para los activos del Top 20."""
    if top10_df.empty:
        return

    tickers = top10_df["Ticker"].tolist()
    if not tickers:
        return

    st.subheader("Graficos de velas — Semanal (10 anos) con Ondas de Elliott")

    # Pre-seleccionar solo los COMPRAR si hay scoreboard
    default_tickers = tickers[:5]
    if not score_df.empty and "Decision sugerida" in score_df.columns:
        buy_tickers = score_df[
            score_df["Decision sugerida"].str.upper() == "COMPRAR"
        ]["Ticker"].tolist()
        if buy_tickers:
            default_tickers = buy_tickers[:5]

    selected = st.multiselect(
        "Selecciona activos para ver velas:",
        options=tickers,
        default=[t for t in default_tickers if t in tickers],
        key=f"candle_select_{pass_id}",
        format_func=_ticker_label,
    )

    if not selected:
        st.info("Selecciona al menos un activo para ver su grafico.")
        return

    for ticker in selected:
        ohlcv = _fetch_ohlcv(ticker)
        if ohlcv is None:
            st.warning(f"No se pudieron obtener datos para {ticker}.")
            continue

        st.markdown(f"#### {_ticker_label(ticker)}")

        candle_fig, ew = _build_candlestick_figure(ohlcv, ticker)
        st.plotly_chart(candle_fig, use_container_width=True, key=f"candle_{pass_id}_{ticker}")

        # Info Elliott Wave
        if ew.get("onda") and ew["onda"] != "N/D":
            _favorable = {"1↑", "2↓", "3↑", "4↓", "C↓"}
            _color = "#4ade80" if ew["onda"] in _favorable else "#f87171"
            e1, e2, e3, e4 = st.columns(4)
            e1.markdown(f"**Onda:** <span style='color:{_color};font-size:1.3em;font-weight:700'>{ew['onda']}</span> — {ew.get('fase', '')}", unsafe_allow_html=True)
            e2.metric("Target 1y", f"${ew['target_1y']:.2f}", f"{ew['rent_1y']:+.1f}%")
            e3.metric("Target 3y", f"${ew['target_3y']:.2f}", f"{ew['rent_3y']:+.1f}%")
            e4.metric("Target 6y", f"${ew['target_6y']:.2f}", f"{ew['rent_6y']:+.1f}%")
            st.caption(f"📊 {ew.get('prevision', '')}")

        volume_fig = _build_volume_figure(ohlcv, ticker)
        st.plotly_chart(volume_fig, use_container_width=True, key=f"vol_{pass_id}_{ticker}")

        rsi_fig = _build_rsi_figure(ohlcv, ticker)
        st.plotly_chart(rsi_fig, use_container_width=True, key=f"rsi_{pass_id}_{ticker}")

        stoch_fig = _build_stoch_rsi_figure(ohlcv, ticker)
        st.plotly_chart(stoch_fig, use_container_width=True, key=f"stochrsi_{pass_id}_{ticker}")

        # Metricas rapidas
        last_close = ohlcv["Close"].iloc[-1]
        prev_close = ohlcv["Close"].iloc[-2] if len(ohlcv) >= 2 else last_close
        first_close = ohlcv["Close"].iloc[0]
        change_1d = ((last_close - prev_close) / prev_close) * 100
        change_period = ((last_close - first_close) / first_close) * 100
        high_period = ohlcv["High"].max()
        low_period = ohlcv["Low"].min()

        rsi_val = _compute_rsi(ohlcv["Close"], 14).iloc[-1]
        k_val, _d_val = _compute_stoch_rsi(ohlcv["Close"])
        stoch_val = k_val.iloc[-1]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Cierre", f"${last_close:.2f}", f"{change_1d:+.2f}%")
        c2.metric("Cambio 5a", f"{change_period:+.1f}%")
        c3.metric("Maximo", f"${high_period:.2f}")
        c4.metric("Minimo", f"${low_period:.2f}")
        c5.metric("RSI(14)", f"{rsi_val:.1f}")
        c6.metric("Stoch RSI", f"{stoch_val:.1f}")
        st.divider()


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

            t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14 = st.tabs([
                "Resumen Pasada 1",
                "Tablas Pasada 1",
                "Resumen Pasada 2 (sin EVITAR)",
                "Tablas Pasada 2",
                "Resumen Pasada 3 (filtrado final)",
                "Tablas Pasada 3",
                "Flujo entre pasadas",
                "Alertas de precio",
                "Señales de salida",
                "Niveles de entrada",
                "Performance",
                "Analítica Pro",
                "Dashboard",
                "Historico",
            ])

            with t1:
                _render_pass_summary(pass1_output, pass_id="p1")
            with t2:
                _render_pass_tables(pass1_output)
            with t3:
                _render_pass_summary(pass2_output, pass_id="p2")
            with t4:
                _render_pass_tables(pass2_output)
            with t5:
                _render_pass_summary(pass3_output, pass_id="p3")
            with t6:
                _render_pass_tables(pass3_output)
            with t7:
                render_funnel(full_output)
            with t8:
                render_price_alerts()
            with t9:
                render_exit_signals()
            with t10:
                render_entry_levels()
            with t11:
                render_performance()
            with t12:
                render_professional_analytics()
            with t13:
                render_dashboard()
            with t14:
                render_history_tab()
        else:
            pass2_output = full_output[idx2:]

            t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12 = st.tabs([
                "Resumen Pasada 1",
                "Tablas Pasada 1",
                "Resumen Pasada 2 (sin EVITAR)",
                "Tablas Pasada 2",
                "Flujo entre pasadas",
                "Alertas de precio",
                "Señales de salida",
                "Niveles de entrada",
                "Performance",
                "Analítica Pro",
                "Dashboard",
                "Historico",
            ])

            with t1:
                _render_pass_summary(pass1_output, pass_id="p1")
            with t2:
                _render_pass_tables(pass1_output)
            with t3:
                _render_pass_summary(pass2_output, pass_id="p2")
            with t4:
                _render_pass_tables(pass2_output)
            with t5:
                render_funnel(full_output)
            with t6:
                render_price_alerts()
            with t7:
                render_exit_signals()
            with t8:
                render_entry_levels()
            with t9:
                render_performance()
            with t10:
                render_professional_analytics()
            with t11:
                render_dashboard()
            with t12:
                render_history_tab()
    else:
        t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
            "Resumen", "Tablas de evaluacion", "Alertas de precio",
            "Señales de salida", "Niveles de entrada", "Performance", "Analítica Pro",
            "Dashboard", "Historico",
        ])
        with t1:
            _render_pass_summary(full_output)
        with t2:
            _render_pass_tables(full_output)
        with t3:
            render_price_alerts()
        with t4:
            render_exit_signals()
        with t5:
            render_entry_levels()
        with t6:
            render_performance()
        with t7:
            render_professional_analytics()
        with t8:
            render_dashboard()
        with t9:
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

    tab_debate, tab_portfolio = st.tabs(["Debate Macro", "Mi Cartera"])

    # Inicializar session_state para persistir resultados entre reruns
    for _key in ("debate_code", "debate_output", "portfolio_code", "portfolio_output"):
        if _key not in st.session_state:
            st.session_state[_key] = None

    # ── TAB 1: Debate Macro (flujo original) ──
    with tab_debate:
        run_clicked = st.button("Iniciar debate", type="primary", use_container_width=True, key="btn_debate")

        if run_clicked:
            cmd = build_command(
                model=model.strip(),
                host=host.strip(),
                seconds=int(seconds),
                max_turns=int(max_turns),
                context_lines=int(context_lines),
            )
            code, full_output = stream_process(cmd)
            st.session_state["debate_code"] = code
            st.session_state["debate_output"] = full_output

            # Guardar en historico (siempre la ultima pasada disponible)
            if code == 0:
                last_pass = _extract_latest_pass_output(full_output)
                tables = extract_named_tables(last_pass)
                top10_df = extract_top10_table(last_pass)
                score_df = build_scoreboard(top10_df, tables)
                _save_run(model.strip(), score_df, full_output)

        # Renderizar siempre que haya resultados guardados
        if st.session_state["debate_output"] is not None:
            full_output = st.session_state["debate_output"]
            code = st.session_state["debate_code"]

            st.divider()
            st.subheader("Resultado")
            if code == 0:
                st.success("Ejecucion completada.")
            else:
                st.error("La ejecucion termino con errores. Revisa la salida completa.")

            consensus = extract_consensus(full_output)
            if consensus:
                st.info(f"**Conclusion macroeconomica de los agentes:**\n\n{consensus}")

            render_timer(full_output)
            render_results(full_output)

            st.download_button(
                label="Descargar log completo",
                data=full_output.encode("utf-8"),
                file_name="debate_macro_output.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # ── TAB 2: Mi Cartera ──
    with tab_portfolio:
        pf_tab1, pf_tab2, pf_tab3 = st.tabs(["Mis posiciones", "Analizar cartera", "Historial cerradas"])

        # --- Sub-tab 1: Gestor de posiciones ---
        with pf_tab1:
            st.subheader("Mis posiciones reales")

            # Formulario para añadir posición
            with st.expander("Añadir posición", expanded=False):
                fc1, fc2, fc3, fc4 = st.columns(4)
                new_ticker = fc1.text_input("Ticker", placeholder="AAPL", key="pf_new_ticker")
                new_shares = fc2.number_input("Acciones", min_value=0.001, value=1.0, step=1.0, key="pf_new_shares")
                new_price = fc3.number_input("Precio entrada", min_value=0.01, value=100.0, step=0.01, key="pf_new_price")
                new_date = fc4.date_input("Fecha entrada", value=datetime.now(), key="pf_new_date")
                new_notes = st.text_input("Notas (opcional)", placeholder="Compra por fundamentales sólidos", key="pf_new_notes")

                if st.button("Añadir posición", key="btn_add_position"):
                    if new_ticker.strip():
                        add_position(
                            new_ticker.strip(),
                            new_shares,
                            new_price,
                            new_date.strftime("%Y-%m-%d"),
                            new_notes,
                        )
                        st.success(f"Posición añadida: {new_ticker.strip().upper()}")
                        st.rerun()
                    else:
                        st.warning("Introduce un ticker válido.")

            # Snapshot actual
            snapshot = portfolio_snapshot()
            if snapshot.empty:
                st.info("No tienes posiciones registradas. Usa el formulario de arriba para añadir tus posiciones reales.")
            else:
                # Resumen agregado
                risk = portfolio_risk_summary(snapshot)
                if risk:
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Valor total", f"${risk['total_value']:,.0f}")
                    m2.metric("P&L total", f"${risk['total_pnl']:+,.0f}",
                              delta=f"{risk['total_pnl_pct']:+.1f}%")
                    m3.metric("Posiciones", risk["num_positions"])
                    m4.metric("En ganancia", risk["num_winning"])
                    m5.metric("En pérdida", risk["num_losing"])

                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Concentración Top 3", f"{risk['top3_concentration']:.0f}%")
                    mc2.metric("Mayor ganancia", f"{risk['max_gain_ticker']} ({risk['max_gain_pct']:+.1f}%)")
                    mc3.metric("Mayor pérdida", f"{risk['max_loss_ticker']} ({risk['max_loss_pct']:+.1f}%)")

                    # Exposición por sector
                    if risk.get("sector_exposure"):
                        with st.expander("Exposición por sector"):
                            sector_df = pd.DataFrame([
                                {"Sector": k, "Peso %": v}
                                for k, v in risk["sector_exposure"].items()
                            ])
                            st.dataframe(sector_df, use_container_width=True, hide_index=True)

                # Tabla de posiciones con P&L
                def _color_pnl(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "color: #22c55e; font-weight: 600"
                        elif val < 0:
                            return "color: #ef4444; font-weight: 600"
                    return ""

                display_cols = ["Ticker", "Acciones", "Precio entrada", "Precio actual",
                                "P&L", "P&L %", "Peso %", "Fecha entrada", "Notas"]
                display_df = snapshot[[c for c in display_cols if c in snapshot.columns]]
                styled = display_df.style.applymap(_color_pnl, subset=["P&L", "P&L %"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Cerrar / eliminar posiciones
                with st.expander("Cerrar o eliminar posición"):
                    pos_options = {
                        f"{row['Ticker']} ({row['Acciones']} acc. @ ${row['Precio entrada']:.2f}) — id:{row['id']}": int(row["id"])
                        for _, row in snapshot.iterrows()
                    }
                    selected_pos = st.selectbox("Seleccionar posición", list(pos_options.keys()), key="pf_close_select")
                    pos_id = pos_options.get(selected_pos, 0)

                    ac1, ac2 = st.columns(2)
                    close_price_input = ac1.number_input("Precio de venta", min_value=0.01, value=100.0, key="pf_close_price")
                    if ac1.button("Cerrar (vender)", key="btn_close_pos"):
                        if pos_id:
                            close_position(pos_id, close_price_input)
                            st.success("Posición cerrada.")
                            st.rerun()
                    if ac2.button("Eliminar registro", key="btn_del_pos", type="secondary"):
                        if pos_id:
                            delete_position(pos_id)
                            st.warning("Posición eliminada.")
                            st.rerun()

        # --- Sub-tab 2: Debate focalizado ---
        with pf_tab2:
            st.subheader("Analiza tu cartera con debate focalizado")
            st.markdown(
                "Introduce los tickers de los activos que tienes en cartera. "
                "El sistema ejecutara un **debate focalizado** entre los 7 gestores sobre tus activos, "
                "seguido de las **8 evaluaciones independientes** y el **veredicto final**."
            )

            # Pre-rellenar con posiciones abiertas si existen
            open_pos = get_open_positions()
            default_tickers = ", ".join(open_pos["ticker"].unique()) if not open_pos.empty else ""

            portfolio_input = st.text_area(
                "Tickers (separados por comas)",
                value=default_tickers,
                placeholder="Ej: AAPL, MSFT, TSLA, NVDA, IBE.MC, ITX.MC",
                height=80,
                key="portfolio_tickers_input",
            )

            portfolio_run = st.button("Analizar Mi Cartera", type="primary", use_container_width=True, key="btn_portfolio")

            if portfolio_run:
                raw_tickers = portfolio_input.strip()
                if not raw_tickers:
                    st.warning("Introduce al menos un ticker para analizar.")
                else:
                    # Limpiar y validar
                    tickers_clean = ",".join(t.strip().upper() for t in raw_tickers.replace(";", ",").split(",") if t.strip())
                    ticker_list = [t for t in tickers_clean.split(",") if t]
                    st.info(f"Analizando **{len(ticker_list)}** activos: {', '.join(ticker_list)}")

                    cmd = build_command(
                        model=model.strip(),
                        host=host.strip(),
                        seconds=int(seconds),
                        max_turns=int(max_turns),
                        context_lines=int(context_lines),
                        portfolio=tickers_clean,
                    )
                    code, full_output = stream_process(cmd)
                    st.session_state["portfolio_code"] = code
                    st.session_state["portfolio_output"] = full_output

                    # Guardar en historico
                    if code == 0:
                        last_pass = _extract_latest_pass_output(full_output)
                        tables = extract_named_tables(last_pass)
                        top10_df = extract_top10_table(last_pass)
                        score_df = build_scoreboard(top10_df, tables)
                        _save_run(f"{model.strip()} [Cartera]", score_df, full_output)

            # Renderizar siempre que haya resultados guardados
            if st.session_state["portfolio_output"] is not None:
                full_output = st.session_state["portfolio_output"]
                code = st.session_state["portfolio_code"]

                st.divider()
                st.subheader("Resultado - Mi Cartera")
                if code == 0:
                    st.success("Analisis completado.")
                else:
                    st.error("El analisis termino con errores. Revisa la salida completa.")

                consensus = extract_consensus(full_output)
                if consensus:
                    st.info(f"**Conclusion de los agentes sobre tu cartera:**\n\n{consensus}")

                render_timer(full_output)

                # Renderizar resultados (1 sola pasada, sin buscar P2/P3)
                _render_portfolio_results(full_output)

                st.download_button(
                    label="Descargar log Mi Cartera",
                    data=full_output.encode("utf-8"),
                    file_name="mi_cartera_output.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        # --- Sub-tab 3: Historial de posiciones cerradas ---
        with pf_tab3:
            st.subheader("Historial de posiciones cerradas")
            closed = get_closed_positions()
            if closed.empty:
                st.info("No hay posiciones cerradas todavía.")
            else:
                closed["P&L"] = (closed["close_price"] - closed["entry_price"]) * closed["shares"]
                closed["P&L %"] = ((closed["close_price"] - closed["entry_price"]) / closed["entry_price"] * 100).round(1)
                closed = closed.rename(columns={
                    "ticker": "Ticker", "shares": "Acciones",
                    "entry_price": "Precio entrada", "entry_date": "Fecha entrada",
                    "close_price": "Precio venta", "close_date": "Fecha venta",
                    "notes": "Notas",
                })

                total_pnl = closed["P&L"].sum()
                wins = (closed["P&L"] > 0).sum()
                losses = (closed["P&L"] <= 0).sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("P&L total cerradas", f"${total_pnl:+,.0f}")
                c2.metric("Operaciones ganadoras", wins)
                c3.metric("Operaciones perdedoras", losses)

                display_cols = ["Ticker", "Acciones", "Precio entrada", "Fecha entrada",
                                "Precio venta", "Fecha venta", "P&L", "P&L %", "Notas"]
                display_df = closed[[c for c in display_cols if c in closed.columns]]

                def _color_pnl_closed(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "color: #22c55e; font-weight: 600"
                        elif val < 0:
                            return "color: #ef4444; font-weight: 600"
                    return ""

                styled = display_df.style.applymap(_color_pnl_closed, subset=["P&L", "P&L %"])
                st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_portfolio_results(full_output: str) -> None:
    """Renderiza resultados de Mi Cartera: resumen + tablas en una sola pasada."""
    if not full_output.strip():
        return
    tab_resumen, tab_tablas = st.tabs(["Resumen", "Tablas de evaluacion"])
    with tab_resumen:
        _render_pass_summary(full_output, pass_id="cartera")
    with tab_tablas:
        _render_pass_tables(full_output)


if __name__ == "__main__":
    main()
