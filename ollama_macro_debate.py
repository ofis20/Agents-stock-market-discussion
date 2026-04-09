#!/usr/bin/env python3
"""Debate macroeconomico entre 7 agentes con Ollama local y streaming en vivo."""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import requests

from debate_agents import AGENTS, Agent
from debate_evaluators import (
    extract_evitar_tickers,
    final_verdict,
    fundamental_analysis_review,
    institutional_analysis_review,
    macd_analysis_review,
    relative_analysis_review,
    risk_management_review,
    sentiment_analysis_review,
    technical_analysis_review,
    wyckoff_analysis_review,
)
from debate_portfolio import top20_investments
from debate_prompts import (
    CONSENSUS_FILLERS,
    build_agent_turn_prompt,
    build_consensus_messages,
)
from guru_holdings import (
    fetch_all_guru_holdings,
    compute_guru_conviction,
    guru_holdings_section,
)
from macro_context import load_macro_context
from market_data import TICKERS, load_all_market_data
from ollama_client import DEFAULT_HOST, OllamaClient, assign_models_to_agents


# ═══════════════════════════════════════════════════════════════════════════
# CRONOMETRO POR ETAPAS
# ═══════════════════════════════════════════════════════════════════════════

class _StageTimer:
    """Registra tiempos por etapa y muestra resumen al final."""

    def __init__(self):
        self._t0 = time.monotonic()
        self._stages: list[tuple[str, float]] = []
        self._lap = self._t0

    def stage(self, name: str) -> None:
        """Marca el fin de la etapa actual y el inicio de la siguiente."""
        now = time.monotonic()
        elapsed = now - self._lap
        self._stages.append((name, elapsed))
        print(f"  ⏱  {name}: {elapsed:.1f}s", flush=True)
        self._lap = now

    def summary(self) -> None:
        total = time.monotonic() - self._t0
        print("\n" + "=" * 60, flush=True)
        print("CRONOMETRO DE EJECUCION", flush=True)
        print("=" * 60, flush=True)
        for name, secs in self._stages:
            pct = (secs / total * 100) if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {name:<35} {secs:>7.1f}s  ({pct:4.1f}%) {bar}", flush=True)
        print(f"  {'─' * 55}", flush=True)
        mm, ss = divmod(int(total), 60)
        print(f"  {'TOTAL':<35} {total:>7.1f}s  ({mm}m {ss}s)", flush=True)
        print("=" * 60 + "\n", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# EVALUADORES SECUENCIALES (CPU-bound – threads no ayudan por el GIL)
# ═══════════════════════════════════════════════════════════════════════════

def _run_evaluators(
    top20_assets: list[dict[str, Any]],
    raw_prices: dict[str, dict[str, Any]],
    raw_fundamentals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Ejecuta los 8 evaluadores deterministas en orden secuencial.

    Se ejecutan en serie para que la salida stdout sea ordenada y
    ``extract_named_tables`` pueda asociar cada tabla a su seccion.
    """
    ta = technical_analysis_review(top10_assets=top20_assets, prices=raw_prices)
    fa = fundamental_analysis_review(top10_assets=top20_assets, fundamentals=raw_fundamentals, prices=raw_prices)
    risk = risk_management_review(top10_assets=top20_assets, prices=raw_prices)
    sent = sentiment_analysis_review(top10_assets=top20_assets, prices=raw_prices)
    macd = macd_analysis_review(top10_assets=top20_assets, prices=raw_prices)
    inst = institutional_analysis_review(top10_assets=top20_assets, fundamentals=raw_fundamentals, prices=raw_prices)
    wyckoff = wyckoff_analysis_review(top10_assets=top20_assets, prices=raw_prices)
    relative = relative_analysis_review(top10_assets=top20_assets, prices=raw_prices, fundamentals=raw_fundamentals)

    return {
        "ta": ta, "fa": fa, "risk": risk, "sent": sent,
        "macd": macd, "inst": inst, "wyckoff": wyckoff, "relative": relative,
    }


def elliott_llm_review(
    client: OllamaClient,
    model: str,
    ew_table: str,
    top20_assets: list[dict[str, Any]],
) -> str:
    """Usa el mejor modelo LLM para interpretar los datos de Elliott Wave a largo plazo."""
    print("\n" + "=" * 60, flush=True)
    print("INTERPRETACION ELLIOTT WAVE LARGO PLAZO (LLM)", flush=True)
    print("=" * 60, flush=True)

    tickers_str = ", ".join(a["ticker"] for a in top20_assets[:10])
    system = (
        "Eres un analista experto en Ondas de Elliott con enfoque exclusivo de MUY LARGO PLAZO (1-6 anos). "
        "Se te proporcionan datos algoritmicos de Elliott Wave basados en 6 anos de historico, con targets a 1, 3 y 6 anos. "
        "Tu trabajo es:\n"
        "1. Interpretar en que fase del SUPERCICLO se encuentra cada activo (usando teoria de grados de onda de Elliott)\n"
        "2. Identificar los 3-5 activos con mejor estructura de onda para acumulacion generacional\n"
        "3. Senalar los activos en ondas peligrosas (5↑, A↓) donde NO se debe invertir\n"
        "4. Dar una vision del SUPERCICLO completo: donde estamos en la gran estructura de mercado a 6 anos\n"
        "Responde en espanol, maximo 15 lineas, sin saludos ni despedidas. Se conciso y directo."
    )
    user = (
        f"Activos analizados: {tickers_str}\n\n"
        f"Resultados algoritmicos de Elliott Wave (6 anos de historico, targets a 1y/3y/6y):\n{ew_table}\n\n"
        "Interpreta estos datos con vision de SUPERCICLO (1-6 anos). "
        "Identifica los mejores activos para acumulacion generacional y los que deben evitarse."
    )

    print(f"\n[Analista Elliott LLM — modelo: {model}]", flush=True)
    text = client.stream_chat(
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        num_predict=600,
        temperature=0.3,
    )
    return text

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debate en vivo entre 7 agentes financieros con Ollama local."
    )
    parser.add_argument("--model", default="llama3.1", help="Modelo local de Ollama")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host base de Ollama")
    parser.add_argument("--seconds", type=int, default=180, help="Duracion del debate en segundos")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=56,
        help="Limite de turnos para evitar bucles largos si el modelo responde rapido",
    )
    parser.add_argument(
        "--context-lines",
        type=int,
        default=18,
        help="Numero de lineas recientes del debate compartidas en cada turno",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="",
        help="Tickers separados por comas para analizar Mi Cartera (ej: AAPL,MSFT,TSLA). "
             "Si se indica, ejecuta debate focalizado + evaluaciones sobre esos activos.",
    )
    return parser.parse_args()


def recent_context(transcript: list[str], max_lines: int) -> str:
    if not transcript:
        return "(sin mensajes previos)"
    return "\n".join(transcript[-max_lines:])


def _sanitize_agent_reply(text: str, max_lines: int = 6, max_chars: int = 900) -> str:
    """Recorta respuestas largas/repetitivas para mantener el debate util."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    clipped = lines[:max_lines] if lines else [text.strip()]
    out = " ".join(clipped).strip()
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "..."
    return out


def agent_turn(
    client: OllamaClient,
    agent: Agent,
    model: str,
    transcript: list[str],
    context_lines: int,
    market_briefing: str = "",
) -> str:
    context = recent_context(transcript, context_lines)
    user_prompt = build_agent_turn_prompt(context=context, market_briefing=market_briefing)

    print(f"\n[{agent.name}] (modelo: {model})", flush=True)
    reply = client.stream_chat(
        model,
        messages=[
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        num_predict=agent.max_tokens,
        temperature=agent.temperature,
    )

    clean_reply = _sanitize_agent_reply(reply)
    line = f"{agent.name}: {clean_reply}"
    transcript.append(line)
    return clean_reply


def consensus_and_summary(client: OllamaClient, model: str, transcript: list[str], context_lines: int, market_briefing: str = "") -> str:
    context = recent_context(transcript, max_lines=max(context_lines * 4, 60))
    system, user = build_consensus_messages(context=context, market_briefing=market_briefing)

    print("\n[Moderador - consenso final 10 lineas]", flush=True)
    text = client.stream_chat(
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )

    # Asegura exactamente 10 lineas limpias para cumplir la especificacion.
    cleaned: list[str] = []
    for ln in text.splitlines():
        line = ln.strip()
        if not line:
            continue
        # Elimina numeracion al inicio: "1. ...", "2) ..."
        line = re.sub(r"^\s*\d+[\.)]\s*", "", line).strip()
        # Descarta lineas truncadas o demasiado cortas.
        if len(line) < 25:
            continue
        if line not in cleaned:
            cleaned.append(line)

    lines = cleaned[:10]
    fill_i = 0
    while len(lines) < 10:
        lines.append(CONSENSUS_FILLERS[fill_i % len(CONSENSUS_FILLERS)])
        fill_i += 1

    final_text = "\n".join(lines)
    print("\n[Resumen validado - 10 lineas]", flush=True)
    print(final_text, flush=True)
    return final_text


def run_debate(args: argparse.Namespace) -> int:
    timer = _StageTimer()
    client = OllamaClient(args.host)
    available_models = client.refresh_models()
    model = client.resolve_model(args.model)

    # Asignar modelos diversos a cada agente y analista
    analyst_roles = [
        "Moderador Consenso", "Moderador Top 20",
        "Analista Tecnico", "Analista Fundamental",
        "Gestor Riesgos", "Analista Sentimiento", "Analista Elliott", "Veredicto Final",
    ]
    model_map = assign_models_to_agents(available_models, AGENTS, analyst_roles)

    agent_names = ", ".join(a.name for a in AGENTS)
    print("=== Debate Macro Global en Vivo ===")
    print(f"Modelo principal: {model}")
    print(f"Duracion objetivo: {args.seconds} segundos")
    print(f"Participantes debate: {agent_names}")
    print("\nAsignacion de modelos:")
    for role, m in model_map.items():
        print(f"  {role:<25} -> {m}")
    print("=" * 60)
    timer.stage("Inicializacion")

    # Fase 0: Cargar datos reales de mercado, carteras de gurus y contexto macro en paralelo
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_market = pool.submit(load_all_market_data, 72, True)
        fut_gurus = pool.submit(fetch_all_guru_holdings)
        fut_macro = pool.submit(load_macro_context)
        briefings = fut_market.result()
        all_guru_holdings = fut_gurus.result()
        macro_ctx = fut_macro.result()

    market_general = briefings.get("general", "")
    raw_prices: dict[str, dict[str, Any]] = briefings.get("raw_prices", {})
    raw_fundamentals: dict[str, dict[str, Any]] = briefings.get("raw_fundamentals", {})

    # Enriquecer briefing con contexto macro adicional (economia, noticias, earnings)
    extra_sections: list[str] = []
    if macro_ctx.get("economic"):
        extra_sections.append(macro_ctx["economic"])
    if macro_ctx.get("earnings"):
        extra_sections.append(macro_ctx["earnings"])
    if macro_ctx.get("news"):
        extra_sections.append(macro_ctx["news"])
    if extra_sections:
        market_general = market_general + "\n\n" + "\n\n".join(extra_sections)

    if market_general:
        print("\n[Datos de mercado + contexto macro cargados correctamente]")
    else:
        print("\n[AVISO: Sin datos de mercado - el debate usara solo conocimiento del modelo]")

    guru_conv = compute_guru_conviction(all_guru_holdings, TICKERS)
    timer.stage("Carga datos mercado + gurus + contexto macro")

    transcript: List[str] = []
    start = time.monotonic()
    deadline = start + max(5, args.seconds)

    turn_count = 0
    while time.monotonic() < deadline and turn_count < args.max_turns:
        for agent in AGENTS:
            if time.monotonic() >= deadline or turn_count >= args.max_turns:
                break
            agent_turn(
                client=client,
                agent=agent,
                model=model_map[agent.name],
                transcript=transcript,
                context_lines=args.context_lines,
                market_briefing=market_general,
            )
            turn_count += 1

    elapsed = time.monotonic() - start
    print(f"\nDebate finalizado en {elapsed:.1f} segundos con {turn_count} turnos.")
    timer.stage(f"Debate ({turn_count} turnos)")

    # Fase 2: Consenso
    consensus_and_summary(
        client=client,
        model=model_map["Moderador Consenso"],
        transcript=transcript,
        context_lines=args.context_lines,
        market_briefing=market_general,
    )
    timer.stage("Consenso")

    # Fase 3: Top 20 inversiones
    _top20_text, top20_assets = top20_investments(
        client=client,
        model=model_map["Moderador Top 20"],
        transcript=transcript,
        recent_context_fn=recent_context,
        prices=raw_prices,
        fundamentals=raw_fundamentals,
        market_briefing=market_general,
        guru_conviction=guru_conv,
    )
    timer.stage("Top 20 inversiones")

    # Fase 4: Evaluaciones independientes EN PARALELO (con datos reales)
    print("\n[Ejecutando 8 evaluadores...]", flush=True)
    ev = _run_evaluators(top20_assets, raw_prices, raw_fundamentals)
    ta_combined, ew_table = ev["ta"]
    fa_table = ev["fa"]
    risk_table = ev["risk"]
    sent_table = ev["sent"]
    macd_table = ev["macd"]
    inst_table = ev["inst"]
    wyckoff_table = ev["wyckoff"]
    relative_table = ev["relative"]
    timer.stage("Evaluadores (8 en paralelo)")

    # Fase 4b: Interpretacion Elliott Wave largo plazo (LLM, mejor modelo)
    elliott_llm_review(
        client=client,
        model=model_map["Analista Elliott"],
        ew_table=ew_table,
        top20_assets=top20_assets,
    )
    timer.stage("Elliott LLM largo plazo")

    # Fase 5: Veredicto final consolidado
    verdict_output = final_verdict(
        top10_assets=top20_assets,
        ta_table=ta_combined,
        fa_table=fa_table,
        risk_table=risk_table,
        sent_table=sent_table,
        ew_table=ew_table,
        macd_table=macd_table,
        inst_table=inst_table,
        wyckoff_table=wyckoff_table,
        relative_table=relative_table,
    )

    timer.stage("Veredicto pasada 1")

    # Fase 5b: Carteras reales de gurus
    guru_holdings_section(
        all_holdings=all_guru_holdings,
        tickers_dict=TICKERS,
        top20_tickers=[a["ticker"] for a in top20_assets],
    )

    # === SEGUNDA PASADA: excluir activos EVITAR ===
    evitar_tickers = extract_evitar_tickers(verdict_output)
    if evitar_tickers:
        print("\n" + "=" * 60, flush=True)
        print("[=== SEGUNDA PASADA (sin EVITAR) ===]", flush=True)
        print("=" * 60, flush=True)
        print(f"Excluidos ({len(evitar_tickers)}): {', '.join(sorted(evitar_tickers))}", flush=True)

        _top20_text2, top20_assets2 = top20_investments(
            client=client,
            model=model_map["Moderador Top 20"],
            transcript=transcript,
            recent_context_fn=recent_context,
            prices=raw_prices,
            fundamentals=raw_fundamentals,
            market_briefing=market_general,
            exclude_tickers=evitar_tickers,
            guru_conviction=guru_conv,
        )

        print("\n[Ejecutando 8 evaluadores (pasada 2)...]", flush=True)
        ev2 = _run_evaluators(top20_assets2, raw_prices, raw_fundamentals)
        ta_combined2, ew_table2 = ev2["ta"]
        fa_table2 = ev2["fa"]
        risk_table2 = ev2["risk"]
        sent_table2 = ev2["sent"]
        macd_table2 = ev2["macd"]
        inst_table2 = ev2["inst"]
        wyckoff_table2 = ev2["wyckoff"]
        relative_table2 = ev2["relative"]
        verdict_output2 = final_verdict(
            top10_assets=top20_assets2,
            ta_table=ta_combined2,
            fa_table=fa_table2,
            risk_table=risk_table2,
            sent_table=sent_table2,
            ew_table=ew_table2,
            macd_table=macd_table2,
            inst_table=inst_table2,
            wyckoff_table=wyckoff_table2,
            relative_table=relative_table2,
        )

        # === TERCERA PASADA: excluir tambien los EVITAR de la pasada 2 ===
        evitar_tickers2 = extract_evitar_tickers(verdict_output2)
        if evitar_tickers2:
            all_excluded = evitar_tickers | evitar_tickers2
            print("\n" + "=" * 60, flush=True)
            print("[=== TERCERA PASADA (sin EVITAR P1+P2) ===]", flush=True)
            print("=" * 60, flush=True)
            print(f"Nuevos excluidos P2 ({len(evitar_tickers2)}): {', '.join(sorted(evitar_tickers2))}", flush=True)
            print(f"Total excluidos ({len(all_excluded)}): {', '.join(sorted(all_excluded))}", flush=True)

            _top20_text3, top20_assets3 = top20_investments(
                client=client,
                model=model_map["Moderador Top 20"],
                transcript=transcript,
                recent_context_fn=recent_context,
                prices=raw_prices,
                fundamentals=raw_fundamentals,
                market_briefing=market_general,
                exclude_tickers=all_excluded,
                guru_conviction=guru_conv,
            )

            print("\n[Ejecutando 8 evaluadores (pasada 3)...]", flush=True)
            ev3 = _run_evaluators(top20_assets3, raw_prices, raw_fundamentals)
            ta_combined3, ew_table3 = ev3["ta"]
            fa_table3 = ev3["fa"]
            risk_table3 = ev3["risk"]
            sent_table3 = ev3["sent"]
            macd_table3 = ev3["macd"]
            inst_table3 = ev3["inst"]
            wyckoff_table3 = ev3["wyckoff"]
            relative_table3 = ev3["relative"]
            verdict_output3 = final_verdict(
                top10_assets=top20_assets3,
                ta_table=ta_combined3,
                fa_table=fa_table3,
                risk_table=risk_table3,
                sent_table=sent_table3,
                ew_table=ew_table3,
                macd_table=macd_table3,
                inst_table=inst_table3,
                wyckoff_table=wyckoff_table3,
                relative_table=relative_table3,
            )

    timer.stage("Pasadas adicionales")

    # Fase 6: Diagrama de arquitectura
    print_architecture_diagram()
    timer.summary()

    return 0


# ═══════════════════════════════════════════════════════════════════════════
# MODO MI CARTERA: analisis focalizado sobre tickers del usuario
# ═══════════════════════════════════════════════════════════════════════════

def run_portfolio(args: argparse.Namespace) -> int:
    """Ejecuta debate focalizado + evaluaciones completas sobre los tickers del usuario."""
    timer = _StageTimer()
    portfolio_tickers = [t.strip().upper() for t in args.portfolio.split(",") if t.strip()]
    if not portfolio_tickers:
        print("ERROR: --portfolio requiere al menos un ticker.", file=sys.stderr)
        return 1

    client = OllamaClient(args.host)
    available_models = client.refresh_models()
    model = client.resolve_model(args.model)

    analyst_roles = [
        "Moderador Consenso", "Moderador Top 20",
        "Analista Tecnico", "Analista Fundamental",
        "Gestor Riesgos", "Analista Sentimiento", "Analista Elliott", "Veredicto Final",
    ]
    model_map = assign_models_to_agents(available_models, AGENTS, analyst_roles)

    n = len(portfolio_tickers)
    print("=== Analisis Mi Cartera ===")
    print(f"Modelo principal: {model}")
    print(f"Activos en cartera: {n} -> {', '.join(portfolio_tickers)}")
    print(f"Duracion debate: {args.seconds} segundos")
    print("=" * 60)

    # Fase 0: Cargar datos de mercado y carteras de gurus en paralelo
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_market = pool.submit(load_all_market_data, 72, True)
        fut_gurus = pool.submit(fetch_all_guru_holdings)
        briefings = fut_market.result()
        all_guru_holdings = fut_gurus.result()

    market_general = briefings.get("general", "")
    raw_prices: dict[str, dict[str, Any]] = briefings.get("raw_prices", {})
    raw_fundamentals: dict[str, dict[str, Any]] = briefings.get("raw_fundamentals", {})

    if market_general:
        print("\n[Datos de mercado cargados correctamente]")
    else:
        print("\n[AVISO: Sin datos de mercado]")

    guru_conv = compute_guru_conviction(all_guru_holdings, TICKERS)
    timer.stage("Carga datos mercado + gurus")

    # Verificar que los tickers existen en datos de mercado
    valid_tickers = [t for t in portfolio_tickers if t in raw_prices]
    missing = [t for t in portfolio_tickers if t not in raw_prices]
    if missing:
        print(f"\n[AVISO: Tickers sin datos de mercado (ignorados): {', '.join(missing)}]", flush=True)
    if not valid_tickers:
        print("ERROR: Ningun ticker tiene datos de mercado disponibles.", file=sys.stderr)
        return 1

    # Fase 1: Debate focalizado sobre los activos de la cartera
    ticker_list_str = ", ".join(valid_tickers)
    portfolio_briefing = (
        f"{market_general}\n\n"
        f"IMPORTANTE: El usuario tiene la siguiente cartera de inversiones: {ticker_list_str}.\n"
        f"Analiza EXCLUSIVAMENTE estos {len(valid_tickers)} activos. "
        f"Debate sobre su situacion actual, perspectivas, riesgos y catalizadores para 2026-2027."
    )

    transcript: list[str] = []
    start = time.monotonic()
    deadline = start + max(5, args.seconds)

    turn_count = 0
    while time.monotonic() < deadline and turn_count < args.max_turns:
        for agent in AGENTS:
            if time.monotonic() >= deadline or turn_count >= args.max_turns:
                break
            agent_turn(
                client=client,
                agent=agent,
                model=model_map[agent.name],
                transcript=transcript,
                context_lines=args.context_lines,
                market_briefing=portfolio_briefing,
            )
            turn_count += 1

    elapsed = time.monotonic() - start
    print(f"\nDebate finalizado en {elapsed:.1f} segundos con {turn_count} turnos.")
    timer.stage(f"Debate ({turn_count} turnos)")

    # Fase 2: Consenso focalizado
    consensus_and_summary(
        client=client,
        model=model_map["Moderador Consenso"],
        transcript=transcript,
        context_lines=args.context_lines,
        market_briefing=portfolio_briefing,
    )
    timer.stage("Consenso")

    # Fase 3: Construir lista de activos con pesos iguales (cartera del usuario)
    peso_por_activo = round(100 / len(valid_tickers), 1)
    remainder = round(100 - peso_por_activo * len(valid_tickers), 1)
    portfolio_assets: list[dict[str, Any]] = []
    for idx, ticker in enumerate(valid_tickers):
        meta = TICKERS.get(ticker, {})
        peso = peso_por_activo + (remainder if idx == 0 else 0)
        portfolio_assets.append({
            "rank": idx + 1,
            "ticker": ticker,
            "nombre": meta.get("nombre", ticker),
            "tipo": raw_prices.get(ticker, {}).get("tipo", meta.get("tipo", "Accion")),
            "peso": peso,
            "tesis": "Activo en cartera del usuario",
        })

    # Mostrar la cartera como Top N
    print("\n" + "=" * 60, flush=True)
    print(f"MI CARTERA ({len(valid_tickers)} activos)", flush=True)
    print("=" * 60, flush=True)
    for asset in portfolio_assets:
        print(f"{asset['rank']}. {asset['ticker']} - {asset['nombre']} - {asset['tipo']} - {asset['peso']}% - {asset['tesis']}", flush=True)
    print("\n[Top 20 validado]", flush=True)

    # Fase 4: Evaluaciones independientes EN PARALELO (datos reales)
    print("\n[Ejecutando 8 evaluadores en paralelo...]", flush=True)
    ev = _run_evaluators(portfolio_assets, raw_prices, raw_fundamentals)
    ta_combined, ew_table = ev["ta"]
    fa_table = ev["fa"]
    risk_table = ev["risk"]
    sent_table = ev["sent"]
    macd_table = ev["macd"]
    inst_table = ev["inst"]
    wyckoff_table = ev["wyckoff"]
    relative_table = ev["relative"]
    timer.stage("Evaluadores (8 en paralelo)")

    # Fase 4b: Interpretacion Elliott Wave largo plazo (LLM, mejor modelo)
    elliott_llm_review(
        client=client,
        model=model_map["Analista Elliott"],
        ew_table=ew_table,
        top20_assets=portfolio_assets,
    )
    timer.stage("Elliott LLM largo plazo")

    # Fase 5: Veredicto final consolidado
    verdict_output = final_verdict(
        top10_assets=portfolio_assets,
        ta_table=ta_combined,
        fa_table=fa_table,
        risk_table=risk_table,
        sent_table=sent_table,
        ew_table=ew_table,
        macd_table=macd_table,
        inst_table=inst_table,
        wyckoff_table=wyckoff_table,
        relative_table=relative_table,
    )

    # Fase 5b: Comparativa con carteras de gurus
    guru_holdings_section(
        all_holdings=all_guru_holdings,
        tickers_dict=TICKERS,
        top20_tickers=valid_tickers,
    )
    timer.stage("Veredicto + gurus")

    print_architecture_diagram()
    timer.summary()
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.portfolio:
            return run_portfolio(args)
        return run_debate(args)
    except requests.HTTPError as exc:
        print(f"\nError HTTP con Ollama: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEjecucion interrumpida por usuario.", file=sys.stderr)
        return 130


def print_architecture_diagram() -> None:
    """Imprime un diagrama ASCII del flujo completo de agentes."""
    diagram = r"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    ARQUITECTURA DE AGENTES - FLUJO COMPLETO                    ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌──────────────────────────────────────────────────────────────────────────┐   ║
║  │                    FASE 1: DEBATE EN VIVO (streaming)                   │   ║
║  │                     Duracion: --seconds (def. 180s)                     │   ║
║  │                                                                        │   ║
║  │   ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐          │   ║
║  │   │   Warren     │  │   Peter     │  │     Stanley          │          │   ║
║  │   │   Buffett    │  │   Lynch     │  │   Druckenmiller      │          │   ║
║  │   │  (Valor)     │  │ (Crecim.)   │  │   (Macro/Timing)     │          │   ║
║  │   └──────┬───────┘  └──────┬──────┘  └──────────┬───────────┘          │   ║
║  │          │                 │                     │                      │   ║
║  │   ┌──────┴───────┐  ┌─────┴──────┐  ┌──────────┴───────────┐          │   ║
║  │   │    Ray       │  │   Cathie   │  │      Howard          │          │   ║
║  │   │   Dalio      │  │   Wood     │  │      Marks           │          │   ║
║  │   │ (All Weather)│  │(Disrupcion)│  │  (Credito/Ciclos)    │          │   ║
║  │   └──────┬───────┘  └─────┬──────┘  └──────────┬───────────┘          │   ║
║  │          │                │                     │                      │   ║
║  │          │         ┌──────┴──────┐              │                      │   ║
║  │          │         │     Jim     │              │                      │   ║
║  │          │         │   Rogers    │              │                      │   ║
║  │          │         │(Commodities)│              │                      │   ║
║  │          │         └──────┬──────┘              │                      │   ║
║  │          └────────────────┼──────────────────────┘                      │   ║
║  │                           │                                            │   ║
║  │                    ┌──────▼──────┐                                     │   ║
║  │                    │ Transcript  │                                     │   ║
║  │                    │  (debate)   │                                     │   ║
║  │                    └──────┬──────┘                                     │   ║
║  └───────────────────────────┼──────────────────────────────────────────────┘   ║
║                              │                                                 ║
║  ┌───────────────────────────▼──────────────────────────────────────────────┐   ║
║  │                  FASE 2: CONSENSO (Moderador)                           │   ║
║  │                  Salida: 10 lineas de consenso macro                    │   ║
║  └───────────────────────────┬──────────────────────────────────────────────┘   ║
║                              │                                                 ║
║  ┌───────────────────────────▼──────────────────────────────────────────────┐   ║
║  │                  FASE 3: TOP 20 INVERSIONES (Moderador)                 │   ║
║  │                  Salida: 20 activos con % de cartera (suma 100%)        │   ║
║  └───────────────────────────┬──────────────────────────────────────────────┘   ║
║                              │                                                 ║
║  ┌───────────────────────────▼──────────────────────────────────────────────┐   ║
║  │                  FASE 4: EVALUACIONES INDEPENDIENTES                    │   ║
║  │                                                                         │   ║
║  │   Tecnico | Elliott | Fundamental | Riesgo | Sentimiento | MACD |        │   ║
║  │   Institucional                                                      │   ║
║  │                                                                         │   ║
║  │   Cada evaluador puntua y asigna OK/NOK por activo usando datos reales. │   ║
║  └──────────────────────────────┬────────────────────────────────────────────┘   ║
║                                 │                                              ║
║  ┌──────────────────────────────▼────────────────────────────────────────────┐   ║
║  │                  FASE 5: VEREDICTO FINAL CONSOLIDADO                     │   ║
║  │                                                                          │   ║
║  │   Regla de decision por activo:                                          │   ║
║  │     >= 6 OK de 7 evaluadores  ──►  COMPRAR                              │   ║
║  │     >= 4 OK de 7 evaluadores  ──►  VIGILAR                              │   ║
║  │      < 4 OK de 7 evaluadores  ──►  EVITAR                               │   ║
║  │                                                                          │   ║
║  │   Salida: Tabla final + 3 lineas de recomendacion                        │   ║
║  └──────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                ║
║  ROLES LLM: 7 debate + 2 moderadores | EVALUADORES: 7 reglas deterministas     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
