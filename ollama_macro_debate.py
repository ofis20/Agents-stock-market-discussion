#!/usr/bin/env python3
"""Debate macroeconomico entre 7 agentes con Ollama local y streaming en vivo."""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, List

import requests

from market_data import FUNDAMENTAL_TICKERS, TICKERS, load_all_market_data
from guru_holdings import (
    fetch_all_guru_holdings,
    compute_guru_conviction,
    guru_holdings_section,
)


DEFAULT_HOST = "http://127.0.0.1:11434"


@dataclass
class Agent:
    name: str
    system_prompt: str
    model: str = ""  # Si vacio, se asigna automaticamente


# Modelos generalistas preferidos para el debate (orden de preferencia).
# Se excluyen modelos de codigo (deepseek-coder, codellama) que no rinden bien en debates.
PREFERRED_MODELS = [
    "llama3.1:8b", "llama3.1", "gemma2:9b", "gemma2",
    "qwen2.5:7b", "qwen2.5", "mistral:latest", "mistral",
    "phi3.5:latest", "phi3.5", "phi3:latest", "phi3",
    "llama3:8b", "llama3", "llama2:7b", "llama2",
    "command-r:latest", "command-r",
]

# Modelos de codigo que no deben usarse para debatir
CODE_MODELS = {"deepseek-coder", "codellama", "starcoder", "codegemma", "codeqwen"}

# Lista global de modelos disponibles (se rellena en run_debate).
_AVAILABLE_MODELS: list[str] = []


def _is_code_model(name: str) -> bool:
    """Devuelve True si el modelo es de codigo (no apto para debate)."""
    base = name.split(":")[0].lower()
    return base in CODE_MODELS


def assign_models_to_agents(available_models: list[str], agents: list[Agent],
                            analyst_roles: list[str]) -> dict[str, str]:
    """Asigna modelos diversos a agentes y analistas.

    Devuelve un dict {rol: modelo} donde rol es el nombre del agente o rol de analista.
    Distribuye los modelos disponibles de forma round-robin para maximizar diversidad.
    """
    # Filtrar solo modelos generalistas
    general_models = [m for m in available_models if not _is_code_model(m)]

    if not general_models:
        # Si no hay generalistas, usar todos los disponibles
        general_models = available_models[:]

    if not general_models:
        raise RuntimeError("No hay modelos instalados en Ollama.")

    # Ordenar por preferencia: los que estan en PREFERRED_MODELS primero
    def _priority(m: str) -> int:
        for i, pref in enumerate(PREFERRED_MODELS):
            if m == pref or m.startswith(pref.split(":")[0]):
                return i
        return len(PREFERRED_MODELS)

    general_models.sort(key=_priority)

    # Todos los roles que necesitan modelo
    all_roles = [a.name for a in agents] + analyst_roles
    assignment: dict[str, str] = {}

    # Primero respetar modelos asignados manualmente en Agent.model
    available_for_auto: list[str] = general_models[:]
    for agent in agents:
        if agent.model and agent.model in available_models:
            assignment[agent.name] = agent.model

    # Distribuir el resto round-robin
    idx = 0
    for role in all_roles:
        if role not in assignment:
            assignment[role] = available_for_auto[idx % len(available_for_auto)]
            idx += 1

    return assignment


AGENTS = [
    Agent(
        name="Warren Buffett",
        system_prompt=(
            "Eres Warren Buffett en marzo de 2026. Inversor orientado a valor de largo plazo. "
            "Priorizas calidad del negocio, ventajas competitivas duraderas, flujo de caja y disciplina de riesgo. "
            "IMPORTANTE: Tu analisis debe ser PROSPECTIVO - que empresas y activos van a hacerlo BIEN en 2026-2027, "
            "no cuales lo hicieron bien en el pasado. Los datos historicos son referencia, no prediccion. "
            "SIEMPRE nombras empresas o activos concretos y reales (acciones, ETFs, bonos, commodities, cripto, etc.) "
            "cuando propones inversiones. Habla en espanol claro, sin jerga excesiva, y de forma concisa."
        ),
    ),
    Agent(
        name="Peter Lynch",
        system_prompt=(
            "Eres Peter Lynch en marzo de 2026. Gestor enfocado en crecimiento razonable y negocios entendibles. "
            "Buscas oportunidades en tendencias FUTURAS de la economia para 2026-2027, con enfoque practico. "
            "IMPORTANTE: No recomiendes activos solo porque subieron en el pasado. Analiza que sectores y empresas "
            "tienen catalizadores concretos para crecer en los proximos 12-24 meses. "
            "SIEMPRE nombras empresas o activos concretos y reales (acciones, ETFs, sectores, commodities, cripto, etc.). "
            "Habla en espanol claro, directo y con ejemplos concretos."
        ),
    ),
    Agent(
        name="Stanley Druckenmiller",
        system_prompt=(
            "Eres Stanley Druckenmiller en marzo de 2026. Macro trader enfocado en regimenes de liquidez, "
            "tipos de interes, divisas y riesgo global. Das peso al timing y a los cambios de politica monetaria/fiscal. "
            "IMPORTANTE: Tu analisis debe ser PROSPECTIVO para 2026-2027. Identifica cambios de ciclo, "
            "rotaciones sectoriales y oportunidades macro que estan por venir, no las que ya pasaron. "
            "SIEMPRE nombras activos concretos y reales (divisas, bonos, commodities, acciones, ETFs, cripto, etc.). "
            "Habla en espanol claro, breve y orientado a escenarios futuros."
        ),
    ),
    Agent(
        name="Ray Dalio",
        system_prompt=(
            "Eres Ray Dalio en marzo de 2026. Fundador de Bridgewater, el mayor hedge fund del mundo. "
            "Tu marco es el All Weather: equilibrar riesgo entre crecimiento, inflacion, deflacion y recesion. "
            "IMPORTANTE: Analiza en que fase del ciclo economico estamos y que activos se beneficiaran "
            "en 2026-2027 segun tu modelo de ciclos. No te bases solo en rendimientos pasados. "
            "SIEMPRE nombras activos concretos y reales (bonos TIPS, oro, acciones chinas, ETFs, etc.). "
            "Habla en espanol claro, con vision de ciclos largos y perspectiva 2026-2027."
        ),
    ),
    Agent(
        name="Cathie Wood",
        system_prompt=(
            "Eres Cathie Wood en marzo de 2026. CEO de ARK Invest, enfocada en innovacion disruptiva. "
            "Tu tesis central: la convergencia de IA, robotica, blockchain, secuenciacion genomica y energia "
            "esta creando la mayor oportunidad de inversion en decadas. "
            "IMPORTANTE: Identifica empresas disruptivas con catalizadores concretos para 2026-2027 "
            "(lanzamientos de productos, adopcion masiva, regulacion favorable). No solo mires rendimiento pasado. "
            "SIEMPRE nombras empresas concretas y reales (Tesla, Coinbase, CRISPR, Palantir, etc.). "
            "Habla en espanol con entusiasmo medido y datos concretos sobre catalizadores futuros."
        ),
    ),
    Agent(
        name="Howard Marks",
        system_prompt=(
            "Eres Howard Marks en marzo de 2026. Co-fundador de Oaktree Capital, experto en credito y ciclos de mercado. "
            "Tu filosofia: comprar cuando hay sangre, vender cuando hay euforia. "
            "IMPORTANTE: Analiza donde estan las oportunidades INFRAVALORADAS para 2026-2027. "
            "Busca sectores castigados con potencial de recuperacion, no los que ya subieron. "
            "SIEMPRE nombras activos concretos y reales (high yield, bonos corporativos, deuda emergente, etc.). "
            "Habla en espanol sobrio, prudente y con foco en margen de seguridad y perspectiva futura."
        ),
    ),
    Agent(
        name="Jim Rogers",
        system_prompt=(
            "Eres Jim Rogers en marzo de 2026. Legendario inversor en commodities y mercados frontera. "
            "Tu vision: los ciclos de materias primas son largos y hay que anticipar el proximo movimiento. "
            "IMPORTANTE: Identifica que commodities y mercados emergentes tienen catalizadores concretos "
            "para 2026-2027 (deficit de oferta, cambios regulatorios, demografia). No solo mires retornos pasados. "
            "SIEMPRE nombras activos concretos y reales (plata, soja, petroleo, acciones vietnamitas, etc.). "
            "Habla en espanol directo, contracorriente y con perspectiva global sobre los proximos 12-24 meses."
        ),
    ),
]


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
    return parser.parse_args()


def check_ollama(host: str, timeout: float = 4.0) -> list[str]:
    try:
        resp = requests.get(f"{host}/api/tags", timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
        return models
    except Exception as exc:
        raise RuntimeError(
            "No se pudo conectar con Ollama local. "
            "Asegurate de tener el servicio activo (ej: `ollama serve`)."
        ) from exc


def resolve_model(requested_model: str, available_models: list[str]) -> str:
    if not available_models:
        raise RuntimeError("No hay modelos instalados en Ollama. Ejecuta, por ejemplo: `ollama pull qwen2.5:7b`.")

    if requested_model in available_models:
        return requested_model

    fallback = available_models[0]
    print(
        f"Aviso: el modelo '{requested_model}' no esta instalado. Se usara '{fallback}'.",
        file=sys.stderr,
    )
    return fallback


def _stream_chat_single(host: str, model: str, messages: list[dict[str, str]], num_predict: int = 2048, silent: bool = False) -> str:
    """Intenta una sola llamada a /api/chat (sin retry)."""
    chat_url = f"{host}/api/chat"
    chat_payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.5,
            "num_predict": num_predict,
        },
    }

    full_text = ""
    chat_resp = requests.post(chat_url, json=chat_payload, stream=True, timeout=120)

    # Compatibilidad con versiones antiguas de Ollama que no incluyen /api/chat.
    if chat_resp.status_code == 404:
        chat_resp.close()
        return stream_generate(host=host, model=model, messages=messages, num_predict=num_predict, silent=silent)

    with chat_resp as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            chunk = event.get("message", {}).get("content", "")
            if chunk:
                if not silent:
                    print(chunk, end="", flush=True)
                full_text += chunk

            if event.get("done"):
                break

    if not silent:
        print(flush=True)
    return full_text.strip()


def stream_chat(host: str, model: str, messages: list[dict[str, str]], num_predict: int = 2048, silent: bool = False) -> str:
    """Llama a /api/chat con retry automatico usando modelos alternativos si falla."""
    try:
        return _stream_chat_single(host, model, messages, num_predict, silent)
    except (requests.RequestException, OSError) as first_err:
        # Construir lista de modelos alternativos
        alternates = [m for m in _AVAILABLE_MODELS if m != model and not _is_code_model(m)]
        if not alternates:
            raise
        for alt_model in alternates:
            print(f"\n[RETRY: {model} fallo ({first_err}). Probando {alt_model}...]", flush=True)
            try:
                return _stream_chat_single(host, alt_model, messages, num_predict, silent)
            except (requests.RequestException, OSError):
                continue
        # Todos fallaron: relanzar el error original
        raise first_err


def stream_generate(host: str, model: str, messages: list[dict[str, str]], num_predict: int = 2048, silent: bool = False) -> str:
    """Fallback a /api/generate para servidores Ollama sin endpoint /api/chat."""
    generate_url = f"{host}/api/generate"

    prompt_lines = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "system":
            prompt_lines.append(f"[INSTRUCCIONES]\n{msg.get('content', '')}\n")
        elif role == "user":
            prompt_lines.append(f"[USUARIO]\n{msg.get('content', '')}\n")
        else:
            prompt_lines.append(f"[ASISTENTE]\n{msg.get('content', '')}\n")
    prompt_lines.append("[ASISTENTE]\n")
    prompt = "\n".join(prompt_lines)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.5,
            "num_predict": num_predict,
        },
    }

    full_text = ""
    with requests.post(generate_url, json=payload, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            chunk = event.get("response", "")
            if chunk:
                if not silent:
                    print(chunk, end="", flush=True)
                full_text += chunk

            if event.get("done"):
                break

    if not silent:
        print(flush=True)
    return full_text.strip()


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
    agent: Agent,
    host: str,
    model: str,
    transcript: list[str],
    context_lines: int,
    market_briefing: str = "",
) -> str:
    context = recent_context(transcript, context_lines)
    market_block = ""
    if market_briefing:
        market_block = f"\n\nDATOS REALES DE MERCADO (usa estos datos, NO inventes cifras):\n{market_briefing}\n"
    user_prompt = textwrap.dedent(
        f"""
        Debate de macroeconomia global e inversiones en marzo de 2026 entre siete inversores legendarios.
        ENFOQUE: Que activos van a hacerlo BIEN en 2026 y 2027. NO recomiendes activos solo porque subieron en el pasado.
        {market_block}
        Reglas para este turno:
        - Responde en 3-5 lineas maximo.
        - Mantente en tu personaje y no inventes ni cites nombres de otras personas.
        - Debes reaccionar al ultimo contexto del debate.
        - OBLIGATORIO: menciona al menos 1 empresa, ETF, commodity, divisa, bono o activo concreto y real.
        - PROSPECTIVO: explica POR QUE ese activo va a hacerlo bien en 2026-2027 (catalizadores futuros, no rendimiento pasado).
        - USA LOS DATOS REALES proporcionados arriba como referencia del estado actual.
        - Incluye 1 tesis concreta sobre el FUTURO de ese activo y 1 riesgo principal.
        - Cierra con una propuesta breve para acercar consenso sobre las mejores inversiones para 2026-2027.

        Contexto reciente:
        {context}
        """
    ).strip()

    print(f"\n[{agent.name}] (modelo: {model})", flush=True)
    reply = stream_chat(
        host,
        model,
        messages=[
            {"role": "system", "content": agent.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        num_predict=420,
    )

    clean_reply = _sanitize_agent_reply(reply)
    line = f"{agent.name}: {clean_reply}"
    transcript.append(line)
    return clean_reply


def consensus_and_summary(host: str, model: str, transcript: list[str], context_lines: int, market_briefing: str = "") -> str:
    context = recent_context(transcript, max_lines=max(context_lines * 4, 60))
    system = (
        "Eres moderador neutral. Debes extraer consenso final entre los siete participantes. "
        "Tu salida final debe ser exactamente 10 lineas en espanol, sin encabezados ni numeracion. "
        "USA LOS DATOS REALES DE MERCADO proporcionados para fundamentar el consenso, NO inventes cifras."
    )
    data_block = f"\n\nDATOS REALES DE MERCADO:\n{market_briefing}\n" if market_briefing else ""
    user = textwrap.dedent(
        f"""
        A partir de este debate, redacta consenso final.{data_block}

        Requisitos estrictos:
        - Exactamente 10 lineas.
        - Cada linea debe aportar una idea util sobre macroeconomia e inversion.
        - Debe reflejar puntos en comun entre los 7 enfoques: valor, crecimiento, macro, paridad de riesgo, disrupcion, credito y commodities.
        - Incluye al menos una advertencia de riesgo y una accion prudente.
        - Basa tus conclusiones en los DATOS REALES proporcionados.

        Debate:
        {context}
        """
    ).strip()

    print("\n[Moderador - consenso final 10 lineas]", flush=True)
    text = stream_chat(host, model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])

    # Asegura exactamente 10 lineas limpias para cumplir la especificacion.
    base_fillers = [
        "Mantener disciplina de riesgo y revisar datos macro de forma continua.",
        "Priorizar calidad de activos y evitar concentraciones excesivas por region o sector.",
        "Ajustar posicionamiento segun liquidez global y trayectoria esperada de tipos.",
        "Combinar horizonte de largo plazo con control tactico de volatilidad en el corto plazo.",
        "Usar cobertura selectiva cuando aumente la incertidumbre geopolItica o inflacionaria.",
    ]

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
        lines.append(base_fillers[fill_i % len(base_fillers)])
        fill_i += 1

    final_text = "\n".join(lines)
    print("\n[Resumen validado - 10 lineas]", flush=True)
    print(final_text, flush=True)
    return final_text


def _to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  ANALISIS DE ONDAS DE ELLIOTT (determinista)
# ═══════════════════════════════════════════════════════════════════════════

def _find_zigzag_pivots(closes: list[float], pct_threshold: float = 5.0) -> list[tuple[int, float, str]]:
    """Detecta pivots (High/Low) usando filtro de zigzag con umbral porcentual."""
    if len(closes) < 10:
        return []

    pivots: list[tuple[int, float, str]] = []  # (idx, precio, "H"/"L")
    last_pivot_type = ""
    last_pivot_price = closes[0]
    last_pivot_idx = 0

    for i in range(1, len(closes)):
        change_from_last = ((closes[i] - last_pivot_price) / last_pivot_price) * 100

        if change_from_last >= pct_threshold:
            if last_pivot_type != "L":
                pivots.append((last_pivot_idx, last_pivot_price, "L"))
            last_pivot_type = "H"
            last_pivot_price = closes[i]
            last_pivot_idx = i
        elif change_from_last <= -pct_threshold:
            if last_pivot_type != "H":
                pivots.append((last_pivot_idx, last_pivot_price, "H"))
            last_pivot_type = "L"
            last_pivot_price = closes[i]
            last_pivot_idx = i
        else:
            if last_pivot_type == "H" and closes[i] > last_pivot_price:
                last_pivot_price = closes[i]
                last_pivot_idx = i
            elif last_pivot_type == "L" and closes[i] < last_pivot_price:
                last_pivot_price = closes[i]
                last_pivot_idx = i

    pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type or "H"))
    return pivots


def _classify_elliott_wave(closes: list[float]) -> dict[str, Any]:
    """Clasifica la fase de Elliott Wave actual, da prevision y calcula targets de precio.

    Retorna dict con claves:
    - onda, fase, prevision (str)
    - precio_actual (float)
    - target_1m, target_3m, target_6m (float) — precios objetivo
    - rent_1m, rent_3m, rent_6m (float) — rentabilidad esperada en %
    """
    nd = {
        "onda": "N/D", "fase": "Datos insuficientes",
        "prevision": "Sin datos suficientes para analisis.",
        "precio_actual": 0.0,
        "target_1m": 0.0, "target_3m": 0.0, "target_6m": 0.0,
        "rent_1m": 0.0, "rent_3m": 0.0, "rent_6m": 0.0,
    }
    if len(closes) < 60:
        return nd

    # Usar ultimos ~12 meses
    series = closes[-252:] if len(closes) >= 252 else closes
    current = series[-1]
    pivots = _find_zigzag_pivots(series, pct_threshold=5.0)

    # Helpers para targets
    highs_all = [p for p in pivots if p[2] == "H"]
    lows_all = [p for p in pivots if p[2] == "L"]
    last_high = max(p[1] for p in highs_all) if highs_all else current
    last_low = min(p[1] for p in lows_all) if lows_all else current
    swing_range = last_high - last_low if last_high > last_low else current * 0.10

    # Fibonacci ratios
    fib_038 = swing_range * 0.382
    fib_050 = swing_range * 0.500
    fib_062 = swing_range * 0.618
    fib_100 = swing_range
    fib_162 = swing_range * 1.618

    def _build(onda: str, fase: str, prevision: str,
               pct_1m: float, pct_3m: float, pct_6m: float) -> dict[str, Any]:
        return {
            "onda": onda, "fase": fase, "prevision": prevision,
            "precio_actual": round(current, 2),
            "target_1m": round(current * (1 + pct_1m / 100), 2),
            "target_3m": round(current * (1 + pct_3m / 100), 2),
            "target_6m": round(current * (1 + pct_6m / 100), 2),
            "rent_1m": round(pct_1m, 1),
            "rent_3m": round(pct_3m, 1),
            "rent_6m": round(pct_6m, 1),
        }

    # Calcular ATR% para escalar targets segun volatilidad del activo
    diffs = [abs(series[i] - series[i - 1]) for i in range(max(1, len(series) - 20), len(series))]
    atr_pct = (sum(diffs) / len(diffs) / current * 100) if diffs and current else 1.5

    if len(pivots) < 3:
        total_change = ((current - series[0]) / series[0]) * 100
        if total_change > 15:
            # Onda 3 fuerte — proyectar extension fib 1.618
            up_3m = min(fib_162 / current * 100, 35)
            return _build("3↑", "Impulso alcista fuerte",
                          "Posible extension de onda 3 o transicion a onda 4 correctiva.",
                          up_3m * 0.3, up_3m * 0.65, up_3m)
        elif total_change > 0:
            up_3m = min(fib_100 / current * 100, 20)
            return _build("1↑", "Inicio de impulso alcista",
                          "Potencial continuacion alcista si confirma estructura.",
                          up_3m * 0.25, up_3m * 0.55, up_3m)
        elif total_change > -15:
            dn_1m = -fib_038 / current * 100
            return _build("A↓", "Inicio de correccion",
                          "Posible rebote en onda B antes de completar correccion.",
                          dn_1m, dn_1m * 0.5, fib_038 / current * 50)
        else:
            up_6m = fib_062 / current * 100
            return _build("C↓", "Correccion profunda",
                          "Cerca de suelo si completa onda C. Oportunidad de entrada.",
                          -atr_pct * 2, up_6m * 0.3, up_6m)

    # Analizar patron de pivots
    highs = highs_all
    lows = lows_all
    higher_highs = len(highs) >= 2 and highs[-1][1] > highs[0][1]
    higher_lows = len(lows) >= 2 and lows[-1][1] > lows[0][1]
    lower_highs = len(highs) >= 2 and highs[-1][1] < highs[0][1]
    lower_lows = len(lows) >= 2 and lows[-1][1] < lows[0][1]

    last_pivot = pivots[-1]
    num_swings = len(pivots)
    pct_from_high = ((current - last_high) / last_high) * 100

    if higher_highs and higher_lows:
        if num_swings <= 3:
            if last_pivot[2] == "H":
                # Onda 1 completada → esperar retroceso fib 38-50%
                dn = -fib_038 / current * 100
                return _build("1↑", "Impulso alcista inicial",
                              "Esperar correccion de onda 2 para entrada. Alcista medio plazo.",
                              dn * 0.5, dn, fib_050 / current * 100)
            else:
                # Onda 2 en curso → comprar oportunidad, onda 3 la mas fuerte
                up_6m = fib_162 / current * 100
                return _build("2↓", "Correccion dentro de impulso alcista",
                              "Oportunidad de compra si respeta soporte. Onda 3 seria la mas fuerte.",
                              -atr_pct, up_6m * 0.35, min(up_6m, 40))
        elif num_swings <= 5:
            if last_pivot[2] == "H":
                # Onda 3 en curso — la mas potente
                up_6m = min(fib_162 / current * 100, 40)
                return _build("3↑", "Impulso alcista principal",
                              "Onda mas fuerte en curso. Mantener posiciones, objetivo al alza.",
                              up_6m * 0.25, up_6m * 0.6, up_6m)
            else:
                # Onda 4 — ultima oportunidad antes de onda 5
                up_6m = fib_100 / current * 100
                return _build("4↓", "Correccion intermedia en tendencia alcista",
                              "Correccion sana. Ultima oportunidad de compra antes de onda 5.",
                              -atr_pct, up_6m * 0.4, min(up_6m, 30))
        else:
            if last_pivot[2] == "H":
                # Onda 5 — cercano a techo
                dn_6m = -fib_038 / current * 100
                return _build("5↑", "Ultimo impulso alcista",
                              "Cercano a techo de ciclo. Considerar reducir posiciones gradualmente.",
                              atr_pct * 0.5, dn_6m * 0.3, dn_6m)
            else:
                # A↓ — inicio de ABC
                dn_3m = -fib_050 / current * 100
                return _build("A↓", "Inicio de correccion tras impulso completo",
                              "Correccion ABC en curso. Rebote temporal posible en onda B.",
                              dn_3m * 0.5, dn_3m, -fib_038 / current * 100)

    elif lower_highs and lower_lows:
        if pct_from_high < -25:
            up_6m = fib_062 / current * 100
            return _build("C↓", "Correccion profunda bajista",
                          "Posible suelo proximo. Zona de acumulacion para 2026-2027.",
                          -atr_pct * 2, up_6m * 0.25, up_6m)
        elif num_swings <= 3:
            dn_3m = -fib_050 / current * 100
            return _build("A↓", "Primera onda correctiva bajista",
                          "Esperar rebote en onda B. No comprar aun, riesgo de mas caida.",
                          dn_3m * 0.4, dn_3m, -fib_062 / current * 100)
        else:
            up_6m = fib_050 / current * 100
            return _build("B↑/C↓", "Correccion avanzada",
                          "En fase final correctiva. Buscar senales de agotamiento para entrada.",
                          -atr_pct, -fib_038 / current * 50, up_6m)

    else:
        if pct_from_high < -10:
            up_6m = fib_062 / current * 100
            return _build("4↓", "Consolidacion/correccion lateral",
                          "Posible acumulacion. Ruptura al alza activaria onda 5 alcista.",
                          -atr_pct, up_6m * 0.35, min(up_6m, 25))
        else:
            dn_6m = -fib_038 / current * 100
            return _build("B↑", "Rebote dentro de correccion",
                          "Rebote tecnico. Confirmar si rompe maximos para cambio de tendencia.",
                          atr_pct, -atr_pct * 2, dn_6m)


def _normalize_weights_100(scores: list[float]) -> list[int]:
    safe = [max(0.01, s) for s in scores]
    total = sum(safe)
    raw = [(s / total) * 100.0 for s in safe]
    base = [int(v) for v in raw]
    missing = 100 - sum(base)
    remainders = sorted([(raw[i] - base[i], i) for i in range(len(raw))], reverse=True)
    for _, idx in remainders[:max(0, missing)]:
        base[idx] += 1
    return base


# Tipos de activo permitidos en la cartera (excluye indices, divisas y macro)
_ALLOWED_TYPES = {"Accion", "ETF", "Commodity", "Cripto"}


def _top20_fallback_from_data(prices: dict[str, dict[str, Any]], fundamentals: dict[str, dict[str, Any]], exclude_tickers: set[str] | None = None, guru_conviction: dict[str, float] | None = None) -> list[dict[str, Any]]:
    # Calcular sector momentum para bonus sectorial (top 3 sectores)
    sector_momentum: dict[str, list[float]] = {}
    for _tk, _f in fundamentals.items():
        sec = _f.get("sector", "N/D")
        d = prices.get(_tk, {})
        mom = _to_float(d.get("momentum_score"))
        if sec != "N/D" and mom is not None:
            sector_momentum.setdefault(sec, []).append(mom)
    sector_avg = {sec: sum(vals) / len(vals) for sec, vals in sector_momentum.items() if vals}
    top_sectors = sorted(sector_avg, key=sector_avg.get, reverse=True)[:3] if sector_avg else []

    rows: list[tuple[str, float, str, str, float]] = []  # (ticker, score, thesis, tipo, vol)
    for ticker, meta in TICKERS.items():
        tipo = meta.get("tipo", "")
        if tipo not in _ALLOWED_TYPES:
            continue
        if exclude_tickers and ticker in exclude_tickers:
            continue
        d = prices.get(ticker)
        if not d:
            continue

        ret_12m = _to_float(d.get("ret_12m")) or 0.0
        ret_3m = _to_float(d.get("ret_3m")) or 0.0
        rsi = _to_float(d.get("rsi14"))
        sma50 = _to_float(d.get("sma50"))
        sma200 = _to_float(d.get("sma200"))
        price = _to_float(d.get("precio"))
        vol = _to_float(d.get("vol_20d")) or 60.0
        close_hist = d.get("close_hist", [])
        sharpe = _to_float(d.get("sharpe_6m"))
        rs_sp500 = _to_float(d.get("rs_vs_sp500"))
        momentum = _to_float(d.get("momentum_score"))
        avg_vol_20d = _to_float(d.get("avg_vol_20d"))
        macd_hist = _to_float(d.get("macd_hist"))

        # FILTRO DE LIQUIDEZ: descartar activos con volumen medio diario < 100k
        if avg_vol_20d is not None and avg_vol_20d < 100_000 and tipo == "Accion":
            continue

        # Tendencia tecnica (0-6)
        trend_bonus = 0.0
        if price and sma50 and sma200:
            if price > sma50 > sma200:
                trend_bonus = 6.0
            elif price > sma200:
                trend_bonus = 3.0
            else:
                trend_bonus = -2.0

        # RSI (0-3)
        rsi_bonus = 0.0
        if rsi is not None:
            if 40 <= rsi <= 68:
                rsi_bonus = 3.0
            elif rsi > 78:
                rsi_bonus = -3.0
            elif rsi < 30:
                rsi_bonus = 1.0

        # Elliott Wave
        ew_bonus = 0.0
        ew_onda = "N/D"
        if close_hist:
            ew = _classify_elliott_wave(close_hist)
            ew_onda = ew["onda"]
            if ew_onda in ("2↓", "4↓"):
                ew_bonus = 6.0
            elif ew_onda == "C↓":
                ew_bonus = 5.0
            elif ew_onda == "3↑":
                ew_bonus = 3.0
            elif ew_onda == "1↑":
                ew_bonus = 4.0
            elif ew_onda == "5↑":
                ew_bonus = -4.0
            elif ew_onda in ("A↓", "B↑"):
                ew_bonus = -1.0

        # Fundamentales
        f = fundamentals.get(ticker, {})
        per = _to_float(f.get("per"))
        forward_pe = _to_float(f.get("forward_pe"))
        peg = _to_float(f.get("peg"))
        roe = _to_float(f.get("roe"))
        growth = _to_float(f.get("crec_ingresos"))
        sector = f.get("sector", "N/D")

        fundamental_bonus = 0.0
        if roe is not None and roe >= 0.10:
            fundamental_bonus += 3.0
        if growth is not None and growth > 0:
            fundamental_bonus += 2.0
        eff_pe = forward_pe if forward_pe else per
        if eff_pe is not None and eff_pe <= 30:
            fundamental_bonus += 2.0
        # PEG bonus: <1 excelente, <1.5 bueno
        if peg is not None:
            if peg <= 1.0:
                fundamental_bonus += 4.0
            elif peg <= 1.5:
                fundamental_bonus += 2.0

        # FUERZA RELATIVA vs S&P 500 (0-5)
        rs_bonus = 0.0
        if rs_sp500 is not None:
            if rs_sp500 > 15:
                rs_bonus = 5.0
            elif rs_sp500 > 5:
                rs_bonus = 3.0
            elif rs_sp500 > 0:
                rs_bonus = 1.5

        # MOMENTUM COMPUESTO (0-5)
        mom_bonus = 0.0
        if momentum is not None:
            if momentum > 20:
                mom_bonus = 5.0
            elif momentum > 10:
                mom_bonus = 3.5
            elif momentum > 0:
                mom_bonus = 2.0
            elif momentum > -5:
                mom_bonus = 0.5

        # SHARPE RATIO (0-5)
        sharpe_bonus = 0.0
        if sharpe is not None:
            if sharpe >= 2.0:
                sharpe_bonus = 5.0
            elif sharpe >= 1.0:
                sharpe_bonus = 3.5
            elif sharpe >= 0.5:
                sharpe_bonus = 2.0
            elif sharpe >= 0:
                sharpe_bonus = 0.5

        # MACD confirmation (0-3)
        macd_bonus = 0.0
        if macd_hist is not None:
            if macd_hist > 0:
                macd_bonus = 3.0
            elif macd_hist > -0.5:
                macd_bonus = 1.0

        # SECTOR MOMENTUM (0-3)
        sector_bonus = 0.0
        if sector in top_sectors:
            sector_bonus = 3.0

        # VOLUMEN CONFIRMATION (0-2)
        vol_confirm = 0.0
        vol_ratio = _to_float(d.get("vol_ratio"))
        if vol_ratio is not None and vol_ratio >= 1.2:
            vol_confirm = 2.0

        # GURU CONVICTION (0-15): num gurus + % cartera + elite bonus
        guru_bonus = 0.0
        if guru_conviction:
            guru_bonus = guru_conviction.get(ticker, 0.0)

        # Score total: nuevo sistema multi-factor
        score = (
            trend_bonus + rsi_bonus + ew_bonus + fundamental_bonus +
            rs_bonus + mom_bonus + sharpe_bonus + macd_bonus +
            sector_bonus + vol_confirm + guru_bonus - (0.03 * vol)
        )
        guru_tag = f", Guru {guru_bonus:.0f}" if guru_bonus > 0 else ""
        thesis = f"Elliott {ew_onda}, Mom {_fmt_num(momentum, 1)}, Sharpe {_fmt_num(sharpe, 2)}, RS {_fmt_num(rs_sp500, 1)}{guru_tag}."
        rows.append((ticker, score, thesis, tipo, vol))

    rows.sort(key=lambda x: x[1], reverse=True)

    # Diversificar: maximo 12 acciones, al menos 2 ETF/commodity/cripto si hay
    selected: list[tuple[str, float, str, str, float]] = []
    count_by_type: dict[str, int] = {}
    for row in rows:
        t_type = row[3]
        ct = count_by_type.get(t_type, 0)
        if t_type == "Accion" and ct >= 12:
            continue
        if ct >= 5:
            continue
        selected.append(row)
        count_by_type[t_type] = ct + 1
        if len(selected) == 20:
            break

    # Si no alcanzan 20, rellenar con los mejores restantes
    if len(selected) < 20:
        used = {s[0] for s in selected}
        for row in rows:
            if row[0] not in used:
                selected.append(row)
                if len(selected) == 20:
                    break

    if len(selected) < 20:
        raise RuntimeError("No hay suficientes activos con datos para construir Top 20.")

    # PESOS POR INVERSE-VOLATILITY: menos peso a los mas volatiles
    inv_vols = [1.0 / max(s[4], 5.0) for s in selected]
    total_inv = sum(inv_vols)
    raw_weights = [(iv / total_inv) * 100 for iv in inv_vols]
    # Aplicar floor minimo del 2% y normalizar a 100
    floored = [max(2.0, w) for w in raw_weights]
    total_floored = sum(floored)
    weights_float = [(w / total_floored) * 100 for w in floored]
    weights = _normalize_weights_100(weights_float)

    out: list[dict[str, Any]] = []
    for i, (ticker, _, thesis, tipo, _vol) in enumerate(selected):
        nombre = TICKERS.get(ticker, {}).get("nombre", ticker)
        out.append(
            {
                "rank": i + 1,
                "ticker": ticker,
                "nombre": nombre,
                "tipo": tipo,
                "peso": weights[i],
                "tesis": thesis,
            }
        )
    return out


def _parse_top20_lines(text: str, prices: dict[str, dict[str, Any]], exclude_tickers: set[str] | None = None) -> list[dict[str, Any]]:
    lines = [ln.strip().replace("**", "") for ln in text.splitlines() if ln.strip()]
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ln in lines:
        m = re.match(r"^\s*(\d+)\.\s+([A-Z0-9\-\.=]+)\s*-\s*([A-Za-z]+)\s*-\s*(\d+)%\s*-\s*(.+)$", ln)
        if not m:
            continue
        ticker = m.group(2).upper()
        tipo = m.group(3).capitalize()
        peso = int(m.group(4))
        tesis = m.group(5).strip()
        if ticker in seen:
            continue
        if exclude_tickers and ticker in exclude_tickers:
            continue
        if ticker not in prices:
            continue
        # Validar que el tipo sea uno de los permitidos
        real_tipo = prices[ticker].get("tipo", tipo)
        if real_tipo not in _ALLOWED_TYPES:
            continue
        seen.add(ticker)
        nombre = TICKERS.get(ticker, {}).get("nombre", ticker)
        parsed.append(
            {
                "rank": int(m.group(1)),
                "ticker": ticker,
                "nombre": nombre,
                "tipo": real_tipo,
                "peso": peso,
                "tesis": tesis,
            }
        )
    parsed.sort(key=lambda x: x["rank"])
    return parsed


def _render_top20_lines(assets: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{i+1}. {a['ticker']} - {a.get('nombre', a['ticker'])} - {a['tipo']} - {a['peso']}% - {a['tesis']}"
        for i, a in enumerate(assets)
    )


def _render_md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_num(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "N/D"
    return f"{val:.{decimals}f}"


def top20_investments(
    host: str,
    model: str,
    transcript: list[str],
    context_lines: int,
    prices: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
    market_briefing: str = "",
    exclude_tickers: set[str] | None = None,
    guru_conviction: dict[str, float] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Genera top20 diversificado (acciones, ETFs, commodities, cripto). Fallback determinista si el LLM falla."""
    context = recent_context(transcript, max_lines=len(transcript))

    # Construir lista de candidatos por tipo
    candidates_by_type: dict[str, list[str]] = {}
    for t, meta in TICKERS.items():
        tipo = meta.get("tipo", "")
        if tipo in _ALLOWED_TYPES and t in prices:
            if exclude_tickers and t in exclude_tickers:
                continue
            candidates_by_type.setdefault(tipo, []).append(t)

    cand_lines = []
    for tipo, tickers_list in sorted(candidates_by_type.items()):
        cand_lines.append(f"  {tipo}: {', '.join(tickers_list[:60])}")
    candidates_str = "\n".join(cand_lines)

    system = (
        "Eres moderador financiero. Debes devolver EXACTAMENTE 20 lineas numeradas, sin texto extra. "
        "Formato estricto por linea: N. TICKER - Tipo - XX% - tesis corta y riesgo principal. "
        "Tipo debe ser Accion, ETF, Commodity o Cripto segun corresponda. "
        "IMPORTANTE: Selecciona activos que van a HACERLO BIEN en 2026-2027 (catalizadores futuros, "
        "no rendimiento pasado). Prioriza oportunidades prospectivas. "
        "La cartera debe estar DIVERSIFICADA: incluir al menos 2 ETFs o commodities. La suma debe ser 100%."
    )
    user = textwrap.dedent(
        f"""
        Debate:
        {context}

        Candidatos validos por tipo:
{candidates_str}

        Reglas:
        - Exactamente 20 lineas.
        - Cada linea: N. TICKER - Tipo - XX% - tesis PROSPECTIVA y riesgo.
        - Tipo valido: Accion, ETF, Commodity, Cripto.
        - No repitas ticker.
        - Suma total 100%.
        - Diversifica entre tipos de activo.
        - ENFOQUE PROSPECTIVO: elige por catalizadores futuros para 2026-2027, no por rendimiento pasado.
        """
    ).strip()

    print("\n" + "=" * 60, flush=True)
    print("TOP 20 INVERSIONES CONSENSUADAS (marzo 2026)", flush=True)
    print("=" * 60, flush=True)

    text = stream_chat(
        host,
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        num_predict=900,
        silent=True,
    )

    parsed = _parse_top20_lines(text, prices, exclude_tickers=exclude_tickers)
    valid_sum = sum(a["peso"] for a in parsed) == 100
    if len(parsed) != 20 or not valid_sum:
        print(f"\n  [AVISO: Top20 del modelo invalido ({len(parsed)} filas, suma={sum(a['peso'] for a in parsed)}). Usando fallback.]", flush=True)
        parsed = _top20_fallback_from_data(prices, fundamentals, exclude_tickers=exclude_tickers, guru_conviction=guru_conviction)

    final = _render_top20_lines(parsed)
    print("\n[Top 20 validado]", flush=True)
    print(final, flush=True)
    return final, parsed


def technical_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE ANALISIS TECNICO + ELLIOTT WAVES (determinista)", flush=True)
    print("=" * 60, flush=True)

    # Tabla clasica de indicadores tecnicos con score 0-100
    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        print(f"[Evaluando {i}/{total}: {t}]", flush=True)
        d = prices.get(t, {})
        p = _to_float(d.get("precio"))
        sma50 = _to_float(d.get("sma50"))
        sma200 = _to_float(d.get("sma200"))
        rsi = _to_float(d.get("rsi14"))
        vol = _to_float(d.get("vol_20d"))
        vol_ratio = _to_float(d.get("vol_ratio"))

        # Score numerico 0-100
        pts = 0.0
        # Tendencia (0-35 pts)
        if p and sma50 and sma200:
            if p > sma50 > sma200:
                pts += 35
            elif p > sma200:
                pts += 20
            elif p > sma50:
                pts += 10
        # RSI (0-30 pts): ideal 45-65
        if rsi is not None:
            if 45 <= rsi <= 65:
                pts += 30
            elif 40 <= rsi <= 70:
                pts += 22
            elif 30 <= rsi <= 78:
                pts += 12
            elif rsi < 30:
                pts += 8  # oversold puede ser oportunidad
        # Volatilidad (0-20 pts)
        if vol is not None:
            if vol <= 25:
                pts += 20
            elif vol <= 35:
                pts += 15
            elif vol <= 45:
                pts += 10
            elif vol <= 55:
                pts += 5
        # Confirmacion por volumen (0-15 pts)
        if vol_ratio is not None:
            if vol_ratio >= 1.2:
                pts += 15  # volumen activo confirma tendencia
            elif vol_ratio >= 0.8:
                pts += 10  # volumen normal
            else:
                pts += 3   # baja liquidez, senal debil

        score_num = min(100, max(0, round(pts)))
        verdict = "OK" if score_num >= 60 else "NOK"
        if verdict == "OK":
            ok_count += 1

        reason = f"Score {score_num}/100 | RSI {_fmt_num(rsi, 1)} | SMA50 {_fmt_num(sma50, 2)} | Vol {_fmt_num(vol, 1)}% | VolRatio {_fmt_num(vol_ratio, 2)}."
        nombre = asset.get("nombre", t)
        rows.append([str(i), t, nombre, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = _render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Razon tecnica"], rows)
    print("\n[Tabla tecnica validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)

    # Tabla de ondas de Elliott (con veredicto OK/NOK)
    print("\n--- ONDAS DE ELLIOTT ---", flush=True)
    ew_rows: list[list[str]] = []
    ew_ok_count = 0
    # Ondas favorables para compra (oportunidad o impulso sano)
    _FAVORABLE_WAVES = {"1↑", "2↓", "3↑", "4↓", "C↓"}
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        d = prices.get(t, {})
        close_hist = d.get("close_hist", [])
        ew = _classify_elliott_wave(close_hist)
        precio = _fmt_num(ew["precio_actual"], 2)
        t1m = f"{_fmt_num(ew['target_1m'], 2)} ({ew['rent_1m']:+.1f}%)"
        t3m = f"{_fmt_num(ew['target_3m'], 2)} ({ew['rent_3m']:+.1f}%)"
        t6m = f"{_fmt_num(ew['target_6m'], 2)} ({ew['rent_6m']:+.1f}%)"
        # OK si onda favorable Y rentabilidad 6m positiva
        ew_verdict = "OK" if (ew["onda"] in _FAVORABLE_WAVES and ew["rent_6m"] > 0) else "NOK"
        if ew_verdict == "OK":
            ew_ok_count += 1
        ew_rows.append([str(i), t, asset.get("nombre", t), ew["onda"], ew["fase"], precio, t1m, t3m, t6m, ew_verdict, ew["prevision"]])

    ew_table = _render_md_table(
        ["#", "Ticker", "Nombre", "Onda", "Fase", "Precio", "Target 1m", "Target 3m", "Target 6m", "Veredicto", "Prevision"],
        ew_rows,
    )
    print(ew_table, flush=True)
    print(f"\nResultado Elliott: {ew_ok_count} OK / {len(top10_assets) - ew_ok_count} NOK de {len(top10_assets)} activos.", flush=True)

    combined = table + "\n\n" + ew_table
    return combined, ew_table


def fundamental_analysis_review(top10_assets: list[dict[str, Any]], fundamentals: dict[str, dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE ANALISIS FUNDAMENTAL (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        print(f"[Evaluando {i}/{total}: {t}]", flush=True)
        tipo = asset.get("tipo", "Accion")
        f = fundamentals.get(t, {})
        per = _to_float(f.get("per"))
        forward_pe = _to_float(f.get("forward_pe"))
        peg = _to_float(f.get("peg"))
        roe = _to_float(f.get("roe"))
        de = _to_float(f.get("deuda_equity"))
        growth = _to_float(f.get("crec_ingresos"))
        earnings_g = _to_float(f.get("earnings_growth"))

        if tipo == "Accion":
            # Score 0-100 por fundamentales
            pts = 0.0
            # PER (0-25 pts): menor es mejor
            eff_pe = forward_pe if forward_pe else per
            if eff_pe is not None:
                if eff_pe <= 15:
                    pts += 25
                elif eff_pe <= 25:
                    pts += 20
                elif eff_pe <= 35:
                    pts += 12
                elif eff_pe <= 50:
                    pts += 5
            # PEG ratio (0-15 pts): <1 es excelente
            if peg is not None:
                if peg <= 1.0:
                    pts += 15
                elif peg <= 1.5:
                    pts += 10
                elif peg <= 2.5:
                    pts += 5
            # ROE (0-25 pts)
            if roe is not None:
                if roe >= 0.25:
                    pts += 25
                elif roe >= 0.15:
                    pts += 20
                elif roe >= 0.10:
                    pts += 12
                elif roe >= 0.05:
                    pts += 5
            # Deuda/Equity (0-20 pts)
            if de is not None:
                if de <= 50:
                    pts += 20
                elif de <= 100:
                    pts += 15
                elif de <= 180:
                    pts += 8
            # Crecimiento ingresos (0-15 pts)
            if growth is not None:
                if growth > 0.20:
                    pts += 15
                elif growth > 0.10:
                    pts += 10
                elif growth > 0:
                    pts += 5
            score_num = min(100, max(0, round(pts)))
            verdict = "OK" if score_num >= 55 else "NOK"
            peg_str = f"{peg:.2f}" if peg else "N/D"
            fpe_str = f"{forward_pe:.1f}" if forward_pe else "N/D"
            reason = f"Score {score_num}/100 | PER {per if per is not None else 'N/D'} | FwdPE {fpe_str} | PEG {peg_str} | ROE {roe if roe is not None else 'N/D'} | D/E {de if de is not None else 'N/D'}."
        else:
            # ETFs, Commodities, Cripto: score 0-100 por momentum + volatilidad
            d = prices.get(t, {})
            r12 = _to_float(d.get("ret_12m"))
            r3 = _to_float(d.get("ret_3m"))
            vol = _to_float(d.get("vol_20d"))
            sharpe = _to_float(d.get("sharpe_6m"))
            pts = 0.0
            # Retorno 12m (0-35 pts)
            if r12 is not None:
                if r12 > 20:
                    pts += 35
                elif r12 > 10:
                    pts += 25
                elif r12 > 0:
                    pts += 15
                elif r12 > -10:
                    pts += 5
            # Retorno 3m (0-25 pts)
            if r3 is not None:
                if r3 > 5:
                    pts += 25
                elif r3 > 0:
                    pts += 18
                elif r3 > -5:
                    pts += 8
            # Volatilidad (0-20 pts)
            if vol is not None:
                if vol <= 20:
                    pts += 20
                elif vol <= 35:
                    pts += 15
                elif vol <= 50:
                    pts += 8
            # Sharpe (0-20 pts)
            if sharpe is not None:
                if sharpe >= 2.0:
                    pts += 20
                elif sharpe >= 1.0:
                    pts += 15
                elif sharpe >= 0.5:
                    pts += 8
                elif sharpe >= 0:
                    pts += 3
            score_num = min(100, max(0, round(pts)))
            verdict = "OK" if score_num >= 55 else "NOK"
            reason = f"Score {score_num}/100 | Ret12m {r12 if r12 is not None else 'N/D'}% | Ret3m {r3 if r3 is not None else 'N/D'}% | Sharpe {_fmt_num(sharpe, 2)}."

        if verdict == "OK":
            ok_count += 1
        nombre = asset.get("nombre", t)
        rows.append([str(i), t, nombre, tipo, f"{asset['peso']}%", verdict, reason])

    table = _render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Razon fundamental"], rows)
    print("\n[Tabla fundamental validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def risk_management_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE GESTION DE RIESGOS (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        print(f"[Evaluando {i}/{total}: {t}]", flush=True)
        d = prices.get(t, {})
        vol = _to_float(d.get("vol_20d"))
        dd = _to_float(d.get("max_dd_12m"))
        dist = _to_float(d.get("dist_high_52w"))
        sharpe = _to_float(d.get("sharpe_6m"))

        # Score 0-100
        pts = 0.0
        # Volatilidad (0-30 pts)
        if vol is not None:
            if vol <= 20:
                pts += 30
            elif vol <= 30:
                pts += 25
            elif vol <= 40:
                pts += 18
            elif vol <= 50:
                pts += 8
        # Max Drawdown (0-30 pts)
        if dd is not None:
            if dd >= -15:
                pts += 30
            elif dd >= -25:
                pts += 22
            elif dd >= -35:
                pts += 15
            elif dd >= -45:
                pts += 8
        # Distancia a max 52w (0-20 pts)
        if dist is not None:
            if dist >= -10:
                pts += 20
            elif dist >= -20:
                pts += 15
            elif dist >= -30:
                pts += 8
        # Sharpe 6m (0-20 pts)
        if sharpe is not None:
            if sharpe >= 1.5:
                pts += 20
            elif sharpe >= 1.0:
                pts += 15
            elif sharpe >= 0.5:
                pts += 10
            elif sharpe >= 0:
                pts += 5

        score_num = min(100, max(0, round(pts)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1

        reason = f"Score {score_num}/100 | Vol {_fmt_num(vol, 1)}% | MaxDD {_fmt_num(dd, 1)}% | Dist52w {_fmt_num(dist, 1)}% | Sharpe {_fmt_num(sharpe, 2)}."
        nombre = asset.get("nombre", t)
        rows.append([str(i), t, nombre, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = _render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Riesgo principal"], rows)
    print("\n[Tabla de riesgos validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def sentiment_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE SENTIMIENTO DE MERCADO (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        print(f"[Evaluando {i}/{total}: {t}]", flush=True)
        d = prices.get(t, {})
        r1 = _to_float(d.get("ret_1m"))
        r3 = _to_float(d.get("ret_3m"))
        r12 = _to_float(d.get("ret_12m"))
        rs = _to_float(d.get("rs_vs_sp500"))
        momentum = _to_float(d.get("momentum_score"))

        # Score 0-100
        pts = 0.0
        # Momentum 1m (0-20 pts)
        if r1 is not None:
            if r1 > 5:
                pts += 20
            elif r1 > 0:
                pts += 15
            elif r1 > -5:
                pts += 10
            elif r1 > -10:
                pts += 5
        # Momentum 3m (0-20 pts)
        if r3 is not None:
            if r3 > 10:
                pts += 20
            elif r3 > 0:
                pts += 15
            elif r3 > -5:
                pts += 8
        # Momentum 12m (0-20 pts)
        if r12 is not None:
            if r12 > 20:
                pts += 20
            elif r12 > 5:
                pts += 15
            elif r12 > 0:
                pts += 8
        # Fuerza relativa vs S&P 500 (0-20 pts)
        if rs is not None:
            if rs > 10:
                pts += 20
            elif rs > 0:
                pts += 15
            elif rs > -10:
                pts += 8
        # Momentum compuesto (0-20 pts)
        if momentum is not None:
            if momentum > 15:
                pts += 20
            elif momentum > 5:
                pts += 15
            elif momentum > 0:
                pts += 10
            elif momentum > -5:
                pts += 3

        score_num = min(100, max(0, round(pts)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1

        reason = f"Score {score_num}/100 | 1m {_fmt_num(r1, 1)}% | 3m {_fmt_num(r3, 1)}% | 12m {_fmt_num(r12, 1)}% | RS {_fmt_num(rs, 1)} | Mom {_fmt_num(momentum, 1)}."
        nombre = asset.get("nombre", t)
        rows.append([str(i), t, nombre, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = _render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Sentimiento clave"], rows)
    print("\n[Tabla de sentimiento validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def macd_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    """6to evaluador: MACD(12,26,9) — confirma momentum y timing de entrada."""
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE MACD (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        print(f"[Evaluando {i}/{total}: {t}]", flush=True)
        d = prices.get(t, {})
        macd = _to_float(d.get("macd"))
        signal = _to_float(d.get("macd_signal"))
        hist = _to_float(d.get("macd_hist"))

        # Score 0-100
        pts = 0.0
        if macd is not None and signal is not None and hist is not None:
            # MACD por encima de signal (0-40 pts)
            if macd > signal:
                pts += 40
                # Cruce reciente (histograma positivo y creciente)
                if hist > 0:
                    pts += 10
            else:
                # MACD bajo signal pero convergiendo (histograma subiendo)
                if hist is not None and hist > -0.5:
                    pts += 15

            # MACD positivo (tendencia alcista confirmada, 0-25 pts)
            if macd > 0:
                pts += 25
            elif macd > -1:
                pts += 10

            # Histograma positivo (0-25 pts)
            if hist > 0:
                pts += 15
            elif hist > -0.5:
                pts += 5

        score_num = min(100, max(0, round(pts)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1

        reason = f"Score {score_num}/100 | MACD {_fmt_num(macd, 4)} | Signal {_fmt_num(signal, 4)} | Hist {_fmt_num(hist, 4)}."
        nombre = asset.get("nombre", t)
        rows.append([str(i), t, nombre, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = _render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "MACD detalle"], rows)
    print("\n[Tabla MACD validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def _parse_verdict_table(table: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    lines = [ln.strip() for ln in table.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return verdicts

    headers = [c.strip() for c in lines[0].strip("|").split("|")]
    hmap = {h.upper(): idx for idx, h in enumerate(headers)}
    tidx = hmap.get("TICKER")
    vidx = hmap.get("VEREDICTO")
    if tidx is None or vidx is None:
        return verdicts

    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) <= max(tidx, vidx):
            continue
        ticker = cells[tidx].upper()
        verdict = cells[vidx].upper()
        if ticker and verdict in {"OK", "NOK"}:
            verdicts[ticker] = verdict
    return verdicts


def _extract_evitar_tickers(verdict_output: str) -> set[str]:
    """Extrae los tickers con decision EVITAR del veredicto final."""
    evitar: set[str] = set()
    lines = [ln.strip() for ln in verdict_output.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return evitar
    headers = [c.strip().upper() for c in lines[0].strip("|").split("|")]
    hmap = {h: idx for idx, h in enumerate(headers)}
    tidx = hmap.get("TICKER")
    didx = hmap.get("DECISION FINAL")
    if tidx is None or didx is None:
        return evitar
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) <= max(tidx, didx):
            continue
        ticker = cells[tidx].upper()
        decision = cells[didx].upper()
        if ticker and "EVITAR" in decision:
            evitar.add(ticker)
    return evitar


def final_verdict(top10_assets: list[dict[str, Any]], ta_table: str, fa_table: str, risk_table: str, sent_table: str, ew_table: str, macd_table: str) -> str:
    print("\n" + "=" * 60, flush=True)
    print("VEREDICTO FINAL CONSOLIDADO", flush=True)
    print("=" * 60, flush=True)

    ta_map = _parse_verdict_table(ta_table)
    fa_map = _parse_verdict_table(fa_table)
    risk_map = _parse_verdict_table(risk_table)
    sent_map = _parse_verdict_table(sent_table)
    ew_map = _parse_verdict_table(ew_table)
    macd_map = _parse_verdict_table(macd_table)

    rows: list[list[str]] = []
    buy = watch = avoid = 0

    for i, asset in enumerate(top10_assets, start=1):
        t = asset["ticker"]
        nombre = asset.get("nombre", t)
        tech = ta_map.get(t, "NOK")
        fund = fa_map.get(t, "NOK")
        risk = risk_map.get(t, "NOK")
        sent = sent_map.get(t, "NOK")
        elliott = ew_map.get(t, "NOK")
        macd = macd_map.get(t, "NOK")
        ok_total = [tech, fund, risk, sent, elliott, macd].count("OK")
        if ok_total >= 5:
            decision = "COMPRAR"
            buy += 1
        elif ok_total >= 3:
            decision = "VIGILAR"
            watch += 1
        else:
            decision = "EVITAR"
            avoid += 1

        rows.append([str(i), t, nombre, f"{asset['peso']}%", tech, fund, risk, sent, elliott, macd, decision])

    table = _render_md_table(
        ["#", "Ticker", "Nombre", "%Cartera", "Tecnico", "Fundamental", "Riesgo", "Sentimiento", "Elliott", "MACD", "DECISION FINAL"],
        rows,
    )

    reco_lines = [
        f"Priorizar activos COMPRAR ({buy} de {len(top10_assets)}) con entradas escalonadas.",
        f"Mantener en observacion activos VIGILAR ({watch} de {len(top10_assets)}) y revisar en cada cierre semanal.",
        f"Reducir exposicion en EVITAR ({avoid} de {len(top10_assets)}) hasta mejora de senales tecnicas y riesgo.",
    ]

    final_output = table + "\n\nRECOMENDACIONES FINALES:\n" + "\n".join(reco_lines)
    print("\n[Veredicto final validado]", flush=True)
    print(final_output, flush=True)
    return final_output


def run_debate(args: argparse.Namespace) -> int:
    global _AVAILABLE_MODELS
    available_models = check_ollama(args.host)
    _AVAILABLE_MODELS = available_models
    model = resolve_model(args.model, available_models)

    # Asignar modelos diversos a cada agente y analista
    analyst_roles = [
        "Moderador Consenso", "Moderador Top 20",
        "Analista Tecnico", "Analista Fundamental",
        "Gestor Riesgos", "Analista Sentimiento", "Veredicto Final",
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

    # Fase 0: Cargar datos reales de mercado
    briefings = load_all_market_data(months=12, include_raw=True)
    market_general = briefings.get("general", "")
    market_tech = briefings.get("tecnico", "")
    market_fund = briefings.get("fundamental", "")
    market_risk = briefings.get("riesgo", "")
    raw_prices: dict[str, dict[str, Any]] = briefings.get("raw_prices", {})
    raw_fundamentals: dict[str, dict[str, Any]] = briefings.get("raw_fundamentals", {})

    if market_general:
        print("\n[Datos de mercado cargados correctamente]")
    else:
        print("\n[AVISO: Sin datos de mercado - el debate usara solo conocimiento del modelo]")

    # Fase 0b: Cargar carteras reales de gurús (13F SEC EDGAR)
    print("\n[Cargando carteras reales de gurus (13F)...]", flush=True)
    all_guru_holdings = fetch_all_guru_holdings()
    guru_conv = compute_guru_conviction(all_guru_holdings, TICKERS)

    transcript: List[str] = []
    start = time.monotonic()
    deadline = start + max(5, args.seconds)

    turn_count = 0
    while time.monotonic() < deadline and turn_count < args.max_turns:
        for agent in AGENTS:
            if time.monotonic() >= deadline or turn_count >= args.max_turns:
                break
            agent_turn(
                agent=agent,
                host=args.host,
                model=model_map[agent.name],
                transcript=transcript,
                context_lines=args.context_lines,
                market_briefing=market_general,
            )
            turn_count += 1

    elapsed = time.monotonic() - start
    print(f"\nDebate finalizado en {elapsed:.1f} segundos con {turn_count} turnos.")

    # Fase 2: Consenso
    consensus_and_summary(
        host=args.host,
        model=model_map["Moderador Consenso"],
        transcript=transcript,
        context_lines=args.context_lines,
        market_briefing=market_general,
    )

    # Fase 3: Top 20 inversiones
    top20_text, top20_assets = top20_investments(
        host=args.host,
        model=model_map["Moderador Top 20"],
        transcript=transcript,
        context_lines=args.context_lines,
        prices=raw_prices,
        fundamentals=raw_fundamentals,
        market_briefing=market_general,
        guru_conviction=guru_conv,
    )

    # Fase 4: Evaluaciones independientes (con datos reales)
    ta_combined, ew_table = technical_analysis_review(
        top10_assets=top20_assets,
        prices=raw_prices,
    )

    fa_table = fundamental_analysis_review(
        top10_assets=top20_assets,
        fundamentals=raw_fundamentals,
        prices=raw_prices,
    )

    risk_table = risk_management_review(
        top10_assets=top20_assets,
        prices=raw_prices,
    )

    sent_table = sentiment_analysis_review(
        top10_assets=top20_assets,
        prices=raw_prices,
    )

    macd_table = macd_analysis_review(
        top10_assets=top20_assets,
        prices=raw_prices,
    )

    # Fase 5: Veredicto final consolidado
    verdict_output = final_verdict(
        top10_assets=top20_assets,
        ta_table=ta_combined,
        fa_table=fa_table,
        risk_table=risk_table,
        sent_table=sent_table,
        ew_table=ew_table,
        macd_table=macd_table,
    )

    # Fase 5b: Carteras reales de gurus
    guru_holdings_section(
        all_holdings=all_guru_holdings,
        tickers_dict=TICKERS,
        top20_tickers=[a["ticker"] for a in top20_assets],
    )

    # === SEGUNDA PASADA: excluir activos EVITAR ===
    evitar_tickers = _extract_evitar_tickers(verdict_output)
    if evitar_tickers:
        print("\n" + "=" * 60, flush=True)
        print("[=== SEGUNDA PASADA (sin EVITAR) ===]", flush=True)
        print("=" * 60, flush=True)
        print(f"Excluidos ({len(evitar_tickers)}): {', '.join(sorted(evitar_tickers))}", flush=True)

        top20_text2, top20_assets2 = top20_investments(
            host=args.host,
            model=model_map["Moderador Top 20"],
            transcript=transcript,
            context_lines=args.context_lines,
            prices=raw_prices,
            fundamentals=raw_fundamentals,
            market_briefing=market_general,
            exclude_tickers=evitar_tickers,
            guru_conviction=guru_conv,
        )

        ta_combined2, ew_table2 = technical_analysis_review(
            top10_assets=top20_assets2,
            prices=raw_prices,
        )
        fa_table2 = fundamental_analysis_review(
            top10_assets=top20_assets2,
            fundamentals=raw_fundamentals,
            prices=raw_prices,
        )
        risk_table2 = risk_management_review(
            top10_assets=top20_assets2,
            prices=raw_prices,
        )
        sent_table2 = sentiment_analysis_review(
            top10_assets=top20_assets2,
            prices=raw_prices,
        )
        macd_table2 = macd_analysis_review(
            top10_assets=top20_assets2,
            prices=raw_prices,
        )
        verdict_output2 = final_verdict(
            top10_assets=top20_assets2,
            ta_table=ta_combined2,
            fa_table=fa_table2,
            risk_table=risk_table2,
            sent_table=sent_table2,
            ew_table=ew_table2,
            macd_table=macd_table2,
        )

        # === TERCERA PASADA: excluir tambien los EVITAR de la pasada 2 ===
        evitar_tickers2 = _extract_evitar_tickers(verdict_output2)
        if evitar_tickers2:
            all_excluded = evitar_tickers | evitar_tickers2
            print("\n" + "=" * 60, flush=True)
            print("[=== TERCERA PASADA (sin EVITAR P1+P2) ===]", flush=True)
            print("=" * 60, flush=True)
            print(f"Nuevos excluidos P2 ({len(evitar_tickers2)}): {', '.join(sorted(evitar_tickers2))}", flush=True)
            print(f"Total excluidos ({len(all_excluded)}): {', '.join(sorted(all_excluded))}", flush=True)

            top20_text3, top20_assets3 = top20_investments(
                host=args.host,
                model=model_map["Moderador Top 20"],
                transcript=transcript,
                context_lines=args.context_lines,
                prices=raw_prices,
                fundamentals=raw_fundamentals,
                market_briefing=market_general,
                exclude_tickers=all_excluded,
                guru_conviction=guru_conv,
            )

            ta_combined3, ew_table3 = technical_analysis_review(
                top10_assets=top20_assets3,
                prices=raw_prices,
            )
            fa_table3 = fundamental_analysis_review(
                top10_assets=top20_assets3,
                fundamentals=raw_fundamentals,
                prices=raw_prices,
            )
            risk_table3 = risk_management_review(
                top10_assets=top20_assets3,
                prices=raw_prices,
            )
            sent_table3 = sentiment_analysis_review(
                top10_assets=top20_assets3,
                prices=raw_prices,
            )
            macd_table3 = macd_analysis_review(
                top10_assets=top20_assets3,
                prices=raw_prices,
            )
            verdict_output3 = final_verdict(
                top10_assets=top20_assets3,
                ta_table=ta_combined3,
                fa_table=fa_table3,
                risk_table=risk_table3,
                sent_table=sent_table3,
                ew_table=ew_table3,
                macd_table=macd_table3,
            )

    # Fase 6: Diagrama de arquitectura
    print_architecture_diagram()

    return 0


def main() -> int:
    args = parse_args()
    try:
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
║  │   ┌────────────────┐ ┌────────────────┐ ┌──────────┐ ┌──────────────┐  │   ║
║  │   │   Analista     │ │   Analista     │ │  Gestor  │ │  Analista    │  │   ║
║  │   │   Tecnico      │ │  Fundamental   │ │   de     │ │     de       │  │   ║
║  │   │                │ │                │ │ Riesgos  │ │ Sentimiento  │  │   ║
║  │   │ RSI, MACD,     │ │ PER, ROE,      │ │ Vol, VaR │ │ COT, flujos │  │   ║
║  │   │ medias, vol.   │ │ FCF, deuda     │ │ drawdown │ │ put/call    │  │   ║
║  │   │                │ │                │ │ correl.  │ │ short int.  │  │   ║
║  │   │  OK / NOK      │ │  OK / NOK      │ │ OK / NOK │ │  OK / NOK   │  │   ║
║  │   └───────┬────────┘ └───────┬────────┘ └────┬─────┘ └──────┬──────┘  │   ║
║  │           └──────────────────┼───────────────┼───────────────┘         │   ║
║  └──────────────────────────────┼───────────────┼────────────────────────────┘   ║
║                                 │               │                              ║
║  ┌──────────────────────────────▼───────────────▼────────────────────────────┐   ║
║  │                  FASE 5: VEREDICTO FINAL CONSOLIDADO                     │   ║
║  │                                                                          │   ║
║  │   Regla de decision por activo:                                          │   ║
║  │     >= 3 OK de 4 evaluadores  ──►  COMPRAR                              │   ║
║  │        2 OK de 4 evaluadores  ──►  VIGILAR                              │   ║
║  │     <= 1 OK de 4 evaluadores  ──►  EVITAR                               │   ║
║  │                                                                          │   ║
║  │   Salida: Tabla final + 3 lineas de recomendacion                        │   ║
║  └──────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                ║
║  TOTAL AGENTES: 7 debate + 1 moderador + 4 evaluadores + 1 veredicto = 13     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    print(diagram, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
