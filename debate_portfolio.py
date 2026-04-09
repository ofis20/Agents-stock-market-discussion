"""Construccion de cartera Top 20 y utilidades deterministas asociadas."""

from __future__ import annotations

import re
from typing import Any

from debate_prompts import build_top20_messages
from market_data import TICKERS
from ollama_client import OllamaClient


def to_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def fmt_num(val: float | None, decimals: int = 2) -> str:
    if val is None:
        return "N/D"
    return f"{val:.{decimals}f}"


def render_md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def normalize_weights_100(scores: list[float]) -> list[int]:
    safe = [max(0.01, score) for score in scores]
    total = sum(safe)
    raw = [(score / total) * 100.0 for score in safe]
    base = [int(value) for value in raw]
    missing = 100 - sum(base)
    remainders = sorted([(raw[idx] - base[idx], idx) for idx in range(len(raw))], reverse=True)
    for _, idx in remainders[:max(0, missing)]:
        base[idx] += 1
    return base


def find_zigzag_pivots(closes: list[float], pct_threshold: float = 5.0) -> list[tuple[int, float, str]]:
    """Detecta pivots (High/Low) usando filtro de zigzag con umbral porcentual."""
    if len(closes) < 10:
        return []

    pivots: list[tuple[int, float, str]] = []
    last_pivot_type = ""
    last_pivot_price = closes[0]
    last_pivot_idx = 0

    for idx in range(1, len(closes)):
        change_from_last = ((closes[idx] - last_pivot_price) / last_pivot_price) * 100

        if change_from_last >= pct_threshold:
            if last_pivot_type != "H":
                pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type or "L"))
            last_pivot_type = "H"
            last_pivot_price = closes[idx]
            last_pivot_idx = idx
        elif change_from_last <= -pct_threshold:
            if last_pivot_type != "L":
                pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type or "H"))
            last_pivot_type = "L"
            last_pivot_price = closes[idx]
            last_pivot_idx = idx
        else:
            if last_pivot_type == "H" and closes[idx] > last_pivot_price:
                last_pivot_price = closes[idx]
                last_pivot_idx = idx
            elif last_pivot_type == "L" and closes[idx] < last_pivot_price:
                last_pivot_price = closes[idx]
                last_pivot_idx = idx

    pivots.append((last_pivot_idx, last_pivot_price, last_pivot_type or "H"))
    return pivots


def find_zigzag_pivots_adaptive(
    closes: list[float],
    min_pivots: int = 5,
    max_pivots: int = 11,
) -> tuple[list[tuple[int, float, str]], float]:
    """Busca pivots con umbral adaptativo para ondas de grado mayor (superciclo).

    Prueba umbrales desde 25 % bajando hasta 5 % y devuelve el primero que
    produzca entre *min_pivots* y *max_pivots*.  Si ninguno cae en rango,
    devuelve el de menor exceso por encima de *min_pivots*.
    """
    _THRESHOLDS = [40, 35, 30, 25, 22, 20, 18, 16, 14, 12, 10, 8, 7, 6, 5]
    best: tuple[list[tuple[int, float, str]], float] | None = None
    for pct in _THRESHOLDS:
        pivots = find_zigzag_pivots(closes, pct_threshold=pct)
        n = len(pivots)
        if min_pivots <= n <= max_pivots:
            return pivots, float(pct)
        if n >= min_pivots and (best is None or n < len(best[0])):
            best = (pivots, float(pct))
    if best is not None:
        return best
    return find_zigzag_pivots(closes, pct_threshold=5.0), 5.0


def classify_elliott_wave(closes: list[float]) -> dict[str, Any]:
    """Clasifica la fase de Elliott Wave actual usando zigzag adaptativo y etiquetado real de ondas.

    Devuelve dict con onda, fase, prevision, targets 1y/3y/6y y datos internos
    (_pivots, _used_pct) para reutilizacion en el overlay grafico.
    """
    nd: dict[str, Any] = {
        "onda": "N/D", "fase": "Datos insuficientes",
        "prevision": "Sin datos suficientes para analisis.",
        "precio_actual": 0.0,
        "target_1y": 0.0, "target_3y": 0.0, "target_6y": 0.0,
        "rent_1y": 0.0, "rent_3y": 0.0, "rent_6y": 0.0,
        # Aliases para compatibilidad con codigo legacy (streamlit charts)
        "target_1m": 0.0, "target_3m": 0.0, "target_6m": 0.0,
        "rent_1m": 0.0, "rent_3m": 0.0, "rent_6m": 0.0,
        "_pivots": [], "_used_pct": 5.0,
    }
    if len(closes) < 60:
        return nd

    series = closes[-1512:] if len(closes) >= 1512 else closes
    current = series[-1]

    # -- Zigzag adaptativo: busca umbrales que den 5-11 pivots (grado mayor) --
    pivots, used_pct = find_zigzag_pivots_adaptive(series)

    highs_all = [p for p in pivots if p[2] == "H"]
    lows_all = [p for p in pivots if p[2] == "L"]
    last_high = max(p[1] for p in highs_all) if highs_all else current
    last_low = min(p[1] for p in lows_all) if lows_all else current
    swing_range = last_high - last_low if last_high > last_low else current * 0.10

    fib_038 = swing_range * 0.382
    fib_050 = swing_range * 0.500
    fib_062 = swing_range * 0.618
    fib_100 = swing_range
    fib_162 = swing_range * 1.618

    def _build(onda: str, fase: str, prevision: str, pct_1y: float, pct_3y: float, pct_6y: float) -> dict[str, Any]:
        result = {
            "onda": onda, "fase": fase, "prevision": prevision,
            "precio_actual": round(current, 2),
            "target_1y": round(current * (1 + pct_1y / 100), 2),
            "target_3y": round(current * (1 + pct_3y / 100), 2),
            "target_6y": round(current * (1 + pct_6y / 100), 2),
            "rent_1y": round(pct_1y, 1),
            "rent_3y": round(pct_3y, 1),
            "rent_6y": round(pct_6y, 1),
            "_pivots": pivots,
            "_used_pct": used_pct,
        }
        # Aliases para compatibilidad con codigo legacy (streamlit charts)
        result["target_1m"] = result["target_1y"]
        result["target_3m"] = result["target_3y"]
        result["target_6m"] = result["target_1y"]
        result["rent_1m"] = result["rent_1y"]
        result["rent_3m"] = result["rent_3y"]
        result["rent_6m"] = result["rent_1y"]
        return result

    diffs = [abs(series[idx] - series[idx - 1]) for idx in range(max(1, len(series) - 20), len(series))]
    atr_pct = (sum(diffs) / len(diffs) / current * 100) if diffs and current else 1.5

    # -- Fallback si muy pocos pivots --
    if len(pivots) < 3:
        total_change = ((current - series[0]) / series[0]) * 100
        if total_change > 15:
            up_3y = min(fib_162 / current * 100, 80)
            return _build("3↑", "Impulso alcista fuerte de superciclo", "Superciclo alcista en curso. Extension de onda 3 con objetivos Fibonacci superiores a 3-6 anos.", up_3y * 0.35, up_3y, up_3y * 1.8)
        if total_change > 0:
            up_3y = min(fib_100 / current * 100, 60)
            return _build("1↑", "Inicio de superciclo alcista", "Arranque de tendencia secular. Onda 3 (la mas potente) podria desarrollarse en los proximos 2-4 anos.", up_3y * 0.3, up_3y, up_3y * 1.7)
        if total_change > -15:
            dn_1y = -fib_038 / current * 100
            return _build("A↓", "Inicio de correccion secular", "Correccion dentro de superciclo. Esperar finalizacion completa de ABC (1-3 anos) antes de acumular.", dn_1y, fib_050 / current * 100, fib_100 / current * 100)
        up_6y = fib_062 / current * 100
        return _build("C↓", "Correccion profunda de superciclo", "Cerca de suelo secular. Zona de acumulacion generacional para los proximos 3-6 anos.", -atr_pct * 2, up_6y * 0.5, up_6y * 1.5)

    # -- Etiquetar pivots con ondas de Elliott --
    labeled = label_elliott_wave_pivots(pivots)
    wave_info = [(lbl, price, ptype) for _, price, ptype, lbl in labeled if lbl and lbl != "0"]

    if len(wave_info) < 2:
        # Insuficientes ondas etiquetadas — clasificacion por tendencia
        total_change = ((current - series[0]) / series[0]) * 100
        if total_change > 15:
            up_3y = min(fib_162 / current * 100, 80)
            return _build("3↑", "Impulso alcista fuerte de superciclo", "Superciclo alcista en curso.", up_3y * 0.35, up_3y, up_3y * 1.8)
        if total_change > 0:
            up_3y = min(fib_100 / current * 100, 60)
            return _build("1↑", "Inicio de superciclo alcista", "Arranque de tendencia secular.", up_3y * 0.3, up_3y, up_3y * 1.7)
        dn_1y = -fib_038 / current * 100
        return _build("A↓", "Inicio de correccion secular", "Correccion dentro de superciclo.", dn_1y, fib_050 / current * 100, fib_100 / current * 100)

    # -- Determinar direccion del impulso --
    wave_0_price = wave_1_price = None
    for _, price, _, lbl in labeled:
        if lbl == "0":
            wave_0_price = price
        elif lbl == "1":
            wave_1_price = price
    is_bullish = (wave_1_price > wave_0_price) if (wave_0_price is not None and wave_1_price is not None) else (current > series[0])

    # -- Ultima onda etiquetada -> fase actual (que onda esta EN PROGRESO) --
    last_lbl = wave_info[-1][0]

    if is_bullish:
        _NEXT: dict[str, str] = {
            "1": "2↓", "2": "3↑", "3": "4↓", "4": "5↑", "5": "A↓",
            "A": "B↑", "B": "C↓", "C": "1↑",
        }
    else:
        _NEXT = {
            "1": "2↑", "2": "3↓", "3": "4↑", "4": "5↓", "5": "A↑",
            "A": "B↓", "B": "C↑", "C": "1↓",
        }

    onda = _NEXT.get(last_lbl, "")

    # -- Targets y previsiones por onda --
    if onda == "1↑":
        up_3y = min(fib_100 / current * 100, 60)
        return _build(onda, "Inicio de superciclo alcista", "Arranque de tendencia secular alcista. Onda 3 (la mas potente) podria desarrollarse en los proximos 2-4 anos.", up_3y * 0.3, up_3y, up_3y * 1.7)
    if onda == "2↓":
        up_3y = fib_162 / current * 100
        return _build(onda, "Correccion secular dentro de superciclo", "Oportunidad de acumulacion generacional. Onda 3 (la mas explosiva) se desarrollaria en los proximos 2-5 anos.", -atr_pct * 2, min(up_3y * 0.5, 70), min(up_3y, 120))
    if onda == "3↑":
        up_3y = min(fib_162 / current * 100, 90)
        return _build(onda, "Impulso principal del superciclo", "Onda mas potente del superciclo en curso. Mantener posiciones core, objetivos Fibonacci a 3-6 anos.", up_3y * 0.3, up_3y, up_3y * 1.6)
    if onda == "4↓":
        up_3y = fib_100 / current * 100
        return _build(onda, "Correccion intermedia del superciclo", "Correccion estructural sana. Ultima gran oportunidad de acumulacion antes de onda 5 del superciclo.", -atr_pct * 2, min(up_3y * 0.6, 60), min(up_3y, 90))
    if onda == "5↑":
        dn_3y = -fib_038 / current * 100
        return _build(onda, "Ultimo impulso del superciclo", "Cercano a techo secular. Reducir exposicion progresivamente en los proximos 1-3 anos. Riesgo de correccion mayor.", atr_pct, dn_3y * 0.5, dn_3y)
    if onda == "A↓":
        dn_3y = -fib_050 / current * 100
        return _build(onda, "Inicio de correccion tras superciclo completo", "Gran correccion ABC en curso. Duracion tipica 2-4 anos antes de nuevo superciclo alcista.", dn_3y * 0.4, dn_3y, -fib_038 / current * 100)
    if onda == "B↑":
        dn_3y = -fib_038 / current * 100
        return _build(onda, "Rebote dentro de correccion secular", "Rebote tecnico dentro de estructura correctiva de superciclo. Confirmar ruptura de maximos historicos para cambio de ciclo.", atr_pct, -atr_pct * 4, dn_3y)
    if onda == "C↓":
        up_6y = fib_062 / current * 100
        return _build(onda, "Correccion profunda de superciclo", "Posible suelo secular proximo. Zona de acumulacion generacional. Horizonte de recuperacion 3-6 anos.", -atr_pct * 2, up_6y * 0.4, up_6y * 1.5)

    # -- Ondas de impulso bajista --
    if onda == "1↓":
        dn_3y = min(fib_100 / current * 100, 60)
        return _build(onda, "Inicio de superciclo bajista", "Arranque de tendencia secular bajista. Proteger capital y esperar confirmacion de suelo.", -dn_3y * 0.3, -dn_3y, -dn_3y * 1.5)
    if onda == "2↑":
        dn_3y = fib_162 / current * 100
        return _build(onda, "Rebote correctivo en superciclo bajista", "Rebote dentro de impulso bajista. Onda 3 bajista (mas violenta) podria desarrollarse.", atr_pct * 2, -min(dn_3y * 0.5, 70), -min(dn_3y, 120))
    if onda == "3↓":
        dn_3y = min(fib_162 / current * 100, 90)
        return _build(onda, "Impulso bajista principal del superciclo", "Onda mas potente del superciclo bajista en curso. Liquidar posiciones largas.", -dn_3y * 0.3, -dn_3y, -dn_3y * 1.6)
    if onda == "4↑":
        dn_3y = fib_100 / current * 100
        return _build(onda, "Rebote en superciclo bajista", "Rebote estructural dentro de tendencia bajista mayor.", atr_pct * 2, -min(dn_3y * 0.6, 60), -min(dn_3y, 90))
    if onda == "5↓":
        up_3y = fib_038 / current * 100
        return _build(onda, "Ultimo impulso bajista del superciclo", "Cercano a suelo secular. Fase de capitulacion, preparar acumulacion a largo plazo.", -atr_pct, up_3y * 0.5, up_3y)
    if onda == "A↑":
        up_3y = fib_050 / current * 100
        return _build(onda, "Primer rebote tras superciclo bajista completo", "Primera reaccion alcista despues de 5 ondas bajistas. Potencial inicio de recuperacion secular.", up_3y * 0.4, up_3y, fib_062 / current * 100)
    if onda == "B↓":
        up_3y = fib_038 / current * 100
        return _build(onda, "Retroceso dentro de correccion alcista secular", "Retroceso dentro de estructura correctiva. Mantener perspectiva de recuperacion a largo plazo.", -atr_pct, atr_pct * 4, up_3y)
    if onda == "C↑":
        up_6y = fib_062 / current * 100
        return _build(onda, "Fase final de correccion alcista secular", "Nuevo superciclo podria seguir, o confirmacion de cambio de tendencia.", atr_pct * 2, up_6y * 0.4, up_6y * 0.8)

    # -- Fallback general --
    pct_from_high = ((current - last_high) / last_high) * 100
    if is_bullish:
        up_3y = fib_050 / current * 100
        return _build("3↑", "Impulso alcista de superciclo", "Tendencia alcista en curso.", up_3y * 0.3, up_3y, fib_100 / current * 100)
    dn_1y = -fib_038 / current * 100
    return _build("A↓", "Inicio de correccion secular", "Tendencia bajista en desarrollo.", dn_1y, -fib_050 / current * 100, fib_062 / current * 100)


# ---------------------------------------------------------------------------
# Etiquetado de ondas de Elliott sobre pivotes de zigzag
# ---------------------------------------------------------------------------

def _count_elliott_from(
    pivots: list[tuple[int, float, str]], start: int, bullish: bool,
) -> tuple[list[str], int, int]:
    """Cuenta ondas de Elliott desde *start* usando dirección de precio.

    Devuelve (labels, wave_count, last_labeled_idx).
    """
    n = len(pivots)
    labels = [""] * n
    SEQ = ["1", "2", "3", "4", "5", "A", "B", "C"]

    # Direcciones esperadas: alcista empieza subiendo, bajista bajando
    if bullish:
        DIRS = ["UP", "DOWN", "UP", "DOWN", "UP", "DOWN", "UP", "DOWN"]
    else:
        DIRS = ["DOWN", "UP", "DOWN", "UP", "DOWN", "UP", "DOWN", "UP"]

    origin = pivots[start][1]
    labels[start] = "0"
    wave_count = 1
    last_labeled = start
    prices: dict[str, float] = {"0": origin}
    wave_i = 0
    pos = start + 1
    prev_price = origin

    while pos < n and wave_i < len(SEQ):
        price = pivots[pos][1]
        actual_dir = "UP" if price > prev_price else "DOWN"

        if actual_dir != DIRS[wave_i]:
            break

        label = SEQ[wave_i]

        # Validación de reglas de Elliott
        ok = True
        if bullish:
            if label == "2" and price <= origin:
                ok = False
            elif label == "3" and "1" in prices and price <= prices["1"]:
                ok = False
        else:
            if label == "2" and price >= origin:
                ok = False
            elif label == "3" and "1" in prices and price >= prices["1"]:
                ok = False

        if not ok:
            break

        labels[pos] = label
        prices[label] = price
        wave_count += 1
        last_labeled = pos
        wave_i += 1
        prev_price = price
        pos += 1

    return labels, wave_count, last_labeled


def label_elliott_wave_pivots(
    pivots: list[tuple[int, float, str]],
) -> list[tuple[int, float, str, str]]:
    """
    Etiqueta pivotes de zigzag con ondas de Elliott (1-5 impulso, A-B-C corrección).

    Analiza la dirección de precios entre pivotes consecutivos (no depende de
    las etiquetas H/L del zigzag). Prueba todos los posibles puntos de inicio
    y devuelve la estructura que mejor cubra los datos más recientes con el
    mayor número de ondas válidas.

    Retorna lista de (index, price, type, wave_label).
    """
    if len(pivots) < 4:
        return [(idx, p, t, "") for idx, p, t in pivots]

    best_labels: list[str] | None = None
    best_score = -1

    for si in range(len(pivots) - 3):
        # Determinar dirección a partir del movimiento de precio, no del tipo H/L
        if si + 1 >= len(pivots):
            continue
        bullish = pivots[si + 1][1] > pivots[si][1]
        labels, wcount, last_lab = _count_elliott_from(pivots, si, bullish)
        if wcount < 4:
            continue
        # Priorizar: alcanzar pivotes recientes > más ondas
        total = last_lab * 100 + wcount * 10
        if total > best_score:
            best_score = total
            best_labels = labels

    if best_labels is None:
        return [(idx, p, t, "") for idx, p, t in pivots]

    return [
        (pivots[i][0], pivots[i][1], pivots[i][2], best_labels[i])
        for i in range(len(pivots))
    ]


ALLOWED_TYPES = {"Accion", "ETF", "Commodity", "Cripto"}


def top20_fallback_from_data(
    prices: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
    exclude_tickers: set[str] | None = None,
    guru_conviction: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    sector_momentum: dict[str, list[float]] = {}
    for ticker_name, fundamental_row in fundamentals.items():
        sector = fundamental_row.get("sector", "N/D")
        price_row = prices.get(ticker_name, {})
        momentum = to_float(price_row.get("momentum_score"))
        if sector != "N/D" and momentum is not None:
            sector_momentum.setdefault(sector, []).append(momentum)
    sector_avg = {sector: sum(values) / len(values) for sector, values in sector_momentum.items() if values}
    top_sectors = sorted(sector_avg, key=sector_avg.get, reverse=True)[:3] if sector_avg else []

    rows: list[tuple[str, float, str, str, float]] = []
    for ticker, meta in TICKERS.items():
        asset_type = meta.get("tipo", "")
        if asset_type not in ALLOWED_TYPES:
            continue
        if exclude_tickers and ticker in exclude_tickers:
            continue
        price_row = prices.get(ticker)
        if not price_row:
            continue

        rsi = to_float(price_row.get("rsi14"))
        sma50 = to_float(price_row.get("sma50"))
        sma200 = to_float(price_row.get("sma200"))
        price = to_float(price_row.get("precio"))
        vol = to_float(price_row.get("vol_20d")) or 60.0
        close_hist = price_row.get("close_hist", [])
        sharpe = to_float(price_row.get("sharpe_6m"))
        rs_sp500 = to_float(price_row.get("rs_vs_sp500"))
        momentum = to_float(price_row.get("momentum_score"))
        avg_vol_20d = to_float(price_row.get("avg_vol_20d"))
        macd_hist = to_float(price_row.get("macd_hist"))

        if avg_vol_20d is not None and avg_vol_20d < 100_000 and asset_type == "Accion":
            continue

        trend_bonus = 0.0
        if price and sma50 and sma200:
            if price > sma50 > sma200:
                trend_bonus = 6.0
            elif price > sma200:
                trend_bonus = 3.0
            else:
                trend_bonus = -2.0

        rsi_bonus = 0.0
        if rsi is not None:
            if 40 <= rsi <= 68:
                rsi_bonus = 3.0
            elif rsi > 78:
                rsi_bonus = -3.0
            elif rsi < 30:
                rsi_bonus = 1.0

        ew_bonus = 0.0
        ew_onda = "N/D"
        if close_hist:
            ew = classify_elliott_wave(close_hist)
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
            # Ondas de impulso bajista — penalizan
            elif ew_onda in ("1↓", "3↓", "5↓"):
                ew_bonus = -5.0
            elif ew_onda in ("2↑", "4↑"):
                ew_bonus = -2.0
            # Ondas correctivas tras impulso bajista — posible suelo
            elif ew_onda in ("A↑", "C↑"):
                ew_bonus = 3.0
            elif ew_onda == "B↓":
                ew_bonus = -1.0

        fundamental_row = fundamentals.get(ticker, {})
        per = to_float(fundamental_row.get("per"))
        forward_pe = to_float(fundamental_row.get("forward_pe"))
        peg = to_float(fundamental_row.get("peg"))
        roe = to_float(fundamental_row.get("roe"))
        growth = to_float(fundamental_row.get("crec_ingresos"))
        sector = fundamental_row.get("sector", "N/D")

        fundamental_bonus = 0.0
        if roe is not None and roe >= 0.10:
            fundamental_bonus += 3.0
        if growth is not None and growth > 0:
            fundamental_bonus += 2.0
        eff_pe = forward_pe if forward_pe else per
        if eff_pe is not None and eff_pe <= 30:
            fundamental_bonus += 2.0
        if peg is not None:
            if peg <= 1.0:
                fundamental_bonus += 4.0
            elif peg <= 1.5:
                fundamental_bonus += 2.0

        rs_bonus = 0.0
        if rs_sp500 is not None:
            if rs_sp500 > 15:
                rs_bonus = 5.0
            elif rs_sp500 > 5:
                rs_bonus = 3.0
            elif rs_sp500 > 0:
                rs_bonus = 1.5

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

        macd_bonus = 0.0
        if macd_hist is not None:
            if macd_hist > 0:
                macd_bonus = 3.0
            elif macd_hist > -0.5:
                macd_bonus = 1.0

        sector_bonus = 3.0 if sector in top_sectors else 0.0

        vol_confirm = 0.0
        vol_ratio = to_float(price_row.get("vol_ratio"))
        if vol_ratio is not None and vol_ratio >= 1.2:
            vol_confirm = 2.0

        guru_bonus = guru_conviction.get(ticker, 0.0) if guru_conviction else 0.0

        score = (
            trend_bonus + rsi_bonus + ew_bonus + fundamental_bonus +
            rs_bonus + mom_bonus + sharpe_bonus + macd_bonus +
            sector_bonus + vol_confirm + guru_bonus - (0.03 * vol)
        )
        guru_tag = f", Guru {guru_bonus:.0f}" if guru_bonus > 0 else ""
        thesis = f"Elliott {ew_onda}, Mom {fmt_num(momentum, 1)}, Sharpe {fmt_num(sharpe, 2)}, RS {fmt_num(rs_sp500, 1)}{guru_tag}."
        rows.append((ticker, score, thesis, asset_type, vol))

    rows.sort(key=lambda row: row[1], reverse=True)

    selected: list[tuple[str, float, str, str, float]] = []
    count_by_type: dict[str, int] = {}
    for row in rows:
        asset_type = row[3]
        count = count_by_type.get(asset_type, 0)
        if asset_type == "Accion" and count >= 12:
            continue
        if count >= 5:
            continue
        selected.append(row)
        count_by_type[asset_type] = count + 1
        if len(selected) == 20:
            break

    if len(selected) < 20:
        used = {item[0] for item in selected}
        for row in rows:
            if row[0] not in used:
                selected.append(row)
                if len(selected) == 20:
                    break

    if len(selected) < 20:
        raise RuntimeError("No hay suficientes activos con datos para construir Top 20.")

    inv_vols = [1.0 / max(row[4], 5.0) for row in selected]
    total_inv = sum(inv_vols)
    raw_weights = [(inv_vol / total_inv) * 100 for inv_vol in inv_vols]
    floored = [max(2.0, weight) for weight in raw_weights]
    total_floored = sum(floored)
    weights_float = [(weight / total_floored) * 100 for weight in floored]
    weights = normalize_weights_100(weights_float)

    out: list[dict[str, Any]] = []
    for idx, (ticker, _, thesis, asset_type, _vol) in enumerate(selected):
        name = TICKERS.get(ticker, {}).get("nombre", ticker)
        out.append({
            "rank": idx + 1,
            "ticker": ticker,
            "nombre": name,
            "tipo": asset_type,
            "peso": weights[idx],
            "tesis": thesis,
        })
    return out


def parse_top20_lines(text: str, prices: dict[str, dict[str, Any]], exclude_tickers: set[str] | None = None) -> list[dict[str, Any]]:
    lines = [line.strip().replace("**", "") for line in text.splitlines() if line.strip()]
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in lines:
        match = re.match(r"^\s*(\d+)\.\s+([A-Z0-9\-\.=]+)\s*-\s*([A-Za-z]+)\s*-\s*(\d+)%\s*-\s*(.+)$", line)
        if not match:
            continue
        ticker = match.group(2).upper()
        asset_type = match.group(3).capitalize()
        weight = int(match.group(4))
        thesis = match.group(5).strip()
        if ticker in seen:
            continue
        if exclude_tickers and ticker in exclude_tickers:
            continue
        if ticker not in prices:
            continue
        real_type = prices[ticker].get("tipo", asset_type)
        if real_type not in ALLOWED_TYPES:
            continue
        seen.add(ticker)
        name = TICKERS.get(ticker, {}).get("nombre", ticker)
        parsed.append({
            "rank": int(match.group(1)),
            "ticker": ticker,
            "nombre": name,
            "tipo": real_type,
            "peso": weight,
            "tesis": thesis,
        })
    parsed.sort(key=lambda item: item["rank"])
    return parsed


def render_top20_lines(assets: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{idx+1}. {asset['ticker']} - {asset.get('nombre', asset['ticker'])} - {asset['tipo']} - {asset['peso']}% - {asset['tesis']}"
        for idx, asset in enumerate(assets)
    )


def top20_investments(
    client: OllamaClient,
    model: str,
    transcript: list[str],
    recent_context_fn,
    prices: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
    market_briefing: str = "",
    exclude_tickers: set[str] | None = None,
    guru_conviction: dict[str, float] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Genera top20 diversificado (acciones, ETFs, commodities, cripto)."""
    del market_briefing
    context = recent_context_fn(transcript, max_lines=len(transcript))

    candidates_by_type: dict[str, list[str]] = {}
    for ticker, meta in TICKERS.items():
        asset_type = meta.get("tipo", "")
        if asset_type in ALLOWED_TYPES and ticker in prices:
            if exclude_tickers and ticker in exclude_tickers:
                continue
            candidates_by_type.setdefault(asset_type, []).append(ticker)

    cand_lines = []
    for asset_type, tickers_list in sorted(candidates_by_type.items()):
        cand_lines.append(f"  {asset_type}: {', '.join(tickers_list[:60])}")
    candidates_str = "\n".join(cand_lines)

    system, user = build_top20_messages(context=context, candidates_str=candidates_str)

    print("\n" + "=" * 60, flush=True)
    print("TOP 20 INVERSIONES CONSENSUADAS (marzo 2026)", flush=True)
    print("=" * 60, flush=True)

    text = client.stream_chat(
        model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        num_predict=900,
        temperature=0.3,
        silent=True,
    )

    parsed = parse_top20_lines(text, prices, exclude_tickers=exclude_tickers)
    valid_sum = sum(asset["peso"] for asset in parsed) == 100
    if len(parsed) != 20 or not valid_sum:
        print(f"\n  [AVISO: Top20 del modelo invalido ({len(parsed)} filas, suma={sum(asset['peso'] for asset in parsed)}). Usando fallback.]", flush=True)
        parsed = top20_fallback_from_data(prices, fundamentals, exclude_tickers=exclude_tickers, guru_conviction=guru_conviction)

    final = render_top20_lines(parsed)
    print("\n[Top 20 validado]", flush=True)
    print(final, flush=True)
    return final, parsed