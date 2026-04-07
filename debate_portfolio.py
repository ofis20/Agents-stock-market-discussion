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
            if last_pivot_type != "L":
                pivots.append((last_pivot_idx, last_pivot_price, "L"))
            last_pivot_type = "H"
            last_pivot_price = closes[idx]
            last_pivot_idx = idx
        elif change_from_last <= -pct_threshold:
            if last_pivot_type != "H":
                pivots.append((last_pivot_idx, last_pivot_price, "H"))
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


def classify_elliott_wave(closes: list[float]) -> dict[str, Any]:
    """Clasifica la fase de Elliott Wave actual, da prevision y calcula targets de precio."""
    nd = {
        "onda": "N/D", "fase": "Datos insuficientes",
        "prevision": "Sin datos suficientes para analisis.",
        "precio_actual": 0.0,
        "target_1m": 0.0, "target_3m": 0.0, "target_6m": 0.0,
        "rent_1m": 0.0, "rent_3m": 0.0, "rent_6m": 0.0,
    }
    if len(closes) < 60:
        return nd

    series = closes[-252:] if len(closes) >= 252 else closes
    current = series[-1]
    pivots = find_zigzag_pivots(series, pct_threshold=5.0)

    highs_all = [pivot for pivot in pivots if pivot[2] == "H"]
    lows_all = [pivot for pivot in pivots if pivot[2] == "L"]
    last_high = max(pivot[1] for pivot in highs_all) if highs_all else current
    last_low = min(pivot[1] for pivot in lows_all) if lows_all else current
    swing_range = last_high - last_low if last_high > last_low else current * 0.10

    fib_038 = swing_range * 0.382
    fib_050 = swing_range * 0.500
    fib_062 = swing_range * 0.618
    fib_100 = swing_range
    fib_162 = swing_range * 1.618

    def _build(onda: str, fase: str, prevision: str, pct_1m: float, pct_3m: float, pct_6m: float) -> dict[str, Any]:
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

    diffs = [abs(series[idx] - series[idx - 1]) for idx in range(max(1, len(series) - 20), len(series))]
    atr_pct = (sum(diffs) / len(diffs) / current * 100) if diffs and current else 1.5

    if len(pivots) < 3:
        total_change = ((current - series[0]) / series[0]) * 100
        if total_change > 15:
            up_3m = min(fib_162 / current * 100, 35)
            return _build("3↑", "Impulso alcista fuerte", "Posible extension de onda 3 o transicion a onda 4 correctiva.", up_3m * 0.3, up_3m * 0.65, up_3m)
        if total_change > 0:
            up_3m = min(fib_100 / current * 100, 20)
            return _build("1↑", "Inicio de impulso alcista", "Potencial continuacion alcista si confirma estructura.", up_3m * 0.25, up_3m * 0.55, up_3m)
        if total_change > -15:
            dn_1m = -fib_038 / current * 100
            return _build("A↓", "Inicio de correccion", "Posible rebote en onda B antes de completar correccion.", dn_1m, dn_1m * 0.5, fib_038 / current * 50)
        up_6m = fib_062 / current * 100
        return _build("C↓", "Correccion profunda", "Cerca de suelo si completa onda C. Oportunidad de entrada.", -atr_pct * 2, up_6m * 0.3, up_6m)

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
                dn = -fib_038 / current * 100
                return _build("1↑", "Impulso alcista inicial", "Esperar correccion de onda 2 para entrada. Alcista medio plazo.", dn * 0.5, dn, fib_050 / current * 100)
            up_6m = fib_162 / current * 100
            return _build("2↓", "Correccion dentro de impulso alcista", "Oportunidad de compra si respeta soporte. Onda 3 seria la mas fuerte.", -atr_pct, up_6m * 0.35, min(up_6m, 40))
        if num_swings <= 5:
            if last_pivot[2] == "H":
                up_6m = min(fib_162 / current * 100, 40)
                return _build("3↑", "Impulso alcista principal", "Onda mas fuerte en curso. Mantener posiciones, objetivo al alza.", up_6m * 0.25, up_6m * 0.6, up_6m)
            up_6m = fib_100 / current * 100
            return _build("4↓", "Correccion intermedia en tendencia alcista", "Correccion sana. Ultima oportunidad de compra antes de onda 5.", -atr_pct, up_6m * 0.4, min(up_6m, 30))
        if last_pivot[2] == "H":
            dn_6m = -fib_038 / current * 100
            return _build("5↑", "Ultimo impulso alcista", "Cercano a techo de ciclo. Considerar reducir posiciones gradualmente.", atr_pct * 0.5, dn_6m * 0.3, dn_6m)
        dn_3m = -fib_050 / current * 100
        return _build("A↓", "Inicio de correccion tras impulso completo", "Correccion ABC en curso. Rebote temporal posible en onda B.", dn_3m * 0.5, dn_3m, -fib_038 / current * 100)

    if lower_highs and lower_lows:
        if pct_from_high < -25:
            up_6m = fib_062 / current * 100
            return _build("C↓", "Correccion profunda bajista", "Posible suelo proximo. Zona de acumulacion para 2026-2027.", -atr_pct * 2, up_6m * 0.25, up_6m)
        if num_swings <= 3:
            dn_3m = -fib_050 / current * 100
            return _build("A↓", "Primera onda correctiva bajista", "Esperar rebote en onda B. No comprar aun, riesgo de mas caida.", dn_3m * 0.4, dn_3m, -fib_062 / current * 100)
        up_6m = fib_050 / current * 100
        return _build("B↑/C↓", "Correccion avanzada", "En fase final correctiva. Buscar senales de agotamiento para entrada.", -atr_pct, -fib_038 / current * 50, up_6m)

    if pct_from_high < -10:
        up_6m = fib_062 / current * 100
        return _build("4↓", "Consolidacion/correccion lateral", "Posible acumulacion. Ruptura al alza activaria onda 5 alcista.", -atr_pct, up_6m * 0.35, min(up_6m, 25))

    dn_6m = -fib_038 / current * 100
    return _build("B↑", "Rebote dentro de correccion", "Rebote tecnico. Confirmar si rompe maximos para cambio de tendencia.", atr_pct, -atr_pct * 2, dn_6m)


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