"""Evaluadores deterministas y veredicto consolidado."""

from __future__ import annotations

from typing import Any

from debate_portfolio import classify_elliott_wave, fmt_num, render_md_table, to_float


def technical_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> tuple[str, str]:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE ANALISIS TECNICO + ELLIOTT WAVES (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        data = prices.get(ticker, {})
        price = to_float(data.get("precio"))
        sma50 = to_float(data.get("sma50"))
        sma200 = to_float(data.get("sma200"))
        rsi = to_float(data.get("rsi14"))
        vol = to_float(data.get("vol_20d"))
        vol_ratio = to_float(data.get("vol_ratio"))
        bb_pct_b = to_float(data.get("bb_pct_b"))
        adx = to_float(data.get("adx"))
        rsi_div = data.get("rsi_divergence", "none")

        points = 0.0
        if price and sma50 and sma200:
            if price > sma50 > sma200:
                points += 25
            elif price > sma200:
                points += 15
            elif price > sma50:
                points += 8
        if rsi is not None:
            if 45 <= rsi <= 65:
                points += 20
            elif 40 <= rsi <= 70:
                points += 15
            elif 30 <= rsi <= 78:
                points += 8
            elif rsi < 30:
                points += 5
        if vol is not None:
            if vol <= 25:
                points += 10
            elif vol <= 35:
                points += 7
            elif vol <= 45:
                points += 4
        if vol_ratio is not None:
            if vol_ratio >= 1.2:
                points += 10
            elif vol_ratio >= 0.8:
                points += 6
            else:
                points += 2
        if bb_pct_b is not None:
            if 0.4 <= bb_pct_b <= 0.8:
                points += 15
            elif 0.2 <= bb_pct_b <= 0.95:
                points += 10
            elif bb_pct_b < 0.1:
                points += 5
        if adx is not None:
            if adx >= 25:
                points += 10
            elif adx >= 20:
                points += 6
            elif adx >= 15:
                points += 3
        if rsi_div == "bullish":
            points += 10
        elif rsi_div == "bearish":
            points -= 5

        score_num = min(100, max(0, round(points)))
        verdict = "OK" if score_num >= 60 else "NOK"
        if verdict == "OK":
            ok_count += 1

        reason = f"Score {score_num}/100 | RSI {fmt_num(rsi, 1)} | BB%B {fmt_num(bb_pct_b, 2)} | ADX {fmt_num(adx, 1)} | Div {rsi_div} | Vol {fmt_num(vol, 1)}%."
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Razon tecnica"], rows)
    print("\n[Tabla tecnica validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)

    print("\n--- ONDAS DE ELLIOTT ---", flush=True)
    ew_rows: list[list[str]] = []
    ew_ok_count = 0
    favorable_waves = {"1↑", "2↓", "3↑", "4↓", "C↓"}
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        data = prices.get(ticker, {})
        close_hist = data.get("close_hist", [])
        ew = classify_elliott_wave(close_hist)
        price = fmt_num(ew["precio_actual"], 2)
        t1m = f"{fmt_num(ew['target_1y'], 2)} ({ew['rent_1y']:+.1f}%)"
        t3m = f"{fmt_num(ew['target_3y'], 2)} ({ew['rent_3y']:+.1f}%)"
        t6m = f"{fmt_num(ew['target_6y'], 2)} ({ew['rent_6y']:+.1f}%)"
        ew_verdict = "OK" if (ew["onda"] in favorable_waves and ew["rent_6y"] > 0) else "NOK"
        if ew_verdict == "OK":
            ew_ok_count += 1
        ew_rows.append([str(idx), ticker, asset.get("nombre", ticker), ew["onda"], ew["fase"], price, t1m, t3m, t6m, ew_verdict, ew["prevision"]])

    ew_table = render_md_table(["#", "Ticker", "Nombre", "Onda", "Fase", "Precio", "Target 1y", "Target 3y", "Target 6y", "Veredicto", "Prevision"], ew_rows)
    print(ew_table, flush=True)
    print(f"\nResultado Elliott: {ew_ok_count} OK / {len(top10_assets) - ew_ok_count} NOK de {len(top10_assets)} activos.", flush=True)

    return table + "\n\n" + ew_table, ew_table


def fundamental_analysis_review(top10_assets: list[dict[str, Any]], fundamentals: dict[str, dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE ANALISIS FUNDAMENTAL (determinista)", flush=True)
    print("=" * 60, flush=True)

    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        asset_type = asset.get("tipo", "Accion")
        fundamental_row = fundamentals.get(ticker, {})
        per = to_float(fundamental_row.get("per"))
        forward_pe = to_float(fundamental_row.get("forward_pe"))
        peg = to_float(fundamental_row.get("peg"))
        roe = to_float(fundamental_row.get("roe"))
        debt_equity = to_float(fundamental_row.get("deuda_equity"))
        growth = to_float(fundamental_row.get("crec_ingresos"))
        target_mean = to_float(fundamental_row.get("target_mean_price"))
        current_price = to_float(fundamental_row.get("current_price"))

        if asset_type == "Accion":
            points = 0.0
            eff_pe = forward_pe if forward_pe else per
            if eff_pe is not None:
                if eff_pe <= 15:
                    points += 20
                elif eff_pe <= 25:
                    points += 16
                elif eff_pe <= 35:
                    points += 10
                elif eff_pe <= 50:
                    points += 4
            if peg is not None:
                if peg <= 1.0:
                    points += 12
                elif peg <= 1.5:
                    points += 8
                elif peg <= 2.5:
                    points += 5
            if roe is not None:
                if roe >= 0.25:
                    points += 20
                elif roe >= 0.15:
                    points += 16
                elif roe >= 0.10:
                    points += 10
                elif roe >= 0.05:
                    points += 4
            if debt_equity is not None:
                if debt_equity <= 50:
                    points += 16
                elif debt_equity <= 100:
                    points += 12
                elif debt_equity <= 180:
                    points += 6
            if growth is not None:
                if growth > 0.20:
                    points += 12
                elif growth > 0.10:
                    points += 8
                elif growth > 0:
                    points += 4
            upside_pct = None
            if target_mean and current_price and current_price > 0:
                upside_pct = ((target_mean / current_price) - 1) * 100
                if upside_pct >= 30:
                    points += 20
                elif upside_pct >= 15:
                    points += 15
                elif upside_pct >= 5:
                    points += 10
                elif upside_pct >= 0:
                    points += 5
            score_num = min(100, max(0, round(points)))
            verdict = "OK" if score_num >= 55 else "NOK"
            peg_str = f"{peg:.2f}" if peg else "N/D"
            fpe_str = f"{forward_pe:.1f}" if forward_pe else "N/D"
            up_str = f"{upside_pct:+.1f}%" if upside_pct is not None else "N/D"
            reason = f"Score {score_num}/100 | FwdPE {fpe_str} | PEG {peg_str} | ROE {roe if roe is not None else 'N/D'} | Upside {up_str}."
        else:
            data = prices.get(ticker, {})
            ret_12m = to_float(data.get("ret_12m"))
            ret_3m = to_float(data.get("ret_3m"))
            vol = to_float(data.get("vol_20d"))
            sharpe = to_float(data.get("sharpe_6m"))
            points = 0.0
            if ret_12m is not None:
                if ret_12m > 20:
                    points += 35
                elif ret_12m > 10:
                    points += 25
                elif ret_12m > 0:
                    points += 15
                elif ret_12m > -10:
                    points += 5
            if ret_3m is not None:
                if ret_3m > 5:
                    points += 25
                elif ret_3m > 0:
                    points += 18
                elif ret_3m > -5:
                    points += 8
            if vol is not None:
                if vol <= 20:
                    points += 20
                elif vol <= 35:
                    points += 15
                elif vol <= 50:
                    points += 8
            if sharpe is not None:
                if sharpe >= 2.0:
                    points += 20
                elif sharpe >= 1.0:
                    points += 15
                elif sharpe >= 0.5:
                    points += 8
                elif sharpe >= 0:
                    points += 3
            score_num = min(100, max(0, round(points)))
            verdict = "OK" if score_num >= 55 else "NOK"
            reason = f"Score {score_num}/100 | Ret12m {ret_12m if ret_12m is not None else 'N/D'}% | Ret3m {ret_3m if ret_3m is not None else 'N/D'}% | Sharpe {fmt_num(sharpe, 2)}."

        if verdict == "OK":
            ok_count += 1
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset_type, f"{asset['peso']}%", verdict, reason])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Razon fundamental"], rows)
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
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        data = prices.get(ticker, {})
        vol = to_float(data.get("vol_20d"))
        max_dd = to_float(data.get("max_dd_12m"))
        dist = to_float(data.get("dist_high_52w"))
        sharpe = to_float(data.get("sharpe_6m"))
        points = 0.0
        if vol is not None:
            if vol <= 20:
                points += 30
            elif vol <= 30:
                points += 25
            elif vol <= 40:
                points += 18
            elif vol <= 50:
                points += 8
        if max_dd is not None:
            if max_dd >= -15:
                points += 30
            elif max_dd >= -25:
                points += 22
            elif max_dd >= -35:
                points += 15
            elif max_dd >= -45:
                points += 8
        if dist is not None:
            if dist >= -10:
                points += 20
            elif dist >= -20:
                points += 15
            elif dist >= -30:
                points += 8
        if sharpe is not None:
            if sharpe >= 1.5:
                points += 20
            elif sharpe >= 1.0:
                points += 15
            elif sharpe >= 0.5:
                points += 10
            elif sharpe >= 0:
                points += 5

        score_num = min(100, max(0, round(points)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1
        reason = f"Score {score_num}/100 | Vol {fmt_num(vol, 1)}% | MaxDD {fmt_num(max_dd, 1)}% | Dist52w {fmt_num(dist, 1)}% | Sharpe {fmt_num(sharpe, 2)}."
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Riesgo principal"], rows)
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
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        data = prices.get(ticker, {})
        ret_1m = to_float(data.get("ret_1m"))
        ret_3m = to_float(data.get("ret_3m"))
        ret_12m = to_float(data.get("ret_12m"))
        rs = to_float(data.get("rs_vs_sp500"))
        momentum = to_float(data.get("momentum_score"))
        points = 0.0
        if ret_1m is not None:
            if ret_1m > 5:
                points += 20
            elif ret_1m > 0:
                points += 15
            elif ret_1m > -5:
                points += 10
            elif ret_1m > -10:
                points += 5
        if ret_3m is not None:
            if ret_3m > 10:
                points += 20
            elif ret_3m > 0:
                points += 15
            elif ret_3m > -5:
                points += 8
        if ret_12m is not None:
            if ret_12m > 20:
                points += 20
            elif ret_12m > 5:
                points += 15
            elif ret_12m > 0:
                points += 8
        if rs is not None:
            if rs > 10:
                points += 20
            elif rs > 0:
                points += 15
            elif rs > -10:
                points += 8
        if momentum is not None:
            if momentum > 15:
                points += 20
            elif momentum > 5:
                points += 15
            elif momentum > 0:
                points += 10
            elif momentum > -5:
                points += 3
        score_num = min(100, max(0, round(points)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1
        reason = f"Score {score_num}/100 | 1m {fmt_num(ret_1m, 1)}% | 3m {fmt_num(ret_3m, 1)}% | 12m {fmt_num(ret_12m, 1)}% | RS {fmt_num(rs, 1)} | Mom {fmt_num(momentum, 1)}."
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Sentimiento clave"], rows)
    print("\n[Tabla de sentimiento validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def macd_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION DE MACD (determinista)", flush=True)
    print("=" * 60, flush=True)
    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        data = prices.get(ticker, {})
        macd = to_float(data.get("macd"))
        signal = to_float(data.get("macd_signal"))
        hist = to_float(data.get("macd_hist"))
        points = 0.0
        if macd is not None and signal is not None and hist is not None:
            if macd > signal:
                points += 40
                if hist > 0:
                    points += 10
            elif hist > -0.5:
                points += 15
            if macd > 0:
                points += 25
            elif macd > -1:
                points += 10
            if hist > 0:
                points += 15
            elif hist > -0.5:
                points += 5
        score_num = min(100, max(0, round(points)))
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1
        reason = f"Score {score_num}/100 | MACD {fmt_num(macd, 4)} | Signal {fmt_num(signal, 4)} | Hist {fmt_num(hist, 4)}."
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset.get("tipo", "Accion"), f"{asset['peso']}%", verdict, reason])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "MACD detalle"], rows)
    print("\n[Tabla MACD validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def institutional_analysis_review(top10_assets: list[dict[str, Any]], fundamentals: dict[str, dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION INSTITUCIONAL (determinista)", flush=True)
    print("=" * 60, flush=True)
    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        asset_type = asset.get("tipo", "Accion")
        fundamental_row = fundamentals.get(ticker, {})
        data = prices.get(ticker, {})

        rec_mean = to_float(fundamental_row.get("recommendation_mean"))
        rec_key = fundamental_row.get("recommendation_key", "")
        num_analysts = to_float(fundamental_row.get("num_analysts"))
        target_mean = to_float(fundamental_row.get("target_mean_price"))
        current_price = to_float(fundamental_row.get("current_price"))
        inst_pct = to_float(fundamental_row.get("institutional_pct"))
        short_ratio = to_float(fundamental_row.get("short_ratio"))
        short_pct = to_float(fundamental_row.get("short_pct_float"))
        points = 0.0

        if asset_type == "Accion":
            if rec_mean is not None:
                if rec_mean <= 1.5:
                    points += 30
                elif rec_mean <= 2.0:
                    points += 25
                elif rec_mean <= 2.5:
                    points += 18
                elif rec_mean <= 3.0:
                    points += 10
                elif rec_mean <= 3.5:
                    points += 5
            if num_analysts is not None:
                if num_analysts >= 20:
                    points += 10
                elif num_analysts >= 10:
                    points += 7
                elif num_analysts >= 5:
                    points += 4
            upside = None
            if target_mean and current_price and current_price > 0:
                upside = ((target_mean / current_price) - 1) * 100
                if upside >= 30:
                    points += 25
                elif upside >= 15:
                    points += 20
                elif upside >= 5:
                    points += 12
                elif upside >= 0:
                    points += 5
            if inst_pct is not None:
                if inst_pct >= 0.70:
                    points += 15
                elif inst_pct >= 0.50:
                    points += 10
                elif inst_pct >= 0.30:
                    points += 5
            if short_pct is not None:
                if short_pct <= 0.02:
                    points += 20
                elif short_pct <= 0.05:
                    points += 15
                elif short_pct <= 0.10:
                    points += 8
                elif short_pct <= 0.15:
                    points += 3
            elif short_ratio is not None:
                if short_ratio <= 2:
                    points += 15
                elif short_ratio <= 5:
                    points += 8
        else:
            momentum = to_float(data.get("momentum_score"))
            vol_ratio = to_float(data.get("vol_ratio"))
            sharpe = to_float(data.get("sharpe_6m"))
            if momentum is not None:
                if momentum > 15:
                    points += 40
                elif momentum > 5:
                    points += 30
                elif momentum > 0:
                    points += 20
                elif momentum > -5:
                    points += 10
            if vol_ratio is not None:
                if vol_ratio >= 1.5:
                    points += 30
                elif vol_ratio >= 1.0:
                    points += 20
                elif vol_ratio >= 0.7:
                    points += 10
            if sharpe is not None:
                if sharpe >= 1.5:
                    points += 30
                elif sharpe >= 0.5:
                    points += 20
                elif sharpe >= 0:
                    points += 10

        score_num = min(100, max(0, round(points)))
        verdict = "OK" if score_num >= 50 else "NOK"
        if verdict == "OK":
            ok_count += 1
        rec_str = f"{rec_mean:.1f}({rec_key})" if rec_mean is not None else "N/D"
        n_str = f"{int(num_analysts)}" if num_analysts is not None else "N/D"
        upside = None
        if target_mean and current_price and current_price > 0:
            upside = ((target_mean / current_price) - 1) * 100
        up_str = f"{upside:+.1f}%" if upside is not None else "N/D"
        inst_str = f"{inst_pct * 100:.0f}%" if inst_pct else "N/D"
        short_str = f"{short_pct * 100:.1f}%" if short_pct else "N/D"
        name = asset.get("nombre", ticker)
        rows.append([str(idx), ticker, name, asset_type, f"{asset['peso']}%", rec_str, n_str, up_str, inst_str, short_str, verdict])

    table = render_md_table(["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Consenso", "Analistas", "Upside", "Institucional", "Short%", "Veredicto"], rows)
    print("\n[Tabla institucional validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


# ---------------------------------------------------------------------------
# Evaluador Wyckoff (precio-volumen, fases de acumulacion/distribucion)
# ---------------------------------------------------------------------------

def _detect_wyckoff_phase(close_hist: list[float], volume_hist: list[float]) -> dict[str, Any]:
    """Clasifica la fase Wyckoff de un activo usando precio y volumen.

    Fases: Acumulacion, Markup, Distribucion, Markdown, Indefinida.
    Detecta springs (caida bajo soporte + recuperacion) y upthrusts.
    """
    result: dict[str, Any] = {
        "fase": "Indefinida",
        "spring": False,
        "upthrust": False,
        "vol_confirmacion": False,
        "score": 0,
        "detalle": "Datos insuficientes",
    }

    min_len = 60  # ~3 meses de datos diarios
    if len(close_hist) < min_len:
        return result

    # Usar ultimos 120 dias (o lo que haya)
    window = min(120, len(close_hist))
    prices = close_hist[-window:]
    volumes = volume_hist[-window:] if len(volume_hist) >= window else volume_hist[-len(volume_hist):]

    # Si no hay volumen suficiente, analizar solo precio
    has_volume = len(volumes) >= len(prices) // 2 and any(v > 0 for v in volumes if v is not None)

    # Dividir en 3 tercios para detectar fase
    third = len(prices) // 3
    p1 = prices[:third]       # primer tercio (historico)
    p2 = prices[third:2*third]  # segundo tercio (medio)
    p3 = prices[2*third:]      # tercer tercio (reciente)

    avg1, avg2, avg3 = sum(p1)/len(p1), sum(p2)/len(p2), sum(p3)/len(p3)
    high_range = max(prices)
    low_range = min(prices)
    rango = high_range - low_range if high_range != low_range else 1.0
    rango_pct = (rango / low_range * 100) if low_range > 0 else 0

    # Volatilidad del ultimo tercio vs primer tercio
    std1 = (sum((p - avg1)**2 for p in p1) / len(p1)) ** 0.5
    std3 = (sum((p - avg3)**2 for p in p3) / len(p3)) ** 0.5

    # Volumen: comparar promedio del primer vs tercer tercio
    vol_increasing = False
    vol_decreasing = False
    if has_volume:
        v1 = volumes[:third]
        v3 = volumes[2*third:]
        avg_v1 = sum(v for v in v1 if v) / max(len(v1), 1)
        avg_v3 = sum(v for v in v3 if v) / max(len(v3), 1)
        if avg_v1 > 0:
            vol_ratio = avg_v3 / avg_v1
            vol_increasing = vol_ratio > 1.3
            vol_decreasing = vol_ratio < 0.7

    # Detectar spring: caida bajo minimo reciente seguida de recuperacion
    recent_20 = prices[-20:]
    recent_low = min(recent_20[:15]) if len(recent_20) >= 15 else min(recent_20)
    if len(recent_20) >= 15:
        # Spring: precio cae bajo soporte y cierra por encima
        dip = min(recent_20[10:15])
        if dip < recent_low * 0.98 and recent_20[-1] > recent_low:
            result["spring"] = True

    # Detectar upthrust: subida sobre maximo reciente seguida de caida
    recent_high = max(recent_20[:15]) if len(recent_20) >= 15 else max(recent_20)
    if len(recent_20) >= 15:
        peak = max(recent_20[10:15])
        if peak > recent_high * 1.02 and recent_20[-1] < recent_high:
            result["upthrust"] = True

    # Clasificacion de fase
    price_trend_up = avg3 > avg2 > avg1
    price_trend_down = avg3 < avg2 < avg1
    price_lateral = rango_pct < 20 and abs(avg3 - avg1) / avg1 < 0.05

    if price_lateral and std3 < std1:
        # Rango estrecho, volatilidad decreciente → posible acumulacion
        if vol_decreasing or not has_volume:
            result["fase"] = "Acumulacion"
        else:
            result["fase"] = "Distribucion" if vol_increasing else "Acumulacion"
    elif price_trend_up:
        if vol_increasing:
            result["fase"] = "Markup"
            result["vol_confirmacion"] = True
        else:
            result["fase"] = "Markup"
    elif price_trend_down:
        if vol_increasing:
            result["fase"] = "Markdown"
            result["vol_confirmacion"] = True
        else:
            result["fase"] = "Markdown"
    elif avg3 > avg1 and rango_pct >= 20:
        result["fase"] = "Markup"
    elif avg3 < avg1 and rango_pct >= 20:
        result["fase"] = "Markdown"
    else:
        # Rango lateral amplio
        result["fase"] = "Distribucion" if avg3 < avg2 else "Acumulacion"

    # Puntuacion Wyckoff (0-100)
    score = 0.0
    fase = result["fase"]

    # Fases favorables para compra
    if fase == "Acumulacion":
        score += 35
    elif fase == "Markup":
        score += 30
    elif fase == "Distribucion":
        score += 10
    elif fase == "Markdown":
        score += 5

    # Spring es senal muy alcista
    if result["spring"]:
        score += 25
    # Upthrust es senal bajista
    if result["upthrust"]:
        score -= 15

    # Confirmacion de volumen
    if result["vol_confirmacion"] and fase in ("Markup", "Acumulacion"):
        score += 20
    elif result["vol_confirmacion"] and fase in ("Markdown", "Distribucion"):
        score -= 10

    # Precio actual vs soporte/resistencia del rango
    current = prices[-1]
    mid_range = (high_range + low_range) / 2
    if current > mid_range:
        score += 10  # por encima de la mitad del rango
    if current > high_range * 0.95:
        score += 5  # cerca del maximo

    score = min(100, max(0, round(score)))
    result["score"] = score

    vol_str = "con vol" if result["vol_confirmacion"] else "sin conf vol"
    spring_str = " | Spring detectado" if result["spring"] else ""
    upthrust_str = " | Upthrust detectado" if result["upthrust"] else ""
    result["detalle"] = f"{fase} ({vol_str}){spring_str}{upthrust_str} | Rango {rango_pct:.1f}%"

    return result


def wyckoff_analysis_review(top10_assets: list[dict[str, Any]], prices: dict[str, dict[str, Any]]) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION WYCKOFF (determinista - precio/volumen)", flush=True)
    print("=" * 60, flush=True)
    total = len(top10_assets)
    rows: list[list[str]] = []
    ok_count = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        data = prices.get(ticker, {})
        close_hist = data.get("close_hist", [])
        volume_hist = data.get("volume_hist", [])

        wk = _detect_wyckoff_phase(close_hist, volume_hist)
        score_num = wk["score"]
        verdict = "OK" if score_num >= 55 else "NOK"
        if verdict == "OK":
            ok_count += 1

        name = asset.get("nombre", ticker)
        rows.append([
            str(idx), ticker, name, asset.get("tipo", "Accion"),
            f"{asset['peso']}%", wk["fase"],
            "Si" if wk["spring"] else "No",
            "Si" if wk["upthrust"] else "No",
            str(score_num), verdict, wk["detalle"],
        ])

    table = render_md_table(
        ["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Fase Wyckoff", "Spring", "Upthrust", "Score", "Veredicto", "Detalle"],
        rows,
    )
    print("\n[Tabla Wyckoff validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


# ---------------------------------------------------------------------------
# Evaluador de Analisis Relativo / Intermarket  (Percentile Cross-Sectional)
# ---------------------------------------------------------------------------

def _classify_asset_group(tipo: str) -> str:
    """Agrupa tipo de activo para seleccion de benchmark."""
    tipo_upper = tipo.upper()
    if tipo_upper in ("ACCION", "ETF", "INDICE"):
        return "RV"
    elif tipo_upper == "COMMODITY":
        return "CMD"
    elif tipo_upper == "CRIPTO":
        return "CRYPTO"
    else:
        return "RV"  # default


# Benchmarks por grupo de activo (ticker yfinance)
_GROUP_BENCHMARKS: dict[str, str] = {
    "RV": "^GSPC",       # S&P 500
    "CMD": "DJP",         # iPath Bloomberg Commodity ETN
    "CRYPTO": "BTC-USD",  # Bitcoin como benchmark cripto
}


def _percentile_rank(value: float, population: list[float]) -> float:
    """Devuelve el percentil (0-100) de *value* dentro de *population*."""
    if not population:
        return 50.0  # sin datos → neutral
    below = sum(1 for v in population if v < value)
    equal = sum(1 for v in population if v == value)
    return ((below + 0.5 * equal) / len(population)) * 100


def relative_analysis_review(
    top10_assets: list[dict[str, Any]],
    prices: dict[str, dict[str, Any]],
    fundamentals: dict[str, dict[str, Any]],
) -> str:
    print("\n" + "=" * 60, flush=True)
    print("REVISION ANALISIS RELATIVO / INTERMARKET (percentil cross-sectional)", flush=True)
    print("=" * 60, flush=True)
    total = len(top10_assets)

    # ------------------------------------------------------------------
    # 1. Recopilar vectores de la cartera para calcular percentiles
    # ------------------------------------------------------------------
    port_rs: list[float] = []
    port_momentum: list[float] = []
    port_sharpe: list[float] = []
    port_accel: list[float] = []      # ann_1m − ann_3m
    port_vol_ratio: list[float] = []

    for asset in top10_assets:
        d = prices.get(asset["ticker"], {})
        rs = to_float(d.get("rs_vs_sp500"))
        mom = to_float(d.get("momentum_score"))
        sh = to_float(d.get("sharpe_6m"))
        vr = to_float(d.get("vol_ratio"))
        r1 = to_float(d.get("ret_1m"))
        r3 = to_float(d.get("ret_3m"))
        if rs is not None:
            port_rs.append(rs)
        if mom is not None:
            port_momentum.append(mom)
        if sh is not None:
            port_sharpe.append(sh)
        if vr is not None:
            port_vol_ratio.append(vr)
        if r1 is not None and r3 is not None:
            port_accel.append(r1 * 12 - r3 * 4)

    # Retornos de benchmarks por grupo (para RS por tipo de activo)
    bench_ret_12m: dict[str, float | None] = {}
    for grp, bench_ticker in _GROUP_BENCHMARKS.items():
        bd = prices.get(bench_ticker, {})
        bench_ret_12m[grp] = to_float(bd.get("ret_12m"))

    # ------------------------------------------------------------------
    # 2. Evaluar cada activo con percentil cross-sectional
    # ------------------------------------------------------------------
    # Pesos de cada dimension
    W_RS = 0.25          # RS vs benchmark apropiado
    W_RS_TREND = 0.15    # Tendencia de RS (aceleracion del exceso de retorno)
    W_MOMENTUM = 0.20    # Momentum compuesto
    W_SHARPE = 0.15      # Sharpe relativo
    W_ACCEL = 0.15       # Aceleracion del momentum
    W_VOLUME = 0.10      # Volumen relativo (conviction)

    rows: list[list[str]] = []
    ok_count = 0

    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        print(f"[Evaluando {idx}/{total}: {ticker}]", flush=True)
        d = prices.get(ticker, {})
        fund = fundamentals.get(ticker, {})

        ret_1m = to_float(d.get("ret_1m"))
        ret_3m = to_float(d.get("ret_3m"))
        ret_6m = to_float(d.get("ret_6m"))
        ret_12m = to_float(d.get("ret_12m"))
        rs_sp500 = to_float(d.get("rs_vs_sp500"))
        sharpe = to_float(d.get("sharpe_6m"))
        momentum = to_float(d.get("momentum_score"))
        vol_ratio = to_float(d.get("vol_ratio"))
        tipo = asset.get("tipo", d.get("tipo", "Accion"))

        # --- A) RS vs benchmark apropiado por tipo de activo (25%) ----
        grupo = _classify_asset_group(tipo)
        bench_r12 = bench_ret_12m.get(grupo)
        rs_vs_bench: float | None = None
        if ret_12m is not None and bench_r12 is not None:
            rs_vs_bench = ret_12m - bench_r12
        elif rs_sp500 is not None:
            rs_vs_bench = rs_sp500  # fallback a S&P 500

        pct_rs = _percentile_rank(rs_vs_bench, port_rs) if rs_vs_bench is not None else 50.0

        # --- B) Tendencia de RS: ¿mejorando o empeorando? (15%) -------
        # Compara RS corto plazo (3m) vs RS largo plazo (12m)
        rs_trend: float | None = None
        if ret_3m is not None and ret_12m is not None:
            sp_d = prices.get("^GSPC", {})
            sp_3m = to_float(sp_d.get("ret_3m"))
            sp_12m = to_float(sp_d.get("ret_12m"))
            if sp_3m is not None and sp_12m is not None:
                rs_short = ret_3m - sp_3m
                rs_long = ret_12m - sp_12m
                rs_trend = rs_short - rs_long  # positivo = mejorando

        # Sin poblacion especifica, normalizamos: >5 excelente, <-5 pobre
        if rs_trend is not None:
            pct_rs_trend = min(100, max(0, 50 + rs_trend * 5))
        else:
            pct_rs_trend = 50.0

        # --- C) Momentum compuesto (20%) -------------------------------
        pct_momentum = _percentile_rank(momentum, port_momentum) if momentum is not None else 50.0

        # --- D) Sharpe relativo (15%) ----------------------------------
        pct_sharpe = _percentile_rank(sharpe, port_sharpe) if sharpe is not None else 50.0

        # --- E) Aceleracion del momentum (15%) -------------------------
        accel_val: float | None = None
        if ret_1m is not None and ret_3m is not None:
            accel_val = ret_1m * 12 - ret_3m * 4
        pct_accel = _percentile_rank(accel_val, port_accel) if accel_val is not None else 50.0

        # --- F) Volumen relativo / conviction (10%) --------------------
        # vol_ratio > 1 = volumen superior a la media → confirma movimiento
        pct_vol = 50.0
        if vol_ratio is not None:
            if port_vol_ratio:
                pct_vol = _percentile_rank(vol_ratio, port_vol_ratio)
            else:
                # sin peers, usamos 1.0 como neutro
                pct_vol = min(100, max(0, vol_ratio * 50))

        # --- Score compuesto (media ponderada de percentiles) ----------
        score = (
            W_RS * pct_rs
            + W_RS_TREND * pct_rs_trend
            + W_MOMENTUM * pct_momentum
            + W_SHARPE * pct_sharpe
            + W_ACCEL * pct_accel
            + W_VOLUME * pct_vol
        )
        score_num = min(100, max(0, round(score)))

        verdict = "OK" if score_num >= 50 else "NOK"
        if verdict == "OK":
            ok_count += 1

        # --- Formato de salida -----------------------------------------
        rs_str = f"{rs_vs_bench:+.1f}%" if rs_vs_bench is not None else "N/D"
        trend_str = f"{rs_trend:+.1f}" if rs_trend is not None else "N/D"
        accel_str = ""
        if accel_val is not None:
            accel_str = "Accel" if accel_val > 0 and ret_1m and ret_1m > 0 else "Decel"
        vol_str = f"{vol_ratio:.2f}" if vol_ratio is not None else "N/D"

        name = asset.get("nombre", ticker)
        bench_label = _GROUP_BENCHMARKS.get(grupo, "^GSPC").replace("^", "")
        reason = (
            f"Score {score_num}/100 | RS vs {bench_label} {rs_str} "
            f"| Tend {trend_str} | {accel_str} | Sharpe p{round(pct_sharpe)} "
            f"| Vol {vol_str}"
        )
        rows.append([
            str(idx), ticker, name, asset.get("tipo", "Accion"),
            f"{asset['peso']}%", verdict, reason,
        ])

    table = render_md_table(
        ["#", "Ticker", "Nombre", "Tipo", "%Cartera", "Veredicto", "Analisis relativo"],
        rows,
    )
    print("\n[Tabla analisis relativo validada]", flush=True)
    print(table, flush=True)
    print(f"\nResultado: {ok_count} OK / {len(top10_assets) - ok_count} NOK de {len(top10_assets)} activos.", flush=True)
    return table


def parse_verdict_table(table: str) -> dict[str, str]:
    verdicts: dict[str, str] = {}
    lines = [line.strip() for line in table.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return verdicts
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    hmap = {header.upper(): idx for idx, header in enumerate(headers)}
    tidx = hmap.get("TICKER")
    vidx = hmap.get("VEREDICTO")
    if tidx is None or vidx is None:
        return verdicts
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) <= max(tidx, vidx):
            continue
        ticker = cells[tidx].upper()
        verdict = cells[vidx].upper()
        if ticker and verdict in {"OK", "NOK"}:
            verdicts[ticker] = verdict
    return verdicts


def extract_evitar_tickers(verdict_output: str) -> set[str]:
    evitar: set[str] = set()
    lines = [line.strip() for line in verdict_output.splitlines() if line.strip().startswith("|")]
    if len(lines) < 3:
        return evitar
    headers = [cell.strip().upper() for cell in lines[0].strip("|").split("|")]
    hmap = {header: idx for idx, header in enumerate(headers)}
    tidx = hmap.get("TICKER")
    didx = hmap.get("DECISION FINAL")
    if tidx is None or didx is None:
        return evitar
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) <= max(tidx, didx):
            continue
        ticker = cells[tidx].upper()
        decision = cells[didx].upper()
        if ticker and "EVITAR" in decision:
            evitar.add(ticker)
    return evitar


def final_verdict(top10_assets: list[dict[str, Any]], ta_table: str, fa_table: str, risk_table: str, sent_table: str, ew_table: str, macd_table: str, inst_table: str, wyckoff_table: str, relative_table: str) -> str:
    print("\n" + "=" * 60, flush=True)
    print("VEREDICTO FINAL CONSOLIDADO", flush=True)
    print("=" * 60, flush=True)

    ta_map = parse_verdict_table(ta_table)
    fa_map = parse_verdict_table(fa_table)
    risk_map = parse_verdict_table(risk_table)
    sent_map = parse_verdict_table(sent_table)
    ew_map = parse_verdict_table(ew_table)
    macd_map = parse_verdict_table(macd_table)
    inst_map = parse_verdict_table(inst_table)
    wyckoff_map = parse_verdict_table(wyckoff_table)
    relative_map = parse_verdict_table(relative_table)

    rows: list[list[str]] = []
    buy = watch = avoid = 0
    for idx, asset in enumerate(top10_assets, start=1):
        ticker = asset["ticker"]
        name = asset.get("nombre", ticker)
        tech = ta_map.get(ticker, "NOK")
        fund = fa_map.get(ticker, "NOK")
        risk = risk_map.get(ticker, "NOK")
        sent = sent_map.get(ticker, "NOK")
        elliott = ew_map.get(ticker, "NOK")
        macd = macd_map.get(ticker, "NOK")
        inst = inst_map.get(ticker, "NOK")
        wyckoff = wyckoff_map.get(ticker, "NOK")
        relative = relative_map.get(ticker, "NOK")
        ok_total = [tech, fund, risk, sent, elliott, macd, inst, wyckoff, relative].count("OK")
        if ok_total >= 8:
            decision = "COMPRAR"
            buy += 1
        elif ok_total >= 5:
            decision = "VIGILAR"
            watch += 1
        else:
            decision = "EVITAR"
            avoid += 1
        rows.append([str(idx), ticker, name, f"{asset['peso']}%", tech, fund, risk, sent, elliott, macd, inst, wyckoff, relative, decision])

    table = render_md_table(["#", "Ticker", "Nombre", "%Cartera", "Tecnico", "Fundamental", "Riesgo", "Sentimiento", "Elliott", "MACD", "Institucional", "Wyckoff", "Relativo", "DECISION FINAL"], rows)
    reco_lines = [
        f"Priorizar activos COMPRAR ({buy} de {len(top10_assets)}) con entradas escalonadas.",
        f"Mantener en observacion activos VIGILAR ({watch} de {len(top10_assets)}) y revisar en cada cierre semanal.",
        f"Reducir exposicion en EVITAR ({avoid} de {len(top10_assets)}) hasta mejora de senales tecnicas y riesgo.",
    ]
    final_output = table + "\n\nRECOMENDACIONES FINALES:\n" + "\n".join(reco_lines)
    print("\n[Veredicto final validado]", flush=True)
    print(final_output, flush=True)
    return final_output