"""Descarga y analiza las carteras reales (13F filings) de los gurús inversores.

Fuentes:
 - SEC EDGAR API (gratis, sin API key) para 13F-HR filings trimestrales.
 - ARK Invest holdings diarios (CSV público).

Gurús disponibles (con formulario 13F):
 - Warren Buffett → Berkshire Hathaway Inc (CIK 1067983)
 - Stanley Druckenmiller → Duquesne Family Office LLC (CIK 1536411)
 - Ray Dalio → Bridgewater Associates LP (CIK 1350694)
 - Cathie Wood → ARK Investment Management LLC (CIK 1697748)
 - Howard Marks → Oaktree Capital Management LP (CIK 1403528)

No disponibles (sin fondo público): Peter Lynch (retirado), Jim Rogers.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent / ".cache"
_CACHE_TTL = 86_400 * 7  # 7 días (los 13F son trimestrales)

_SEC_BASE = "https://data.sec.gov"
_SEC_HEADERS = {
    "User-Agent": "MacroDebateApp/1.0 (ofis20@hotmail.com)",
    "Accept-Encoding": "gzip, deflate",
}

# CIKs oficiales de SEC EDGAR (10 dígitos con padding)
GURU_FUNDS: dict[str, dict[str, Any]] = {
    "Warren Buffett": {
        "fund": "Berkshire Hathaway Inc",
        "cik": "0001067983",
    },
    "Stanley Druckenmiller": {
        "fund": "Duquesne Family Office LLC",
        "cik": "0001536411",
    },
    "Ray Dalio": {
        "fund": "Bridgewater Associates LP",
        "cik": "0001350694",
    },
    "Cathie Wood": {
        "fund": "ARK Investment Management LLC",
        "cik": "0001697748",
    },
    "Howard Marks": {
        "fund": "Oaktree Capital Management LP",
        "cik": "0001403528",
    },
}

# Namespace del XML de 13F (detectado dinámicamente)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", key)
    return _CACHE_DIR / f"guru_{safe}.pkl"


def _cache_get(key: str) -> Any | None:
    p = _cache_path(key)
    if p.exists():
        age = time.time() - p.stat().st_mtime
        if age < _CACHE_TTL:
            with open(p, "rb") as f:
                return pickle.load(f)
    return None


def _cache_set(key: str, data: Any) -> None:
    p = _cache_path(key)
    with open(p, "wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# SEC EDGAR: obtener el último 13F-HR filing
# ---------------------------------------------------------------------------

def _sec_get(url: str) -> requests.Response:
    """GET con rate-limiting SEC (10 req/s max)."""
    time.sleep(0.12)  # ~8 req/s para estar seguros
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def _get_latest_13f_url(cik: str) -> str | None:
    """Busca el filing 13F-HR más reciente y devuelve la URL del infotable XML."""
    url = f"{_SEC_BASE}/submissions/CIK{cik}.json"
    data = _sec_get(url).json()

    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])

    cik_num = cik.lstrip("0")
    for i, form in enumerate(forms):
        if form in ("13F-HR", "13F-HR/A"):
            acc = accessions[i].replace("-", "")
            base = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}"
            # Buscar el infotable XML en el filing index
            idx_url = f"{base}/index.json"
            try:
                idx = _sec_get(idx_url).json()
                items = idx.get("directory", {}).get("item", [])
                # El infotable es el XML que NO es primary_doc.xml
                xml_files = [
                    it["name"] for it in items
                    if it["name"].endswith(".xml")
                    and "primary" not in it["name"].lower()
                ]
                if xml_files:
                    return f"{base}/{xml_files[0]}"
                # Fallback: cualquier XML
                any_xml = [it["name"] for it in items if it["name"].endswith(".xml")]
                if any_xml:
                    return f"{base}/{any_xml[0]}"
            except Exception:
                pass
            break

    return None


def _parse_13f_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parsea el XML infotable del 13F y extrae las posiciones."""
    root = ET.fromstring(xml_text)

    # Detectar namespace dinámicamente desde el tag raíz
    tag_match = re.search(r"\{([^}]+)\}", root.tag or "")
    ns = tag_match.group(1) if tag_match else ""
    prefix = f"{{{ns}}}" if ns else ""

    holdings: list[dict[str, Any]] = []
    # Agrupar por CUSIP para sumar posiciones del mismo emisor
    cusip_agg: dict[str, dict[str, Any]] = {}

    for entry in root.findall(f".//{prefix}infoTable"):
        cusip = _xml_text(entry, f"{prefix}cusip")
        name = _xml_text(entry, f"{prefix}nameOfIssuer")
        value = _xml_float(entry, f"{prefix}value")  # en USD
        shares_el = entry.find(f"{prefix}shrsOrPrnAmt")
        shares = _xml_float(shares_el, f"{prefix}sshPrnamt") if shares_el is not None else 0

        if name and cusip:
            key = cusip.strip().upper()
            if key in cusip_agg:
                cusip_agg[key]["value_thousands"] += value
                cusip_agg[key]["shares"] += shares
            else:
                cusip_agg[key] = {
                    "name": name.strip(),
                    "cusip": key,
                    "value_thousands": value,
                    "shares": shares,
                }

    return list(cusip_agg.values())


def _xml_text(el: ET.Element | None, tag: str) -> str | None:
    if el is None:
        return None
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _xml_float(el: ET.Element | None, tag: str) -> float:
    t = _xml_text(el, tag)
    if t is None:
        return 0.0
    try:
        return float(t.replace(",", ""))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# Fetch holdings de un gurú
# ---------------------------------------------------------------------------

def fetch_guru_holdings(guru_name: str) -> list[dict[str, Any]]:
    """Descarga las posiciones del último 13F para un gurú.

    Returns:
        Lista de dicts con keys: name, cusip, value_thousands, shares.
    """
    info = GURU_FUNDS.get(guru_name)
    if not info:
        return []

    cache_key = f"13f_{guru_name}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    cik = info["cik"]
    try:
        xml_url = _get_latest_13f_url(cik)
        if not xml_url:
            print(f"  [13F] No se encontró filing para {guru_name}", flush=True)
            _cache_set(cache_key, [])
            return []

        resp = _sec_get(xml_url)
        holdings = _parse_13f_xml(resp.text)
        print(f"  [13F] {guru_name}: {len(holdings)} posiciones encontradas", flush=True)
        _cache_set(cache_key, holdings)
        return holdings
    except Exception as e:
        print(f"  [13F] Error {guru_name}: {e}", flush=True)
        _cache_set(cache_key, [])
        return []


def fetch_all_guru_holdings() -> dict[str, list[dict[str, Any]]]:
    """Descarga holdings de todos los gurús disponibles."""
    result: dict[str, list[dict[str, Any]]] = {}
    for guru_name in GURU_FUNDS:
        result[guru_name] = fetch_guru_holdings(guru_name)
    return result


# ---------------------------------------------------------------------------
# Matching: mapear nombres del 13F a tickers de nuestro universo
# ---------------------------------------------------------------------------

def _build_name_to_ticker_map(tickers_dict: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Construye un mapa nombre_normalizado → ticker para matching fuzzy."""
    mapping: dict[str, str] = {}
    for ticker, meta in tickers_dict.items():
        nombre = meta.get("nombre", "").upper().strip()
        if nombre:
            # Guardar variantes: nombre completo, sin Inc/Corp/Ltd, primera palabra
            mapping[nombre] = ticker
            for suffix in (" INC", " CORP", " LTD", " PLC", " CO", " GROUP", " HOLDINGS", " NV", " SA", " AG", " SE"):
                if nombre.endswith(suffix):
                    mapping[nombre[: -len(suffix)].strip()] = ticker
            # Primera palabra si tiene >4 chars (para match parcial)
            first = nombre.split()[0] if nombre.split() else ""
            if len(first) >= 5:
                mapping[first] = ticker
        # También ticker directo
        mapping[ticker.upper()] = ticker
    return mapping


def match_holdings_to_universe(
    all_holdings: dict[str, list[dict[str, Any]]],
    tickers_dict: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cruza las posiciones de los gurús con nuestro universo de tickers.

    Returns:
        Dict ticker → {guru_count, gurus: [nombre_guru, ...], total_value, details: [...]}
    """
    name_map = _build_name_to_ticker_map(tickers_dict)

    ticker_info: dict[str, dict[str, Any]] = {}

    for guru_name, holdings in all_holdings.items():
        matched_this_guru: set[str] = set()
        for h in holdings:
            issuer = h["name"].upper().strip()
            ticker = None

            # Intento 1: nombre exacto
            if issuer in name_map:
                ticker = name_map[issuer]
            else:
                # Intento 2: sin sufijos
                for suffix in (" INC", " CORP", " LTD", " PLC", " CO", " GROUP", " HOLDINGS", " CL A", " CL B", " CLASS A", " CLASS B", " COM"):
                    cleaned = issuer.replace(suffix, "").strip()
                    if cleaned in name_map:
                        ticker = name_map[cleaned]
                        break

                # Intento 3: primeras 2 palabras
                if not ticker:
                    words = issuer.split()[:2]
                    partial = " ".join(words)
                    if partial in name_map:
                        ticker = name_map[partial]

            if ticker and ticker not in matched_this_guru:
                matched_this_guru.add(ticker)
                if ticker not in ticker_info:
                    ticker_info[ticker] = {
                        "guru_count": 0,
                        "gurus": [],
                        "total_value": 0.0,
                        "details": [],
                    }
                info = ticker_info[ticker]
                info["guru_count"] += 1
                info["gurus"].append(guru_name)
                info["total_value"] += h["value_thousands"]
                info["details"].append({
                    "guru": guru_name,
                    "value_thousands": h["value_thousands"],
                    "shares": h["shares"],
                })

    return ticker_info


# ---------------------------------------------------------------------------
# Conviction score: bonus para el ranking
# ---------------------------------------------------------------------------

def compute_guru_conviction(
    all_holdings: dict[str, list[dict[str, Any]]],
    tickers_dict: dict[str, dict[str, Any]],
) -> dict[str, float]:
    """Calcula un bonus de convicción guru para cada ticker (0-15 puntos).

    Factores:
    1. Numero de gurus que tienen el activo (0-8 pts)
    2. Peso medio en cartera de los gurus que lo tienen (0-5 pts)
       - >5% del portfolio → +5 (alta conviccion)
       - 2-5% → +3
       - 0.5-2% → +1.5
       - <0.5% → +0.5
    3. Bonus elite +2 si Buffett o Druckenmiller lo tienen
    """
    # Calcular valor total de cartera por guru
    guru_totals: dict[str, float] = {}
    for guru_name, holdings in all_holdings.items():
        total = sum(h["value_thousands"] for h in holdings)
        if total > 0:
            guru_totals[guru_name] = total

    matched = match_holdings_to_universe(all_holdings, tickers_dict)
    scores: dict[str, float] = {}

    for ticker, info in matched.items():
        count = info["guru_count"]

        # Factor 1: numero de gurus (0-8)
        if count >= 3:
            count_score = 8.0
        elif count == 2:
            count_score = 5.0
        else:
            count_score = 2.0

        # Factor 2: peso medio en cartera (0-5)
        pct_list: list[float] = []
        for detail in info["details"]:
            guru_name = detail["guru"]
            guru_total = guru_totals.get(guru_name, 0)
            if guru_total > 0:
                pct = (detail["value_thousands"] / guru_total) * 100
                pct_list.append(pct)

        avg_pct = sum(pct_list) / len(pct_list) if pct_list else 0.0
        if avg_pct > 5.0:
            weight_score = 5.0
        elif avg_pct > 2.0:
            weight_score = 3.0
        elif avg_pct > 0.5:
            weight_score = 1.5
        else:
            weight_score = 0.5

        # Factor 3: bonus elite (0-2)
        elite_bonus = 0.0
        for g in info["gurus"]:
            if g in ("Warren Buffett", "Stanley Druckenmiller"):
                elite_bonus = 2.0
                break

        scores[ticker] = min(15.0, count_score + weight_score + elite_bonus)

    return scores


# ---------------------------------------------------------------------------
# Sección de output para el pipeline
# ---------------------------------------------------------------------------

def guru_holdings_section(
    all_holdings: dict[str, list[dict[str, Any]]],
    tickers_dict: dict[str, dict[str, Any]],
    top20_tickers: list[str],
) -> str:
    """Genera la sección textual de carteras de gurús para el output."""
    matched = match_holdings_to_universe(all_holdings, tickers_dict)

    # Calcular totales por guru para porcentajes
    guru_totals: dict[str, float] = {}
    for guru_name, holdings in all_holdings.items():
        total = sum(h["value_thousands"] for h in holdings)
        if total > 0:
            guru_totals[guru_name] = total

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CARTERAS REALES DE GURUS (13F SEC EDGAR)")
    lines.append("=" * 60)
    lines.append("")

    # Resumen por gurú
    for guru_name, holdings in all_holdings.items():
        fund = GURU_FUNDS[guru_name]["fund"]
        total_val = sum(h["value_thousands"] for h in holdings) / 1_000  # millones
        lines.append(f"  {guru_name} ({fund}): {len(holdings)} posiciones, ${total_val:,.0f}M")
    lines.append("")

    # Top 20 que coinciden con carteras de gurús
    lines.append("COINCIDENCIAS CON TOP 20:")
    lines.append(f"  {'Ticker':<10} {'Gurus':>5} {'Quienes':<40} {'%Cartera':>10} {'Valor ($M)':>12}")
    lines.append("  " + "-" * 77)

    found_any = False
    for ticker in top20_tickers:
        if ticker in matched:
            info = matched[ticker]
            # Calcular % medio de cartera
            pcts = []
            for d in info["details"]:
                gt = guru_totals.get(d["guru"], 0)
                if gt > 0:
                    pcts.append((d["value_thousands"] / gt) * 100)
            avg_pct = sum(pcts) / len(pcts) if pcts else 0.0
            gurus_str = ", ".join(info["gurus"])
            val_m = info["total_value"] / 1_000
            lines.append(f"  {ticker:<10} {info['guru_count']:>5} {gurus_str:<40} {avg_pct:>9.2f}% {val_m:>12,.1f}")
            found_any = True

    if not found_any:
        lines.append("  (Ningun activo del Top 20 coincide con las carteras de gurus)")

    lines.append("")

    # Activos fuera del Top 20 con alta convicción (2+ gurús)
    high_conviction = [
        (t, i) for t, i in matched.items()
        if i["guru_count"] >= 2 and t not in top20_tickers
    ]
    if high_conviction:
        high_conviction.sort(key=lambda x: x[1]["guru_count"], reverse=True)
        lines.append("ACTIVOS CON ALTA CONVICCION GURU (fuera del Top 20):")
        lines.append(f"  {'Ticker':<10} {'Gurus':>5} {'Quienes':<40} {'%Cartera':>10}")
        lines.append("  " + "-" * 65)
        for ticker, info in high_conviction[:10]:
            pcts = []
            for d in info["details"]:
                gt = guru_totals.get(d["guru"], 0)
                if gt > 0:
                    pcts.append((d["value_thousands"] / gt) * 100)
            avg_pct = sum(pcts) / len(pcts) if pcts else 0.0
            gurus_str = ", ".join(info["gurus"])
            lines.append(f"  {ticker:<10} {info['guru_count']:>5} {gurus_str:<40} {avg_pct:>9.2f}%")
        lines.append("")

    output = "\n".join(lines)
    print(output, flush=True)
    return output
