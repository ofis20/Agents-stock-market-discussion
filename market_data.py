#!/usr/bin/env python3
"""Modulo de datos de mercado reales para alimentar el debate macro.

Usa yfinance (gratuito, sin API key) para obtener:
- Precios y rendimientos de indices, acciones, ETFs, commodities, cripto
- Indicadores macro via proxies de mercado (yields, dolar, VIX)
- Metricas fundamentales basicas de acciones
- Datos tecnicos (medias moviles, RSI, volatilidad)

Rango por defecto: 12 meses.
Universo: ~660 activos (500+ acciones, 100 ETFs, 20 commodities, 10 cripto, etc.)
"""

from __future__ import annotations

import gc
import json as _json
import os
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
import pandas as pd

# Workers dinamicos: usa 2× nucleos logicos (IO-bound), minimo 8
_DEFAULT_WORKERS = max(8, (os.cpu_count() or 4) * 2)


# ═══════════════════════════════════════════════════════════════════════════
# CACHE LOCAL (pickle, TTL 1 hora por defecto)
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
_CACHE_TTL_SECONDS = int(os.environ.get("MARKET_CACHE_TTL", 3600))


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(exist_ok=True)
    return _CACHE_DIR / f"{key}.pkl"


def _cache_get(key: str) -> Any | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = datetime.now().timestamp() - p.stat().st_mtime
    if age > _CACHE_TTL_SECONDS:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    try:
        with open(_cache_path(key), "wb") as f:
            pickle.dump(value, f)
    except Exception:
        pass


def _cache_cleanup(max_age_seconds: int = 86_400 * 3) -> int:
    """Elimina archivos de cache con mas de max_age_seconds de antiguedad."""
    if not _CACHE_DIR.exists():
        return 0
    now = datetime.now().timestamp()
    removed = 0
    for p in _CACHE_DIR.glob("*.pkl"):
        try:
            if now - p.stat().st_mtime > max_age_seconds:
                p.unlink()
                removed += 1
        except Exception:
            pass
    return removed


# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSO DE ACTIVOS
# ═══════════════════════════════════════════════════════════════════════════

def _build(tipo: str, tickers: dict[str, str]) -> dict[str, dict[str, str]]:
    return {t: {"nombre": n, "tipo": tipo} for t, n in tickers.items()}


# ── INDICES GLOBALES (~30) ──
_INDICES: dict[str, str] = {
    # Americas
    "^GSPC": "S&P 500", "^IXIC": "NASDAQ Composite", "^DJI": "Dow Jones",
    "^RUT": "Russell 2000", "^BVSP": "Bovespa", "^MXX": "IPC Mexico",
    "^MERV": "MERVAL Argentina", "^IPSA": "IPSA Chile",
    # Europa
    "^STOXX50E": "Euro Stoxx 50", "^GDAXI": "DAX", "^FTSE": "FTSE 100",
    "^FCHI": "CAC 40", "^IBEX": "IBEX 35", "FTSEMIB.MI": "FTSE MIB",
    "^AEX": "AEX Amsterdam", "^SSMI": "SMI Suiza", "^OMX": "OMX Stockholm 30",
    "^OMXC25": "OMX Copenhagen 25", "^OMXH25": "OMX Helsinki 25",
    "^OSEAX": "Oslo All-Share", "WIG20.WA": "WIG 20 Polonia",
    "XU100.IS": "BIST 100 Turquia", "^ATG": "Atenas General",
    "^PSI20": "PSI 20 Portugal",
    # Asia-Pacifico
    "^N225": "Nikkei 225", "^HSI": "Hang Seng", "000001.SS": "Shanghai Composite",
    "^KS11": "KOSPI", "^TWII": "TAIEX", "^NSEI": "Nifty 50",
    "^STI": "STI Singapur", "^JKSE": "Jakarta Composite",
    "^SET.BK": "SET Tailandia", "^KLSE": "KLCI Malasia",
    "^AXJO": "ASX 200 Australia", "^TASI.SR": "Tadawul Arabia Saudi",
    # Africa
    "^J203.JO": "JSE All-Share Sudafrica",
}

# ── USA: Tecnologia (~60) ──
_US_TECH: dict[str, str] = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "GOOGL": "Alphabet",
    "META": "Meta Platforms", "AVGO": "Broadcom", "ORCL": "Oracle", "CRM": "Salesforce",
    "ADBE": "Adobe", "AMD": "AMD", "INTC": "Intel", "CSCO": "Cisco",
    "IBM": "IBM", "TXN": "Texas Instruments", "QCOM": "Qualcomm", "NOW": "ServiceNow",
    "INTU": "Intuit", "AMAT": "Applied Materials", "MU": "Micron", "LRCX": "Lam Research",
    "KLAC": "KLA Corp", "SNPS": "Synopsys", "CDNS": "Cadence Design", "MRVL": "Marvell",
    "NXPI": "NXP Semi", "PANW": "Palo Alto Networks", "CRWD": "CrowdStrike",
    "FTNT": "Fortinet", "ZS": "Zscaler", "NET": "Cloudflare", "DDOG": "Datadog",
    "SNOW": "Snowflake", "PLTR": "Palantir", "SHOP": "Shopify", "BILL": "Bill.com",
    "PYPL": "PayPal", "UBER": "Uber", "ABNB": "Airbnb", "SMCI": "Super Micro",
    "ON": "ON Semiconductor", "MCHP": "Microchip Tech", "SWKS": "Skyworks",
    "MPWR": "Monolithic Power", "ADI": "Analog Devices", "DELL": "Dell Technologies",
    "HPQ": "HP Inc", "HPE": "HPE", "GDDY": "GoDaddy", "TEAM": "Atlassian",
    "HUBS": "HubSpot", "TTD": "Trade Desk", "ROKU": "Roku", "U": "Unity Software",
    "ARM": "ARM Holdings", "IOT": "Samsara", "COIN": "Coinbase", "APP": "AppLovin",
    "DUOL": "Duolingo", "MNDY": "monday.com", "GLOB": "Globant", "DOCN": "DigitalOcean",
    "MDB": "MongoDB", "CFLT": "Confluent", "PATH": "UiPath",
}

# ── USA: Salud (~45) ──
_US_HEALTH: dict[str, str] = {
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "LLY": "Eli Lilly",
    "ABBV": "AbbVie", "MRK": "Merck", "PFE": "Pfizer", "TMO": "Thermo Fisher",
    "ABT": "Abbott Labs", "DHR": "Danaher", "BMY": "Bristol-Myers",
    "AMGN": "Amgen", "GILD": "Gilead Sciences", "VRTX": "Vertex Pharma",
    "REGN": "Regeneron", "ISRG": "Intuitive Surgical", "MDT": "Medtronic",
    "SYK": "Stryker", "BSX": "Boston Scientific", "EW": "Edwards Lifesciences",
    "ZTS": "Zoetis", "DXCM": "DexCom", "IDXX": "IDEXX Labs", "ALGN": "Align Tech",
    "MRNA": "Moderna", "CRSP": "CRISPR Therapeutics", "ILMN": "Illumina",
    "BIO": "Bio-Rad", "A": "Agilent Technologies", "HOLX": "Hologic",
    "PODD": "Insulet", "VEEV": "Veeva Systems", "HCA": "HCA Healthcare",
    "GEHC": "GE HealthCare", "CI": "Cigna Group", "ELV": "Elevance Health",
    "HUM": "Humana", "CNC": "Centene", "MOH": "Molina Healthcare",
    "IQV": "IQVIA", "WST": "West Pharma", "BAX": "Baxter", "BDX": "Becton Dickinson",
    "MTD": "Mettler-Toledo", "TECH": "Bio-Techne", "WAT": "Waters Corp",
}

# ── USA: Financiero (~40) ──
_US_FINANCE: dict[str, str] = {
    "JPM": "JPMorgan Chase", "V": "Visa", "MA": "Mastercard", "BAC": "Bank of America",
    "WFC": "Wells Fargo", "GS": "Goldman Sachs", "MS": "Morgan Stanley",
    "BLK": "BlackRock", "SCHW": "Charles Schwab", "AXP": "American Express",
    "C": "Citigroup", "USB": "US Bancorp", "PNC": "PNC Financial",
    "SPGI": "S&P Global", "MCO": "Moody's", "ICE": "Intercontinental Exchange",
    "CME": "CME Group", "TROW": "T. Rowe Price", "AON": "Aon", "MET": "MetLife",
    "PRU": "Prudential Financial", "ALL": "Allstate", "TRV": "Travelers",
    "CB": "Chubb", "FITB": "Fifth Third", "KEY": "KeyCorp", "CFG": "Citizens Financial",
    "HBAN": "Huntington Bancshares", "RJF": "Raymond James", "NTRS": "Northern Trust",
    "MKTX": "MarketAxess", "CBOE": "Cboe Global Markets", "NDAQ": "Nasdaq Inc",
    "CINF": "Cincinnati Financial", "GL": "Globe Life", "WRB": "Berkley Corp",
    "AFG": "American Financial", "ERIE": "Erie Indemnity", "BRO": "Brown & Brown",
    "FIS": "Fidelity National", "FISV": "Fiserv",
}

# ── USA: Consumo Discrecional (~35) ──
_US_CONSUMER_DISC: dict[str, str] = {
    "AMZN": "Amazon", "TSLA": "Tesla", "HD": "Home Depot", "NKE": "Nike",
    "MCD": "McDonald's", "SBUX": "Starbucks", "LOW": "Lowe's", "TJX": "TJX Companies",
    "ROST": "Ross Stores", "CMG": "Chipotle", "DPZ": "Domino's", "YUM": "Yum! Brands",
    "BKNG": "Booking Holdings", "MAR": "Marriott", "HLT": "Hilton", "DG": "Dollar General",
    "DLTR": "Dollar Tree", "BBY": "Best Buy", "POOL": "Pool Corp", "TSCO": "Tractor Supply",
    "DHI": "DR Horton", "LEN": "Lennar", "PHM": "PulteGroup", "DECK": "Deckers Outdoor",
    "LULU": "Lululemon", "WYNN": "Wynn Resorts", "LVS": "Las Vegas Sands",
    "MGM": "MGM Resorts", "EXPE": "Expedia", "ORLY": "O'Reilly Auto", "AZO": "AutoZone",
    "CPRT": "Copart", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian",
}

# ── USA: Consumo Basico (~20) ──
_US_CONSUMER_STAPLES: dict[str, str] = {
    "PG": "Procter & Gamble", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "COST": "Costco", "WMT": "Walmart", "CL": "Colgate-Palmolive",
    "CLX": "Clorox", "KMB": "Kimberly-Clark", "GIS": "General Mills",
    "MDLZ": "Mondelez International", "HSY": "Hershey", "SJM": "JM Smucker", "MKC": "McCormick",
    "MNST": "Monster Beverage", "STZ": "Constellation Brands", "TAP": "Molson Coors",
    "PM": "Philip Morris", "MO": "Altria", "KR": "Kroger", "SYY": "Sysco",
}

# ── USA: Industrial (~35) ──
_US_INDUSTRIAL: dict[str, str] = {
    "CAT": "Caterpillar", "DE": "Deere & Co", "BA": "Boeing",
    "RTX": "RTX Corp", "LMT": "Lockheed Martin", "NOC": "Northrop Grumman",
    "GD": "General Dynamics", "GE": "GE Aerospace", "HON": "Honeywell",
    "MMM": "3M", "UNP": "Union Pacific", "UPS": "UPS", "FDX": "FedEx",
    "CSX": "CSX Corp", "NSC": "Norfolk Southern", "WM": "Waste Management",
    "RSG": "Republic Services", "EMR": "Emerson Electric", "ETN": "Eaton",
    "ROK": "Rockwell Automation", "PH": "Parker Hannifin", "ODFL": "Old Dominion Freight",
    "FAST": "Fastenal", "CTAS": "Cintas", "PCAR": "PACCAR", "IR": "Ingersoll Rand",
    "TT": "Trane Technologies", "CARR": "Carrier Global", "OTIS": "Otis Worldwide",
    "XYL": "Xylem", "VRSK": "Verisk Analytics", "DOV": "Dover", "AME": "Ametek",
    "ITW": "Illinois Tool Works", "SWK": "Stanley Black & Decker",
}

# ── USA: Energia (~20) ──
_US_ENERGY: dict[str, str] = {
    "XOM": "ExxonMobil", "CVX": "Chevron", "COP": "ConocoPhillips",
    "EOG": "EOG Resources", "SLB": "Schlumberger", "OXY": "Occidental Petroleum",
    "MPC": "Marathon Petroleum", "VLO": "Valero Energy", "PSX": "Phillips 66",
    "DVN": "Devon Energy", "FANG": "Diamondback Energy", "DINO": "HF Sinclair",
    "HAL": "Halliburton", "BKR": "Baker Hughes", "TRGP": "Targa Resources",
    "WMB": "Williams Companies", "KMI": "Kinder Morgan", "OKE": "ONEOK",
    "ET": "Energy Transfer", "CTRA": "Coterra Energy",
}

# ── USA: Materiales (~15) ──
_US_MATERIALS: dict[str, str] = {
    "LIN": "Linde", "APD": "Air Products", "SHW": "Sherwin-Williams",
    "DD": "DuPont", "NEM": "Newmont Mining", "FCX": "Freeport-McMoRan",
    "NUE": "Nucor", "CF": "CF Industries", "MOS": "Mosaic",
    "ALB": "Albemarle", "ECL": "Ecolab", "PPG": "PPG Industries",
    "VMC": "Vulcan Materials", "MLM": "Martin Marietta", "BALL": "Ball Corp",
}

# ── USA: Inmobiliario (~12) ──
_US_REIT: dict[str, str] = {
    "PLD": "Prologis", "AMT": "American Tower", "CCI": "Crown Castle",
    "EQIX": "Equinix", "DLR": "Digital Realty", "SPG": "Simon Property",
    "O": "Realty Income", "WELL": "Welltower", "AVB": "AvalonBay",
    "ARE": "Alexandria RE", "PSA": "Public Storage", "VICI": "VICI Properties",
}

# ── USA: Utilities (~10) ──
_US_UTILITIES: dict[str, str] = {
    "NEE": "NextEra Energy", "DUK": "Duke Energy", "SO": "Southern Co",
    "AEP": "American Electric Power", "D": "Dominion Energy", "SRE": "Sempra",
    "EXC": "Exelon", "XEL": "Xcel Energy", "WEC": "WEC Energy", "ES": "Eversource",
}

# ── USA: Comunicaciones (~15) ──
_US_COMM: dict[str, str] = {
    "DIS": "Disney", "NFLX": "Netflix", "CMCSA": "Comcast",
    "T": "AT&T", "VZ": "Verizon", "TMUS": "T-Mobile US",
    "EA": "Electronic Arts", "TTWO": "Take-Two Interactive", "RBLX": "Roblox",
    "MTCH": "Match Group", "PINS": "Pinterest", "SNAP": "Snap",
    "FOX": "Fox Corp", "WBD": "Warner Bros Discovery", "LYV": "Live Nation",
}

# ── USA: Adicionales mid-cap / growth (~25) ──
_US_EXTRA: dict[str, str] = {
    "BRK-B": "Berkshire Hathaway", "SOFI": "SoFi Technologies", "NU": "Nu Holdings",
    "MELI": "MercadoLibre", "SE": "Sea Limited", "GRAB": "Grab Holdings",
    "CAVA": "CAVA Group", "ONON": "On Holding", "BIRK": "Birkenstock",
    "CELH": "Celsius Holdings", "TOST": "Toast", "DT": "Dynatrace",
    "DKNG": "DraftKings", "CVNA": "Carvana", "ZIM": "ZIM Shipping",
    "ENPH": "Enphase Energy", "SEDG": "SolarEdge", "FSLR": "First Solar",
    "RUN": "Sunrun", "W": "Wayfair", "ETSY": "Etsy", "CHWY": "Chewy",
    "HIMS": "Hims & Hers", "SOUN": "SoundHound AI", "IONQ": "IonQ",
}

# ── Internacional: ADRs en bolsa USA (~80) ──
_INTL_ADR: dict[str, str] = {
    # UK
    "AZN": "AstraZeneca (UK)", "BP": "BP (UK)", "SHEL": "Shell (UK/NL)",
    "GSK": "GSK (UK)", "UL": "Unilever (UK/NL)", "BTI": "BAT (UK)",
    "DEO": "Diageo (UK)", "RIO": "Rio Tinto (UK/AU)", "BHP": "BHP Group (AU/UK)",
    "VOD": "Vodafone (UK)", "LYG": "Lloyds Banking (UK)", "BCS": "Barclays (UK)",
    # Alemania
    "SAP": "SAP (Alemania)", "SIEGY": "Siemens (Alemania)",
    # Paises Bajos
    "ASML": "ASML (Paises Bajos)", "ING": "ING Group (Paises Bajos)",
    "PHG": "Philips (Paises Bajos)",
    # Francia
    "TTE": "TotalEnergies (Francia)", "SNY": "Sanofi (Francia)",
    # Suiza
    "UBS": "UBS (Suiza)",
    # Dinamarca / Nordicos
    "NVO": "Novo Nordisk (Dinamarca)", "ERIC": "Ericsson (Suecia)",
    "SPOT": "Spotify (Suecia)",
    # Irlanda
    "CRH": "CRH (Irlanda)", "ACGL": "Arch Capital (Bermudas)",
    # Japon
    "TM": "Toyota (Japon)", "SONY": "Sony (Japon)", "HMC": "Honda (Japon)",
    "MUFG": "Mitsubishi UFJ (Japon)", "MFG": "Mizuho Financial (Japon)",
    "NMR": "Nomura Holdings (Japon)",
    # China
    "BABA": "Alibaba (China)", "JD": "JD.com (China)", "PDD": "PDD Holdings (China)",
    "BIDU": "Baidu (China)", "NIO": "NIO (China)", "XPEV": "XPeng (China)",
    "LI": "Li Auto (China)", "NTES": "NetEase (China)", "TCOM": "Trip.com (China)",
    "YMM": "Full Truck Alliance (China)", "FUTU": "Futu Holdings (China)",
    "BILI": "Bilibili (China)", "WB": "Weibo (China)", "VNET": "VNET Group (China)",
    "TME": "Tencent Music (China)", "ZTO": "ZTO Express (China)",
    # Taiwan
    "TSM": "TSMC (Taiwan)", "UMC": "UMC (Taiwan)", "ASX": "ASE Technology (Taiwan)",
    # Corea
    "KB": "KB Financial (Corea)", "SHG": "Shinhan Financial (Corea)",
    # India
    "INFY": "Infosys (India)", "WIT": "Wipro (India)", "HDB": "HDFC Bank (India)",
    "IBN": "ICICI Bank (India)", "RDY": "Dr. Reddy's (India)",
    "SIFY": "Sify Technologies (India)",
    # Brasil
    "VALE": "Vale (Brasil)", "PBR": "Petrobras (Brasil)", "ITUB": "Itau Unibanco (Brasil)",
    "BBD": "Bradesco (Brasil)", "ABEV": "Ambev (Brasil)", "SID": "CSN (Brasil)",
    "GGB": "Gerdau (Brasil)", "CIG": "CEMIG (Brasil)",
    # Mexico
    "AMX": "America Movil (Mexico)",
    # Argentina
    "YPF": "YPF (Argentina)", "GGAL": "Grupo Galicia (Argentina)",
    # Chile
    "SQM": "SQM (Chile)", "BSAC": "Banco Santander Chile",
    # Canada
    "CP": "Canadian Pacific (Canada)", "CNI": "CN Rail (Canada)",
    "ENB": "Enbridge (Canada)", "BMO": "Bank of Montreal (Canada)",
    "TD": "Toronto-Dominion (Canada)", "RY": "Royal Bank Canada",
    # Israel
    "TEVA": "Teva Pharma (Israel)", "NICE": "NICE Ltd (Israel)",
    "CYBR": "CyberArk (Israel)",
}

# ── Internacional: Bolsas europeas locales (~55) ──
_EU_LOCAL: dict[str, str] = {
    # UK – FTSE 100 principales (London)
    "HSBA.L": "HSBC (London)", "LLOY.L": "Lloyds (London)", "BARC.L": "Barclays (London)",
    "GLEN.L": "Glencore (London)", "AAL.L": "Anglo American (London)",
    "DGE.L": "Diageo (London)", "REL.L": "RELX (London)", "LSEG.L": "LSEG (London)",
    "CPG.L": "Compass Group (London)", "ABF.L": "AB Foods (London)",
    "ANTO.L": "Antofagasta (London)", "RKT.L": "Reckitt (London)",
    "AZN.L": "AstraZeneca (London)", "SHEL.L": "Shell (London)",
    "ULVR.L": "Unilever (London)", "GSK.L": "GSK (London)",
    "RIO.L": "Rio Tinto (London)", "NG.L": "National Grid (London)",
    "BATS.L": "BAT (London)", "PRU.L": "Prudential (London)",
    "SSE.L": "SSE (London)", "VOD.L": "Vodafone (London)",
    "TSCO.L": "Tesco (London)", "BA.L": "BAE Systems (London)",
    "SN.L": "Smith & Nephew (London)", "WPP.L": "WPP (London)",
    "RR.L": "Rolls-Royce (London)", "III.L": "3i Group (London)",
    "EXPN.L": "Experian (London)", "AHT.L": "Ashtead Group (London)",
    "SBRY.L": "Sainsbury's (London)", "LAND.L": "British Land (London)",
    "SGE.L": "Sage Group (London)", "SMDS.L": "DS Smith (London)",
    "MNDI.L": "Mondi (London)", "NWG.L": "NatWest Group (London)",
    "IMB.L": "Imperial Brands (London)", "IHG.L": "IHG Hotels (London)",
    "BNZL.L": "Bunzl (London)", "CRH.L": "CRH (London)",
    "STAN.L": "Standard Chartered (London)",
    # Alemania – DAX 40 (Frankfurt)
    "SIE.DE": "Siemens (Frankfurt)", "ALV.DE": "Allianz (Frankfurt)",
    "MBG.DE": "Mercedes-Benz (Frankfurt)", "BMW.DE": "BMW (Frankfurt)",
    "DTE.DE": "Deutsche Telekom (Frankfurt)", "MUV2.DE": "Munich Re (Frankfurt)",
    "IFX.DE": "Infineon (Frankfurt)", "ADS.DE": "Adidas (Frankfurt)",
    "BEI.DE": "Beiersdorf (Frankfurt)", "HEN3.DE": "Henkel (Frankfurt)",
    "FRE.DE": "Fresenius (Frankfurt)", "DB1.DE": "Deutsche Boerse (Frankfurt)",
    "VOW3.DE": "Volkswagen (Frankfurt)", "BAS.DE": "BASF (Frankfurt)",
    "DTG.DE": "Daimler Truck (Frankfurt)", "DBK.DE": "Deutsche Bank (Frankfurt)",
    "RWE.DE": "RWE (Frankfurt)", "HEI.DE": "HeidelbergCement (Frankfurt)",
    "SHL.DE": "Siemens Healthineers (Frankfurt)", "MTX.DE": "MTU Aero (Frankfurt)",
    "SY1.DE": "Symrise (Frankfurt)", "PAH3.DE": "Porsche SE (Frankfurt)",
    "P911.DE": "Porsche AG (Frankfurt)", "ZAL.DE": "Zalando (Frankfurt)",
    "ENR.DE": "Siemens Energy (Frankfurt)", "1COV.DE": "Covestro (Frankfurt)",
    "QIA.DE": "Qiagen (Frankfurt)", "AIR.DE": "Airbus (Frankfurt)",
    "HNR1.DE": "Hannover Re (Frankfurt)", "CON.DE": "Continental (Frankfurt)",
    "MRK.DE": "Merck KGaA (Frankfurt)", "EOAN.DE": "E.ON (Frankfurt)",
    "SRT3.DE": "Sartorius (Frankfurt)", "PUM.DE": "Puma (Frankfurt)",
    # Francia – CAC 40 (Paris)
    "MC.PA": "LVMH (Paris)", "OR.PA": "L'Oreal (Paris)", "AI.PA": "Air Liquide (Paris)",
    "SAN.PA": "Sanofi (Paris)", "BNP.PA": "BNP Paribas (Paris)",
    "CS.PA": "AXA (Paris)", "SU.PA": "Schneider Electric (Paris)",
    "AIR.PA": "Airbus (Paris)", "DG.PA": "Vinci (Paris)", "RI.PA": "Pernod Ricard (Paris)",
    "KER.PA": "Kering (Paris)", "SAF.PA": "Safran (Paris)",
    "STM.PA": "STMicro (Paris)", "GLE.PA": "Societe Generale (Paris)",
    "CAP.PA": "Capgemini (Paris)", "VIV.PA": "Vivendi (Paris)",
    "SGO.PA": "Saint-Gobain (Paris)", "CA.PA": "Carrefour (Paris)",
    "HO.PA": "Thales (Paris)", "ML.PA": "Michelin (Paris)",
    "EN.PA": "Bouygues (Paris)", "TEP.PA": "Teleperformance (Paris)",
    "ORA.PA": "Orange (Paris)", "ERA.PA": "Eramet (Paris)",
    "DSY.PA": "Dassault Systemes (Paris)", "RNO.PA": "Renault (Paris)",
    "LR.PA": "Legrand (Paris)", "WLN.PA": "Worldline (Paris)",
    "EL.PA": "EssilorLuxottica (Paris)", "ATO.PA": "Atos (Paris)",
    "BN.PA": "Danone (Paris)", "URW.PA": "Unibail-Rodamco (Paris)",
    "PUB.PA": "Publicis (Paris)", "ENGI.PA": "Engie (Paris)",
    # Espana – IBEX 35 completo (Madrid)
    "SAN.MC": "Banco Santander (Madrid)", "TEF.MC": "Telefonica (Madrid)",
    "IBE.MC": "Iberdrola (Madrid)", "ITX.MC": "Inditex (Madrid)",
    "REP.MC": "Repsol (Madrid)", "BBVA.MC": "BBVA (Madrid)",
    "FER.MC": "Ferrovial (Madrid)", "AMS.MC": "Amadeus IT (Madrid)",
    "CABK.MC": "CaixaBank (Madrid)", "ACS.MC": "ACS (Madrid)",
    "AENA.MC": "Aena (Madrid)", "ANA.MC": "Acciona (Madrid)",
    "ANE.MC": "Acciona Energia (Madrid)", "ACX.MC": "Acerinox (Madrid)",
    "BKT.MC": "Bankinter (Madrid)", "CLNX.MC": "Cellnex (Madrid)",
    "COL.MC": "Inmobiliaria Colonial (Madrid)", "ELE.MC": "Endesa (Madrid)",
    "ENG.MC": "Enagas (Madrid)", "FDR.MC": "Fluidra (Madrid)",
    "GRF.MC": "Grifols (Madrid)", "IAG.MC": "IAG (Madrid)",
    "IDR.MC": "Indra (Madrid)", "LOG.MC": "Logista (Madrid)",
    "MAP.MC": "Mapfre (Madrid)", "MEL.MC": "Melia Hotels (Madrid)",
    "MRL.MC": "Merlin Properties (Madrid)", "MTS.MC": "ArcelorMittal (Madrid)",
    "NTGY.MC": "Naturgy (Madrid)", "PHM.MC": "Pharma Mar (Madrid)",
    "RED.MC": "Redeia (Madrid)", "ROVI.MC": "Laboratorios Rovi (Madrid)",
    "SAB.MC": "Banco Sabadell (Madrid)", "SCYR.MC": "Sacyr (Madrid)",
    "SLR.MC": "Solaria (Madrid)", "UNI.MC": "Unicaja (Madrid)",
    # Portugal – PSI 20 (Lisboa)
    "EDP.LS": "EDP (Lisboa)", "GALP.LS": "Galp Energia (Lisboa)",
    "SON.LS": "Sonae (Lisboa)", "JMT.LS": "Jeronimo Martins (Lisboa)",
    "EDPR.LS": "EDP Renovaveis (Lisboa)", "BCP.LS": "Banco Comercial Portugues (Lisboa)",
    "NOS.LS": "NOS SGPS (Lisboa)", "ALTR.LS": "Altri (Lisboa)",
    "CTT.LS": "CTT Correios (Lisboa)", "SEM.LS": "Semapa (Lisboa)",
    # Italia – FTSE MIB (Milan)
    "ENEL.MI": "Enel (Milan)", "UCG.MI": "UniCredit (Milan)",
    "ISP.MI": "Intesa Sanpaolo (Milan)", "ENI.MI": "Eni (Milan)",
    "STM": "STMicroelectronics (ADR)", "RACE.MI": "Ferrari (Milan)",
    "G.MI": "Generali (Milan)", "TIT.MI": "Telecom Italia (Milan)",
    "PIRC.MI": "Pirelli (Milan)", "BAMI.MI": "Banco BPM (Milan)",
    "PRY.MI": "Prysmian (Milan)", "STLAM.MI": "Stellantis (Milan)",
    "TEN.MI": "Tenaris (Milan)", "A2A.MI": "A2A (Milan)",
    "CPR.MI": "Campari (Milan)", "MONC.MI": "Moncler (Milan)",
    "SPM.MI": "Saipem (Milan)", "SRG.MI": "Snam (Milan)",
    "TRN.MI": "Terna (Milan)", "AMP.MI": "Amplifon (Milan)",
    "PST.MI": "Poste Italiane (Milan)", "MB.MI": "Mediobanca (Milan)",
    "LDO.MI": "Leonardo (Milan)", "HER.MI": "Hera (Milan)",
    "BGN.MI": "Brembo (Milan)", "REC.MI": "Recordati (Milan)",
    "NWL.MI": "Newlat Food (Milan)",
    # Paises Bajos (Amsterdam)
    "PHIA.AS": "Philips (Amsterdam)", "AD.AS": "Ahold Delhaize (Amsterdam)",
    "WKL.AS": "Wolters Kluwer (Amsterdam)", "UNA.AS": "Unilever (Amsterdam)",
    "HEIA.AS": "Heineken (Amsterdam)", "MT.AS": "ArcelorMittal (Amsterdam)",
    "RAND.AS": "Randstad (Amsterdam)", "AKZA.AS": "AkzoNobel (Amsterdam)",
    "INGA.AS": "ING Group (Amsterdam)", "ASM.AS": "ASM International (Amsterdam)",
    "IMCD.AS": "IMCD (Amsterdam)", "PRX.AS": "Prosus (Amsterdam)",
    "DSM.AS": "DSM-Firmenich (Amsterdam)", "NN.AS": "NN Group (Amsterdam)",
    # Belgica (Bruselas)
    "ABI.BR": "AB InBev (Bruselas)", "UCB.BR": "UCB (Bruselas)",
    "KBC.BR": "KBC Group (Bruselas)", "SOLB.BR": "Solvay (Bruselas)",
    "ACKB.BR": "Ackermans & van Haaren (Bruselas)", "GBLB.BR": "GBL (Bruselas)",
    # Nordicos – Suecia (Estocolmo)
    "VOLV-B.ST": "Volvo (Estocolmo)", "ATCO-A.ST": "Atlas Copco (Estocolmo)",
    "SAND.ST": "Sandvik (Estocolmo)", "SEB-A.ST": "SEB (Estocolmo)",
    "INVE-B.ST": "Investor AB (Estocolmo)", "HM-B.ST": "H&M (Estocolmo)",
    "ABB.ST": "ABB (Estocolmo)", "HEXA-B.ST": "Hexagon (Estocolmo)",
    "ALFA.ST": "Alfa Laval (Estocolmo)", "ASSA-B.ST": "Assa Abloy (Estocolmo)",
    "ERIC-B.ST": "Ericsson (Estocolmo)", "TEL2-B.ST": "Tele2 (Estocolmo)",
    "SCA-B.ST": "SCA (Estocolmo)", "SKF-B.ST": "SKF (Estocolmo)",
    "ESSITY-B.ST": "Essity (Estocolmo)", "SWED-A.ST": "Swedbank (Estocolmo)",
    "KINV-B.ST": "Kinnevik (Estocolmo)",
    # Nordicos – Dinamarca (Copenhague)
    "NOVO-B.CO": "Novo Nordisk (Copenhague)", "CARL-B.CO": "Carlsberg (Copenhague)",
    "MAERSK-B.CO": "Maersk (Copenhague)", "VWS.CO": "Vestas Wind (Copenhague)",
    "DSV.CO": "DSV (Copenhague)", "PNDORA.CO": "Pandora (Copenhague)",
    "COLO-B.CO": "Coloplast (Copenhague)", "GN.CO": "GN Store Nord (Copenhague)",
    "TRYG.CO": "Tryg (Copenhague)", "ORSTED.CO": "Orsted (Copenhague)",
    "DEMANT.CO": "Demant (Copenhague)",
    # Nordicos – Finlandia (Helsinki)
    "NOKIA.HE": "Nokia (Helsinki)", "NESTE.HE": "Neste (Helsinki)",
    "FORTUM.HE": "Fortum (Helsinki)", "SAMPO.HE": "Sampo (Helsinki)",
    "KNEBV.HE": "Kone (Helsinki)", "UPM.HE": "UPM-Kymmene (Helsinki)",
    "STERV.HE": "Stora Enso (Helsinki)", "WRT1V.HE": "Wartsila (Helsinki)",
    # Nordicos – Noruega (Oslo)
    "EQNR.OL": "Equinor (Oslo)", "DNB.OL": "DNB Bank (Oslo)",
    "TEL.OL": "Telenor (Oslo)", "MOWI.OL": "Mowi (Oslo)",
    "ORK.OL": "Orkla (Oslo)", "YAR.OL": "Yara International (Oslo)",
    "SALM.OL": "SalMar (Oslo)", "KOG.OL": "Kongsberg Gruppen (Oslo)",
    "AKER.OL": "Aker (Oslo)", "AKRBP.OL": "Aker BP (Oslo)",
    "SUBC.OL": "Subsea 7 (Oslo)", "BWLPG.OL": "BW LPG (Oslo)",
    # Suiza
    "NESN.SW": "Nestle (Zurich)", "ROG.SW": "Roche (Zurich)",
    "NOVN.SW": "Novartis (Zurich)", "UBSG.SW": "UBS (Zurich)",
    "ABBN.SW": "ABB (Zurich)", "SREN.SW": "Swiss Re (Zurich)",
    "ZURN.SW": "Zurich Insurance (Zurich)", "LONN.SW": "Lonza (Zurich)",
    "GEBN.SW": "Geberit (Zurich)", "SGSN.SW": "SGS (Zurich)",
    "GIVN.SW": "Givaudan (Zurich)", "SCMN.SW": "Swisscom (Zurich)",
    "SLHN.SW": "Swiss Life (Zurich)", "SIKA.SW": "Sika (Zurich)",
    # Polonia (Varsovia)
    "PKN.WA": "PKN Orlen (Varsovia)", "PEO.WA": "Bank Pekao (Varsovia)",
    "PKO.WA": "PKO BP (Varsovia)", "PZU.WA": "PZU (Varsovia)",
    "KGH.WA": "KGHM (Varsovia)", "DNP.WA": "Dino Polska (Varsovia)",
    "LPP.WA": "LPP (Varsovia)", "ALE.WA": "Allegro (Varsovia)",
    "CDR.WA": "CD Projekt (Varsovia)", "SPR.WA": "Spyrosoft (Varsovia)",
    # Turquia (Estambul)
    "THYAO.IS": "Turkish Airlines (Estambul)", "GARAN.IS": "Garanti BBVA (Estambul)",
    "EREGL.IS": "Eregli Demir (Estambul)", "AKBNK.IS": "Akbank (Estambul)",
    "BIMAS.IS": "BIM (Estambul)", "SISE.IS": "Sisecam (Estambul)",
    "KCHOL.IS": "Koc Holding (Estambul)", "TUPRS.IS": "Tupras (Estambul)",
    # Grecia (Atenas)
    "OPAP.AT": "OPAP (Atenas)", "ETE.AT": "National Bank Greece (Atenas)",
    "EUROB.AT": "Eurobank (Atenas)", "HTO.AT": "Hellenic Telecom (Atenas)",
    "MYTIL.AT": "Mytilineos (Atenas)",
}

# ── Internacional: Asia / Pacifico locales (~45) ──
_ASIA_LOCAL: dict[str, str] = {
    # Japon – Nikkei 225 principales (Tokyo)
    "7203.T": "Toyota (Tokyo)", "6758.T": "Sony (Tokyo)", "8306.T": "MUFG (Tokyo)",
    "9984.T": "SoftBank Group (Tokyo)", "6501.T": "Hitachi (Tokyo)",
    "7267.T": "Honda (Tokyo)", "4502.T": "Takeda Pharma (Tokyo)",
    "6902.T": "Denso (Tokyo)", "8035.T": "Tokyo Electron (Tokyo)",
    "6861.T": "Keyence (Tokyo)", "9432.T": "NTT (Tokyo)", "8001.T": "ITOCHU (Tokyo)",
    "4063.T": "Shin-Etsu Chemical (Tokyo)", "6098.T": "Recruit Holdings (Tokyo)",
    "3382.T": "Seven & i (Tokyo)", "6367.T": "Daikin (Tokyo)",
    "4568.T": "Daiichi Sankyo (Tokyo)", "9983.T": "Fast Retailing (Tokyo)",
    "6594.T": "Nidec (Tokyo)", "7741.T": "HOYA (Tokyo)",
    "6857.T": "Advantest (Tokyo)", "8058.T": "Mitsubishi Corp (Tokyo)",
    "8031.T": "Mitsui & Co (Tokyo)", "2914.T": "Japan Tobacco (Tokyo)",
    "6273.T": "SMC Corp (Tokyo)", "4519.T": "Chugai Pharma (Tokyo)",
    "7751.T": "Canon (Tokyo)", "6971.T": "Kyocera (Tokyo)",
    "8316.T": "SMFG (Tokyo)", "6954.T": "Fanuc (Tokyo)",
    "4661.T": "Oriental Land (Tokyo)", "8766.T": "Tokio Marine (Tokyo)",
    # Hong Kong
    "0700.HK": "Tencent (HK)", "9988.HK": "Alibaba (HK)", "3690.HK": "Meituan (HK)",
    "1810.HK": "Xiaomi (HK)", "2318.HK": "Ping An Insurance (HK)",
    "0005.HK": "HSBC (HK)", "1299.HK": "AIA Group (HK)", "2020.HK": "ANTA Sports (HK)",
    "0941.HK": "China Mobile (HK)", "1211.HK": "BYD (HK)",
    "0388.HK": "HKEX (HK)", "0001.HK": "CK Hutchison (HK)",
    "0016.HK": "Sun Hung Kai (HK)", "0002.HK": "CLP Holdings (HK)",
    "0003.HK": "HK & China Gas (HK)", "2628.HK": "China Life (HK)",
    "0027.HK": "Galaxy Entertainment (HK)", "1928.HK": "Sands China (HK)",
    "9618.HK": "JD.com (HK)", "9999.HK": "NetEase (HK)",
    # China Continental (Shanghai / Shenzhen)
    "600519.SS": "Kweichow Moutai (Shanghai)", "601318.SS": "Ping An (Shanghai)",
    "600036.SS": "China Merchants Bank (Shanghai)", "601888.SS": "China Tourism (Shanghai)",
    "600276.SS": "Jiangsu Hengrui (Shanghai)", "601899.SS": "Zijin Mining (Shanghai)",
    "000858.SZ": "Wuliangye (Shenzhen)", "300750.SZ": "CATL (Shenzhen)",
    "002594.SZ": "BYD (Shenzhen)", "000333.SZ": "Midea Group (Shenzhen)",
    "002415.SZ": "Hikvision (Shenzhen)", "000651.SZ": "Gree Electric (Shenzhen)",
    # Corea del Sur (KSE)
    "005930.KS": "Samsung Electronics (Corea)", "000660.KS": "SK Hynix (Corea)",
    "035420.KS": "NAVER (Corea)", "051910.KS": "LG Chem (Corea)",
    "006400.KS": "Samsung SDI (Corea)", "035720.KS": "Kakao (Corea)",
    "003670.KS": "POSCO Holdings (Corea)", "105560.KS": "KB Financial (Corea)",
    "055550.KS": "Shinhan Financial (Corea)", "068270.KS": "Celltrion (Corea)",
    "005380.KS": "Hyundai Motor (Corea)", "000270.KS": "Kia (Corea)",
    "066570.KS": "LG Electronics (Corea)", "034730.KS": "SK Inc (Corea)",
    "017670.KS": "SK Telecom (Corea)",
    # India (NSE)
    "RELIANCE.NS": "Reliance Industries (India)", "TCS.NS": "TCS (India)",
    "HDFCBANK.NS": "HDFC Bank (India)", "ICICIBANK.NS": "ICICI Bank (India)",
    "BHARTIARTL.NS": "Bharti Airtel (India)", "ITC.NS": "ITC Ltd (India)",
    "SBIN.NS": "SBI (India)", "BAJFINANCE.NS": "Bajaj Finance (India)",
    "INFY.NS": "Infosys (India)", "LT.NS": "Larsen & Toubro (India)",
    "HINDUNILVR.NS": "Hindustan Unilever (India)", "KOTAKBANK.NS": "Kotak Mahindra (India)",
    "TATAMOTORS.NS": "Tata Motors (India)", "TATASTEEL.NS": "Tata Steel (India)",
    "SUNPHARMA.NS": "Sun Pharma (India)", "M&M.NS": "Mahindra & Mahindra (India)",
    "ASIANPAINT.NS": "Asian Paints (India)", "ADANIENT.NS": "Adani Enterprises (India)",
    "ADANIPORTS.NS": "Adani Ports (India)", "WIPRO.NS": "Wipro (India)",
    "POWERGRID.NS": "Power Grid India", "NTPC.NS": "NTPC (India)",
    "JSWSTEEL.NS": "JSW Steel (India)", "TITAN.NS": "Titan Company (India)",
    "ULTRACEMCO.NS": "UltraTech Cement (India)", "ONGC.NS": "ONGC (India)",
    "COALINDIA.NS": "Coal India", "MARUTI.NS": "Maruti Suzuki (India)",
    "HCLTECH.NS": "HCL Technologies (India)", "DRREDDY.NS": "Dr. Reddy's (India)",
    # Taiwan (TWSE)
    "2330.TW": "TSMC (Taiwan)", "2317.TW": "Hon Hai / Foxconn (Taiwan)",
    "2454.TW": "MediaTek (Taiwan)", "2412.TW": "Chunghwa Telecom (Taiwan)",
    "2881.TW": "Fubon Financial (Taiwan)", "2882.TW": "Cathay Financial (Taiwan)",
    "1301.TW": "Formosa Plastics (Taiwan)", "2308.TW": "Delta Electronics (Taiwan)",
    "3711.TW": "ASE Technology (Taiwan)", "2303.TW": "United Microelectronics (Taiwan)",
    # Australia – ASX 50 principales (Sydney)
    "CSL.AX": "CSL Limited (Australia)", "CBA.AX": "Comm Bank Australia",
    "WES.AX": "Wesfarmers (Australia)", "MQG.AX": "Macquarie Group (Australia)",
    "NAB.AX": "NAB (Australia)", "WOW.AX": "Woolworths (Australia)",
    "BHP.AX": "BHP Group (Australia)", "FMG.AX": "Fortescue Metals (Australia)",
    "WBC.AX": "Westpac (Australia)", "ANZ.AX": "ANZ Banking (Australia)",
    "RIO.AX": "Rio Tinto (Australia)", "WDS.AX": "Woodside Energy (Australia)",
    "TLS.AX": "Telstra (Australia)", "ALL.AX": "Aristocrat Leisure (Australia)",
    "GMG.AX": "Goodman Group (Australia)", "TCL.AX": "Transurban (Australia)",
    "COL.AX": "Coles Group (Australia)", "SHL.AX": "Sonic Healthcare (Australia)",
    "JHX.AX": "James Hardie (Australia)", "QBE.AX": "QBE Insurance (Australia)",
    "STO.AX": "Santos (Australia)", "ORG.AX": "Origin Energy (Australia)",
    "REA.AX": "REA Group (Australia)", "XRO.AX": "Xero (Australia)",
    "MIN.AX": "Mineral Resources (Australia)", "APA.AX": "APA Group (Australia)",
    "IAG.AX": "Insurance Aust Group", "AMC.AX": "Amcor (Australia)",
    "S32.AX": "South32 (Australia)", "COH.AX": "Cochlear (Australia)",
    "CPU.AX": "Computershare (Australia)", "TWE.AX": "Treasury Wine (Australia)",
    "NCM.AX": "Newcrest Mining (Australia)", "MPL.AX": "Medibank (Australia)",
    "ALD.AX": "Ampol (Australia)", "DXS.AX": "Dexus (Australia)",
    "SGP.AX": "Stockland (Australia)", "ORI.AX": "Orica (Australia)",
    "SUN.AX": "Suncorp (Australia)", "RMD.AX": "ResMed (Australia)",
    "SCG.AX": "Scentre Group (Australia)", "MGR.AX": "Mirvac (Australia)",
    "IEL.AX": "IDP Education (Australia)", "LYC.AX": "Lynas Rare Earths (Australia)",
    "PLS.AX": "Pilbara Minerals (Australia)", "NHF.AX": "nib Holdings (Australia)",
    "ALX.AX": "Atlas Arteria (Australia)", "LLC.AX": "Lendlease (Australia)",
    "CAR.AX": "CAR Group (Australia)", "SEK.AX": "Seek (Australia)",
    # Singapur
    "D05.SI": "DBS Group (Singapur)", "O39.SI": "OCBC Bank (Singapur)",
    "U11.SI": "UOB (Singapur)", "Z74.SI": "SingTel (Singapur)",
    "C38U.SI": "CapitaLand Invest (Singapur)", "Y92.SI": "Thai Beverage (Singapur)",
    "BN4.SI": "Keppel Corp (Singapur)",
    # Indonesia (Jakarta)
    "BBCA.JK": "Bank Central Asia (Indonesia)", "TLKM.JK": "Telkom Indonesia",
    "BBRI.JK": "Bank Rakyat Indonesia", "BMRI.JK": "Bank Mandiri (Indonesia)",
    "ASII.JK": "Astra International (Indonesia)", "UNVR.JK": "Unilever Indonesia",
    "ICBP.JK": "Indofood CBP (Indonesia)",
    # Tailandia
    "PTT.BK": "PTT (Tailandia)", "AOT.BK": "Airports of Thailand",
    "ADVANC.BK": "AIS (Tailandia)", "CPALL.BK": "CP All (Tailandia)",
    "SCC.BK": "Siam Cement (Tailandia)", "BDMS.BK": "Bangkok Dusit Medical (Tailandia)",
    # Malasia (Kuala Lumpur)
    "1155.KL": "Maybank (Malasia)", "1295.KL": "Public Bank (Malasia)",
    "6888.KL": "Petronas Chemicals (Malasia)", "3182.KL": "Genting (Malasia)",
    "5225.KL": "IHH Healthcare (Malasia)", "4715.KL": "Genting Malaysia",
    # Filipinas (Manila)
    "SM.PS": "SM Investments (Filipinas)", "BDO.PS": "BDO Unibank (Filipinas)",
    "TEL.PS": "PLDT (Filipinas)", "ALI.PS": "Ayala Land (Filipinas)",
    "JFC.PS": "Jollibee Foods (Filipinas)",
    # Vietnam
    "VNM.HM": "Vinamilk (Vietnam)", "VHM.HM": "Vinhomes (Vietnam)",
    "VIC.HM": "Vingroup (Vietnam)", "HPG.HM": "Hoa Phat (Vietnam)",
    "FPT.HM": "FPT Corp (Vietnam)",
    # Arabia Saudi (Tadawul)
    "2222.SR": "Saudi Aramco (Tadawul)", "1180.SR": "Al Rajhi Bank (Tadawul)",
    "2010.SR": "SABIC (Tadawul)", "7010.SR": "STC (Tadawul)",
    "1150.SR": "Saudi Awakening (Tadawul)", "2350.SR": "Saudi Kayan (Tadawul)",
    # Emiratos Arabes (Abu Dhabi / Dubai)
    "FAB.AE": "First Abu Dhabi Bank", "ADNOCDIST.AE": "ADNOC Distribution",
    "IHC.AE": "Intl Holding Company (Abu Dhabi)", "DFM.AE": "Dubai Financial Market",
    "EMAAR.AE": "Emaar Properties (Dubai)",
    # Israel (Tel Aviv)
    "LUMI.TA": "Bank Leumi (Tel Aviv)", "DSCT.TA": "Bank Discount (Tel Aviv)",
    "BEZQ.TA": "Bezeq (Tel Aviv)", "ICL.TA": "ICL Group (Tel Aviv)",
    "TEVA.TA": "Teva Pharma (Tel Aviv)",
    # Sudafrica (JSE Johannesburg)
    "NPN.JO": "Naspers (Johannesburg)", "SOL.JO": "Sasol (Johannesburg)",
    "SBK.JO": "Standard Bank (Johannesburg)", "FSR.JO": "FirstRand (Johannesburg)",
    "AGL.JO": "Anglo American (Johannesburg)", "BTI.JO": "BAT (Johannesburg)",
    "MTN.JO": "MTN Group (Johannesburg)", "VOD.JO": "Vodacom (Johannesburg)",
    "SHP.JO": "Shoprite (Johannesburg)", "CFR.JO": "Richemont (Johannesburg)",
    "BID.JO": "Bid Corp (Johannesburg)", "AMS.JO": "Anglo American Plat (Johannesburg)",
}

# ── Internacional: Latinoamerica locales ──
_LATAM_LOCAL: dict[str, str] = {
    # Brasil – Bovespa principales (B3 Sao Paulo)
    "PETR4.SA": "Petrobras (Brasil)", "VALE3.SA": "Vale (Brasil)",
    "ITUB4.SA": "Itau Unibanco (Brasil)", "BBDC4.SA": "Bradesco (Brasil)",
    "WEGE3.SA": "WEG (Brasil)", "RENT3.SA": "Localiza (Brasil)",
    "BBAS3.SA": "Banco do Brasil (Brasil)", "ABEV3.SA": "Ambev (Brasil)",
    "SUZB3.SA": "Suzano (Brasil)", "ELET3.SA": "Eletrobras (Brasil)",
    "B3SA3.SA": "B3 Bolsa Brasil", "HAPV3.SA": "Hapvida (Brasil)",
    "RAIL3.SA": "Rumo (Brasil)", "JBSS3.SA": "JBS (Brasil)",
    "RDOR3.SA": "Rede D'Or (Brasil)", "LREN3.SA": "Lojas Renner (Brasil)",
    "EQTL3.SA": "Equatorial Energia (Brasil)", "PRIO3.SA": "PRIO (Brasil)",
    "CSAN3.SA": "Cosan (Brasil)", "VIVT3.SA": "Telefonica Brasil",
    "SBSP3.SA": "Sabesp (Brasil)", "TOTS3.SA": "TOTVS (Brasil)",
    "RADL3.SA": "RaiaDrogasil (Brasil)", "ENEV3.SA": "Eneva (Brasil)",
    "BBSE3.SA": "BB Seguridade (Brasil)", "CMIG4.SA": "Cemig (Brasil)",
    "CPLE6.SA": "Copel (Brasil)", "EMBR3.SA": "Embraer (Brasil)",
    "VBBR3.SA": "Vibra Energia (Brasil)", "MGLU3.SA": "Magazine Luiza (Brasil)",
    # Mexico – BMV principales
    "FEMSAUBD.MX": "FEMSA (Mexico)", "WALMEX.MX": "Walmart Mexico",
    "GFNORTEO.MX": "Banorte (Mexico)", "CEMEXCPO.MX": "Cemex (Mexico)",
    "GMEXICOB.MX": "Grupo Mexico", "BIMBOA.MX": "Bimbo (Mexico)",
    "TABORAB.MX": "Organizacion Soriana (Mexico)",
    "AC.MX": "Arca Continental (Mexico)", "ALSEA.MX": "Alsea (Mexico)",
    "LIVEPOLC-1.MX": "Liverpool (Mexico)", "PE&OLES.MX": "Penoles (Mexico)",
    "GCARSOA1.MX": "Grupo Carso (Mexico)", "ORABORAB.MX": "Grupo Elektra (Mexico)",
    "GENTERA.MX": "Gentera (Mexico)", "MEGACPO.MX": "Megacable (Mexico)",
    "GRUMAB.MX": "Gruma (Mexico)",
    # Argentina (Buenos Aires)
    "GGAL.BA": "Grupo Galicia (Buenos Aires)", "YPF.BA": "YPF (Buenos Aires)",
    "BMA.BA": "Banco Macro (Buenos Aires)", "PAMP.BA": "Pampa Energia (Buenos Aires)",
    "TXAR.BA": "Ternium Argentina (Buenos Aires)", "CEPU.BA": "Central Puerto (Buenos Aires)",
    "TGSU2.BA": "TGS (Buenos Aires)", "SUPV.BA": "Supervielle (Buenos Aires)",
    # Colombia (BVC Bogota)
    "EC": "Ecopetrol (ADR Colombia)", "PFBCOLOM.CL": "Bancolombia (Bogota)",
    "ISA.CL": "ISA (Bogota)", "NUTRESA.CL": "Nutresa (Bogota)",
    "GRUPOARGOS.CL": "Grupo Argos (Bogota)",
    # Peru
    "BVN": "Buenaventura (ADR Peru)", "SCCO": "Southern Copper (Peru)",
    "BAP": "Credicorp (ADR Peru)", "IFS": "Intercorp Financial (Peru)",
    # Chile (Santiago)
    "FALABELLA.SN": "Falabella (Chile)", "CHILE.SN": "Banco de Chile",
    "CENCOSUD.SN": "Cencosud (Chile)", "SQM-B.SN": "SQM (Chile)",
    "COPEC.SN": "Copec (Chile)", "VAPORES.SN": "CSAV (Chile)",
    "BSANTANDER.SN": "Santander Chile", "CCU.SN": "CCU (Chile)",
    "ENELAM.SN": "Enel Americas (Chile)", "CMPC.SN": "CMPC (Chile)",
}

# ── ETFs (100) ──
_ETFS: dict[str, str] = {
    # Mercado amplio USA
    "SPY": "SPDR S&P 500", "QQQ": "Invesco NASDAQ 100", "IWM": "iShares Russell 2000",
    "DIA": "SPDR Dow Jones", "VOO": "Vanguard S&P 500", "VTI": "Vanguard Total Market",
    "MDY": "SPDR S&P MidCap 400", "RSP": "Invesco S&P 500 Equal Weight",
    # Internacional
    "EFA": "iShares MSCI EAFE", "VEA": "Vanguard FTSE Developed",
    "VWO": "Vanguard Emerging Markets", "IEMG": "iShares Core EM",
    "FXI": "iShares China Large-Cap", "MCHI": "iShares MSCI China",
    "EWJ": "iShares MSCI Japan", "EWZ": "iShares MSCI Brazil",
    "EWG": "iShares MSCI Germany", "EWU": "iShares MSCI UK",
    "EWQ": "iShares MSCI France", "INDA": "iShares MSCI India",
    "VNM": "VanEck Vietnam", "EZA": "iShares MSCI South Africa",
    "AFK": "VanEck Africa Index", "EWY": "iShares MSCI South Korea",
    "EWT": "iShares MSCI Taiwan", "EWH": "iShares MSCI Hong Kong",
    "THD": "iShares MSCI Thailand", "EPOL": "iShares MSCI Poland",
    "TUR": "iShares MSCI Turkey", "KWEB": "KraneShares China Internet",
    # Sectores USA
    "XLE": "Energy Select SPDR", "XLF": "Financial Select SPDR",
    "XLK": "Technology Select SPDR", "XLV": "Health Care Select SPDR",
    "XLRE": "Real Estate Select SPDR", "XLI": "Industrial Select SPDR",
    "XLP": "Consumer Staples Select SPDR", "XLY": "Consumer Discret Select SPDR",
    "XLB": "Materials Select SPDR", "XLU": "Utilities Select SPDR",
    "XLC": "Communication Svcs Select SPDR",
    # Tematicos
    "SMH": "VanEck Semiconductor", "ARKK": "ARK Innovation",
    "ARKG": "ARK Genomic Revolution", "ARKF": "ARK Fintech Innovation",
    "BOTZ": "Global X Robotics & AI", "ROBO": "ROBO Global Robotics",
    "HACK": "ETFMG Prime Cyber Security", "TAN": "Invesco Solar",
    "ICLN": "iShares Global Clean Energy", "LIT": "Global X Lithium & Battery",
    "URA": "Global X Uranium", "REMX": "VanEck Rare Earth/Strategic Metals",
    "COPX": "Global X Copper Miners", "SOXX": "iShares Semiconductor",
    "IGV": "iShares Expanded Tech-Software", "XBI": "SPDR S&P Biotech",
    "IBB": "iShares Biotechnology", "PAVE": "Global X US Infrastructure",
    "MOO": "VanEck Agribusiness",
    # Renta Fija
    "TLT": "iShares 20+ Yr Treasury", "IEF": "iShares 7-10 Yr Treasury",
    "SHY": "iShares 1-3 Yr Treasury", "TIP": "iShares TIPS Bond",
    "HYG": "iShares High Yield Corp", "EMB": "iShares JP EM Bond",
    "LQD": "iShares Invest Grade Corp", "AGG": "iShares Core US Agg Bond",
    "BND": "Vanguard Total Bond", "BNDX": "Vanguard Total Intl Bond",
    "VCIT": "Vanguard Intermed-Term Corp", "VCSH": "Vanguard Short-Term Corp",
    "MBB": "iShares MBS",
    # Commodities
    "GLD": "SPDR Gold Shares", "SLV": "iShares Silver Trust",
    "DBA": "Invesco DB Agriculture", "DBC": "Invesco DB Commodity",
    "PDBC": "Invesco Optimum Yield Commodity", "USO": "United States Oil Fund",
    "UNG": "United States Natural Gas", "CPER": "United States Copper Fund",
    # Dividendos / Valor
    "VYM": "Vanguard High Dividend Yield", "SCHD": "Schwab US Dividend",
    "HDV": "iShares Core High Dividend", "DVY": "iShares Select Dividend",
    "SDY": "SPDR S&P Dividend", "DGRW": "WisdomTree US Dividend Growth",
    # Crecimiento
    "VUG": "Vanguard Growth", "IWF": "iShares Russell 1000 Growth",
    "SPYG": "SPDR Portfolio S&P 500 Growth", "SPYV": "SPDR Portfolio S&P 500 Value",
    # Real Estate
    "VNQ": "Vanguard Real Estate", "IYR": "iShares US Real Estate",
    # Income / Overlay
    "JEPI": "JPMorgan Equity Premium Income", "JEPQ": "JPMorgan NASDAQ Equity Premium",
    # Cripto
    "BITO": "ProShares Bitcoin Strategy", "IBIT": "iShares Bitcoin Trust",
    "GBTC": "Grayscale Bitcoin Trust",
    # Multi-asset / Hedge
    "QQQM": "Invesco NASDAQ 100 ETF", "SPLG": "SPDR Portfolio S&P 500",
}

# ── COMMODITIES - Futuros (20) ──
_COMMODITIES: dict[str, str] = {
    "GC=F": "Oro", "SI=F": "Plata", "CL=F": "Petroleo WTI",
    "BZ=F": "Petroleo Brent", "NG=F": "Gas Natural", "HG=F": "Cobre",
    "PL=F": "Platino", "PA=F": "Paladio", "ZS=F": "Soja",
    "ZW=F": "Trigo", "ZC=F": "Maiz", "KC=F": "Cafe",
    "SB=F": "Azucar", "CT=F": "Algodon", "CC=F": "Cacao",
    "OJ=F": "Zumo Naranja", "LE=F": "Ganado Vivo", "HE=F": "Cerdo Magro",
    "RB=F": "Gasolina RBOB", "ALI=F": "Aluminio",
}

# ── CRIPTO (10) ──
_CRYPTO: dict[str, str] = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana",
    "BNB-USD": "BNB", "XRP-USD": "XRP", "ADA-USD": "Cardano",
    "DOT-USD": "Polkadot", "AVAX-USD": "Avalanche", "LINK-USD": "Chainlink",
    "DOGE-USD": "Dogecoin",
}

# ── DIVISAS (12) ──
_FOREX: dict[str, str] = {
    "EURUSD=X": "EUR/USD", "JPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD",
    "AUDUSD=X": "AUD/USD", "CNY=X": "USD/CNY", "MXN=X": "USD/MXN",
    "BRL=X": "USD/BRL", "INR=X": "USD/INR", "KRW=X": "USD/KRW",
    "CHF=X": "USD/CHF", "CAD=X": "USD/CAD", "NZDUSD=X": "NZD/USD",
}

# ── MACRO PROXIES (10) ──
_MACRO: dict[str, str] = {
    "^TNX": "Yield 10Y USA", "^TYX": "Yield 30Y USA", "^IRX": "Yield 3M USA",
    "^FVX": "Yield 5Y USA", "^VIX": "VIX (volatilidad S&P)",
    "DX-Y.NYB": "Indice Dolar (DXY)", "^SKEW": "SKEW (tail risk)",
    "^OVX": "OVX (vol petroleo)", "^GVZ": "GVZ (vol oro)",
    "^VVIX": "VVIX (vol del VIX)",
}


# ═══════════════════════════════════════════════════════════════════════════
# ENSAMBLAJE DEL UNIVERSO COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

TICKERS: dict[str, dict[str, str]] = {}
TICKERS.update(_build("Indice", _INDICES))
for _group in (_US_TECH, _US_HEALTH, _US_FINANCE, _US_CONSUMER_DISC,
               _US_CONSUMER_STAPLES, _US_INDUSTRIAL, _US_ENERGY,
               _US_MATERIALS, _US_REIT, _US_UTILITIES, _US_COMM, _US_EXTRA):
    TICKERS.update(_build("Accion", _group))
TICKERS.update(_build("Accion", _INTL_ADR))
TICKERS.update(_build("Accion", _EU_LOCAL))
TICKERS.update(_build("Accion", _ASIA_LOCAL))
TICKERS.update(_build("Accion", _LATAM_LOCAL))
TICKERS.update(_build("ETF", _ETFS))
TICKERS.update(_build("Commodity", _COMMODITIES))
TICKERS.update(_build("Cripto", _CRYPTO))
TICKERS.update(_build("Divisa", _FOREX))
TICKERS.update(_build("Macro", _MACRO))

# Top 50 acciones para fundamentales (mega caps + clave globales)
FUNDAMENTAL_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "BRK-B",
    "JPM", "V", "MA", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE",
    "XOM", "CVX", "AVGO", "TMO", "ORCL", "CRM", "ADBE", "AMD",
    "BAC", "WFC", "GS", "NKE", "COST", "WMT", "HD", "PG", "KO",
    "CAT", "BA", "NFLX", "DIS", "PLTR", "COIN", "CRSP", "ARM", "SMCI",
    "BABA", "TSM", "NVO", "SAP", "ASML", "TM", "VALE", "INFY",
]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════════════════

def _safe_pct(val: float | None) -> str:
    if val is None or pd.isna(val):
        return "N/D"
    return f"{val:+.1f}%"


def _safe_val(val: float | None, decimals: int = 2) -> str:
    if val is None or pd.isna(val):
        return "N/D"
    return f"{val:,.{decimals}f}"


def _compute_rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) < period + 1:
        return None
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    last_gain = gain.iloc[-1]
    last_loss = loss.iloc[-1]
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_sma(series: pd.Series, period: int) -> float | None:
    if len(series) < period:
        return None
    return series.rolling(window=period).mean().iloc[-1]


def _compute_macd(series: pd.Series) -> dict[str, float | None]:
    """Calcula MACD(12,26,9). Devuelve dict con macd, signal, histogram."""
    if len(series) < 35:
        return {"macd": None, "macd_signal": None, "macd_hist": None}
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal_line.iloc[-1]),
        "macd_hist": float(histogram.iloc[-1]),
    }


def _compute_sharpe(series: pd.Series, days: int = 126) -> float | None:
    """Sharpe Ratio anualizado sobre los ultimos *days* dias (por defecto 6m)."""
    tail = series.tail(days)
    if len(tail) < 30:
        return None
    rets = tail.pct_change().dropna()
    if rets.std() == 0:
        return None
    return float((rets.mean() / rets.std()) * (252 ** 0.5))


def _compute_bollinger(series: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, float | None]:
    """Bollinger Bands: upper, lower, %B (posicion relativa), bandwidth (squeeze)."""
    if len(series) < period:
        return {"bb_upper": None, "bb_lower": None, "bb_pct_b": None, "bb_bandwidth": None}
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    last_upper = float(upper.iloc[-1])
    last_lower = float(lower.iloc[-1])
    last_sma = float(sma.iloc[-1])
    last_price = float(series.iloc[-1])
    width = last_upper - last_lower
    pct_b = (last_price - last_lower) / width if width > 0 else None
    bandwidth = width / last_sma * 100 if last_sma > 0 else None
    return {
        "bb_upper": last_upper,
        "bb_lower": last_lower,
        "bb_pct_b": pct_b,
        "bb_bandwidth": bandwidth,
    }


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float | None:
    """ADX (Average Directional Index): mide fuerza de tendencia (0-100)."""
    if len(close) < period * 3:
        return None
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr
    di_sum = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, float("nan"))
    adx = dx.rolling(window=period).mean()
    last_vals = adx.dropna()
    return float(last_vals.iloc[-1]) if not last_vals.empty else None


def _detect_rsi_divergence(close: pd.Series, period: int = 14, lookback: int = 40) -> str:
    """Detecta divergencias RSI vs precio: 'bullish', 'bearish' o 'none'.

    Bullish: precio hace lower low pero RSI hace higher low.
    Bearish: precio hace higher high pero RSI hace lower high.
    """
    if len(close) < period + lookback:
        return "none"
    # Calcular RSI para los ultimos lookback dias
    rsi_full = []
    for i in range(lookback):
        idx = len(close) - lookback + i
        sub = close.iloc[:idx + 1]
        if len(sub) >= period + 1:
            rsi_full.append(_compute_rsi(sub, period))
        else:
            rsi_full.append(None)
    rsi_series = pd.Series(rsi_full)
    price_series = close.tail(lookback).reset_index(drop=True)
    # Dividir en dos mitades y buscar extremos
    mid = lookback // 2
    p_first = price_series.iloc[:mid]
    p_second = price_series.iloc[mid:]
    r_first = rsi_series.iloc[:mid]
    r_second = rsi_series.iloc[mid:]
    # Bullish: precio lower low, RSI higher low
    p_low1 = p_first.min()
    p_low2 = p_second.min()
    r_low1 = r_first.dropna().min() if not r_first.dropna().empty else None
    r_low2 = r_second.dropna().min() if not r_second.dropna().empty else None
    if r_low1 is not None and r_low2 is not None:
        if p_low2 < p_low1 and r_low2 > r_low1:
            return "bullish"
    # Bearish: precio higher high, RSI lower high
    p_high1 = p_first.max()
    p_high2 = p_second.max()
    r_high1 = r_first.dropna().max() if not r_first.dropna().empty else None
    r_high2 = r_second.dropna().max() if not r_second.dropna().empty else None
    if r_high1 is not None and r_high2 is not None:
        if p_high2 > p_high1 and r_high2 < r_high1:
            return "bearish"
    return "none"


# ═══════════════════════════════════════════════════════════════════════════
# DESCARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════

def fetch_price_data(months: int = 12) -> dict[str, dict[str, Any]]:
    """Descarga datos de precios para todos los tickers del universo."""
    end = datetime.now()
    start = end - timedelta(days=max(months * 30, 365) + 60)

    tickers_list = list(TICKERS.keys())
    print(f"  Descargando precios de {len(tickers_list)} activos...", flush=True)

    try:
        data = yf.download(
            tickers_list,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False,
            threads=True,
        )
    except Exception as exc:
        print(f"  Error descargando datos: {exc}", file=sys.stderr)
        return {}

    results: dict[str, dict[str, Any]] = {}

    # Pre-calcular retorno del S&P 500 para fuerza relativa
    _sp500_rets: dict[str, float | None] = {}
    try:
        if len(tickers_list) > 1 and "^GSPC" in tickers_list:
            sp_close = data["Close"]["^GSPC"].dropna()
        else:
            sp_close = pd.Series(dtype=float)
        if not sp_close.empty:
            for d_label, d_days in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
                if len(sp_close) >= d_days + 1:
                    _sp500_rets[d_label] = ((sp_close.iloc[-1] / sp_close.iloc[-d_days - 1]) - 1) * 100
    except Exception:
        pass

    failed_tickers: list[str] = []
    for ticker, meta in TICKERS.items():
        try:
            if len(tickers_list) > 1:
                close = data["Close"][ticker].dropna()
            else:
                close = data["Close"].dropna()

            if close.empty:
                failed_tickers.append(ticker)
                continue

            precio_actual = close.iloc[-1]

            def _ret(days: int) -> float | None:
                if len(close) < days + 1:
                    return None
                return ((close.iloc[-1] / close.iloc[-days - 1]) - 1) * 100

            ret_1w = _ret(5)
            ret_1m = _ret(21)
            ret_3m = _ret(63)
            ret_6m = _ret(126)
            ret_12m = _ret(252)

            sma50 = _compute_sma(close, 50)
            sma200 = _compute_sma(close, 200)
            rsi14 = _compute_rsi(close, 14)

            vol_20d = None
            if len(close) > 20:
                vol_20d = close.pct_change().tail(20).std() * (252 ** 0.5) * 100

            last_year = close.tail(252)
            max_dd = None
            if len(last_year) > 1:
                roll_max = last_year.cummax()
                drawdown = ((last_year - roll_max) / roll_max) * 100
                max_dd = drawdown.min()

            high_52w = close.tail(252).max()
            dist_high = ((precio_actual / high_52w) - 1) * 100 if high_52w else None

            # --- NUEVOS INDICADORES ---
            # MACD(12,26,9)
            macd_data = _compute_macd(close)

            # Sharpe Ratio 6 meses
            sharpe_6m = _compute_sharpe(close, 126)

            # Volumen medio 20d y ratio volumen actual / media
            avg_vol_20d = None
            vol_ratio = None
            try:
                if len(tickers_list) > 1:
                    vol_series = data["Volume"][ticker].dropna()
                else:
                    vol_series = data["Volume"].dropna()
                if len(vol_series) >= 20:
                    avg_vol_20d = float(vol_series.tail(20).mean())
                    last_vol = float(vol_series.iloc[-1])
                    if avg_vol_20d > 0:
                        vol_ratio = last_vol / avg_vol_20d
                volume_hist = vol_series.tolist() if len(vol_series) > 0 else []
            except Exception:
                volume_hist = []

            # Fuerza relativa vs S&P 500 (exceso de retorno)
            rs_vs_sp500 = None
            if ret_12m is not None and _sp500_rets.get("12m") is not None:
                rs_vs_sp500 = ret_12m - _sp500_rets["12m"]

            # Momentum compuesto (1m×0.15 + 3m×0.25 + 6m×0.30 + 12m×0.30)
            momentum_score = None
            if all(r is not None for r in [ret_1m, ret_3m, ret_6m, ret_12m]):
                momentum_score = ret_1m * 0.15 + ret_3m * 0.25 + ret_6m * 0.30 + ret_12m * 0.30

            # Bollinger Bands (20, 2σ)
            bb_data = _compute_bollinger(close)

            # OHLC historico (para graficos de velas)
            open_hist: list[float] = []
            high_hist: list[float] = []
            low_hist: list[float] = []
            dates_hist: list[str] = []
            try:
                if len(tickers_list) > 1:
                    open_s = data["Open"][ticker].dropna()
                    high_s = data["High"][ticker].dropna()
                    low_s = data["Low"][ticker].dropna()
                else:
                    open_s = data["Open"].dropna()
                    high_s = data["High"].dropna()
                    low_s = data["Low"].dropna()
                open_hist = open_s.tolist()
                high_hist = high_s.tolist()
                low_hist = low_s.tolist()
                dates_hist = [d.strftime("%Y-%m-%d") for d in close.index]
            except Exception:
                open_s = pd.Series(dtype=float)
                high_s = pd.Series(dtype=float)
                low_s = pd.Series(dtype=float)

            # ADX (14) - necesita High y Low
            adx_val = None
            try:
                # Alinear indices
                if not high_s.empty and not low_s.empty:
                    common_idx = close.index.intersection(high_s.index).intersection(low_s.index)
                    if len(common_idx) > 42:
                        adx_val = _compute_adx(
                            high_s.loc[common_idx],
                            low_s.loc[common_idx],
                            close.loc[common_idx],
                        )
            except Exception:
                pass

            # RSI Divergencia
            rsi_divergence = _detect_rsi_divergence(close)

            results[ticker] = {
                "nombre": meta["nombre"], "tipo": meta["tipo"],
                "precio": precio_actual,
                "ret_1w": ret_1w, "ret_1m": ret_1m, "ret_3m": ret_3m,
                "ret_6m": ret_6m, "ret_12m": ret_12m,
                "sma50": sma50, "sma200": sma200, "rsi14": rsi14,
                "vol_20d": vol_20d, "max_dd_12m": max_dd, "dist_high_52w": dist_high,
                "close_hist": close.tolist(),
                "open_hist": open_hist,
                "high_hist": high_hist,
                "low_hist": low_hist,
                "dates_hist": dates_hist,
                "volume_hist": volume_hist,
                # Indicadores de momentum
                "macd": macd_data["macd"],
                "macd_signal": macd_data["macd_signal"],
                "macd_hist": macd_data["macd_hist"],
                "sharpe_6m": sharpe_6m,
                "avg_vol_20d": avg_vol_20d,
                "vol_ratio": vol_ratio,
                "rs_vs_sp500": rs_vs_sp500,
                "momentum_score": momentum_score,
                # Bollinger Bands
                "bb_upper": bb_data["bb_upper"],
                "bb_lower": bb_data["bb_lower"],
                "bb_pct_b": bb_data["bb_pct_b"],
                "bb_bandwidth": bb_data["bb_bandwidth"],
                # ADX + RSI divergencia
                "adx": adx_val,
                "rsi_divergence": rsi_divergence,
            }
        except Exception:
            failed_tickers.append(ticker)
            continue

    print(f"  Datos obtenidos para {len(results)} de {len(tickers_list)} activos.", flush=True)
    if failed_tickers:
        print(f"  Sin datos ({len(failed_tickers)}): {', '.join(sorted(failed_tickers))}", flush=True)
    return results


def _fetch_single_fundamental(ticker_str: str) -> tuple[str, dict[str, Any] | None]:
    """Obtiene fundamentales de un solo ticker (para threading)."""
    try:
        tk = yf.Ticker(ticker_str)
        info = tk.info
        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        earnings_growth = info.get("earningsQuarterlyGrowth") or info.get("earningsGrowth")
        peg = info.get("pegRatio")
        # Calcular PEG manualmente si no viene y tenemos forward PE + crecimiento
        if peg is None and forward_pe and earnings_growth and earnings_growth > 0:
            peg = forward_pe / (earnings_growth * 100)
        return ticker_str, {
            "per": trailing_pe or forward_pe,
            "forward_pe": forward_pe,
            "peg": peg,
            "roe": info.get("returnOnEquity"),
            "margen_neto": info.get("profitMargins"),
            "deuda_equity": info.get("debtToEquity"),
            "market_cap": info.get("marketCap"),
            "crec_ingresos": info.get("revenueGrowth"),
            "earnings_growth": earnings_growth,
            "div_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "sector": info.get("sector", "N/D"),
            # Datos institucionales / analistas (de .info)
            "target_mean_price": info.get("targetMeanPrice"),
            "target_low_price": info.get("targetLowPrice"),
            "target_high_price": info.get("targetHighPrice"),
            "recommendation_mean": info.get("recommendationMean"),  # 1=strong buy .. 5=strong sell
            "recommendation_key": info.get("recommendationKey", ""),
            "num_analysts": info.get("numberOfAnalystOpinions"),
            "insider_pct": info.get("heldPercentInsiders"),
            "institutional_pct": info.get("heldPercentInstitutions"),
            "short_ratio": info.get("shortRatio"),
            "short_pct_float": info.get("shortPercentOfFloat"),
            "current_price": info.get("currentPrice") or info.get("previousClose"),
        }
    except Exception:
        return ticker_str, None


def fetch_fundamentals(max_workers: int = 0) -> dict[str, dict[str, Any]]:
    """Obtiene metricas fundamentales con descarga concurrente."""
    if max_workers <= 0:
        max_workers = _DEFAULT_WORKERS
    n = len(FUNDAMENTAL_TICKERS)
    print(f"  Obteniendo fundamentales de {n} acciones (concurrente x{max_workers})...", flush=True)
    results: dict[str, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_fundamental, t): t
            for t in FUNDAMENTAL_TICKERS
        }
        done = 0
        for future in as_completed(futures):
            ticker_str, data = future.result()
            done += 1
            if data is not None:
                results[ticker_str] = data
            if done % 10 == 0:
                print(f"    ... {done}/{n} procesados", flush=True)

    print(f"  Fundamentales obtenidos para {len(results)} acciones.", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# FORMATEO DE BRIEFINGS
# ═══════════════════════════════════════════════════════════════════════════

def format_market_briefing(prices: dict[str, dict], fundamentals: dict[str, dict]) -> str:
    """Briefing general para los agentes del debate.

    Incluye: macro, divisas, indices, commodities, cripto,
    top movers, resumen sectorial, y mega caps con fundamentales.
    """
    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== DATOS REALES DE MERCADO ({today}) - {len(prices)} activos monitorizados ===\n")

    # MACRO
    lines.append("-- INDICADORES MACRO --")
    for t, d in prices.items():
        if d["tipo"] == "Macro":
            lines.append(f"  {d['nombre']}: {_safe_val(d['precio'])} | 1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])}")

    # DIVISAS
    lines.append("\n-- DIVISAS --")
    for t, d in prices.items():
        if d["tipo"] == "Divisa":
            lines.append(f"  {d['nombre']}: {_safe_val(d['precio'], 4)} | 1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])}")

    # INDICES
    lines.append("\n-- INDICES GLOBALES --")
    for t, d in prices.items():
        if d["tipo"] == "Indice":
            lines.append(
                f"  {d['nombre']}: {_safe_val(d['precio'], 0)} | "
                f"1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])} | "
                f"12m: {_safe_pct(d['ret_12m'])} | SMA200: {_safe_val(d.get('sma200'), 0)}"
            )

    # COMMODITIES
    lines.append("\n-- COMMODITIES --")
    for t, d in prices.items():
        if d["tipo"] == "Commodity":
            lines.append(
                f"  {d['nombre']}: {_safe_val(d['precio'])} | "
                f"1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])} | "
                f"12m: {_safe_pct(d['ret_12m'])}"
            )

    # CRIPTO
    lines.append("\n-- CRIPTO --")
    for t, d in prices.items():
        if d["tipo"] == "Cripto":
            lines.append(
                f"  {d['nombre']}: {_safe_val(d['precio'], 0)} | "
                f"1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])} | "
                f"12m: {_safe_pct(d['ret_12m'])}"
            )

    # TOP MOVERS: 10 mejores y 10 peores en 1 mes (acciones + ETFs)
    tradeable = [(t, d) for t, d in prices.items()
                 if d["tipo"] in ("Accion", "ETF") and d.get("ret_1m") is not None]
    tradeable.sort(key=lambda x: x[1]["ret_1m"])

    if tradeable:
        lines.append("\n-- TOP 10 PEORES (1 mes) --")
        for t, d in tradeable[:10]:
            lines.append(f"  {d['nombre']} ({t}): {_safe_val(d['precio'])} | 1m: {_safe_pct(d['ret_1m'])} | 12m: {_safe_pct(d.get('ret_12m'))}")

        lines.append("\n-- TOP 10 MEJORES (1 mes) --")
        for t, d in tradeable[-10:]:
            lines.append(f"  {d['nombre']} ({t}): {_safe_val(d['precio'])} | 1m: {_safe_pct(d['ret_1m'])} | 12m: {_safe_pct(d.get('ret_12m'))}")

    # SECTOR ETFs
    sector_etfs = ["XLE", "XLF", "XLK", "XLV", "XLRE", "XLI", "XLP", "XLY", "XLB", "XLU", "XLC"]
    lines.append("\n-- RENDIMIENTO SECTORIAL (ETFs sectoriales) --")
    for t in sector_etfs:
        d = prices.get(t)
        if d:
            lines.append(f"  {d['nombre']}: 1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])} | 12m: {_safe_pct(d['ret_12m'])}")

    # MEGA CAPS con fundamentales
    lines.append("\n-- ACCIONES CLAVE (precio + fundamentales) --")
    for ticker in FUNDAMENTAL_TICKERS:
        if ticker not in prices:
            continue
        d = prices[ticker]
        f = fundamentals.get(ticker, {})
        per_str = _safe_val(f.get("per"), 1) if f.get("per") else "N/D"
        roe_str = _safe_pct(f.get("roe", 0) * 100) if f.get("roe") else "N/D"
        beta_str = _safe_val(f.get("beta"), 2) if f.get("beta") else "N/D"
        lines.append(
            f"  {d['nombre']} ({ticker}): ${_safe_val(d['precio'])} | "
            f"1m: {_safe_pct(d['ret_1m'])} | 12m: {_safe_pct(d['ret_12m'])} | "
            f"PER: {per_str} | ROE: {roe_str} | Beta: {beta_str}"
        )

    # ETFs PRINCIPALES
    top_etfs = [
        "SPY", "QQQ", "IWM", "VWO", "EFA", "FXI", "EWJ", "EWZ", "INDA", "VNM",
        "TLT", "HYG", "EMB", "TIP", "LQD", "AGG",
        "GLD", "SLV", "DBA", "USO",
        "SMH", "ARKK", "XBI", "URA", "LIT", "KWEB",
        "VYM", "SCHD", "VNQ", "BITO",
    ]
    lines.append("\n-- ETFs PRINCIPALES --")
    for t in top_etfs:
        d = prices.get(t)
        if d:
            lines.append(
                f"  {d['nombre']} ({t}): ${_safe_val(d['precio'])} | "
                f"1m: {_safe_pct(d['ret_1m'])} | 3m: {_safe_pct(d['ret_3m'])} | "
                f"12m: {_safe_pct(d['ret_12m'])}"
            )

    return "\n".join(lines)


def format_technical_briefing(prices: dict[str, dict]) -> str:
    """Briefing tecnico completo - incluye TODOS los activos."""
    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== DATOS TECNICOS REALES ({today}) - {len(prices)} activos ===\n")
    lines.append(f"{'Ticker':<12} {'Nombre':<25} {'Precio':>10} {'SMA50':>10} {'SMA200':>10} {'RSI':>5} {'Vol%':>6} {'DD%':>7} {'v52H%':>7}")
    lines.append("-" * 100)

    for ticker, d in sorted(prices.items(), key=lambda x: x[1]["tipo"]):
        lines.append(
            f"{ticker:<12} {d['nombre'][:24]:<25} {_safe_val(d['precio']):>10} "
            f"{_safe_val(d.get('sma50')):>10} {_safe_val(d.get('sma200')):>10} "
            f"{_safe_val(d.get('rsi14'), 0):>5} {_safe_pct(d.get('vol_20d')):>6} "
            f"{_safe_pct(d.get('max_dd_12m')):>7} {_safe_pct(d.get('dist_high_52w')):>7}"
        )

    return "\n".join(lines)


def format_fundamental_briefing(fundamentals: dict[str, dict], prices: dict[str, dict]) -> str:
    """Briefing fundamental detallado."""
    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== DATOS FUNDAMENTALES REALES ({today}) ===\n")
    lines.append(f"{'Accion':<20} {'PER':>8} {'PEG':>7} {'ROE':>8} {'Margen':>8} {'D/E':>8} {'CrecIng':>8} {'DivYld':>7} {'Beta':>6} {'Sector':<20}")
    lines.append("-" * 115)

    for ticker, f in fundamentals.items():
        per = _safe_val(f.get("per"), 1) if f.get("per") else "N/D"
        peg = _safe_val(f.get("peg"), 2) if f.get("peg") else "N/D"
        roe = _safe_pct(f.get("roe", 0) * 100) if f.get("roe") else "N/D"
        margen = _safe_pct(f.get("margen_neto", 0) * 100) if f.get("margen_neto") else "N/D"
        de = _safe_val(f.get("deuda_equity"), 1) if f.get("deuda_equity") else "N/D"
        crec = _safe_pct(f.get("crec_ingresos", 0) * 100) if f.get("crec_ingresos") else "N/D"
        div = _safe_pct(f.get("div_yield", 0) * 100) if f.get("div_yield") else "N/D"
        beta = _safe_val(f.get("beta"), 2) if f.get("beta") else "N/D"
        nombre = prices.get(ticker, {}).get("nombre", ticker)
        sector = f.get("sector", "N/D")[:19]
        lines.append(
            f"{nombre:<20} {per:>8} {peg:>7} {roe:>8} {margen:>8} {de:>8} {crec:>8} {div:>7} {beta:>6} {sector:<20}"
        )

    return "\n".join(lines)


def format_risk_briefing(prices: dict[str, dict]) -> str:
    """Briefing de riesgos: volatilidad, drawdown, para todos los activos."""
    lines: list[str] = []
    today = datetime.now().strftime("%d/%m/%Y")
    lines.append(f"=== DATOS DE RIESGO REALES ({today}) - {len(prices)} activos ===\n")
    lines.append(f"{'Ticker':<12} {'Nombre':<25} {'Vol20d':>8} {'MaxDD12m':>9} {'vs52wH':>8} {'Ret12m':>8}")
    lines.append("-" * 78)

    for ticker, d in sorted(prices.items(), key=lambda x: x[1].get("vol_20d") or 0, reverse=True):
        lines.append(
            f"{ticker:<12} {d['nombre'][:24]:<25} {_safe_pct(d.get('vol_20d')):>8} "
            f"{_safe_pct(d.get('max_dd_12m')):>9} "
            f"{_safe_pct(d.get('dist_high_52w')):>8} "
            f"{_safe_pct(d.get('ret_12m')):>8}"
        )

    vix = prices.get("^VIX")
    if vix:
        lines.append(f"\nVIX actual: {_safe_val(vix['precio'], 1)} (<15 calma, 15-25 normal, >25 estres, >35 panico)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# FUNCION PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def load_all_market_data(months: int = 12, include_raw: bool = False) -> dict[str, Any]:
    """Carga todos los datos y devuelve los briefings formateados.

    Usa cache local (pickle) con TTL de 1 hora para evitar re-descargar.
    Si include_raw=True, tambien incluye estructuras crudas:
    - raw_prices: dict con precios/metricas por ticker
    - raw_fundamentals: dict con fundamentales por ticker
    """
    cache_key = f"market_data_m{months}_raw{int(include_raw)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        print("\n" + "=" * 60)
        print("DATOS DE MERCADO CARGADOS DESDE CACHE")
        print(f"(TTL: {_CACHE_TTL_SECONDS}s — para forzar recarga, borra .cache/)")
        print("=" * 60 + "\n")
        return cached

    # Limpiar cache obsoleta (>3 dias)
    removed = _cache_cleanup()
    if removed:
        print(f"  Cache: {removed} archivos obsoletos eliminados.")

    print("\n" + "=" * 60)
    print("CARGANDO DATOS REALES DE MERCADO")
    print(f"Universo: {len(TICKERS)} activos | Workers: {_DEFAULT_WORKERS}")
    print("=" * 60)

    # Descargar precios y fundamentales en paralelo (son independientes)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_prices = pool.submit(fetch_price_data, months)
        fut_fundamentals = pool.submit(fetch_fundamentals)
        prices = fut_prices.result()
        fundamentals = fut_fundamentals.result()

    # Liberar memoria de DataFrames temporales de yfinance
    gc.collect()

    if not prices:
        print("  ADVERTENCIA: No se pudieron obtener datos de mercado.", file=sys.stderr)
        return {}

    tipos: dict[str, int] = {}
    for d in prices.values():
        tipos[d["tipo"]] = tipos.get(d["tipo"], 0) + 1
    for tipo, count in sorted(tipos.items()):
        print(f"  {tipo}: {count} activos cargados")

    briefings = {
        "general": format_market_briefing(prices, fundamentals),
        "tecnico": format_technical_briefing(prices),
        "fundamental": format_fundamental_briefing(fundamentals, prices),
        "riesgo": format_risk_briefing(prices),
    }

    if include_raw:
        briefings["raw_prices"] = prices
        briefings["raw_fundamentals"] = fundamentals

    print(f"\nBriefing general: {len(briefings['general']):,} chars ({briefings['general'].count(chr(10))} lineas)")
    print(f"Briefing tecnico: {len(briefings['tecnico']):,} chars ({briefings['tecnico'].count(chr(10))} lineas)")
    print(f"Briefing fundamental: {len(briefings['fundamental']):,} chars")
    print(f"Briefing riesgo: {len(briefings['riesgo']):,} chars ({briefings['riesgo'].count(chr(10))} lineas)")
    print("=" * 60 + "\n")

    _cache_set(cache_key, briefings)
    return briefings


if __name__ == "__main__":
    tipos: dict[str, int] = {}
    for d in TICKERS.values():
        tipos[d["tipo"]] = tipos.get(d["tipo"], 0) + 1
    print(f"UNIVERSO TOTAL: {len(TICKERS)} activos")
    for tipo, count in sorted(tipos.items(), key=lambda x: -x[1]):
        print(f"  {tipo}: {count}")
    print(f"Fundamentales: {len(FUNDAMENTAL_TICKERS)} acciones\n")

    briefings = load_all_market_data()
    for key, text in briefings.items():
        print(f"\n{'='*40} {key.upper()} {'='*40}")
        preview = "\n".join(text.split("\n")[:20])
        print(preview)
        print(f"... ({text.count(chr(10))} lineas totales)")
