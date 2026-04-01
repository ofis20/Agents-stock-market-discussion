# Debate Macro con 7 Agentes (Ollama local)

Programa en Python que crea un debate en vivo entre tres agentes inspirados en enfoques de:

- Warren Buffett (valor y largo plazo)
- Peter Lynch (crecimiento entendible)
- Stanley Druckenmiller (macro y liquidez)

Tras un tiempo objetivo (por defecto 50 segundos), el sistema fuerza un consenso final y muestra un resumen de 10 lineas.

## Requisitos

- Tener Ollama instalado y corriendo en local.
- Tener descargado un modelo (ejemplo: `ollama pull llama3.1`).

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
python ollama_macro_debate.py --model llama3.1 --seconds 50
```

## Ejecucion con Streamlit (recomendado para visualizar mejor)

```bash
streamlit run streamlit_app.py
```

Desde la interfaz puedes:

- Ajustar modelo, host y parametros del debate.
- Ver toda la salida en tiempo real.
- Descargar el log completo al terminar.

Opciones utiles:

- `--host http://127.0.0.1:11434`
- `--max-turns 24`
- `--context-lines 18`

## Nota

El programa imprime el debate en streaming para que lo leas mientras se escribe.
