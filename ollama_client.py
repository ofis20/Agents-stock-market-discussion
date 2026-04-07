"""Cliente Ollama, politicas de modelos y streaming."""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

import requests


if TYPE_CHECKING:
    from debate_agents import Agent


DEFAULT_HOST = "http://127.0.0.1:11434"

# Opciones de rendimiento para Ollama (configurables via env)
_NUM_THREAD = int(os.environ.get("OLLAMA_NUM_THREAD", 0)) or os.cpu_count() or 4
_NUM_GPU = int(os.environ.get("OLLAMA_NUM_GPU", 99))    # 99 = todas las capas en GPU
_NUM_BATCH = int(os.environ.get("OLLAMA_NUM_BATCH", 512))
_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


PREFERRED_MODELS = [
    # Qwen 3.5 (ultima generacion, 256K contexto, multimodal)
    "qwen3.5:27b", "qwen3.5:9b",
    # Gemma 4 (ultima gen Google, MoE, thinking, 128K contexto)
    "gemma4:26b", "gemma4:e4b",
    # Modelos grandes (buen razonamiento)
    "mistral-small:22b",
    "phi4:14b",
    "qwen2.5:14b",
    "mistral-nemo:12b",
]


CODE_MODELS = {"deepseek-coder", "codellama", "starcoder", "codegemma", "codeqwen"}


def is_code_model(name: str) -> bool:
    """Devuelve True si el modelo es de codigo (no apto para debate)."""
    base = name.split(":")[0].lower()
    return base in CODE_MODELS


def filter_general_models(available_models: list[str]) -> list[str]:
    general_models = [model for model in available_models if not is_code_model(model)]
    return general_models or available_models[:]


def sort_models_by_preference(models: list[str]) -> list[str]:
    def _priority(model_name: str) -> int:
        for idx, preferred in enumerate(PREFERRED_MODELS):
            if model_name == preferred or model_name.startswith(preferred.split(":")[0]):
                return idx
        return len(PREFERRED_MODELS)

    return sorted(models, key=_priority)


def assign_models_to_agents(available_models: list[str], agents: list["Agent"], analyst_roles: list[str]) -> dict[str, str]:
    """Asigna modelos diversos a agentes y analistas."""
    general_models = sort_models_by_preference(filter_general_models(available_models))

    if not general_models:
        raise RuntimeError("No hay modelos instalados en Ollama.")

    all_roles = [agent.name for agent in agents] + analyst_roles
    assignment: dict[str, str] = {}
    available_for_auto = general_models[:]

    for agent in agents:
        if agent.model and agent.model in available_models:
            assignment[agent.name] = agent.model

    idx = 0
    for role in all_roles:
        if role not in assignment:
            assignment[role] = available_for_auto[idx % len(available_for_auto)]
            idx += 1

    return assignment


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST):
        self.host = host.rstrip("/")
        self.available_models: list[str] = []

    def check_ollama(self, timeout: float = 4.0) -> list[str]:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            models = [m.get("name", "") for m in payload.get("models", []) if m.get("name")]
            return models
        except Exception as exc:
            raise RuntimeError(
                "No se pudo conectar con Ollama local. "
                "Asegurate de tener el servicio activo (ej: `ollama serve`)."
            ) from exc

    def refresh_models(self, timeout: float = 4.0) -> list[str]:
        self.available_models = self.check_ollama(timeout=timeout)
        return self.available_models

    def resolve_model(self, requested_model: str) -> str:
        if not self.available_models:
            raise RuntimeError("No hay modelos instalados en Ollama. Ejecuta, por ejemplo: `ollama pull qwen2.5:7b`.")

        if requested_model in self.available_models:
            return requested_model

        fallback = self.available_models[0]
        print(
            f"Aviso: el modelo '{requested_model}' no esta instalado. Se usara '{fallback}'.",
            file=sys.stderr,
        )
        return fallback

    def _stream_chat_single(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_predict: int = 2048,
        temperature: float = 0.5,
        silent: bool = False,
    ) -> str:
        """Intenta una sola llamada a /api/chat (sin retry)."""
        chat_url = f"{self.host}/api/chat"
        chat_payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_thread": _NUM_THREAD,
                "num_gpu": _NUM_GPU,
                "num_batch": _NUM_BATCH,
            },
        }

        full_text = ""
        chat_resp = requests.post(chat_url, json=chat_payload, stream=True, timeout=120)

        if chat_resp.status_code == 404:
            chat_resp.close()
            return self.stream_generate(
                model=model,
                messages=messages,
                num_predict=num_predict,
                temperature=temperature,
                silent=silent,
            )

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

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_predict: int = 2048,
        temperature: float = 0.5,
        silent: bool = False,
    ) -> str:
        """Llama a /api/chat con retry automatico usando modelos alternativos si falla."""
        try:
            return self._stream_chat_single(model, messages, num_predict, temperature, silent)
        except (requests.RequestException, OSError) as first_err:
            alternates = [model_name for model_name in self.available_models if model_name != model and not is_code_model(model_name)]
            if not alternates:
                raise
            for alt_model in alternates:
                print(f"\n[RETRY: {model} fallo ({first_err}). Probando {alt_model}...]", flush=True)
                try:
                    return self._stream_chat_single(alt_model, messages, num_predict, temperature, silent)
                except (requests.RequestException, OSError):
                    continue
            raise first_err

    def stream_generate(
        self,
        model: str,
        messages: list[dict[str, str]],
        num_predict: int = 2048,
        temperature: float = 0.5,
        silent: bool = False,
    ) -> str:
        """Fallback a /api/generate para servidores Ollama sin endpoint /api/chat."""
        generate_url = f"{self.host}/api/generate"

        prompt_lines = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                prompt_lines.append(f"[INSTRUCCIONES]\n{message.get('content', '')}\n")
            elif role == "user":
                prompt_lines.append(f"[USUARIO]\n{message.get('content', '')}\n")
            else:
                prompt_lines.append(f"[ASISTENTE]\n{message.get('content', '')}\n")
        prompt_lines.append("[ASISTENTE]\n")
        prompt = "\n".join(prompt_lines)

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": _KEEP_ALIVE,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_thread": _NUM_THREAD,
                "num_gpu": _NUM_GPU,
                "num_batch": _NUM_BATCH,
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