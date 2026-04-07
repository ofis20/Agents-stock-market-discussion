"""Definicion declarativa de agentes del debate."""

from __future__ import annotations

from dataclasses import dataclass

from debate_prompts import AGENT_SYSTEM_PROMPTS


@dataclass(frozen=True)
class Agent:
    name: str
    system_prompt: str
    model: str = ""
    temperature: float = 0.5
    max_tokens: int = 420
    role: str = "debate"


AGENTS = [
    Agent(name="Warren Buffett", system_prompt=AGENT_SYSTEM_PROMPTS["Warren Buffett"]),
    Agent(name="Peter Lynch", system_prompt=AGENT_SYSTEM_PROMPTS["Peter Lynch"]),
    Agent(name="Stanley Druckenmiller", system_prompt=AGENT_SYSTEM_PROMPTS["Stanley Druckenmiller"]),
    Agent(name="Ray Dalio", system_prompt=AGENT_SYSTEM_PROMPTS["Ray Dalio"]),
    Agent(name="Cathie Wood", system_prompt=AGENT_SYSTEM_PROMPTS["Cathie Wood"]),
    Agent(name="Howard Marks", system_prompt=AGENT_SYSTEM_PROMPTS["Howard Marks"]),
    Agent(name="Jim Rogers", system_prompt=AGENT_SYSTEM_PROMPTS["Jim Rogers"]),
]