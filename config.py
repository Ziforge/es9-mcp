"""Configuration for ES-9 MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class ES9Config:
    """Configuration loaded from environment / .env file."""

    midi_output_port: str = ""
    midi_input_port: str = ""
    audio_device: str = ""
    sample_rate: int = 48000
    auto_connect: bool = False

    @classmethod
    def from_env(cls, env_path: str | None = None) -> ES9Config:
        load_dotenv(env_path)
        return cls(
            midi_output_port=os.getenv("ES9_MIDI_OUTPUT_PORT", ""),
            midi_input_port=os.getenv("ES9_MIDI_INPUT_PORT", ""),
            audio_device=os.getenv("ES9_AUDIO_DEVICE", ""),
            sample_rate=int(os.getenv("ES9_SAMPLE_RATE", "48000")),
            auto_connect=os.getenv("ES9_AUTO_CONNECT", "false").lower()
            in ("true", "1", "yes"),
        )
