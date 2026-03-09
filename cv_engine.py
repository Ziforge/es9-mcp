"""CV/Gate engine using sounddevice for ES-9 DC-coupled USB audio I/O."""

from __future__ import annotations

import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# CV Source Classes
# ---------------------------------------------------------------------------

class CVSource(ABC):
    """Base class for CV output sources."""

    @abstractmethod
    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        """Generate audio samples for this source.

        Returns:
            1-D float32 array of length `frames`, values in [-1.0, 1.0].
        """
        ...

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of this source."""
        ...


class OffSource(CVSource):
    """Zero output (silence)."""

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        return np.zeros(frames, dtype=np.float32)

    def describe(self) -> str:
        return "Off"


class StaticCV(CVSource):
    """Constant voltage output."""

    def __init__(self, voltage: float):
        self.voltage = voltage
        self._sample = max(-1.0, min(1.0, voltage / 10.0))

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        return np.full(frames, self._sample, dtype=np.float32)

    def describe(self) -> str:
        return f"Static {self.voltage:+.3f}V"


class GateCV(CVSource):
    """Gate output: high or low voltage."""

    def __init__(self, high: bool = False, voltage: float = 5.0):
        self.high = high
        self.voltage = voltage
        self._sample = max(-1.0, min(1.0, voltage / 10.0)) if high else 0.0

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        return np.full(frames, self._sample, dtype=np.float32)

    def describe(self) -> str:
        state = "HIGH" if self.high else "LOW"
        return f"Gate {state} ({self.voltage}V)"


class LfoCv(CVSource):
    """Low-frequency oscillator source."""

    def __init__(
        self,
        shape: str = "sine",
        rate_hz: float = 1.0,
        depth_v: float = 5.0,
        offset_v: float = 0.0,
    ):
        self.shape = shape.lower()
        self.rate_hz = rate_hz
        self.depth_v = depth_v
        self.offset_v = offset_v
        self._phase = 0.0

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        t = np.arange(frames, dtype=np.float64)
        phase_inc = self.rate_hz / sample_rate
        phases = self._phase + t * phase_inc
        self._phase = (self._phase + frames * phase_inc) % 1.0

        if self.shape == "sine":
            wave = np.sin(2.0 * np.pi * phases)
        elif self.shape == "triangle" or self.shape == "tri":
            wave = 2.0 * np.abs(2.0 * (phases % 1.0) - 1.0) - 1.0
        elif self.shape == "saw":
            wave = 2.0 * (phases % 1.0) - 1.0
        elif self.shape == "square":
            wave = np.where((phases % 1.0) < 0.5, 1.0, -1.0)
        elif self.shape == "random" or self.shape == "s&h":
            # Sample-and-hold: new random value each cycle
            rng = np.random.default_rng()
            cycle_indices = np.floor(phases).astype(int)
            changes = np.diff(cycle_indices, prepend=cycle_indices[0] - 1) > 0
            wave = np.zeros(frames)
            current = rng.uniform(-1, 1)
            for i in range(frames):
                if changes[i]:
                    current = rng.uniform(-1, 1)
                wave[i] = current
        else:
            wave = np.sin(2.0 * np.pi * phases)

        # Scale: depth_v is peak-to-peak in volts, offset_v is center
        amplitude = self.depth_v / 2.0 / 10.0  # convert to sample range
        offset = self.offset_v / 10.0
        output = (wave * amplitude + offset).astype(np.float32)
        return np.clip(output, -1.0, 1.0)

    def describe(self) -> str:
        return (
            f"LFO {self.shape} {self.rate_hz:.2f}Hz "
            f"depth={self.depth_v:.1f}V offset={self.offset_v:.1f}V"
        )


class EnvelopeCV(CVSource):
    """Attack-release envelope (one-shot)."""

    def __init__(
        self,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
        peak_v: float = 5.0,
    ):
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.peak_v = peak_v
        self._peak_sample = max(-1.0, min(1.0, peak_v / 10.0))
        self._position = 0  # sample counter
        self._done = False

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        if self._done:
            return np.zeros(frames, dtype=np.float32)

        attack_samples = int(self.attack_ms * sample_rate / 1000.0)
        release_samples = int(self.release_ms * sample_rate / 1000.0)
        total = attack_samples + release_samples

        output = np.zeros(frames, dtype=np.float32)
        for i in range(frames):
            pos = self._position + i
            if pos < attack_samples:
                # Attack phase
                output[i] = self._peak_sample * (pos / max(1, attack_samples))
            elif pos < total:
                # Release phase
                release_pos = pos - attack_samples
                output[i] = self._peak_sample * (1.0 - release_pos / max(1, release_samples))
            else:
                self._done = True
                break

        self._position += frames
        return output

    def describe(self) -> str:
        return (
            f"Envelope A={self.attack_ms:.0f}ms R={self.release_ms:.0f}ms "
            f"peak={self.peak_v:.1f}V"
        )


class SequenceCV(CVSource):
    """Step sequence with pitch and gate."""

    def __init__(
        self,
        steps: list[dict],
        loop: bool = True,
        gate_channel: int | None = None,
    ):
        """
        Args:
            steps: List of dicts with keys: voltage (float), gate (bool),
                   duration_ms (float).
            loop: Whether to loop the sequence.
            gate_channel: If set, this source outputs gate for that channel.
        """
        self.steps = steps
        self.loop = loop
        self.gate_channel = gate_channel
        self._step_idx = 0
        self._step_pos = 0  # samples into current step
        self._done = False

    def render(self, frames: int, sample_rate: int) -> np.ndarray:
        if self._done or not self.steps:
            return np.zeros(frames, dtype=np.float32)

        output = np.zeros(frames, dtype=np.float32)
        pos = 0

        while pos < frames and not self._done:
            step = self.steps[self._step_idx]
            dur_samples = int(step.get("duration_ms", 250) * sample_rate / 1000.0)
            remaining_in_step = dur_samples - self._step_pos
            remaining_in_buffer = frames - pos
            count = min(remaining_in_step, remaining_in_buffer)

            voltage = step.get("voltage", 0.0)
            sample_val = max(-1.0, min(1.0, voltage / 10.0))
            output[pos:pos + count] = sample_val

            pos += count
            self._step_pos += count

            if self._step_pos >= dur_samples:
                self._step_pos = 0
                self._step_idx += 1
                if self._step_idx >= len(self.steps):
                    if self.loop:
                        self._step_idx = 0
                    else:
                        self._done = True

        return output

    def describe(self) -> str:
        mode = "loop" if self.loop else "one-shot"
        return f"Sequence ({len(self.steps)} steps, {mode})"


# ---------------------------------------------------------------------------
# CV Engine
# ---------------------------------------------------------------------------

class CVEngine:
    """Manages sounddevice duplex stream for CV/gate I/O."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 48000,
        num_outputs: int = 8,
        num_inputs: int = 14,
        block_size: int = 256,
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._num_outputs = num_outputs
        self._num_inputs = num_inputs
        self._block_size = block_size

        self._lock = threading.Lock()
        self._sources: dict[int, CVSource] = {}
        self._stream: Any = None
        self._running = False

        # Input voltage monitoring
        self._input_ring: dict[int, deque] = {
            ch: deque(maxlen=4096) for ch in range(num_inputs)
        }
        self._monitoring_inputs = False
        self._input_log: list[dict] = []
        self._monitor_interval: float = 0.1

    @property
    def running(self) -> bool:
        return self._running

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def start(self) -> str:
        """Open and start the audio stream."""
        import sounddevice as sd

        if self._running:
            return "Audio stream already running"

        self._stream = sd.Stream(
            device=self._device,
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=(self._num_inputs, self._num_outputs),
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

        device_info = sd.query_devices(self._device)
        name = device_info["name"] if isinstance(device_info, dict) else str(device_info)
        return f"Audio stream started: {name} @ {self._sample_rate}Hz"

    def stop(self) -> str:
        """Stop and close the audio stream."""
        if not self._running:
            return "Audio stream not running"

        self._monitoring_inputs = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        return "Audio stream stopped"

    def _callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """Audio callback: render CV sources to outputs, capture inputs."""
        with self._lock:
            for ch in range(self._num_outputs):
                source = self._sources.get(ch)
                if source:
                    try:
                        outdata[:, ch] = source.render(frames, self._sample_rate)
                    except Exception:
                        outdata[:, ch] = 0.0
                else:
                    outdata[:, ch] = 0.0

        # Capture input samples
        if self._monitoring_inputs and indata is not None:
            for ch in range(min(self._num_inputs, indata.shape[1])):
                self._input_ring[ch].extend(indata[:, ch])

    def set_source(self, channel: int, source: CVSource) -> None:
        """Set the CV source for an output channel (thread-safe)."""
        if not 0 <= channel < self._num_outputs:
            raise ValueError(
                f"Channel {channel} out of range (0-{self._num_outputs - 1})"
            )
        with self._lock:
            self._sources[channel] = source

    def clear_source(self, channel: int) -> None:
        """Remove/zero a CV output channel."""
        with self._lock:
            self._sources.pop(channel, None)

    def get_source_info(self) -> dict[int, str]:
        """Get descriptions of all active CV sources."""
        with self._lock:
            return {
                ch: src.describe()
                for ch, src in self._sources.items()
            }

    def read_voltage(self, channel: int, samples: int = 64) -> float | None:
        """Read average voltage from an input channel's ring buffer."""
        if channel not in self._input_ring:
            return None
        ring = self._input_ring[channel]
        if not ring:
            return None
        recent = list(ring)[-samples:]
        if not recent:
            return None
        avg_sample = sum(recent) / len(recent)
        return avg_sample * 10.0  # sample → voltage

    def start_input_monitor(self, interval: float = 0.1) -> None:
        """Start monitoring input voltages."""
        self._monitor_interval = interval
        self._input_log.clear()
        self._monitoring_inputs = True

    def stop_input_monitor(self) -> None:
        """Stop monitoring input voltages."""
        self._monitoring_inputs = False

    def get_input_log(self, channel: int, count: int = 50) -> list[dict]:
        """Get recent voltage readings for a channel."""
        if not self._monitoring_inputs and not self._input_ring.get(channel):
            return []
        # Read current values
        ring = self._input_ring.get(channel)
        if not ring:
            return []
        recent = list(ring)[-count:]
        return [{"sample": i, "voltage": s * 10.0} for i, s in enumerate(recent)]

    @staticmethod
    def list_audio_devices() -> str:
        """List all audio devices via sounddevice."""
        import sounddevice as sd
        devices = sd.query_devices()
        lines = ["=== Audio Devices ==="]
        for i, dev in enumerate(devices):
            ins = dev["max_input_channels"]
            outs = dev["max_output_channels"]
            sr = int(dev["default_samplerate"])
            direction = []
            if ins > 0:
                direction.append(f"{ins}in")
            if outs > 0:
                direction.append(f"{outs}out")
            lines.append(f"  [{i}] {dev['name']} ({', '.join(direction)}) @ {sr}Hz")
        return "\n".join(lines)

    @staticmethod
    def find_es9_device() -> int | None:
        """Find ES-9 audio device index by name."""
        import sounddevice as sd
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "es-9" in dev["name"].lower() or "es9" in dev["name"].lower():
                return i
        return None
