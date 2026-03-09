"""ES-9 engine: thread-safe MIDI SysEx wrapper + mixer state + CV engine."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import rtmidi

from cv_engine import CVEngine
from protocol import (
    CMD_CONFIG_DUMP_CHUNK,
    CMD_CONFIG_DUMP_START,
    CMD_CPU_USAGE_RESPONSE,
    CMD_MIX_STATE_RESPONSE,
    CMD_SAMPLE_RATE_RESPONSE,
    DEVICE_ID,
    MANUFACTURER_ID,
    db_to_eq_gain_raw,
    db_to_raw_mix,
    db_to_vmix,
    eq_raw_to_hz,
    eq_raw_to_q,
    eq_gain_raw_to_db,
    hz_to_eq_raw,
    msg_codec_reset,
    msg_factory_reset,
    msg_get_config,
    msg_get_cpu_usage,
    msg_get_firmware,
    msg_get_mix_state,
    msg_get_sample_rate,
    msg_restore_config,
    msg_save_config,
    msg_set_capture_routing,
    msg_set_dc_offset,
    msg_set_eq,
    msg_set_hpf,
    msg_set_options,
    msg_set_output_routing,
    msg_set_raw_mix,
    msg_set_smoothing,
    msg_set_stereo_link,
    msg_set_virtual_mix,
    parse_config_response,
    parse_cpu_usage_response,
    parse_mix_state_response,
    parse_sample_rate_response,
    q_to_eq_raw,
    raw_mix_to_db,
    vmix_to_db,
    EQ_FILTER_TYPES,
    EQ_FILTER_NAMES,
    encode_21bit,
)


@dataclass
class MidiMessage:
    """Parsed MIDI message with timestamp."""

    raw: list[int]
    timestamp: float
    type: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.type:
            self._parse()

    def _parse(self) -> None:
        if not self.raw:
            self.type = "unknown"
            self.description = "empty"
            return
        status = self.raw[0]
        if status == 0xF0:
            self.type = "sysex"
            self.description = f"SysEx ({len(self.raw)} bytes)"
        elif status == 0xFE:
            self.type = "active_sense"
            self.description = "Active Sensing"
        else:
            self.type = "other"
            self.description = f"0x{status:02X} ({len(self.raw)} bytes)"


class ES9Engine:
    """Thread-safe ES-9 MIDI + CV engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._midi_out: rtmidi.MidiOut | None = None
        self._midi_in: rtmidi.MidiIn | None = None
        self._out_port_name: str = ""
        self._in_port_name: str = ""
        self._midi_connected = False

        # SysEx response handling
        self._sysex_buffer: list[list[int]] = []
        self._sysex_event = threading.Event()
        self._collecting_sysex = False

        # Monitoring
        self._monitoring = False
        self._message_log: deque[MidiMessage] = deque(maxlen=1000)

        # CV engine (created on audio connect)
        self._cv: CVEngine | None = None

        # Cached state
        self._config_cache: dict[str, Any] | None = None

    # -- Properties --

    @property
    def midi_connected(self) -> bool:
        return self._midi_connected

    @property
    def audio_running(self) -> bool:
        return self._cv is not None and self._cv.running

    @property
    def cv_engine(self) -> CVEngine | None:
        return self._cv

    @property
    def port_info(self) -> str:
        if not self._midi_connected:
            return "Not connected"
        return f"out='{self._out_port_name}', in='{self._in_port_name}'"

    # -- Port Discovery --

    @staticmethod
    def list_output_ports() -> list[str]:
        tmp = rtmidi.MidiOut()
        try:
            return tmp.get_ports()
        finally:
            del tmp

    @staticmethod
    def list_input_ports() -> list[str]:
        tmp = rtmidi.MidiIn()
        try:
            return tmp.get_ports()
        finally:
            del tmp

    @staticmethod
    def _find_port(ports: list[str], query: str) -> int | None:
        q = query.lower()
        for i, name in enumerate(ports):
            if q in name.lower():
                return i
        return None

    # -- MIDI Connection --

    def connect_midi(
        self,
        output_port: str | int = "",
        input_port: str | int = "",
    ) -> str:
        """Connect to ES-9 MIDI ports."""
        with self._lock:
            if self._midi_connected:
                self._disconnect_midi_locked()

            self._midi_out = rtmidi.MidiOut()
            out_ports = self._midi_out.get_ports()

            if isinstance(output_port, int):
                out_idx = output_port
            elif output_port:
                out_idx = self._find_port(out_ports, output_port)
                if out_idx is None:
                    raise ValueError(
                        f"Output port '{output_port}' not found. "
                        f"Available: {out_ports}"
                    )
            else:
                # Auto-detect ES-9
                out_idx = self._find_port(out_ports, "ES-9")
                if out_idx is None:
                    raise ValueError(
                        f"ES-9 not found in output ports. Available: {out_ports}"
                    )

            self._midi_out.open_port(out_idx)
            self._out_port_name = out_ports[out_idx]

            # Input
            self._midi_in = rtmidi.MidiIn()
            self._midi_in.ignore_types(
                sysex=False, timing=False, active_sense=True
            )
            in_ports = self._midi_in.get_ports()

            if isinstance(input_port, int):
                in_idx = input_port
            elif input_port:
                in_idx = self._find_port(in_ports, input_port)
                if in_idx is None:
                    raise ValueError(
                        f"Input port '{input_port}' not found. "
                        f"Available: {in_ports}"
                    )
            else:
                # Match output query or auto-detect
                query = output_port if isinstance(output_port, str) and output_port else "ES-9"
                in_idx = self._find_port(in_ports, query)

            if in_idx is not None:
                self._midi_in.open_port(in_idx)
                self._in_port_name = in_ports[in_idx]
                self._midi_in.set_callback(self._input_callback)
            else:
                self._in_port_name = "(none)"

            self._midi_connected = True
            return (
                f"MIDI connected: out='{self._out_port_name}', "
                f"in='{self._in_port_name}'"
            )

    def _disconnect_midi_locked(self) -> None:
        """Disconnect MIDI (must hold lock)."""
        self._monitoring = False
        if self._midi_out:
            self._midi_out.close_port()
            del self._midi_out
            self._midi_out = None
        if self._midi_in:
            self._midi_in.close_port()
            del self._midi_in
            self._midi_in = None
        self._midi_connected = False
        self._out_port_name = ""
        self._in_port_name = ""

    def disconnect_midi(self) -> str:
        with self._lock:
            self._disconnect_midi_locked()
        return "MIDI disconnected"

    # -- Audio Connection --

    def connect_audio(
        self,
        device: int | str | None = None,
        sample_rate: int = 48000,
    ) -> str:
        """Connect to ES-9 audio device."""
        if self._cv and self._cv.running:
            self._cv.stop()

        self._cv = CVEngine(
            device=device,
            sample_rate=sample_rate,
            num_outputs=8,
            num_inputs=14,
        )
        return self._cv.start()

    def disconnect_audio(self) -> str:
        if self._cv:
            result = self._cv.stop()
            self._cv = None
            return result
        return "Audio not connected"

    # -- Full Connect / Disconnect --

    def connect(
        self,
        midi_output: str | int = "",
        midi_input: str | int = "",
        audio_device: int | str | None = None,
        sample_rate: int = 48000,
    ) -> str:
        """Connect both MIDI and audio."""
        parts = []
        parts.append(self.connect_midi(midi_output, midi_input))
        try:
            parts.append(self.connect_audio(audio_device, sample_rate))
        except Exception as e:
            parts.append(f"Audio: {e} (MIDI-only mode)")
        return "\n".join(parts)

    def disconnect(self) -> str:
        parts = []
        if self._cv and self._cv.running:
            parts.append(self.disconnect_audio())
        if self._midi_connected:
            parts.append(self.disconnect_midi())
        return "\n".join(parts) if parts else "Not connected"

    # -- MIDI I/O --

    def _input_callback(
        self, event: tuple[list[int], float], data: Any = None
    ) -> None:
        message, delta = event

        if self._monitoring:
            msg = MidiMessage(raw=message, timestamp=time.time())
            self._message_log.append(msg)

        if self._collecting_sysex and message and message[0] == 0xF0:
            self._sysex_buffer.append(message)
            self._sysex_event.set()

    def _send(self, message: list[int]) -> None:
        with self._lock:
            if not self._midi_connected or not self._midi_out:
                raise RuntimeError("Not connected to ES-9 MIDI")
            self._midi_out.send_message(message)

    def send_sysex(self, data: list[int]) -> None:
        """Send a SysEx message (auto-frames if needed)."""
        if data[0] != 0xF0:
            data = [0xF0] + data
        if data[-1] != 0xF7:
            data = data + [0xF7]
        self._send(data)

    def send_and_wait(
        self, message: list[int], timeout: float = 2.0
    ) -> list[int] | None:
        """Send a SysEx message and wait for a response."""
        self._sysex_buffer.clear()
        self._sysex_event.clear()
        self._collecting_sysex = True
        self._send(message)
        got = self._sysex_event.wait(timeout=timeout)
        self._collecting_sysex = False
        if got and self._sysex_buffer:
            return self._sysex_buffer[0]
        return None

    def send_and_collect(
        self, message: list[int], timeout: float = 3.0, idle_timeout: float = 1.0
    ) -> list[list[int]]:
        """Send a SysEx and collect multiple responses (for config dump)."""
        self._sysex_buffer.clear()
        self._sysex_event.clear()
        self._collecting_sysex = True
        self._send(message)

        start = time.time()
        last_count = 0
        last_activity = time.time()

        while True:
            time.sleep(0.1)
            count = len(self._sysex_buffer)
            if count > last_count:
                last_count = count
                last_activity = time.time()

            if time.time() - start > timeout:
                break
            if count > 0 and time.time() - last_activity > idle_timeout:
                break

        self._collecting_sysex = False
        return list(self._sysex_buffer)

    # -- Monitor --

    def start_monitor(self) -> None:
        self._message_log.clear()
        self._monitoring = True

    def stop_monitor(self) -> None:
        self._monitoring = False

    @property
    def is_monitoring(self) -> bool:
        return self._monitoring

    @property
    def log_count(self) -> int:
        return len(self._message_log)

    def get_log(
        self, count: int = 50, type_filter: str | None = None
    ) -> list[MidiMessage]:
        msgs = list(self._message_log)
        if type_filter:
            f = type_filter.lower()
            msgs = [m for m in msgs if f in m.type]
        return msgs[-count:]

    # -- ES-9 SysEx Commands --

    def get_firmware_version(self) -> str:
        """Query firmware version."""
        resp = self.send_and_wait(msg_get_firmware())
        if resp is None:
            return "No response (is ES-9 connected?)"
        # Parse version from response payload
        payload = resp[6:-1] if len(resp) > 7 else resp
        if payload:
            return f"Firmware v{'.'.join(str(b) for b in payload[:3])}"
        return f"Response: {' '.join(f'{b:02X}' for b in resp)}"

    def get_config(self) -> dict[str, Any]:
        """Request and parse full config dump."""
        messages = self.send_and_collect(msg_get_config(), timeout=5.0, idle_timeout=2.0)
        config = parse_config_response(messages)
        self._config_cache = config
        return config

    def get_mix_state(self) -> dict[str, Any]:
        """Request current mix state."""
        resp = self.send_and_wait(msg_get_mix_state())
        if resp is None:
            return {"error": "No response"}
        return parse_mix_state_response(resp)

    def get_cpu_usage(self) -> dict[str, Any]:
        """Request DSP CPU usage."""
        resp = self.send_and_wait(msg_get_cpu_usage())
        if resp is None:
            return {"error": "No response"}
        return parse_cpu_usage_response(resp)

    def get_sample_rate(self) -> dict[str, Any]:
        """Request current sample rate."""
        resp = self.send_and_wait(msg_get_sample_rate())
        if resp is None:
            return {"error": "No response"}
        return parse_sample_rate_response(resp)

    def save_config(self, slot: int = 0) -> None:
        self._send(msg_save_config(slot))

    def restore_config(self) -> None:
        self._send(msg_restore_config())

    def factory_reset(self) -> None:
        self._send(msg_factory_reset())

    def codec_reset(self) -> None:
        self._send(msg_codec_reset())

    # -- Mixer --

    def set_virtual_mix(
        self, mix_bus: int, channel: int, db: float
    ) -> None:
        """Set virtual mix level in dB."""
        vmix = db_to_vmix(db)
        self._send(msg_set_virtual_mix(mix_bus, channel, vmix, is_pan=False))

    def set_virtual_pan(
        self, mix_bus: int, channel: int, pan: int
    ) -> None:
        """Set virtual mix pan (-63 to +63, 0=center)."""
        # Convert -63..+63 to 0..127 (64=center)
        value = max(0, min(127, pan + 64))
        self._send(msg_set_virtual_mix(mix_bus, channel, value, is_pan=True))

    def set_raw_mix(self, mix_bus: int, channel: int, db: float) -> None:
        """Set raw 21-bit mix level in dB."""
        raw = db_to_raw_mix(db)
        self._send(msg_set_raw_mix(mix_bus, channel, raw))

    def reset_mixer(self) -> None:
        """Zero all mix levels (set to -inf dB)."""
        for bus in range(8):
            for ch in range(16):
                self._send(msg_set_virtual_mix(bus, ch, 0, is_pan=False))

    # -- Routing --

    def set_capture_routing(self, dsp: int, channels: list[int]) -> None:
        self._send(msg_set_capture_routing(dsp, channels))

    def set_output_routing(self, dsp: int, channels: list[int]) -> None:
        self._send(msg_set_output_routing(dsp, channels))

    def set_options(self, mixer2_spdif: bool = False, midi_thru: bool = False) -> None:
        self._send(msg_set_options(mixer2_spdif, midi_thru))

    # -- EQ --

    def set_eq(
        self,
        channel: int,
        slot: int,
        filter_type: str | int,
        enabled: bool,
        freq_hz: float,
        q: float = 1.0,
        gain_db: float = 0.0,
    ) -> None:
        """Set EQ filter with human-readable parameters."""
        if isinstance(filter_type, str):
            ft = EQ_FILTER_TYPES.get(filter_type.lower())
            if ft is None:
                raise ValueError(
                    f"Unknown filter type '{filter_type}'. "
                    f"Available: {list(EQ_FILTER_TYPES.keys())}"
                )
        else:
            ft = filter_type

        freq_raw = hz_to_eq_raw(freq_hz)
        q_raw = q_to_eq_raw(q)
        gain_raw = db_to_eq_gain_raw(gain_db)

        self._send(msg_set_eq(channel, slot, ft, enabled, freq_raw, q_raw, gain_raw))

    def bypass_eq(self, channel: int, slot: int) -> None:
        """Disable an EQ filter slot."""
        # Send with enabled=False, keep other params at defaults
        self._send(msg_set_eq(channel, slot, 0, False, 0, 0, 0))

    # -- DC / HPF --

    def set_dc_offset(self, output: int, voltage: float) -> None:
        """Set DC offset in volts (converted to 21-bit raw)."""
        # Map voltage to raw value. Assume ±10V range maps to 0-2097151
        raw = int(((voltage + 10.0) / 20.0) * 0x1FFFFF)
        raw = max(0, min(0x1FFFFF, raw))
        self._send(msg_set_dc_offset(output, raw))

    def set_hpf(self, channel: int, enabled: bool) -> None:
        self._send(msg_set_hpf(channel, enabled))

    def set_smoothing(self, channel: int, enabled: bool) -> None:
        self._send(msg_set_smoothing(channel, enabled))
