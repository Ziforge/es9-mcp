"""ES-9 SysEx protocol: message builders, encoding, dB conversions, routing maps."""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# SysEx Header
# ---------------------------------------------------------------------------

MANUFACTURER_ID = [0x00, 0x21, 0x27]  # Expert Sleepers
DEVICE_ID = 0x19  # ES-9

SYSEX_HEADER = [0xF0] + MANUFACTURER_ID + [DEVICE_ID]


def sysex_msg(cmd: int, data: list[int] | None = None) -> list[int]:
    """Build a complete ES-9 SysEx message: F0 00 21 27 19 [cmd] [data] F7."""
    msg = SYSEX_HEADER + [cmd]
    if data:
        msg.extend(data)
    msg.append(0xF7)
    return msg


# ---------------------------------------------------------------------------
# 21-bit Value Encoding (3 x 7-bit bytes)
# ---------------------------------------------------------------------------

def encode_21bit(value: int) -> list[int]:
    """Encode an integer as 3 x 7-bit bytes (21-bit max)."""
    value = max(0, min(0x1FFFFF, value))
    byte0 = (value >> 14) & 0x03   # 2 MSBs
    byte1 = (value >> 7) & 0x7F    # middle 7
    byte2 = value & 0x7F           # 7 LSBs
    return [byte0, byte1, byte2]


def decode_21bit(data: list[int], offset: int = 0) -> int:
    """Decode 3 x 7-bit bytes at offset into a 21-bit integer."""
    return (
        ((data[offset] & 0x03) << 14)
        | ((data[offset + 1] & 0x7F) << 7)
        | (data[offset + 2] & 0x7F)
    )


# ---------------------------------------------------------------------------
# Command Constants
# ---------------------------------------------------------------------------

# Host → Device
CMD_GET_FIRMWARE = 0x22
CMD_GET_CONFIG = 0x23
CMD_SAVE_CONFIG = 0x24
CMD_RESTORE_CONFIG = 0x25
CMD_FACTORY_RESET = 0x26
CMD_CODEC_RESET = 0x28
CMD_GET_MIX_STATE = 0x2A
CMD_GET_CPU_USAGE = 0x2B
CMD_GET_SAMPLE_RATE = 0x2C
CMD_SET_HPF = 0x31
CMD_SET_OPTIONS = 0x32
CMD_SET_STEREO_LINK = 0x33
CMD_SET_VIRTUAL_MIX = 0x34
CMD_SET_MIDI_CHANNEL = 0x35
CMD_SET_DC_OFFSET = 0x36
CMD_SET_EQ = 0x39
CMD_SET_SMOOTHING = 0x3A

# Capture routing: DSP 0-3
CMD_CAPTURE_ROUTING_BASE = 0x40  # 0x40-0x43

# Output routing: DSP 0-3
CMD_OUTPUT_ROUTING_BASE = 0x50  # 0x50-0x53

# Raw mix levels: mix 0-15
CMD_RAW_MIX_BASE = 0x60  # 0x60-0x6F

# Device → Host
CMD_CONFIG_DUMP_START = 0x08
CMD_CONFIG_DUMP_CHUNK = 0x09
CMD_MIX_STATE_RESPONSE = 0x11
CMD_CPU_USAGE_RESPONSE = 0x12
CMD_SAMPLE_RATE_RESPONSE = 0x14


# ---------------------------------------------------------------------------
# Command Builders
# ---------------------------------------------------------------------------

def msg_get_firmware() -> list[int]:
    return sysex_msg(CMD_GET_FIRMWARE)


def msg_get_config() -> list[int]:
    return sysex_msg(CMD_GET_CONFIG)


def msg_save_config(slot: int = 0) -> list[int]:
    """Save config to flash. slot: 0=hosted, 1=standalone."""
    return sysex_msg(CMD_SAVE_CONFIG, [slot & 0x01])


def msg_restore_config() -> list[int]:
    return sysex_msg(CMD_RESTORE_CONFIG)


def msg_factory_reset() -> list[int]:
    return sysex_msg(CMD_FACTORY_RESET)


def msg_codec_reset() -> list[int]:
    return sysex_msg(CMD_CODEC_RESET)


def msg_get_mix_state() -> list[int]:
    return sysex_msg(CMD_GET_MIX_STATE)


def msg_get_cpu_usage() -> list[int]:
    return sysex_msg(CMD_GET_CPU_USAGE)


def msg_get_sample_rate() -> list[int]:
    return sysex_msg(CMD_GET_SAMPLE_RATE)


def msg_set_hpf(channel: int, enabled: bool) -> list[int]:
    """Enable/disable DC blocking high-pass filter on an input channel."""
    return sysex_msg(CMD_SET_HPF, [channel & 0x7F, 1 if enabled else 0])


def msg_set_options(mixer2_spdif: bool = False, midi_thru: bool = False) -> list[int]:
    """Set options: mixer2/SPDIF mode, MIDI thru."""
    flags = (1 if mixer2_spdif else 0) | (2 if midi_thru else 0)
    return sysex_msg(CMD_SET_OPTIONS, [flags])


def msg_set_stereo_link(pair: int, enabled: bool) -> list[int]:
    """Set stereo link for a channel pair. pair: 0-based index."""
    return sysex_msg(CMD_SET_STEREO_LINK, [pair & 0x7F, 1 if enabled else 0])


def msg_set_virtual_mix(
    mix_bus: int, channel: int, value: int, is_pan: bool = False
) -> list[int]:
    """Set virtual mix level or pan.

    Args:
        mix_bus: Mix bus number (0-7).
        channel: Input channel (0-based).
        value: Level (0-127) or pan (0-127, 64=center).
        is_pan: True for pan, False for level.
    """
    param_type = 1 if is_pan else 0
    return sysex_msg(CMD_SET_VIRTUAL_MIX, [
        mix_bus & 0x07, channel & 0x7F, param_type, value & 0x7F
    ])


def msg_set_midi_channel(output: int, channel: int) -> list[int]:
    """Set MIDI channel assignment for an output."""
    return sysex_msg(CMD_SET_MIDI_CHANNEL, [output & 0x7F, channel & 0x0F])


def msg_set_dc_offset(output: int, offset_value: int) -> list[int]:
    """Set DC offset for an output channel. offset_value: 21-bit encoded."""
    return sysex_msg(CMD_SET_DC_OFFSET, [output & 0x7F] + encode_21bit(offset_value))


def msg_set_eq(
    channel: int, slot: int, filter_type: int, enabled: bool,
    freq_raw: int, q_raw: int, gain_raw: int
) -> list[int]:
    """Set EQ filter configuration.

    Args:
        channel: Channel index (0-based).
        slot: EQ slot (0-based, typically 0-3).
        filter_type: 0=LP1, 1=HP1, 2=LP2, 3=HP2, 4=LowShelf, 5=HighShelf, 6=Peak, 7=PhaseInvert.
        enabled: Whether filter is active.
        freq_raw: 21-bit raw frequency value.
        q_raw: 21-bit raw Q value.
        gain_raw: 21-bit raw gain value (signed).
    """
    type_byte = ((filter_type & 0x07) << 1) | (1 if enabled else 0)
    data = [channel & 0x7F, slot & 0x7F, type_byte]
    data.extend(encode_21bit(freq_raw))
    data.extend(encode_21bit(q_raw))
    data.extend(encode_21bit(gain_raw))
    return sysex_msg(CMD_SET_EQ, data)


def msg_set_smoothing(channel: int, enabled: bool) -> list[int]:
    return sysex_msg(CMD_SET_SMOOTHING, [channel & 0x7F, 1 if enabled else 0])


def msg_set_capture_routing(dsp: int, channels: list[int]) -> list[int]:
    """Set input capture routing for a DSP block (0-3).

    Args:
        dsp: DSP block 0-3 (each handles a group of channels).
        channels: List of source codes for each channel in this block.
    """
    cmd = CMD_CAPTURE_ROUTING_BASE + (dsp & 0x03)
    return sysex_msg(cmd, [c & 0x7F for c in channels])


def msg_set_output_routing(dsp: int, channels: list[int]) -> list[int]:
    """Set output routing for a DSP block (0-3)."""
    cmd = CMD_OUTPUT_ROUTING_BASE + (dsp & 0x03)
    return sysex_msg(cmd, [c & 0x7F for c in channels])


def msg_set_raw_mix(mix_bus: int, channel: int, level_raw: int) -> list[int]:
    """Set raw 21-bit mix level for a channel in a mix bus."""
    cmd = CMD_RAW_MIX_BASE + (mix_bus & 0x0F)
    return sysex_msg(cmd, [channel & 0x7F] + encode_21bit(level_raw))


# ---------------------------------------------------------------------------
# dB Conversions
# ---------------------------------------------------------------------------

# Raw mix: 0-32767. 0x2000 (8192) = 0dB, 0x7FFF = +12dB, 0 = -inf
RAW_MIX_UNITY = 0x2000  # 8192


def raw_mix_to_db(value: int) -> float:
    """Convert raw mix value (0-32767) to dB."""
    if value <= 0:
        return float("-inf")
    return 6.0 * math.log2(value / RAW_MIX_UNITY)


def db_to_raw_mix(db: float) -> int:
    """Convert dB to raw mix value (0-32767)."""
    if db <= -96.0 or math.isinf(db):
        return 0
    value = round(RAW_MIX_UNITY * (2.0 ** (db / 6.0)))
    return max(0, min(0x7FFF, value))


# Virtual mix: 0-127. 0=-inf, 55=-24dB, 103=0dB, 127=+12dB
# Piecewise linear mapping from ES-9 config tool source

_VMIX_BREAKPOINTS = [
    # (vmix_value, db)
    (0, float("-inf")),
    (1, -72.0),
    (55, -24.0),
    (103, 0.0),
    (127, 12.0),
]


def vmix_to_db(vmix: int) -> float:
    """Convert virtual mix value (0-127) to dB."""
    if vmix <= 0:
        return float("-inf")
    vmix = max(0, min(127, vmix))

    for i in range(1, len(_VMIX_BREAKPOINTS)):
        v1, db1 = _VMIX_BREAKPOINTS[i - 1]
        v2, db2 = _VMIX_BREAKPOINTS[i]
        if vmix <= v2:
            if math.isinf(db1):
                db1 = -72.0
            t = (vmix - v1) / (v2 - v1) if v2 != v1 else 0
            return db1 + t * (db2 - db1)

    return 12.0


def db_to_vmix(db: float) -> int:
    """Convert dB to virtual mix value (0-127)."""
    if math.isinf(db) or db <= -72.0:
        return 0

    db = max(-72.0, min(12.0, db))

    for i in range(1, len(_VMIX_BREAKPOINTS)):
        v1, db1 = _VMIX_BREAKPOINTS[i - 1]
        v2, db2 = _VMIX_BREAKPOINTS[i]
        if math.isinf(db1):
            db1 = -72.0
        if db <= db2:
            t = (db - db1) / (db2 - db1) if db2 != db1 else 0
            return max(0, min(127, round(v1 + t * (v2 - v1))))

    return 127


# ---------------------------------------------------------------------------
# EQ Frequency / Q / Gain Mapping
# ---------------------------------------------------------------------------

# Frequency: 21-bit log-mapped, 20Hz to 20kHz
EQ_FREQ_MIN = 20.0
EQ_FREQ_MAX = 20000.0
EQ_RAW_MAX = 0x1FFFFF  # 2097151


def hz_to_eq_raw(freq_hz: float) -> int:
    """Convert frequency in Hz to 21-bit raw EQ value (log-mapped)."""
    freq_hz = max(EQ_FREQ_MIN, min(EQ_FREQ_MAX, freq_hz))
    t = math.log(freq_hz / EQ_FREQ_MIN) / math.log(EQ_FREQ_MAX / EQ_FREQ_MIN)
    return round(t * EQ_RAW_MAX)


def eq_raw_to_hz(raw: int) -> float:
    """Convert 21-bit raw EQ value to frequency in Hz."""
    t = raw / EQ_RAW_MAX
    return EQ_FREQ_MIN * ((EQ_FREQ_MAX / EQ_FREQ_MIN) ** t)


# Q: 21-bit log-mapped, 0.1 to 20.0
EQ_Q_MIN = 0.1
EQ_Q_MAX = 20.0


def q_to_eq_raw(q: float) -> int:
    """Convert Q factor to 21-bit raw value."""
    q = max(EQ_Q_MIN, min(EQ_Q_MAX, q))
    t = math.log(q / EQ_Q_MIN) / math.log(EQ_Q_MAX / EQ_Q_MIN)
    return round(t * EQ_RAW_MAX)


def eq_raw_to_q(raw: int) -> float:
    """Convert 21-bit raw value to Q factor."""
    t = raw / EQ_RAW_MAX
    return EQ_Q_MIN * ((EQ_Q_MAX / EQ_Q_MIN) ** t)


# EQ Gain: 21-bit signed, -24 to +24 dB
EQ_GAIN_MIN = -24.0
EQ_GAIN_MAX = 24.0
EQ_GAIN_CENTER = EQ_RAW_MAX // 2  # midpoint = 0dB


def db_to_eq_gain_raw(db: float) -> int:
    """Convert EQ gain in dB to 21-bit raw value."""
    db = max(EQ_GAIN_MIN, min(EQ_GAIN_MAX, db))
    t = (db - EQ_GAIN_MIN) / (EQ_GAIN_MAX - EQ_GAIN_MIN)
    return round(t * EQ_RAW_MAX)


def eq_gain_raw_to_db(raw: int) -> float:
    """Convert 21-bit raw value to EQ gain in dB."""
    t = raw / EQ_RAW_MAX
    return EQ_GAIN_MIN + t * (EQ_GAIN_MAX - EQ_GAIN_MIN)


# EQ Filter type names
EQ_FILTER_TYPES = {
    "lp1": 0, "hp1": 1, "lp2": 2, "hp2": 3,
    "low_shelf": 4, "high_shelf": 5, "peak": 6, "phase_invert": 7,
}

EQ_FILTER_NAMES = {v: k for k, v in EQ_FILTER_TYPES.items()}


# ---------------------------------------------------------------------------
# Routing Maps
# ---------------------------------------------------------------------------

# Input capture source codes (14 physical inputs on base ES-9)
INPUT_CAPTURE_LOOKUP = [
    0x78, 0x79, 0x76, 0x75, 0x74, 0x7B, 0x7A, 0x77,
    0x73, 0x72, 0x7D, 0x7C, 0x7E, 0x7F,
]

# Named input sources → routing codes
INPUT_SOURCES: dict[str, int] = {}

# USB inputs 1-16
for _i in range(8):
    INPUT_SOURCES[f"usb_{_i + 1}"] = _i           # 0x00-0x07
for _i in range(8):
    INPUT_SOURCES[f"usb_{_i + 9}"] = 0x10 + _i    # 0x10-0x17

# Mix buses 1-8
for _i in range(8):
    INPUT_SOURCES[f"mix_{_i + 1}"] = 0x20 + _i    # 0x20-0x27

# Buses 1-16
for _i in range(16):
    INPUT_SOURCES[f"bus_{_i + 1}"] = 0x60 + _i     # 0x60-0x6F

# S/PDIF
INPUT_SOURCES["spdif_l"] = 0x30
INPUT_SOURCES["spdif_r"] = 0x31

# Physical inputs by number
for _i, _code in enumerate(INPUT_CAPTURE_LOOKUP):
    INPUT_SOURCES[f"input_{_i + 1}"] = _code

# Reverse lookup
INPUT_SOURCE_NAMES = {v: k for k, v in INPUT_SOURCES.items()}


# Output destination codes (non-sequential mapping!)
OUTPUT_MAP: dict[str | int, int] = {
    1: 0x08, 2: 0x09, 3: 0x04, 4: 0x05,
    5: 0x02, 6: 0x03, 7: 0x0B, 8: 0x0A,
    "main_l": 0x06, "main_r": 0x07,
    "phones_l": 0x0C, "phones_r": 0x0D,
    "es5_l": 0x0E, "es5_r": 0x0F,
}

# Bus outputs 1-16
for _i in range(16):
    OUTPUT_MAP[f"bus_{_i + 1}"] = 0x10 + _i

# Also accept string "output_N" for numbered outputs
for _n in range(1, 9):
    OUTPUT_MAP[f"output_{_n}"] = OUTPUT_MAP[_n]

# Reverse lookup
OUTPUT_NAMES: dict[int, str] = {}
for _key, _val in OUTPUT_MAP.items():
    if isinstance(_key, str) and _val not in OUTPUT_NAMES:
        OUTPUT_NAMES[_val] = _key
    elif isinstance(_key, int) and _val not in OUTPUT_NAMES:
        OUTPUT_NAMES[_val] = f"output_{_key}"


def resolve_input_source(name: str) -> int:
    """Resolve a human-readable input source name to its routing code."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in INPUT_SOURCES:
        return INPUT_SOURCES[key]
    raise ValueError(
        f"Unknown input source '{name}'. "
        f"Available: {sorted(INPUT_SOURCES.keys())}"
    )


def resolve_output_dest(name: str | int) -> int:
    """Resolve a human-readable output name or number to its routing code."""
    if isinstance(name, int):
        if name in OUTPUT_MAP:
            return OUTPUT_MAP[name]
        raise ValueError(f"Unknown output number {name}. Available: 1-8")

    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in OUTPUT_MAP:
        return OUTPUT_MAP[key]
    # Try parsing as integer
    try:
        num = int(key)
        if num in OUTPUT_MAP:
            return OUTPUT_MAP[num]
    except ValueError:
        pass
    raise ValueError(
        f"Unknown output '{name}'. "
        f"Available: {sorted(k for k in OUTPUT_MAP if isinstance(k, str))}"
    )


# ---------------------------------------------------------------------------
# Note / Voltage Helpers (V/Oct)
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_NOTE_NAME_TO_SEMITONE = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}


def note_name_to_midi(name: str) -> int:
    """Parse note name like 'C4', 'F#3', 'Bb5' to MIDI note number."""
    name = name.strip()
    if not name:
        raise ValueError("Empty note name")

    idx = 0
    base = name[idx].upper()
    if base not in _NOTE_NAME_TO_SEMITONE:
        raise ValueError(f"Invalid note letter: {base}")
    semitone = _NOTE_NAME_TO_SEMITONE[base]
    idx += 1

    if idx < len(name) and name[idx] in ("#", "s"):
        semitone += 1
        idx += 1
    elif idx < len(name) and name[idx] == "b":
        semitone -= 1
        idx += 1

    try:
        octave = int(name[idx:])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot parse octave from '{name}'")

    midi = (octave + 1) * 12 + semitone
    if not 0 <= midi <= 127:
        raise ValueError(f"MIDI note {midi} out of range (0-127)")
    return midi


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name like 'C4'."""
    midi = max(0, min(127, midi))
    octave = (midi // 12) - 1
    semitone = midi % 12
    return f"{_NOTE_NAMES[semitone]}{octave}"


def midi_to_voltage(midi: int) -> float:
    """Convert MIDI note to V/Oct voltage. C0 (MIDI 12) = 0V."""
    return (midi - 12) / 12.0


def voltage_to_midi(voltage: float) -> int:
    """Convert V/Oct voltage to nearest MIDI note. 0V = C0 (MIDI 12)."""
    return max(0, min(127, round(voltage * 12.0 + 12)))


def voltage_to_sample(voltage: float) -> float:
    """Convert voltage (-10V to +10V) to audio sample value (-1.0 to +1.0)."""
    return max(-1.0, min(1.0, voltage / 10.0))


def sample_to_voltage(sample: float) -> float:
    """Convert audio sample value (-1.0 to +1.0) to voltage (-10V to +10V)."""
    return sample * 10.0


# ---------------------------------------------------------------------------
# Config Dump Parser
# ---------------------------------------------------------------------------

def parse_config_response(messages: list[list[int]]) -> dict[str, Any]:
    """Parse config dump SysEx messages into a structured dict.

    The config dump consists of a start message (CMD 0x08) followed by
    chunk messages (CMD 0x09). This extracts what we can into a readable dict.
    """
    config: dict[str, Any] = {
        "raw_messages": len(messages),
        "parsed": False,
    }

    if not messages:
        return config

    # Concatenate payload bytes from all chunks
    payload = bytearray()
    for msg in messages:
        # Strip SysEx header (F0 00 21 27 19 CMD) and F7
        if len(msg) > 7:
            payload.extend(msg[6:-1])

    config["payload_bytes"] = len(payload)
    config["parsed"] = True

    return config


def parse_mix_state_response(data: list[int]) -> dict[str, Any]:
    """Parse mix state response (CMD 0x11) payload."""
    # Strip header: F0 00 21 27 19 11 [payload] F7
    if len(data) < 8:
        return {"error": "Message too short"}

    payload = data[6:-1]
    return {
        "raw_payload_bytes": len(payload),
        "raw": [f"0x{b:02X}" for b in payload[:32]],  # first 32 bytes
    }


def parse_cpu_usage_response(data: list[int]) -> dict[str, Any]:
    """Parse CPU usage response (CMD 0x12)."""
    if len(data) < 8:
        return {"error": "Message too short"}
    payload = data[6:-1]
    if len(payload) >= 3:
        usage = decode_21bit(payload, 0)
        return {"cpu_percent": usage / 100.0}
    return {"raw": [f"0x{b:02X}" for b in payload]}


def parse_sample_rate_response(data: list[int]) -> dict[str, Any]:
    """Parse sample rate response (CMD 0x14)."""
    if len(data) < 8:
        return {"error": "Message too short"}
    payload = data[6:-1]
    if len(payload) >= 3:
        rate = decode_21bit(payload, 0)
        return {"sample_rate": rate}
    return {"raw": [f"0x{b:02X}" for b in payload]}
