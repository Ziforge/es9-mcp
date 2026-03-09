"""ES-9 MCP Server — control an Expert Sleepers ES-9 over USB MIDI + Audio."""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import Context, FastMCP

from config import ES9Config
from cv_engine import (
    CVEngine,
    EnvelopeCV,
    GateCV,
    LfoCv,
    OffSource,
    SequenceCV,
    StaticCV,
)
from es9_engine import ES9Engine
from protocol import (
    EQ_FILTER_NAMES,
    EQ_FILTER_TYPES,
    INPUT_SOURCE_NAMES,
    INPUT_SOURCES,
    OUTPUT_MAP,
    OUTPUT_NAMES,
    db_to_raw_mix,
    db_to_vmix,
    midi_to_note_name,
    midi_to_voltage,
    note_name_to_midi,
    raw_mix_to_db,
    resolve_input_source,
    resolve_output_dest,
    sample_to_voltage,
    vmix_to_db,
    voltage_to_midi,
    voltage_to_sample,
)

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    config = ES9Config.from_env()
    engine = ES9Engine()

    if config.auto_connect:
        try:
            result = engine.connect(
                midi_output=config.midi_output_port,
                midi_input=config.midi_input_port,
                audio_device=config.audio_device or None,
                sample_rate=config.sample_rate,
            )
            print(f"[es9-mcp] Auto-connected: {result}")
        except Exception as e:
            print(f"[es9-mcp] Auto-connect failed: {e}")

    yield {"engine": engine, "config": config}

    engine.disconnect()


mcp = FastMCP(
    "es9-midi",
    instructions=(
        "Control an Expert Sleepers ES-9 Eurorack USB audio interface. "
        "Provides mixer routing, EQ, CV/gate generation, and voltage reading."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Context Helpers
# ---------------------------------------------------------------------------


def _engine(ctx: Context) -> ES9Engine:
    return ctx.request_context.lifespan_context["engine"]


def _config(ctx: Context) -> ES9Config:
    return ctx.request_context.lifespan_context["config"]


def _require_midi(ctx: Context) -> ES9Engine:
    engine = _engine(ctx)
    if not engine.midi_connected:
        raise ValueError("Not connected to ES-9 MIDI. Use connect_es9 first.")
    return engine


def _require_audio(ctx: Context) -> ES9Engine:
    engine = _engine(ctx)
    if not engine.audio_running:
        raise ValueError(
            "ES-9 audio not running. Use connect_es9 with audio, "
            "or audio will start automatically on first CV command."
        )
    return engine


def _ensure_audio(ctx: Context) -> ES9Engine:
    """Get engine, auto-starting audio if MIDI is connected but audio isn't."""
    engine = _engine(ctx)
    if not engine.midi_connected:
        raise ValueError("Not connected to ES-9. Use connect_es9 first.")
    if not engine.audio_running:
        config = _config(ctx)
        device = config.audio_device or None
        engine.connect_audio(device=device, sample_rate=config.sample_rate)
    return engine


# ===================================================================
# 1. CONNECTION TOOLS (4)
# ===================================================================


@mcp.tool()
async def list_midi_ports(ctx: Context) -> str:
    """List all available MIDI input and output ports."""
    engine = _engine(ctx)
    loop = asyncio.get_event_loop()
    out_ports = await loop.run_in_executor(None, engine.list_output_ports)
    in_ports = await loop.run_in_executor(None, engine.list_input_ports)

    lines = ["=== MIDI Output Ports ==="]
    for i, p in enumerate(out_ports):
        lines.append(f"  [{i}] {p}")
    if not out_ports:
        lines.append("  (none)")

    lines.append("\n=== MIDI Input Ports ===")
    for i, p in enumerate(in_ports):
        lines.append(f"  [{i}] {p}")
    if not in_ports:
        lines.append("  (none)")

    return "\n".join(lines)


@mcp.tool()
async def list_audio_devices(ctx: Context) -> str:
    """List all audio devices (sounddevice). Shows input/output channel counts."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, CVEngine.list_audio_devices)


@mcp.tool()
async def connect_es9(
    ctx: Context,
    midi_output: str = "",
    midi_input: str = "",
    audio_device: str = "",
    sample_rate: int = 0,
) -> str:
    """Connect to ES-9 MIDI and audio. Auto-detects if parameters are empty.

    Args:
        midi_output: MIDI output port name/substring or index. Empty = auto-detect "ES-9".
        midi_input: MIDI input port name/substring or index. Empty = match output.
        audio_device: Audio device name/substring or index. Empty = auto-detect "ES-9".
        sample_rate: Sample rate (48000 or 96000). 0 = use config default.
    """
    engine = _engine(ctx)
    config = _config(ctx)

    out = midi_output or config.midi_output_port
    inp = midi_input or config.midi_input_port or out

    # Try numeric index
    for val_name in ["out", "inp"]:
        val = locals()[val_name]
        try:
            locals()[val_name] = int(val)
        except (ValueError, TypeError):
            pass

    sr = sample_rate or config.sample_rate

    # Audio device
    audio_dev: int | str | None = audio_device or config.audio_device or None
    if audio_dev:
        try:
            audio_dev = int(audio_dev)
        except (ValueError, TypeError):
            pass

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: engine.connect(
            midi_output=out,
            midi_input=inp,
            audio_device=audio_dev,
            sample_rate=sr,
        ),
    )
    return result


@mcp.tool()
async def disconnect_es9(ctx: Context) -> str:
    """Disconnect from ES-9 MIDI and audio."""
    engine = _engine(ctx)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, engine.disconnect)


# ===================================================================
# 2. MIXER — VIRTUAL (5)
# ===================================================================


@mcp.tool()
async def set_mix_level(
    ctx: Context, mix_bus: int, channel: int, db: float
) -> str:
    """Set virtual mix level in dB for a channel in a mix bus.

    Args:
        mix_bus: Mix bus number (0-7).
        channel: Input channel (0-based).
        db: Level in dB. 0 = unity, -inf = silence, +12 = max.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_virtual_mix(mix_bus, channel, db)
    )
    vmix = db_to_vmix(db)
    return f"Mix bus {mix_bus} ch {channel}: {db:+.1f} dB (vmix={vmix})"


@mcp.tool()
async def set_mix_pan(
    ctx: Context, mix_bus: int, channel: int, pan: int
) -> str:
    """Set pan for a channel in a mix bus.

    Args:
        mix_bus: Mix bus number (0-7).
        channel: Input channel (0-based).
        pan: Pan value -63 (full left) to +63 (full right), 0 = center.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_virtual_pan(mix_bus, channel, pan)
    )
    return f"Mix bus {mix_bus} ch {channel}: pan {pan:+d}"


@mcp.tool()
async def get_mix_state(ctx: Context) -> str:
    """Request and show current virtual mix levels and pans from ES-9."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    state = await loop.run_in_executor(None, engine.get_mix_state)
    if "error" in state:
        return f"Error: {state['error']}"

    lines = ["=== Mix State ==="]
    for key, val in state.items():
        lines.append(f"  {key}: {val}")
    return "\n".join(lines)


@mcp.tool()
async def set_raw_mix_level(
    ctx: Context, mix_bus: int, channel: int, db: float
) -> str:
    """Set raw 21-bit mix level (advanced). Uses dB input.

    Args:
        mix_bus: Mix bus number (0-15).
        channel: Channel in the mix bus (0-based).
        db: Level in dB. 0 = unity (0x2000), +12 = max (0x7FFF), -inf = 0.
    """
    engine = _require_midi(ctx)
    raw = db_to_raw_mix(db)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_raw_mix(mix_bus, channel, db)
    )
    return f"Raw mix bus {mix_bus} ch {channel}: {db:+.1f} dB (raw=0x{raw:04X})"


@mcp.tool()
async def reset_mixer(ctx: Context) -> str:
    """Zero all virtual mix levels (set everything to -inf dB)."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.reset_mixer)
    return "All mix levels reset to -inf dB"


# ===================================================================
# 3. ROUTING (4)
# ===================================================================


@mcp.tool()
async def set_input_routing(
    ctx: Context, dsp_block: int, source_names: list[str]
) -> str:
    """Route named sources to DSP capture channels.

    Args:
        dsp_block: DSP block 0-3 (each handles a group of channels).
        source_names: List of source names, e.g. ["input_1", "usb_3", "mix_1"].
            Available: input_1..14, usb_1..16, mix_1..8, bus_1..16, spdif_l, spdif_r.
    """
    engine = _require_midi(ctx)
    codes = [resolve_input_source(name) for name in source_names]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_capture_routing(dsp_block, codes)
    )
    mapping = ", ".join(
        f"{name}→ch{i}" for i, name in enumerate(source_names)
    )
    return f"DSP {dsp_block} capture routing: {mapping}"


@mcp.tool()
async def set_output_routing(
    ctx: Context, dsp_block: int, dest_names: list[str]
) -> str:
    """Route DSP output channels to named physical outputs.

    Args:
        dsp_block: DSP block 0-3.
        dest_names: List of destination names, e.g. ["output_1", "main_l", "phones_r"].
            Available: output_1..8, main_l, main_r, phones_l, phones_r, es5_l, es5_r, bus_1..16.
    """
    engine = _require_midi(ctx)
    codes = [resolve_output_dest(name) for name in dest_names]
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_output_routing(dsp_block, codes)
    )
    mapping = ", ".join(
        f"ch{i}→{name}" for i, name in enumerate(dest_names)
    )
    return f"DSP {dsp_block} output routing: {mapping}"


@mcp.tool()
async def get_routing(ctx: Context) -> str:
    """Show current routing matrix in human-readable form.

    Requires a config dump to have been fetched first (use get_config).
    """
    engine = _require_midi(ctx)
    if engine._config_cache is None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, engine.get_config)

    config = engine._config_cache
    if not config or not config.get("parsed"):
        return "Config not available. Try get_config first."

    lines = ["=== Routing ==="]
    lines.append("(Routing details from config dump)")
    for key, val in config.items():
        if key.startswith("routing") or key.startswith("capture") or key.startswith("output"):
            lines.append(f"  {key}: {val}")
    if len(lines) == 2:
        lines.append("  (routing details not yet parsed from config dump)")
    return "\n".join(lines)


@mcp.tool()
async def set_options(
    ctx: Context,
    mixer2_spdif: bool = False,
    midi_thru: bool = False,
) -> str:
    """Toggle ES-9 options: mixer2/SPDIF mode, MIDI thru.

    Args:
        mixer2_spdif: Enable mixer2/SPDIF mode.
        midi_thru: Enable MIDI thru.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_options(mixer2_spdif, midi_thru)
    )
    parts = []
    parts.append(f"mixer2/SPDIF={'on' if mixer2_spdif else 'off'}")
    parts.append(f"MIDI thru={'on' if midi_thru else 'off'}")
    return f"Options set: {', '.join(parts)}"


# ===================================================================
# 4. EQ (3)
# ===================================================================


@mcp.tool()
async def set_eq_filter(
    ctx: Context,
    channel: int,
    slot: int,
    filter_type: str,
    freq_hz: float,
    q: float = 1.0,
    gain_db: float = 0.0,
) -> str:
    """Configure an EQ filter with human-readable parameters.

    Args:
        channel: Channel index (0-based).
        slot: EQ slot (0-3).
        filter_type: One of: lp1, hp1, lp2, hp2, low_shelf, high_shelf, peak, phase_invert.
        freq_hz: Filter frequency in Hz (20-20000).
        q: Q factor (0.1-20.0). Relevant for peak/shelf filters.
        gain_db: Gain in dB (-24 to +24). Relevant for shelf/peak filters.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: engine.set_eq(channel, slot, filter_type, True, freq_hz, q, gain_db),
    )
    return (
        f"EQ ch {channel} slot {slot}: {filter_type} @ {freq_hz:.0f}Hz "
        f"Q={q:.1f} gain={gain_db:+.1f}dB"
    )


@mcp.tool()
async def bypass_eq_filter(ctx: Context, channel: int, slot: int) -> str:
    """Disable/bypass an EQ filter slot.

    Args:
        channel: Channel index (0-based).
        slot: EQ slot (0-3).
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.bypass_eq(channel, slot)
    )
    return f"EQ ch {channel} slot {slot}: bypassed"


@mcp.tool()
async def get_eq_state(ctx: Context) -> str:
    """Show all EQ filter configurations from cached config.

    Requires get_config to have been called first.
    """
    engine = _require_midi(ctx)
    config = engine._config_cache
    if not config or not config.get("parsed"):
        return "Config not available. Use get_config first."

    lines = ["=== EQ State ==="]
    for key, val in config.items():
        if "eq" in key.lower():
            lines.append(f"  {key}: {val}")
    if len(lines) == 1:
        lines.append("  (EQ details not yet parsed from config dump)")
    return "\n".join(lines)


# ===================================================================
# 5. CONFIG (6)
# ===================================================================


@mcp.tool()
async def get_config(ctx: Context) -> str:
    """Request and parse full configuration dump from ES-9.

    Caches the result for use by get_routing, get_eq_state, etc.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    config = await loop.run_in_executor(None, engine.get_config)

    lines = ["=== ES-9 Config Dump ==="]
    lines.append(f"  Messages received: {config.get('raw_messages', 0)}")
    lines.append(f"  Payload bytes: {config.get('payload_bytes', 0)}")
    lines.append(f"  Parsed: {config.get('parsed', False)}")
    return "\n".join(lines)


@mcp.tool()
async def save_config(ctx: Context, slot: int = 0) -> str:
    """Save current configuration to flash.

    Args:
        slot: 0 = hosted mode config, 1 = standalone mode config.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: engine.save_config(slot))
    mode = "hosted" if slot == 0 else "standalone"
    return f"Config saved to flash (slot {slot}: {mode})"


@mcp.tool()
async def restore_config(ctx: Context) -> str:
    """Restore configuration from flash."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.restore_config)
    return "Config restored from flash"


@mcp.tool()
async def reset_config(ctx: Context, confirm: bool = False) -> str:
    """Factory reset the ES-9. DESTRUCTIVE — resets all settings.

    Args:
        confirm: Must be True to execute. Safety check.
    """
    if not confirm:
        return "Factory reset requires confirm=True. This will erase all settings!"
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.factory_reset)
    return "Factory reset sent. ES-9 will restart with default settings."


@mcp.tool()
async def get_firmware_version(ctx: Context) -> str:
    """Query ES-9 firmware version."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, engine.get_firmware_version)


@mcp.tool()
async def get_sample_rate(ctx: Context) -> str:
    """Query current sample rate from ES-9."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.get_sample_rate)
    if "error" in result:
        return f"Error: {result['error']}"
    sr = result.get("sample_rate", "unknown")
    return f"Sample rate: {sr} Hz"


# ===================================================================
# 6. CV OUTPUT (6)
# ===================================================================


@mcp.tool()
async def set_cv_voltage(ctx: Context, channel: int, volts: float) -> str:
    """Set a static voltage on an ES-9 output channel.

    Args:
        channel: Output channel (0-7).
        volts: Voltage in volts (-10.0 to +10.0).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    source = StaticCV(volts)
    cv.set_source(channel, source)
    return f"Output {channel}: {volts:+.3f}V (static)"


@mcp.tool()
async def set_cv_gate(
    ctx: Context,
    channel: int,
    high: bool,
    voltage: float = 5.0,
) -> str:
    """Set a gate output high or low.

    Args:
        channel: Output channel (0-7).
        high: True = gate high, False = gate low (0V).
        voltage: Gate high voltage (default 5V, typical for Eurorack).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    source = GateCV(high=high, voltage=voltage)
    cv.set_source(channel, source)
    state = "HIGH" if high else "LOW"
    return f"Output {channel}: gate {state} ({voltage}V)"


@mcp.tool()
async def generate_lfo(
    ctx: Context,
    channel: int,
    shape: str = "sine",
    rate_hz: float = 1.0,
    depth_v: float = 5.0,
    offset_v: float = 0.0,
) -> str:
    """Start an LFO on an output channel.

    Args:
        channel: Output channel (0-7).
        shape: Waveform shape: sine, triangle, saw, square, random.
        rate_hz: LFO frequency in Hz.
        depth_v: Peak-to-peak depth in volts.
        offset_v: DC offset in volts.
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    source = LfoCv(shape=shape, rate_hz=rate_hz, depth_v=depth_v, offset_v=offset_v)
    cv.set_source(channel, source)
    return (
        f"Output {channel}: LFO {shape} @ {rate_hz:.2f}Hz, "
        f"depth={depth_v:.1f}V, offset={offset_v:.1f}V"
    )


@mcp.tool()
async def trigger_envelope(
    ctx: Context,
    channel: int,
    attack_ms: float = 10.0,
    release_ms: float = 100.0,
    peak_v: float = 5.0,
) -> str:
    """Trigger a one-shot attack-release envelope on an output channel.

    Args:
        channel: Output channel (0-7).
        attack_ms: Attack time in milliseconds.
        release_ms: Release time in milliseconds.
        peak_v: Peak voltage.
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    source = EnvelopeCV(attack_ms=attack_ms, release_ms=release_ms, peak_v=peak_v)
    cv.set_source(channel, source)
    total = attack_ms + release_ms
    return (
        f"Output {channel}: envelope A={attack_ms:.0f}ms R={release_ms:.0f}ms "
        f"peak={peak_v:.1f}V (total {total:.0f}ms)"
    )


@mcp.tool()
async def send_cv_sequence(
    ctx: Context,
    channel: int,
    steps: list[dict],
    loop: bool = True,
) -> str:
    """Play a pitch/voltage sequence on an output channel.

    Args:
        channel: Output channel (0-7).
        steps: List of step dicts. Each has: voltage (float, in volts),
               duration_ms (float, step duration). Example:
               [{"voltage": 0.0, "duration_ms": 500}, {"voltage": 0.417, "duration_ms": 500}]
        loop: Whether to loop the sequence (default True).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    source = SequenceCV(steps=steps, loop=loop)
    cv.set_source(channel, source)
    mode = "looping" if loop else "one-shot"
    return f"Output {channel}: sequence ({len(steps)} steps, {mode})"


@mcp.tool()
async def stop_cv(ctx: Context, channel: int) -> str:
    """Stop and zero a CV output channel.

    Args:
        channel: Output channel (0-7).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    cv.clear_source(channel)
    return f"Output {channel}: stopped (0V)"


# ===================================================================
# 7. CV INPUT (3)
# ===================================================================


@mcp.tool()
async def read_cv_voltage(ctx: Context, channel: int) -> str:
    """Read the current voltage on an ES-9 input channel.

    Args:
        channel: Input channel (0-13).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    voltage = cv.read_voltage(channel)
    if voltage is None:
        return f"Input {channel}: no data (start monitoring first)"
    return f"Input {channel}: {voltage:+.3f}V"


@mcp.tool()
async def monitor_cv_input(
    ctx: Context, interval: float = 0.1
) -> str:
    """Start logging input voltage readings at regular intervals.

    Args:
        interval: Seconds between readings (default 0.1).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    cv.start_input_monitor(interval)
    return f"Input monitoring started (interval={interval}s)"


@mcp.tool()
async def get_cv_log(
    ctx: Context, channel: int, count: int = 50
) -> str:
    """Get recent input voltage readings for a channel.

    Args:
        channel: Input channel (0-13).
        count: Number of readings to return (max 200).
    """
    engine = _ensure_audio(ctx)
    cv = engine.cv_engine
    count = min(count, 200)
    readings = cv.get_input_log(channel, count)

    if not readings:
        return f"Input {channel}: no readings. Use monitor_cv_input first."

    lines = [f"=== Input {channel} Voltage Log ({len(readings)} samples) ==="]
    for r in readings[-20:]:  # Show last 20
        lines.append(f"  #{r['sample']}: {r['voltage']:+.3f}V")
    if len(readings) > 20:
        lines.insert(1, f"  (showing last 20 of {len(readings)})")
    return "\n".join(lines)


# ===================================================================
# 8. UTILITY (4)
# ===================================================================


@mcp.tool()
async def note_to_voltage(ctx: Context, note: str) -> str:
    """Convert a MIDI note name or number to V/Oct voltage.

    Args:
        note: Note name ("C4", "F#3", "Bb5") or MIDI number ("60").
    """
    try:
        midi = int(note)
    except ValueError:
        midi = note_name_to_midi(note)

    voltage = midi_to_voltage(midi)
    name = midi_to_note_name(midi)
    sample = voltage_to_sample(voltage)
    return f"{name} (MIDI {midi}) = {voltage:+.4f}V (sample: {sample:+.4f})"


@mcp.tool()
async def voltage_to_note(ctx: Context, volts: float) -> str:
    """Convert a voltage to the nearest MIDI note name (V/Oct).

    Args:
        volts: Voltage in V/Oct.
    """
    midi = voltage_to_midi(volts)
    name = midi_to_note_name(midi)
    exact_v = midi_to_voltage(midi)
    cents_off = (volts - exact_v) * 1200.0  # 1V = 1200 cents
    return f"{volts:+.4f}V ≈ {name} (MIDI {midi}, {cents_off:+.0f} cents)"


@mcp.tool()
async def set_dc_offset(
    ctx: Context, output: int, voltage: float
) -> str:
    """Set DC offset trim on an ES-9 output channel.

    Args:
        output: Output channel (0-based).
        voltage: DC offset in volts.
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_dc_offset(output, voltage)
    )
    return f"Output {output}: DC offset = {voltage:+.3f}V"


@mcp.tool()
async def set_hpf(
    ctx: Context, channel: int, enabled: bool
) -> str:
    """Enable or disable the input DC blocking high-pass filter.

    Args:
        channel: Input channel (0-based).
        enabled: True to enable HPF, False to disable (pass DC).
    """
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: engine.set_hpf(channel, enabled)
    )
    state = "enabled" if enabled else "disabled"
    return f"Input {channel}: DC blocking HPF {state}"


# ===================================================================
# 9. STATUS (2)
# ===================================================================


@mcp.tool()
async def query_status(ctx: Context) -> str:
    """Show full ES-9 MCP server status: MIDI, audio, mixer, routing, CV, EQ."""
    engine = _engine(ctx)
    config = _config(ctx)

    lines = ["=== ES-9 MCP Status ==="]

    # MIDI
    lines.append(f"  MIDI connected: {engine.midi_connected}")
    if engine.midi_connected:
        lines.append(f"    Output: {engine._out_port_name}")
        lines.append(f"    Input:  {engine._in_port_name}")
    lines.append(f"  MIDI monitor: {'running' if engine.is_monitoring else 'stopped'}")

    # Audio
    lines.append(f"  Audio running: {engine.audio_running}")
    if engine.audio_running and engine.cv_engine:
        cv = engine.cv_engine
        lines.append(f"    Sample rate: {cv.sample_rate}Hz")
        sources = cv.get_source_info()
        if sources:
            lines.append("    CV outputs:")
            for ch, desc in sorted(sources.items()):
                lines.append(f"      ch {ch}: {desc}")
        else:
            lines.append("    CV outputs: (none active)")

    # Config
    lines.append(f"  Config cached: {engine._config_cache is not None}")
    lines.append(f"  Config sample rate: {config.sample_rate}Hz")

    return "\n".join(lines)


@mcp.tool()
async def get_cpu_usage(ctx: Context) -> str:
    """Request DSP CPU usage from ES-9."""
    engine = _require_midi(ctx)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.get_cpu_usage)
    if "error" in result:
        return f"Error: {result['error']}"
    cpu = result.get("cpu_percent", "unknown")
    return f"DSP CPU usage: {cpu}%"


# ===================================================================
# Entry point
# ===================================================================


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
