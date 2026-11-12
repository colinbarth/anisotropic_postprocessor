"""Programmable G-code procedures for startup, purge, layer change, cutting, and finishing.

Every procedure is a plain Python function that returns a ``list[str]`` of
G-code lines.  There is no hidden code generation at runtime: what a function
appends to its list is what appears in the output ``.gcode`` file.

Procedures accept an optional :class:`PrintContext` as their first argument so
they can read machine and filament parameters (temperatures, build volume,
retraction settings) and inspect or update the live :class:`PrintState`
(current ``x, y, z, e``, active tool, layer index).  Every parameter can also
be passed or overridden explicitly as a keyword argument.

Printer JSON profiles reference procedures by name in their ``"procedures"``
dictionary (resolved via :data:`PROCEDURES`), and callers can register new
procedures with :func:`register_procedure` or pass Python functions directly to
:func:`~anisotropic_postprocessor.postprocessor.generate_gcode`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from anisotropic_postprocessor.profiles import FilamentProfile, PrinterProfile

__all__ = [
    "PrintState",
    "PrintContext",
    "ProcedureFn",
    "PROCEDURES",
    "register_procedure",
    "resolve_procedure",
    "raise3d_start",
    "raise3d_purge",
    "raise3d_layer_change",
    "raise3d_end",
    "generic_start",
    "generic_purge",
    "generic_layer_change",
    "generic_end",
    "standard_retract",
    "standard_unretract",
    "standard_travel",
    "standard_tool_change",
    "prusa_xl_tool_change",
    "continuous_fiber_cut",
]


# ── Live machine state and context ───────────────────────────────────────────

@dataclass
class PrintState:
    """Mutable machine state tracked linearly during G-code generation.

    Coordinates ``x, y, z`` and ``e`` are in printer units (mm), and
    ``feedrate`` is in mm/min.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    e: float = 0.0
    feedrate: float = 1000.0
    tool: int = 0
    fan_percent: float = 0.0
    is_retracted: bool = False
    relative_extrusion: bool = False
    layer: int = 0
    layer_z: float = 0.0
    layer_height: float = 0.2
    current_kind: str | None = None
    has_moved_xy: bool = False
    total_extruded_by_tool: dict[int, float] = field(default_factory=dict)

    def record_extrusion(self, delta_e: float) -> None:
        """Advance extruder state by ``delta_e`` mm of filament."""
        if self.relative_extrusion:
            self.e = float(delta_e)
        else:
            self.e += float(delta_e)
        if delta_e > 0:
            self.total_extruded_by_tool[self.tool] = (
                self.total_extruded_by_tool.get(self.tool, 0.0) + float(delta_e)
            )


@dataclass
class PrintContext:
    """Bundle of printer profile, active filament profile, and live state.

    Passed to every procedure function during G-code generation.
    """

    printer: PrinterProfile
    filament: FilamentProfile
    state: PrintState = field(default_factory=PrintState)
    width: float = 0.4
    layer_height: float = 0.2
    print_speed: float | None = None
    retract: bool = True
    hop: float | None = None
    filaments_by_kind: dict[str, FilamentProfile] = field(default_factory=dict)
    tool_map: dict[str, int] = field(default_factory=dict)

    def effective_retract_length(self) -> float:
        if self.filament.retract_length is not None:
            return float(self.filament.retract_length)
        return float(self.printer.retract_length)

    def effective_retract_speed(self) -> float:
        if self.filament.retract_speed is not None:
            return float(self.filament.retract_speed)
        return float(self.printer.retract_speed)

    def effective_unretract_speed(self) -> float:
        if self.filament.unretract_speed is not None:
            return float(self.filament.unretract_speed)
        return float(self.printer.unretract_speed)

    def effective_hop(self) -> float:
        if self.hop is not None:
            return float(self.hop)
        if self.filament.z_hop is not None:
            return float(self.filament.z_hop)
        return float(self.printer.z_hop)

    def effective_print_speed(self, layer: int | None = None) -> float:
        if self.print_speed is not None:
            return float(self.print_speed)
        active_layer = self.state.layer if layer is None else layer
        if active_layer == 0:
            return float(self.printer.first_layer_print_speed)
        return float(self.printer.default_print_speed)


ProcedureFn = Callable[..., list[str]]


def _fan_pwm(percent: float) -> int:
    """Convert a 0-100% fan speed to a 0-255 PWM integer for ``M106 S``."""
    return max(0, min(255, int(round(float(percent) * 255.0 / 100.0))))


# ── Raise3D procedures ───────────────────────────────────────────────────────

def raise3d_start(
    ctx: PrintContext | None = None,
    *,
    extrusion_multiplier: float | None = None,
    heatbed_temp: float | None = None,
    nozzle_temp_extruder0: float | None = None,
    pressure_advance: float | None = None,
    fan_beginning: float | None = None,
    rapid_speed: float | None = None,
    tool: int | None = None,
) -> list[str]:
    """Raise3D start-up sequence as G-code lines, without the purge line.

    Sets the extrusion multiplier, starts heating the bed, heats the active
    nozzle and waits, selects the active tool, waits for the bed, sets pressure
    advance on drives D0 and D1, switches to millimeters (``G21``), absolute
    coordinates (``G90``), and the configured extrusion mode (``M82`` / ``M83``),
    sets initial fan speed, homes all axes, raises the bed to Z0.5, and resets
    E to zero.
    """
    active_tool = tool if tool is not None else (ctx.state.tool if ctx else 0)
    flow_pct = (
        extrusion_multiplier
        if extrusion_multiplier is not None
        else (ctx.filament.flow_percent if ctx else 100.0)
    )
    bed_temp = (
        heatbed_temp
        if heatbed_temp is not None
        else (
            ctx.filament.bed_temperature_initial_layer
            if ctx and ctx.filament.bed_temperature_initial_layer is not None
            else (ctx.filament.bed_temperature if ctx else 55.0)
        )
    )
    nozzle_temp = (
        nozzle_temp_extruder0
        if nozzle_temp_extruder0 is not None
        else (
            ctx.filament.nozzle_temperature_initial_layer
            if ctx and ctx.filament.nozzle_temperature_initial_layer is not None
            else (ctx.filament.nozzle_temperature if ctx else 215.0)
        )
    )
    pa = (
        pressure_advance
        if pressure_advance is not None
        else (ctx.filament.pressure_advance if ctx else 0.03)
    )
    fan_init = (
        fan_beginning
        if fan_beginning is not None
        else (ctx.filament.fan_percent_for_layer(0) if ctx else 0.0)
    )
    travel_f = (
        rapid_speed
        if rapid_speed is not None
        else (ctx.printer.rapid_speed if ctx else 9000.0)
    )
    relative_e = ctx.printer.relative_extrusion if ctx else False

    lines = [
        "; --- START OF STARTING PROCEDURE (Raise3D) ---",
        f"M221 T{active_tool} S{flow_pct:.2f} ; Extrusion multiplier",
        f"M140 S{bed_temp:.2f} ; Start heating bed",
        f"M109 T{active_tool} S{nozzle_temp:.2f} ; Nozzle heating with tool specification",
        f"T{active_tool} ; Select Tool {active_tool}",
        f"M190 S{bed_temp:.2f} ; Wait for bed temperature",
        f"M572 D0 S{pa:.2f} ; Pressure advance setting",
        f"M572 D1 S{pa:.2f} ; Pressure advance setting",
        "G21 ; Set units to millimeters",
        "G90 ; Absolute coordinates",
        "M83 ; Relative extrusion" if relative_e else "M82 ; Absolute extrusion",
        "M107 ; Fan off",
    ]
    if fan_init > 0:
        lines.append(f"M106 S{_fan_pwm(fan_init)} ; Initial fan speed")
    lines.extend(
        [
            "G28 ; Home axes",
            "G1 Z0.500 F500 ; Raise bed",
            "G92 E0 ; Reset extruder position",
            f"G0 F{travel_f:.0f} ; Set rapid travel speed",
            "; --- END OF STARTING PROCEDURE ---",
        ]
    )

    if ctx is not None:
        ctx.state.tool = active_tool
        ctx.state.z = 0.5
        ctx.state.e = 0.0
        ctx.state.feedrate = travel_f
        ctx.state.fan_percent = fan_init
        ctx.state.relative_extrusion = relative_e
    return lines


def raise3d_purge(
    ctx: PrintContext | None = None,
    *,
    purge_x: float = 60.0,
    purge_y: float = -2.0,
    purge_z: float = 0.2,
    purge_length: float = 80.0,
    purge_feedrate: float = 200.0,
    purge_length_extruded: float = 12.0,
    purge_end_height: float = 5.0,
    z_feedrate: float | None = None,
    rapid_speed: float | None = None,
) -> list[str]:
    """Raise3D purge line as G-code lines.

    Travels to ``(purge_x, purge_y, purge_z)`` and extrudes a straight line in
    +X to prime the nozzle, then lifts to ``purge_end_height`` while pushing
    out 3 mm more filament and resets ``E`` to zero.
    """
    z_f = (
        z_feedrate
        if z_feedrate is not None
        else (ctx.printer.z_feedrate if ctx else 300.0)
    )
    travel_f = (
        rapid_speed
        if rapid_speed is not None
        else (ctx.printer.rapid_speed if ctx else 9000.0)
    )
    travel_accel = ctx.printer.travel_acceleration if ctx else 1500.0
    relative_e = ctx.state.relative_extrusion if ctx else False

    purge_end_x = purge_x + purge_length
    hover_extrusion = 3.0 if relative_e else (purge_length_extruded + 3.0)

    lines = [
        "; --- START OF PURGE PROCEDURE ---",
        ";TYPE:PURGE",
        f"G0 F{z_f:.0f} X{purge_x:.3f} Y{purge_y:.3f} Z{purge_z:.3f}",
        f"G1 X{purge_end_x:.3f} E{purge_length_extruded:.4f} F{purge_feedrate:.0f} ; Purge line",
        f"G0 F{z_f:.0f} Z{purge_end_height:.3f}",
        f"G1 E{hover_extrusion:.4f} F{purge_feedrate:.0f} ; Purge end hover",
        "G92 E0 ; Reset extruder position",
        f"G0 F{travel_f:.0f}",
        "M117 Printing...",
        "M1001",
        f"M204 T{travel_accel:.0f}",
        "; --- END OF PURGE PROCEDURE ---",
    ]

    if ctx is not None:
        ctx.state.x = purge_end_x
        ctx.state.y = purge_y
        ctx.state.z = purge_end_height
        ctx.state.e = 0.0
        ctx.state.feedrate = travel_f
        ctx.state.has_moved_xy = True
        ctx.state.record_extrusion(purge_length_extruded + 3.0)
        ctx.state.e = 0.0
    return lines


def raise3d_layer_change(
    ctx: PrintContext | None = None,
    *,
    layer: int | None = None,
    z: float | None = None,
    layer_height: float | None = None,
    next_x: float | None = None,
    next_y: float | None = None,
    max_z: float | None = None,
    fan_percent: float | None = None,
    aux_fan: int | None = None,
    z_feedrate: float | None = None,
    rapid_speed: float | None = None,
    acceleration: float | None = None,
) -> list[str]:
    """Raise3D layer-change sequence mirroring ideaMaker.

    Emits ``;LAYER``, ``;Z``, and ``;HEIGHT`` comments, sets part-cooling and
    auxiliary fans, moves slowly to the new layer height ``z``, optionally
    travels to ``(next_x, next_y)``, and sets print acceleration ``M204 P``.
    """
    layer_idx = layer if layer is not None else (ctx.state.layer if ctx else 0)
    layer_z = z if z is not None else (ctx.state.layer_z if ctx else 0.2)
    lh = (
        layer_height
        if layer_height is not None
        else (ctx.state.layer_height if ctx else 0.2)
    )
    limit_z = max_z if max_z is not None else (ctx.printer.max_z if ctx else 300.0)
    if layer_z > limit_z:
        raise ValueError(
            f"z ({layer_z}) is above the printer's max_z ({limit_z})."
        )

    fan_pct = (
        fan_percent
        if fan_percent is not None
        else (ctx.filament.fan_percent_for_layer(layer_idx) if ctx else 100.0)
    )
    aux_pwm = (
        aux_fan
        if aux_fan is not None
        else (_fan_pwm(ctx.filament.auxiliary_fan_speed) if ctx else 0)
    )
    z_f = (
        z_feedrate
        if z_feedrate is not None
        else (ctx.printer.z_feedrate if ctx else 300.0)
    )
    travel_f = (
        rapid_speed
        if rapid_speed is not None
        else (ctx.printer.rapid_speed if ctx else 9000.0)
    )
    accel = (
        acceleration
        if acceleration is not None
        else (ctx.printer.print_acceleration if ctx else 500.0)
    )

    lines = [
        f";LAYER:{layer_idx}",
        f";Z:{layer_z:.3f}",
        f";HEIGHT:{lh:.3f}",
        f"M106 S{_fan_pwm(fan_pct)}",
        f"M106 P2 S{aux_pwm} ; Auxiliary fan",
        f"G0 F{z_f:.0f} Z{layer_z:.3f}",
    ]
    if next_x is not None and next_y is not None:
        lines.append(f"G0 F{travel_f:.0f} X{next_x:.3f} Y{next_y:.3f}")
    else:
        lines.append(f"G0 F{travel_f:.0f}")
    lines.append(f"M204 P{accel:.2f} ; Set acceleration")

    if ctx is not None:
        ctx.state.layer = layer_idx
        ctx.state.layer_z = layer_z
        ctx.state.z = layer_z
        ctx.state.layer_height = lh
        ctx.state.fan_percent = fan_pct
        ctx.state.feedrate = travel_f
        if next_x is not None and next_y is not None:
            ctx.state.x = float(next_x)
            ctx.state.y = float(next_y)
            ctx.state.has_moved_xy = True
    return lines


def raise3d_end(
    ctx: PrintContext | None = None,
    *,
    last_z: float | None = None,
    max_z: float | None = None,
    acceleration: float | None = None,
    retraction: float | None = None,
    retraction_feedrate: float | None = None,
    z_lift: float = 4.8,
    z_lift_feedrate: float | None = None,
    final_z_lift: float = 0.5,
) -> list[str]:
    """Raise3D end-of-print sequence mirroring ideaMaker.

    Both Z lifts are clamped so the nozzle never exceeds ``max_z``.
    """
    current_z = last_z if last_z is not None else (ctx.state.z if ctx else 200.0)
    limit_z = max_z if max_z is not None else (ctx.printer.max_z if ctx else 300.0)
    if current_z > limit_z:
        raise ValueError(
            f"last_z ({current_z}) is above the printer's max_z ({limit_z})."
        )
    lift_z = min(current_z + z_lift, limit_z)
    final_lift = max(0.0, min(final_z_lift, limit_z - lift_z))

    accel = (
        acceleration
        if acceleration is not None
        else (ctx.printer.print_acceleration if ctx else 1000.0)
    )
    retract_len = (
        retraction
        if retraction is not None
        else (ctx.effective_retract_length() if ctx else 0.5)
    )
    retract_f = (
        retraction_feedrate
        if retraction_feedrate is not None
        else (ctx.effective_retract_speed() if ctx else 2400.0)
    )
    z_f = (
        z_lift_feedrate
        if z_lift_feedrate is not None
        else (ctx.printer.z_feedrate if ctx else 300.0)
    )

    lines = [
        "; --- START OF ENDING PROCEDURE (Raise3D) ---",
        f"M204 P{accel:.2f} ; Set acceleration",
        "M106 S0",
        "M82",
        "G92 E0",
        f"G1 F{retract_f:.0f} E-{retract_len:.4f} ; Retract",
        f"G0 F{z_f:.0f} Z{lift_z:.3f} ; Lift nozzle",
        "M221 T0 S100 ; Reset extrusion multiplier",
        "M82",
        "G92 E0",
        "M1002 ; End of print",
        "M104 S0 T0",
        "M104 S0",
        "M140 S0",
        "M106 P2 S0 ; Fan 2 off",
        "M107 ; Fans off",
        "G91 ; Relative coordinates",
        "G1 E-1.0000 F300 ; Retract",
        f"G1 Z{final_lift:.3f} E-5.0000 Y-20.000 F9000 ; Retract, lift and move away",
        "G28 X0 U0 ; Home X",
        "M84 ; Disable motors",
        "G90 ; Absolute coordinates",
        "; --- END OF ENDING PROCEDURE ---",
    ]

    if ctx is not None:
        ctx.state.z = lift_z + final_lift
        ctx.state.e = 0.0
        ctx.state.fan_percent = 0.0
    return lines


# ── Generic / standard RepRap & Marlin procedures ────────────────────────────

def generic_start(
    ctx: PrintContext | None = None,
    *,
    bed_temp: float | None = None,
    nozzle_temp: float | None = None,
    rapid_speed: float | None = None,
    tool: int | None = None,
) -> list[str]:
    """Standard RepRap / Marlin start-up procedure."""
    active_tool = tool if tool is not None else (ctx.state.tool if ctx else 0)
    b_temp = (
        bed_temp
        if bed_temp is not None
        else (
            ctx.filament.bed_temperature_initial_layer
            if ctx and ctx.filament.bed_temperature_initial_layer is not None
            else (ctx.filament.bed_temperature if ctx else 60.0)
        )
    )
    n_temp = (
        nozzle_temp
        if nozzle_temp is not None
        else (
            ctx.filament.nozzle_temperature_initial_layer
            if ctx and ctx.filament.nozzle_temperature_initial_layer is not None
            else (ctx.filament.nozzle_temperature if ctx else 215.0)
        )
    )
    travel_f = (
        rapid_speed
        if rapid_speed is not None
        else (ctx.printer.rapid_speed if ctx else 9000.0)
    )
    relative_e = ctx.printer.relative_extrusion if ctx else False
    flow_pct = ctx.filament.flow_percent if ctx else 100.0

    lines = [
        "; --- START OF STARTING PROCEDURE ---",
        "G21 ; Set units to millimeters",
        "G90 ; Absolute coordinates",
        "M83 ; Relative extrusion" if relative_e else "M82 ; Absolute extrusion",
        f"M140 S{b_temp:.2f} ; Start heating bed",
        f"M104 T{active_tool} S{n_temp:.2f} ; Start heating nozzle",
        "G28 ; Home axes",
    ]
    if ctx is not None and ctx.printer.features.get("mesh_bed_leveling", False):
        lines.append("G29 ; Mesh bed leveling")
    lines.extend(
        [
            f"T{active_tool} ; Select tool",
            f"M190 S{b_temp:.2f} ; Wait for bed temperature",
            f"M109 T{active_tool} S{n_temp:.2f} ; Wait for nozzle temperature",
            f"M221 S{flow_pct:.2f} ; Set flow percentage",
            "G92 E0 ; Reset extruder position",
            f"G0 F{travel_f:.0f} Z2.000",
            "; --- END OF STARTING PROCEDURE ---",
        ]
    )

    if ctx is not None:
        ctx.state.tool = active_tool
        ctx.state.z = 2.0
        ctx.state.e = 0.0
        ctx.state.feedrate = travel_f
        ctx.state.relative_extrusion = relative_e
    return lines


def generic_purge(
    ctx: PrintContext | None = None,
    *,
    purge_x: float = 10.0,
    purge_y: float = 5.0,
    purge_z: float = 0.2,
    purge_length: float = 60.0,
    purge_feedrate: float = 600.0,
    purge_extrusion: float = 8.0,
) -> list[str]:
    """Standard straight-line nozzle purge along the front edge of the bed."""
    z_f = ctx.printer.z_feedrate if ctx else 300.0
    travel_f = ctx.printer.rapid_speed if ctx else 9000.0
    end_x = purge_x + purge_length

    lines = [
        "; --- START OF PURGE PROCEDURE ---",
        ";TYPE:PURGE",
        f"G0 F{travel_f:.0f} X{purge_x:.3f} Y{purge_y:.3f}",
        f"G0 F{z_f:.0f} Z{purge_z:.3f}",
        "G92 E0",
        f"G1 X{end_x:.3f} Y{purge_y:.3f} E{purge_extrusion:.4f} F{purge_feedrate:.0f} ; Purge line",
        "G92 E0",
        f"G0 F{z_f:.0f} Z1.000",
        f"G0 F{travel_f:.0f}",
        "; --- END OF PURGE PROCEDURE ---",
    ]
    if ctx is not None:
        ctx.state.x = end_x
        ctx.state.y = purge_y
        ctx.state.z = 1.0
        ctx.state.e = 0.0
        ctx.state.has_moved_xy = True
        ctx.state.record_extrusion(purge_extrusion)
        ctx.state.e = 0.0
    return lines


def generic_layer_change(
    ctx: PrintContext | None = None,
    *,
    layer: int | None = None,
    z: float | None = None,
    layer_height: float | None = None,
    next_x: float | None = None,
    next_y: float | None = None,
) -> list[str]:
    """Standard layer-change sequence."""
    layer_idx = layer if layer is not None else (ctx.state.layer if ctx else 0)
    layer_z = z if z is not None else (ctx.state.layer_z if ctx else 0.2)
    lh = (
        layer_height
        if layer_height is not None
        else (ctx.state.layer_height if ctx else 0.2)
    )
    limit_z = ctx.printer.max_z if ctx else 250.0
    if layer_z > limit_z:
        raise ValueError(
            f"z ({layer_z}) is above the printer's max_z ({limit_z})."
        )

    fan_pct = ctx.filament.fan_percent_for_layer(layer_idx) if ctx else 100.0
    z_f = ctx.printer.z_feedrate if ctx else 300.0
    travel_f = ctx.printer.rapid_speed if ctx else 9000.0

    lines = [
        f";LAYER:{layer_idx}",
        f";Z:{layer_z:.3f}",
        f";HEIGHT:{lh:.3f}",
        f"M106 S{_fan_pwm(fan_pct)}",
        f"G0 F{z_f:.0f} Z{layer_z:.3f}",
    ]
    if next_x is not None and next_y is not None:
        lines.append(f"G0 F{travel_f:.0f} X{next_x:.3f} Y{next_y:.3f}")
    if ctx is not None:
        ctx.state.layer = layer_idx
        ctx.state.layer_z = layer_z
        ctx.state.z = layer_z
        ctx.state.layer_height = lh
        ctx.state.fan_percent = fan_pct
        if next_x is not None and next_y is not None:
            ctx.state.x = float(next_x)
            ctx.state.y = float(next_y)
            ctx.state.has_moved_xy = True
    return lines


def generic_end(
    ctx: PrintContext | None = None,
    *,
    last_z: float | None = None,
    z_lift: float = 5.0,
) -> list[str]:
    """Standard end-of-print sequence."""
    current_z = last_z if last_z is not None else (ctx.state.z if ctx else 10.0)
    limit_z = ctx.printer.max_z if ctx else 250.0
    if current_z > limit_z:
        raise ValueError(
            f"last_z ({current_z}) is above the printer's max_z ({limit_z})."
        )
    lift_z = min(current_z + z_lift, limit_z)
    retract_len = ctx.effective_retract_length() if ctx else 0.5
    retract_f = ctx.effective_retract_speed() if ctx else 2400.0
    z_f = ctx.printer.z_feedrate if ctx else 300.0

    lines = [
        "; --- START OF ENDING PROCEDURE ---",
        "M107 ; Fan off",
        "G92 E0",
        f"G1 F{retract_f:.0f} E-{retract_len:.4f} ; Retract",
        f"G0 F{z_f:.0f} Z{lift_z:.3f} ; Lift nozzle",
        "G28 X0 Y0 ; Home X and Y",
        "M104 S0 ; Turn off nozzle heater",
        "M140 S0 ; Turn off bed heater",
        "M84 ; Disable motors",
        "; --- END OF ENDING PROCEDURE ---",
    ]
    if ctx is not None:
        ctx.state.z = lift_z
        ctx.state.e = 0.0
        ctx.state.fan_percent = 0.0
    return lines


# ── Motion, retraction, toolchange, and fiber cutting procedures ─────────────

def standard_retract(
    ctx: PrintContext | None = None,
    *,
    length: float | None = None,
    speed: float | None = None,
) -> list[str]:
    """Retract filament before a travel move.

    Uses ``M83`` relative extrusion for the retract move (mirroring
    ``path_optimizer.paths.path_steps``) and restores ``M82`` afterwards if the
    printer is configured for absolute extrusion.
    """
    retract_len = (
        length
        if length is not None
        else (ctx.effective_retract_length() if ctx else 0.5)
    )
    retract_f = (
        speed
        if speed is not None
        else (ctx.effective_retract_speed() if ctx else 2400.0)
    )
    if retract_len <= 0:
        return []
    if ctx is not None and ctx.state.is_retracted:
        return []

    relative_e = ctx.state.relative_extrusion if ctx else False
    lines = [";ACTION:RETRACT"]
    if not relative_e:
        lines.append("M83")
    lines.append(f"G1 F{retract_f:.0f} E-{retract_len:.4f}")
    if not relative_e:
        lines.append("M82")

    if ctx is not None:
        ctx.state.is_retracted = True
        ctx.state.feedrate = retract_f
    return lines


def standard_unretract(
    ctx: PrintContext | None = None,
    *,
    length: float | None = None,
    speed: float | None = None,
) -> list[str]:
    """Unretract (prime) filament after arriving at the start of a path."""
    retract_len = (
        length
        if length is not None
        else (ctx.effective_retract_length() if ctx else 0.5)
    )
    unretract_f = (
        speed
        if speed is not None
        else (ctx.effective_unretract_speed() if ctx else 1500.0)
    )
    if retract_len <= 0:
        return []
    if ctx is not None and not ctx.state.is_retracted:
        return []

    relative_e = ctx.state.relative_extrusion if ctx else False
    lines = [";ACTION:UNRETRACT"]
    if not relative_e:
        lines.append("M83")
    lines.append(f"G1 F{unretract_f:.0f} E{retract_len:.4f}")
    if not relative_e:
        lines.append("M82")

    if ctx is not None:
        ctx.state.is_retracted = False
        ctx.state.feedrate = unretract_f
    return lines


def standard_travel(
    ctx: PrintContext | None = None,
    *,
    x: float,
    y: float,
    z: float,
    retract: bool | None = None,
    hop: float | None = None,
) -> list[str]:
    """Travel to ``(x, y, z)`` with optional retraction and Z-hop.

    Suppresses the Z-hop lift if the print head has not yet moved to any XY
    coordinate in the design, avoiding a blind vertical move before position is
    established.
    """
    do_retract = (
        retract if retract is not None else (ctx.retract if ctx is not None else False)
    )
    hop_height = (
        hop if hop is not None else (ctx.effective_hop() if ctx is not None else 0.0)
    )
    z_f = ctx.printer.z_feedrate if ctx is not None else 300.0
    travel_f = ctx.printer.rapid_speed if ctx is not None else 9000.0
    max_z = ctx.printer.max_z if ctx is not None else 300.0
    can_hop = (
        hop_height is not None
        and hop_height > 0.0
        and (ctx is None or ctx.state.has_moved_xy)
    )

    retract_fn = (
        resolve_procedure(ctx.printer.procedures.get("retract", "standard_retract"))
        if ctx is not None
        else standard_retract
    )
    unretract_fn = (
        resolve_procedure(ctx.printer.procedures.get("unretract", "standard_unretract"))
        if ctx is not None
        else standard_unretract
    )

    lines: list[str] = []
    if do_retract:
        lines.extend(retract_fn(ctx))

    if can_hop:
        base_z = max(ctx.state.z, z) if ctx is not None else z
        lift_z = min(base_z + float(hop_height), max_z)
        lines.extend(
            [
                ";TYPE:Z_HOP",
                f"G0 F{z_f:.0f} Z{lift_z:.3f}",
                ";TYPE:TRAVEL",
                f"G0 F{travel_f:.0f} X{x:.3f} Y{y:.3f} Z{lift_z:.3f}",
                ";TYPE:Z_HOP",
                f"G0 F{z_f:.0f} Z{z:.3f}",
            ]
        )
    else:
        lines.extend(
            [
                ";TYPE:TRAVEL",
                f"G0 F{travel_f:.0f} X{x:.3f} Y{y:.3f} Z{z:.3f}",
            ]
        )

    if do_retract:
        lines.extend(unretract_fn(ctx))

    if ctx is not None:
        ctx.state.x = float(x)
        ctx.state.y = float(y)
        ctx.state.z = float(z)
        ctx.state.feedrate = travel_f
        ctx.state.has_moved_xy = True
    return lines


def standard_tool_change(
    ctx: PrintContext | None = None,
    *,
    tool: int | str,
    filament: FilamentProfile | None = None,
    wait_for_temp: bool = False,
    reset_e: bool = True,
) -> list[str]:
    """Select ``tool`` and reset the extruder datum so absolute E stays valid."""
    target_profile = filament or (ctx.filament if ctx is not None else None)
    tool_cmd = tool if isinstance(tool, str) else f"T{tool}"
    if isinstance(tool, str) and tool.startswith("T") and tool[1:].isdigit():
        tool_idx = int(tool[1:])
    elif isinstance(tool, int):
        tool_idx = int(tool)
    else:
        tool_idx = ctx.state.tool if ctx else 0

    lines = [
        f";ACTION:TOOL_CHANGE T{tool_idx}",
        f"{tool_cmd} ; Select tool",
    ]
    if target_profile is not None:
        layer = ctx.state.layer if ctx is not None else 0
        temp = (
            target_profile.nozzle_temperature_initial_layer
            if layer == 0 and target_profile.nozzle_temperature_initial_layer is not None
            else target_profile.nozzle_temperature
        )
        m_code = "M109" if wait_for_temp else "M104"
        lines.append(f"{m_code} T{tool_idx} S{temp:.2f} ; Set incoming tool temperature")
        lines.append(
            f"M221 T{tool_idx} S{target_profile.flow_percent:.2f} ; Set incoming tool flow"
        )
    if reset_e:
        lines.append("G92 E0 ; Reset extruder datum after tool change")

    if ctx is not None:
        ctx.state.tool = tool_idx
        if target_profile is not None:
            ctx.filament = target_profile
        if reset_e:
            ctx.state.e = 0.0
    return lines


def prusa_xl_tool_change(
    ctx: PrintContext | None = None,
    *,
    tool: int | str,
    filament: FilamentProfile | None = None,
    wait_for_temp: bool = True,
) -> list[str]:
    """Tool-change sequence for multi-tool changers such as the Prusa XL.

    Retracts the outgoing tool if not already retracted, performs a Z lift to
    clear the part before docking, switches tools with ``P0``/``T{tool}``,
    waits for the incoming tool's target temperature, and resets ``E``.
    """
    lines: list[str] = []
    if ctx is not None and not ctx.state.is_retracted:
        lines.extend(standard_retract(ctx))
    if ctx is not None and ctx.state.has_moved_xy:
        lift_z = min(ctx.state.z + max(ctx.effective_hop(), 1.0), ctx.printer.max_z)
        lines.append(f"G0 F{ctx.printer.z_feedrate:.0f} Z{lift_z:.3f} ; Lift before tool change")
        ctx.state.z = lift_z
    lines.extend(
        standard_tool_change(
            ctx,
            tool=tool,
            filament=filament,
            wait_for_temp=wait_for_temp,
            reset_e=True,
        )
    )
    return lines


def continuous_fiber_cut(
    ctx: PrintContext | None = None,
    *,
    cut_command: str = "M280 P0 S90",
    release_command: str = "M280 P0 S0",
    dwell_ms: int = 250,
    pause_feed: bool = True,
) -> list[str]:
    """Actuate a continuous-fiber cutter mid-path or at the end of a path.

    In continuous-fiber printing, the cutter typically sits ``cut_offset_mm``
    upstream of the nozzle tip.  Insert this procedure at ``cut_offset_mm``
    before the end of a fiber path using
    :func:`~anisotropic_postprocessor.postprocessor.insert_at_distance` with
    ``from_end=True`` so the remaining tail is pulled cleanly out of the nozzle.
    """
    lines = [";ACTION:CUT_FIBER"]
    if pause_feed and dwell_ms > 0:
        lines.append("M400 ; Wait for current moves to finish before cutting")
    lines.append(f"{cut_command} ; Actuate fiber cutter")
    if dwell_ms > 0:
        lines.append(f"G4 P{dwell_ms} ; Hold cutter")
    if release_command:
        lines.append(f"{release_command} ; Release fiber cutter")
    return lines


# ── Explicit procedure registry ──────────────────────────────────────────────

PROCEDURES: dict[str, ProcedureFn] = {
    "raise3d_start": raise3d_start,
    "raise3d_purge": raise3d_purge,
    "raise3d_layer_change": raise3d_layer_change,
    "raise3d_end": raise3d_end,
    "generic_start": generic_start,
    "generic_purge": generic_purge,
    "generic_layer_change": generic_layer_change,
    "generic_end": generic_end,
    "standard_retract": standard_retract,
    "standard_unretract": standard_unretract,
    "standard_travel": standard_travel,
    "standard_tool_change": standard_tool_change,
    "prusa_xl_tool_change": prusa_xl_tool_change,
    "continuous_fiber_cut": continuous_fiber_cut,
}


def register_procedure(name: str, fn: ProcedureFn) -> None:
    """Register a Python procedure function so printer JSON files can name it."""
    if not name or not isinstance(name, str):
        raise ValueError(f"procedure name must be a non-empty string, got {name!r}")
    if not callable(fn):
        raise TypeError(f"procedure {name!r} must be callable, got {type(fn).__name__}")
    PROCEDURES[name] = fn


def resolve_procedure(spec: str | ProcedureFn | list[str] | None) -> ProcedureFn | None:
    """Resolve a procedure name, callable, or static G-code list into a callable."""
    if spec is None:
        return None
    if callable(spec):
        return spec
    if isinstance(spec, list):
        fixed_lines = [str(line) for line in spec]
        return lambda *_args, **_kwargs: list(fixed_lines)
    if isinstance(spec, str):
        if spec not in PROCEDURES:
            raise KeyError(
                f"Unknown procedure {spec!r}. Available procedures: "
                f"{sorted(PROCEDURES)}"
            )
        return PROCEDURES[spec]
    raise TypeError(
        f"Procedure specification must be a string name, callable, list[str], "
        f"or None; got {type(spec).__name__}"
    )
