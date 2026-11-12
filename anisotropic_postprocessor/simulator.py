"""G-code virtual machine parser and Siemens NX-style 3D machine simulator.

Rather than plotting input geometry before postprocessing, this module parses
the **actual generated G-code** line by line, simulating machine state
transitions (``G0``/``G1``, ``G28``, ``G90``/``G91``, ``M82``/``M83``, ``G92``,
tool changes ``Tn``, retractions, Z-hops, and continuous-fiber cuts).

:func:`plot_gcode` renders the resulting :class:`GcodeSimulation` as an
interactive 3D Plotly figure with:

* The printer build plate and coordinate triad ``(+X, +Y, +Z)``.
* Toggleable traces for extrusion beads, purge lines, rapid travels, Z-hops,
  retractions, tool changes, and continuous-fiber cut markers.
* Colouring by ``"feedrate"``, ``"tool"``, ``"layer"``, ``"z"``, ``"time"``, or
  ``"type"``.
* A virtual nozzle/toolhead indicator and an interactive simulation scrubbing
  slider (with Play/Pause controls) to step through the print chronologically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path as FilePath
from typing import Any, Literal

import numpy as onp

from anisotropic_postprocessor.profiles import PrinterProfile, load_printer

__all__ = [
    "GcodeMove",
    "GcodeEvent",
    "GcodeSimulation",
    "parse_gcode",
    "plot_gcode",
]

MoveType = Literal["extrude", "purge", "travel", "z_hop"]
EventType = Literal["retract", "unretract", "tool_change", "cut_fiber"]
ColorBy = Literal["feedrate", "tool", "layer", "z", "time", "type"]


# ── Simulation data structures ───────────────────────────────────────────────

@dataclass(frozen=True)
class GcodeMove:
    """One simulated spatial machine motion extracted from a ``G0``/``G1`` line."""

    step_index: int
    line_number: int
    raw_line: str
    move_type: MoveType
    feature_type: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    distance_mm: float
    delta_e: float
    feedrate: float
    tool: int
    layer: int
    duration_s: float
    cumulative_time_s: float


@dataclass(frozen=True)
class GcodeEvent:
    """A discrete machine action (retraction, tool change, fiber cut) in space."""

    line_number: int
    raw_line: str
    event_type: EventType
    position: tuple[float, float, float]
    tool: int
    layer: int
    cumulative_time_s: float
    details: str = ""


@dataclass
class GcodeSimulation:
    """Complete record of a simulated G-code program."""

    moves: list[GcodeMove] = field(default_factory=list)
    events: list[GcodeEvent] = field(default_factory=list)
    total_time_s: float = 0.0
    extruded_distance_mm: float = 0.0
    travel_distance_mm: float = 0.0
    filament_used_mm: dict[int, float] = field(default_factory=dict)
    num_layers: int = 0
    bounds_min: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_max: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def summary(self) -> str:
        """Human-readable verification summary of the simulated G-code."""
        cuts = sum(1 for e in self.events if e.event_type == "cut_fiber")
        retracts = sum(1 for e in self.events if e.event_type == "retract")
        tool_changes = sum(1 for e in self.events if e.event_type == "tool_change")
        tools_str = ", ".join(
            f"T{t}: {length:.1f} mm"
            for t, length in sorted(self.filament_used_mm.items())
        ) or "none"
        mins, secs = divmod(int(round(self.total_time_s)), 60)
        return (
            f"GcodeSimulation({len(self.moves)} moves, {self.num_layers} layers | "
            f"extrude {self.extruded_distance_mm:.1f} mm, "
            f"travel {self.travel_distance_mm:.1f} mm | "
            f"filament [{tools_str}] | "
            f"retracts: {retracts}, tool changes: {tool_changes}, fiber cuts: {cuts} | "
            f"est. time: {mins}m {secs:02d}s)"
        )


# ── G-code parser & virtual machine interpreter ──────────────────────────────

def _parse_words(cmd_part: str) -> dict[str, float]:
    """Parse ``X10.0 Y20.5 F1000`` parameter words into a dict."""
    words: dict[str, float] = {}
    for token in cmd_part.split()[1:]:
        if len(token) >= 2 and token[0].isalpha():
            key = token[0].upper()
            try:
                words[key] = float(token[1:])
            except ValueError:
                continue
    return words


def parse_gcode(gcode_or_path: str | FilePath) -> GcodeSimulation:
    """Simulate a G-code string or ``.gcode`` file and extract all moves and events.

    Tracks absolute/relative positioning (``G90``/``G91``), absolute/relative
    extrusion (``M82``/``M83``), coordinate resets (``G92``), tool selections
    (``Tn``), layer comments (``;LAYER:``), feature type comments (``;TYPE:``),
    and action comments (``;ACTION:``).
    """
    if isinstance(gcode_or_path, FilePath) or (
        isinstance(gcode_or_path, str)
        and "\n" not in gcode_or_path
        and gcode_or_path.strip().lower().endswith(".gcode")
        and FilePath(gcode_or_path).is_file()
    ):
        text = FilePath(gcode_or_path).read_text(encoding="utf-8")
    else:
        text = str(gcode_or_path)

    x = 0.0
    y = 0.0
    z = 0.0
    e = 0.0
    feedrate = 1000.0
    unit_scale = 1.0  # 1.0 for mm (G21), 25.4 for inches (G20)
    abs_xyz = True
    abs_e = True
    tool = 0
    layer = 0
    seen_layers: set[int] = set()
    feature_type = "PATH"
    pending_action: str | None = None
    in_ending = False

    moves: list[GcodeMove] = []
    events: list[GcodeEvent] = []
    filament_used: dict[int, float] = {}
    extruded_dist = 0.0
    travel_dist = 0.0
    cum_time = 0.0

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Parse structured comments
        if stripped.startswith(";"):
            upper = stripped.upper()
            if "START OF ENDING PROCEDURE" in upper:
                in_ending = True
            elif "END OF ENDING PROCEDURE" in upper:
                in_ending = False
            elif upper.startswith(";LAYER:"):
                try:
                    layer = int(stripped.split(":", 1)[1].strip())
                    seen_layers.add(layer)
                except ValueError:
                    pass
                feature_type = "PATH"
            elif upper.startswith(";TYPE:"):
                feature_type = stripped.split(":", 1)[1].strip().upper()
            elif upper.startswith(";ACTION:"):
                action_body = stripped.split(":", 1)[1].strip().upper()
                if action_body.startswith("CUT"):
                    events.append(
                        GcodeEvent(
                            line_number=line_no,
                            raw_line=stripped,
                            event_type="cut_fiber",
                            position=(x, y, z),
                            tool=tool,
                            layer=layer,
                            cumulative_time_s=cum_time,
                            details="Continuous fiber cut",
                        )
                    )
                else:
                    pending_action = action_body
            continue

        cmd_part = stripped.split(";", 1)[0].strip()
        if not cmd_part:
            continue
        tokens = cmd_part.split()
        opcode = tokens[0].upper()

        if opcode == "G21":
            unit_scale = 1.0
        elif opcode == "G20":
            unit_scale = 25.4
        elif opcode == "G90":
            abs_xyz = True
        elif opcode == "G91":
            abs_xyz = False
        elif opcode == "M82":
            abs_e = True
        elif opcode == "M83":
            abs_e = False
        elif opcode == "G92":
            words = _parse_words(cmd_part)
            if "X" in words:
                x = words["X"] * unit_scale
            if "Y" in words:
                y = words["Y"] * unit_scale
            if "Z" in words:
                z = words["Z"] * unit_scale
            if "E" in words:
                e = words["E"] * unit_scale
        elif opcode == "G28":
            words = _parse_words(cmd_part)
            if not words or "X" in words:
                x = 0.0
            if not words or "Y" in words:
                y = 0.0
            if not words or "Z" in words:
                z = 0.0
        elif opcode.startswith("T") and opcode[1:].isdigit():
            new_tool = int(opcode[1:])
            events.append(
                GcodeEvent(
                    line_number=line_no,
                    raw_line=stripped,
                    event_type="tool_change",
                    position=(x, y, z),
                    tool=new_tool,
                    layer=layer,
                    cumulative_time_s=cum_time,
                    details=f"T{tool} -> T{new_tool}",
                )
            )
            tool = new_tool
            pending_action = None
        elif opcode in ("G0", "G00", "G1", "G01"):
            words = _parse_words(cmd_part)
            if "F" in words and words["F"] > 0:
                feedrate = words["F"] * unit_scale

            nx, ny, nz = x, y, z
            if "X" in words:
                nx = (words["X"] * unit_scale) if abs_xyz else (x + words["X"] * unit_scale)
            if "Y" in words:
                ny = (words["Y"] * unit_scale) if abs_xyz else (y + words["Y"] * unit_scale)
            if "Z" in words:
                nz = (words["Z"] * unit_scale) if abs_xyz else (z + words["Z"] * unit_scale)

            delta_e = 0.0
            if "E" in words:
                target_e = words["E"] * unit_scale
                if abs_e:
                    delta_e = target_e - e
                    e = target_e
                else:
                    delta_e = target_e
                    e += delta_e

            dx = nx - x
            dy = ny - y
            dz = nz - z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Stationary filament move (retract / unretract / purge hover)
            if dist <= 1e-9:
                if abs(delta_e) > 1e-9:
                    dt = abs(delta_e) / max(feedrate / 60.0, 1e-6)
                    cum_time += dt
                    if delta_e < 0:
                        events.append(
                            GcodeEvent(
                                line_number=line_no,
                                raw_line=stripped,
                                event_type="retract",
                                position=(x, y, z),
                                tool=tool,
                                layer=layer,
                                cumulative_time_s=cum_time,
                                details=f"dE={delta_e:.3f} mm",
                            )
                        )
                    elif pending_action == "UNRETRACT" or feature_type != "PURGE":
                        events.append(
                            GcodeEvent(
                                line_number=line_no,
                                raw_line=stripped,
                                event_type="unretract",
                                position=(x, y, z),
                                tool=tool,
                                layer=layer,
                                cumulative_time_s=cum_time,
                                details=f"dE=+{delta_e:.3f} mm",
                            )
                        )
                    else:
                        filament_used[tool] = filament_used.get(tool, 0.0) + delta_e
                pending_action = None
                continue

            # Spatial move (dist > 0)
            dt = dist / max(feedrate / 60.0, 1e-6)
            cum_time += dt

            if in_ending:
                x, y, z = nx, ny, nz
                pending_action = None
                continue

            if delta_e > 1e-9:
                m_type: MoveType = "purge" if feature_type == "PURGE" else "extrude"
                if m_type == "extrude":
                    extruded_dist += dist
                filament_used[tool] = filament_used.get(tool, 0.0) + delta_e
                seen_layers.add(layer)
            else:
                xy_dist = math.hypot(dx, dy)
                if feature_type == "Z_HOP" or (xy_dist <= 1e-9 and abs(dz) > 1e-9):
                    m_type = "z_hop"
                else:
                    m_type = "travel"
                travel_dist += dist

            moves.append(
                GcodeMove(
                    step_index=len(moves),
                    line_number=line_no,
                    raw_line=stripped,
                    move_type=m_type,
                    feature_type=feature_type,
                    start=(x, y, z),
                    end=(nx, ny, nz),
                    distance_mm=dist,
                    delta_e=delta_e,
                    feedrate=feedrate,
                    tool=tool,
                    layer=layer,
                    duration_s=dt,
                    cumulative_time_s=cum_time,
                )
            )
            x, y, z = nx, ny, nz
            pending_action = None

    # Compute bounding box over extrusion moves (or all moves if none extrude)
    box_moves = [m for m in moves if m.move_type == "extrude"] or moves
    if box_moves:
        pts = onp.array([m.start for m in box_moves] + [m.end for m in box_moves])
        b_min = (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 2].min()))
        b_max = (float(pts[:, 0].max()), float(pts[:, 1].max()), float(pts[:, 2].max()))
    else:
        b_min = (0.0, 0.0, 0.0)
        b_max = (0.0, 0.0, 0.0)

    return GcodeSimulation(
        moves=moves,
        events=events,
        total_time_s=cum_time,
        extruded_distance_mm=extruded_dist,
        travel_distance_mm=travel_dist,
        filament_used_mm=filament_used,
        num_layers=len(seen_layers),
        bounds_min=b_min,
        bounds_max=b_max,
    )


# ── Interactive 3D machine simulation viewer ─────────────────────────────────

def _segments_to_polyline_arrays(
    moves: list[GcodeMove],
    metric_fn: Any,
) -> tuple[list[float | None], list[float | None], list[float | None], list[float], list[str]]:
    """Pack a list of moves into NaN/None-separated 3D polyline arrays for Plotly."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    vals: list[float] = []
    hovers: list[str] = []

    for i, m in enumerate(moves):
        if i == 0 or m.start != moves[i - 1].end:
            if xs:
                xs.append(None)
                ys.append(None)
                zs.append(None)
                vals.append(float(metric_fn(m)))
                hovers.append("")
            xs.append(m.start[0])
            ys.append(m.start[1])
            zs.append(m.start[2])
            vals.append(float(metric_fn(m)))
            hovers.append(
                f"Line {m.line_number}: {m.raw_line}<br>"
                f"Start: ({m.start[0]:.2f}, {m.start[1]:.2f}, {m.start[2]:.2f}) mm"
            )
        xs.append(m.end[0])
        ys.append(m.end[1])
        zs.append(m.end[2])
        vals.append(float(metric_fn(m)))
        hovers.append(
            f"Line {m.line_number}: {m.raw_line}<br>"
            f"End: ({m.end[0]:.2f}, {m.end[1]:.2f}, {m.end[2]:.2f}) mm<br>"
            f"F: {m.feedrate:.0f} mm/min | ΔE: {m.delta_e:.4f} mm<br>"
            f"Tool: T{m.tool} | Layer: {m.layer} | t: {m.cumulative_time_s:.1f} s"
        )
    return xs, ys, zs, vals, hovers


def plot_gcode(
    gcode_or_sim: str | FilePath | GcodeSimulation,
    *,
    printer: PrinterProfile | str | FilePath | None = None,
    color_by: ColorBy = "feedrate",
    show_travel: bool = True,
    show_z_hop: bool = True,
    show_purge: bool = True,
    show_events: bool = True,
    show_bed: bool = True,
    slider: bool = True,
    slider_steps: int = 25,
    line_width: float = 4.0,
    title: str | None = None,
) -> Any:
    """Render an interactive 3D Siemens NX-style machine simulation in Plotly.

    Parameters
    ----------
    gcode_or_sim : str, Path, or GcodeSimulation
        Produced G-code string, path to a ``.gcode`` file, or an already-parsed
        :class:`GcodeSimulation`.
    printer : PrinterProfile, str, or Path, optional
        Printer profile used to draw the build plate envelope.
    color_by : {"feedrate", "tool", "layer", "z", "time", "type"}
        Property used to colour the extruded toolpath.
    show_travel, show_z_hop, show_purge, show_events, show_bed : bool
        Toggle visibility of rapid travels, Z-hops, purge lines, discrete
        machine events (retractions, tool changes, fiber cuts), and the build
        plate outline.
    slider : bool
        Add an interactive step-scrubbing slider and virtual nozzle marker to
        inspect the toolpath buildup chronologically.
    slider_steps : int
        Number of scrub positions along the slider (default ``25``).
    line_width : float
        Width of extruded toolpath lines in px.
    title : str, optional
        Custom figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D figure (renders inline in Google Colab and Jupyter).
    """
    import plotly.graph_objects as go

    sim = (
        gcode_or_sim
        if isinstance(gcode_or_sim, GcodeSimulation)
        else parse_gcode(gcode_or_sim)
    )
    printer_prof = load_printer(printer) if printer is not None else None

    fig = go.Figure()

    # 1. Build plate outline and coordinate triad
    if show_bed:
        if printer_prof is not None:
            bx0, by0 = printer_prof.bed_origin
            bx1 = bx0 + printer_prof.bed_size[0]
            by1 = by0 + printer_prof.bed_size[1]
        else:
            pad = 10.0
            bx0 = min(0.0, sim.bounds_min[0] - pad)
            by0 = min(0.0, sim.bounds_min[1] - pad)
            bx1 = max(200.0, sim.bounds_max[0] + pad)
            by1 = max(200.0, sim.bounds_max[1] + pad)

        fig.add_trace(
            go.Scatter3d(
                x=[bx0, bx1, bx1, bx0, bx0],
                y=[by0, by0, by1, by1, by0],
                z=[0.0, 0.0, 0.0, 0.0, 0.0],
                mode="lines",
                name="Build Plate",
                line={"color": "#64748b", "width": 2},
                hoverinfo="skip",
            )
        )
        triad_len = max(15.0, 0.06 * max(bx1 - bx0, by1 - by0))
        for axis_name, ax_end, col in (
            ("+X", (bx0 + triad_len, by0, 0.0), "#ef4444"),
            ("+Y", (bx0, by0 + triad_len, 0.0), "#22c55e"),
            ("+Z", (bx0, by0, triad_len), "#3b82f6"),
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=[bx0, ax_end[0]],
                    y=[by0, ax_end[1]],
                    z=[0.0, ax_end[2]],
                    mode="lines",
                    name=f"Axis {axis_name}",
                    showlegend=False,
                    line={"color": col, "width": 4},
                    hovertemplate=f"Machine {axis_name}<extra></extra>",
                )
            )

    # 2. Extrusion moves
    extrude_moves = [m for m in sim.moves if m.move_type == "extrude"]
    metric_map: dict[str, tuple[Any, str, str]] = {
        "feedrate": (lambda m: m.feedrate, "Feedrate (mm/min)", "Plasma"),
        "tool": (lambda m: float(m.tool), "Tool Index", "Turbo"),
        "layer": (lambda m: float(m.layer), "Layer", "Viridis"),
        "z": (lambda m: m.end[2], "Z Height (mm)", "Viridis"),
        "time": (lambda m: m.cumulative_time_s, "Time (s)", "Cividis"),
        "type": (
            lambda m: float(hash(m.feature_type) % 16),
            "Feature Type",
            "Turbo",
        ),
    }
    if color_by not in metric_map:
        raise ValueError(
            f"color_by must be one of {sorted(metric_map)}, got {color_by!r}"
        )
    metric_fn, cbar_title, colorscale = metric_map[color_by]

    extrude_trace_idx: int | None = None
    if extrude_moves:
        ex_x, ex_y, ex_z, ex_val, ex_hover = _segments_to_polyline_arrays(
            extrude_moves, metric_fn
        )
        extrude_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=ex_x,
                y=ex_y,
                z=ex_z,
                mode="lines",
                name="Extrusion Bead",
                text=ex_hover,
                hovertemplate="%{text}<extra>Extrusion</extra>",
                line={
                    "width": line_width,
                    "color": ex_val,
                    "colorscale": colorscale,
                    "colorbar": {
                        "title": cbar_title,
                        "thickness": 14,
                        "len": 0.65,
                    },
                },
            )
        )

    # 3. Purge moves
    purge_moves = [m for m in sim.moves if m.move_type == "purge"]
    if show_purge and purge_moves:
        px, py, pz, _, phover = _segments_to_polyline_arrays(
            purge_moves, lambda m: 0.0
        )
        fig.add_trace(
            go.Scatter3d(
                x=px,
                y=py,
                z=pz,
                mode="lines",
                name="Purge Line",
                text=phover,
                hovertemplate="%{text}<extra>Purge</extra>",
                line={"color": "#a855f7", "width": 3},
            )
        )

    # 4. Travel moves
    travel_moves = [m for m in sim.moves if m.move_type == "travel"]
    travel_trace_idx: int | None = None
    if show_travel and travel_moves:
        tx, ty, tz, _, thover = _segments_to_polyline_arrays(
            travel_moves, lambda m: 0.0
        )
        travel_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=tx,
                y=ty,
                z=tz,
                mode="lines",
                name="Rapid Travel (G0)",
                text=thover,
                hovertemplate="%{text}<extra>Travel</extra>",
                line={"color": "#94a3b8", "width": 1.5, "dash": "dash"},
            )
        )

    # 5. Z-hop moves
    hop_moves = [m for m in sim.moves if m.move_type == "z_hop"]
    if show_z_hop and hop_moves:
        hx, hy, hz, _, hhover = _segments_to_polyline_arrays(
            hop_moves, lambda m: 0.0
        )
        fig.add_trace(
            go.Scatter3d(
                x=hx,
                y=hy,
                z=hz,
                mode="lines",
                name="Z-Hop",
                text=hhover,
                hovertemplate="%{text}<extra>Z-Hop</extra>",
                line={"color": "#38bdf8", "width": 1.5, "dash": "dot"},
            )
        )

    # 6. Discrete machine event markers (fiber cuts, retractions, tool changes)
    if show_events and sim.events:
        event_styles: list[tuple[EventType, str, str, str, int]] = [
            ("cut_fiber", "Fiber Cut", "#ef4444", "diamond", 6),
            ("tool_change", "Tool Change", "#eab308", "square", 5),
            ("retract", "Retraction", "#f97316", "circle", 3),
        ]
        for ev_type, label, col, symbol, sz in event_styles:
            ev_group = [e for e in sim.events if e.event_type == ev_type]
            if not ev_group:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=[e.position[0] for e in ev_group],
                    y=[e.position[1] for e in ev_group],
                    z=[e.position[2] for e in ev_group],
                    mode="markers",
                    name=label,
                    text=[
                        f"Line {e.line_number}: {e.raw_line}<br>"
                        f"Pos: ({e.position[0]:.2f}, {e.position[1]:.2f}, "
                        f"{e.position[2]:.2f}) mm<br>"
                        f"Tool: T{e.tool} | Layer: {e.layer} | {e.details}"
                        for e in ev_group
                    ],
                    hovertemplate=f"%{{text}}<extra>{label}</extra>",
                    marker={"color": col, "symbol": symbol, "size": sz},
                )
            )

    # 7. Virtual Nozzle marker + interactive simulation slider
    nozzle_trace_idx: int | None = None
    if extrude_moves:
        last_pt = extrude_moves[-1].end
        nozzle_trace_idx = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=[last_pt[0]],
                y=[last_pt[1]],
                z=[last_pt[2]],
                mode="markers",
                name="Nozzle Tip",
                marker={"color": "#dc2626", "size": 6, "symbol": "diamond"},
                hovertemplate="Active Nozzle Position<extra></extra>",
            )
        )

    if slider and extrude_moves and extrude_trace_idx is not None:
        n_ext = len(extrude_moves)
        n_steps = max(2, min(slider_steps, n_ext))
        step_indices = [
            max(1, int(round(k * n_ext / n_steps))) for k in range(1, n_steps + 1)
        ]
        slider_items = []
        for count in step_indices:
            sub_ext = extrude_moves[:count]
            sx, sy, sz, sval, shover = _segments_to_polyline_arrays(sub_ext, metric_fn)
            cur_move = sub_ext[-1]
            restyle_x: list[Any] = [sx]
            restyle_y: list[Any] = [sy]
            restyle_z: list[Any] = [sz]
            target_traces: list[int] = [extrude_trace_idx]

            if travel_trace_idx is not None:
                sub_trav = [
                    m for m in travel_moves if m.step_index <= cur_move.step_index
                ]
                if sub_trav:
                    tx_s, ty_s, tz_s, _, _ = _segments_to_polyline_arrays(
                        sub_trav, lambda m: 0.0
                    )
                else:
                    tx_s, ty_s, tz_s = [], [], []
                restyle_x.append(tx_s)
                restyle_y.append(ty_s)
                restyle_z.append(tz_s)
                target_traces.append(travel_trace_idx)

            if nozzle_trace_idx is not None:
                restyle_x.append([cur_move.end[0]])
                restyle_y.append([cur_move.end[1]])
                restyle_z.append([cur_move.end[2]])
                target_traces.append(nozzle_trace_idx)

            pct = int(round(100.0 * count / n_ext))
            slider_items.append(
                {
                    "method": "restyle",
                    "label": f"{pct}% (L{cur_move.layer})",
                    "args": [
                        {"x": restyle_x, "y": restyle_y, "z": restyle_z},
                        target_traces,
                    ],
                }
            )

        fig.update_layout(
            sliders=[
                {
                    "active": len(slider_items) - 1,
                    "currentvalue": {"prefix": "Simulation Progress: "},
                    "pad": {"t": 30},
                    "steps": slider_items,
                }
            ]
        )

    fig_title = title or sim.summary()
    fig.update_layout(
        title={"text": fig_title, "font": {"size": 13}},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        legend={"x": 0.01, "y": 0.98, "bgcolor": "rgba(255,255,255,0.8)"},
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
            "aspectmode": "data",
        },
    )
    return fig
