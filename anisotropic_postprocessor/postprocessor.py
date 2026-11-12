"""Core G-code postprocessor for 2D, 2.5D, and 3D anisotropic toolpaths.

Accepts paths produced by ``path_optimizer`` (either ``path_optimizer.paths.Path``
objects or bare ``(n, 2)`` / ``(n, 3)`` coordinate arrays in SI meters) and
combines them with a :class:`~anisotropic_postprocessor.profiles.PrinterProfile`,
one or more :class:`~anisotropic_postprocessor.profiles.FilamentProfile`
configs, and Python procedure functions to emit complete, printable G-code.

Three dimensionality modes are supported via ``mode``:

* ``"2d"`` — a single 2D layer pattern (``list[Path]``), repeated ``n_layers``
  times from ``z0`` at spacing ``layer_height``.
* ``"2.5d"`` — multiple stacked 2D layers (``list[list[Path]]``), where each
  layer has its own distinct paths (for example, differently oriented plies in
  a composite laminate).
* ``"3d"`` — full 3D paths where every vertex is ``(x, y, z)`` in space.
"""
from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path as FilePath
from typing import Any, Literal

import numpy as onp

from anisotropic_postprocessor.procedures import (
     PrintContext,
    PrintState,
    ProcedureFn,
    resolve_procedure,
    standard_travel,
)
from anisotropic_postprocessor.profiles import (
    FilamentProfile,
    PrinterProfile,
    load_filament,
    load_printer,
    validate_compatibility,
)

__all__ = [
    "PathEvent",
    "ProcedureCall",
    "PrintPath",
    "insert_at_index",
    "insert_at_distance",
    "add_fiber_cuts",
    "bed_offset",
    "extrusion_length",
    "path_gcode",
    "generate_gcode",
    "generate_gcode_layers",
    "write_gcode",
]

Mode = Literal["2d", "2.5d", "3d"]


# ── Path & event types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathEvent:
    """A procedure scheduled to fire when a path reaches ``vertex_index``.

    ``vertex_index`` is 0-based into :attr:`PrintPath.nodes`.  When ``0``, the
    procedure runs right after arriving at the start of the path; when
    ``len(nodes) - 1``, it runs right after the final segment of the path; and
    for interior vertices it runs between the incoming segment and the outgoing
    segment at that vertex.
    """

    vertex_index: int
    procedure: str | ProcedureFn | list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcedureCall:
    """A standalone procedure invocation placed between paths in a sequence.

    Insert a :class:`ProcedureCall` directly into any path list to trigger a
    procedure (such as a custom purge, pause, or fiber cut) between two paths.
    """

    procedure: str | ProcedureFn | list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        procedure: str | ProcedureFn | list[str],
        **kwargs: Any,
    ) -> None:
        object.__setattr__(self, "procedure", procedure)
        object.__setattr__(self, "kwargs", dict(kwargs))


@dataclass
class PrintPath:
    """One continuous extrusion (or travel) polyline with optional mid-path events.

    Compatible with ``path_optimizer.paths.Path`` by duck typing (``.nodes``,
    ``.kind``, ``.path_id``), while also supporting 3D ``(n, 3)`` polylines,
    per-path extrusion overrides, and mid-path :class:`PathEvent` callbacks.
    """

    nodes: onp.ndarray
    kind: str = "path"
    path_id: int | None = None
    width: float | None = None
    height: float | None = None
    speed: float | None = None
    extrude: bool = True
    travel: bool = True
    label: str | None = None
    events: tuple[PathEvent, ...] = ()

    def __post_init__(self) -> None:
        self.nodes = onp.asarray(self.nodes, dtype=float)
        if self.nodes.ndim != 2 or self.nodes.shape[1] not in (2, 3):
            raise ValueError(
                f"nodes must have shape (n, 2) or (n, 3), got {self.nodes.shape}"
            )
        if len(self.nodes) < 2:
            raise ValueError("a path needs at least 2 nodes")
        self.events = tuple(self.events)
        for ev in self.events:
            if not (0 <= ev.vertex_index < len(self.nodes)):
                raise ValueError(
                    f"PathEvent vertex_index {ev.vertex_index} out of bounds "
                    f"for path with {len(self.nodes)} nodes"
                )

    @property
    def ndim(self) -> int:
        """Spatial dimension of the path points (``2`` or ``3``)."""
        return int(self.nodes.shape[1])

    @property
    def start(self) -> onp.ndarray:
        return self.nodes[0]

    @property
    def end(self) -> onp.ndarray:
        return self.nodes[-1]

    @property
    def length(self) -> float:
        return float(onp.linalg.norm(onp.diff(self.nodes, axis=0), axis=1).sum())

    @property
    def is_closed(self) -> bool:
        return bool(
            onp.linalg.norm(self.end - self.start) <= 1e-3 * max(self.length, 1e-30)
        )

    def reversed(self) -> PrintPath:
        """The same path walked in reverse order, flipping event indices."""
        n = len(self.nodes)
        rev_events = tuple(
            PathEvent(n - 1 - ev.vertex_index, ev.procedure, dict(ev.kwargs))
            for ev in reversed(self.events)
        )
        return PrintPath(
            nodes=self.nodes[::-1].copy(),
            kind=self.kind,
            path_id=self.path_id,
            width=self.width,
            height=self.height,
            speed=self.speed,
            extrude=self.extrude,
            travel=self.travel,
            label=self.label,
            events=rev_events,
        )


def _as_print_path(item: Any, default_id: int | None = None) -> PrintPath:
    """Convert a ``path_optimizer.paths.Path``, ``PrintPath``, or array to ``PrintPath``."""
    if isinstance(item, PrintPath):
        if item.path_id is None and default_id is not None:
            return PrintPath(
                nodes=item.nodes,
                kind=item.kind,
                path_id=default_id,
                width=item.width,
                height=item.height,
                speed=item.speed,
                extrude=item.extrude,
                travel=item.travel,
                label=item.label,
                events=item.events,
            )
        return item
    if hasattr(item, "nodes"):
        return PrintPath(
            nodes=onp.asarray(item.nodes, dtype=float),
            kind=getattr(item, "kind", "path"),
            path_id=getattr(item, "path_id", default_id),
        )
    return PrintPath(nodes=onp.asarray(item, dtype=float), path_id=default_id)


def _normalize_sequence(
    items: Sequence[Any],
) -> list[PrintPath | ProcedureCall]:
    """Convert a list of paths and ProcedureCalls into normalized objects."""
    out: list[PrintPath | ProcedureCall] = []
    path_counter = 0
    for item in items:
        if isinstance(item, ProcedureCall):
            out.append(item)
        else:
            out.append(_as_print_path(item, default_id=path_counter))
            path_counter += 1
    return out


# ── Programmatic mid-path procedure insertion ────────────────────────────────

def insert_at_index(
    path: Any,
    index: int,
    procedure: str | ProcedureFn | list[str],
    **kwargs: Any,
) -> PrintPath:
    """Attach ``procedure`` to vertex ``index`` of ``path``.

    Negative indices count from the end of the path (``-1`` is the final
    vertex).  Returns a new :class:`PrintPath` without mutating the input.
    """
    p = _as_print_path(path)
    n = len(p.nodes)
    idx = index if index >= 0 else n + index
    if not (0 <= idx < n):
        raise IndexError(
            f"vertex index {index} (resolved to {idx}) out of range for path "
            f"with {n} vertices"
        )
    new_event = PathEvent(vertex_index=idx, procedure=procedure, kwargs=dict(kwargs))
    return PrintPath(
        nodes=p.nodes.copy(),
        kind=p.kind,
        path_id=p.path_id,
        width=p.width,
        height=p.height,
        speed=p.speed,
        extrude=p.extrude,
        travel=p.travel,
        label=p.label,
        events=(*p.events, new_event),
    )


def insert_at_distance(
    path: Any,
    distance: float | None = None,
    procedure: str | ProcedureFn | list[str] = "continuous_fiber_cut",
    *,
    distance_mm: float | None = None,
    from_end: bool = False,
    scale: float = 1000.0,
    **kwargs: Any,
) -> PrintPath:
    """Insert ``procedure`` at a given arc-length along ``path``.

    If the target arc-length falls inside a segment rather than on an existing
    vertex, that segment is split by linear interpolation so the procedure
    fires at the exact geometric position along the path.

    Parameters
    ----------
    path : Path, PrintPath, or array
        The input path.
    distance : float, optional
        Distance along the path **in the path's own units** (e.g. meters when
        coming from ``path_optimizer``).  Provide either ``distance`` or
        ``distance_mm``.
    procedure : str, callable, or list[str]
        The procedure to trigger at that point (default
        ``"continuous_fiber_cut"``).
    distance_mm : float, optional
        Distance along the path in printer millimeters, converted to path units
        by dividing by ``scale``.
    from_end : bool
        If ``True``, measure the distance backwards from the end of the path
        (for example, cutting continuous fiber ``15.0`` mm before the path
        ends).
    scale : float
        Coordinate scale from path units to millimeters (default ``1000.0``).
    **kwargs
        Keyword arguments forwarded to ``procedure`` when it executes.
    """
    if (distance is None) == (distance_mm is None):
        raise ValueError("Provide exactly one of `distance` or `distance_mm`.")
    dist_path_units = (
        float(distance)
        if distance is not None
        else float(distance_mm) / float(scale)  # type: ignore[arg-type]
    )
    if dist_path_units < 0:
        raise ValueError(f"distance must be non-negative, got {dist_path_units}")

    p = _as_print_path(path)
    nodes = p.nodes.copy()
    seg_vecs = onp.diff(nodes, axis=0)
    seg_lengths = onp.linalg.norm(seg_vecs, axis=1)
    cum_lengths = onp.concatenate([[0.0], onp.cumsum(seg_lengths)])
    total_len = float(cum_lengths[-1])

    target = (total_len - dist_path_units) if from_end else dist_path_units
    if target <= 0.0:
        return insert_at_index(p, 0, procedure, **kwargs)
    if target >= total_len:
        return insert_at_index(p, len(nodes) - 1, procedure, **kwargs)

    tol = 1e-9 * max(total_len, 1.0)
    diffs = onp.abs(cum_lengths - target)
    nearest_idx = int(onp.argmin(diffs))
    if diffs[nearest_idx] <= tol:
        return insert_at_index(p, nearest_idx, procedure, **kwargs)

    # Interpolate a new vertex on segment [k, k + 1]
    k = int(onp.searchsorted(cum_lengths, target, side="right") - 1)
    k = max(0, min(k, len(nodes) - 2))
    seg_len = float(seg_lengths[k])
    alpha = (target - float(cum_lengths[k])) / seg_len if seg_len > 0 else 0.0
    new_pt = nodes[k] + alpha * (nodes[k + 1] - nodes[k])

    new_nodes = onp.vstack([nodes[: k + 1], new_pt[None, :], nodes[k + 1 :]])
    shifted_events: list[PathEvent] = []
    for ev in p.events:
        if ev.vertex_index <= k:
            shifted_events.append(ev)
        else:
            shifted_events.append(
                PathEvent(ev.vertex_index + 1, ev.procedure, dict(ev.kwargs))
            )
    shifted_events.append(
        PathEvent(vertex_index=k + 1, procedure=procedure, kwargs=dict(kwargs))
    )

    return PrintPath(
        nodes=new_nodes,
        kind=p.kind,
        path_id=p.path_id,
        width=p.width,
        height=p.height,
        speed=p.speed,
        extrude=p.extrude,
        travel=p.travel,
        label=p.label,
        events=tuple(shifted_events),
    )


def add_fiber_cuts(
    paths: Sequence[Any],
    *,
    cut_offset_mm: float = 15.0,
    procedure: str | ProcedureFn | list[str] = "continuous_fiber_cut",
    kind: str | None = "fibre",
    scale: float = 1000.0,
    **kwargs: Any,
) -> list[PrintPath | ProcedureCall]:
    """Attach a continuous-fiber cut ``cut_offset_mm`` before the end of matching paths.

    Parameters
    ----------
    paths : sequence of Path or PrintPath
        Input paths for one layer.
    cut_offset_mm : float
        Distance from the fiber cutter to the nozzle tip, in mm.
    procedure : str or callable
        Cut procedure to invoke (default ``"continuous_fiber_cut"``).
    kind : str or None
        Only attach cuts to paths whose ``.kind == kind``.  Pass ``None`` to
        attach to every path in ``paths``.
    scale : float
        Coordinate scale from path units to mm (default ``1000.0``).
    """
    out: list[PrintPath | ProcedureCall] = []
    for item in _normalize_sequence(paths):
        if isinstance(item, ProcedureCall):
            out.append(item)
        elif kind is None or item.kind == kind:
            out.append(
                insert_at_distance(
                    item,
                    procedure=procedure,
                    distance_mm=cut_offset_mm,
                    from_end=True,
                    scale=scale,
                    **kwargs,
                )
            )
        else:
            out.append(item)
    return out


# ── Bed centering and extrusion math ─────────────────────────────────────────

def bed_offset(
    paths: Sequence[Any],
    *,
    bed_center: tuple[float, float],
    scale: float = 1000.0,
) -> tuple[float, float]:
    """Return ``(dx, dy)`` in mm that centers the XY bounding box on ``bed_center``."""
    arrays: list[onp.ndarray] = []
    for item in paths:
        if isinstance(item, ProcedureCall):
            continue
        p = _as_print_path(item)
        arrays.append(p.nodes[:, :2])
    if not arrays:
        raise ValueError("cannot center an empty design with no paths")
    pts = onp.vstack(arrays) * float(scale)
    mid = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    return (float(bed_center[0] - mid[0]), float(bed_center[1] - mid[1]))


def extrusion_length(
    distance_mm: float,
    *,
    width_mm: float,
    height_mm: float,
    filament_diameter_mm: float = 1.75,
    cross_section: str = "rectangle",
) -> float:
    """Filament length ``ΔE`` (in mm) needed to lay down a segment of ``distance_mm``.

    Parameters
    ----------
    distance_mm : float
        Length of the nozzle move in mm.
    width_mm, height_mm : float
        Bead cross-section dimensions in mm.
    filament_diameter_mm : float
        Feedstock filament diameter in mm (typically ``1.75`` or ``2.85``).
    cross_section : {"rectangle", "stadium"}
        ``"rectangle"`` uses ``width * height`` (matching FullControl);
        ``"stadium"`` uses ``(width - height) * height + pi * (height / 2)**2``
        (matching PrusaSlicer / Slic3r).
    """
    if distance_mm <= 0.0:
        return 0.0
    if cross_section == "rectangle":
        bead_area = float(width_mm) * float(height_mm)
    elif cross_section == "stadium":
        bead_area = (float(width_mm) - float(height_mm)) * float(
            height_mm
        ) + math.pi * (0.5 * float(height_mm)) ** 2
    else:
        raise ValueError(
            f"cross_section must be 'rectangle' or 'stadium', got {cross_section!r}"
        )
    filament_area = math.pi * (0.5 * float(filament_diameter_mm)) ** 2
    return (bead_area * float(distance_mm)) / filament_area


# ── Single-path G-code emission ──────────────────────────────────────────────

def _run_procedure(
    spec: str | ProcedureFn | list[str] | None,
    ctx: PrintContext,
    **kwargs: Any,
) -> list[str]:
    """Execute a procedure specification with ``ctx`` and return its G-code lines."""
    fn = resolve_procedure(spec)
    if fn is None:
        return []
    try:
        return list(fn(ctx, **kwargs))
    except TypeError:
        # Allow simple user functions with signature `fn(**kwargs) -> list[str]`
        return list(fn(**kwargs))


def path_gcode(
    path: Any,
    ctx: PrintContext,
    *,
    z: float | None = None,
    scale: float | None = None,
    offset: tuple[float, float] | None = None,
    annotate: bool = True,
    before_path: str | ProcedureFn | list[str] | None = None,
    after_path: str | ProcedureFn | list[str] | None = None,
) -> list[str]:
    """Emit G-code lines for a single :class:`PrintPath`.

    Parameters
    ----------
    path : Path, PrintPath, or array
        The continuous path to print.
    ctx : PrintContext
        Active printer/filament/state context.
    z : float, optional
        Layer height in mm for 2D paths.  Ignored if ``path`` already has 3D
        coordinates ``(n, 3)``.
    scale : float, optional
        Coordinate multiplier from path units to printer units (defaults to
        ``ctx.printer.coordinate_scale``).
    offset : (float, float), optional
        ``(dx, dy)`` bed shift in printer units (mm), added after scaling.
    annotate : bool
        Emit a leading comment naming the path's kind and ID.
    before_path, after_path : procedure spec, optional
        Procedures to invoke immediately before or after this path.
    """
    p = _as_print_path(path)
    coord_scale = float(scale if scale is not None else ctx.printer.coordinate_scale)

    raw_pts = p.nodes * coord_scale
    if p.ndim == 2:
        layer_z = float(z if z is not None else ctx.state.layer_z)
        pts_3d = onp.column_stack(
            [raw_pts[:, 0], raw_pts[:, 1], onp.full(len(raw_pts), layer_z)]
        )
    else:
        pts_3d = raw_pts.copy()

    if offset is not None:
        pts_3d[:, 0] += float(offset[0])
        pts_3d[:, 1] += float(offset[1])

    # Check printer build volume
    min_xyz = pts_3d.min(axis=0)
    max_xyz = pts_3d.max(axis=0)
    bx0, by0 = ctx.printer.bed_origin
    bx1, by1 = bx0 + ctx.printer.bed_size[0], by0 + ctx.printer.bed_size[1]
    if (
        min_xyz[0] < bx0 - 1e-6
        or max_xyz[0] > bx1 + 1e-6
        or min_xyz[1] < by0 - 1e-6
        or max_xyz[1] > by1 + 1e-6
    ):
        raise ValueError(
            f"Path coordinates X[{min_xyz[0]:.2f}, {max_xyz[0]:.2f}], "
            f"Y[{min_xyz[1]:.2f}, {max_xyz[1]:.2f}] exceed printer "
            f"{ctx.printer.name!r} bed bounds X[{bx0:.1f}, {bx1:.1f}], "
            f"Y[{by0:.1f}, {by1:.1f}] (try bed_center=True or check coordinate_scale)."
        )
    if min_xyz[2] < -1e-6 or max_xyz[2] > ctx.printer.max_z + 1e-6:
        raise ValueError(
            f"Path Z[{min_xyz[2]:.2f}, {max_xyz[2]:.2f}] exceeds printer "
            f"{ctx.printer.name!r} Z bounds [0.0, {ctx.printer.max_z:.1f}]."
        )

    lines: list[str] = []
    if annotate:
        comment = (
            p.label
            if p.label is not None
            else (
                f"{p.kind} path {p.path_id if p.path_id is not None else 0}"
                f"{' (closed)' if p.is_closed else ''}"
            )
        )
        lines.append(f"; {comment}")

    if before_path is not None:
        lines.extend(_run_procedure(before_path, ctx))

    # Group mid-path events by vertex index
    events_by_vertex: dict[int, list[PathEvent]] = {}
    for ev in p.events:
        events_by_vertex.setdefault(ev.vertex_index, []).append(ev)

    start_x, start_y, start_z = (
        float(pts_3d[0, 0]),
        float(pts_3d[0, 1]),
        float(pts_3d[0, 2]),
    )
    if p.travel:
        travel_spec = ctx.printer.procedures.get("travel", standard_travel)
        lines.extend(
            _run_procedure(
                travel_spec,
                ctx,
                x=start_x,
                y=start_y,
                z=start_z,
                retract=ctx.retract,
                hop=ctx.hop,
            )
        )
    else:
        ctx.state.x, ctx.state.y, ctx.state.z = start_x, start_y, start_z
        ctx.state.has_moved_xy = True

    # Fire any events attached to vertex 0 (after arrival, before extruding)
    for ev in events_by_vertex.get(0, ()):
        lines.extend(_run_procedure(ev.procedure, ctx, **ev.kwargs))

    width_mm = float(p.width if p.width is not None else ctx.width)
    height_mm = float(p.height if p.height is not None else ctx.layer_height)
    feed_mm_min = float(
        p.speed if p.speed is not None else ctx.effective_print_speed()
    )

    lines.append(f";TYPE:WALL-OUTER" if p.kind == "outline" else f";TYPE:{p.kind.upper()}")

    for i in range(1, len(pts_3d)):
        x0, y0, z0 = pts_3d[i - 1]
        x1, y1, z1 = pts_3d[i]
        seg_dist = float(math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2))

        if p.extrude and seg_dist > 0.0:
            d_e = extrusion_length(
                seg_dist,
                width_mm=width_mm,
                height_mm=height_mm,
                filament_diameter_mm=ctx.filament.filament_diameter,
                cross_section=ctx.printer.extrusion_cross_section,
            )
            ctx.state.record_extrusion(d_e)
            e_word = f" E{ctx.state.e:.5f}"
            move_code = "G1"
        else:
            e_word = ""
            move_code = "G0"

        f_word = f" F{feed_mm_min:.0f}" if i == 1 or ctx.state.feedrate != feed_mm_min else ""
        z_word = f" Z{z1:.3f}" if (p.ndim == 3 or abs(z1 - ctx.state.z) > 1e-6) else ""
        lines.append(f"{move_code}{f_word} X{x1:.4f} Y{y1:.4f}{z_word}{e_word}")

        ctx.state.x = float(x1)
        ctx.state.y = float(y1)
        ctx.state.z = float(z1)
        ctx.state.feedrate = feed_mm_min

        for ev in events_by_vertex.get(i, ()):
            lines.extend(_run_procedure(ev.procedure, ctx, **ev.kwargs))

    if after_path is not None:
        lines.extend(_run_procedure(after_path, ctx))

    return lines


# ── Complete G-code orchestration (2D, 2.5D, 3D) ─────────────────────────────

def _resolve_filaments(
    filament: FilamentProfile | str | FilePath | dict[str, Any],
) -> tuple[FilamentProfile, dict[str, FilamentProfile]]:
    """Return ``(default_filament, filaments_by_kind)``."""
    if isinstance(filament, dict) and not (
        "name" in filament or "filament_type" in filament or "nozzle_temperature" in filament
    ):
        if not filament:
            raise ValueError("filament mapping dict cannot be empty")
        by_kind = {k: load_filament(v) for k, v in filament.items()}
        default_prof = next(iter(by_kind.values()))
        return default_prof, by_kind
    prof = load_filament(filament)
    return prof, {}


def generate_gcode(
    paths: Sequence[Any],
    *,
    printer: PrinterProfile | str | FilePath | dict[str, Any] = "raise3d_pro3",
    filament: FilamentProfile | str | FilePath | dict[str, Any] = "voxelpla",
    mode: Mode = "2d",
    n_layers: int = 1,
    z0: float = 0.2,
    layer_height: float = 0.2,
    width: float = 0.4,
    print_speed: float | None = None,
    retract: bool = True,
    hop: float | None = None,
    bed_center: bool | tuple[float, float] = False,
    offset: tuple[float, float] | None = None,
    scale: float | None = None,
    tool_map: dict[str, int] | None = None,
    on_kind: dict[str, str | ProcedureFn | list[str]] | None = None,
    procedures: dict[str, str | ProcedureFn | list[str] | None] | None = None,
    before_path: str | ProcedureFn | list[str] | None = None,
    after_path: str | ProcedureFn | list[str] | None = None,
    auto_cut_fiber: bool = False,
    annotate: bool = True,
) -> str:
    """Generate a complete, printable G-code file from paths.

    Parameters
    ----------
    paths : sequence
        * When ``mode="2d"``: a sequence of 2D paths (``Path``, ``PrintPath``,
          ``(n, 2)`` arrays, or :class:`ProcedureCall`) representing one layer,
          which is stacked ``n_layers`` times.
        * When ``mode="2.5d"``: a sequence of layer sequences
          (``list[list[Path]]``), where each entry is a 2D layer with its own
          paths.
        * When ``mode="3d"``: a sequence of 3D paths with ``(n, 3)`` vertices.
    printer : PrinterProfile, str, Path, or dict
        Printer configuration or built-in name (e.g. ``"raise3d_pro3"``,
        ``"prusa_xl_multi"``, ``"generic_reprap"``).
    filament : FilamentProfile, str, Path, or dict[str, FilamentProfile]
        Single filament profile (e.g. ``"voxelpla"``) or a per-kind mapping
        ``{"outline": "voxelpla", "fibre": "generic_cf_pa"}``.
    mode : {"2d", "2.5d", "3d"}
        Dimensionality mode switch.
    n_layers : int
        Number of layers to stack when ``mode="2d"`` (ignored in ``"2.5d"`` and
        ``"3d"`` modes).
    z0 : float
        First-layer Z height in mm (for ``"2d"`` and ``"2.5d"`` modes).
    layer_height : float
        Vertical spacing between layers and default bead height in mm.
    width : float
        Default bead extrusion width in mm.
    print_speed : float, optional
        Override print speed in mm/min (defaults to printer's first-layer and
        regular print speeds).
    retract : bool
        Enable filament retraction on travel moves.
    hop : float, optional
        Override Z-hop height in mm on travel moves.
    bed_center : bool or (float, float)
        If ``True``, center the design on ``printer.bed_center``; if a tuple
        ``(cx, cy)``, center on those printer coordinates in mm.
    offset : (float, float), optional
        Explicit ``(dx, dy)`` shift in mm (mutually exclusive with
        ``bed_center``).
    scale : float, optional
        Override ``printer.coordinate_scale`` (which defaults to ``1000.0`` to
        convert ``path_optimizer`` meters into millimeters).
    tool_map : dict[str, int], optional
        Map ``path.kind`` to extruder tool index ``0 .. num_extruders - 1``.
        When a kind change alters the active tool, the printer's
        ``tool_change`` procedure is invoked automatically.
    on_kind : dict[str, procedure], optional
        Additional procedures to invoke whenever ``path.kind`` changes.
    procedures : dict[str, procedure | None], optional
        Override any printer procedure (``"start"``, ``"purge"``,
        ``"layer_change"``, ``"travel"``, ``"retract"``, ``"unretract"``,
        ``"tool_change"``, ``"cut"``, ``"end"``) for this job.
    before_path, after_path : procedure, optional
        Procedures invoked before or after every path.
    auto_cut_fiber : bool
        If ``True``, automatically insert the printer's ``"cut"`` procedure
        ``cut_offset_mm`` before the end of any path whose filament is marked
        ``continuous_fiber=True`` (unless that path already carries events).
    annotate : bool
        Emit descriptive G-code comments for each path.
    """
    if mode not in ("2d", "2.5d", "3d"):
        raise ValueError(f"mode must be '2d', '2.5d', or '3d', got {mode!r}")

    printer_prof = load_printer(printer)
    if procedures:
        merged_procs = dict(printer_prof.procedures)
        for k, v in procedures.items():
            if v is None:
                merged_procs.pop(k, None)
            else:
                merged_procs[k] = v  # type: ignore[assignment]
        printer_prof = PrinterProfile(
            **{
                **printer_prof.__dict__,
                "procedures": merged_procs,
            }
        )

    default_filament, filaments_by_kind = _resolve_filaments(filament)
    validate_compatibility(printer_prof, default_filament, tool=0)
    for kind_key, fil_prof in filaments_by_kind.items():
        mapped_tool = (tool_map or {}).get(kind_key, 0)
        validate_compatibility(printer_prof, fil_prof, tool=mapped_tool)

    coord_scale = float(
        scale if scale is not None else printer_prof.coordinate_scale
    )

    # Normalize layers according to `mode`
    layers: list[list[PrintPath | ProcedureCall]] = []
    if mode == "2d":
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        base_layer = _normalize_sequence(paths)
        for item in base_layer:
            if isinstance(item, PrintPath) and item.ndim != 2:
                raise ValueError(
                    f"mode='2d' requires 2D paths (n, 2), got shape {item.nodes.shape}"
                )
        layers = [list(base_layer) for _ in range(n_layers)]
    elif mode == "2.5d":
        if not paths:
            raise ValueError("mode='2.5d' requires a non-empty sequence of layers")
        for layer_idx, raw_layer in enumerate(paths):
            norm_layer = _normalize_sequence(raw_layer)
            for item in norm_layer:
                if isinstance(item, PrintPath) and item.ndim != 2:
                    raise ValueError(
                        f"mode='2.5d' layer {layer_idx} requires 2D paths (n, 2), "
                        f"got shape {item.nodes.shape}"
                    )
            layers.append(norm_layer)
    else:  # mode == "3d"
        norm_3d = _normalize_sequence(paths)
        for item in norm_3d:
            if isinstance(item, PrintPath) and item.ndim != 3:
                raise ValueError(
                    f"mode='3d' requires 3D paths (n, 3), got shape {item.nodes.shape}"
                )
        layers = [norm_3d]

    all_paths = [
        item for layer in layers for item in layer if isinstance(item, PrintPath)
    ]
    if not all_paths:
        raise ValueError("No paths provided to generate_gcode.")

    # Resolve bed_center / offset across all layers together
    if bed_center is not False and bed_center is not None:
        if offset is not None:
            raise ValueError("Provide either bed_center or offset, not both.")
        target_center = (
            printer_prof.bed_center
            if bed_center is True
            else (float(bed_center[0]), float(bed_center[1]))
        )
        assert target_center is not None
        offset = bed_offset(all_paths, bed_center=target_center, scale=coord_scale)

    # Optional automatic upstream fiber cutting
    if auto_cut_fiber:
        cut_spec = printer_prof.procedures.get("cut", "continuous_fiber_cut")
        for layer_idx, layer in enumerate(layers):
            updated_layer: list[PrintPath | ProcedureCall] = []
            for item in layer:
                if isinstance(item, PrintPath) and not item.events:
                    fil_for_path = filaments_by_kind.get(item.kind, default_filament)
                    if fil_for_path.is_continuous_fiber and fil_for_path.cut_offset_mm:
                        item = insert_at_distance(
                            item,
                            procedure=cut_spec,
                            distance_mm=fil_for_path.cut_offset_mm,
                            from_end=True,
                            scale=coord_scale,
                        )
                updated_layer.append(item)
            layers[layer_idx] = updated_layer

    # Initialize state and context
    state = PrintState(
        relative_extrusion=printer_prof.relative_extrusion,
        layer_height=layer_height,
        layer_z=z0,
    )
    ctx = PrintContext(
        printer=printer_prof,
        filament=default_filament,
        state=state,
        width=width,
        layer_height=layer_height,
        print_speed=print_speed,
        retract=retract,
        hop=hop,
        filaments_by_kind=filaments_by_kind,
        tool_map=dict(tool_map or {}),
    )

    stamp = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        f"; Generated by anisotropic_postprocessor on {stamp}",
        f"; Printer: {printer_prof.name} ({printer_prof.firmware})",
        f"; Filament: {default_filament.name} ({default_filament.filament_type})",
        f"; Mode: {mode} | Layers: {len(layers)} | Width: {width:.3f} mm "
        f"| Layer height: {layer_height:.3f} mm",
        f"; Coordinate scale (path -> {printer_prof.units}): {coord_scale:g}",
    ]

    # 1. Start procedure
    start_spec = printer_prof.procedures.get("start")
    if start_spec is not None:
        lines.extend(_run_procedure(start_spec, ctx))

    # 2. Purge procedure
    purge_spec = printer_prof.procedures.get("purge")
    if purge_spec is not None:
        lines.extend(_run_procedure(purge_spec, ctx))

    # 3. Layers and paths
    layer_change_spec = printer_prof.procedures.get("layer_change")
    tool_change_spec = printer_prof.procedures.get("tool_change", "standard_tool_change")

    for k, layer_items in enumerate(layers):
        layer_z = z0 + k * layer_height if mode in ("2d", "2.5d") else float(
            next(it for it in layer_items if isinstance(it, PrintPath)).nodes[0, 2]
            * coord_scale
        )
        ctx.state.layer = k
        ctx.state.layer_z = layer_z

        if layer_change_spec is not None and mode in ("2d", "2.5d"):
            lines.extend(
                _run_procedure(
                    layer_change_spec,
                    ctx,
                    layer=k,
                    z=layer_z,
                    layer_height=layer_height,
                )
            )
        elif mode == "3d":
            lines.append(";LAYER:0")

        for item in layer_items:
            if isinstance(item, ProcedureCall):
                lines.extend(_run_procedure(item.procedure, ctx, **item.kwargs))
                continue

            # Check for `kind` transition (multi-material / tool change / on_kind hooks)
            if item.kind != ctx.state.current_kind:
                if item.kind in filaments_by_kind:
                    ctx.filament = filaments_by_kind[item.kind]
                if tool_map and item.kind in tool_map:
                    target_tool = tool_map[item.kind]
                    if target_tool != ctx.state.tool:
                        lines.extend(
                            _run_procedure(
                                tool_change_spec,
                                ctx,
                                tool=target_tool,
                                filament=ctx.filament,
                            )
                        )
                if on_kind and item.kind in on_kind:
                    lines.extend(_run_procedure(on_kind[item.kind], ctx))
                ctx.state.current_kind = item.kind

            lines.extend(
                path_gcode(
                    item,
                    ctx,
                    z=layer_z if mode in ("2d", "2.5d") else None,
                    scale=coord_scale,
                    offset=offset,
                    annotate=annotate,
                    before_path=before_path,
                    after_path=after_path,
                )
            )

    # 4. End procedure
    end_spec = printer_prof.procedures.get("end")
    if end_spec is not None:
        lines.extend(_run_procedure(end_spec, ctx, last_z=ctx.state.z))

    return "\n".join(lines) + "\n"


def generate_gcode_layers(
    layers: Sequence[Sequence[Any]],
    *,
    printer: PrinterProfile | str | FilePath | dict[str, Any] = "raise3d_pro3",
    filament: FilamentProfile | str | FilePath | dict[str, Any] = "voxelpla",
    z0: float = 0.2,
    layer_height: float = 0.2,
    **kwargs: Any,
) -> str:
    """Convenience wrapper around :func:`generate_gcode` with ``mode="2.5d"``."""
    return generate_gcode(
        layers,
        printer=printer,
        filament=filament,
        mode="2.5d",
        z0=z0,
        layer_height=layer_height,
        **kwargs,
    )


def write_gcode(
    gcode_or_paths: str | Sequence[Any],
    filename: str | FilePath,
    **kwargs: Any,
) -> str:
    """Write a G-code string (or generate one from paths) to ``filename``."""
    if isinstance(gcode_or_paths, str):
        if kwargs:
            raise ValueError(
                "Keyword arguments are only used when passing paths to write_gcode, "
                "not an already-generated G-code string."
            )
        gcode_text = gcode_or_paths
    else:
        gcode_text = generate_gcode(gcode_or_paths, **kwargs)
    PathObj = FilePath(filename)
    PathObj.write_text(gcode_text, encoding="utf-8")
    return gcode_text
