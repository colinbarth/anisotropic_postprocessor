"""Filament and printer profiles for G-code postprocessing.

Filament profiles follow the conventions of PrusaSlicer and Bambu Studio /
OrcaSlicer: material identity, filament diameter and density, nozzle and bed
temperatures (initial and regular layers), flow multiplier, pressure advance,
cooling fan rules, and optional retraction overrides.

Printer profiles describe the machine envelope, coordinate scaling from
``path_optimizer``'s SI units (meters) to the printer's native G-code units
(millimeters), temperature and speed limits, default retraction settings,
hardware feature flags, and the names of the Python procedure functions used
to start, purge, change layers, change tools, cut fiber, and finish a print.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FilamentProfile",
    "PrinterProfile",
    "load_filament",
    "load_printer",
    "list_builtin_filaments",
    "list_builtin_printers",
    "validate_compatibility",
]

_DATA_DIR = Path(__file__).resolve().parent / "data"
_FILAMENTS_DIR = _DATA_DIR / "filaments"
_PRINTERS_DIR = _DATA_DIR / "printers"

_VALID_UNITS = {"mm", "cm", "m"}
_VALID_CROSS_SECTIONS = {"rectangle", "stadium"}


# ── Filament profile ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FilamentProfile:
    """Material and extrusion parameters for one filament.

    Field names and semantics match PrusaSlicer / Bambu Studio filament
    profiles.  Temperatures are in °C, filament diameter and retraction lengths
    in mm, speeds in mm/min, density in g/cm³, and ``extrusion_multiplier`` is
    a dimensionless flow ratio where ``1.0`` means 100% flow.
    """

    name: str
    filament_type: str = "PLA"
    filament_vendor: str = "Generic"
    filament_diameter: float = 1.75
    filament_density: float = 1.24
    filament_cost: float = 0.0
    nozzle_temperature: float = 215.0
    nozzle_temperature_initial_layer: float | None = None
    nozzle_temperature_range_low: float | None = None
    nozzle_temperature_range_high: float | None = None
    bed_temperature: float = 55.0
    bed_temperature_initial_layer: float | None = None
    chamber_temperature: float = 0.0
    extrusion_multiplier: float = 1.0
    pressure_advance: float = 0.0
    max_volumetric_speed: float = 15.0
    fan_min_speed: float = 100.0
    fan_max_speed: float = 100.0
    close_fan_the_first_x_layers: int = 1
    auxiliary_fan_speed: float = 0.0
    retract_length: float | None = None
    retract_speed: float | None = None
    unretract_speed: float | None = None
    z_hop: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.nozzle_temperature_initial_layer is None:
            object.__setattr__(
                self, "nozzle_temperature_initial_layer", float(self.nozzle_temperature)
            )
        if self.bed_temperature_initial_layer is None:
            object.__setattr__(
                self, "bed_temperature_initial_layer", float(self.bed_temperature)
            )
        if self.filament_diameter <= 0:
            raise ValueError(
                f"filament_diameter must be positive, got {self.filament_diameter}"
            )
        if self.filament_density <= 0:
            raise ValueError(
                f"filament_density must be positive, got {self.filament_density}"
            )
        if self.extrusion_multiplier <= 0:
            raise ValueError(
                f"extrusion_multiplier must be positive, got {self.extrusion_multiplier}"
            )
        if self.nozzle_temperature < 0:
            raise ValueError(
                f"nozzle_temperature must be non-negative, got {self.nozzle_temperature}"
            )
        if self.bed_temperature < 0:
            raise ValueError(
                f"bed_temperature must be non-negative, got {self.bed_temperature}"
            )
        if not (0.0 <= self.fan_min_speed <= 100.0):
            raise ValueError(
                f"fan_min_speed must be in [0, 100], got {self.fan_min_speed}"
            )
        if not (0.0 <= self.fan_max_speed <= 100.0):
            raise ValueError(
                f"fan_max_speed must be in [0, 100], got {self.fan_max_speed}"
            )
        if not (0.0 <= self.auxiliary_fan_speed <= 100.0):
            raise ValueError(
                f"auxiliary_fan_speed must be in [0, 100], got {self.auxiliary_fan_speed}"
            )

    @property
    def flow_percent(self) -> float:
        """Flow percentage for ``M221 S<pct>`` (e.g. ``1.0`` -> ``100.0``)."""
        return float(self.extrusion_multiplier) * 100.0

    def fan_percent_for_layer(self, layer: int) -> float:
        """Part-cooling fan speed in percent for 0-based ``layer``."""
        if layer < self.close_fan_the_first_x_layers:
            return 0.0
        return float(self.fan_max_speed)

    @property
    def is_continuous_fiber(self) -> bool:
        """Whether this filament is marked as continuous fiber in ``extra``."""
        return bool(self.extra.get("continuous_fiber", False))

    @property
    def cut_offset_mm(self) -> float | None:
        """Distance from cutter to nozzle tip in mm, if specified in ``extra``."""
        val = self.extra.get("cut_offset_mm")
        return float(val) if val is not None else None


# ── Printer profile ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PrinterProfile:
    """Machine geometry, limits, coordinate scaling, and procedure bindings.

    ``coordinate_scale`` converts path coordinates from ``path_optimizer``
    (which works in SI meters) into the printer's native units (``units``,
    default ``"mm"`` with ``coordinate_scale = 1000.0``).

    ``procedures`` maps lifecycle and action names (``"start"``, ``"purge"``,
    ``"layer_change"``, ``"travel"``, ``"retract"``, ``"unretract"``,
    ``"tool_change"``, ``"cut"``, ``"end"``) to Python procedure function names
    registered in :mod:`anisotropic_postprocessor.procedures`.
    """

    name: str
    firmware: str = "reprap"
    units: str = "mm"
    coordinate_scale: float = 1000.0
    bed_size: tuple[float, float] = (300.0, 300.0)
    max_z: float = 300.0
    bed_origin: tuple[float, float] = (0.0, 0.0)
    bed_center: tuple[float, float] | None = None
    nozzle_diameter: float = 0.4
    num_extruders: int = 1
    min_extrusion_temp: float = 170.0
    max_nozzle_temp: float = 300.0
    max_bed_temp: float = 120.0
    default_print_speed: float = 1000.0
    first_layer_print_speed: float = 900.0
    rapid_speed: float = 9000.0
    z_feedrate: float = 300.0
    max_print_speed: float = 18000.0
    print_acceleration: float = 1000.0
    travel_acceleration: float = 1500.0
    relative_extrusion: bool = False
    extrusion_cross_section: str = "rectangle"
    retract_length: float = 0.5
    retract_speed: float = 2400.0
    unretract_speed: float = 1500.0
    z_hop: float = 0.6
    features: dict[str, bool] = field(default_factory=dict)
    procedures: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bed_size = (float(self.bed_size[0]), float(self.bed_size[1]))
        bed_origin = (float(self.bed_origin[0]), float(self.bed_origin[1]))
        object.__setattr__(self, "bed_size", bed_size)
        object.__setattr__(self, "bed_origin", bed_origin)
        if self.bed_center is None:
            center = (
                bed_origin[0] + bed_size[0] / 2.0,
                bed_origin[1] + bed_size[1] / 2.0,
            )
            object.__setattr__(self, "bed_center", center)
        else:
            object.__setattr__(
                self,
                "bed_center",
                (float(self.bed_center[0]), float(self.bed_center[1])),
            )

        if self.units not in _VALID_UNITS:
            raise ValueError(
                f"units must be one of {sorted(_VALID_UNITS)}, got {self.units!r}"
            )
        if self.coordinate_scale <= 0:
            raise ValueError(
                f"coordinate_scale must be positive, got {self.coordinate_scale}"
            )
        if bed_size[0] <= 0 or bed_size[1] <= 0:
            raise ValueError(f"bed_size dimensions must be positive, got {bed_size}")
        if self.max_z <= 0:
            raise ValueError(f"max_z must be positive, got {self.max_z}")
        if self.nozzle_diameter <= 0:
            raise ValueError(
                f"nozzle_diameter must be positive, got {self.nozzle_diameter}"
            )
        if self.num_extruders < 1:
            raise ValueError(
                f"num_extruders must be at least 1, got {self.num_extruders}"
            )
        if self.extrusion_cross_section not in _VALID_CROSS_SECTIONS:
            raise ValueError(
                f"extrusion_cross_section must be one of {sorted(_VALID_CROSS_SECTIONS)}, "
                f"got {self.extrusion_cross_section!r}"
            )


# ── Loading and PrusaSlicer / BambuSlicer normalization ──────────────────────

def _unwrap_slicer_value(val: Any) -> Any:
    """Unwrap single-element list values used by Bambu Studio / OrcaSlicer JSONs."""
    if isinstance(val, list) and len(val) == 1:
        val = val[0]
    if isinstance(val, str):
        stripped = val.strip().rstrip("%")
        if stripped.lower() in ("nil", "none", "n/a", ""):
            return None
        try:
            if "." in stripped or "e" in stripped.lower():
                return float(stripped)
            return int(stripped)
        except ValueError:
            return val
    return val


_SLICER_FILAMENT_ALIASES: dict[str, str] = {
    "filament_settings_id": "name",
    "filament_flow_ratio": "extrusion_multiplier",
    "hot_plate_temp": "bed_temperature",
    "hot_plate_temp_initial_layer": "bed_temperature_initial_layer",
    "bed_temperature_initial_layer_single": "bed_temperature_initial_layer",
    "first_layer_bed_temperature": "bed_temperature_initial_layer",
    "first_layer_temperature": "nozzle_temperature_initial_layer",
    "temperature": "nozzle_temperature",
    "filament_max_volumetric_speed": "max_volumetric_speed",
    "filament_retract_length": "retract_length",
    "filament_retraction_length": "retract_length",
    "filament_z_hop": "z_hop",
    "min_fan_speed": "fan_min_speed",
    "max_fan_speed": "fan_max_speed",
    "disable_fan_first_layers": "close_fan_the_first_x_layers",
}


def _normalize_filament_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a dictionary or PrusaSlicer/BambuSlicer JSON into FilamentProfile kwargs."""
    known_fields = {
        "name",
        "filament_type",
        "filament_vendor",
        "filament_diameter",
        "filament_density",
        "filament_cost",
        "nozzle_temperature",
        "nozzle_temperature_initial_layer",
        "nozzle_temperature_range_low",
        "nozzle_temperature_range_high",
        "bed_temperature",
        "bed_temperature_initial_layer",
        "chamber_temperature",
        "extrusion_multiplier",
        "pressure_advance",
        "max_volumetric_speed",
        "fan_min_speed",
        "fan_max_speed",
        "close_fan_the_first_x_layers",
        "auxiliary_fan_speed",
        "retract_length",
        "retract_speed",
        "unretract_speed",
        "z_hop",
    }
    out: dict[str, Any] = {}
    extra: dict[str, Any] = dict(raw.get("extra") or {})

    for k, v in raw.items():
        if k == "extra":
            continue
        target_key = _SLICER_FILAMENT_ALIASES.get(k, k)
        unwrapped = _unwrap_slicer_value(v)
        if target_key in known_fields:
            if unwrapped is not None and target_key not in out:
                if target_key in ("name", "filament_type", "filament_vendor"):
                    out[target_key] = str(unwrapped)
                elif target_key == "close_fan_the_first_x_layers":
                    out[target_key] = int(unwrapped)
                else:
                    out[target_key] = float(unwrapped)
        else:
            extra[k] = v

    if "name" not in out:
        out["name"] = str(extra.pop("filament_id", "Custom Filament"))
    out["extra"] = extra
    return out


def _normalize_printer_dict(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a dictionary into PrinterProfile kwargs."""
    known_fields = {
        "name",
        "firmware",
        "units",
        "coordinate_scale",
        "bed_size",
        "max_z",
        "bed_origin",
        "bed_center",
        "nozzle_diameter",
        "num_extruders",
        "min_extrusion_temp",
        "max_nozzle_temp",
        "max_bed_temp",
        "default_print_speed",
        "first_layer_print_speed",
        "rapid_speed",
        "z_feedrate",
        "max_print_speed",
        "print_acceleration",
        "travel_acceleration",
        "relative_extrusion",
        "extrusion_cross_section",
        "retract_length",
        "retract_speed",
        "unretract_speed",
        "z_hop",
        "features",
        "procedures",
    }
    out: dict[str, Any] = {}
    extra: dict[str, Any] = dict(raw.get("extra") or {})
    for k, v in raw.items():
        if k == "extra":
            continue
        if k in known_fields:
            out[k] = v
        else:
            extra[k] = v
    out["extra"] = extra
    return out


def _resolve_json_path(source: str | Path, builtin_dir: Path, kind_label: str) -> Path:
    """Resolve either an explicit file path or a built-in profile name."""
    candidate = Path(source)
    if candidate.is_file():
        return candidate
    if candidate.suffix == "":
        with_json = candidate.with_suffix(".json")
        if with_json.is_file():
            return with_json
        builtin = builtin_dir / f"{source}.json"
        if builtin.is_file():
            return builtin
    else:
        builtin = builtin_dir / candidate.name
        if builtin.is_file():
            return builtin
    available = sorted(p.stem for p in builtin_dir.glob("*.json"))
    raise FileNotFoundError(
        f"{kind_label} profile {str(source)!r} was not found as a file or in "
        f"built-in profiles {available}"
    )


def list_builtin_filaments() -> list[str]:
    """Names of filament profiles shipped with ``anisotropic_postprocessor``."""
    return sorted(p.stem for p in _FILAMENTS_DIR.glob("*.json"))


def list_builtin_printers() -> list[str]:
    """Names of printer profiles shipped with ``anisotropic_postprocessor``."""
    return sorted(p.stem for p in _PRINTERS_DIR.glob("*.json"))


def load_filament(
    source: str | Path | dict[str, Any] | FilamentProfile,
) -> FilamentProfile:
    """Load a :class:`FilamentProfile` from a file path, built-in name, or dict.

    Parameters
    ----------
    source : str, Path, dict, or FilamentProfile
        A path to a ``.json`` file (e.g. ``"voxelpla.json"``), the stem of a
        built-in profile (e.g. ``"voxelpla"`` or ``"generic_cf_pa"``), a
        dictionary of parameters (including raw PrusaSlicer / Bambu Studio JSON
        dictionaries), or an already-constructed :class:`FilamentProfile`.
    """
    if isinstance(source, FilamentProfile):
        return source
    if isinstance(source, dict):
        return FilamentProfile(**_normalize_filament_dict(source))
    path = _resolve_json_path(source, _FILAMENTS_DIR, "Filament")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FilamentProfile(**_normalize_filament_dict(raw))


def load_printer(
    source: str | Path | dict[str, Any] | PrinterProfile,
) -> PrinterProfile:
    """Load a :class:`PrinterProfile` from a file path, built-in name, or dict.

    Parameters
    ----------
    source : str, Path, dict, or PrinterProfile
        A path to a ``.json`` file (e.g. ``"raise3d_pro3.json"``), the stem of
        a built-in profile (e.g. ``"raise3d_pro3"``, ``"prusa_xl_multi"``, or
        ``"generic_reprap"``), a dictionary of parameters, or an
        already-constructed :class:`PrinterProfile`.
    """
    if isinstance(source, PrinterProfile):
        return source
    if isinstance(source, dict):
        return PrinterProfile(**_normalize_printer_dict(source))
    path = _resolve_json_path(source, _PRINTERS_DIR, "Printer")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PrinterProfile(**_normalize_printer_dict(raw))


def validate_compatibility(
    printer: PrinterProfile,
    filament: FilamentProfile,
    *,
    tool: int = 0,
) -> None:
    """Verify that ``filament`` temperatures and ``tool`` fit ``printer`` limits.

    Raises
    ------
    ValueError
        If the filament requests a nozzle or bed temperature outside the
        printer's allowed range, or if ``tool`` exceeds ``printer.num_extruders``.
    """
    if tool < 0 or tool >= printer.num_extruders:
        raise ValueError(
            f"Tool T{tool} requested for filament {filament.name!r}, but printer "
            f"{printer.name!r} only has {printer.num_extruders} extruder(s) "
            f"(T0..T{printer.num_extruders - 1})."
        )
    for label, temp in (
        ("nozzle_temperature", filament.nozzle_temperature),
        ("nozzle_temperature_initial_layer", filament.nozzle_temperature_initial_layer),
    ):
        if temp is None:
            continue
        if temp > printer.max_nozzle_temp:
            raise ValueError(
                f"Filament {filament.name!r} {label} ({temp} °C) exceeds printer "
                f"{printer.name!r} max_nozzle_temp ({printer.max_nozzle_temp} °C)."
            )
        if 0 < temp < printer.min_extrusion_temp:
            raise ValueError(
                f"Filament {filament.name!r} {label} ({temp} °C) is below printer "
                f"{printer.name!r} min_extrusion_temp ({printer.min_extrusion_temp} °C)."
            )
    for label, temp in (
        ("bed_temperature", filament.bed_temperature),
        ("bed_temperature_initial_layer", filament.bed_temperature_initial_layer),
    ):
        if temp is None:
            continue
        if temp > printer.max_bed_temp:
            raise ValueError(
                f"Filament {filament.name!r} {label} ({temp} °C) exceeds printer "
                f"{printer.name!r} max_bed_temp ({printer.max_bed_temp} °C)."
            )
