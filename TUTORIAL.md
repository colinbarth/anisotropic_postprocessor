# Implementing `anisotropic_postprocessor` in `path_optimizer`

This tutorial walks through integrating `anisotropic_postprocessor` into a `path_optimizer` workflow to convert optimized anisotropic toolpaths (in SI meters) into machine-ready G-code and verify them with interactive 3D machine simulation.

---

## 1. How `path_optimizer` Connects to `anisotropic_postprocessor`

`path_optimizer` computes structural and fiber toolpaths in **SI units (meters)** and outputs sequences of `path_optimizer.paths.Path` objects, where each path has:
- `.nodes`: a `(n, 2)` (or `(n, 3)`) NumPy array of vertex coordinates in meters (`m`).
- `.kind`: a semantic category string such as `"outline"` or `"fibre"`.
- `.path_id`: an integer identifier.

`anisotropic_postprocessor` accepts `path_optimizer.paths.Path` objects **directly via duck typing** (no conversion wrapper or `path_optimizer` dependency required) as well as raw `(n, 2)` / `(n, 3)` NumPy arrays or its own `ap.PrintPath` objects.

```
path_optimizer (SI meters)
   └── final: list[Path]  (.nodes, .kind, .path_id)
          │
          ▼
anisotropic_postprocessor
   ├── PrinterProfile  (coordinate_scale=1000.0 -> mm, bed bounds, speeds, procedure map)
   ├── FilamentProfile (temps, flow, fan rules, retraction, cut_offset_mm)
   ├── Procedures      (start, purge, layer_change, travel, retract, tool_change, cut, end)
   │      │
          ▼
   generate_gcode() / write_gcode()  ──►  printable .gcode
          │
          ▼
   parse_gcode() / plot_gcode()      ──►  Siemens NX-style 3D G-code simulation
```

---

## 2. Installation

Install `anisotropic_postprocessor` into the same Python environment (or Google Colab notebook) as `path_optimizer`:

```bash
# Local editable install from a cloned repository
pip install -e /path/to/anisotropic_postprocessor

# Or directly from GitHub (e.g. in Google Colab)
pip install git+https://github.com/colinbarth/anisotropic_postprocessor
```

Verify available built-in profiles in Python:

```python
import anisotropic_postprocessor as ap

print("Built-in printers:", ap.list_builtin_printers())
# ['generic_reprap', 'prusa_xl_multi', 'raise3d_pro3']

print("Built-in filaments:", ap.list_builtin_filaments())
# ['generic_cf_pa', 'voxelpla']
```

---

## 3. Basic Integration: 2D Layer Stacking (`mode="2d"`)

In a standard 2D `path_optimizer` run (such as an MBB beam or cantilever), your optimization pipeline produces a single 2D layer list `final: list[Path]` with coordinates in meters.

Use `mode="2d"` to scale the paths from meters to millimeters (`coordinate_scale = 1000.0`), center the part on the build plate, and repeat the 2D pattern across `n_layers`:

```python
import anisotropic_postprocessor as ap

# `final` is the list[Path] returned by path_optimizer (coordinates in meters)
printer = ap.load_printer("raise3d_pro3")
filament = ap.load_filament("voxelpla")

gcode = ap.generate_gcode(
    final,
    printer=printer,
    filament=filament,
    mode="2d",            # Repeat a single 2D layer pattern n_layers times
    n_layers=4,           # Number of stacked layers
    z0=0.2,               # First-layer Z height (mm)
    layer_height=0.2,     # Layer thickness & default bead height (mm)
    width=0.4,            # Default extrusion bead width (mm)
    retract=True,         # Retract and unretract across travel moves
    hop=0.6,              # Z-hop lift during travel moves (mm)
    bed_center=True,      # Center XY bounding box on the printer bed
)

ap.write_gcode(gcode, "mbb_beam.gcode")
```

> **Note on Coordinate Scaling & Build Volume Checks:**
> Printer profiles default to `"coordinate_scale": 1000.0`, which multiplies `path.nodes` by `1000` so `path_optimizer` stays strictly in SI meters while emitted G-code is in millimeters. If any scaled path exceeds the printer's `bed_size` or `max_z`, `generate_gcode()` raises a `ValueError` immediately. Use `bed_center=True`, `bed_center=(cx, cy)`, or `offset=(dx, dy)` (in mm) to position the part on the build plate.

---

## 4. 2.5D Multi-Angle Laminates & 3D Non-Planar Toolpaths

### 4.1 Distinct Stacked Layers (`mode="2.5d"`)

When each layer in your composite layup has its own distinct toolpaths (for example, alternating $[0^\circ / 90^\circ / \pm 45^\circ]$ plies or layer-by-layer topology optimization), pass a sequence of layers `list[list[Path]]` with `mode="2.5d"` (or use the `ap.generate_gcode_layers` helper).

Bed centering is computed across the combined bounding box of **all** layers so layer-to-layer registration is preserved:

```python
import anisotropic_postprocessor as ap

# Each entry in `layers` is a list[Path] for that layer (in SI meters)
layers = [
    layer_0_paths,  # z = 0.20 mm
    layer_1_paths,  # z = 0.40 mm
    layer_2_paths,  # z = 0.60 mm
]

gcode = ap.generate_gcode_layers(
    layers,
    printer="raise3d_pro3",
    filament="voxelpla",
    z0=0.2,
    layer_height=0.2,
    width=0.4,
    bed_center=True,
)
```

### 4.2 Full 3D Non-Planar Toolpaths (`mode="3d"`)

If your optimizer generates spatial `(n, 3)` polylines where `Z` varies continuously along a path, set `mode="3d"`. Every segment emits coordinated `X Y Z E` moves:

```python
import numpy as np
import anisotropic_postprocessor as ap

# List of Path / PrintPath / (n, 3) arrays in SI meters
nonplanar_paths = [
    ap.PrintPath(
        nodes=np.array([
            [0.020, 0.020, 0.0002],
            [0.050, 0.020, 0.0012],
            [0.080, 0.020, 0.0022],
        ]),
        kind="fibre",
    )
]

gcode_3d = ap.generate_gcode(
    nonplanar_paths,
    printer="generic_reprap",
    filament="voxelpla",
    mode="3d",
    bed_center=True,
)
```

---

## 5. Multi-Material Printing & Continuous Fiber Cutting

`path_optimizer` labels paths with `.kind` (commonly `"outline"` for thermoplastic perimeters/matrix and `"fibre"` for continuous carbon/glass fiber reinforcement). `anisotropic_postprocessor` uses `.kind` to automate tool changes, filament switching, and upstream fiber cutting.

### 5.1 Mapping `.kind` to Extruders and Filaments

Pass a dictionary to `filament` and `tool_map`. Whenever `path.kind` changes between paths:
1. The active `FilamentProfile` updates to match the new `kind`.
2. If the mapped tool index changes, the printer's `tool_change` procedure (e.g., `prusa_xl_tool_change` or `standard_tool_change`) fires automatically, setting the incoming nozzle temperature (`M104`/`M109`), updating flow (`M221`), and resetting the extruder datum (`G92 E0`).

```python
import anisotropic_postprocessor as ap

gcode = ap.generate_gcode(
    final,
    printer="prusa_xl_multi",
    filament={
        "outline": "voxelpla",       # Thermoplastic matrix on Tool 0
        "fibre": "generic_cf_pa",    # Continuous carbon fiber on Tool 1
    },
    tool_map={
        "outline": 0,
        "fibre": 1,
    },
    mode="2d",
    n_layers=4,
    bed_center=True,
    auto_cut_fiber=True,             # Automatically cut continuous fiber before path ends
)
```

### 5.2 Upstream Continuous Fiber Cutting

In continuous-fiber printheads, the mechanical cutter sits a fixed distance `cut_offset_mm` upstream of the nozzle tip (e.g. `15.0` mm in `generic_cf_pa.json`). To finish a fiber path cleanly, the cutter must actuate **before** the nozzle reaches the end of the path so the remaining tail is pulled through the nozzle.

You can trigger fiber cuts in two ways:

1. **Automatic (`auto_cut_fiber=True`)**: Any path whose assigned `FilamentProfile` has `"continuous_fiber": true` and `"cut_offset_mm": 15.0` in its `extra` metadata automatically receives a cut procedure `cut_offset_mm` before the end of the path.
2. **Explicit (`ap.add_fiber_cuts` or `ap.insert_at_distance`)**: Pre-process your `path_optimizer` path list before calling `generate_gcode`. If the cut location falls in the middle of a segment, `insert_at_distance` linearly interpolates and splits the segment at the exact arc-length so the cut fires at the exact geometric position:

```python
import anisotropic_postprocessor as ap

# Option A: Attach cuts to all paths with kind == "fibre" in a layer
paths_with_cuts = ap.add_fiber_cuts(
    final,
    cut_offset_mm=15.0,
    procedure="continuous_fiber_cut",
    kind="fibre",
    dwell_ms=300,  # Forwarded as kwargs to continuous_fiber_cut()
)

# Option B: Insert a procedure on a single path by arc-length or vertex index
single_cut_path = ap.insert_at_distance(
    final[0],
    distance_mm=15.0,
    from_end=True,
    procedure="continuous_fiber_cut",
)
vertex_event_path = ap.insert_at_index(
    final[0],
    index=-1,  # Last vertex
    procedure=["M400", "M117 Path complete"],
)
```

### 5.3 Inter-Path Procedures & Per-Path Overrides

- **Between paths (`ap.ProcedureCall`)**: Insert `ap.ProcedureCall` items directly into your path sequence to run a procedure (pause, camera trigger, custom wipe) between two paths:
  ```python
  sequence = [
      final[0],
      ap.ProcedureCall("continuous_fiber_cut", dwell_ms=250),
      ap.ProcedureCall(["M400 ; Wait for moves", "G4 P500 ; Dwell 0.5 s"]),
      final[1],
  ]
  ```
- **Per-path overrides (`ap.PrintPath`)**: Wrap a `Path` or coordinate array in `ap.PrintPath` to override bead `width` (mm), `height` (mm), `speed` (mm/min), `extrude` (`True`/`False`), or `travel` (`True`/`False`) for that specific trajectory:
  ```python
  reinforced_path = ap.PrintPath(
      nodes=final[0].nodes,
      kind="fibre",
      width=0.6,    # Wider bead for fiber tow (mm)
      speed=600.0,  # Slower print speed for fiber placement (mm/min)
  )
  ```

---

## 6. Customizing Profiles & Programmable Procedures

### 6.1 Loading Custom or Slicer JSON Profiles

- **Filaments**: Pass a built-in name (`"voxelpla"`), a path to a custom `.json` file, a Python `dict`, or a raw **PrusaSlicer / Bambu Studio / OrcaSlicer** filament JSON file to `ap.load_filament()`:
  ```python
  custom_filament = ap.load_filament("path/to/bambu_pla_basic.json")
  ```
- **Printers**: Pass a built-in name (`"raise3d_pro3"`), a path to a `.json` file, or a Python `dict` to `ap.load_printer()`. You can check printer/filament compatibility at any time with:
  ```python
  ap.validate_compatibility(printer, custom_filament, tool=0)
  ```

### 6.2 Writing Custom Python Procedures

Every lifecycle hook (`start`, `purge`, `layer_change`, `travel`, `retract`, `unretract`, `tool_change`, `cut`, `end`) is a plain Python function taking `(ctx: ap.PrintContext, **kwargs) -> list[str]`.

You can inspect or update `ctx.state` (`x`, `y`, `z`, `e`, `tool`, `layer`, `is_retracted`) and read `ctx.printer` or `ctx.filament`:

```python
import anisotropic_postprocessor as ap

def custom_nozzle_wipe(ctx: ap.PrintContext, *, wipe_x: float = 10.0) -> list[str]:
    """Wipe the active nozzle at the edge of the bed."""
    f = ctx.printer.rapid_speed
    ctx.state.x = wipe_x
    return [
        "; --- CUSTOM NOZZLE WIPE ---",
        f"G0 F{f:.0f} X{wipe_x:.3f} Y5.000",
        f"G0 F{f:.0f} X{wipe_x + 20.0:.3f} Y5.000",
    ]

# Register by name so printer JSON files or generate_gcode() can reference it
ap.register_procedure("custom_nozzle_wipe", custom_nozzle_wipe)

# Use in generate_gcode() via `procedures`, `on_kind`, `before_path`, or `after_path`
gcode = ap.generate_gcode(
    final,
    printer="raise3d_pro3",
    filament="voxelpla",
    procedures={"purge": "custom_nozzle_wipe"},  # Override default purge
    on_kind={"fibre": custom_nozzle_wipe},       # Or run whenever switching to 'fibre'
    bed_center=True,
)
```

---

## 7. Simulating and Visualizing G-code

Instead of only plotting input geometry before postprocessing, `anisotropic_postprocessor` includes a virtual machine interpreter (`parse_gcode`) and a Siemens NX-style 3D Plotly viewer (`plot_gcode`) that parse the **actual emitted G-code** line by line.

```python
import anisotropic_postprocessor as ap

# 1. Parse the generated G-code string or .gcode file
sim = ap.parse_gcode("mbb_beam.gcode")
print(sim.summary())
# GcodeSimulation(412 moves, 4 layers | extrude 1840.0 mm, travel 312.4 mm |
# filament [T0: 61.2 mm] | retracts: 16, tool changes: 0, fiber cuts: 0 | est. time: 2m 14s)

# 2. Launch interactive 3D verification view (works inline in Jupyter / Google Colab)
fig = ap.plot_gcode(
    sim,
    printer="raise3d_pro3",
    color_by="feedrate",   # Options: "feedrate", "tool", "layer", "z", "time", "type"
    show_travel=True,      # Dashed rapid travel lines (G0)
    show_z_hop=True,       # Dotted Z-hop lifts
    show_purge=True,       # Purge line trace
    show_events=True,      # Markers for retractions, tool changes, and fiber cuts
    slider=True,           # Interactive chronological progress slider + virtual nozzle tip
)
fig.show()
```

---

## 8. Quick Migration Checklist from Legacy `path_optimizer` G-code Export

If your `path_optimizer` notebook previously used `path_steps()` or FullControl to export G-code, replace that block with:

```python
# --- BEFORE (legacy path_optimizer G-code export) ---
# steps = path_steps(final, z0=0.2, layer_height=0.2, n_layers=4, ...)
# fc.transform(steps, "gcode", ...)

# --- AFTER (with anisotropic_postprocessor) ---
import anisotropic_postprocessor as ap

gcode = ap.generate_gcode(
    final,
    printer="raise3d_pro3",
    filament="voxelpla",
    mode="2d",
    n_layers=4,
    z0=0.2,
    layer_height=0.2,
    width=0.4,
    retract=True,
    hop=0.6,
    bed_center=True,
)
ap.write_gcode(gcode, "output.gcode")

fig = ap.plot_gcode(gcode, printer="raise3d_pro3", color_by="feedrate")
fig.show()
```
