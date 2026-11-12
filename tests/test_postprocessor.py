from __future__ import annotations

from dataclasses import dataclass
import numpy as onp
import pytest

import anisotropic_postprocessor as ap


@dataclass
class DummyOptimizerPath:
    """Mimics path_optimizer.paths.Path without requiring path_optimizer installed."""

    nodes: onp.ndarray
    kind: str = "path"
    path_id: int | None = None


def make_line_si(
    y_m: float = 0.010,
    x0_m: float = 0.010,
    x1_m: float = 0.060,
    n: int = 3,
    kind: str = "fibre",
    path_id: int = 0,
) -> DummyOptimizerPath:
    """Return a straight 2D line in SI meters (default 50 mm long)."""
    xs = onp.linspace(x0_m, x1_m, n)
    ys = onp.full(n, y_m)
    return DummyOptimizerPath(onp.column_stack([xs, ys]), kind=kind, path_id=path_id)


def test_2d_mode_scales_from_si_and_stacks_layers():
    p0 = make_line_si(y_m=0.010, x0_m=0.010, x1_m=0.060, kind="fibre", path_id=0)
    p1 = make_line_si(y_m=0.020, x0_m=0.010, x1_m=0.060, kind="outline", path_id=1)

    gcode = ap.generate_gcode(
        [p0, p1],
        printer="raise3d_pro3",
        filament="voxelpla",
        mode="2d",
        n_layers=3,
        z0=0.2,
        layer_height=0.2,
        width=0.4,
        bed_center=True,
    )

    assert ";LAYER:0" in gcode
    assert ";LAYER:1" in gcode
    assert ";LAYER:2" in gcode
    assert ";LAYER:3" not in gcode
    assert "; fibre path 0" in gcode
    assert "; outline path 1" in gcode

    # Verify bed centering on Raise3D Pro3 (300 x 300 -> center at 150, 150)
    sim = ap.parse_gcode(gcode)
    assert sim.num_layers == 3
    mid_x = 0.5 * (sim.bounds_min[0] + sim.bounds_max[0])
    mid_y = 0.5 * (sim.bounds_min[1] + sim.bounds_max[1])
    assert mid_x == pytest.approx(150.0, abs=1e-3)
    assert mid_y == pytest.approx(150.0, abs=1e-3)
    # 2 paths of 50 mm per layer * 3 layers = 300 mm of extrusion distance
    assert sim.extruded_distance_mm == pytest.approx(300.0, rel=1e-4)


def test_2_5d_mode_stacks_distinct_layers_with_shared_centering():
    layer_0 = [make_line_si(y_m=0.010, x0_m=0.0, x1_m=0.040, kind="fibre", path_id=0)]
    layer_1 = [make_line_si(y_m=0.030, x0_m=0.020, x1_m=0.080, kind="fibre", path_id=1)]

    gcode = ap.generate_gcode(
        [layer_0, layer_1],
        printer="raise3d_pro3",
        filament="voxelpla",
        mode="2.5d",
        z0=0.25,
        layer_height=0.25,
        bed_center=True,
    )
    sim = ap.parse_gcode(gcode)
    assert sim.num_layers == 2
    # Layer 0 is 40 mm + Layer 1 is 60 mm = 100 mm
    assert sim.extruded_distance_mm == pytest.approx(100.0, rel=1e-4)
    # Combined X span is [0, 80] mm -> 80 mm wide, centered at 150 -> [110, 190]
    assert sim.bounds_min[0] == pytest.approx(110.0, abs=1e-3)
    assert sim.bounds_max[0] == pytest.approx(190.0, abs=1e-3)
    assert sim.bounds_min[2] == pytest.approx(0.25, abs=1e-3)
    assert sim.bounds_max[2] == pytest.approx(0.50, abs=1e-3)


def test_3d_mode_emits_nonplanar_xyz_coordinates():
    # A 3D helical/ramp path in SI meters: X from 0.02 to 0.07 m, Z from 0.0002 to 0.0022 m
    pts_3d = onp.array(
        [
            [0.020, 0.020, 0.0002],
            [0.045, 0.020, 0.0012],
            [0.070, 0.020, 0.0022],
        ]
    )
    gcode = ap.generate_gcode(
        [pts_3d],
        printer="generic_reprap",
        filament="voxelpla",
        mode="3d",
        bed_center=False,
    )
    sim = ap.parse_gcode(gcode)
    extrude_moves = [m for m in sim.moves if m.move_type == "extrude"]
    assert len(extrude_moves) == 2
    assert extrude_moves[0].start == pytest.approx((20.0, 20.0, 0.2), abs=1e-3)
    assert extrude_moves[0].end == pytest.approx((45.0, 20.0, 1.2), abs=1e-3)
    assert extrude_moves[1].end == pytest.approx((70.0, 20.0, 2.2), abs=1e-3)


def test_mode_validation_rejects_mismatched_dimensions():
    path_2d = make_line_si()
    path_3d = onp.array([[0.01, 0.01, 0.001], [0.02, 0.01, 0.002]])

    with pytest.raises(ValueError, match="mode='3d' requires 3D paths"):
        ap.generate_gcode([path_2d], mode="3d")

    with pytest.raises(ValueError, match="mode='2d' requires 2D paths"):
        ap.generate_gcode([path_3d], mode="2d")


def test_insert_at_distance_interpolates_vertex_and_fires_fiber_cut():
    # Single 2-node segment of length 100 mm (0.100 m in SI)
    raw_path = DummyOptimizerPath(
        onp.array([[0.010, 0.020], [0.110, 0.020]]), kind="fibre", path_id=0
    )
    # Insert fiber cut 15 mm before the end -> at X = 0.010 + (0.100 - 0.015) = 0.095 m (95 mm)
    cut_path = ap.insert_at_distance(
        raw_path,
        procedure="continuous_fiber_cut",
        distance_mm=15.0,
        from_end=True,
        scale=1000.0,
    )
    assert len(cut_path.nodes) == 3
    assert cut_path.nodes[1, 0] == pytest.approx(0.095)
    assert len(cut_path.events) == 1
    assert cut_path.events[0].vertex_index == 1

    gcode = ap.generate_gcode(
        [cut_path],
        printer="raise3d_pro3",
        filament="voxelpla",
        mode="2d",
        n_layers=1,
    )
    sim = ap.parse_gcode(gcode)
    cuts = [e for e in sim.events if e.event_type == "cut_fiber"]
    assert len(cuts) == 1
    assert cuts[0].position == pytest.approx((95.0, 20.0, 0.2), abs=1e-3)


def test_multi_tool_and_auto_cut_fiber_on_prusa_xl():
    outline = make_line_si(y_m=0.020, x0_m=0.020, x1_m=0.080, n=2, kind="outline", path_id=0)
    fibre = make_line_si(y_m=0.030, x0_m=0.020, x1_m=0.080, n=2, kind="fibre", path_id=1)

    gcode = ap.generate_gcode(
        [outline, fibre],
        printer="prusa_xl_multi",
        filament={"outline": "voxelpla", "fibre": "generic_cf_pa"},
        tool_map={"outline": 0, "fibre": 1},
        mode="2d",
        n_layers=1,
        auto_cut_fiber=True,
    )
    sim = ap.parse_gcode(gcode)

    # Tool 0 prints 60 mm of outline; Tool 1 prints 60 mm of continuous fiber
    assert 0 in sim.filament_used_mm
    assert 1 in sim.filament_used_mm

    # Auto fiber cut fires 15 mm before the end of the 60 mm fiber path (at X = 80 - 15 = 65 mm)
    cuts = [e for e in sim.events if e.event_type == "cut_fiber"]
    assert len(cuts) == 1
    assert cuts[0].position[0] == pytest.approx(65.0, abs=1e-3)
    assert cuts[0].tool == 1


def test_inter_path_procedure_call_and_out_of_bounds_check():
    p0 = make_line_si(y_m=0.020, x0_m=0.020, x1_m=0.050)
    p1 = make_line_si(y_m=0.030, x0_m=0.020, x1_m=0.050)

    gcode = ap.generate_gcode(
        [p0, ap.ProcedureCall(["M400 ; Pause between paths"]), p1],
        printer="generic_reprap",
        filament="voxelpla",
    )
    assert "M400 ; Pause between paths" in gcode

    # Path outside the 250x250 mm bed of generic_reprap raises ValueError
    way_outside = make_line_si(y_m=0.500, x0_m=0.020, x1_m=0.050)
    with pytest.raises(ValueError, match="exceed printer"):
        ap.generate_gcode([way_outside], printer="generic_reprap", filament="voxelpla")
