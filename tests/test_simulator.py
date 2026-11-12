from __future__ import annotations

import numpy as onp
import pytest

import anisotropic_postprocessor as ap


def test_parse_and_plot_gcode_simulation(tmp_path):
    p0 = ap.PrintPath(
        onp.array([[0.020, 0.020], [0.080, 0.020], [0.080, 0.050]]),
        kind="outline",
        path_id=0,
    )
    p1 = ap.insert_at_distance(
        ap.PrintPath(
            onp.array([[0.025, 0.030], [0.075, 0.030]]),
            kind="fibre",
            path_id=1,
        ),
        procedure="continuous_fiber_cut",
        distance_mm=10.0,
        from_end=True,
    )

    gcode_file = tmp_path / "sim_test.gcode"
    ap.write_gcode(
        [p0, p1],
        gcode_file,
        printer="prusa_xl_multi",
        filament={"outline": "voxelpla", "fibre": "generic_cf_pa"},
        tool_map={"outline": 0, "fibre": 1},
        mode="2d",
        n_layers=2,
        bed_center=True,
    )

    sim = ap.parse_gcode(gcode_file)
    assert sim.num_layers == 2
    assert sim.total_time_s > 0.0
    assert sim.extruded_distance_mm == pytest.approx((90.0 + 50.0) * 2, rel=1e-4)
    assert any(e.event_type == "cut_fiber" for e in sim.events)
    assert any(e.event_type == "tool_change" for e in sim.events)
    assert any(e.event_type == "retract" for e in sim.events)
    assert "GcodeSimulation(" in sim.summary()

    for color_mode in ("feedrate", "tool", "layer", "z", "time", "type"):
        fig = ap.plot_gcode(
            sim,
            printer="prusa_xl_multi",
            color_by=color_mode,  # type: ignore[arg-type]
            slider=True,
            slider_steps=10,
        )
        assert len(fig.data) >= 5
        assert len(fig.layout.sliders) == 1

    with pytest.raises(ValueError, match="color_by must be one of"):
        ap.plot_gcode(sim, color_by="invalid_metric")  # type: ignore[arg-type]
