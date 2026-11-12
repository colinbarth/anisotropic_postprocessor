from __future__ import annotations

import pytest

import anisotropic_postprocessor as ap


def make_context() -> ap.PrintContext:
    printer = ap.load_printer("raise3d_pro3")
    filament = ap.load_filament("voxelpla")
    return ap.PrintContext(printer=printer, filament=filament)


def test_raise3d_start_and_purge_reflect_context_and_overrides():
    ctx = make_context()
    start_lines = ap.raise3d_start(ctx, nozzle_temp_extruder0=225.0)
    text = "\n".join(start_lines)
    assert "M221 T0 S100.00" in text
    assert "M140 S55.00" in text
    assert "M109 T0 S225.00" in text
    assert "M572 D0 S0.03" in text
    assert "G21" in text
    assert "G90" in text
    assert "M82" in text
    assert "G28" in text
    assert ctx.state.z == pytest.approx(0.5)

    purge_lines = ap.raise3d_purge(ctx)
    purge_text = "\n".join(purge_lines)
    assert ";TYPE:PURGE" in purge_text
    assert "G1 X140.000 E12.0000 F200" in purge_text
    assert "G1 E15.0000 F200" in purge_text
    assert "G92 E0" in purge_text
    assert ctx.state.has_moved_xy is True
    assert ctx.state.e == pytest.approx(0.0)


def test_raise3d_layer_change_and_end_clamp_to_max_z():
    ctx = make_context()
    layer_lines = ap.raise3d_layer_change(
        ctx, layer=1, z=0.4, layer_height=0.2, next_x=100.0, next_y=120.0
    )
    text = "\n".join(layer_lines)
    assert ";LAYER:1" in text
    assert ";Z:0.400" in text
    assert "M106 S255" in text
    assert "G0 F300 Z0.400" in text
    assert "G0 F9000 X100.000 Y120.000" in text

    with pytest.raises(ValueError, match="above the printer's max_z"):
        ap.raise3d_layer_change(ctx, layer=2, z=305.0, layer_height=0.2)

    # Near the top of the 300 mm build volume, Z lifts are clamped to 300.0
    end_lines = ap.raise3d_end(ctx, last_z=298.0, max_z=300.0, z_lift=4.8, final_z_lift=0.5)
    end_text = "\n".join(end_lines)
    assert "G0 F300 Z300.000 ; Lift nozzle" in end_text
    assert "G1 Z0.000 E-5.0000 Y-20.000 F9000" in end_text

    with pytest.raises(ValueError, match="above the printer's max_z"):
        ap.raise3d_end(ctx, last_z=301.0, max_z=300.0)


def test_custom_procedure_registration_and_resolution():
    def custom_wipe(ctx: ap.PrintContext, *, wipe_x: float = 15.0) -> list[str]:
        return [f"G0 X{wipe_x:.1f} ; Custom nozzle wipe"]

    ap.register_procedure("custom_wipe", custom_wipe)
    fn = ap.resolve_procedure("custom_wipe")
    assert fn is custom_wipe
    assert fn(make_context(), wipe_x=25.0) == ["G0 X25.0 ; Custom nozzle wipe"]

    with pytest.raises(KeyError, match="Unknown procedure"):
        ap.resolve_procedure("nonexistent_procedure_xyz")
