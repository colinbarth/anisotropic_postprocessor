from __future__ import annotations

import json
import pytest

import anisotropic_postprocessor as ap


def test_builtin_profiles_are_discoverable_and_loadable():
    filaments = ap.list_builtin_filaments()
    printers = ap.list_builtin_printers()
    assert "voxelpla" in filaments
    assert "generic_cf_pa" in filaments
    assert "raise3d_pro3" in printers
    assert "prusa_xl_multi" in printers
    assert "generic_reprap" in printers

    pla = ap.load_filament("voxelpla")
    assert pla.filament_type == "PLA"
    assert pla.filament_diameter == pytest.approx(1.75)
    assert pla.nozzle_temperature == pytest.approx(215.0)
    assert pla.bed_temperature == pytest.approx(55.0)
    assert pla.flow_percent == pytest.approx(100.0)
    assert not pla.is_continuous_fiber

    cf = ap.load_filament("generic_cf_pa")
    assert cf.is_continuous_fiber
    assert cf.cut_offset_mm == pytest.approx(15.0)

    r3d = ap.load_printer("raise3d_pro3")
    assert r3d.coordinate_scale == pytest.approx(1000.0)
    assert r3d.bed_size == (300.0, 300.0)
    assert r3d.max_z == pytest.approx(300.0)
    assert r3d.procedures["start"] == "raise3d_start"
    assert r3d.procedures["end"] == "raise3d_end"


def test_load_filament_from_bambu_or_prusaslicer_style_json(tmp_path):
    bambu_json = {
        "filament_settings_id": ["Bambu PLA Basic"],
        "filament_type": ["PLA"],
        "filament_diameter": ["1.75"],
        "filament_density": ["1.25"],
        "filament_flow_ratio": ["0.98"],
        "nozzle_temperature": ["220"],
        "nozzle_temperature_initial_layer": ["225"],
        "hot_plate_temp": ["60"],
        "hot_plate_temp_initial_layer": ["65"],
        "fan_min_speed": ["80%"],
        "fan_max_speed": ["100%"],
        "close_fan_the_first_x_layers": ["2"],
        "filament_retraction_length": ["0.8"],
        "custom_vendor_note": "kept in extra",
    }
    p = tmp_path / "bambu_pla.json"
    p.write_text(json.dumps(bambu_json), encoding="utf-8")

    prof = ap.load_filament(p)
    assert prof.name == "Bambu PLA Basic"
    assert prof.filament_type == "PLA"
    assert prof.extrusion_multiplier == pytest.approx(0.98)
    assert prof.flow_percent == pytest.approx(98.0)
    assert prof.nozzle_temperature == pytest.approx(220.0)
    assert prof.nozzle_temperature_initial_layer == pytest.approx(225.0)
    assert prof.bed_temperature == pytest.approx(60.0)
    assert prof.bed_temperature_initial_layer == pytest.approx(65.0)
    assert prof.fan_min_speed == pytest.approx(80.0)
    assert prof.fan_max_speed == pytest.approx(100.0)
    assert prof.close_fan_the_first_x_layers == 2
    assert prof.retract_length == pytest.approx(0.8)
    assert prof.extra["custom_vendor_note"] == "kept in extra"
    assert prof.fan_percent_for_layer(0) == 0.0
    assert prof.fan_percent_for_layer(1) == 0.0
    assert prof.fan_percent_for_layer(2) == 100.0


def test_profile_validation_rejects_invalid_values():
    with pytest.raises(ValueError, match="filament_diameter"):
        ap.FilamentProfile(name="Bad", filament_diameter=0.0)

    with pytest.raises(ValueError, match="coordinate_scale"):
        ap.PrinterProfile(name="Bad", coordinate_scale=-1.0)

    with pytest.raises(ValueError, match="extrusion_cross_section"):
        ap.PrinterProfile(name="Bad", extrusion_cross_section="triangle")

    printer = ap.load_printer("generic_reprap")  # 1 extruder, max_nozzle_temp=300
    hot_filament = ap.FilamentProfile(name="TooHot", nozzle_temperature=350.0)
    with pytest.raises(ValueError, match="max_nozzle_temp"):
        ap.validate_compatibility(printer, hot_filament)

    cold_filament = ap.FilamentProfile(name="TooCold", nozzle_temperature=120.0)
    with pytest.raises(ValueError, match="min_extrusion_temp"):
        ap.validate_compatibility(printer, cold_filament)

    valid_filament = ap.load_filament("voxelpla")
    with pytest.raises(ValueError, match="only has 1 extruder"):
        ap.validate_compatibility(printer, valid_filament, tool=1)
