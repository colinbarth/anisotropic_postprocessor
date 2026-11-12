"""anisotropic_postprocessor: G-code generation and machine simulation for path_optimizer."""
from __future__ import annotations

from anisotropic_postprocessor import postprocessor, procedures, profiles, simulator
from anisotropic_postprocessor.postprocessor import (
    PathEvent,
    PrintPath,
    ProcedureCall,
    add_fiber_cuts,
    bed_offset,
    extrusion_length,
    generate_gcode,
    generate_gcode_layers,
    insert_at_distance,
    insert_at_index,
    path_gcode,
    write_gcode,
)
from anisotropic_postprocessor.procedures import (
    PROCEDURES,
    PrintContext,
    PrintState,
    continuous_fiber_cut,
    generic_end,
    generic_layer_change,
    generic_purge,
    generic_start,
    prusa_xl_tool_change,
    raise3d_end,
    raise3d_layer_change,
    raise3d_purge,
    raise3d_start,
    register_procedure,
    resolve_procedure,
    standard_retract,
    standard_tool_change,
    standard_travel,
    standard_unretract,
)
from anisotropic_postprocessor.profiles import (
    FilamentProfile,
    PrinterProfile,
    list_builtin_filaments,
    list_builtin_printers,
    load_filament,
    load_printer,
    validate_compatibility,
)
from anisotropic_postprocessor.simulator import (
    GcodeEvent,
    GcodeMove,
    GcodeSimulation,
    parse_gcode,
    plot_gcode,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # submodules
    "profiles",
    "procedures",
    "postprocessor",
    "simulator",
    # profiles
    "FilamentProfile",
    "PrinterProfile",
    "load_filament",
    "load_printer",
    "list_builtin_filaments",
    "list_builtin_printers",
    "validate_compatibility",
    # procedures
    "PrintState",
    "PrintContext",
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
    # postprocessor
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
    # simulator
    "GcodeMove",
    "GcodeEvent",
    "GcodeSimulation",
    "parse_gcode",
    "plot_gcode",
]
