
# -------------------------------------------------------------
# Global synthesis settings
# -------------------------------------------------------------
set_db hdl_error_on_blackbox true
set_db max_cpus_per_server 8

# -------------------------------------------------------------
# Read library files
# -------------------------------------------------------------
read_mmmc $env(SYN_SCRIPTS_DIR)/mmmc.tcl
read_physical -lef "$env(LEF_TECH) $env(LEF_STDCELL)"

# -------------------------------------------------------------
# Set top module
# -------------------------------------------------------------
set nr_lanes $env(NR_LANES)
set vlen $env(VLEN)
set os_support $env(OS_SUPPORT)
elaborate -parameters [list [list NrLanes $nr_lanes] [list VLEN $vlen] [list OSSupport $os_support]] $env(TOP_MODULE)
init_design
check_design -unresolved

# -------------------------------------------------------------
# Set retime modules
# -------------------------------------------------------------

# -------------------------------------------------------------
# Set dont use cells
# -------------------------------------------------------------

# set_dont_touch tc_sram

# -------------------------------------------------------------
# Synthesize the design to target library
# -------------------------------------------------------------
set_db syn_generic_effort $env(SYN_GENERIC_EFFORT)
syn_generic  

set_db syn_map_effort $env(SYN_MAP_EFFORT)
syn_map 

set_db syn_opt_effort $env(SYN_OPT_EFFORT)
syn_opt

# -------------------------------------------------------------
# Write out data
# -------------------------------------------------------------
write_hdl -mapped > $env(SYN_DIR)/data/$env(TOP_MODULE)-mapped.v
write_sdf > $env(SYN_DIR)/data/$env(TOP_MODULE).sdf
write_sdc -view setup_view > $env(SYN_DIR)/data/constraint_setup.sdc
write_sdc -view hold_view > $env(SYN_DIR)/data/constraint_hold.sdc

# -------------------------------------------------------------
# Save the design
# -------------------------------------------------------------
write_db $env(SYN_DIR)/data/syn.db
exit 0
