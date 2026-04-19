
# -------------------------------------------------------------
# Read the design
# -------------------------------------------------------------
read_db $env(SYN_DIR)/data/syn.db

# -------------------------------------------------------------
# Generate reports
# -------------------------------------------------------------
set report_dir $env(SYN_DIR)/reports
file mkdir $report_dir

report_timing > ${report_dir}/timing.rpt
report_power -by_hierarchy -levels all > ${report_dir}/power.rpt
report_area > ${report_dir}/area.rpt
report_design_rules > ${report_dir}/drc.rpt
report_qor > ${report_dir}/qor.rpt


# -------------------------------------------------------------
# Report path group timing
# -------------------------------------------------------------
report_timing -from [ all_inputs ] -to [ all_outputs ] -nworst 1 > ${report_dir}/timing_in_to_out.rpt
report_timing -from [ all_inputs ] -to [ all_registers ] -nworst 1 > ${report_dir}/timing_in_to_reg.rpt
report_timing -from [ all_registers ] -to [ all_outputs ] -nworst 1 > ${report_dir}/timing_reg_to_out.rpt
report_timing -from [ all_registers ] -to [ all_registers ] -nworst 1 > ${report_dir}/timing_reg_to_reg.rpt

# -------------------------------------------------------------
# Save the design
# -------------------------------------------------------------
write_db $env(SYN_DIR)/data/report.db
exit 0
