// Copyright 2021 ETH Zurich and University of Bologna.
// Solderpad Hardware License, Version 0.51, see LICENSE for details.
// SPDX-License-Identifier: SHL-0.51
//
// Author: Matheus Cavalcante <matheusd@iis.ee.ethz.ch>
// Date: 21/10/2020
// Description: Top level testbench module for Verilator.

module ara_tb_verilator #(
    parameter int unsigned NrLanes = 0,
    parameter int unsigned VLEN    = 0
  )(
    input  logic        clk_i,
    input  logic        rst_ni,
    output logic [63:0] exit_o
  );

  /*****************
   *  Definitions  *
   *****************/

  localparam AxiAddrWidth     = 64;
  localparam AxiWideDataWidth = tc_pkg::AXI_DATA_WIDTH;

  /*********
   *  DUT  *
   *********/

  ara_testharness #(
    .NrLanes     (NrLanes         ),
    .VLEN        (VLEN            ),
    .AxiAddrWidth(AxiAddrWidth    ),
    .AxiDataWidth(AxiWideDataWidth)
  ) dut (
    .clk_i (clk_i ),
    .rst_ni(rst_ni),
    .exit_o(exit_o)
  );

`ifdef CVA6_PC_STALL_DBG
  /**
   * 卡死定位：CVA6 commit 队首长期无法提交时，pc_commit 会稳定在阻塞指令的 PC
   *（例如等待 AXI B 的 sd 写 TensorCore MMIO）。与 commit_stage.sv 中 pc_o = commit_instr_i[0].pc 一致。
   */
  localparam int unsigned CVA6_STALL_CYCLE_THRESH = 32'd10000; 
  logic [63:0] cva6_pc_commit_d;
  assign cva6_pc_commit_d = dut.i_ara_soc.i_system.i_ariane.pc_commit;

  logic [63:0] cva6_pc_prev_q;
  int unsigned cva6_pc_stable_cycles_q;
  logic        cva6_stall_announced_q;

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      cva6_pc_prev_q           <= 64'h0;
      cva6_pc_stable_cycles_q  <= 0;
      cva6_stall_announced_q   <= 1'b0;
    end else begin
      if (cva6_pc_commit_d == cva6_pc_prev_q) begin
        if (!cva6_stall_announced_q) begin
          cva6_pc_stable_cycles_q <= cva6_pc_stable_cycles_q + 1;
        end
        if (cva6_pc_stable_cycles_q == CVA6_STALL_CYCLE_THRESH) begin
          cva6_stall_announced_q <= 1'b1;
          $display(
              "[CVA6-HANG] commit PC held at 0x%016h for >= %0d cycles (instruction not retired).",
              cva6_pc_commit_d, CVA6_STALL_CYCLE_THRESH);
          $display(
              "  Map to asm: riscv64-unknown-elf-addr2line -e <elf> -f -C 0x%016h  OR  grep in apps/bin/*.dump",
              cva6_pc_commit_d);
          $display(
              "  dbg: lsu_commit_ready=%0d commit_fu=%0d (ariane fu_t STORE=2)",
              int'(dut.i_ara_soc.i_system.i_ariane.lsu_commit_ready_ex_commit),
              int'(dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].fu));
        end
      end else begin
        cva6_pc_stable_cycles_q <= 0;
        cva6_pc_prev_q          <= cva6_pc_commit_d;
        cva6_stall_announced_q  <= 1'b0;
      end
    end
  end
`endif

  /*********
   *  EOC  *
   *********/

  always @(posedge clk_i) begin
    if (exit_o[0]) begin
      if (exit_o >> 1) begin
        $warning("Core Test ", $sformatf("*** FAILED *** (tohost = %0d)", (exit_o >> 1)));
      end else begin
        // Print vector HW runtime
        $display("[hw-cycles]: %d", int'(dut.runtime_buf_q));
        $info("Core Test ", $sformatf("*** SUCCESS *** (tohost = %0d)", (exit_o >> 1)));
      end

      $finish(exit_o >> 1);
    end
  end

endmodule : ara_tb_verilator
