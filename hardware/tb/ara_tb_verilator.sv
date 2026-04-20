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
  localparam int unsigned TC_PORT_IDX = 3;
  localparam logic [63:0] TC_STATE_MMIO_ADDR = 64'h00000000d0001000;
  logic [63:0] cva6_pc_commit_d;
  assign cva6_pc_commit_d = dut.i_ara_soc.i_system.i_ariane.pc_commit;

  logic [63:0] cva6_pc_prev_q;
  int unsigned cva6_pc_stable_cycles_q;
  logic        cva6_stall_announced_q;
  integer      dbg_up_axi_fd;
  logic        dbg_cpu_ar_valid_prev_q;
  logic        dbg_tc_ar_valid_prev_q;
  logic        dbg_cpu_r_valid_prev_q;
  logic        dbg_tc_r_valid_prev_q;
  integer      dbg_cva6_ld_fd;
  logic        dbg_lu_req_prev_q;
  logic        dbg_lu_rvalid_prev_q;
  logic        dbg_lu_valid_prev_q;
  logic        dbg_lu_wb_prev_q;
  logic        dbg_commit_valid_prev_q;

  task automatic dbg_dump_cva6_load;
    begin
      if (dbg_cva6_ld_fd != 0) begin
        $fdisplay(
            dbg_cva6_ld_fd,
            "[CVA6-LD] t=%0t pc=0x%016h head_v=%0b head_tid=%0d head_fu=%0d lsu_commit_ready=%0b load_wb_v=%0b load_wb_tid=%0d load_wb_res=0x%016h load_wb_ex=%0b lu_state=%0d lu_valid_i=%0b lu_tid=%0d lu_vaddr=0x%016h lu_op=%0d pop_ld=%0b data_req=%0b data_gnt=%0b tag_v=%0b kill=%0b data_id=%0d rvalid=%0b rid=%0d rdata=0x%016h valid_o=%0b trans_id_o=%0d ex_o=%0b ldbuf_valid=%b ldbuf_flushed=%b last=%0d slot0_tid=%0d slot1_tid=%0d",
            $time,
            cva6_pc_commit_d,
            dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].valid,
            dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].trans_id,
            int'(dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].fu),
            dut.i_ara_soc.i_system.i_ariane.lsu_commit_ready_ex_commit,
            dut.i_ara_soc.i_system.i_ariane.load_valid_ex_id,
            dut.i_ara_soc.i_system.i_ariane.load_trans_id_ex_id,
            dut.i_ara_soc.i_system.i_ariane.load_result_ex_id,
            dut.i_ara_soc.i_system.i_ariane.load_exception_ex_id.valid,
            int'(dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.state_q),
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.valid_i,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.lsu_ctrl_i.trans_id,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.lsu_ctrl_i.vaddr,
            int'(dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.lsu_ctrl_i.operation),
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.pop_ld,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.data_req,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_gnt,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.tag_valid,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.kill_req,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.data_id,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rvalid,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rid,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rdata,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.valid_o,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.trans_id_o,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ex_o.valid,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ldbuf_valid_q,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ldbuf_flushed_q,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ldbuf_last_id_q,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ldbuf_q[0].trans_id,
            dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.ldbuf_q[1].trans_id);
        $fflush(dbg_cva6_ld_fd);
      end
    end
  endtask

  initial begin
    dbg_up_axi_fd = $fopen(
        "/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/data_log/upstream_axi.log", "w");
    if (dbg_up_axi_fd == 0) begin
      $display("[UP-AXI] ERROR: cannot open .../data_log/upstream_axi.log");
    end else begin
      $fdisplay(dbg_up_axi_fd,
          "[UP-AXI] CPU=ariane_axi_req/resp, TC=periph_wide_axi_req[3]/resp; trace ar_valid/r_valid edges");
      $fflush(dbg_up_axi_fd);
    end

    dbg_cva6_ld_fd = $fopen(
        "/home/zhoujinwei/pulp/ara/apps/cpu_vector_tensorecore/data_log/cva6_load_debug.log",
        "w");
    if (dbg_cva6_ld_fd == 0) begin
      $display("[CVA6-LD] ERROR: cannot open .../data_log/cva6_load_debug.log");
    end else begin
      $fdisplay(dbg_cva6_ld_fd,
          "[CVA6-LD] trace commit/load_unit around MMIO load to 0x%016h", TC_STATE_MMIO_ADDR);
      $fflush(dbg_cva6_ld_fd);
    end
  end

  final begin
    if (dbg_up_axi_fd != 0) begin
      $fclose(dbg_up_axi_fd);
    end
    if (dbg_cva6_ld_fd != 0) begin
      $fclose(dbg_cva6_ld_fd);
    end
  end

  always_ff @(posedge clk_i) begin
    if (!rst_ni) begin
      cva6_pc_prev_q           <= 64'h0;
      cva6_pc_stable_cycles_q  <= 0;
      cva6_stall_announced_q   <= 1'b0;
      dbg_cpu_ar_valid_prev_q  <= 1'b0;
      dbg_tc_ar_valid_prev_q   <= 1'b0;
      dbg_cpu_r_valid_prev_q   <= 1'b0;
      dbg_tc_r_valid_prev_q    <= 1'b0;
      dbg_lu_req_prev_q        <= 1'b0;
      dbg_lu_rvalid_prev_q     <= 1'b0;
      dbg_lu_valid_prev_q      <= 1'b0;
      dbg_lu_wb_prev_q         <= 1'b0;
      dbg_commit_valid_prev_q  <= 1'b0;
    end else begin
      if (dbg_up_axi_fd != 0) begin
        if (dut.i_ara_soc.i_system.ariane_axi_req.ar_valid != dbg_cpu_ar_valid_prev_q) begin
          $fdisplay(dbg_up_axi_fd,
              "[UP-AXI][CPU][AR^] t=%0t %0b->%0b ar_ready=%0b addr=0x%016h id=%0d",
              $time, dbg_cpu_ar_valid_prev_q, dut.i_ara_soc.i_system.ariane_axi_req.ar_valid,
              dut.i_ara_soc.i_system.ariane_axi_resp.ar_ready,
              dut.i_ara_soc.i_system.ariane_axi_req.ar.addr,
              dut.i_ara_soc.i_system.ariane_axi_req.ar.id);
          $fflush(dbg_up_axi_fd);
        end
        if (dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar_valid != dbg_tc_ar_valid_prev_q) begin
          $fdisplay(dbg_up_axi_fd,
              "[UP-AXI][TC ][AR^] t=%0t %0b->%0b ar_ready=%0b addr=0x%016h id=%0d",
              $time, dbg_tc_ar_valid_prev_q, dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar_valid,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].ar_ready,
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar.addr,
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar.id);
          $fflush(dbg_up_axi_fd);
        end
        if (dut.i_ara_soc.i_system.ariane_axi_resp.r_valid != dbg_cpu_r_valid_prev_q) begin
          $fdisplay(dbg_up_axi_fd,
              "[UP-AXI][CPU][R^ ] t=%0t %0b->%0b r_ready=%0b r_last=%0b r_resp=%0d low64=0x%016h",
              $time, dbg_cpu_r_valid_prev_q, dut.i_ara_soc.i_system.ariane_axi_resp.r_valid,
              dut.i_ara_soc.i_system.ariane_axi_req.r_ready,
              dut.i_ara_soc.i_system.ariane_axi_resp.r.last,
              dut.i_ara_soc.i_system.ariane_axi_resp.r.resp,
              dut.i_ara_soc.i_system.ariane_axi_resp.r.data);
          $fflush(dbg_up_axi_fd);
        end
        if (dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r_valid != dbg_tc_r_valid_prev_q) begin
          $fdisplay(dbg_up_axi_fd,
              "[UP-AXI][TC ][R^ ] t=%0t %0b->%0b r_ready=%0b r_last=%0b r_resp=%0d low64=0x%016h",
              $time, dbg_tc_r_valid_prev_q, dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r_valid,
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].r_ready,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r.last,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r.resp,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r.data[63:0]);
          $fflush(dbg_up_axi_fd);
        end
        dbg_cpu_ar_valid_prev_q <= dut.i_ara_soc.i_system.ariane_axi_req.ar_valid;
        dbg_tc_ar_valid_prev_q  <= dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar_valid;
        dbg_cpu_r_valid_prev_q  <= dut.i_ara_soc.i_system.ariane_axi_resp.r_valid;
        dbg_tc_r_valid_prev_q   <= dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r_valid;
      end

      if (dbg_cva6_ld_fd != 0) begin
        if (((dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.lsu_ctrl_i.vaddr
              == TC_STATE_MMIO_ADDR)
             || dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rvalid
             || dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.valid_o
             || dut.i_ara_soc.i_system.i_ariane.load_valid_ex_id
             || (cva6_pc_commit_d == 64'h000000008000025a))
            && ((dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.data_req
                 != dbg_lu_req_prev_q)
                || (dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rvalid
                    != dbg_lu_rvalid_prev_q)
                || (dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.valid_o
                    != dbg_lu_valid_prev_q)
                || (dut.i_ara_soc.i_system.i_ariane.load_valid_ex_id != dbg_lu_wb_prev_q)
                || (dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].valid
                    != dbg_commit_valid_prev_q))) begin
          dbg_dump_cva6_load();
        end

        dbg_lu_req_prev_q <= dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_o.data_req;
        dbg_lu_rvalid_prev_q
            <= dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.req_port_i.data_rvalid;
        dbg_lu_valid_prev_q <= dut.i_ara_soc.i_system.i_ariane.ex_stage_i.lsu_i.i_load_unit.valid_o;
        dbg_lu_wb_prev_q    <= dut.i_ara_soc.i_system.i_ariane.load_valid_ex_id;
        dbg_commit_valid_prev_q <= dut.i_ara_soc.i_system.i_ariane.commit_instr_id_commit[0].valid;
      end

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
          $display(
              "  cpu_axi: ar_v=%0b ar_rdy=%0b ar_addr=0x%016h r_v=%0b r_rdy=%0b r_last=%0b",
              dut.i_ara_soc.i_system.ariane_axi_req.ar_valid,
              dut.i_ara_soc.i_system.ariane_axi_resp.ar_ready,
              dut.i_ara_soc.i_system.ariane_axi_req.ar.addr,
              dut.i_ara_soc.i_system.ariane_axi_resp.r_valid,
              dut.i_ara_soc.i_system.ariane_axi_req.r_ready,
              dut.i_ara_soc.i_system.ariane_axi_resp.r.last);
          $display(
              "  tc_axi : ar_v=%0b ar_rdy=%0b ar_addr=0x%016h r_v=%0b r_rdy=%0b r_last=%0b low64=0x%016h",
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar_valid,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].ar_ready,
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].ar.addr,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r_valid,
              dut.i_ara_soc.periph_wide_axi_req[TC_PORT_IDX].r_ready,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r.last,
              dut.i_ara_soc.periph_wide_axi_resp[TC_PORT_IDX].r.data[63:0]);
          if (dbg_cva6_ld_fd != 0) begin
            $fdisplay(dbg_cva6_ld_fd, "[CVA6-LD][HANG] t=%0t", $time);
            $fflush(dbg_cva6_ld_fd);
          end
          dbg_dump_cva6_load();
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
