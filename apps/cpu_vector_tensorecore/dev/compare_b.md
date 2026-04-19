[FA] GEMM2 P*V: MMA (M,N,K)=(64,128,64)
[FA][GEMM2_PxV] send_tensorcore_instruction: M_block=1 N_block=2 K_block=1
[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x0000000080018000 buf=0x4 transpose=0 len=127
[TC][BUF_DIN][DMA-WR] bank=0 addr=0 val=0
[TC][DMA][R0] req_addr=0x0000000080018000 buf=0x4 transpose=0 rdata[31:0]=0x00000000 rdata[7:0]=0x00
[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x000000008000c000 buf=0x0 transpose=0 len=31
[TC][BUF_A][DMA-WR] bank=0 addr=0 data16=0xc84e (k0=78 k1=-56)
[TC][DMA][R0] req_addr=0x000000008000c000 buf=0x0 transpose=0 rdata[31:0]=0x0000004e rdata[7:0]=0x4e
[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x0000000010004000 buf=0x2 transpose=0 len=31
[TC][BUF_B][DMA-WR] bank=0 addr=0 data16=0x06f9 (k0=-7 k1=6)
[TC][DMA][R0] req_addr=0x0000000010004000 buf=0x2 transpose=0 rdata[31:0]=0xfffdfbf9 rdata[7:0]=0xf9
[TC][BUF_DIN][SA-RD] bank=0 addr=0 val=0
[TC][SA][DIN] addr=0 Din00=0
[FA][TC][BUF_B][SA-RD] bank=0 addr=0 part=0 val=-4
[TC][BUF_A][SA-RD] bank=0 addr=0 part=0 val=-2
[TC][SA][AB] addr=0 A00=0 B00=0
[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x000000008001c000 buf=0x4 transpose=0 len=127
[[TC][BUF_DIN][DMA-WR] bank=0 addr=0 val=0
[TC][DMA][R0] req_addr=0x000000008001c000 buf=0x4 transpose=0 rdata[31:0]=0x00000000 rdata[7:0]=0x00
[TC][BUF_DOUT][SA-WR] bank=0 addr=0 val=-155
GE[TC][DMA][AW HS] aw_valid=1 aw_ready=1 addr=0x0000000080020000 len=127
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=-155
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=-215
[TC][DMA][W.LAST] 317830 w.last 0 -> 1 (cnt=127 len=127 w_valid=1)
[TC][DMA][W.LAST] 317832 w.last 1 -> 0 (cnt=128 len=127 w_valid=1)
M[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x000000008000c000 buf=0x1 transpose=0 len=31
[TC][DMA][R0] req_addr=0x000000008000c000 buf=0x1 transpose=0 rdata[31:0]=0x0000004e rdata[7:0]=0x4e
[TC][DMA][AR HS] ar_valid=1 ar_ready=1 addr=0x0000000010005000 buf=0x3 transpose=0 len=31
[TC][DMA][R0] req_addr=0x0000000010005000 buf=0x3 transpose=0 rdata[31:0]=0x07050301 rdata[7:0]=0x01
[TC][BUF_DIN][SA-RD] bank=0 addr=0 val=0
[TC][SA][DIN] addr=0 Din00=0
[TC][BUF_DOUT][SA-WR] bank=0 addr=0 val=-187
[TC][DMA][AW HS] aw_valid=1 aw_ready=1 addr=0x0000000080024000 len=127
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=-187
[TC][BUF_DOUT][DMA-RD] dma_addr=0 word0(Dout00)=1086
M[TC][DMA][W.LAST] 318918 w.last 0 -> 1 (cnt=127 len=127 w_valid=1)
[TC][DMA][W.LAST] 318920 w.last 1 -> 0 (cnt=128 len=127 w_valid=1)