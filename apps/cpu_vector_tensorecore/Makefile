# cpu_vector_tensorecore 子目录 Makefile：提示如何从 apps/ 构建裸机 ELF。
# 一键 verilate + 编译 + 仿真见 ../../hardware/all_unit.sh
.PHONY: all clean
ARA_DIR ?= $(shell git rev-parse --show-toplevel 2>/dev/null)
all:
	@echo "Bare-metal: cd $(ARA_DIR)/apps && make bin/cpu_vector_tensorecore"
clean:
	@:
