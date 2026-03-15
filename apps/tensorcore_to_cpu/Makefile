# Build TensorCore-to-CPU device plugin for Spike
ARA_DIR := $(shell git rev-parse --show-toplevel 2>/dev/null || echo $$ARA_DIR)
SPIKE_DIR := $(ARA_DIR)/install/riscv-isa-sim
CXX = g++
CXXFLAGS = -fPIC -shared -std=c++11 -O2 -Wall
INCLUDES = -I$(SPIKE_DIR)/include

DEVICE_LIB = libtensorcore_to_cpu.so

all: $(DEVICE_LIB)

$(DEVICE_LIB): device.cxx
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -ldl

clean:
	rm -f $(DEVICE_LIB)

.PHONY: all clean
