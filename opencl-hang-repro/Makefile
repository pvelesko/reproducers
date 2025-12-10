# Makefile for OpenCL Callback Hang Reproducer

CXX := g++

# OpenCL paths
OCL_INCLUDE ?= /usr/include
OCL_LIB ?= /usr/lib/x86_64-linux-gnu

# Compiler flags
CXXFLAGS := -std=c++17 -Wall -Wextra -g -O2
CXXFLAGS += -I$(OCL_INCLUDE)

# Linker flags
LDFLAGS := -L$(OCL_LIB)
LDLIBS := -lOpenCL -lpthread

# Targets
TARGETS := minimal_repro

.PHONY: all clean run

all: $(TARGETS)

minimal_repro: minimal_repro.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)

# For Aurora with different OpenCL path
aurora: OCL_LIB=/opt/aurora/25.190.0/support/libraries/khronos/default/lib64
aurora: $(TARGETS)
