# Makefile for Level Zero Callback Reproducer
# Mirrors chipStar Level Zero backend build settings

CXX := g++

# Level Zero SDK paths - adjust if needed
L0_INCLUDE ?= /usr/include/level_zero
L0_LIB ?= /usr/lib/x86_64-linux-gnu

# Compiler flags matching chipStar
CXXFLAGS := -std=c++17 -Wall -Wextra -g -O2
CXXFLAGS += -I$(L0_INCLUDE)

# Linker flags
LDFLAGS := -L$(L0_LIB)
LDLIBS := -lze_loader -lpthread

# Target
TARGET := callback_repro
SRCS := callback_repro.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CXXFLAGS += -DDEBUG -O0
debug: $(TARGET)

# Verbose run with L0 debug
run-debug: $(TARGET)
	ZE_ENABLE_TRACING_LAYER=1 ./$(TARGET)
