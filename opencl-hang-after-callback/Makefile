CXX := g++

CXXFLAGS := -std=c++17 -Wall -Wextra -g -O2
LDLIBS := -lOpenCL -lpthread

TARGET := main

.PHONY: all clean run

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)
