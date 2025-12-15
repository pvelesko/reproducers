all: test-printf

test-printf: main.cpp
	g++ -std=c++17 -o $@ $< -lOpenCL

run: test-printf
	./test-printf strings_bad.spv

run-good: test-printf
	./test-printf dynamic_good.spv

clean:
	rm -f test-printf
