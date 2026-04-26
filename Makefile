NVCC        := nvcc
NVCC_FLAGS  := -O2 -arch=sm_75 -I include
SRCS        := $(wildcard src/*.cu)
TARGETS     := $(patsubst src/%.cu, bin/%, $(SRCS))

.PHONY: all clean run

all: bin $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.cu include/*.cuh
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

run: all
	@echo "=== Bitonic Sort ===" && bin/bitonic_sort
	@echo "=== Shell Sort ===" && bin/shell_sort
	@echo "=== Odd-Even Sort ===" && bin/odd_even_sort
	@echo "=== Ranking Sort ===" && bin/ranking_sort
	@echo "=== Merge Sort ===" && bin/merge_sort

clean:
	rm -rf bin/
