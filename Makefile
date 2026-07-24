# nutllm — build system
#
#   make run     # correctness checks + the benchmark table
#   make test    # correctness only (what CI gates on)
#
# -march=native lets the compiler use this machine's AVX2/FMA. For a portable
# binary, build with: make ARCH="-mavx2 -mfma"

CXX   := g++
ARCH  ?= -march=native
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra $(ARCH) -Iinclude

BUILD := build
BIN   := $(BUILD)/nutllm

SRC := $(wildcard src/*.cpp)
OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))

.PHONY: all run test bench clean

all: $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

run: $(BIN)
	@$(BIN)

# Correctness gate: exits non-zero if any kernel disagrees with the reference.
test: $(BIN)
	@$(BIN) 128 > $(BUILD)/test.out 2>&1; \
	  status=$$?; \
	  cat $(BUILD)/test.out; \
	  if [ $$status -ne 0 ]; then echo "FAILED"; exit 1; fi; \
	  grep -q FAIL $(BUILD)/test.out && { echo "FAILED"; exit 1; }; \
	  echo "all kernels agree with the reference"

bench: $(BIN)
	@$(BIN) 1024

clean:
	rm -rf $(BUILD)
