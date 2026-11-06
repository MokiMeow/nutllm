# nutllm — build system
#
#   make run     # correctness checks + the benchmark table
#   make test    # correctness only (what CI gates on)
#
# -march=native lets the compiler use this machine's AVX2/FMA. For a portable
# binary, build with: make ARCH="-mavx2 -mfma"

CXX   := g++
ARCH  ?= -march=native
SANITIZE ?= 0
ifeq ($(SANITIZE),1)
OPTFLAGS := -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all
else
OPTFLAGS := -O3
endif
CXXFLAGS := -std=c++17 $(OPTFLAGS) -Wall -Wextra -Werror -pthread $(ARCH) -Iinclude

BUILD ?= build
BIN   := $(BUILD)/nutllm

SRC := $(wildcard src/*.cpp)
OBJ := $(patsubst src/%.cpp,$(BUILD)/%.o,$(SRC))
DEP := $(OBJ:.o=.d)

.PHONY: all run test bench sanitize clean

all: $(BIN)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

run: $(BIN)
	@$(BIN)

# Correctness gate: exits non-zero if any kernel disagrees with the reference.
test: $(BIN)
	@mkdir -p build # Shared scratch directory for generated test fixtures.
	@$(BIN) --test > $(BUILD)/test.out 2>&1; \
	  status=$$?; \
	  cat $(BUILD)/test.out; \
	  if [ $$status -ne 0 ]; then echo "FAILED"; exit 1; fi; \
	  grep -q FAIL $(BUILD)/test.out && { echo "FAILED"; exit 1; }; \
	  echo "all kernels agree with the reference"

# Keep sanitizer objects separate from optimized builds; use scalar fallbacks.
sanitize:
	$(MAKE) BUILD=build/sanitize ARCH="" SANITIZE=1 test

bench: $(BIN)
	@$(BIN) 1024

clean:
	rm -rf $(BUILD)

-include $(DEP)
