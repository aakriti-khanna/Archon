# Makefile
LLAMA_DIR := internal/inference/llama
BUILD_DIR := bin

.PHONY: all clean llama archon

all: llama archon

llama:
	@echo "Building llama.cpp static libraries with Metal support via CMake..."
	# Generate build files in a 'build' directory and compile
	cd $(LLAMA_DIR) && cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j8

archon:
	@echo "Building Archon Go binary..."
	go build -o $(BUILD_DIR)/archon ./cmd/archon/main.go

clean:
	rm -rf $(BUILD_DIR)
	rm -rf $(LLAMA_DIR)/build