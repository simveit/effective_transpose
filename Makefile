NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = sudo $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

swizzle: swizzle.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

transpose_naive: transpose_naive.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

transpose_swizzle: transpose_swizzle.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

transpose_swizzle_batched: transpose_swizzle_batched.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)


compile_all: 
	make swizzle
	make transpose_naive
	make transpose_swizzle
	make transpose_swizzle_batched

run_all: compile_all
	./$(OUT_DIR)/swizzle
	./$(OUT_DIR)/transpose_naive
	./$(OUT_DIR)/transpose_swizzle
	./$(OUT_DIR)/transpose_swizzle_batched

clean:
	rm $(OUT_DIR)/*