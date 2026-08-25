# Makefile for almost-Goldilocks CUDA kernels
#
# Prime: P = 2^64 - 2^32 + 1 - 32 = 2^64 - 2^32 - 31 = 0xFFFFFFFEFFFFFFE1

NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++17

HEADERS = \
	almost_goldilocks.cuh \
	almost_extension.cuh \
	almost_eq_lagrange.cuh \
	almost_partial_eval.cuh \
	almost_fused_permute_peval.cuh \
	almost_sumcheck_prover.cuh \
	ajtai.cuh \
	ajtai_chacha8.cuh \
	ajtai_cpu_reference.cuh

.PHONY: all clean test \
	test-field test-extension test-eq-lagrange \
	test-partial-eval test-sumcheck test-fused test-ajtai

all: almost_field_test almost_extension_test almost_eq_lagrange_test \
	almost_partial_eval_test almost_sumcheck_test almost_fused_test \
	ajtai_test

# Field tests
almost_field_test: almost_field_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_field_test.cu

# Ext2 tests
almost_extension_test: almost_extension_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_extension_test.cu

# eq_lagrange tests (DP + WHT vs CPU reference)
almost_eq_lagrange_test: almost_eq_lagrange_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_eq_lagrange_kernels.cu

# Partial eval tests
almost_partial_eval_test: almost_partial_eval_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_partial_eval_test.cu

# Sumcheck tests
almost_sumcheck_test: almost_sumcheck_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_sumcheck_test.cu

# Fused permute + partial eval tests
almost_fused_test: almost_fused_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ almost_fused_test.cu

# Ajtai commitment tests
ajtai_test: ajtai_test.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -o $@ ajtai_test.cu

test: test-field test-extension test-eq-lagrange test-partial-eval test-sumcheck test-fused test-ajtai

test-field: almost_field_test
	./almost_field_test

test-extension: almost_extension_test
	./almost_extension_test

test-eq-lagrange: almost_eq_lagrange_test
	./almost_eq_lagrange_test

test-partial-eval: almost_partial_eval_test
	./almost_partial_eval_test

test-sumcheck: almost_sumcheck_test
	./almost_sumcheck_test

test-fused: almost_fused_test
	./almost_fused_test

test-ajtai: ajtai_test
	./ajtai_test

clean:
	rm -f almost_field_test almost_extension_test almost_eq_lagrange_test \
		almost_partial_eval_test almost_sumcheck_test almost_fused_test \
		ajtai_test \
		*.o *.a *.so

# Library (object archive)
lib: almost_goldilocks.o almost_extension.o
	ar rcs libalmost_goldilocks_cuda.a almost_goldilocks.o almost_extension.o

almost_goldilocks.o: almost_goldilocks_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ almost_goldilocks_kernels.cu

almost_extension.o: almost_extension_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c -o $@ almost_extension_kernels.cu
