# Makefile for Goldilocks CUDA kernels

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_80 -std=c++17

# Debug flags (uncomment for debugging)
# NVCC_FLAGS = -g -G -O0 -arch=sm_70 -std=c++17

# Headers
HEADERS = goldilocks.cuh poseidon2.cuh extension.cuh challenger.cuh eq_lagrange.cuh basefold.cuh

.PHONY: all clean test test-field test-poseidon2 test-extension test-challenger test-eq-lagrange test-basefold lib shared

all: goldilocks_test poseidon2_test extension_test challenger_test eq_lagrange_test basefold_test

# Goldilocks field test
goldilocks_test: goldilocks_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DGOLDILOCKS_TEST -o $@ goldilocks_kernels.cu

# Poseidon2 test
poseidon2_test: poseidon2_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DPOSEIDON2_TEST -o $@ poseidon2_kernels.cu

# Extension field test
extension_test: extension_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DEXTENSION_TEST -o $@ extension_kernels.cu

# Challenger (Fiat-Shamir) test
challenger_test: challenger_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DCHALLENGER_TEST -o $@ challenger_kernels.cu

# Eq Lagrange test
eq_lagrange_test: eq_lagrange_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DEQ_LAGRANGE_TEST -o $@ eq_lagrange_kernels.cu

# Basefold test
basefold_test: basefold_kernels.cu basefold.cuh $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -DBASEFOLD_TEST -o $@ basefold_kernels.cu

test: test-field test-poseidon2 test-extension test-challenger test-eq-lagrange test-basefold

test-field: goldilocks_test
	./goldilocks_test

test-poseidon2: poseidon2_test
	./poseidon2_test

test-extension: extension_test
	./extension_test

test-challenger: challenger_test
	./challenger_test

test-eq-lagrange: eq_lagrange_test
	./eq_lagrange_test

test-basefold: basefold_test
	./basefold_test

clean:
	rm -f goldilocks_test poseidon2_test extension_test challenger_test eq_lagrange_test basefold_test *.o *.a *.so

# Build as a library (static)
lib: goldilocks_kernels.cu poseidon2_kernels.cu extension_kernels.cu challenger_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) -c -o goldilocks.o goldilocks_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o poseidon2.o poseidon2_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o extension.o extension_kernels.cu
	$(NVCC) $(NVCC_FLAGS) -c -o challenger.o challenger_kernels.cu
	ar rcs libgoldilocks_cuda.a goldilocks.o poseidon2.o extension.o challenger.o

# Build as a shared library
shared: goldilocks_kernels.cu poseidon2_kernels.cu extension_kernels.cu challenger_kernels.cu $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) --shared -o libgoldilocks_cuda.so \
		goldilocks_kernels.cu poseidon2_kernels.cu extension_kernels.cu challenger_kernels.cu
