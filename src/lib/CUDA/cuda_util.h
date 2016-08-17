#ifndef CUDA_DEFINES_H_
#define CUDA_DEFINES_H_

static const int CUDA_BLOCK_DIM_SIZE = 16;
static const int CUDA_BLOCK_SIZE = CUDA_BLOCK_DIM_SIZE*CUDA_BLOCK_DIM_SIZE;

#define CUDA_CHECK(ans){cudaAssert((ans), __FILE__, __LINE__);}

inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(code);
    }
}

#endif
