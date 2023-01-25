#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#define FAST_MEMORY_CONDITIONS_MAX_COUNT 8192

#define VERTEX_COUNT 17
#define SUBGRAPH_VERTEX_COUNT 4
#define SEARCH_SYMMETRIC_SOLUTIONS false

// only for non symmetric solutions
#define SOLUTIONS_START (1ull << (VERTEX_COUNT - SUBGRAPH_VERTEX_COUNT))
#define SOLUTIONS_END ((1ull << (VERTEX_COUNT - 1)) - 1)

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 24
#define TOTAL_THREADS_COUNT (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#define SECONDS_BETWEEN_SOLUTION_CHECKS 2

#define SUBCONDITION_FLAG 0x8000000000000000

#ifndef cudaCheckError
#define cudaCheckError()  __cudaCheckError(__FILE__, __LINE__)

#include <iostream>

inline void __cudaCheckError(const char *file, const int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#endif // cudaCheckError

__global__ void findSolutions(size_t fastConditionsCount,
                              unsigned long long *slowConditions,
                              size_t slowConditionsCount,
                              unsigned long long *solutions_d);

__global__ void findSymmetricSolutions(size_t fastConditionsCount,
                                       unsigned long long *slowConditions,
                                       size_t slowConditionsCount,
                                       unsigned long long *solutions_d);

#include <vector>
void initConditions(std::vector<unsigned long long> &conditions);
void checkForSolution(unsigned long long *solutions_h, const unsigned long long *solutions_d, cudaStream_t stream = nullptr);

[[maybe_unused]] void readConditions(std::vector<unsigned long long> &conditions, const char *filename);
[[maybe_unused]] void printConditions(std::vector<unsigned long long> &conditions, std::ostream &out = std::cout, bool binary = true);

#endif // DEFINITIONS_CUH
