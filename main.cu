#include <iostream>
#include <vector>
using std::vector;
#include <unistd.h>
#include "definitions.cuh"

__constant__ unsigned long long fastConditions[FAST_MEMORY_CONDITIONS_MAX_COUNT];

int main() {
    vector<unsigned long long> conditions = vector<unsigned long long>();
    initConditions(conditions);
    std::cout << "Number of unique conditions = " << conditions.size() << std::endl;

    const size_t fastConditionsCount = conditions.size() > FAST_MEMORY_CONDITIONS_MAX_COUNT ? FAST_MEMORY_CONDITIONS_MAX_COUNT : conditions.size();
    const size_t slowConditionsCount = conditions.size() - fastConditionsCount;
    const size_t fastMemorySize = fastConditionsCount * sizeof(unsigned long long);
    const size_t slowMemorySize = slowConditionsCount * sizeof(unsigned long long);

    cudaMemcpyToSymbol(fastConditions, conditions.data(), fastMemorySize);
    unsigned long long *slowConditions;
    cudaMalloc(&slowConditions, slowMemorySize);
    cudaMemcpy(slowConditions,conditions.data() + fastConditionsCount,slowMemorySize,cudaMemcpyHostToDevice);

    unsigned long long *solutions_h;
    cudaMallocHost(&solutions_h, TOTAL_THREADS_COUNT * sizeof(unsigned long long));
    unsigned long long *solutions_d;
    cudaMalloc(&solutions_d, TOTAL_THREADS_COUNT * sizeof(unsigned long long));

    cudaEvent_t event;
    cudaEventCreate(&event);
    std::cout << "Searching for solutions..." << std::endl;
#if SEARCH_SYMMETRIC_SOLUTIONS
    findSymmetricSolutions<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>
            (fastConditionsCount, slowConditions, slowConditionsCount, solutions_d);
#else
    findSolutions<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(fastConditionsCount, slowConditions, slowConditionsCount, solutions_d);
#endif
    cudaEventRecord(event);
    cudaCheckError();

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    while (cudaEventQuery(event) == cudaErrorNotReady) {
        sleep(SECONDS_BETWEEN_SOLUTION_CHECKS);
        checkForSolution(solutions_h, solutions_d, stream);
    }
    cudaStreamDestroy(stream);
    cudaEventDestroy(event);

    cudaDeviceSynchronize();
    checkForSolution(solutions_h, solutions_d);
    cudaCheckError();

    cudaFree(slowConditions);
    cudaFreeHost(solutions_h);
    cudaFree(solutions_d);

    return EXIT_SUCCESS;
}
