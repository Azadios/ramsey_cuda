#include "definitions.cuh"

extern __constant__ unsigned long long fastConditions[FAST_MEMORY_CONDITIONS_MAX_COUNT];

__device__ static inline unsigned long long getGlobalThreadIdx() {
    return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ static bool isSolution(const unsigned long fastConditionsCount,
                                  const unsigned long long candidate,
                                  const unsigned long long *slowConditions,
                                  const unsigned long slowConditionsCount) {
    for (int i = 0; i < fastConditionsCount; ++i) {
        const unsigned long long condition = fastConditions[i];
        if ((candidate & condition) == condition || (~candidate & condition) == condition) {
            return false;
        }
    }
    for (int i = 0; i < slowConditionsCount; ++i) {
        const unsigned long long condition = slowConditions[i];
        if ((candidate & condition) == condition || (~candidate & condition) == condition) {
            return false;
        }
    }
    return true;
}

__global__ void findSolutions(size_t fastConditionsCount,
                              unsigned long long *slowConditions,
                              size_t slowConditionsCount,
                              unsigned long long *solutions_d) {
    const unsigned long long globalThreadIdx = getGlobalThreadIdx();

    for (auto candidate = SOLUTIONS_START + globalThreadIdx; candidate < SOLUTIONS_END; candidate += TOTAL_THREADS_COUNT) {
        if (isSolution(fastConditionsCount, candidate, slowConditions, slowConditionsCount)) {
            solutions_d[globalThreadIdx] = candidate;
            break;
        }
    }
}

__device__ static inline unsigned long long getSymmetricCandidate(const unsigned long long candidateHalf) {
    unsigned long long reversedHalf = 0;
    for (int i = 0; i < VERTEX_COUNT / 2; ++i) {
        reversedHalf |= ((candidateHalf >> i) & 1ull) << (VERTEX_COUNT / 2 - i - 1);
    }
    return (reversedHalf << (VERTEX_COUNT / 2)) | candidateHalf;
}

__global__ void findSymmetricSolutions(unsigned long fastConditionsCount,
                                       unsigned long long *slowConditions,
                                       unsigned long slowConditionsCount,
                                       unsigned long long *solutions_d) {
    const unsigned long long globalThreadIdx = getGlobalThreadIdx();
    constexpr unsigned long long lastCandidateHalf = (1ull << ((VERTEX_COUNT - 1) / 2)) - 1;

    for (auto candidateHalf = globalThreadIdx; candidateHalf < lastCandidateHalf; candidateHalf += TOTAL_THREADS_COUNT) {
        const unsigned long long candidate = getSymmetricCandidate(candidateHalf);
        if (isSolution(fastConditionsCount, candidate, slowConditions, slowConditionsCount)) {
            solutions_d[globalThreadIdx] = candidate;
            break;
        }
    }
}
