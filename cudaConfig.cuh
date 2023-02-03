#ifndef CUDA_CONFIG_CUH
#define CUDA_CONFIG_CUH

// Value is required at compile time to allocate the maximum amount of device constant memory.
// You can get this value by running this code:

// cudaDeviceProp props;
// cudaGetDeviceProperties(&props, 0);
// std::cout << props.totalConstMem / sizeof(unsigned long long) << std::endl;

#define FAST_MEMORY_CONDITIONS_MAX_COUNT 8192

// These values are harder to get right but 256 threads per block usually good number to go with.
// I just picked 24 blocks per grid to get 6144 threads as it is number of CUDA cores on my RTX 3070 Ti.
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 24
#define TOTAL_THREADS_COUNT (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

#endif //CUDA_CONFIG_CUH
