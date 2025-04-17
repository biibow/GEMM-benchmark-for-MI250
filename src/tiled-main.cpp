#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      T sum = 0;
      for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
      }
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!");
}

// constexpr int warpSize = 64;

// Now require a square tile for C where each thread computes one element.
// For example, setting bM = bN = bK = 8 means one block has 8*8 = 64 threads.
template<typename T, const size_t bM, const size_t bN, const size_t bK>
__global__ void gemm_kernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  static_assert(bM == bN, "Tile row and column dimensions must be equal.");
  static_assert((bM * bK) % warpSize == 0, "Total threads per block must be a multiple of warpSize.");
  // assert(bM * bK == blockDim.x);
  // assert(bK * bN == blockDim.x);

  // Determine thread position in the tile
  const size_t threadCol = threadIdx.x % bN;      // bN threads per row
  const size_t threadRow = threadIdx.x / bN;        // There will be bK rows in the block

  // Calculate tile indices (each block computes an 8x8 tile of C)
  const size_t tileRow = blockIdx.y;
  const size_t tileCol = blockIdx.x;

  // Move pointers to the start of this tile
  d_a += tileRow * bM * K;
  d_b += tileCol * bN;
  d_c += tileRow * bM * N + tileCol * bN;

  // Allocate shared memory for the sub-tiles of A and B
  __shared__ T As[bM * bK];
  __shared__ T Bs[bK * bN];

  T tmp = 0;

  // Loop over the K dimension in chunks of bK
  for (int bk = 0; bk < K; bk += bK) {
    // Load a tile of A and B into shared memory.
    // (Ensure that these accesses are coalesced on MI250.)
    As[threadRow * bK + threadCol] = d_a[threadRow * K + threadCol];
    Bs[threadRow * bN + threadCol] = d_b[threadRow * N + threadCol];

    __syncthreads();

    // Compute the partial dot-product for this tile
    for (size_t k = 0; k < bK; k++) {
      tmp += As[threadRow * bK + k] * Bs[k * bN + threadCol];
    }

    __syncthreads();
    // Move to the next sub-tile in the K dimension
    d_a += bK;
    d_b += bK * N;
  }

  // Write the result back to global memory
  d_c[threadRow * N + threadCol] = tmp;
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N , size_t K) {
  size_t a_bytes = sizeof(T) * M * K;
  size_t b_bytes = sizeof(T) * K * N;
  hipError_t err = hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_a to d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy h_b to d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T, const size_t bM, const size_t bN, const size_t bK>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block(bM * bK, 1, 1);
  dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM, 1);
  // Ways to affect occupancy
  // 1. Changing template parameter affects the shared memory size (bM, bN, bK)
  // 2. Changing block size
  gemm_kernel<T, bM, bN, bK><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
  hipDeviceSynchronize();
}

template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
  size_t c_bytes = sizeof(T) * M * N;
  hipError_t err = hipMemcpy(h_c, d_c, c_bytes, hipMemcpyDeviceToHost);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to copy from d_c to h_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
  hipError_t err = hipFree(d_a);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_a (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_b);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_b (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = hipFree(d_c);
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to deallocate d_c (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ void cleanUpDevice() {
  hipError_t err = hipDeviceReset();
  if (err != hipSuccess) {
    fprintf(stderr, "Failed to clean up device (error code: %s)", hipGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
  int M = 1024;
  int N = 1024;
  int K = 1024;

  for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    std::string value(argv[i+1]);
    i++;
    if (option.compare("-m") == 0) {
      M = std::stoi(value);
    }
    else if (option.compare("-n") == 0) {
      N = std::stoi(value);
    }
    else if (option.compare("-k") == 0) {
      K = std::stoi(value);
    }
  }
  return {M, N, K};
}

int main(int argc, char *argv[]) {
  std::tuple<int, int, int>parsedCmdLineArgsTuple = parseCmdLineArgs(argc, argv);
  int M = std::get<0>(parsedCmdLineArgsTuple);
  int N = std::get<1>(parsedCmdLineArgsTuple);
  int K = std::get<2>(parsedCmdLineArgsTuple);

  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));

  // initialize
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      h_a[i * K + j] = rand() % 10;
    }
  }

  for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
      h_b[i * N + j] = rand() % 10;
    }
  }

  // allocate memory on device side
  float *d_a, *d_b, *d_c;
  hipMalloc((float **)&d_a, M * K * sizeof(float));
  hipMalloc((float **)&d_b, K * N * sizeof(float));
  hipMalloc((float **)&d_c, M * N * sizeof(float));

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

  hipEvent_t start, stop;
  float time;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord( start, 0 );
  //executeKernel<float, 16, 16, 16>(d_a, d_b, d_c, M, N, K);
  executeKernel<float, 32, 32, 32>(d_a, d_b, d_c, M, N, K);

  hipEventRecord( stop, 0 );
  hipEventSynchronize( stop );

  hipEventElapsedTime( &time, start, stop );
  printf("Time taken for GEMM: %f ms\n", time);
  hipEventDestroy( start );
  hipEventDestroy( stop );

  std::cout << "Performance: " << 2LL*M*N*K/(time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  // verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}