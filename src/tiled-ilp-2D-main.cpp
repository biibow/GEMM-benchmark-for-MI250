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
      //printf("sum: %f, h_c[%d * K + %d]: %f\n", sum, i, j, h_c[i * K + j]);
      assert(h_c[i * N + j] == sum);
    }
  }
  printf("Correct!\n");
}
template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN))
void gemm_kernel_optimized(T* __restrict__ A, T* __restrict__ B, T* __restrict__ C, size_t M, size_t N, size_t K) {
  constexpr uint numThreads = (BM * BN) / (TM * TN);
  static_assert(numThreads <= 1024, "Too many threads per block!");

  // Compute thread row/col within the C tile
  const uint tid = threadIdx.x;
  const uint warpCol = tid % (BN / TN);
  const uint warpRow = tid / (BN / TN);

  // Block origin
  const size_t blockRow = blockIdx.y * BM;
  const size_t blockCol = blockIdx.x * BN;

  // Accumulator registers
  float threadResults[TM][TN] = {0.0f};
  float regA[TM];
  float regB[TN];

  // Shared memory (double buffered)
  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  const size_t numTilesK = K / BK;
  int buf = 0;

  // Move A, B, C pointers
  A += blockRow * K;
  B += blockCol;
  C += blockRow * N + blockCol;

  // Preload tile 0
  for (uint i = tid; i < BM * BK; i += numThreads)
    As[buf][i] = A[(i / BK) * K + (i % BK)];
  for (uint i = tid; i < BK * BN; i += numThreads)
    Bs[buf][i] = B[(i / BN) * N + (i % BN)];

  __syncthreads();

  for (uint t = 0; t < numTilesK; ++t) {
    uint nextBuf = 1 - buf;
    if (t + 1 < numTilesK) {
      const T* Aptr = A + BK;
      const T* Bptr = B + BK * N;

      for (uint i = tid; i < BM * BK; i += numThreads)
        As[nextBuf][i] = Aptr[(i / BK) * K + (i % BK)];
      for (uint i = tid; i < BK * BN; i += numThreads)
        Bs[nextBuf][i] = Bptr[(i / BN) * N + (i % BN)];
    }

    __syncthreads();

    // Compute tile
    for (uint k = 0; k < BK; ++k) {
      for (uint i = 0; i < TM; ++i)
        regA[i] = As[buf][(warpRow * TM + i) * BK + k];

      for (uint j = 0; j < TN; ++j)
        regB[j] = Bs[buf][k * BN + warpCol * TN + j];

      for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
          threadResults[i][j] += regA[i] * regB[j];
    }

    buf = nextBuf;
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  // Write result
  for (uint i = 0; i < TM; ++i)
    for (uint j = 0; j < TN; ++j) {
      size_t row = warpRow * TM + i;
      size_t col = warpCol * TN + j;
      if ((blockRow + row < M) && (blockCol + col < N))
        C[row * N + col] = threadResults[i][j];
    }
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

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  gemm_kernel_optimized<T, BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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

  executeKernel<float, 128, 128, 16, 4, 4>(d_a, d_b, d_c, M, N, K);

  hipEventRecord( stop, 0 );
  hipEventSynchronize( stop );

  hipEventElapsedTime( &time, start, stop );
  printf("Time taken for GEMM: %f ms\n", time);
  hipEventDestroy( start );
  hipEventDestroy( stop );
//
  std::cout << "Performance: " << 2LL*M*N*K/(time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}