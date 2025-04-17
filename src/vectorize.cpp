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
  printf("Correct!\n");
}

// BM = 256, BN = 256, BK = 16, TM = 8, TN = 8
template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ __launch_bounds__((BM * BN) / (TM * TN))
void gemm_kernel(const T* __restrict__ A,
                 const T* __restrict__ B,
                 T* __restrict__ C,
                 size_t M, size_t N, size_t K) {
  __shared__ float As[BM][BK + 1];
  __shared__ float Bs[BK][BN + 1];

  const int tid = threadIdx.x;
  const int threadCol = tid % (BN / TN);
  const int threadRow = tid / (BN / TN);

  const size_t blockRow = blockIdx.y * BM;
  const size_t blockCol = blockIdx.x * BN;

  float accum[TM][TN] = {0};

  for (size_t bkIdx = 0; bkIdx < K; bkIdx += BK) {
    for (int i = tid; i < BM * BK; i += blockDim.x) {
      int row = i / BK;
      int col = i % BK;
      As[row][col] = A[(blockRow + row) * K + bkIdx + col];
    }
    for (int i = tid; i < BK * BN; i += blockDim.x) {
      int row = i / BN;
      int col = i % BN;
      Bs[row][col] = B[(bkIdx + row) * N + blockCol + col];
    }
    __syncthreads();

    for (int k = 0; k < BK; ++k) {
      float aFrag[TM];
      float bFrag[TN];
      for (int i = 0; i < TM; ++i)
        aFrag[i] = As[threadRow * TM + i][k];
      for (int j = 0; j < TN; ++j)
        bFrag[j] = Bs[k][threadCol * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
          accum[i][j] += aFrag[i] * bFrag[j];
    }
    __syncthreads();
  }

  for (int i = 0; i < TM; ++i) {
    for (int j = 0; j < TN; ++j) {
      size_t row = blockRow + threadRow * TM + i;
      size_t col = blockCol + threadCol * TN + j;
      if (row < M && col < N)
        C[row * N + col] = accum[i][j];
    }
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
  // Block dimension: (BM*BN)/(TM*TN) = (256*256)/(8*8) = 1024 threads per block.
  dim3 block((BM * BN) / (TM * TN), 1, 1);
  // Grid covers the output matrix tiled by BM x BN.
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  gemm_kernel<float, BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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

  // Initialize matrices A and B.
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

  // Allocate device memory.
  float *d_a, *d_b, *d_c;
  hipMalloc((float **)&d_a, M * K * sizeof(float));
  hipMalloc((float **)&d_b, K * N * sizeof(float));
  hipMalloc((float **)&d_c, M * N * sizeof(float));

  copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

  hipEvent_t start, stop;
  float time;
  hipEventCreate(&start);
  hipEventCreate(&stop);

  hipEventRecord(start, 0);

  // Launch with the new configuration for MI250.
  executeKernel<float, 64, 64, 16, 4, 4>(d_a, d_b, d_c, M, N, K);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time, start, stop);
  printf("Time taken for GEMM: %f ms\n", time);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  std::cout << "Performance: " << 2LL * M * N * K / (time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}