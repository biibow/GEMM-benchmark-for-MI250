#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>

// Hàm kiểm tra kết quả sau khi tính GEMM trên host.
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

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM>
__global__ void gemm_kernel_optimized(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
  // Determine the starting tile position in the output matrix C.
  const int tileRow = blockIdx.y * BM;
  const int tileCol = blockIdx.x * BN;
  
  // Offset the pointers to the current tile.
  A += tileRow * K;
  B += tileCol;
  C += tileRow * N + tileCol;
  
  // Double-buffered shared memory for A and B tiles.
  __shared__ T As[2][BM * BK];  // BM x BK tile of A (64x8)
  __shared__ T Bs[2][BK * BN];    // BK x BN tile of B (8x64)
  
  const int tid = threadIdx.x;
  
  // For loading A, use a mapping based on BK.
  int a_row = tid / BK;       // ranges 0 to BM-1 (0..63)
  int a_col = tid % BK;       // ranges 0 to BK-1 (0..7)
  
  // For loading B, use a mapping based on BN.
  int b_row = tid / BN;       // ranges 0 to BK-1 (0..7)
  int b_col = tid % BN;       // ranges 0 to BN-1 (0..63)
  
  // For computing results, group threads by BN.
  int compute_row_group = tid / BN; // ranges 0 to BM/TM - 1 (0..7)
  int compute_col = tid % BN;         // ranges 0 to BN-1 (0..63)
  
  // Each thread accumulates TM results (consecutive rows in C).
  T threadRes[TM] = {0};
  
  int buf = 0;
  for (int bk = 0; bk < K; bk += BK) {
    // Load one element per thread into shared memory for A and B.
    As[buf][a_row * BK + a_col] = A[a_row * K + a_col];
    Bs[buf][b_row * BN + b_col] = B[b_row * N + b_col];
    __syncthreads();
    
    // Compute the partial product for the current tile.
    #pragma unroll
    for (int k = 0; k < BK; k++) {
      T bVal = Bs[buf][k * BN + compute_col];
      #pragma unroll
      for (int r = 0; r < TM; r++) {
        threadRes[r] += As[buf][((compute_row_group * TM + r) * BK) + k] * bVal;
      }
    }
    __syncthreads();
    
    // Swap buffers for double buffering.
    buf = 1 - buf;
    A += BK;
    B += BK * N;
  }
  
  // Write the accumulated results to global memory.
  for (int r = 0; r < TM; r++) {
    C[((compute_row_group * TM + r) * N) + compute_col] = threadRes[r];
  }
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N, size_t K) {
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

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
  // Block dimension: BM * BK = 64 * 8 = 512 threads (8 wavefront với mỗi wavefront 64 thread)
  dim3 block(BM * BK, 1, 1);
  // Grid được tính dựa trên kích thước tile của C (BM x BN)
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  gemm_kernel_optimized<T, BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);
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
  auto [M, N, K] = parseCmdLineArgs(argc, argv);
  float* h_a = (float*)malloc(M * K * sizeof(float));
  float* h_b = (float*)malloc(K * N * sizeof(float));
  float* h_c = (float*)malloc(M * N * sizeof(float));

  // Khởi tạo ma trận A và B.
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

  // Cấp phát bộ nhớ cho device.
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

  // Khởi chạy kernel với cấu hình tiling phù hợp cho MI250 (64 thread per warp):
  // BM = 64, BN = 64, BK = 8, TM = 8.
  executeKernel<float, 64, 64, 8, 8>(d_a, d_b, d_c, M, N, K);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);

  hipEventElapsedTime(&time, start, stop);
  printf("Time taken for GEMM: %f ms\n", time);
  hipEventDestroy(start);
  hipEventDestroy(stop);

  std::cout << "Performance: " << 2LL * M * N * K / (time * 1e-3 * 1e9) << " GFLOP/s\n";

  copyFromDeviceToHost<float>(d_c, h_c, M, N);
  // verifyResult<float>(h_a, h_b, h_c, M, N, K);
  deallocateMemory<float>(d_a, d_b, d_c);
  cleanUpDevice();
  return 0;
}