## 🚀 Matmul Benchmark on AMD MI250

**GEMM (General Matrix Multiply) Kernel Optimization Benchmark on AMD Instinct MI250 using HIP.**  
This project evaluates the performance of various GEMM kernel optimization strategies, from naive implementation to warp-level tiling, targeting the 64-thread wavefront architecture of AMD MI250.

---

### 📌 Project Goals

- Understand optimization techniques for GEMM kernels on MI250 architecture.
- Evaluate real-world performance using several strategies:
  - Global memory coalescing
  - Shared memory tiling
  - Register tiling
  - Warp-level tiling
  - Vectorized memory access
- Compare performance in GFLOP/s across multiple matrix sizes.

---

### 🏗️ Hardware Architecture

- **AMD MI250:**
  - SIMD Unit: `Wavefront` = 64 threads
  - High-speed Shared Memory (LDS)
  - Best suited for large tile/block-style parallel computation

---

### 📈 Optimization Techniques

| Step | Technique | Performance Boost (vs. Naive) | Description |
|------|-----------|-------------------------------|-------------|
| 1 | Naive Global Memory | - | Basic coalesced access only |
| 2 | Shared Memory Tiling | x3 | Uses LDS to reduce global access |
| 3 | 1D Blocktiling | x1.14–1.5 | Each thread computes multiple outputs |
| 4 | 2D Blocktiling | x2–3 | Higher arithmetic intensity with reuse |
| 5 | Vectorization | +~5% | float4 + LDS.128 usage |
| 6 | Warp Tiling | **x2.25 (best)** | Three-level parallelism strategy |

---

### ⚙️ Best Configuration

```cpp
// Tile per block
BM = 128, BN = 128, BK = 8

// Warp tile
WM = 64, WN = 32

// Thread tile
TM = 8, TN = 8

// Block size
BLOCK_SIZE = 256  // 8 warps on MI250
```

---

### 📊 Benchmark Results (GFLOP/s)

| Kernel | 1024³ | 1024x1024x128 | 8192³ |
|--------|--------|----------------|--------|
| Naive | 804.95 | 3733.44 | 698.80 |
| Shared Memory | 2724.67 | 751.33 | 4570.01 |
| 1D Blocktiling | 3127.16 | 752.68 | 7108.21 |
| 2D Blocktiling | 3483.45 | 745.32 | 10760.8 |
| Vectorized | 3430.91 | 724.09 | 11049.4 |
| **Warp Tiling** | **3365.56** | **736.50** | **24897.7** |

> ✅ Warp Tiling yields the highest performance for large matrix sizes (8192³).

---

### 📂 Source Directory Structure

```bash
.
├── src/
│   ├── naive_main.cpp
│   ├── tile-main.cpp
│   ├── tiled_ilp-1D-main.cpp
│   ├── tiled-ilp-2D-main.cpp
│   ├── vectorize.cpp
│   └── warp-tiling-main.cpp
│   └── best-gemm.cpp
├── README.md
├── matmul.ipynb
```

---

### 📚 System Requirements

- AMD MI250 or ROCm-compatible GPU
- ROCm ≥ 5.7
- GCC ≥ 9

---

### 🛠️ Build & Run

- Run bash block in notebook.

---


