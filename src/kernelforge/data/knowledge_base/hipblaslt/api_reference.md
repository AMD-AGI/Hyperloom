# hipBLASLt API Reference

Source headers (rocm-libraries/projects/hipblaslt/library/include/hipblaslt/):
- `hipblaslt.h` -- C API: handle, descriptors, matmul, matrix transform
- `hipblaslt-ext.hpp` -- C++ extension API: Gemm, GroupedGemm, GemmPreference, GemmTuning
- `hipblaslt-types.h` -- Basic type aliases, extended data type constants

---

## 1. Descriptor Lifecycle Pattern

Every hipBLASLt operation follows: **Create -> Set Attributes -> Use -> Destroy**.

```
hipblasLtHandle_t           handle;
hipblasLtMatmulDesc_t       matmulDesc;
hipblasLtMatrixLayout_t     layoutA, layoutB, layoutC, layoutD;
hipblasLtMatmulPreference_t pref;

// 1. Create
hipblasLtCreate(&handle);
hipblasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
hipblasLtMatrixLayoutCreate(&layoutA, dataType, rows, cols, ld);
hipblasLtMatmulPreferenceCreate(&pref);

// 2. Set attributes
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

// 3. Use
hipblasLtMatmulAlgoGetHeuristic(handle, matmulDesc, layoutA, layoutB, layoutC, layoutD, pref, reqCount, results, &retCount);
hipblasLtMatmul(handle, matmulDesc, &alpha, A, layoutA, B, layoutB, &beta, C, layoutC, D, layoutD, &results[0].algo, workspace, wsSize, stream);

// 4. Destroy (reverse order)
hipblasLtMatmulPreferenceDestroy(pref);
hipblasLtMatrixLayoutDestroy(layoutA); // ...and B, C, D
hipblasLtMatmulDescDestroy(matmulDesc);
hipblasLtDestroy(handle);
```

**Important**: A handle is NOT safe for concurrent use across multiple HIP streams. Create one handle per stream. `hipblasLtDestroy()` implicitly calls `hipDeviceSynchronize()`.

---

## 2. Enumerations

### 2.1 hipblasLtEpilogue_t -- Epilogue Modes

Controls post-GEMM fusion. Combine bias + activation in a single kernel launch.

| Value | Name | Description |
|-------|------|-------------|
| 1 | `HIPBLASLT_EPILOGUE_DEFAULT` | No post-processing (scale/quantize only) |
| 2 | `HIPBLASLT_EPILOGUE_RELU` | `x = max(x, 0)` |
| 4 | `HIPBLASLT_EPILOGUE_BIAS` | Broadcast bias vector (length = rows of D, stride = 1) |
| 6 | `HIPBLASLT_EPILOGUE_RELU_BIAS` | Bias then ReLU |
| 32 | `HIPBLASLT_EPILOGUE_GELU` | `x = GELU(x)` |
| 36 | `HIPBLASLT_EPILOGUE_GELU_BIAS` | Bias then GELU |
| 130 | `HIPBLASLT_EPILOGUE_RELU_AUX` | Output pre-ReLU result to aux buffer |
| 134 | `HIPBLASLT_EPILOGUE_RELU_AUX_BIAS` | Bias, output to aux, then ReLU |
| 136 | `HIPBLASLT_EPILOGUE_DRELU` | Gradient ReLU (requires aux input) |
| 152 | `HIPBLASLT_EPILOGUE_DRELU_BGRAD` | Gradient ReLU + bias gradient (requires aux input) |
| 160 | `HIPBLASLT_EPILOGUE_GELU_AUX` | Output pre-GELU result to aux buffer |
| 164 | `HIPBLASLT_EPILOGUE_GELU_AUX_BIAS` | Bias, output to aux, then GELU |
| 192 | `HIPBLASLT_EPILOGUE_DGELU` | Gradient GELU (requires aux input) |
| 208 | `HIPBLASLT_EPILOGUE_DGELU_BGRAD` | Gradient GELU + bias gradient (requires aux input) |
| 256 | `HIPBLASLT_EPILOGUE_BGRADA` | Bias gradient w.r.t. A + output GEMM result |
| 512 | `HIPBLASLT_EPILOGUE_BGRADB` | Bias gradient w.r.t. B + output GEMM result |
| 1024 | `HIPBLASLT_EPILOGUE_SIGMOID` | Sigmoid activation pointwise |
| 65536 | `HIPBLASLT_EPILOGUE_SWISH_EXT` | `x = Swish(x, 1)` |
| 65540 | `HIPBLASLT_EPILOGUE_SWISH_BIAS_EXT` | Bias then Swish |
| 131072 | `HIPBLASLT_EPILOGUE_CLAMP_EXT` | `x = max(alpha, min(x, beta))` (uses act arg0/arg1) |
| 131076 | `HIPBLASLT_EPILOGUE_CLAMP_BIAS_EXT` | Bias then clamp |
| 131200 | `HIPBLASLT_EPILOGUE_CLAMP_AUX_EXT` | Output pre-clamp to aux |
| 131204 | `HIPBLASLT_EPILOGUE_CLAMP_AUX_BIAS_EXT` | Bias, output to aux, then clamp |

**Pattern**: Values are bit-flags. `BIAS = 4`, so `GELU|BIAS = 32+4 = 36`. `AUX = 128`, so `RELU|AUX = 2+128 = 130`. Gradient variants: `DRELU = 136`, `DGELU = 192`. `BGRAD` adds 16 to the gradient variant.

**Forward epilogue with aux output** (`*_AUX*`): saves pre-activation values to an auxiliary buffer for use in the backward pass.
**Backward epilogues** (`D*`): consume the auxiliary buffer from the forward pass to compute activation gradients.

### 2.2 hipblasLtMatmulDescAttributes_t -- Matmul Descriptor Attributes

| Value | Name | Data Type | Default | Description |
|-------|------|-----------|---------|-------------|
| 0 | `HIPBLASLT_MATMUL_DESC_TRANSA` | `int32_t` | `HIPBLAS_OP_N` | Transpose for A |
| 1 | `HIPBLASLT_MATMUL_DESC_TRANSB` | `int32_t` | `HIPBLAS_OP_N` | Transpose for B |
| 2 | `HIPBLASLT_MATMUL_DESC_EPILOGUE` | `uint32_t` | `HIPBLASLT_EPILOGUE_DEFAULT` | Epilogue function |
| 3 | `HIPBLASLT_MATMUL_DESC_BIAS_POINTER` | `void*` | NULL | Device pointer to bias vector |
| 4 | `HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` | `int32_t` | -- | Bias vector data type (hipDataType) |
| 5 | `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` | `void*` | NULL | Scale factor for A (same type as compute type) |
| 6 | `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER` | `void*` | NULL | Scale factor for B |
| 7 | `HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER` | `void*` | NULL | Scale factor for C |
| 8 | `HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER` | `void*` | NULL | Scale factor for D |
| 9 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER` | `void*` | NULL | Scale factor for aux matrix |
| 10 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER` | `void*` | -- | Device pointer to aux buffer |
| 11 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD` | `int64_t` | -- | Leading dimension of aux buffer |
| 12 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE` | `int64_t` | -- | Batch stride of aux buffer |
| 13 | `HIPBLASLT_MATMUL_DESC_POINTER_MODE` | `int32_t` | `HIPBLASLT_POINTER_MODE_HOST` | Where alpha/beta reside |
| 14 | `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER` | `void*` | NULL | Output: max absolute value in D |
| 22 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE` | `int32_t` | INVALID (uses D type) | Aux buffer data type |
| 31 | `HIPBLASLT_MATMUL_DESC_A_SCALE_MODE` | -- | -- | Scale interpretation mode for A |
| 32 | `HIPBLASLT_MATMUL_DESC_B_SCALE_MODE` | -- | -- | Scale interpretation mode for B |
| 100 | `HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT` | -- | -- | Override compute input type for A |
| 101 | `HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT` | -- | -- | Override compute input type for B |
| 102 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT` | `float` | -- | First activation function argument |
| 103 | `HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT` | `float` | -- | Second activation function argument |

**Deprecated macros**: `HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER_VEC_EXT` and `HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER_VEC_EXT` are deprecated. Use `A_SCALE_MODE` / `B_SCALE_MODE` set to `HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F` instead.

### 2.3 hipblasLtMatmulMatrixScale_t -- Scale Modes

| Value | Name | Description |
|-------|------|-------------|
| 0 | `SCALAR_32F` | Single fp32 scalar for entire tensor (default for fp8) |
| 1 | `VEC16_UE4M3` | **Not supported yet.** Per-16-element block, stored as E4M3 |
| 2 | `VEC32_UE8M0` | Per-32-element block in innermost dim, stored as UE8M0 (MXFP) |
| 3 | `OUTER_VEC_32F` | Per-row (A) or per-column (B) fp32 vectors. A scale has M elements, B scale has N elements. Element (i,j) of A*B scaled by scaleA[i] * scaleB[j] |
| 4 | `VEC128_32F` | **Not supported yet.** Per-128-element block, fp32 |
| 5 | `BLK128x128_32F` | **Not supported yet.** Per-128x128 block, fp32 |
| 1001 | `BLK32_UE8M0_32_8_EXT` | Per-32-element block UE8M0, **pre-swizzled** layout for kernel memory access |
| 1002 | `VEC16_UE8M0_EXT` | **Not supported yet.** Per-16-element block UE8M0 |
| 1003 | `VEC32_UE4M3_EXT` | **Not supported yet.** Per-32-element block E4M3 |
| 1004 | `VEC16_UE5M3_EXT` | **Not supported yet.** Per-16-element block E5M3 |
| 1005 | `VEC32_UE5M3_EXT` | **Not supported yet.** Per-32-element block E5M3 |

**Key constraint**: `OUTER_VEC_32F` is only valid for A and B scale modes. `SCALAR_32F` is the default for FP8 types. `VEC32_UE8M0` is the standard MXFP scale mode. `BLK32_UE8M0_32_8_EXT` is its pre-swizzled variant for performance.

### 2.4 hipblasLtMatrixLayoutAttribute_t -- Matrix Layout Attributes

| Value | Name | Data Type | Default | Description |
|-------|------|-----------|---------|-------------|
| 0 | `BATCH_COUNT` | `int32_t` | 1 | Number of batches |
| 1 | `STRIDED_BATCH_OFFSET` | `int64_t` | 0 | Stride to next batch (elements) |
| 2 | `TYPE` | `uint32_t` | -- | Data type (hipDataType) |
| 3 | `ORDER` | `int32_t` | `HIPBLASLT_ORDER_COL` | Memory ordering |
| 4 | `ROWS` | `uint64_t` | -- | Number of rows |
| 5 | `COLS` | `uint64_t` | -- | Number of columns |
| 6 | `LD` | `int64_t` | -- | Leading dimension (>= ROWS for col-major) |

### 2.5 hipblasLtOrder_t -- Memory Orderings

| Value | Name | Description |
|-------|------|-------------|
| 0 | `HIPBLASLT_ORDER_COL` | Column-major. LD = stride to next column |
| 1 | `HIPBLASLT_ORDER_ROW` | Row-major. LD = stride to next row |
| 99 | `COL16_4R32` | Tiled: 32 cols x 128 rows composite tiles. Offset: `row%32 + 32*col + (row/32)*32*32`. Requires cols % 32 == 0 and rows % 128 == 0 |
| 100 | `COL16_4R16` | Tiled: 16 cols x 64 rows. Offset: `row%16 + 16*col + (row/16)*16*16`. Requires cols % 16 == 0 and rows % 64 == 0 |
| 101 | `COL16_4R8` | Tiled: 16 cols x 32 rows. Offset: `row%8 + 8*col + (row/8)*16*8`. Requires cols % 16 == 0 and rows % 32 == 0 |
| 102 | `COL16_4R4` | Tiled variant (4-row inner tiles) |
| 103 | `COL16_4R2` | Tiled variant (2-row inner tiles) |

The tiled orderings are for specialized MXFP/quantized kernel layouts. Most standard GEMM uses `ORDER_COL` (default) or `ORDER_ROW`.

### 2.6 hipblasLtPointerMode_t

| Value | Name | Description |
|-------|------|-------------|
| 0 | `HIPBLASLT_POINTER_MODE_HOST` | alpha/beta on host (default) |
| 1 | `HIPBLASLT_POINTER_MODE_DEVICE` | alpha/beta on device |
| 4 | `ALPHA_DEVICE_VECTOR_BETA_HOST` | alpha = device vector (length = rows of D), beta = host scalar |

### 2.7 hipblasLtMatmulPreferenceAttributes_t

| Value | Name | Data Type | Default | Description |
|-------|------|-----------|---------|-------------|
| 0 | `SEARCH_MODE` | `uint32_t` | -- | Heuristic search mode |
| 1 | `MAX_WORKSPACE_BYTES` | `uint64_t` | 0 | Maximum workspace memory allowed |

### 2.8 hipblasLtMatrixTransformDescAttributes_t

| Value | Name | Data Type | Default |
|-------|------|-----------|---------|
| 0 | `SCALE_TYPE` | `int32_t` | -- |
| 1 | `POINTER_MODE` | `int32_t` | `POINTER_MODE_HOST` |
| 2 | `TRANSA` | `int32_t` | `HIPBLAS_OP_N` |
| 3 | `TRANSB` | `int32_t` | `HIPBLAS_OP_N` |

---

## 3. C API Functions

### 3.1 Library Handle

```c
hipblasStatus_t hipblasLtCreate(hipblasLtHandle_t* handle);
hipblasStatus_t hipblasLtDestroy(const hipblasLtHandle_t handle);
hipblasStatus_t hipblasLtGetVersion(hipblasLtHandle_t handle, int* version);
hipblasStatus_t hipblasLtGetGitRevision(hipblasLtHandle_t handle, char* rev);
hipblasStatus_t hipblasLtGetArchName(char** archName);
```

- One handle per device per stream.
- `hipblasLtDestroy` calls `hipDeviceSynchronize` -- minimize create/destroy pairs.

### 3.2 Matrix Layout

```c
hipblasStatus_t hipblasLtMatrixLayoutCreate(
    hipblasLtMatrixLayout_t* matLayout,
    hipDataType type, uint64_t rows, uint64_t cols, int64_t ld);

hipblasStatus_t hipblasLtMatrixLayoutDestroy(const hipblasLtMatrixLayout_t matLayout);

hipblasStatus_t hipblasLtMatrixLayoutSetAttribute(
    hipblasLtMatrixLayout_t matLayout,
    hipblasLtMatrixLayoutAttribute_t attr,
    const void* buf, size_t sizeInBytes);

hipblasStatus_t hipblasLtMatrixLayoutGetAttribute(
    hipblasLtMatrixLayout_t matLayout,
    hipblasLtMatrixLayoutAttribute_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);
```

- `ld` must be >= `rows` for column-major.
- For batched GEMM, set `BATCH_COUNT` and `STRIDED_BATCH_OFFSET` after creation.

### 3.3 Matmul Descriptor

```c
hipblasStatus_t hipblasLtMatmulDescCreate(
    hipblasLtMatmulDesc_t* matmulDesc,
    hipblasComputeType_t computeType,
    hipDataType scaleType);

hipblasStatus_t hipblasLtMatmulDescDestroy(const hipblasLtMatmulDesc_t matmulDesc);

hipblasStatus_t hipblasLtMatmulDescSetAttribute(
    hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatmulDescAttributes_t attr,
    const void* buf, size_t sizeInBytes);

hipblasStatus_t hipblasLtMatmulDescGetAttribute(
    hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatmulDescAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);
```

- `computeType`: determines accumulator precision (e.g., `HIPBLAS_COMPUTE_32F`)
- `scaleType`: type of alpha/beta scalars and scale pointers

### 3.4 Matmul Preference

```c
hipblasStatus_t hipblasLtMatmulPreferenceCreate(hipblasLtMatmulPreference_t* pref);
hipblasStatus_t hipblasLtMatmulPreferenceDestroy(const hipblasLtMatmulPreference_t pref);

hipblasStatus_t hipblasLtMatmulPreferenceSetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr,
    const void* buf, size_t sizeInBytes);

hipblasStatus_t hipblasLtMatmulPreferenceGetAttribute(
    hipblasLtMatmulPreference_t pref,
    hipblasLtMatmulPreferenceAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);
```

- Default `MAX_WORKSPACE_BYTES = 0` means no workspace. Increase to allow more algorithm choices.

### 3.5 Algorithm Selection

```c
hipblasStatus_t hipblasLtMatmulAlgoGetHeuristic(
    hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t matmulDesc,
    hipblasLtMatrixLayout_t Adesc, hipblasLtMatrixLayout_t Bdesc,
    hipblasLtMatrixLayout_t Cdesc, hipblasLtMatrixLayout_t Ddesc,
    hipblasLtMatmulPreference_t pref,
    int requestedAlgoCount,
    hipblasLtMatmulHeuristicResult_t heuristicResultsArray[],
    int* returnAlgoCount);
```

- Results sorted by increasing estimated compute time.
- Increasing `requestedAlgoCount` increases wall time.
- Each result contains: `algo` (the algorithm), `workspaceSize` (bytes needed), `state` (check == `HIPBLAS_STATUS_SUCCESS`), `wavesCount` (1.0 = full GPU occupancy).

### 3.6 Matrix Multiply Execution

```c
hipblasStatus_t hipblasLtMatmul(
    hipblasLtHandle_t handle,
    hipblasLtMatmulDesc_t matmulDesc,
    const void* alpha,
    const void* A, hipblasLtMatrixLayout_t Adesc,
    const void* B, hipblasLtMatrixLayout_t Bdesc,
    const void* beta,
    const void* C, hipblasLtMatrixLayout_t Cdesc,
    void* D, hipblasLtMatrixLayout_t Ddesc,
    const hipblasLtMatmulAlgo_t* algo,
    void* workspace, size_t workspaceSizeInBytes,
    hipStream_t stream);
```

**Operation**: `D = alpha * (A * B) + beta * C`

- In-place supported: `C == D` and `Cdesc == Ddesc`.
- Out-of-place: `C != D` allowed if same data type, rows, cols, batch, ordering. Leading dimension of C can differ from D. LD of C can be 0 for broadcast.
- `algo = NULL` triggers implicit heuristic with default preferences.
- `workspace` must be 16-byte aligned.

### 3.7 Matrix Transform

```c
hipblasStatus_t hipblasLtMatrixTransformDescCreate(
    hipblasLtMatrixTransformDesc_t* transformDesc, hipDataType scaleType);
hipblasStatus_t hipblasLtMatrixTransformDescDestroy(hipblasLtMatrixTransformDesc_t transformDesc);

hipblasStatus_t hipblasLtMatrixTransformDescSetAttribute(
    hipblasLtMatrixTransformDesc_t transformDesc,
    hipblasLtMatrixTransformDescAttributes_t attr,
    const void* buf, size_t sizeInBytes);
hipblasStatus_t hipblasLtMatrixTransformDescGetAttribute(
    hipblasLtMatrixTransformDesc_t transformDesc,
    hipblasLtMatrixTransformDescAttributes_t attr,
    void* buf, size_t sizeInBytes, size_t* sizeWritten);

hipblasStatus_t hipblasLtMatrixTransform(
    hipblasLtHandle_t handle,
    hipblasLtMatrixTransformDesc_t transformDesc,
    const void* alpha, const void* A, hipblasLtMatrixLayout_t Adesc,
    const void* beta, const void* B, hipblasLtMatrixLayout_t Bdesc,
    void* C, hipblasLtMatrixLayout_t Cdesc,
    hipStream_t stream);
```

**Operation**: `C = alpha * op(A) + beta * op(B)`

Use for: changing memory order (col to row, row to tiled), scaling, layout conversion before GEMM.

---

## 4. Data Types

### 4.1 Opaque Structures

| Type | Underlying | Description |
|------|-----------|-------------|
| `hipblasLtHandle_t` | `void*` | Library context handle |
| `hipblasLtMatmulDesc_t` | `hipblasLtMatmulDescOpaque_t*` | Matmul operation descriptor |
| `hipblasLtMatrixLayout_t` | `hipblasLtMatrixLayoutOpaque_t*` | Matrix layout descriptor |
| `hipblasLtMatmulPreference_t` | `hipblasLtMatmulPreferenceOpaque_t*` | Heuristic preferences |
| `hipblasLtMatrixTransformDesc_t` | `hipblasLtMatrixTransformDescOpaque_t*` | Matrix transform descriptor |

### 4.2 Result Structures

```c
typedef struct {
    uint8_t data[16];        // Opaque algorithm data
    size_t max_workspace_bytes;
} hipblasLtMatmulAlgo_t;

typedef struct {
    hipblasLtMatmulAlgo_t algo;
    size_t workspaceSize;           // Required workspace bytes
    hipblasStatus_t state;          // Check == HIPBLAS_STATUS_SUCCESS
    float wavesCount;               // 1.0 = full GPU utilization
    int reserved[4];
} hipblasLtMatmulHeuristicResult_t;
```

### 4.3 Basic Type Aliases (hipblaslt-types.h)

| Type | Underlying |
|------|-----------|
| `hipblasLtFloat` | `float` |
| `hipblasLtHalf` | `_Float16` or 16-bit struct |
| `hipblasLtBfloat16` | `hip_bfloat16` |
| `hipblasLtInt8` | `int8_t` |
| `hipblasLtInt32` | `int32_t` |

### 4.4 Extended Data Type Constants

```c
int const HIP_R_6F_E2M3_EXT = 31;   // 6-bit float, 2-bit exponent, 3-bit mantissa
int const HIP_R_6F_E3M2_EXT = 32;   // 6-bit float, 3-bit exponent, 2-bit mantissa
int const HIP_R_4F_E2M1_EXT = 33;   // 4-bit float, 2-bit exponent, 1-bit mantissa
int const HIP_R_8F_E5M3_EXT = 34;   // 8-bit float, 5-bit exponent, 3-bit mantissa (no NaN)
```

---

## 5. Extended C++ API (hipblaslt_ext namespace)

The `hipblaslt_ext` API wraps the C API with RAII-style C++ classes. It adds GroupedGemm, tuning parameters, and a simpler workflow.

### 5.1 GemmType

```cpp
enum class GemmType {
    HIPBLASLT_GEMM         = 1,
    HIPBLASLT_GROUPED_GEMM = 2,
};
```

### 5.2 GemmPreference

```cpp
class GemmPreference {
    void setMaxWorkspaceBytes(size_t workspaceBytes);
    const size_t getMaxWorkspaceBytes() const;
};
```

### 5.3 GemmProblemType

Specifies types and operations for GEMM.

```cpp
class GemmProblemType {
    // Constructor
    GemmProblemType(hipblasOperation_t opA, hipblasOperation_t opB,
                    hipDataType typeA, hipDataType typeB,
                    hipDataType typeC, hipDataType typeD,
                    hipblasComputeType_t typeCompute);

    // Setters
    void setOpA(hipblasOperation_t op);
    void setOpB(hipblasOperation_t op);
    void setTypeA(hipDataType type);
    void setTypeB(hipDataType type);
    void setTypeC(hipDataType type);
    void setTypeD(hipDataType type);
    void setTypeCompute(hipblasComputeType_t type);
    void setOrderA(hipblasLtOrder_t order);
    void setOrderB(hipblasLtOrder_t order);

    // Getters: getOpA(), getOpB(), getTypeA()..getTypeD(), getTypeCompute(), getOrderA(), getOrderB()
};
```

### 5.4 GemmEpilogue

```cpp
class GemmEpilogue {
    void setMode(hipblasLtEpilogue_t mode);        // Default: EPILOGUE_DEFAULT
    void setBiasDataType(hipDataType biasDataType); // Only for bias epilogues
    void setAuxDataType(hipDataType auxDataType);   // Only for aux epilogues
    void setAuxLeadingDimension(int auxLD);         // Only for aux epilogues
    void setAuxBatchStride(int auxBatchStride);     // Only for aux epilogues
    void setScalingAType(hipblasLtMatmulMatrixScale_t type); // FP8 only
    void setScalingBType(hipblasLtMatmulMatrixScale_t type); // FP8 only
    void setAct0(float act0);                       // First activation arg (e.g., clamp min)
    void setAct1(float act1);                       // Second activation arg (e.g., clamp max)

    // All corresponding getters available
};
```

### 5.5 GemmTuning

Runtime tuning knobs applied on top of a selected algorithm.

```cpp
struct GemmTuning {
    void setSplitK(uint16_t splitK);  // 0 = use solution default
    void setWgm(int16_t wgm);        // 0 = use solution default workgroup mapping

    uint16_t getSplitK() const;
    int16_t getWgm() const;
};
```

### 5.6 GemmInputs

All device pointers for a GEMM problem.

```cpp
class GemmInputs {
    void setA(const void* a);
    void setB(const void* b);
    void setC(const void* c);
    void setD(const void* d);
    void setAlpha(const void* alpha);
    void setBeta(const void* beta);
    void setBias(const void* bias);
    void setScaleA(const void* scaleA);
    void setScaleB(const void* scaleB);
    void setScaleC(const void* scaleC);
    void setScaleD(const void* scaleD);
    void setScaleAux(const void* scaleAux);
    void setScaleAlphaVec(const void* scaleAlphaVec);
    void setAux(const void* aux);
    void setAmaxD(const void* amaxD);
    // All corresponding getters available
};
```

### 5.7 UserArguments (for GPU-driven grouped GEMM)

```cpp
struct UserArguments {  // __attribute__((packed))
    uint32_t m, n, batch, k;
    void *d, *c, *a, *b;
    uint32_t strideD1, strideD2, strideC1, strideC2;
    uint32_t strideA1, strideA2, strideB1, strideB2;
    int8_t alpha[16], beta[16];
    void *scaleA, *scaleB, *scaleC, *scaleD, *scaleAlphaVec, *bias;
    int biasType;
    uint32_t reserved;
    void* e;  // aux pointer
    uint32_t strideE1, strideE2;
    float act0, act1;
    int activationType;
};
```

Used with `GroupedGemm::run(void* deviceUserArgs, hipStream_t stream)` for solutions that load arguments from global memory.

### 5.8 GemmInstance (base class)

```cpp
class GemmInstance {
    // Algorithm selection
    hipblasStatus_t algoGetHeuristic(
        int requestedAlgoCount,
        const GemmPreference& pref,
        std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

    hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo, size_t& workspaceSizeInBytes);
    hipblasStatus_t isAlgoSupported(hipblasLtMatmulAlgo_t& algo, GemmTuning& tuning, size_t& workspaceSizeInBytes);

    // Workspace
    void setMaxWorkspaceBytes(size_t workspaceBytes);
    const size_t getMaxWorkspaceBytes() const;

    // Initialize kernel arguments
    hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo, void* workspace,
                               bool useUserArgs = true, hipStream_t stream = 0);
    hipblasStatus_t initialize(const hipblasLtMatmulAlgo_t& algo, GemmTuning& tuning,
                               void* workspace, bool useUserArgs = true, hipStream_t stream = 0);

    // Execute
    hipblasStatus_t run(hipStream_t stream, hipEvent_t start = nullptr, hipEvent_t stop = nullptr);

    // Introspection
    GemmType getGemmType();
    size_t getGemmCount();
    std::string getSolutionName();
    std::string getKernelName();
};
```

### 5.9 Gemm (single GEMM)

```cpp
class Gemm : public GemmInstance {
    // Constructors
    explicit Gemm(hipblasLtHandle_t handle,
                  hipblasOperation_t opA, hipblasOperation_t opB,
                  hipDataType typeA, typeB, typeC, typeD,
                  hipblasComputeType_t typeCompute);

    explicit Gemm(hipblasLtHandle_t handle,
                  hipblasLtMatmulDesc_t matmul_descr,
                  const void* alpha, const void* A, hipblasLtMatrixLayout_t matA,
                  const void* B, hipblasLtMatrixLayout_t matB,
                  const void* beta, const void* C, hipblasLtMatrixLayout_t matC,
                  void* D, hipblasLtMatrixLayout_t matD);

    // Set problem -- simple form (uses problem type from constructor)
    hipblasStatus_t setProblem(int64_t m, int64_t n, int64_t k, int64_t batch_count,
                               GemmEpilogue& epilogue, GemmInputs& inputs);

    // Set problem -- full control (explicit strides and problem type)
    hipblasStatus_t setProblem(int64_t m, n, k, batch_count,
                               int64_t lda, ldb, ldc, ldd,
                               int64_t strideA, strideB, strideC, strideD,
                               GemmEpilogue& epilogue, GemmInputs& inputs,
                               GemmProblemType& problemtype);

    // Set problem from C API descriptors
    hipblasStatus_t setProblem(hipblasLtMatmulDesc_t matmul_descr,
                               const void* alpha, const void* A, hipblasLtMatrixLayout_t matA,
                               const void* B, hipblasLtMatrixLayout_t matB,
                               const void* beta, const void* C, hipblasLtMatrixLayout_t matC,
                               void* D, hipblasLtMatrixLayout_t matD);

    GemmProblemType getProblemTypes();
};
```

### 5.10 GroupedGemm

```cpp
class GroupedGemm : public GemmInstance {
    // Constructor -- type-based
    explicit GroupedGemm(hipblasLtHandle_t handle,
                         hipblasOperation_t opA, opB,
                         hipDataType typeA, typeB, typeC, typeD,
                         hipblasComputeType_t typeCompute);

    // Constructor -- from C API descriptors (vectors)
    explicit GroupedGemm(hipblasLtHandle_t handle,
                         std::vector<hipblasLtMatmulDesc_t>& matmul_descr,
                         std::vector<void*>& alpha,
                         std::vector<void*>& A, std::vector<hipblasLtMatrixLayout_t>& matA,
                         std::vector<void*>& B, std::vector<hipblasLtMatrixLayout_t>& matB,
                         std::vector<void*>& beta,
                         std::vector<void*>& C, std::vector<hipblasLtMatrixLayout_t>& matC,
                         std::vector<void*>& D, std::vector<hipblasLtMatrixLayout_t>& matD);

    // Set problem -- vectors of m/n/k/batch_count
    hipblasStatus_t setProblem(std::vector<int64_t>& m, n, k, batch_count,
                               std::vector<GemmEpilogue>& epilogue,
                               std::vector<GemmInputs>& inputs);

    // Set problem -- full control with strides
    hipblasStatus_t setProblem(std::vector<int64_t>& m, n, k, batch_count,
                               std::vector<int64_t>& lda, ldb, ldc, ldd,
                               std::vector<int64_t>& strideA, strideB, strideC, strideD,
                               std::vector<GemmEpilogue>& epilogue,
                               std::vector<GemmInputs>& inputs,
                               GemmProblemType& problemtype);

    // Set problem from C API descriptors
    hipblasStatus_t setProblem(std::vector<hipblasLtMatmulDesc_t>& matmul_descr, ...);

    std::vector<GemmProblemType> getProblemTypes();

    // GPU-driven execution with UserArguments
    hipblasStatus_t getDefaultValueForDeviceUserArguments(void* hostDeviceUserArgs);
    hipblasStatus_t run(void* deviceUserArgs, hipStream_t stream);
};
```

### 5.11 Free Functions

```cpp
// Convert GemmType to string
std::string gemmType2String(GemmType type);

// Get all algorithms for a given type configuration (not filtered by specific problem dimensions)
hipblasStatus_t getAllAlgos(hipblasLtHandle_t handle, GemmType typeGemm,
                            hipblasOperation_t opA, opB,
                            hipDataType typeA, typeB, typeC, typeD,
                            hipblasComputeType_t typeCompute,
                            std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

// Get algorithm index (for serialization/retrieval)
int getIndexFromAlgo(hipblasLtMatmulAlgo_t& algo);

// Get human-readable names
std::string getSolutionNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo);
std::string getKernelNameFromAlgo(hipblasLtHandle_t handle, hipblasLtMatmulAlgo_t& algo);

// Retrieve algorithms by index (cross-session lookup, same library version only)
hipblasStatus_t getAlgosFromIndex(hipblasLtHandle_t handle,
                                   std::vector<int>& algoIndex,
                                   std::vector<hipblasLtMatmulHeuristicResult_t>& heuristicResults);

// Check if algorithm supports a specific problem (C API descriptors)
hipblasStatus_t matmulIsAlgoSupported(hipblasLtHandle_t handle,
                                       hipblasLtMatmulDesc_t matmulDesc,
                                       const void* alpha,
                                       hipblasLtMatrixLayout_t Adesc, Bdesc,
                                       const void* beta,
                                       hipblasLtMatrixLayout_t Cdesc, Ddesc,
                                       hipblasLtMatmulAlgo_t& algo,
                                       size_t& workspaceSizeInBytes);

// Copy matmul descriptor settings
hipblasStatus_t copyMatmul(hipblasLtMatmulDesc_t src, hipblasLtMatmulDesc_t dst);

// Check if a problem has tuned solutions available
int matmulIsTuned(hipblasLtHandle_t handle,
                  hipblasLtMatmulDesc_t matmulDesc,
                  hipblasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc);
```

---

## 6. Ext API Workflow (C++)

```cpp
using namespace hipblaslt_ext;

// 1. Create handle
hipblasLtHandle_t handle;
hipblasLtCreate(&handle);

// 2. Create Gemm instance
Gemm gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_T,
           HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F,
           HIPBLAS_COMPUTE_32F);

// 3. Configure epilogue
GemmEpilogue epilogue;
epilogue.setMode(HIPBLASLT_EPILOGUE_GELU_BIAS);
epilogue.setBiasDataType(HIP_R_16F);

// 4. Set inputs
GemmInputs inputs;
inputs.setA(d_A); inputs.setB(d_B); inputs.setC(d_C); inputs.setD(d_D);
inputs.setAlpha(&alpha); inputs.setBeta(&beta);
inputs.setBias(d_bias);

// 5. Set problem
gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

// 6. Get algorithms
GemmPreference pref;
pref.setMaxWorkspaceBytes(workspaceSize);
std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
gemm.algoGetHeuristic(10, pref, heuristicResults);

// 7. (Optional) Apply tuning
GemmTuning tuning;
tuning.setSplitK(4);
tuning.setWgm(1);

// 8. Initialize and run
gemm.initialize(heuristicResults[0].algo, tuning, d_workspace);
gemm.run(stream);
```

---

## 7. FP8 Scaling Configuration

For FP8 (E4M3/E5M2) inputs, scaling is critical:

```cpp
// Scalar scaling (default)
// A_scale and B_scale are single fp32 values on device
inputs.setScaleA(d_scaleA);
inputs.setScaleB(d_scaleB);

// Block scaling (MXFP)
epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
// Now scaleA has ceil(K/32) elements per row of A, scaleB has ceil(K/32) per column of B

// Row/column vector scaling
epilogue.setScalingAType(HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
// scaleA is an fp32 vector of length M
epilogue.setScalingBType(HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F);
// scaleB is an fp32 vector of length N

// Output scaling
inputs.setScaleD(d_scaleD);    // Scale applied to D output
inputs.setAmaxD(d_amaxD);      // Receives max|D| for dynamic quantization
```

**Constraint**: `setScalingAType` and `setScalingBType` only work when `DataTypeA == DataTypeB == FP8`.

---

## 8. Deprecated Aliases

The V2 suffixed types are deprecated aliases for the current classes:
- `GemmPreferenceV2` -> `GemmPreference`
- `GemmProblemTypeV2` -> `GemmProblemType`
- `GemmEpilogueV2` -> `GemmEpilogue`
- `GemmTuningV2` -> `GemmTuning`
- `GemmInputsV2` -> `GemmInputs`

---

## 9. Error Codes Reference

All functions return `hipblasStatus_t`:

| Status | Meaning |
|--------|---------|
| `HIPBLAS_STATUS_SUCCESS` | Operation successful |
| `HIPBLAS_STATUS_NOT_INITIALIZED` | Handle not initialized |
| `HIPBLAS_STATUS_ALLOC_FAILED` | Memory allocation failed |
| `HIPBLAS_STATUS_INVALID_VALUE` | Invalid parameter or unsupported combination |
| `HIPBLAS_STATUS_NOT_SUPPORTED` | Operation not supported on this device/config |
| `HIPBLAS_STATUS_ARCH_MISMATCH` | Cannot run on selected device |
| `HIPBLAS_STATUS_EXECUTION_FAILED` | GPU execution error |

---

## 10. Key Validity Constraints

1. **Bias vector length** must equal rows of D, packed (stride = 1).
2. **Aux buffer** is required for `*_AUX*` epilogues (forward: output, backward: input). Must set aux pointer, LD, and batch stride.
3. **Scale pointers** must match compute type precision. If NULL, scale assumed to be 1.0.
4. **Workspace alignment**: 16-byte aligned. `workspaceSizeInBytes` must be >= what the algorithm reports.
5. **Algorithm indices** are NOT portable across library versions.
6. **Tiled orderings** (`COL16_4R*`) require dimensions to be multiples of the tile size.
7. **GroupedGemm**: all problems in a group must share the same type configuration (opA, opB, typeA-D, computeType).
8. **Clamp epilogue**: uses `act_arg0` as lower bound and `act_arg1` as upper bound: `x = max(arg0, min(x, arg1))`.
