# hipBLASLt Implementation Pitfalls and Gotchas

Hard-won lessons from reading the rocblaslt source code. For kernel optimization agents.

---

## 1. Data Type Compatibility Rules

### Valid Type Combinations (from `validateMatmulArgs`)

The validation logic is **whitelist-based negation** -- it checks for known-invalid combos and rejects them, meaning unlisted combinations may silently pass validation but still return 0 solutions from Tensile.

**Hard rules enforced in `rocblaslt_mat_utils.hpp`:**

- `rocblaslt_compute_f32_fast_xf32`: requires ALL of A, B, C, D to be `HIP_R_32F`. Any other type combination returns `rocblaslt_status_not_implemented`.
- `rocblaslt_compute_i32`: requires either `(i8, i8, i32, i32)` or `(i8, i8, i8, i8)` for (A, B, C, D). Any other combo is rejected.
- Any type set to `HIPBLASLT_DATATYPE_INVALID` in A, B, C, or D causes immediate rejection.
- **C type must equal D type.** Validated at the `rocblaslt_matmul` entry point with `matC->type != matD->type` returning `rocblaslt_status_type_mismatch`. This is easy to miss since A and B types can differ from C/D.

**Known valid precision families (from test YAML):**

| A type | B type | C/D type | Compute type | Notes |
|--------|--------|----------|-------------|-------|
| f16 | f16 | f16 | f32 | Standard HPA half |
| bf16 | bf16 | bf16 | f32 | Standard HPA bf16 |
| f32 | f32 | f32 | f32 | Single precision |
| f32 | f32 | f32 | xf32 | TF32 fast path |
| f64 | f64 | f64 | f64 | Double precision (GEMM only, not grouped) |
| i8 | i8 | i32/i8 | i32 | Integer |
| f8/bf8/f8bf8/bf8f8 (OCP+FNUZ) | same | f32/f16/bf16/f8/bf8 | f32 | All cross-combos of f8/bf8 for A/B |
| f16 | f16 | f16 | f32 (compute_input=f8_fnuz/bf8_fnuz) | Compute input type override |
| f16/f32 | f16/f32 | f16/f32 | f32_fast_bf16 | TF32x1 intermediate |
| Mixed f8/f6/f4/bf6/bf8 | all cross-combos | f32 | f32 | Sub-byte with compute_input_type |
| MX types (f8/bf8/f6/bf6/f4) | all cross-combos | f32 | f32 | Requires block scaling (scaleA/scaleB mode 3) |

**Gotcha:** The `_matmul_desc_determine_compute_type` function silently mutates `compute_type` based on `compute_input_typeA/B`. If you set `compute_type=f32` and then set compute input types to f8 variants, the effective compute type silently becomes `f32_fast_f8` etc. This happens in `rocblaslt_matmul_desc_set_attribute` for both `COMPUTE_INPUT_TYPE_A_EXT` and `COMPUTE_INPUT_TYPE_B_EXT` -- setting EITHER one triggers re-derivation. Order of attribute setting matters.

### Scale Type Constraints

- `scale_type` at matmul_desc creation only accepts: `HIP_R_32F`, `HIP_R_64F`, `HIP_R_32I`. All other values throw `rocblaslt_status_invalid_value`.
- Scale types for A and B (`scaleAType`, `scaleBType`) must match: both Scalar, both Vector, or both Block. Mismatched scaling formats (e.g., Scalar A + Block B) return `rocblaslt_status_invalid_value` from `rocblaslt_epilogue_valid_args`.

---

## 2. Leading Dimension and Stride Constraints

### What Causes Silent Corruption vs. Explicit Errors

- **Negative batch strides** are caught: `batch_stride_a < 0` returns `rocblaslt_status_invalid_size`.
- **LDE too small** is caught: `lde < num_rows_e` returns `rocblaslt_status_invalid_value`.
- **Batch stride E too small** is caught: `batch_stride_e < num_cols_e * num_rows_e` returns invalid.
- **LDA/LDB/LDC/LDD too small for the matrix dimensions are NOT validated in the library code.** The library reads ld from the matrix layout struct and passes it straight through to Tensile. If ld < rows, you get silent memory corruption.
- **Batch count mismatch** across A, B, C, D is caught: all four must be equal and >= 1.

### Swizzle Layout Constraints

Swizzle (non-COL/ROW orders) has strict rules:

- **Swizzle A requires `opA = HIPBLAS_OP_T`** (transposed). Any other op returns invalid.
- **Swizzle B requires `opB = HIPBLAS_OP_N`** (non-transposed). Any other op returns invalid.
- Only specific order-datatype pairs are valid:
  - `HIP_R_16F`, `HIP_R_16BF` -> `HIPBLASLT_ORDER_COL16_4R8`
  - `HIP_R_8F_E4M3`, `HIP_R_8F_E4M3_FNUZ` -> `HIPBLASLT_ORDER_COL16_4R16`
  - `HIP_R_4F_E2M1` -> `HIPBLASLT_ORDER_COL16_4R32`
- **LDA/LDB are silently ignored when swizzle is enabled.** The code warns but does NOT error: "The lda parameter is ignored and disabled when swizzle_a is true."

### M/N/K Derivation

- `m = matD->m` (rows of D, NOT rows of A)
- `n = matD->n` (cols of D)
- `k = (opA == HIPBLAS_OP_N) ? matA->n : matA->m` (derived from A layout + transposition)

If you set the matrix layout dimensions inconsistently (e.g., D rows != A rows when opA=N), the library will not catch it -- Tensile will either crash or produce garbage.

---

## 3. Handle Lifetime and Device Binding

### Device is Captured at Handle Creation

The constructor `_rocblaslt_handle::_rocblaslt_handle()` calls `hipGetDevice(&device)` once. The handle permanently binds to **whichever GPU is active at creation time**. If you later switch devices with `hipSetDevice()` but reuse the same handle, Tensile will generate kernels for the wrong architecture.

### ASIC Revision Capture

`asic_rev = properties.asicRevision` is captured once. Solutions are filtered by architecture. A handle created on one GPU cannot be reused on a different ASIC revision even on the same node.

### RocRoller Environment Variable

The `HIPBLASLT_USE_ROCROLLER` env var is read once at handle construction. Values: `"1"` = use RocRoller, anything else = don't, unset = auto (`-1`). Cannot be changed after handle creation.

### Handle Synchronizer is a Raw Pointer

`handle->Synchronizer` is `void* = nullptr` by default. For grouped GEMM, the code arithmetically offsets it: `(char*)handle->Synchronizer + (409600 * i * sizeof(int))`. If the synchronizer buffer is not allocated or is too small, this causes memory corruption.

---

## 4. Epilogue Constraints

### Which Epilogues Require AUX Tensor (E)

The `is_e_enabled()` function returns true for these epilogues, meaning the E pointer **must be non-null**:

**Forward pass:**
- `RELU_AUX`, `RELU_AUX_BIAS`
- `GELU_AUX`, `GELU_AUX_BIAS`
- `CLAMP_AUX_EXT`, `CLAMP_AUX_BIAS_EXT`
- `SIGMOID`

**Backward pass:**
- `DGELU`, `DGELU_BGRAD`
- `DRELU`, `DRELU_BGRAD`

If E is required and null, you get `rocblaslt_status_invalid_pointer`.

### Which Epilogues Require Bias

`is_bias_enabled()` returns true for:
- `BIAS`, `GELU_BIAS`, `RELU_BIAS`
- `RELU_AUX_BIAS`, `GELU_AUX_BIAS`
- `DGELU_BGRAD`, `BGRADA`, `BGRADB`, `DRELU_BGRAD`
- `SWISH_BIAS_EXT`, `CLAMP_BIAS_EXT`, `CLAMP_AUX_BIAS_EXT`

If bias is required and null, you get `rocblaslt_status_invalid_pointer`.

### Gradient Epilogues

`is_grad_enabled()` returns true for:
- `DGELU`, `DGELU_BGRAD`
- `BGRADA`, `BGRADB`
- `DRELU`, `DRELU_BGRAD`

The gradient flag affects how E tensor is treated (input vs. output) and how bias gradient is computed.

### Bias Pointer Dummy Trick in Heuristic Search

During `algo_get_heuristic`, if bias is null but the epilogue requires bias, the library **temporarily sets bias to a dummy stack address** (`dummy_bias_address = true; matmul_desc->bias = &dummy_bias_address`), then resets it after the search. This means you can search for algorithms before setting the bias pointer, but you MUST set it before calling `rocblaslt_matmul`.

### E Tensor Default LDE/Stride

If `lde <= 0`, it defaults to `num_rows_e` (= m). If `stride_e <= 0`, it defaults to `lde * num_cols_e`. These defaults happen inside `rocblaslt_epilogue_valid_args`. But the validation check `lde < num_rows_e` happens AFTER the default, so providing `lde=0` is safe (it auto-sizes), but providing `lde=1` when `m>1` will fail.

### Activation Arguments

`act0` and `act1` (floats) are activation-specific parameters. They default to `0.0f`. For CLAMP, these define the min/max range. For SWISH, act0 is the beta parameter. For RELU with threshold, act0 is the threshold. These are set via `ROCBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT` / `ARG1_EXT`.

---

## 5. Solution Selection Pitfalls

### returnedAlgoCount = 0 Cases

The heuristic search can return 0 algorithms for many non-obvious reasons:

1. **No Tensile solution exists** for your exact (type, op, epilogue) combination. The library does NOT tell you why -- just returns 0 algos with `rocblaslt_status_success`.
2. **xf32 fallback**: If `compute_type = f32_fast_xf32` and the override solution doesn't support xf32, the code falls back to `f32` transparently. But if NO solution exists for either, you get 0.
3. **Workspace too small**: Solutions are filtered by `max_workspace_bytes` from the preference. If your workspace limit is 0, only workspace-free solutions are returned. Many efficient solutions require workspace.

### Three-Phase Algorithm Search

The heuristic search has three phases:

1. **Override file check** (if `HIPBLASLT_TUNING_OVERRIDE_FILE` is set) -- this solution is inserted at position 0.
2. **`getBestSolutions`** -- Tensile's GridBasedMatching and PredictionMatching. These are the primary heuristic results.
3. **`getAllSolutions`** fallback -- if fewer solutions than requested are found, it calls `getAllSolutions` and filters each through `isSolutionSupported`. This explicitly EXCLUDES GridBased and Prediction libraries to avoid duplicates.

**Gotcha:** Phase 3 uses a static mutex (`static std::mutex mtx`). In multi-threaded scenarios, this serializes all fallback searches globally.

### Deduplication Behavior

The code checks for duplicate solutions by comparing `*(int*)(algo.data)` -- the first 4 bytes of the algo data, which is the solution index. If an override solution duplicates a heuristic solution, the heuristic copy is removed and the override stays at position 0.

### Algorithm max_workspace_bytes Clamping

In `rocblaslt_matmul_impl`, the actual workspace is clamped: `workspaceSizeInBytes = std::min<size_t>(workspaceSizeInBytes, algo->max_workspace_bytes)`. If you allocate a huge workspace but the algo's `max_workspace_bytes` is small, you only use the smaller amount. The algo's max_workspace_bytes is set during heuristic search to the preference's max.

### Grouped GEMM Deprecation Warning

For grouped GEMM, `rocblaslt_algo_get_heuristic_cpp` logs: "will be deprecated for groupedgemm in the future, please use get_all_algos instead." Prefer `getAllAlgos` for grouped GEMM.

---

## 6. User-Driven Tuning Override Mechanism

### Environment Variable

Set `HIPBLASLT_TUNING_OVERRIDE_FILE=/path/to/file.csv` before handle creation. The singleton reads it once at first access.

### File Format

CSV file with **alternating header/value line pairs**. Expected format per entry:

```
transA,transB,batch_count,m,n,k,a_type,b_type,c_type,compute_type,solution_index
N,N,1,4096,4096,4096,f16_r,f16_r,f16_r,c_f32_r,42
```

Every two lines form one entry: the first line must be the header (starting with "transA"), the second line is the corresponding values.

### Key Details

- **transA/transB**: "N" for no-transpose, anything else is transposed.
- **a_type, b_type, c_type**: hipDataType strings (e.g., `f16_r`, `f32_r`, `bf16_r`, `f8_r`, `bf8_r`).
- **compute_type**: hipblasComputeType_t string (e.g., `c_f32_r`, `c_f64_r`, `c_i32_r`).
- **solution_index**: integer index into the Tensile solution table. Must be > 0 (index 0 or negative is rejected).
- **Matching is exact**: transA, transB, inputTypeA, inputTypeB, computeType, outputType, M, N, K, batchSize must ALL match for the override to apply. There is no wildcard or fuzzy matching.
- **Multiple entries** for the same problem are supported (multimap). The code tries each in order until one passes `isSolutionSupported`.
- **Duplicate solutions** in the file are silently deduplicated.
- **The file is read lazily** on first heuristic call, not at handle creation. It is only read once (guarded by `m_override.size() == 0`).

### Gotcha: Override + Heuristic Interaction

The override solution is ALWAYS placed at index 0 of the results array, pushing heuristic results to index 1+. If the override duplicates a heuristic result, the heuristic copy is removed. The override solution's workspace is validated against the preference's max.

---

## 7. Common Correctness Validation Patterns from Tests

### Three Validation Modes

Tests use three complementary checks (from `testing_matmul.hpp`):

1. **`unit_check`**: Exact element-wise match (or near-match with tolerance). Uses `unit_check_general` for bitwise equality, `near_check_general` when tolerance > 0.
2. **`norm_check`**: Frobenius norm of the error matrix. `norm_check_general('F', ...)` computes `||D_gpu - D_cpu||_F / ||D_cpu||_F`. The result is compared against a type-dependent threshold via `norm_check(norm_error, To, compute_type)`.
3. **`allclose_check`**: NumPy-style allclose with absolute and relative tolerance: `|a - b| <= atol + rtol * |b|`.

### Integer-Exact Testing Pattern

Tests with `initialization: integer_exact` use carefully constructed inputs where A, C are in {0, 1, 2} and B is in {-2, -1, 0, 1, 2}, with alpha=2, beta in {0, -2}. This ensures the GPU result must be EXACTLY equal to the CPU reference (no floating-point tolerance). This catches accumulation order bugs.

### FP16 Accumulator Probe

The `fp16_accumulator_probe` initialization fills A with near-max f16 values and B with +/-2 pairs designed to detect whether internal accumulation uses f32 (correct) or f16 (incorrect). NN-only (no transpose support). Uses `unit_check: 1` for exact match.

### Alpha=0 NaN Propagation Test

A specific test (`alpha_beta_zero_NaN`) verifies that when `alpha=0`, NaN values in A/B do not propagate to D. When alpha=NaN or beta=NaN, the values are converted to zero.

### Tolerance Calculation

Tolerance depends on data types and is per-GEMM. The test infrastructure computes it based on K dimension and data type precision. For low-precision types (f8, bf8, f4, f6), larger tolerances are needed.

### C == D (In-Place) Testing

Tests with `c_equal_d: true` verify that the operation works correctly when C and D point to the same memory. The test copies D back to C after the GPU GEMM to verify no corruption.

---

## 8. Memory Allocation Pitfalls

### Workspace Allocation

- Workspace is allocated by the **user**, not the library. The library reports required workspace size in `heuristicResult.workspaceSize`.
- Setting `max_workspace_bytes = 0` in the preference severely limits available solutions.
- If `workspace == nullptr && workspaceSizeInBytes > 0`, you get `rocblaslt_status_invalid_pointer`.
- For grouped GEMM, workspace size is `workspace_size * block_count`, where block_count is the number of GEMM groups.

### AUX Tensor (E) Sizing

E tensor dimensions match D: `num_rows_e = m`, `num_cols_e = n`. But aux_type can differ from d_type (set via `ROCBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`). If not set, `aux_type` defaults to `HIPBLASLT_DATATYPE_INVALID`, which the library uses as-is.

### ScaleAlphaVec Override

When `scaleAlphaVec` is provided (pointer mode = alpha vector), the library **silently replaces alpha with 1.0**. The original alpha is discarded. The scale alpha vector is applied per-row instead. This happens in multiple code paths (`rocblaslt_matmul_impl`, `rocblaslt_gemm_create_cpp_impl`, and construct_rocblaslt_problem).

**Stack-use-after-return bug pattern:** The code has a comment warning about this: when alpha is replaced with a stack-local `alpha_1[16]` buffer and used later during async solution search, ASAN catches stack-use-after-return. The `construct_rocblaslt_problem` path now uses `problem.alpha_owned` (an inline array) to avoid this. Be aware if modifying similar code paths.

### User Arguments for Grouped GEMM

User args (`hipblaslt_ext::UserArguments`) require both host (`hipHostMalloc`) and device (`hipMalloc`) allocations. The device allocation is `block_count * gemm_count * sizeof(UserArguments)`.

---

## 9. Batch and Grouped GEMM Constraints

### Strided Batch

- All four matrices (A, B, C, D) must have the **same batch_count** (validated).
- Batch strides must be >= 0 (negative strides are rejected).
- `batch_count` defaults to 1 in the matrix layout struct.
- Quick return: if `m == 0 || num_batches_a == 0`, returns success immediately (valid BLAS behavior). But `k == 0` is NOT a quick return -- C must still be scaled by beta.

### Grouped GEMM

- `opA`, `opB`, `compute_type`, and data types (A, B, C, D) are taken from **the first problem** (`matmul_descr[0]`) and applied to all problems in the group. If individual problems need different types, grouped GEMM cannot be used.
- Individual problems within a group CAN have different M, N, K dimensions.
- If `validArgs == rocblaslt_status_success` (quick return for m=0 etc.), the problem is **skipped entirely** in the grouped GEMM construction loop (`continue`). This means zero-sized problems are silently dropped from the group.
- The Synchronizer offset for grouped GEMM problem i is `(char*)handle->Synchronizer + (409600 * i * sizeof(int))`. This means the Synchronizer buffer must be at least `409600 * gemm_count * sizeof(int)` bytes.

### n=0 Is Allowed in Grouped GEMM

The validation explicitly notes: "we don't check n here since grouped gemm accept some n == 0". Individual problems with n=0 are valid in grouped GEMM but not in regular GEMM (where it's a quick return).

---

## 10. Environment Variables Reference

| Variable | Effect | When Read |
|----------|--------|-----------|
| `HIPBLASLT_TUNING_OVERRIDE_FILE` | Path to CSV override file for solution selection | Singleton first access |
| `HIPBLASLT_USE_ROCROLLER` | Enable/disable RocRoller code generation ("1"=on, else=off) | Handle construction |
| `HIPBLASLT_LOG_MASK` | Controls logging verbosity | Runtime |

---

## 11. Subtle Source Code Patterns to Watch

### The `rocblaslt_status_continue` Convention

Unlike typical APIs where success = 0, hipBLASLt uses a three-value status convention internally:
- `rocblaslt_status_success` (= 0): Quick return, operation is trivially done (e.g., m=0).
- `rocblaslt_status_continue`: Validation passed, proceed with the operation.
- Other values: Actual errors.

The `rocblaslt_matmul_valid_args` function returns `continue` on success. Code that checks `!= rocblaslt_status_continue` is checking for BOTH success (quick return) AND error. This is a trap if you expect the usual `!= success` pattern.

### Alpha/Beta Type Casting

`assignAlphaBeta1` and `setTo1` cast alpha/beta based on compute type, NOT on the data type of the matrices. For `compute_f64`, alpha is `double*`. For `compute_i32`, it's `int32_t*`. For everything else (including f16 compute), it's `float*`. Passing the wrong pointer type causes silent corruption of the scalar value.

### Pointer Mode Alpha Vector

When `pointermode = rocblaslt_pointer_mode_alpha_vector`, the `alpha` parameter to `rocblaslt_matmul` is reinterpreted as the `scaleAlphaVec` pointer. The actual scalar alpha is set to 1.0 internally. This is implicit and not documented in the function signature.

### Epilogue Validation Order Dependency

In `rocblaslt_matmul_valid_args`, there is a critical comment: "rocblaslt_epilogue_valid_args must to be called otherwise bias_type will be garbage value". The epilogue validation initializes `bias_type` from `matmul_desc->bias_type`. If you skip epilogue validation (e.g., by early-returning on matmul_status), bias_type remains uninitialized. The code structure ensures epilogue validation always runs, but be careful if refactoring.
