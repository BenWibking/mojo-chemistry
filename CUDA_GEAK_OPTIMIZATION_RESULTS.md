# Original CUDA Kernel: GEAK Optimization Results

Date: 2026-07-28

## Outcome

The optimized original CUDA implementation reduces the three-run grid-64
application timer from **17.3351 s** to **15.1645 s**, a **12.52% time
reduction** or **1.143x throughput**. Process wall time improves by 12.33%.
Nsight Systems attributes the gain to the chemistry kernel: its aggregate time
over 1,000 launches falls from 15,840.2 ms to 13,655.4 ms, a 13.79%
reduction, while the preparation kernel remains approximately 34 ms.

The changes follow the priority order in AMD GEAK's
[optimization strategies](https://github.com/AMD-AGI/GEAK/blob/main/kernel_workflow/knowledge/optimization_strategies.md)
and use portable counterparts of the patterns in its
[HIP optimization guide](https://github.com/AMD-AGI/GEAK/blob/main/kernel_workflow/knowledge/hip_optimization.md):
reduce duplicated work and improve reuse before launch tuning and compiler
micro-optimization.

## Retained Changes

1. **Reduce the instruction footprint.** `PrimordialChem::rhs` is a shared
   no-inline device function instead of being duplicated at all three ROS2S
   stages. This addresses the baseline profile's dominant instruction-fetch
   stalls. CUDA executable text falls from 3,122,538 to 2,287,042 bytes,
   a 26.76% reduction.
2. **Reuse retry-invariant work.** The first-stage RHS is evaluated once beside
   the Jacobian and reused when a rejected step retries with a different
   timestep. Median grid-64 RHS calls fall from 4,629 to 4,407.
3. **Reuse expensive scalar expressions.** Each generated RHS/Jacobian call
   computes `log(abs(T))` and `sqrt(T)` once. Repeated generated expressions
   consume those values.
4. **Avoid redundant local copies and initialization.** A const view reads
   species directly from `burn_t`, an output adapter writes RHS values directly
   into the integrator array, the generated Jacobian no longer zero-initializes
   scratch values that are assigned before use, and its fully assigned output
   matrix is not cleared first.
5. **Aggregate atomics by warp.** Successful lanes ballot once and one lane
   adds the warp population to `integrated_count`. CUDA uses 32-bit active
   masks; HIP uses its 64-bit ballot and population-count intrinsics.
6. **Allow FMA in the performance build.** `--fmad=true` enables the
   contraction pattern recommended by GEAK. A documented `--fmad=false` build
   remains available for strict, bitwise regression checks.

The CUDA/HIP launch expression remains outside macros so the source stays
hipify-safe. The implementation also retains fixed-size templated arrays,
compile-time loop bounds, and coalesced per-cell grid ownership.

## Build Modes

Performance:

```bash
nvcc -x cu -std=c++20 -O3 -arch=sm_90 --expt-relaxed-constexpr \
  --fmad=true --maxrregcount=255 \
  -DPRIMORDIAL_ROS2S_ENABLE_CUDA \
  -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 \
  -I. reproducer.cpp -o reproducer_cuda
```

Strict regression:

```bash
nvcc -x cu -std=c++20 -O3 -arch=sm_90 --expt-relaxed-constexpr \
  --fmad=false --maxrregcount=255 \
  -DPRIMORDIAL_ROS2S_ENABLE_CUDA \
  -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 \
  -I. reproducer.cpp -o reproducer_cuda_strict
```

## Correctness

- A warning-enabled CPU build (`-Wall -Wextra -Wpedantic
  -Wmaybe-uninitialized`) is clean and completes the grid-1 test.
- The strict CUDA build passes grid-4 and full grid-64 comparisons against the
  untouched original CUDA binary with zero difference in every reported
  category.
- The FMA build passes the same full grid-64 repository comparator. Its maximum
  non-deuterium species relative error is `2.92286e-5`, maximum thermodynamic
  relative error is `6.50519e-6`, and maximum density relative error is
  `9.60112e-15`. The comparator separately reports large relative changes in
  floor-level deuterium-bearing values, as it does for supported
  cross-arithmetic comparisons.

## Measurement Environment

| Item | Value |
|---|---|
| GPU | NVIDIA H200 |
| Driver | 580.159.04 |
| Compute capability | 9.0 |
| CUDA module | CUDA/12.9.1 |
| Nsight Systems | 2025.1.3 |
| Source commit | `411941285954b040bca558445e553debb601a4a1` plus these worktree changes |
| Selected device | GPU 1, tied with GPU 2 as least loaded |
| Pre-timing state | 0% utilization, 541 MiB used |

The timing batch used one warm-up of each executable followed by three
counterbalanced, interleaved runs of the untouched baseline and optimized
build.

| Variant | Internal runs (s) | Mean (s) | Process mean (s) |
|---|---:|---:|---:|
| Original CUDA | 17.2979, 17.2921, 17.4154 | 17.3351 | 17.6200 |
| Optimized CUDA | 15.2283, 15.1274, 15.1378 | **15.1645** | **15.4467** |

Nsight Systems used one paired grid-64 run per variant:

| Kernel | Original total (ms) | Optimized total (ms) | Change |
|---|---:|---:|---:|
| Chemistry advance, 1,000 launches | 15,840.2 | 13,655.4 | **-13.79%** |
| Timestep preparation, 1,000 launches | 34.74 | 34.42 | -0.91% |

Nsight Compute hardware-counter collection was attempted on the selected
shared GPU, but the driver denied access with `ERR_NVGPUCTRPERM`. Compiler and
the existing baseline profile still show the intended resource movement:

| PTXAS metric | Original | Strict optimized source | FMA performance build |
|---|---:|---:|---:|
| Registers/thread | 255 | 255 | 255 |
| Stack frame (bytes/thread) | 6,296 | 6,360 | 6,320 |
| Spill stores (bytes) | 6,290 | 4,512 | 4,340 |
| Spill loads (bytes) | 9,408 | 7,464 | 7,284 |

## Screened and Rejected Variants

- Register caps of 128, 160, 192, and 224 increased spill cost and were slower
  than 255 registers.
- Blocks of 64 and 256 threads were less stable or slower than 128 threads.
- Moving the Jacobian or LU solve out of line was slower.
- Reusing the Jacobian matrix in place saved storage but forced recomputation
  after retries and regressed the grid-32 screen by about 8%.
- Packed pivot indices and compact tolerance storage were neutral to slightly
  slower.
- Rewriting the controller to reuse a cube root changed the adaptive path and
  was about 1% slower, so it was reverted.

These negative results reinforce the baseline diagnosis: this kernel is
latency- and instruction-footprint-bound, not bandwidth-bound. The retained
changes reduce issued work and redundant evaluation without replacing the
chemistry model or changing the thread-per-cell architecture.
