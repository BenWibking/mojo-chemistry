# Structured CUDA Kernel: GEAK Optimization Results

Date: 2026-07-28

## Outcome

Starting from the pure-CUDA structured kernel, the retained changes reduce the
three-run grid-64 application timer from **35.6740 s** to **29.9128 s**. That
is a **16.15% time reduction** or **1.193x speedup**. The complete 262,144-cell
final state remains bitwise identical to the starting structured CUDA binary.

Nsight Systems attributes the gain to the requested kernel: aggregate
structured chemistry time over 1,000 launches falls from **34,047.4 ms** to
**28,340.5 ms**, a **16.76% reduction**, while timestep preparation remains
approximately 37.5 ms.

The implementation follows GEAK's bottleneck-first
[optimization strategy catalog](https://github.com/AMD-AGI/GEAK/blob/main/kernel_workflow/knowledge/optimization_strategies.md)
and applies the warp cooperation, shared-memory lifetime, branch reduction,
and compile-time specialization patterns from its
[HIP optimization guide](https://github.com/AMD-AGI/GEAK/blob/main/kernel_workflow/knowledge/hip_optimization.md).

## Retained Changes

1. **Run the chemistry DAG in one wave.** All eight physical warps now execute
   one logical DAG slice concurrently. The original three-plus-five schedule
   required an intermediate CTA barrier and left part of the block idle in
   each wave.
2. **Remove the serialized exchange producer.** Raising the generated exchange
   threshold to 1,400 selects recomputation over a warp-0 producer. This
   removes two producer barriers and avoids serializing 106 base definitions.
   Exchange storage falls from 15 `Float64` values per cell to zero.
3. **Bound generated live ranges.** The generator splits the DAG into
   416-definition no-inline regions. CUDA now preserves those boundaries
   instead of force-inlining every generated helper. Base scratch falls from
   531 to 190 `Float64` values per cell, dynamic shared memory falls from
   230,576 to 139,696 bytes, and executable text falls from 9,482,826 to
   5,160,968 bytes.
4. **Use cooperative warp primitives.** Eight-lane cell groups cooperatively
   clamp and copy species, search LU pivots with deterministic shuffle
   reductions, and copy accepted candidates. The first warp uses ballots for
   the participant count and singular detection and a shuffle reduction for
   the tile error.
5. **Specialize generated policy at compile time.** Generated CUDA headers
   publish scratch, exchange, and concurrent-warp metadata. `if constexpr`
   removes the empty producer and selects either the legacy two-wave path or
   the tuned single-wave path without runtime policy branches.
6. **Keep strict arithmetic.** The generated chemistry continues to use
   round-to-nearest multiply operations and the build uses `--fmad=false`.
   This preserves the adaptive trajectory exactly.

## Build

```bash
python3 tools/slice_dag.py reproducer.mojo --output-dir generated \
  --threshold 1400 \
  --shared-region-definitions 416 \
  --shared-wave-warps 8

nvcc -x cu -std=c++20 -O3 -arch=sm_90 \
  --expt-relaxed-constexpr --fmad=false --maxrregcount=255 \
  -DPRIMORDIAL_ROS2S_ENABLE_CUDA \
  -DPRIMORDIAL_ROS2S_CUDA_STRUCTURED \
  -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 \
  -I. reproducer.cpp -o reproducer_cuda_structured
```

## Correctness

- Grid 4 passes against the starting structured CUDA reference with zero error
  in every reported species, thermodynamic, and density category.
- Full grid 64 passes against the same reference with zero error in every
  category and identical ROS2S work counters.
- The final strict executable is byte-for-byte the executable used for the
  authoritative timing and profile.
- An opt-in FMA experiment was rejected because it changed the adaptive
  trajectory and failed the existing grid-4 comparator.

## Measurement Environment

| Item | Value |
|---|---|
| GPU | NVIDIA H200 |
| Driver | 580.159.04 |
| Compute capability | 9.0 |
| CUDA module/toolkit | CUDA/12.9.1 / 12.9.86 |
| Nsight Systems | 2025.1.3 |
| Nsight Compute | 2025.2.1 |
| Source commit | `08d84908a92475e76c7f0d8685f0e4d5e651c23f` plus these worktree changes |
| Selected device | GPU 2, least loaded |
| Pre-timing state | 0% utilization, 541 MiB used |

The authoritative batch used one warm-up per executable followed by three
counterbalanced runs on the same device. "Internal" is the application's GPU
integration timer; "process" is `/usr/bin/time -p` real time.

| Variant | Internal runs (s) | Internal mean (s) | Process mean (s) |
|---|---:|---:|---:|
| Starting structured CUDA | 35.7723, 35.5968, 35.6530 | 35.6740 | 35.9267 |
| GEAK-optimized structured CUDA | 29.9142, 29.9532, 29.8710 | **29.9128** | **30.1700** |

Nsight Systems used one grid-64 run per variant:

| Kernel | Starting total (ms) | Optimized total (ms) | Change |
|---|---:|---:|---:|
| Structured chemistry, 1,000 launches | 34,047.4 | 28,340.5 | **-16.76%** |
| Timestep preparation, 1,000 launches | 37.69 | 37.45 | -0.63% |

## Resource Movement

| Metric | Starting | Optimized |
|---|---:|---:|
| Concurrent DAG warps | 3 then 5 | 8 |
| Base scratch slots/cell | 531 | 190 |
| Exchange slots/cell | 15 | 0 |
| Dynamic shared memory/CTA | 230,576 B | 139,696 B |
| Executable text | 9,482,826 B | 5,160,968 B |
| Entry registers/thread | 255 | 255 |
| Entry stack frame/thread | 1,112 B | 640 B |
| Entry spill stores | 5,288 B | 32 B |
| Entry spill loads | 8,552 B | 52 B |

The optimized generated regions are separate device functions, so PTXAS also
reports their local spill traffic separately; the largest region reports
872-byte stores and 1,432-byte loads. The end-to-end timing includes that cost.
Nsight Compute hardware-counter collection was attempted, but the shared-node
driver denied access with `ERR_NVGPUCTRPERM`.

## Screened and Rejected Variants

- Force-inlined bounded regions increased spills and were slower.
- No-inline region sizes from 64 through 544 definitions were screened;
  416 was the best stable grid-32 choice.
- Keeping one or more exchange values retained producer serialization and was
  slower than the zero-exchange schedule.
- A two-wave zero-exchange schedule was faster than the starting kernel but
  slower than dispatching all eight slices concurrently.
- Cost-balanced output reassignment was 9.3% slower than the retained
  unbalanced mapping.
- Fused multiply-add contraction failed the correctness comparator and was
  removed.

## Validation

- `python3 -m unittest tests.test_slice_dag`: 13 passed.
- `pixi run mojo run -I . tests/test_slice_dag.mojo`: 1 passed.
- `pixi run mojo run -I . tests/test_summary.mojo`: 2 passed.
- `pixi run mojo run -I . test_grid_timestep_reference.mojo`: 9 passed.
- The structured Mojo GPU executable builds successfully for H200.
- Strict CUDA grid-4 and full grid-64 comparisons pass with zero error.
- `git diff --check`: passes.
