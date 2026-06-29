# Chemical kinetic reproducer (Quokka)

`reproducer.cpp` is a standalone translation unit for the primordial chemistry
collapse-grid reproducer. Run the commands below from this directory.

## Table of results

| ROCm version | runtime (s) | target | sgpr_count | vgpr_count | sgpr_spill_count | vgpr_spill_count | agpr_count | private_segment_fixed_size |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 6.0.0 | FAILED | gfx90a | 106 | 128 | 2864 | 2340 | 0 | 7448 |
| 6.1.3 | FAILED | gfx90a | 106 | 128 | 2868 | 2375 | 0 | 7440 |
| 6.2.4 | 324.48 | gfx90a | 106 | 128 | 2076 | 2254 | 0 | 7336 |
| 6.3.1 | 341.79 | gfx90a | 106 | 128 | 2112 | 2286 | 0 | 7328 |
| 6.4.2 | 340.22 | gfx90a | 106 | 128 | 2024 | 2357 | 0 | 7360 |
| 7.0.2 | 307.75 | gfx90a | 106 | 128 | 2037 | 2280 | 0 | 7344 |
| 7.1.1 | 333.63 | gfx90a | 106 | 128 | 2037 | 2280 | 0 | 7344 |
| 7.2.0 | 328.39 | gfx90a | 104 | 128 | 1544 | 2082 | 0 | 7344 |

## Build

CPU build:

```bash
c++ -std=c++20 -O3 -I. reproducer.cpp -o reproducer
```

CUDA build with the default project settings:

```bash
nvcc -x cu -std=c++20 -O3 -arch=sm_90 --expt-relaxed-constexpr --fmad=false --maxrregcount=255 -DPRIMORDIAL_ROS2S_ENABLE_CUDA -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 -I. reproducer.cpp -o reproducer_cuda
```

HIP build with the default project settings:

```bash
hipcc -x hip -std=c++20 -O3 --offload-arch=gfx90a -DPRIMORDIAL_ROS2S_ENABLE_HIP -DPRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK=128 -I. reproducer.cpp -o reproducer_hip
```

## Run

Run the CPU reproducer:

```bash
./reproducer
```

Run the CUDA reproducer:

```bash
./reproducer_cuda
```

Run the HIP reproducer:

```bash
./reproducer_hip
```

# Extra options

Show the available options:

```bash
./reproducer --help
```

By default, the run uses a `64^3` cell grid, enables deterministic density
perturbations, and compares the final state against
`final_states_grid64_cpu.bin`. To run without any reference comparison:

```bash
./reproducer --no-compare-final-state
./reproducer_cuda --no-compare-final-state
./reproducer_hip --no-compare-final-state
```

To run a smaller case:

```bash
./reproducer --grid 1 --no-compare-final-state
./reproducer_cuda --grid 1 --no-compare-final-state
./reproducer_hip --grid 1 --no-compare-final-state
```

For a `--grid 1` run, the program prints the final representative cell state
directly. For larger grids, it writes a packed binary final-state file named
`final_states_grid<N>_<backend>.bin`, for example
`final_states_grid64_cpu.bin`, `final_states_grid64_cuda.bin`, or
`final_states_grid64_hip.bin`. If that output file already exists, the old file
is moved aside with an `.old.<suffix>` name.

## Output

The main progress and summary fields are:

- `grid`: the requested cubic grid size and total cell count.
- `perturbations`: whether deterministic density perturbations were enabled.
- `completed global collapse steps`: the number of global collapse iterations
  completed before every cell stopped or a failure occurred.
- `cell completed steps`, `cell physical time`, and `cell density driver`: for
  multi-cell runs these are printed as `[min, median, max]`; for one-cell runs
  the representative cell values are printed directly.
- `wall time`: elapsed host wall-clock time for the integration loop.
- `ROS2S internal steps`, `rhs calls`, `jacobian calls`, `decompositions`,
  `linear solves`, and `accepted/rejected`: integrator work counters. Multi-cell
  runs print `[min, median, max]`; one-cell runs print totals.

If final-state comparison is enabled, the program also prints:

- `final-state comparison file`: the reference file used.
- `final-state comparison: PASS`: the run matched the reference within the
  built-in tolerances.
- `final-state comparison: FAIL`: at least one checked field exceeded tolerance.
  The first failure is printed, and detailed failures are written to
  `final_state_comparison_failures_<backend>.csv`.

The process exits with status `0` on successful integration and, when enabled,
successful final-state comparison. It exits nonzero if integration fails,
argument parsing fails, an output file cannot be written, or comparison fails.
