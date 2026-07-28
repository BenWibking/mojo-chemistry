// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Pure-CUDA port of the 32-cell cooperative structured Mojo kernel.
#pragma once

#if !defined(PRIMORDIAL_ROS2S_ENABLE_CUDA)
#error "structured_cuda.cuh requires PRIMORDIAL_ROS2S_ENABLE_CUDA"
#endif

// These names are intentionally visible to the generated slice namespaces.
inline constexpr double kPi = 3.141592653589793238462643383279502884;
inline constexpr double kLog10 = 2.3025850929940456840179914546843642;

__device__ __forceinline__ double structured_mul(double lhs, double rhs) {
    return __dmul_rn(lhs, rhs);
}

__device__ __forceinline__ double structured_sqrt(double value) {
    return __dsqrt_rn(value);
}

__device__ __forceinline__ bool structured_truthy(bool value) { return value; }

__device__ __forceinline__ bool structured_truthy(double value) {
    return value != 0.0;
}

__device__ __forceinline__ double powi_m3(double x) {
    return 1.0 / (x * (x * x));
}

__device__ __forceinline__ double powi_m5(double x) {
    return 1.0 / (x * ((x * x) * (x * x)));
}

__device__ __forceinline__ double powi_m7(double x) {
    const double x3 = x * (x * x);
    return 1.0 / (x * (x3 * x3));
}

#include "generated/slices_base_shared.cuh"
#include "generated/slices_rhs_shared.cuh"

namespace structured_cuda {

inline constexpr int kNumSpecies = pc::NumSpec;
inline constexpr int kEquations = pc::neqs;
inline constexpr int kEnergy = pc::NumSpec;
inline constexpr int kTileCells = 32;
inline constexpr int kBlockThreads = 256;
inline constexpr int kDagWarps = 8;
inline constexpr int kLuGroupsPerWarp = 4;
inline constexpr int kLuGroupWidth = 8;
inline constexpr int kMatrixValues = kEquations * kEquations;
inline constexpr int kScratchSlots = structured_cuda_base::kScratchSlots;
inline constexpr int kExchangeSlots = structured_cuda_base::kExchangeSlots;
// Standard C++ does not permit a zero-length member array. The tuned schedule
// specializes all exchange accesses away, so this is only an addressable dummy.
inline constexpr int kExchangeStorageSlots =
    kExchangeSlots == 0 ? 1 : kExchangeSlots;

static_assert(kEquations == 15);
static_assert(kScratchSlots >= structured_cuda_rhs::kScratchSlots);
static_assert(kExchangeSlots >= structured_cuda_rhs::kExchangeSlots);
static_assert(structured_cuda_base::kConcurrentWarps ==
              structured_cuda_rhs::kConcurrentWarps);
static_assert(structured_cuda_base::kConcurrentWarps == 3 ||
              structured_cuda_base::kConcurrentWarps == kDagWarps);
static_assert(kDagWarps * kLuGroupsPerWarp == kTileCells);
static_assert(kLuGroupsPerWarp * kLuGroupWidth == 32);

struct alignas(16) SharedStorage {
    double jacobian[kTileCells * kMatrixValues];
    double scratch[kTileCells * kScratchSlots];
    double rhs[kTileCells * kEquations];
    double dag_inputs[kTileCells * kEquations];
    double exchange[kTileCells * kExchangeStorageSlots];
    double base[kTileCells * kEquations];
    double candidate[kTileCells * kEquations];
    double stage_y[kTileCells * kEquations];
    double k1[kTileCells * kEquations];
    double k2[kTileCells * kEquations];
    double work[kTileCells * kEquations];
    double trial_errors[kTileCells];
    double control_f64[3];
    int pivots[kTileCells * kEquations];
    int infos[kTileCells];
    int participants[kTileCells];
    int control_i32[4];
};

inline constexpr std::size_t kSharedBytes = sizeof(SharedStorage);
inline constexpr std::size_t kHopperSharedBytes = 227U * 1024U;
static_assert(kSharedBytes <= kHopperSharedBytes,
              "structured CUDA kernel exceeds Hopper shared-memory capacity");

__device__ __forceinline__ double cbrt_nonnegative(double value) {
    if (value == 0.0) {
        return 0.0;
    }
    return exp(log(value) / 3.0);
}

__device__ __forceinline__ int first_wave_logical_warp(int physical_warp) {
    if (physical_warp == 0) {
        return 7;
    }
    if (physical_warp == 1) {
        return 1;
    }
    return 0;
}

__device__ __forceinline__ int second_wave_logical_warp(int physical_warp) {
    return physical_warp + 2;
}

__device__ __forceinline__ void dispatch_base_slice(
    int logical_warp, const double* inputs, double* jacobian, double* rhs,
    double* exchange, double* scratch, int cell, double z) {
    switch (logical_warp) {
    case 0:
        structured_cuda_base::base_slice_0_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 1:
        structured_cuda_base::base_slice_1_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 2:
        structured_cuda_base::base_slice_2_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 3:
        structured_cuda_base::base_slice_3_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 4:
        structured_cuda_base::base_slice_4_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 5:
        structured_cuda_base::base_slice_5_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    case 6:
        structured_cuda_base::base_slice_6_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    default:
        structured_cuda_base::base_slice_7_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z);
        break;
    }
}

__device__ __forceinline__ void dispatch_rhs_slice(
    int logical_warp, const double* inputs, double* outputs, double* exchange,
    double* scratch, int cell, double z) {
    switch (logical_warp) {
    case 0:
        structured_cuda_rhs::rhs_specie_slice_0_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 1:
        structured_cuda_rhs::rhs_specie_slice_1_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 2:
        structured_cuda_rhs::rhs_specie_slice_2_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 3:
        structured_cuda_rhs::rhs_specie_slice_3_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 4:
        structured_cuda_rhs::rhs_specie_slice_4_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 5:
        structured_cuda_rhs::rhs_specie_slice_5_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    case 6:
        structured_cuda_rhs::rhs_specie_slice_6_shared(
            inputs, outputs, exchange, scratch, cell, z);
        structured_cuda_rhs::rhs_eint_slice_6_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    default:
        structured_cuda_rhs::rhs_specie_slice_7_shared(
            inputs, outputs, exchange, scratch, cell, z);
        break;
    }
}

__device__ __forceinline__ void prepare_dag_inputs(
    const double* base, double* inputs, int cell, int local_lane,
    int active_cells) {
    if (cell < active_cells) {
        for (int species = local_lane; species < kNumSpecies;
             species += kLuGroupWidth) {
            inputs[cell * kEquations + species] =
                fmax(base[cell * kEquations + species],
                     pc::small_number_density_floor());
        }
    }
    __syncwarp();
    if (cell < active_cells && local_lane == 0) {
        double rhotot = 0.0;
        for (int species = 0; species < kNumSpecies; ++species) {
            const double value = inputs[cell * kEquations + species];
            rhotot += structured_mul(value, pc::species_mass(species));
        }
        double sum_abarinv = 0.0;
        double sum_gammasinv = 0.0;
        for (int species = 0; species < kNumSpecies; ++species) {
            const double value = inputs[cell * kEquations + species];
            sum_abarinv += value;
            sum_gammasinv += structured_mul(
                structured_mul(value, pc::constants::m_p) / rhotot,
                1.0 / (pc::species_gamma(species) - 1.0));
        }
        sum_abarinv *= pc::constants::m_p / rhotot;
        sum_gammasinv /= sum_abarinv;
        const double energy = base[cell * kEquations + kEnergy];
        inputs[cell * kEquations + kEnergy] = energy /
            structured_mul(
                structured_mul(sum_gammasinv,
                               pc::constants::n_A * pc::constants::k_B),
                sum_abarinv);
    }
    __syncthreads();
}

__device__ __forceinline__ void evaluate_base(
    SharedStorage& shared, int physical_warp, int physical_lane,
    int active_cells) {
    constexpr double z = pc::default_redshift;
    if constexpr (structured_cuda_base::kExchangeSlots > 0) {
        if (physical_warp == 0 && physical_lane < active_cells) {
            structured_cuda_base::base_exchange_shared(
                shared.dag_inputs, shared.exchange, shared.scratch, physical_lane,
                z);
        }
        __syncthreads();
    }
    if constexpr (structured_cuda_base::kConcurrentWarps == kDagWarps) {
        // The tuned schedule gives each physical warp exactly one DAG slice.
        if (physical_lane < active_cells) {
            dispatch_base_slice(physical_warp, shared.dag_inputs,
                                shared.jacobian, shared.rhs, shared.exchange,
                                shared.scratch, physical_lane, z);
        }
    } else {
        if (physical_warp < 3 && physical_lane < active_cells) {
            dispatch_base_slice(first_wave_logical_warp(physical_warp),
                                shared.dag_inputs, shared.jacobian, shared.rhs,
                                shared.exchange, shared.scratch, physical_lane, z);
        }
        __syncthreads();
        if (physical_warp < 5 && physical_lane < active_cells) {
            dispatch_base_slice(second_wave_logical_warp(physical_warp),
                                shared.dag_inputs, shared.jacobian, shared.rhs,
                                shared.exchange, shared.scratch, physical_lane, z);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void evaluate_rhs(
    SharedStorage& shared, int physical_warp, int physical_lane,
    int active_cells) {
    constexpr double z = pc::default_redshift;
    if constexpr (structured_cuda_rhs::kExchangeSlots > 0) {
        if (physical_warp == 0 && physical_lane < active_cells) {
            structured_cuda_rhs::rhs_specie_exchange_shared(
                shared.dag_inputs, shared.exchange, shared.scratch, physical_lane,
                z);
        }
        __syncthreads();
    }
    if constexpr (structured_cuda_rhs::kConcurrentWarps == kDagWarps) {
        if (physical_lane < active_cells) {
            dispatch_rhs_slice(physical_warp, shared.dag_inputs, shared.rhs,
                               shared.exchange, shared.scratch, physical_lane, z);
        }
    } else {
        if (physical_warp < 3 && physical_lane < active_cells) {
            dispatch_rhs_slice(first_wave_logical_warp(physical_warp),
                               shared.dag_inputs, shared.rhs, shared.exchange,
                               shared.scratch, physical_lane, z);
        }
        __syncthreads();
        if (physical_warp < 5 && physical_lane < active_cells) {
            dispatch_rhs_slice(second_wave_logical_warp(physical_warp),
                               shared.dag_inputs, shared.rhs, shared.exchange,
                               shared.scratch, physical_lane, z);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void factorize_shared(
    double* jacobian, bool participating, int cell, int local_lane,
    int* pivots, int* infos) {
    if (local_lane == 0) {
        infos[cell] = 0;
    }
    __syncwarp();
    for (int k = 0; k < kEquations - 1; ++k) {
        int pivot = k;
        double maximum = -1.0;
        if (participating) {
            for (int row = k + local_lane; row < kEquations;
                 row += kLuGroupWidth) {
                const double value = fabs(
                    jacobian[cell * kMatrixValues + row * kEquations + k]);
                if (value > maximum) {
                    maximum = value;
                    pivot = row;
                }
            }
        }
        const int group_base = (threadIdx.x & 31) - local_lane;
        const unsigned int group_mask = 0xffU << group_base;
        // Prefer the lowest row on ties to preserve the serial pivot policy.
        for (int offset = kLuGroupWidth / 2; offset > 0; offset /= 2) {
            const double other_maximum =
                __shfl_down_sync(group_mask, maximum, offset, kLuGroupWidth);
            const int other_pivot =
                __shfl_down_sync(group_mask, pivot, offset, kLuGroupWidth);
            if (other_maximum > maximum ||
                (other_maximum == maximum && other_pivot < pivot)) {
                maximum = other_maximum;
                pivot = other_pivot;
            }
        }
        if (participating && local_lane == 0) {
            pivots[cell * kEquations + k] = pivot;
        }
        __syncwarp();
        if (participating) {
            pivot = pivots[cell * kEquations + k];
            for (int column = local_lane; column < kEquations;
                 column += kLuGroupWidth) {
                if (column >= k && pivot != k) {
                    const int top = cell * kMatrixValues + k * kEquations + column;
                    const int bottom =
                        cell * kMatrixValues + pivot * kEquations + column;
                    const double temporary = jacobian[bottom];
                    jacobian[bottom] = jacobian[top];
                    jacobian[top] = temporary;
                }
            }
        }
        __syncwarp();
        if (participating && local_lane == k % kLuGroupWidth) {
            const int diagonal_index =
                cell * kMatrixValues + k * kEquations + k;
            const double diagonal = jacobian[diagonal_index];
            if (diagonal == 0.0) {
                infos[cell] = k + 1;
            } else {
                const double scale = -1.0 / diagonal;
                for (int row = k + 1; row < kEquations; ++row) {
                    jacobian[cell * kMatrixValues + row * kEquations + k] *= scale;
                }
            }
        }
        __syncwarp();
        if (participating) {
            for (int column = local_lane; column < kEquations;
                 column += kLuGroupWidth) {
                if (column > k) {
                    const double upper =
                        jacobian[cell * kMatrixValues + k * kEquations + column];
                    for (int row = k + 1; row < kEquations; ++row) {
                        const int index =
                            cell * kMatrixValues + row * kEquations + column;
                        const double multiplier =
                            jacobian[cell * kMatrixValues + row * kEquations + k];
                        jacobian[index] += structured_mul(upper, multiplier);
                    }
                }
            }
        }
        __syncwarp();
    }
    if (participating && local_lane == 0 &&
        jacobian[cell * kMatrixValues + kMatrixValues - 1] == 0.0) {
        infos[cell] = kEquations;
    }
    __syncwarp();
}

__device__ __forceinline__ void solve_shared(
    const double* jacobian, double* rhs, bool participating, int cell,
    int local_lane, const int* pivots, const int* infos) {
    const bool nonsingular = participating && infos[cell] == 0;
    for (int k = 0; k < kEquations - 1; ++k) {
        if (nonsingular && local_lane == 0) {
            const int pivot = pivots[cell * kEquations + k];
            if (pivot != k) {
                const double temporary = rhs[cell * kEquations + pivot];
                rhs[cell * kEquations + pivot] = rhs[cell * kEquations + k];
                rhs[cell * kEquations + k] = temporary;
            }
        }
        __syncwarp();
        if (nonsingular) {
            const double value = rhs[cell * kEquations + k];
            for (int row = local_lane; row < kEquations;
                 row += kLuGroupWidth) {
                if (row > k) {
                    rhs[cell * kEquations + row] += structured_mul(
                        value,
                        jacobian[cell * kMatrixValues + row * kEquations + k]);
                }
            }
        }
        __syncwarp();
    }
    for (int reverse_k = 0; reverse_k < kEquations; ++reverse_k) {
        const int k = kEquations - 1 - reverse_k;
        if (nonsingular && local_lane == k % kLuGroupWidth) {
            rhs[cell * kEquations + k] /=
                jacobian[cell * kMatrixValues + k * kEquations + k];
        }
        __syncwarp();
        if (nonsingular) {
            const double negative_x = -rhs[cell * kEquations + k];
            for (int row = local_lane; row < kEquations;
                 row += kLuGroupWidth) {
                if (row < k) {
                    rhs[cell * kEquations + row] += structured_mul(
                        negative_x,
                        jacobian[cell * kMatrixValues + row * kEquations + k]);
                }
            }
        }
        __syncwarp();
    }
}

__global__ __launch_bounds__(kBlockThreads, 1)
void advance_collapse_gridwide_structured_kernel(
    CollapseState* cells, int num_cells, int completed_global_steps,
    double grid_time, double next_grid_time, double dt_grid,
    double* jacobian_cache, int* integrated_count, int* failure_code) {
    extern __shared__ __align__(16) unsigned char raw_shared[];
    auto& shared = *reinterpret_cast<SharedStorage*>(raw_shared);

    const int tid = threadIdx.x;
    const int physical_warp = tid / 32;
    const int physical_lane = tid % 32;
    const int warp_cell =
        physical_warp * kLuGroupsPerWarp + physical_lane / kLuGroupWidth;
    const int local_lane = physical_lane % kLuGroupWidth;
    const int tile_begin = blockIdx.x * kTileCells;
    const int active_cells = min(kTileCells, num_cells - tile_begin);

    if (tid < kTileCells) {
        bool participating = false;
        int participant_code = 0;
        if (tid < active_cells) {
            const int cell = tile_begin + tid;
            CollapseState collapse = cells[cell];
            if (collapse.completed_steps == completed_global_steps) {
                if (collapse.time != grid_time) {
                    atomicCAS(failure_code,
                              static_cast<int>(integrators::IntegratorResult::SUCCESS),
                              static_cast<int>(integrators::IntegratorResult::BAD_INPUTS));
                } else {
                    const double local_dt = collapse_timestep(collapse);
                    if (local_dt >= 10.0) {
                        const double old_density = collapse.density_driver;
                        collapse.density_driver +=
                            structured_mul(dt_grid, collapse.density_driver) /
                            (local_dt / tff_reduc);
                        if (valid_positive(collapse.density_driver) &&
                            collapse.density_driver <= 2.0e-6) {
                            const double ratio =
                                collapse.density_driver / old_density;
                            for (int species = 0; species < kNumSpecies; ++species) {
                                collapse.current.xn[species] *= ratio;
                            }
                            collapse.current.rho *= ratio;
                            pc::eos_rt(collapse.current);
                            participating = true;
                        }
                    }
                }
            }
            for (int component = 0; component < kNumSpecies; ++component) {
                shared.base[tid * kEquations + component] =
                    collapse.current.xn[component];
            }
            shared.base[tid * kEquations + kEnergy] = collapse.current.e;
            cells[cell] = collapse;
            const int encoded_cell = cell + 1;
            participant_code = participating ? encoded_cell : -encoded_cell;
        }
        shared.participants[tid] = participant_code;
    }
    __syncthreads();

    if (physical_warp == 0) {
        const unsigned int participant_mask = __ballot_sync(
            0xffffffffU,
            physical_lane < active_cells &&
                shared.participants[physical_lane] > 0);
        if (physical_lane == 0) {
            shared.control_i32[0] = __popc(participant_mask);
            shared.control_i32[2] = 0;
            shared.control_i32[3] = 0;
            shared.control_f64[1] = 0.0;
            shared.control_f64[2] = 1.0;
        }
    }
    __syncthreads();

    constexpr double uround = 1.0e-16;
    constexpr double fac_min = 0.2;
    constexpr double fac_max = 6.0;
    constexpr double safe = 0.9;
    constexpr int max_steps = 10000000;
    constexpr double gamma = 0.292893218813452;
    constexpr double a21 = 2.0000000000000036;
    constexpr double a31 = 6.828427124746214;
    constexpr double a32 = 3.4142135623731007;
    constexpr double c21 = -6.828427124746214;
    constexpr double c31 = -10.949747468305889;
    constexpr double c32 = -7.535533905932761;
    constexpr double b1 = 6.828427124746214;
    constexpr double b2 = 3.414213562373101;
    constexpr double e1 = -0.23570226039551292;
    constexpr double e2 = -0.23570226039551567;
    constexpr double e3 = -0.13807118745769906;

    double h = dt_grid;
    double x = 0.0;
    bool reject = false;
    int nsing = 0;
    int n_step = 0;
    int n_accept = 0;
    int status = static_cast<int>(integrators::IntegratorResult::SUCCESS);
    // LU factorization overwrites shared.jacobian.  A rejected trial keeps the
    // same base state and substep time, so retain the raw Jacobian until a
    // trial is accepted and advances both.
    bool jacobian_valid = false;
    const bool participating =
        warp_cell < active_cells && shared.participants[warp_cell] > 0;

    while (true) {
        if (shared.control_i32[0] == 0 || x >= dt_grid ||
            status != static_cast<int>(integrators::IntegratorResult::SUCCESS)) {
            break;
        }
        if (n_step > max_steps) {
            status = static_cast<int>(integrators::IntegratorResult::TOO_MANY_STEPS);
            continue;
        }
        if (0.1 * fabs(h) <= fabs(x) * uround) {
            status = static_cast<int>(integrators::IntegratorResult::DT_UNDERFLOW);
            continue;
        }
        bool final_trial = false;
        if (x + structured_mul(h, 1.0001) >= dt_grid) {
            h = dt_grid - x;
            final_trial = true;
        }
        if (tid < kTileCells) {
            shared.trial_errors[tid] = 0.0;
        }
        if (tid == 0) {
            shared.control_i32[1] = 0;
        }
        __syncthreads();

        const bool reuse_jacobian = jacobian_valid;
        prepare_dag_inputs(shared.base, shared.dag_inputs, warp_cell, local_lane,
                           active_cells);
        if (reuse_jacobian) {
            evaluate_rhs(shared, physical_warp, physical_lane, active_cells);
        } else {
            evaluate_base(shared, physical_warp, physical_lane, active_cells);
            if (tid == 0) {
                ++shared.control_i32[2];
            }
        }

        if (participating) {
            const int global_cell = tile_begin + warp_cell;
            for (int column = local_lane; column < kEquations;
                 column += kLuGroupWidth) {
                for (int row = 0; row < kEquations; ++row) {
                    const int index =
                        warp_cell * kMatrixValues + row * kEquations + column;
                    const std::size_t cache_index =
                        static_cast<std::size_t>(global_cell) * kMatrixValues +
                        row * kEquations + column;
                    const double raw_jacobian =
                        reuse_jacobian ? jacobian_cache[cache_index]
                                       : shared.jacobian[index];
                    if (!reuse_jacobian) {
                        jacobian_cache[cache_index] = raw_jacobian;
                    }
                    shared.jacobian[index] = -raw_jacobian;
                    if (row == column) {
                        shared.jacobian[index] += 1.0 / (h * gamma);
                    }
                }
            }
        }
        jacobian_valid = true;
        __syncthreads();
        factorize_shared(shared.jacobian, participating, warp_cell, local_lane,
                         shared.pivots, shared.infos);
        __syncthreads();
        if (physical_warp == 0) {
            const unsigned int singular_mask = __ballot_sync(
                0xffffffffU,
                physical_lane < active_cells &&
                    shared.infos[physical_lane] != 0);
            if (physical_lane == 0) {
                shared.control_i32[1] = singular_mask != 0;
            }
        }
        __syncthreads();
        solve_shared(shared.jacobian, shared.rhs, participating, warp_cell,
                     local_lane, shared.pivots, shared.infos);
        if (participating) {
            for (int component = local_lane; component < kEquations;
                 component += kLuGroupWidth) {
                const int index = warp_cell * kEquations + component;
                const double solution = shared.rhs[index];
                shared.k1[index] = solution;
                shared.stage_y[index] =
                    shared.base[index] + structured_mul(a21, solution);
                shared.k2[index] = (c21 / h) * solution;
            }
        }
        __syncthreads();

        prepare_dag_inputs(shared.stage_y, shared.dag_inputs, warp_cell,
                           local_lane, active_cells);
        evaluate_rhs(shared, physical_warp, physical_lane, active_cells);
        if (participating) {
            for (int component = local_lane; component < kEquations;
                 component += kLuGroupWidth) {
                const int index = warp_cell * kEquations + component;
                shared.rhs[index] += shared.k2[index];
            }
        }
        __syncthreads();
        solve_shared(shared.jacobian, shared.rhs, participating, warp_cell,
                     local_lane, shared.pivots, shared.infos);
        if (participating) {
            for (int component = local_lane; component < kEquations;
                 component += kLuGroupWidth) {
                const int index = warp_cell * kEquations + component;
                const double solution = shared.rhs[index];
                shared.k2[index] = solution;
                shared.stage_y[index] =
                    shared.base[index] + structured_mul(a31, shared.k1[index]) +
                    structured_mul(a32, solution);
                shared.work[index] =
                    (structured_mul(c31, shared.k1[index]) +
                     structured_mul(c32, solution)) /
                    h;
            }
        }
        __syncthreads();

        prepare_dag_inputs(shared.stage_y, shared.dag_inputs, warp_cell,
                           local_lane, active_cells);
        evaluate_rhs(shared, physical_warp, physical_lane, active_cells);
        if (participating) {
            for (int component = local_lane; component < kEquations;
                 component += kLuGroupWidth) {
                const int index = warp_cell * kEquations + component;
                shared.rhs[index] += shared.work[index];
            }
        }
        __syncthreads();
        solve_shared(shared.jacobian, shared.rhs, participating, warp_cell,
                     local_lane, shared.pivots, shared.infos);
        if (participating) {
            for (int component = local_lane; component < kEquations;
                 component += kLuGroupWidth) {
                const int index = warp_cell * kEquations + component;
                const double first = shared.k1[index];
                const double second = shared.k2[index];
                const double third = shared.rhs[index];
                shared.candidate[index] =
                    shared.base[index] + structured_mul(b1, first) +
                    structured_mul(b2, second) + third;
                shared.work[index] =
                    structured_mul(e1, first) + structured_mul(e2, second) +
                    structured_mul(e3, third);
            }
        }
        __syncthreads();
        if (participating && local_lane == 0) {
            double sum_squared = 0.0;
            for (int component = 0; component < kEquations; ++component) {
                const int index = warp_cell * kEquations + component;
                const double old_value = shared.base[index];
                const double new_value = shared.candidate[index];
                const double rtol = component == kEnergy ? rtol_energy : rtol_spec;
                const double atol = component == kEnergy ? atol_energy : atol_spec;
                const double scale =
                    atol + structured_mul(rtol,
                                          fmax(fabs(old_value), fabs(new_value)));
                const double term = shared.work[index] / scale;
                sum_squared += structured_mul(term, term);
            }
            shared.trial_errors[warp_cell] =
                structured_sqrt(sum_squared / static_cast<double>(kEquations));
        }
        __syncthreads();

        if (shared.control_i32[1] != 0) {
            ++nsing;
            if (nsing >= 5) {
                status = static_cast<int>(
                    integrators::IntegratorResult::LU_DECOMPOSITION_ERROR);
            } else {
                h *= 0.5;
                reject = true;
            }
            continue;
        }

        if (physical_warp == 0) {
            double tile_error = 0.0;
            if (physical_lane < active_cells &&
                shared.participants[physical_lane] > 0) {
                tile_error = shared.trial_errors[physical_lane];
                if (!isfinite(tile_error)) {
                    tile_error = std::numeric_limits<double>::max();
                }
            }
            for (int offset = 16; offset > 0; offset /= 2) {
                tile_error = fmax(
                    tile_error,
                    __shfl_down_sync(0xffffffffU, tile_error, offset));
            }
            if (physical_lane == 0) {
                shared.control_f64[0] = tile_error;
            }
        }
        __syncthreads();
        const double err_tile = shared.control_f64[0];
        ++n_step;
        const double fac_step =
            fmax(1.0 / fac_max,
                 fmin(1.0 / fac_min, cbrt_nonnegative(err_tile) / safe));
        double hnew = h / fac_step;
        if (err_tile <= 1.0) {
            ++n_accept;
            if (n_accept > 1) {
                const double facgus = fmax(
                    1.0 / fac_max,
                    fmin(1.0 / fac_min,
                         (shared.control_f64[1] / h) *
                             cbrt_nonnegative(
                                 (err_tile * err_tile) / shared.control_f64[2]) /
                             safe));
                hnew = h / fmax(fac_step, facgus);
            }
            if (participating) {
                for (int component = local_lane; component < kEquations;
                     component += kLuGroupWidth) {
                    shared.base[warp_cell * kEquations + component] =
                        shared.candidate[warp_cell * kEquations + component];
                }
            }
            __syncthreads();
            if (tid == 0) {
                shared.control_f64[1] = h;
                shared.control_f64[2] = fmax(1.0e-2, err_tile);
            }
            x = final_trial ? dt_grid : x + h;
            hnew = fmin(fabs(hnew), dt_grid);
            if (reject) {
                hnew = fmin(hnew, fabs(h));
            }
            reject = false;
            h = hnew;
            jacobian_valid = false;
        } else {
            if (tid == 0 && n_accept >= 1) {
                ++shared.control_i32[3];
            }
            reject = true;
            h = hnew;
        }
    }

    __syncthreads();
    if (tid < active_cells) {
        const int participant_code = shared.participants[tid];
        const bool cell_participating = participant_code > 0;
        const int cell =
            (cell_participating ? participant_code : -participant_code) - 1;
        CollapseState collapse = cells[cell];
        if (status == static_cast<int>(integrators::IntegratorResult::SUCCESS) &&
            cell_participating) {
            for (int species = 0; species < kNumSpecies; ++species) {
                collapse.current.xn[species] =
                    shared.base[tid * kEquations + species];
            }
            collapse.current.e = shared.base[tid * kEquations + kEnergy];
            pc::floor_and_normalize_number_densities(collapse.current);
            pc::balance_charge(collapse.current);
            pc::floor_and_normalize_number_densities(collapse.current);
            pc::eos_re(collapse.current);
            collapse.time = next_grid_time;
            collapse.completed_steps = completed_global_steps + 1;
            collapse.stats.internal_steps += static_cast<std::uint64_t>(n_step);
            collapse.stats.rhs_calls += static_cast<std::uint64_t>(3 * n_step);
            collapse.stats.jacobian_calls +=
                static_cast<std::uint64_t>(shared.control_i32[2]);
            collapse.stats.decompositions += static_cast<std::uint64_t>(n_step);
            collapse.stats.linear_solves += static_cast<std::uint64_t>(3 * n_step);
            collapse.stats.accepted_steps += static_cast<std::uint64_t>(n_accept);
            collapse.stats.rejected_steps +=
                static_cast<std::uint64_t>(shared.control_i32[3]);
        }
        cells[cell] = collapse;
    }
    __syncthreads();
    if (tid == 0) {
        if (status == static_cast<int>(integrators::IntegratorResult::SUCCESS)) {
            atomicAdd(integrated_count, shared.control_i32[0]);
        } else {
            atomicCAS(failure_code,
                      static_cast<int>(integrators::IntegratorResult::SUCCESS),
                      status);
        }
    }
}

} // namespace structured_cuda
