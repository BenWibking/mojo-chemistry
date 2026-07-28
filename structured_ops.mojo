# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Hopper 32-cell DAG dispatch and shared-memory configuration.

import reproducer
from generated.slices_jac import (
    jac_slice_0,
    jac_slice_1,
    jac_slice_2,
    jac_slice_3,
    jac_slice_4,
    jac_slice_5,
    jac_slice_6,
    jac_slice_7,
)
from generated.slices_rhs import (
    rhs_eint_slice_6,
    rhs_specie_slice_0,
    rhs_specie_slice_1,
    rhs_specie_slice_2,
    rhs_specie_slice_3,
    rhs_specie_slice_4,
    rhs_specie_slice_5,
    rhs_specie_slice_6,
    rhs_specie_slice_7,
)
from generated.slices_rhs_shared import (
    ExchangeSlots as RhsExchangeSlots,
    ScratchSlots as RhsScratchSlots,
    rhs_eint_slice_6_shared,
    rhs_specie_exchange_shared,
    rhs_specie_slice_0_shared,
    rhs_specie_slice_1_shared,
    rhs_specie_slice_2_shared,
    rhs_specie_slice_3_shared,
    rhs_specie_slice_4_shared,
    rhs_specie_slice_5_shared,
    rhs_specie_slice_6_shared,
    rhs_specie_slice_7_shared,
)
from generated.slices_base_shared import (
    ExchangeSlots as BaseExchangeSlots,
    ScratchSlots as BaseScratchSlots,
    base_exchange_shared,
    base_slice_0_shared,
    base_slice_1_shared,
    base_slice_2_shared,
    base_slice_3_shared,
    base_slice_4_shared,
    base_slice_5_shared,
    base_slice_6_shared,
    base_slice_7_shared,
)
from layout import TensorLayout, TileTensor
from std.gpu import WARP_SIZE, barrier, block_idx, thread_idx
from std.gpu.memory import AddressSpace
from std.math import min


comptime DagWarps = 8
comptime DagWaves = 2
comptime DagFirstWaveWarps = 3
comptime DagSecondWaveWarps = 5
comptime OwnerWarps = 8
comptime LuGroupsPerWarp = 4
comptime LuGroupWidth = 8
comptime TileCells = 32
comptime BatchCells = TileCells
comptime BatchesPerTile = 1
comptime BlockThreads = OwnerWarps * WARP_SIZE
comptime DynamicJBytes = TileCells * reproducer.Neqs * reproducer.Neqs * 8
comptime DagScratchSlots = (
    BaseScratchSlots
    if BaseScratchSlots >= RhsScratchSlots
    else RhsScratchSlots
)
comptime DagInputValues = TileCells * reproducer.Neqs
comptime DagScratchValues = TileCells * DagScratchSlots
comptime RhsValues = TileCells * reproducer.Neqs
comptime DynamicJValues = TileCells * reproducer.Neqs * reproducer.Neqs
comptime DagScratchOffset = DynamicJValues
comptime RhsOffset = DagScratchOffset + DagScratchValues
comptime DagInputOffset = RhsOffset + RhsValues
comptime DagExchangeSlots = (
    BaseExchangeSlots
    if BaseExchangeSlots >= RhsExchangeSlots
    else RhsExchangeSlots
)
comptime DagExchangeValues = TileCells * DagExchangeSlots
comptime DagExchangeOffset = DagInputOffset + DagInputValues
comptime DynamicSharedValues = DagExchangeOffset + DagExchangeValues
comptime DynamicSharedBytes = DynamicSharedValues * 8
comptime HopperSharedBytes = 227 * 1024
comptime MaxExchangeBytes = 8 * 1024
comptime GeneratedExchangeBytes = DagExchangeValues * 8


def align_up(value: Int, alignment: Int) -> Int:
    return ((value + alignment - 1) // alignment) * alignment


struct SmemLayout:
    comptime BaseOffset = 0
    comptime BaseBytes = TileCells * reproducer.Neqs * 8
    comptime CandidateOffset = align_up(Self.BaseOffset + Self.BaseBytes, 16)
    comptime CandidateBytes = TileCells * reproducer.Neqs * 8
    comptime VectorsOffset = align_up(
        Self.CandidateOffset + Self.CandidateBytes, 16
    )
    comptime VectorsBytes = 4 * TileCells * reproducer.Neqs * 8
    comptime XtOffset = align_up(Self.VectorsOffset + Self.VectorsBytes, 16)
    comptime XtBytes = 0
    comptime MetadataOffset = align_up(Self.XtOffset + Self.XtBytes, 16)
    comptime MetadataBytes = 2560
    comptime ExchangeOffset = align_up(
        Self.MetadataOffset + Self.MetadataBytes, 16
    )
    comptime ExchangeBytes = 0
    comptime ControllerOffset = align_up(
        Self.ExchangeOffset + Self.ExchangeBytes, 16
    )
    comptime ControllerBytes = 256
    comptime StaticBytes = align_up(
        Self.ControllerOffset + Self.ControllerBytes, 16
    )
    comptime Bytes = DynamicSharedBytes + Self.StaticBytes


struct FullBarrierSync:
    def __init__(out self):
        pass

    def phase_barrier(self):
        barrier()

    def producer_arrive(self, stage: Int):
        _ = stage
        barrier()

    def consumer_wait(self, stage: Int):
        _ = stage
        barrier()

    def consumer_release(self, stage: Int):
        _ = stage
        barrier()


def validate_config():
    comptime assert TileCells == 32
    comptime assert WARP_SIZE == 32, "M4 is Hopper-only"
    comptime assert BatchCells == TileCells
    comptime assert BatchesPerTile == 1
    comptime assert DagWarps == 8
    comptime assert DagWaves == 2
    comptime assert DagFirstWaveWarps + DagSecondWaveWarps == DagWarps
    comptime assert DagFirstWaveWarps == 3
    comptime assert DagSecondWaveWarps == 5
    comptime assert OwnerWarps * LuGroupsPerWarp == TileCells
    comptime assert LuGroupWidth * LuGroupsPerWarp == WARP_SIZE
    comptime assert reproducer.Neqs <= 2 * LuGroupWidth
    comptime assert BlockThreads == 256
    comptime assert DagScratchSlots == BaseScratchSlots
    comptime assert DagExchangeSlots == BaseExchangeSlots
    comptime assert GeneratedExchangeBytes <= MaxExchangeBytes
    comptime assert SmemLayout.Bytes <= HopperSharedBytes


def dispatch_jac_slice(
    warp_id: Int,
    state: reproducer.BurnState,
    mut jac: reproducer.MatTensor,
    X: reproducer.SpeciesTensor,
    z: Float64,
):
    if warp_id == 0:
        jac_slice_0(state, jac, X, z)
    elif warp_id == 1:
        jac_slice_1(state, jac, X, z)
    elif warp_id == 2:
        jac_slice_2(state, jac, X, z)
    elif warp_id == 3:
        jac_slice_3(state, jac, X, z)
    elif warp_id == 4:
        jac_slice_4(state, jac, X, z)
    elif warp_id == 5:
        jac_slice_5(state, jac, X, z)
    elif warp_id == 6:
        jac_slice_6(state, jac, X, z)
    else:
        jac_slice_7(state, jac, X, z)


def dispatch_rhs_slice(
    warp_id: Int,
    state: reproducer.BurnState,
    mut rhs: reproducer.VecTensor,
    X: reproducer.SpeciesTensor,
    z: Float64,
):
    if warp_id == 0:
        rhs_specie_slice_0(state, rhs, X, z)
    elif warp_id == 1:
        rhs_specie_slice_1(state, rhs, X, z)
    elif warp_id == 2:
        rhs_specie_slice_2(state, rhs, X, z)
    elif warp_id == 3:
        rhs_specie_slice_3(state, rhs, X, z)
    elif warp_id == 4:
        rhs_specie_slice_4(state, rhs, X, z)
    elif warp_id == 5:
        rhs_specie_slice_5(state, rhs, X, z)
    elif warp_id == 6:
        rhs_specie_slice_6(state, rhs, X, z)
        reproducer.vset(rhs, reproducer.NetIenuc, rhs_eint_slice_6(state, X, z))
    else:
        rhs_specie_slice_7(state, rhs, X, z)


comptime SharedPointer = UnsafePointer[
    Scalar[DType.float64],
    MutUntrackedOrigin,
    address_space=AddressSpace.SHARED,
]


def first_wave_logical_warp(physical_warp: Int) -> Int:
    if physical_warp == 0:
        return 7
    elif physical_warp == 1:
        return 1
    return 0


def second_wave_logical_warp(physical_warp: Int) -> Int:
    if physical_warp == 0:
        return 2
    elif physical_warp == 1:
        return 3
    elif physical_warp == 2:
        return 4
    elif physical_warp == 3:
        return 5
    return 6


def dispatch_rhs_slice_shared(
    warp_id: Int,
    inputs: SharedPointer,
    outputs: SharedPointer,
    exchange: SharedPointer,
    scratch: SharedPointer,
    cell: Int,
    z: Float64,
):
    if warp_id == 0:
        rhs_specie_slice_0_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 1:
        rhs_specie_slice_1_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 2:
        rhs_specie_slice_2_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 3:
        rhs_specie_slice_3_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 4:
        rhs_specie_slice_4_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 5:
        rhs_specie_slice_5_shared(inputs, outputs, exchange, scratch, cell, z)
    elif warp_id == 6:
        rhs_specie_slice_6_shared(inputs, outputs, exchange, scratch, cell, z)
        rhs_eint_slice_6_shared(inputs, outputs, exchange, scratch, cell, z)
    else:
        rhs_specie_slice_7_shared(inputs, outputs, exchange, scratch, cell, z)


def produce_rhs_exchange_shared(
    inputs: SharedPointer,
    exchange: SharedPointer,
    scratch: SharedPointer,
    cell: Int,
    z: Float64,
):
    rhs_specie_exchange_shared(inputs, exchange, scratch, cell, z)


def dispatch_base_slice_shared(
    warp_id: Int,
    inputs: SharedPointer,
    jacobian: SharedPointer,
    rhs: SharedPointer,
    exchange: SharedPointer,
    scratch: SharedPointer,
    cell: Int,
    z: Float64,
):
    if warp_id == 0:
        base_slice_0_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 1:
        base_slice_1_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 2:
        base_slice_2_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 3:
        base_slice_3_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 4:
        base_slice_4_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 5:
        base_slice_5_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    elif warp_id == 6:
        base_slice_6_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )
    else:
        base_slice_7_shared(
            inputs, jacobian, rhs, exchange, scratch, cell, z
        )


def produce_base_exchange_shared(
    inputs: SharedPointer,
    exchange: SharedPointer,
    scratch: SharedPointer,
    cell: Int,
    z: Float64,
):
    base_exchange_shared(inputs, exchange, scratch, cell, z)


def dag_slice_kernel[
    StateLayout: TensorLayout,
    JacobianLayout: TensorLayout,
    RhsLayout: TensorLayout,
](
    states: TileTensor[DType.float64, StateLayout, MutAnyOrigin],
    jacobians: TileTensor[DType.float64, JacobianLayout, MutAnyOrigin],
    right_hand_sides: TileTensor[DType.float64, RhsLayout, MutAnyOrigin],
    num_cells: Int,
):
    comptime assert states.flat_rank == 2
    comptime assert jacobians.flat_rank == 2
    comptime assert right_hand_sides.flat_rank == 2
    validate_config()
    var sync = FullBarrierSync()
    var warp_id = thread_idx.x // WARP_SIZE
    var lane = thread_idx.x % WARP_SIZE
    var cell = block_idx.x * TileCells + lane
    var evaluating = warp_id < DagWarps and cell < num_cells
    if evaluating:
        var y = reproducer.VecTensor.stack_allocation()
        for component in range(reproducer.Neqs):
            reproducer.vset(
                y,
                component,
                rebind[Scalar[DType.float64]](states[cell, component]),
            )
        var state = reproducer.burn_state_from_y(y)
        var X = reproducer.SpeciesTensor.stack_allocation()
        for species in range(reproducer.NumSpec):
            reproducer.vset(X, species, state.xn[species])
        var jac = reproducer.MatTensor.stack_allocation()
        var rhs = reproducer.VecTensor.stack_allocation()
        dispatch_jac_slice(warp_id, state, jac, X, reproducer.redshift())
        dispatch_rhs_slice(warp_id, state, rhs, X, reproducer.redshift())
        var first_row = warp_id * 2
        var last_row = min(first_row + 2, reproducer.Neqs)
        for row in range(first_row, last_row):
            for column in range(reproducer.Neqs):
                jacobians[cell, row * reproducer.Neqs + column] = (
                    reproducer.mget(jac, row, column)
                )
        var first_component = warp_id
        right_hand_sides[cell, first_component] = reproducer.vget(
            rhs, first_component
        )
        var second_component = warp_id + DagWarps
        if second_component < reproducer.NumSpec:
            right_hand_sides[cell, second_component] = reproducer.vget(
                rhs, second_component
            )
        if warp_id == 6:
            right_hand_sides[cell, reproducer.NetIenuc] = reproducer.vget(
                rhs, reproducer.NetIenuc
            )
    sync.phase_barrier()
