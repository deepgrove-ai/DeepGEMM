#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/scheduler.cuh>
#include <deep_gemm/common/sm90_utils.cuh>

namespace deep_gemm {

using namespace deep_gemm::sm90;

template <uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs,
          GemmType kGemmType, typename cd_dtype_t>
__global__ __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d1d_rowwise_impl(__nv_fp8_e4m3* gmem_a_ptr, __nv_fp8_e4m3* gmem_b_ptr,
                        int* grouped_layout,
                        cute::TmaDescriptor* tensor_map_buffer,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b_base,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(kNumTMAThreads == 128 and kNumMathThreads % 128 == 0, "Invalid Threads");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>, "Invalid C/D data dtype");
    DG_STATIC_ASSERT(kGemmType == GemmType::Normal or kGemmType == GemmType::KGroupedContiguous, "Invalid GEMM type");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory Layout
    // 1. Tensor Maps
    static constexpr uint32_t SMEM_TENSOR_MAP_SIZE = (kGemmType == GemmType::KGroupedContiguous ? sizeof(cute::TmaDescriptor) * 4 : 0);
    // 2. Output D
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_M * BLOCK_N * sizeof(float);
    // 3. Main Pipe (A/B)
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    
    // 4. Scales (Row-wise/Col-wise) - Single Buffered (Not Staged)
    // We only need one buffer per tile because scales don't change along K
    static constexpr uint32_t SMEM_SFA_SIZE = BLOCK_M * sizeof(float);
    static constexpr uint32_t SMEM_SFB_SIZE = BLOCK_N * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFB_SIZE = constexpr_align(SMEM_SFB_SIZE, 128u);
    DG_STATIC_ASSERT(SMEM_SFA_SIZE % 128 == 0, "Invalid TMA alignment");

    // Configs
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = threadIdx.x % 32;

    // Prefetch TMA descriptors
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a_base);
        cute::prefetch_tma_descriptor(&tensor_map_b_base);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();


    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // --- Pointers Setup ---
    
    // Tensor maps
    auto smem_tensor_map_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + static_cast<uint32_t>(sizeof(cute::TmaDescriptor)) * i);
    });
    auto smem_tensor_map_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cute::TmaDescriptor*>(smem_buffer + static_cast<uint32_t>(sizeof(cute::TmaDescriptor)) * (2 + i));
    });
    auto gmem_tensor_map_a = PatternVisitor([=](const uint32_t& i) { return tensor_map_buffer + blockIdx.x * 4 + i; });
    auto gmem_tensor_map_b = PatternVisitor([=](const uint32_t& i) { return tensor_map_buffer + blockIdx.x * 4 + 2 + i; });

    // Data Pointers
    auto smem_d = reinterpret_cast<float*>(smem_buffer + SMEM_TENSOR_MAP_SIZE);
    
    // Staged A/B pointers
    auto smem_a = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE)); 
    });
    auto smem_b = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + (SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE));
    });

    // Single-buffer Scale pointers (Row/Col wise)
    constexpr auto SMEM_SF_OFFSET = SMEM_TENSOR_MAP_SIZE + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa_ptr = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET);
    auto smem_sfb_ptr = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + SMEM_SFA_SIZE);

    // Barriers
    // 1. Pipeline Barriers for A/B (Staged)
    constexpr auto SMEM_BARRIER_OFFSET = SMEM_SF_OFFSET + SMEM_SFA_SIZE + ALIGNED_SMEM_SFB_SIZE;
    auto full_barriers = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + i * static_cast<uint32_t>(sizeof(Barrier))));
    });
    auto empty_barriers = PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<Barrier*>(smem_buffer + (SMEM_BARRIER_OFFSET + (kNumStages + i) * static_cast<uint32_t>(sizeof(Barrier))));
    });

    // 2. Scale Barriers (Single Stage)
    // We need one barrier pair to sync the single buffer of scales per Tile
    constexpr auto SMEM_SCALE_BARRIER_OFFSET = SMEM_BARRIER_OFFSET + 2 * kNumStages * static_cast<uint32_t>(sizeof(Barrier));
    auto scale_full_barrier = reinterpret_cast<Barrier*>(smem_buffer + SMEM_SCALE_BARRIER_OFFSET);
    auto scale_empty_barrier = reinterpret_cast<Barrier*>(smem_buffer + SMEM_SCALE_BARRIER_OFFSET + sizeof(Barrier));

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // Load tensormap A/B to shared memory if grouped
        if constexpr (kGemmType == GemmType::KGroupedContiguous) {
            *smem_tensor_map_a[0] = tensor_map_a_base;
            *smem_tensor_map_a[1] = tensor_map_a_base;
            *smem_tensor_map_b[0] = tensor_map_b_base;
            *smem_tensor_map_b[1] = tensor_map_b_base;
        }

        // Initialize Pipeline barriers
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Initialize Scale barriers
        scale_full_barrier->init(1);
        scale_empty_barrier->init(kNumTMAMulticast * kNumMathThreads / 32);

        cutlass::arch::fence_barrier_init();
    }

    // Synchronize
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Pipeline unroll control
    constexpr uint32_t kNumPipelineUnrolls = (kGemmType == GemmType::KGroupedContiguous ? 0 : kNumStages);

    // Register reconfigurations
    constexpr uint32_t kNumTMARegisters = (kNumPipelineUnrolls == 0 ? 40 : 24);
    constexpr uint32_t kNumMathRegisters = (kNumPipelineUnrolls == 0 ? 232 : 240);

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs, 128u>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline State
    const auto& get_pipeline = [=](const uint32_t& iter_idx) -> cute::tuple<uint32_t, uint32_t> {
        return {iter_idx % kNumStages, (iter_idx / kNumStages) & 1}; 
    };
    uint32_t iter_idx = 0;
    uint32_t tile_idx = 0; // To toggle phase for Scale barriers

    if (warp_idx >= kNumMathThreads / 32) {
        // --- TMA Warp (Producer) ---
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
            const cute::TmaDescriptor* current_tensor_map_a = &tensor_map_a_base;
            const cute::TmaDescriptor* current_tensor_map_b = &tensor_map_b_base;
            uint32_t last_group_idx = kNumGroups, sum_k = 0;

            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                // Multicast setup
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                
                const uint32_t& num_k_blocks = ceil_div(scheduler.current_shape_k, BLOCK_K);
                const uint32_t& m_idx = m_block_idx * BLOCK_M;
                const uint32_t& n_idx = n_block_idx * BLOCK_N;

                // Grouped GEMM Tensor Map Swapping
                if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
                    const uint32_t& stage_idx = scheduler.current_num_valid_groups & 1;
                    const uint32_t& next_stage_idx = stage_idx ^ 1;
                    last_group_idx = scheduler.current_group_idx;

                    sum_k += scheduler.current_shape_k;
                    if (scheduler.next_group_idx < kNumGroups) {
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_a[next_stage_idx], gmem_a_ptr + static_cast<uint64_t>(sum_k) * shape_m);
                        tensor_map_replace_global_addr_in_smem(smem_tensor_map_b[next_stage_idx], gmem_b_ptr + static_cast<uint64_t>(sum_k) * shape_n);
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a[next_stage_idx], scheduler.next_shape_k, scheduler.next_shape_k);
                        tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_b[next_stage_idx], scheduler.next_shape_k, scheduler.next_shape_k);
                        *(gmem_tensor_map_a[next_stage_idx]) = *(smem_tensor_map_a[next_stage_idx]);
                        *(gmem_tensor_map_b[next_stage_idx]) = *(smem_tensor_map_b[next_stage_idx]);
                        tensor_map_release_cta();
                    }

                    if (scheduler.current_num_valid_groups > 0) {
                        tensor_map_acquire_cta(gmem_tensor_map_a[stage_idx]);
                        tensor_map_acquire_cta(gmem_tensor_map_b[stage_idx]);
                        current_tensor_map_a = gmem_tensor_map_a[stage_idx];
                        current_tensor_map_b = gmem_tensor_map_b[stage_idx];
                    }
                }

                // -----------------------------------------------------------
                // 1. Load Scales (Row-wise / Col-wise)
                // -----------------------------------------------------------
                // Wait for consumer to be done with previous scales
                uint32_t scale_phase = (tile_idx & 1);
                scale_empty_barrier->wait(scale_phase ^ 1);
                
                // Issue TMA copy for scales (Once per tile)
                // Use block_K=1 or generic logic, as scales are (M, 1) and (1, N)
                // We pass 0 for the inner-dimension coordinate
                tma_copy<BLOCK_M, 1, 0>(&tensor_map_sfa, scale_full_barrier, smem_sfa_ptr, m_idx, 0, num_tma_multicast_a);
                tma_copy<BLOCK_N, 1, 0>(&tensor_map_sfb, scale_full_barrier, smem_sfb_ptr, n_idx, 0, num_tma_multicast_b);
                
                // Expect bytes for scales
                scale_full_barrier->arrive_and_expect_tx(SMEM_SFA_SIZE + SMEM_SFB_SIZE);

                // -----------------------------------------------------------
                // 2. Load A/B Blocks (Loop over K)
                // -----------------------------------------------------------
                #pragma unroll kNumPipelineUnrolls
                for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                    // Wait consumer release A/B buffer
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);

                    // Issue TMA for A/B
                    auto& full_barrier = *full_barriers[stage_idx];
                    const uint32_t& k_idx = k_block_idx * BLOCK_K;
                    
                    tma_copy<BLOCK_K, BLOCK_M, kSwizzleAMode>(current_tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx, m_idx, num_tma_multicast_a);
                    tma_copy<BLOCK_K, BLOCK_N, kSwizzleBMode>(current_tensor_map_b, &full_barrier, smem_b[stage_idx], k_idx, n_idx, num_tma_multicast_b);
                    
                    full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
                }
                
                tile_idx++;
            }

            // Cleanup
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s) {
                    CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                    empty_barriers[stage_idx]->wait(phase ^ 1);
                }
                // Cleanup scale barrier
                scale_empty_barrier->wait(tile_idx & 1 ^ 1);
            }
        }
    } else {
        // --- Math Warps (Consumer) ---
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto row_idx = lane_idx / 4, col_idx = lane_idx % 4;
        const auto r_0 = warp_idx * 16 + row_idx, r_1 = r_0 + 8;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            DG_STATIC_ASSERT(BLOCK_M == WGMMA::M * (BLOCK_M <= 64 ? 1 : 2), "Invalid block sizes");
            const uint32_t& current_shape_k = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_shape_k : shape_k);
            const uint32_t& current_group_idx = (kGemmType == GemmType::KGroupedContiguous ? scheduler.current_group_idx : 0);
            const uint32_t& num_k_blocks = ceil_div(current_shape_k, BLOCK_K);
            
            // Accumulators
            float accum[WGMMA::kNumAccum] = {0};       // Raw WGMMA output
            float final_accum[WGMMA::kNumAccum] = {0}; // Aggregated sum (unscaled until end)
            
            // -----------------------------------------------------------
            // 1. Wait and Load Scales
            // -----------------------------------------------------------
            uint32_t scale_phase = (tile_idx & 1);
            scale_full_barrier->wait(scale_phase);
            
            // Load Scales into Registers (Constant for the Tile)
            float scale_a_0 = ld_shared(smem_sfa_ptr + r_0);
            float scale_a_1 = ld_shared(smem_sfa_ptr + r_1);
            float2 scales_b[WGMMA::kNumAccum / 4];
            
            #pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum / 4; ++i)
                scales_b[i] = ld_shared(reinterpret_cast<float2*>(smem_sfb_ptr + i * 8 + col_idx * 2));

            // Helper for barrier arrival
            auto empty_barrier_arrive = [&](Barrier* b) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? b->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? b->arrive(target_cta) : void();
                }
            };

            // -----------------------------------------------------------
            // 2. Main Loop (WGMMA over K)
            // -----------------------------------------------------------
            #pragma unroll kNumPipelineUnrolls
            for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++ k_block_idx) {
                // Wait A/B
                CUTE_TIE_DECL(get_pipeline(iter_idx ++), stage_idx, phase);
                full_barriers[stage_idx]->wait(phase);

                // WGMMA
                // Note: No scale reading here.
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_arrive();
                
                #pragma unroll
                for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                    auto desc_a = make_smem_desc(smem_a[stage_idx] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                    auto desc_b = make_smem_desc(smem_b[stage_idx] + k * WGMMA::K, 1);
                    WGMMA::wgmma(desc_a, desc_b, accum, k);
                }
                
                warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                    warpgroup_fence_operand(accum[i]);
                warpgroup_wait<0>();

                // Notify A/B buffer empty
                empty_barrier_arrive(empty_barriers[stage_idx]);

                // Accumulate Raw Results (FP32 add)
                // WGMMA output is in `accum`. We add to `final_accum`.
                #pragma unroll
                for(int i = 0; i < WGMMA::kNumAccum; ++i) {
                     final_accum[i] += accum[i];
                     accum[i] = 0.0f; // Reset for next chunk
                }
            }

            // -----------------------------------------------------------
            // 3. Apply Scales & Store
            // -----------------------------------------------------------
            
            // Now we release the scale barrier since we have used the scales and finished the loops
            // BUT: We loaded scales into regs at start. We can actually release the scale barrier 
            // immediately after loading if we don't read them again. 
            // However, strictly safely, we release here or earlier. 
            // Let's release here to signal "done with this tile's scale buffer".
            empty_barrier_arrive(scale_empty_barrier);

            // Apply Row-wise / Col-wise scaling: Out = Sum(A*B) * sA * sB
            #pragma unroll
            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                const float &scale_b_0 = scales_b[i].x;
                const float &scale_b_1 = scales_b[i].y;
                final_accum[i * 4 + 0] *= (scale_a_0 * scale_b_0);
                final_accum[i * 4 + 1] *= (scale_a_0 * scale_b_1);
                final_accum[i * 4 + 2] *= (scale_a_1 * scale_b_0);
                final_accum[i * 4 + 3] *= (scale_a_1 * scale_b_1);
            }

            // Flush previous stores
            if (warp_idx % 4 == 0 and cute::elect_one_sync())
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // Store to D
            const auto& smem_d_0 = reinterpret_cast<float2*>(smem_d + r_0 * BLOCK_N + col_idx * 2);
            const auto& smem_d_1 = reinterpret_cast<float2*>(smem_d + r_1 * BLOCK_N + col_idx * 2);
            #pragma unroll
            for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                st_shared(smem_d_0 + i * 4, {final_accum[i * 4 + 0], final_accum[i * 4 + 1]});
                st_shared(smem_d_1 + i * 4, {final_accum[i * 4 + 2], final_accum[i * 4 + 3]});
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(128, math_wg_idx);

            // TMA store
            // TMA store
            // Only the first thread of the first Math Warp should trigger the store for the whole group.
            // We use math_wg_idx (0 for the first group) and lane_idx.
            if (math_wg_idx == 0 && lane_idx == 0 and cute::elect_one_sync()) {
                
                // 1. Point to the BASE of the Shared Memory Buffer (smem_d), not the warp offset
                // 2. Point to the BASE of the Global Memory Tile (m_block_idx * BLOCK_M), not + r_0
                
                cute::SM90_TMA_REDUCE_ADD_2D::copy(
                    &tensor_map_cd, 
                    smem_d, // <--- FIX: Base Pointer
                    n_block_idx * BLOCK_N,
                    current_group_idx * shape_m + m_block_idx * BLOCK_M // <--- FIX: Base Coordinate (No + r_0)
                );
                cute::tma_store_arrive();
            }
            __syncwarp();
            
            tile_idx++;
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop