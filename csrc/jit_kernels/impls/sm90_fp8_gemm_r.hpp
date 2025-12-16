#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../heuristics/sm90.hpp"

#include "epilogue.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

class SM90FP8GemmRowWiseRuntime final: public LaunchRuntime<SM90FP8GemmRowWiseRuntime> {
public:
    struct Args {
        cute::UMMA::Major major_sfb;
        int m, n, k, num_groups;
        const std::string& compiled_dims;
        const std::optional<std::string>& epilogue_type;

        GemmConfig gemm_config;
        LaunchArgs launch_args;

        void *sfb; // Raw pointer for SFB (Load directly in kernel)
        void *grouped_layout;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_d;
        CUtensorMap tensor_map_sfa; // TMA for SFA
    };

    static std::string generate_impl(const Args& args) {
        // Matches template signature of: sm90_fp8_gemm_1d_rowwise_impl
        return fmt::format(R"(
#include <deep_gemm/impls/sm90_fp8_gemm_1dr.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm90_fp8_gemm_1d_rowwise_impl<
        {}, // kMajorSFB
        {}, {}, {}, // SHAPE_M, SHAPE_N, SHAPE_K
        {}, // kNumGroups
        {}, {}, {}, // BLOCK_M, BLOCK_N, BLOCK_K
        {}, {}, {}, // Swizzle A, B, D
        {}, {}, // NumStages, NumLastStages
        {}, {}, // NumTMAThreads, NumMathThreads
        {}, {}, // NumTMAMulticast, IsTMAMulticastOnA
        {}, {}, // NumSMs, GemmType
        {}  // EpilogueType
    >);
}};
)",
        to_string(args.major_sfb),
        get_compiled_dim(args.m, 'm', args.compiled_dims), get_compiled_dim(args.n, 'n', args.compiled_dims), get_compiled_dim(args.k, 'k', args.compiled_dims),
        args.num_groups,
        args.gemm_config.block_m, args.gemm_config.block_n, args.gemm_config.block_k,
        args.gemm_config.smem_config.swizzle_a_mode, args.gemm_config.smem_config.swizzle_b_mode, args.gemm_config.smem_config.swizzle_cd_mode,
        args.gemm_config.num_stages, args.gemm_config.num_last_stages,
        args.gemm_config.thread_config.num_tma_threads, args.gemm_config.thread_config.num_math_threads,
        args.gemm_config.multicast_config.num_multicast, args.gemm_config.multicast_config.is_multicast_on_a,
        args.gemm_config.num_sms, to_string(args.gemm_config.gemm_type),
        get_default_epilogue_type(args.epilogue_type));
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        // Matches arguments of: sm90_fp8_gemm_1d_rowwise_impl
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.sfb, args.grouped_layout,
            args.m, args.n, args.k,
            args.tensor_map_a, args.tensor_map_b,
            args.tensor_map_d, args.tensor_map_sfa));
    }
};

static void sm90_fp8_gemm_rowwise(const torch::Tensor& a, const torch::Tensor& sfa,
                                  const torch::Tensor& b, const torch::Tensor& sfb,
                                  const std::optional<torch::Tensor>& c,
                                  const torch::Tensor& d,
                                  const int& m, const int& n, const int& k,
                                  const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b, 
                                  const std::string& compiled_dims,
                                  const std::optional<std::string>& epilogue_type = std::nullopt) {
    DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K);

    // Default to Row-Major (MN) for scaling factors if not specified, 
    // though the kernel currently loads SFB manually via pointer logic.
    const auto major_sfb = cute::UMMA::Major::MN; 

    const auto& aligned_k = align(k, 128);
    // Using Kernel1D2D heuristic as a baseline for Rowwise (1D Row)
    const auto& config = get_best_config<SM90ArchSpec>(
        GemmType::Normal, KernelType::Kernel1D2D,
        m, n, k, 1, major_a, major_b,
        torch::kFloat8_e4m3fn, d.scalar_type(), c.has_value(),
        device_runtime->get_num_sms());

    // TMA A Setup
    const auto& tensor_map_a = make_tma_a_desc(major_a, a, m, k,
                                               SM90ArchSpec::get_ab_load_block_m(config.multicast_config, config.block_m),
                                               config.block_k,
                                               static_cast<int>(a.stride(get_non_contiguous_dim(major_a))), 1,
                                               config.smem_config.swizzle_a_mode);
    
    // TMA B Setup
    const auto& tensor_map_b = make_tma_b_desc(major_b, b, n, k,
                                               SM90ArchSpec::get_ab_load_block_n(config.multicast_config, config.block_n),
                                               config.block_k,
                                               static_cast<int>(b.stride(get_non_contiguous_dim(major_b))), 1,
                                               config.smem_config.swizzle_b_mode);
    
    // TMA D Setup
    const auto& tensor_map_d = make_tma_cd_desc(d, m, static_cast<int>(d.size(-1)),
                                                SM90ArchSpec::get_cd_store_block_m(config.block_m),
                                                SM90ArchSpec::get_cd_store_block_n(config.block_n),
                                                static_cast<int>(d.stride(-2)), 1,
                                                config.smem_config.swizzle_cd_mode);

    // TMA SFA Setup (Row-wise scaling factor for A: Shape [M, 1], treated as [M, K] with stride 0 in K usually, 
    // but here likely treated as a simple column vector loaded via TMA)
    // Note: implementation details of make_tma_sf_desc might need adjustment for pure (M, 1) layout if not already supported.
    // Assuming make_tma_sf_desc handles the striding for broadcast or simple vector loading.
    const auto& tensor_map_sfa = make_tma_sf_desc(cute::UMMA::Major::MN, sfa, m, k,
                                                  config.block_m, config.block_k, 1, 0);

    // Launch
    const SM90FP8GemmRowWiseRuntime::Args& args = {
        .major_sfb = major_sfb,
        .m = m, .n = n, .k = aligned_k,
        .num_groups = 1,
        .compiled_dims = compiled_dims,
        .epilogue_type = epilogue_type,
        .gemm_config = config,
        .launch_args = LaunchArgs(config.num_sms, config.thread_config.num_threads,
                                  config.smem_config.smem_size,
                                  config.multicast_config.num_multicast),
        .sfb = sfb.data_ptr(), // Passing raw pointer for SFB
        .grouped_layout = nullptr,
        .tensor_map_a = tensor_map_a,
        .tensor_map_b = tensor_map_b,
        .tensor_map_d = tensor_map_d,
        .tensor_map_sfa = tensor_map_sfa,
    };

    // JIT Build & Launch
    const auto& code = SM90FP8GemmRowWiseRuntime::generate(args);
    const auto& runtime = compiler->build("sm90_fp8_gemm_rowwise", code);
    SM90FP8GemmRowWiseRuntime::launch(runtime, args);
}

} // namespace deep_gemm