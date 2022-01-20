#include <vector>
#include <ATen/ATen.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
#include <curand_kernel.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <math.h>
#include "util.h"

template <typename MaskType, typename acc_t, typename IndexType>
__global__ void generate_dropout_mask_kernel(MaskType* output, IndexType n, uint64_t seed, uint64_t offset, acc_t p) {
    IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx * sizeof(MaskType) * 8, offset, &state);
    MaskType mask = 0;
    #pragma unroll
    for (int i = 0; i < sizeof(MaskType) * 2; ++i) {
        const float4 rand = curand_uniform4(&state);
        mask |= (((MaskType)(rand.x < p)) << (i * 4))
            | (((MaskType)(rand.y < p)) << (i * 4 + 1))
            | (((MaskType)(rand.z < p)) << (i * 4 + 2))
            | (((MaskType)(rand.w < p)) << (i * 4 + 3));
    }
    if (idx < n) {
        output[idx] = mask;
    }
}

template <typename MaskType>
void generate_dropout_mask(MaskType* mask, int bsz, int dim, float p, uint64_t seed, uint64_t offset) {
    const int mask_elements_per_batch = DIV_CELL(dim, sizeof(MaskType) * 8);
    const int num_elements = bsz * mask_elements_per_batch;
    const int block_size = 128;
    const int grid = DIV_CELL(num_elements, block_size);
    generate_dropout_mask_kernel<MaskType, float, size_t><<<grid, block_size>>>(mask, num_elements, seed, offset, p);
}

template <typename T>
__device__ __forceinline__ T from_uint8(uint8_t input) {
    return (T)input;
}

// template <>
// __device__ __forceinline__ __nv_bfloat16 from_uint8(uint8_t input) {
//     return (__nv_bfloat16)(float)input;
// }

template <typename index_t, typename input_t, typename output_t, bool is_training>
__global__ void bias_dropout_add_forward(output_t *dst, const input_t *x, const input_t *bias,
    const input_t *residual, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
    if (blockIdx.x < bsz) {
        if IF_CONSTEXPR (is_training) {
            const int mask_index = blockIdx.x * DIV_CELL(dim, 8);
            const uint8_t mask_offset = threadIdx.x % 8;
            for (int j = threadIdx.x; j < dim; j += blockDim.x) {
                const index_t idx = blockIdx.x * dim + j;
                const input_t y = x[idx] + bias[j];
                const input_t m = from_uint8<input_t>((mask[mask_index + j / 8] >> mask_offset) & 1);
                dst[idx] = y * m * pinv + residual[idx];
            }
        } else {
            for (int j = threadIdx.x; j < dim; j += blockDim.x) {
                const index_t idx = blockIdx.x * dim + j;
                dst[idx] = x[idx] + bias[j] + residual[idx];
            }
        }
    }
}

// template <typename index_t, typename input_t, typename output_t, bool is_training>
// __global__ void bias_dropout_add_forward_vec(output_t *dst, const input_t *x, const input_t *bias,
//     const input_t *residual, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
//     using VecInType = VecType<input_t, 2>;
//     using VecOutType = VecType<output_t, 2>;
//     if (blockIdx.x < bsz) {
//         if IF_CONSTEXPR (is_training) {
//             const int mask_index = blockIdx.x * DIV_CELL(dim, 8);
//             const uint8_t mask_offset1 = (threadIdx.x * 2) % 8;
//             const uint8_t mask_offset2 = (threadIdx.x * 2 + 1) % 8;
//             for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
//                 const index_t idx = blockIdx.x * dim + j;
//                 const VecInType xi = *(VecInType *)(x + idx);
//                 const VecInType b = *(VecInType *)(bias + j);
//                 const VecInType r = *(VecInType *)(residual + idx);
//                 const uint8_t m = mask[mask_index + j / 8];
//                 const input_t m1 = from_uint8<input_t>((m >> mask_offset1) & 1);
//                 const input_t m2 = from_uint8<input_t>((m >> mask_offset2) & 1);
//                 VecOutType d;
//                 d.x = (xi.x + b.x) * m1 * pinv + r.x;
//                 d.y = (xi.y + b.y) * m2 * pinv + r.y;
//                 *(VecOutType *)(dst + idx) = d;
//             }
//         } else {
//             for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
//                 const index_t idx = blockIdx.x * dim + j;
//                 const VecInType xi = *(VecInType *)(x + idx);
//                 const VecInType b = *(VecInType *)(bias + j);
//                 const VecInType r = *(VecInType *)(residual + idx);
//                 VecOutType d;
//                 d.x = xi.x + b.x + r.x;
//                 d.y = xi.y + b.y + r.y;
//                 *(VecOutType *)(dst + idx) = d;
//             }
//         }
//     }
// }

template <typename index_t, typename input_t, typename output_t>
__global__ void bias_dropout_add_backward(output_t *dst, const input_t *grad, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
    if (blockIdx.x < bsz) {
        const int mask_index = blockIdx.x * DIV_CELL(dim, 8);
        const uint8_t mask_offset = threadIdx.x % 8;
        for (int j = threadIdx.x; j < dim; j += blockDim.x) {
            const index_t idx = blockIdx.x * dim + j;
            uint8_t m = (mask[mask_index + j / 8] >> mask_offset) & 1;
            dst[idx] = grad[idx] * from_uint8<input_t>(m) * pinv;
        }
    }
}

template <typename index_t, typename input_t, typename output_t>
__global__ void bias_dropout_add_backward_vec(output_t *dst, const input_t *grad, const uint8_t *mask, index_t bsz, int dim, input_t pinv) {
    using VecInType = VecType<input_t, 2>;
    using VecOutType = VecType<output_t, 2>;
    if (blockIdx.x < bsz) {
        const int mask_index = blockIdx.x * DIV_CELL(dim, 8);
        const uint8_t mask_offset1 = (threadIdx.x * 2) % 8;
        const uint8_t mask_offset2 = (threadIdx.x * 2 + 1) % 8;
        for (int j = threadIdx.x * 2; j < dim; j += blockDim.x * 2) {
            const index_t idx = blockIdx.x * dim + j;
            const uint8_t m = mask[mask_index + j / 8];
            const VecInType g = *(VecInType *)(grad + idx);
            VecOutType d;
            d.x = g.x * from_uint8<input_t>((m >> mask_offset1) & 1) * pinv;
            d.y = g.y * from_uint8<input_t>((m >> mask_offset2) & 1) * pinv;
            *(VecOutType *)(dst + idx) = d;
        }
    }
}

std::vector<c10::optional<torch::Tensor>> bias_dropout_add_forward_cuda(const torch::Tensor &x, const torch::Tensor &bias,
    const torch::Tensor &residual, bool is_training, float dropout_prob, c10::optional<at::Generator> gen_) {
    using MaskType = uint64_t;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = x.sizes();
    size_t bsz = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bsz *= sizes[i];
    }
    const int dim = sizes[sizes.size() - 1];
    auto dst_options = x.options().requires_grad(false);
    torch::Tensor results = torch::empty(sizes, dst_options);
    auto type = x.scalar_type();
    const int ThreadsPerBlock = 256;
    int ThreadsPerBlockVec = DIV_CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    if (is_training && dropout_prob != 0.0) {
        auto mask_options = dst_options.dtype(torch::kInt64);
        torch::Tensor mask = torch::empty(bsz * DIV_CELL(dim, sizeof(MaskType) * 8), mask_options);
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());
        std::pair<uint64_t, uint64_t> rng_engine_inputs;
        {
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            rng_engine_inputs = gen->philox_engine_inputs(sizeof(MaskType) * 8);
        }
        uint64_t seed = std::get<0>(rng_engine_inputs);
        uint64_t offset = std::get<1>(rng_engine_inputs);
        generate_dropout_mask<MaskType>((MaskType *)mask.data_ptr(), bsz, dim, 1.0 - dropout_prob, seed, offset);
        // if (type == at::ScalarType::BFloat16) {
        //     if (dim % 2 == 0) {
        //         bias_dropout_add_forward_vec<size_t, nv_bfloat16, nv_bfloat16, true><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
        //             (nv_bfloat16 *)results.data_ptr(),
        //             (const nv_bfloat16 *)x.data_ptr(),
        //             (const nv_bfloat16 *)bias.data_ptr(),
        //             (const nv_bfloat16 *)residual.data_ptr(),
        //             (const uint8_t *)mask.data_ptr(),
        //             bsz,
        //             dim,
        //             1.0 / (1.0 - dropout_prob));
        //     } else {
        //         bias_dropout_add_forward<size_t, nv_bfloat16, nv_bfloat16, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
        //             (nv_bfloat16 *)results.data_ptr(),
        //             (const nv_bfloat16 *)x.data_ptr(),
        //             (const nv_bfloat16 *)bias.data_ptr(),
        //             (const nv_bfloat16 *)residual.data_ptr(),
        //             (const uint8_t *)mask.data_ptr(),
        //             bsz,
        //             dim,
        //             1.0 / (1.0 - dropout_prob));
        //     }
        // } else if (type == at::ScalarType::Half) {
        //     if (dim % 2 == 0) {
        //         bias_dropout_add_forward_vec<size_t, half, half, true><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
        //             (half *)results.data_ptr(),
        //             (const half *)x.data_ptr(),
        //             (const half *)bias.data_ptr(),
        //             (const half *)residual.data_ptr(),
        //             (const uint8_t *)mask.data_ptr(),
        //             bsz,
        //             dim,
        //             1.0 / (1.0 - dropout_prob));
        //     } else {
        //         bias_dropout_add_forward<size_t, half, half, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
        //             (half *)results.data_ptr(),
        //             (const half *)x.data_ptr(),
        //             (const half *)bias.data_ptr(),
        //             (const half *)residual.data_ptr(),
        //             (const uint8_t *)mask.data_ptr(),
        //             bsz,
        //             dim,
        //             1.0 / (1.0 - dropout_prob));
        //     }
        // } else if (type == at::ScalarType::Float) {
            if (type == at::ScalarType::Float) {
            bias_dropout_add_forward<size_t, float, float, true><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (float *)results.data_ptr(),
                (const float *)x.data_ptr(),
                (const float *)bias.data_ptr(),
                (const float *)residual.data_ptr(),
                (const uint8_t *)mask.data_ptr(),
                bsz,
                dim,
                1.0 / (1.0 - dropout_prob));
        }
        return {results, mask};
    } else {
        // if (type == at::ScalarType::BFloat16) {
        //     if (dim % 2 == 0) {
        //         bias_dropout_add_forward_vec<size_t, nv_bfloat16, nv_bfloat16, false><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
        //             (nv_bfloat16 *)results.data_ptr(),
        //             (const nv_bfloat16 *)x.data_ptr(),
        //             (const nv_bfloat16 *)bias.data_ptr(),
        //             (const nv_bfloat16 *)residual.data_ptr(),
        //             nullptr,
        //             bsz,
        //             dim,
        //             0.0);
        //     } else {
        //         bias_dropout_add_forward<size_t, nv_bfloat16, nv_bfloat16, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
        //             (nv_bfloat16 *)results.data_ptr(),
        //             (const nv_bfloat16 *)x.data_ptr(),
        //             (const nv_bfloat16 *)bias.data_ptr(),
        //             (const nv_bfloat16 *)residual.data_ptr(),
        //             nullptr,
        //             bsz,
        //             dim,
        //             0.0);
        //     }
        // } else if (type == at::ScalarType::Half) {
        //     if (dim % 2 == 0) {
        //         bias_dropout_add_forward_vec<size_t, half, half, false><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
        //             (half *)results.data_ptr(),
        //             (const half *)x.data_ptr(),
        //             (const half *)bias.data_ptr(),
        //             (const half *)residual.data_ptr(),
        //             nullptr,
        //             bsz,
        //             dim,
        //             0.0);
        //     } else {
        //         bias_dropout_add_forward<size_t, half, half, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
        //             (half *)results.data_ptr(),
        //             (const half *)x.data_ptr(),
        //             (const half *)bias.data_ptr(),
        //             (const half *)residual.data_ptr(),
        //             nullptr,
        //             bsz,
        //             dim,
        //             0.0);
        //     }
        // } else if (type == at::ScalarType::Float) {
            if (type == at::ScalarType::Float) {
            bias_dropout_add_forward<size_t, float, float, false><<<bsz, ThreadsPerBlock, 0, stream>>>(
                (float *)results.data_ptr(),
                (const float *)x.data_ptr(),
                (const float *)bias.data_ptr(),
                (const float *)residual.data_ptr(),
                nullptr,
                bsz,
                dim,
                0.0);
        }
        return {results, c10::optional<torch::Tensor>()};
    }
}

torch::Tensor bias_dropout_add_backward_cuda(const torch::Tensor &grad, const torch::Tensor &mask, float dropout_prob) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto sizes = grad.sizes();
    size_t bsz = 1;
    for (size_t i = 0; i + 1 < sizes.size(); ++i) {
        bsz *= sizes[i];
    }
    int dim = sizes[sizes.size() - 1];
    auto dst_options = grad.options().requires_grad(false);
    torch::Tensor results = torch::empty(sizes, dst_options);
    auto type = grad.scalar_type();
    const int ThreadsPerBlock = 256;
    int ThreadsPerBlockVec = DIV_CELL(dim, 256) * 256 % 512 == 0 ? 256 : 128;
    // if (type == at::ScalarType::BFloat16) {
    //     if (dim % 2 == 0) {
    //         bias_dropout_add_backward_vec<size_t, nv_bfloat16, nv_bfloat16><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
    //             (nv_bfloat16 *)results.data_ptr(),
    //             (const nv_bfloat16 *)grad.data_ptr(),
    //             (const uint8_t *)mask.data_ptr(),
    //             bsz,
    //             dim,
    //             1.0 / (1.0 - dropout_prob));
    //     } else {
    //         bias_dropout_add_backward<size_t, nv_bfloat16, nv_bfloat16><<<bsz, ThreadsPerBlock, 0, stream>>>(
    //             (nv_bfloat16 *)results.data_ptr(),
    //             (const nv_bfloat16 *)grad.data_ptr(),
    //             (const uint8_t *)mask.data_ptr(),
    //             bsz,
    //             dim,
    //             1.0 / (1.0 - dropout_prob));
    //     }
    // } else if (type == at::ScalarType::Half) {
    //     if (dim % 2 == 0) {
    //         bias_dropout_add_backward_vec<size_t, half, half><<<bsz, ThreadsPerBlockVec, 0, stream>>>(
    //             (half *)results.data_ptr(),
    //             (const half *)grad.data_ptr(),
    //             (const uint8_t *)mask.data_ptr(),
    //             bsz,
    //             dim,
    //             1.0 / (1.0 - dropout_prob));
    //     } else {
    //         bias_dropout_add_backward<size_t, half, half><<<bsz, ThreadsPerBlock, 0, stream>>>(
    //             (half *)results.data_ptr(),
    //             (const half *)grad.data_ptr(),
    //             (const uint8_t *)mask.data_ptr(),
    //             bsz,
    //             dim,
    //             1.0 / (1.0 - dropout_prob));
    //     }
    // } else if (type == at::ScalarType::Float) {
        if (type == at::ScalarType::Float) {
        bias_dropout_add_backward<size_t, float, float><<<bsz, ThreadsPerBlock, 0, stream>>>(
            (float *)results.data_ptr(),
            (const float *)grad.data_ptr(),
            (const uint8_t *)mask.data_ptr(),
            bsz,
            dim,
            1.0 / (1.0 - dropout_prob));
    }
    return results;
}