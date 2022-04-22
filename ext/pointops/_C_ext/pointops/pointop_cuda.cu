#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>

#include "checks.h"


template <typename scalar_t>
__global__ void pointwise_add_cuda_kernel( 
    torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> const indices,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> const values,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> out_incr_values
){

    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < values.size(0)){
        auto i_0 = indices[i][0];
        auto i_1 = indices[i][1];
        auto i_2 = indices[i][2];

        auto tmp = X[i_0][i_1][i_2] + values[i];
        X[i_0][i_1][i_2] = tmp;
        out_incr_values[i] = tmp;
    }
}


torch::Tensor pointwise_add_cuda(c10::DeviceIndex const device_idx, torch::Tensor const &X ,\
 torch::Tensor const &input_incr_values, torch::Tensor const &input_incr_indices)
{
    int const threads = 1024;
    int const blocks = 20;
    // c10::cuda::CUDAStream const stream = c10::cuda::getCurrentCUDAStream(device_idx);

    auto output_incr_values = torch::zeros_like(input_incr_values);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "pointwise_cuda", ([&] {
            pointwise_add_cuda_kernel<scalar_t><<<blocks, threads>>>(
                input_incr_indices.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                input_incr_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                X.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
                output_incr_values.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>()
            );
        })
    );

    CUDA_CHECK_ERRORS();
    
    return output_incr_values;
}

